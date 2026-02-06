import argparse
import os
import laspy
import glob
import json
import torch

from pathlib import Path
from torch_geometric.loader import DataLoader as PyGDataLoader
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)

from tqdm import tqdm
from utils_v13_preprod_outliers import TestDataset, seed_everything, collate_custom_test
from omegaconf import OmegaConf
from model_v13_preprod import MinkUNet14C, Binary_model, MinkUNet34A

import numpy as np
from sklearn.cluster import DBSCAN

def post_traitement_et_fusion(full_pred1, full_pred2, coords,
                              classe_poteau=1, classe_plo=2, classe_tronc=3, classe_batiment=5,
                              classe_panneau=2, classe_potelet=3,
                              classe_out_potelet=7, classe_out_poteau=8, classe_out_panneau=9,
                              eps=0.03, min_samples=5):
    """
    Combine post-traitement des objets verticaux (model plo) et fusion des prédictions basemodele et plo.

    Étapes :
    I.Parcours des clusters dans pred plo:
        1. Corrige les potelets en poteaux si panneau est dans le cluster.
        2. Corrige les conflits poteau/potelet par vote majoritaire.
        3. Fusionne la classif plo avec la classif du modele base selon les règles :
            i Si cluster est classé comme potelet → on assigne toujours la classe potelet.
            ii Si cluster est poteau ou panneau :
                • Si modele base contient tronc mais pas de plo → on ne change rien à la prediciton de base (reste tronc).
                • Si modele base contient uniquement bâtiment → on ne change rien à la prediction de base (reste batiment).
                • Sinon → on assigne classe poteau ou panneau.
    II. Parcours des clusters dans la prédiction fusionnée (PLO, Poteau, Potelet). But: compléter les prédictions du modèle PLO.
        1. Si un cluster contient à la fois la clase PLO et Poteau, on met tout le cluster à Poteau
        2. Si un cluster contient à la fois la classe PLO et Potelet, on met tout le cluster à Potelet
    III. Remap de la classif fusionnée
        1. Si il reste des éléments de la classe PLO (2), ils sont mis à Unclassified (0) car ce sont surement des FPs
        2. Les ids des classes sont réordonnés.
    """
    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords)

    labels = full_pred2.astype(np.uint8)
    fused_pred = full_pred1.copy()

    mask_verticaux = np.isin(labels, [classe_poteau, classe_panneau, classe_potelet])
    if not np.any(mask_verticaux):
        print("Aucun objet vertical détecté (poteau, panneau, potelet).")
        return fused_pred

    coords_verticaux = coords[mask_verticaux]
    labels_verticaux = labels[mask_verticaux]

    # Clustering des objets verticaux
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_verticaux[:, :2])
    cluster_ids = clustering.labels_

    print(f"Clustering des objets verticaux : {len(np.unique(cluster_ids)) - (1 if -1 in cluster_ids else 0)} clusters détectés.")

    for cid in np.unique(cluster_ids):
        if cid == -1:
            continue  # bruit

        cluster_mask = cluster_ids == cid
        cluster_indices = np.where(mask_verticaux)[0][cluster_mask]
        cluster_labels = labels_verticaux[cluster_mask]
        cluster_base_labels = full_pred1[cluster_indices]

        # Étape 1 : corriger potelet → poteau si panneau présent
        if classe_panneau in cluster_labels:
            # Conversion des potelets → poteaux uniquement
            mask_potelets = cluster_labels == classe_potelet
            cluster_labels[mask_potelets] = classe_poteau
            labels_verticaux[cluster_mask][mask_potelets] = classe_poteau
            print(f"Cluster {cid} : panneau détecté → potelets convertis en poteaux.")

            # Indices globaux des objets du cluster
            indices_globaux = np.where(mask_verticaux)[0][cluster_mask]
            # Indices globaux des potelets à convertir
            indices_potelets = indices_globaux[mask_potelets]
            # Met à jour la prédiction d'origine
            full_pred2[indices_potelets] = classe_poteau


        else:
            # Vote majoritaire poteau vs potelet
            count_poteau = np.sum(cluster_labels == classe_poteau)
            count_potelet = np.sum(cluster_labels == classe_potelet)

            indices_cluster = np.where(mask_verticaux)[0][cluster_mask]

            if count_poteau > count_potelet:
                cluster_labels[:] = classe_poteau
                full_pred2[indices_cluster] = classe_poteau  # 🔁 mise à jour de la prédiction brute

            elif count_potelet >= count_poteau:
                cluster_labels[:] = classe_potelet
                full_pred2[indices_cluster] = classe_potelet  # 🔁 mise à jour de la prédiction brute

        # Étape 2 : règles métier sur la base
        base_set = set(cluster_base_labels)
        if classe_tronc in base_set and classe_plo not in base_set:
            print(f"Cluster {cid} : tronc sans plo → probablement arbre → on ne change rien.")
            continue
        if base_set == {classe_batiment}:
            print(f"Cluster {cid} : uniquement bâtiment → pas de changement.")
            continue

        # Étape 3 : assignation individuelle dans fused_pred
        for i, idx in enumerate(cluster_indices):
            label = cluster_labels[i]
            if label == classe_poteau:
                fused_pred[idx] = classe_out_poteau
            elif label == classe_panneau:
                fused_pred[idx] = classe_out_panneau

    #rajout de tous les potelets de full_pred_2 à fused_pred
    fused_pred[full_pred2 == 3] = classe_out_potelet


    # ➕ Étape finale : nouveau clustering sur les classes 2, 7 et 8 (regroupés en un seul cluster pour DBSCAN)
    mask_final_cluster = np.isin(fused_pred, [2, 7, 8])
    coords_final = coords[mask_final_cluster]
    labels_final = fused_pred[mask_final_cluster]

    if len(coords_final) > 0:
        clustering_final = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_final[:, :2])
        cluster_ids_final = clustering_final.labels_
        unique_clusters = np.unique(cluster_ids_final)

        print(f"[Post-fusion] {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} clusters détectés pour règles finales.")

        for cid in unique_clusters:
            if cid == -1:
                continue  # bruit

            mask_cluster = cluster_ids_final == cid
            cluster_indices_global = np.where(mask_final_cluster)[0][mask_cluster]
            cluster_labels = fused_pred[cluster_indices_global]

            has_2 = 2 in cluster_labels
            has_7 = 7 in cluster_labels
            has_8 = 8 in cluster_labels

            if has_2 and has_7:
                fused_pred[cluster_indices_global] = 7
                print(f"[Fusion finale] Cluster {cid} : 2 & 7 → tout en 7.")
            elif has_2 and has_8:
                fused_pred[cluster_indices_global] = 8
                print(f"[Fusion finale] Cluster {cid} : 2 & 8 → tout en 8.")

    # Nettoyage final : suppression de la classe plo (2)
    fused_pred[fused_pred == classe_plo] = 0


    # Réordonnancement final des classes
    remap = {
        0: 0, #unclass
        1: 1, #ground
        3: 2, #tronc
        4: 3, #vegetation
        5: 4, #batiment
        8: 5, #poteaux
        9: 6, #panneau
        7: 7  #potelet
    }

    max_label = max(fused_pred.max(), max(remap.keys()))  # 👈 s’assure que le tableau est assez grand
    remap_array = np.arange(max_label + 1)
    for old, new in remap.items():
        remap_array[old] = new
    fused_pred = remap_array[fused_pred]

    return fused_pred


def predict_dual_model(weights1, config1, weights2, config2, pointclouds, savepath) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Charger les deux configs et modèles
    conf1 = OmegaConf.merge(OmegaConf.load(config1), OmegaConf.create())
    conf2 = OmegaConf.merge(OmegaConf.load(config2), OmegaConf.create())

    model1 = MinkUNet34A(conf1.num_features, conf1.num_classes)
    if conf1.num_classes == 1:
        model1 = Binary_model(model1)
    model1.load_state_dict(torch.load(weights1))
    model1.to(device)

    model2 = MinkUNet14C(conf2.num_features, conf2.num_classes)
    if conf2.num_classes == 1:
        model2 = Binary_model(model2)
    model2.load_state_dict(torch.load(weights2))
    model2.to(device)

    seed_everything(conf1.random_seed)

    # -- Récupération des fichiers point cloud
    if Path(pointclouds).suffix == ".json":
        with open(pointclouds, encoding="utf-8") as f:
            all_tiles = json.load(f)
        test_tiles = [
            str(Path(pointclouds).parent) + "/pointclouds/" + x["tile_name"]
            for x in all_tiles if x["split"] == "val"
        ]
    else:
        test_tiles = glob.glob(pointclouds + "*.la[sz]")

    dataset = TestDataset(conf1, test_tiles)
    dataloader = PyGDataLoader(
        dataset, batch_size=conf1.test_batch_size, collate_fn=collate_custom_test, pin_memory=False
    )

    inv_voxel_size = 1.0 / conf1.voxel_size

    with torch.no_grad():
        for j, batch in enumerate(tqdm(dataloader, desc="Predicting pointclouds")):
            coords = torch.cat([
                batch.batch.unsqueeze(1),
                torch.floor(batch.pos * inv_voxel_size)
            ], dim=1)
            features = batch.x.to(device)

            # --- Modèle 1 ---
            field1 = TensorField(
                features=features,
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            out1 = model1(field1.sparse()).slice(field1).F
            pred1 = torch.argmax(out1, axis=1).cpu().numpy()

            # --- Indices des points "non-sol" ---
            mask_not_ground = pred1 != 1
            pred_model2 = np.zeros_like(pred1)

            # --- Modèle 2 sur les points non-sol uniquement ---
            if mask_not_ground.sum() > 0:
                features2 = features[mask_not_ground]
                batch_idx2 = batch.batch[mask_not_ground]

                coords2 = torch.cat([batch_idx2.unsqueeze(1), torch.floor(batch.pos[mask_not_ground] * inv_voxel_size)], dim=1)

                field2 = TensorField(
                    features=features2.to(device),
                    coordinates=coords2.to(device),
                    quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
                )
                out2 = model2(field2.sparse()).slice(field2).F
                pred2 = torch.argmax(out2, axis=1).cpu().numpy()

                pred_model2[mask_not_ground] = pred2
            else:
                continue

            # --- Fusion finale : sol du modèle 1, reste du modèle 2
            fused_pred = np.where(pred1 == 1, 0, pred_model2)

            # --- Lecture du fichier LAS
            las = laspy.read(test_tiles[j])
            n_points = len(las.classification)
            full_pred1 = np.zeros(n_points + 1, dtype=np.uint8)
            full_pred2 = np.zeros(n_points + 1, dtype=np.uint8)

            indices_kept = batch.index_kept.cpu().numpy()

            full_pred1[indices_kept] = pred1
            full_pred2[indices_kept] = fused_pred

            index_quantile_excluded = batch.index_quantile_excluded.cpu().numpy()
            full_pred1[index_quantile_excluded] = 1

            # Supprimer le point artificiel
            full_pred1 = full_pred1[:-1]
            full_pred2 = full_pred2[:-1]

            # # Ajout des champs dans le fichier LAS
            if "basemodel_v13" not in las.point_format.extra_dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name="basemodel_v13", type=np.uint8))
            if "plo_v13" not in las.point_format.extra_dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name="plo_v13", type=np.uint8))

            las.basemodel_v13 = full_pred1
            las.plo_v13 = full_pred2

            ###fusion des prédictions potelet/poteau sur le même objet
            full_pred2_corr = post_traitement_et_fusion(
                full_pred1,
                full_pred2,
                coords=np.vstack((las.x, las.y, las.z)).T
            )

            if "fuze" not in las.point_format.extra_dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name="fuze", type=np.uint8))
            las.fuze = full_pred2_corr

            las.classification = full_pred2_corr

            las.write(os.path.join(savepath, os.path.basename(test_tiles[j])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model point cloud prediction")

    parser.add_argument("--weight_path1", type=str, help="Path to weights for model 1")
    parser.add_argument("--config_file1", type=str, help="Config file for model 1")

    parser.add_argument("--weight_path2", type=str, help="Path to weights for model 2")
    parser.add_argument("--config_file2", type=str, help="Config file for model 2")

    parser.add_argument("--pointclouds", type=str, help="Path to input .json or folder with pointclouds")
    parser.add_argument("--save_path", default="/preds/", type=str, help="Where to save .laz outputs")

    args = parser.parse_args()

    predict_dual_model(
        args.weight_path1,
        args.config_file1,
        args.weight_path2,
        args.config_file2,
        args.pointclouds,
        args.save_path,
    )



if __name__ == "__main__":
    main()
