import argparse
import os
import laspy
import glob
import json
import numpy as np
import torch

from pathlib import Path
from torch_geometric.loader import DataLoader as PyGDataLoader
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)
from typing import List, Dict
from tqdm import tqdm
from utils_v13_preprod import TestDataset, seed_everything, collate_custom_test
from omegaconf import OmegaConf
from model_v13_preprod import MinkUNet14C, Binary_model, MinkUNet34A


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

            # Supprimer le point artificiel
            full_pred1 = full_pred1[:-1]
            full_pred2 = full_pred2[:-1]

            # Ajout des champs dans le fichier LAS
            las.add_extra_dim(laspy.ExtraBytesParams(name="basemodel_v13", type=np.uint8))
            las.add_extra_dim(laspy.ExtraBytesParams(name="plo_v13", type=np.uint8))

            las.basemodel_v13 = full_pred1
            las.plo_v13 = full_pred2

            las.write(os.path.join(savepath, os.path.basename(test_tiles[j])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model point cloud prediction")

    parser.add_argument("--weight_path1", type=str, help="Path to weights for model 1")
    parser.add_argument("--config_file1", type=str, help="Config file for model 1")

    parser.add_argument("--weight_path2", type=str, help="Path to weights for model 2")
    parser.add_argument("--config_file2", type=str, help="Config file for model 2")

    parser.add_argument("--pointclouds", type=str, help="Path to input .json or folder with pointclouds")
    parser.add_argument("--save_path", default="/preds/", type=str, help="Where to save .las outputs")

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
