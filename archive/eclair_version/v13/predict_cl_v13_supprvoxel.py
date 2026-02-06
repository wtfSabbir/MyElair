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
from utils_v13 import TestDataset, seed_everything, collate_custom_test
from omegaconf import OmegaConf
from model_v13 import MinkUNet14C, Binary_model, MinkUNet34A, MinkUNet14, MinkUNet18



def predict(weights, config, pointclouds, savepath) -> None:
    """
    Prediction of all the poinclouds in the poinclouds parameter (either pointclouds in the labels.json file with
    split = test or all the pointclouds in a folder)
    :param weights: path to the weights file (.pth)
    :param config: path to train.yaml
    :param pointclouds: path to the .json file or a folder with .laz files
    :param savepath: path to the folder where the predictions will be saved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reading the config file, weights and instantiating the model
    from_cli = OmegaConf.create()
    base_conf = OmegaConf.load(config)
    conf = OmegaConf.merge(base_conf, from_cli)
    model_pred = MinkUNet14C(conf.num_features, conf.num_classes)
    if conf.num_classes == 1:
        model_pred = Binary_model(model_pred)
    model_pred.load_state_dict(torch.load(weights))
    seed_everything(conf.random_seed)

    # Reading the pointclouds
    if Path(pointclouds).suffix == ".json":

        with open(pointclouds, encoding="utf-8") as f:
            all_tiles: List[Dict] = json.load(f)
            print(str(Path(pointclouds).parent) + "/pointclouds/")
        test_tiles: List[str] = [
            str(Path(pointclouds).parent) + "/pointclouds/" + x["tile_name"]
            for x in all_tiles
            if x["split"] == "val"
        ]
    else:
        try:
            test_tiles: List[str] = glob.glob(pointclouds + "*.la[sz]")
        except FileNotFoundError as exception:
            raise FileNotFoundError(
                f"No pointcloud files found for: {pointclouds}."
            ) from exception

    dataset = TestDataset(conf, test_tiles)
    dataloader = PyGDataLoader(
        dataset,
        batch_size=conf.test_batch_size,
        collate_fn=collate_custom_test,
        pin_memory=False,
    )
    model_pred.to(device)

    inv_voxel_size=1.0/conf.voxel_size

    with torch.no_grad():
        for j, batch in enumerate(tqdm(dataloader, desc="Predicting pointclouds")):

            coords, features, batch_idx = (
                torch.floor(batch.pos * inv_voxel_size),
                batch.x,
                batch.batch,
            )

            coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

            # Identifier les voxels sous-peuplés
            min_points_per_voxel = 4

            unique_coords, inverse_indices, counts = torch.unique(coords, return_inverse=True, return_counts=True,
                                                                  dim=0)
            mask_valid_voxels = counts >= min_points_per_voxel
            # Utilisation de bincount pour un filtrage plus rapide
            points_per_voxel = torch.bincount(inverse_indices, minlength=mask_valid_voxels.shape[0])
            mask_points = points_per_voxel[inverse_indices] >= min_points_per_voxel

            # Mettre à jour batch.index_kept pour ne garder que ceux qui restent après TOUTES les suppressions
            batch.index_kept = torch.where(mask_points)[0]
            coords = coords[mask_points]
            features = features[mask_points]

            in_field = TensorField(
                features=features.to(device),
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            # Convert to a sparse tensor
            sinput = in_field.sparse()
            # sparse model output
            soutput = model_pred(sinput)
            # dense model output
            out_field = soutput.slice(in_field).F

            if conf.num_classes == 1:
                pred = (out_field > 0.5).float()
            else:
                pred = torch.argmax(out_field, axis=1).float()

            # writing the prediction
            las = laspy.read(test_tiles[j])

            # Créer un tableau de classification avec des zéros (classe 0 = invalide par défaut)
            final_classification = np.zeros(len(las.classification)+1, dtype=np.uint8)

            # Remettre les valeurs prédites aux bons indices
            final_classification[batch.index_kept.cpu().numpy()] = pred.view(-1).cpu().numpy().astype(np.uint8)

            # Supprimer le dernier point ajouté artificiellement
            final_classification = final_classification[:-1]

            # Appliquer la classification complète au nuage
            las.classification = final_classification
            las.write(savepath + os.path.basename(test_tiles[j]))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point cloud predictions with eclair model."
    )

    parser.add_argument(
        "--weight_path", type=str, help="Path to the weight file (.pth)"
    )

    parser.add_argument(
        "--config_file", type=str, help="Path to the config file (.yaml)"
    )

    parser.add_argument(
        "--pointclouds",
        type=str,
        help="Either the path to the .json file in a data folder (data folder should contains a folder 'pointclouds' "
        "and a .json file"
        "or the path to a folder of poinctlouds that will be all predicted",
    )

    parser.add_argument(
        "--save_path",
        default="/preds/",
        type=str,
        help="Path to th folder where you want the prediction to be saved",
    )

    args = parser.parse_args()

    predict(args.weight_path, args.config_file, args.pointclouds, args.save_path)


if __name__ == "__main__":
    main()
