import gc
import json
import os
from typing import List, Dict
from torch_geometric.loader import DataLoader as PyGDataLoader
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)

from utils import (

    seed_everything,
    TrainDataset_save,
    TrainDataset_supprvoxel_save, TrainDataset_supprvoxel_save_trainset,

)
import torch
from omegaconf import OmegaConf

def make_data(conf) -> PyGDataLoader:
    """
    Read the .json file with the name of the pointclouds and their repartition in train/val.
    Apply the transformation (data-aug, normalization) and create the train and val dataloaders.

    :param conf: Dictionary from train.yaml
    :return: train and validation dataloader for the training and validation steps.
    """
    label_file = "/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v13/labels_v13_horssol.json"
    # label_file = "/mnt/d/pointcloud_data/iantsa/reanno_16c/gt/labels_iantsa16c.json"
    with open(label_file, encoding="utf-8") as f:
        all_tiles: List[Dict] = json.load(f)

    train_tiles = [
        "/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v13/pointclouds_horssol/" + "/" + x["tile_name"]
        # "/mnt/d/pointcloud_data/iantsa/reanno_16c/gt/pointclouds" + "/" + x["tile_name"]
        for x in all_tiles
        if x["split"] == "val"
    ]

    if len(train_tiles) == 0:
        raise ValueError(
            "No training data. Please verify your labels.json file. Pointclouds should be in data/pointclouds"
        )


    dataset=TrainDataset_supprvoxel_save_trainset(conf, train_tiles, output_dir="/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v13/pointclouds_horssol_filtered",min_point_per_voxel=15)
# ⚠️ Ajout de cette boucle pour forcer l'application des transformations et la sauvegarde
    for i in range(len(dataset)):
        data = dataset[i]  # Charge et applique les transformations
        print(f"✅ {i+1}/{len(dataset)} : Fichier {train_tiles[i]} traité et sauvegardé.")

def main() -> None:
    from_cli = OmegaConf.create()
    base_conf = OmegaConf.load("eclair_version/v15/ex_v15.yaml")
    conf = OmegaConf.merge(base_conf, from_cli)

    seed_everything(conf.random_seed)
    make_data(conf)


if __name__ == "__main__":
    main()
