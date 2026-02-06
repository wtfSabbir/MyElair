import argparse
import os
import laspy
import glob
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
from utilis_10c import TestDataset, seed_everything, collate_custom_test
from omegaconf import OmegaConf
from model_10c import MinkUNet14C, Binary_model
import time


def predict_and_fuse(
    weights_curbgutt, config_curbgutt, weights_10c, config_10c, pointclouds, savepath
) -> None:
    """
    Process each pointcloud, predict using both models, fuse predictions, and save result.
    :param weights_curbgutt: path to the weights file for curbgutt model
    :param config_curbgutt: path to config for curbgutt model
    :param weights_10c: path to the weights file for 10c model
    :param config_10c: path to config for 10c model
    :param pointclouds: path to the folder containing .laz files
    :param savepath: path where the final predictions will be saved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reading the config files and loading models
    from_cli = OmegaConf.create()
    base_conf_curbgutt = OmegaConf.load(config_curbgutt)
    conf_curbgutt = OmegaConf.merge(base_conf_curbgutt, from_cli)
    model_pred_curbgutt = MinkUNet14C(
        conf_curbgutt.num_features, conf_curbgutt.num_classes
    )
    if conf_curbgutt.num_classes == 1:
        model_pred_curbgutt = Binary_model(model_pred_curbgutt)
    model_pred_curbgutt.load_state_dict(torch.load(weights_curbgutt))

    base_conf_10c = OmegaConf.load(config_10c)
    conf_10c = OmegaConf.merge(base_conf_10c, from_cli)
    model_pred_10c = MinkUNet14C(conf_10c.num_features, conf_10c.num_classes)
    if conf_10c.num_classes == 1:
        model_pred_10c = Binary_model(model_pred_10c)
    model_pred_10c.load_state_dict(torch.load(weights_10c))

    # Seed and set models to evaluation mode
    seed_everything(conf_curbgutt.random_seed)
    model_pred_curbgutt.eval()
    model_pred_10c.eval()

    # Get list of pointclouds
    try:
        test_tiles: List[str] = glob.glob(pointclouds + "*.la[sz]")
    except FileNotFoundError as exception:
        raise FileNotFoundError(
            f"No pointcloud files found for: {pointclouds}."
        ) from exception

    dataset = TestDataset(conf_curbgutt, test_tiles)
    dataloader = PyGDataLoader(
        dataset,
        batch_size=conf_curbgutt.test_batch_size,
        collate_fn=collate_custom_test,
        pin_memory=False,
    )

    model_pred_curbgutt.to(device)
    model_pred_10c.to(device)

    with torch.no_grad():
        for j, batch in enumerate(
            tqdm(dataloader, desc="Predicting and fusing pointclouds")
        ):
            coords, features, batch_idx = (
                batch.pos / conf_curbgutt.voxel_size,
                batch.x,
                batch.batch,
            )
            coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

            # Processing with the curbgutt model
            in_field_curbgutt = TensorField(
                features=features.to(device),
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            sinput_curbgutt = in_field_curbgutt.sparse()
            soutput_curbgutt = model_pred_curbgutt(sinput_curbgutt)
            out_field_curbgutt = soutput_curbgutt.slice(in_field_curbgutt).F
            pred_curbgutt = (
                (out_field_curbgutt > 0.5).float()
                if conf_curbgutt.num_classes == 1
                else torch.argmax(out_field_curbgutt, axis=1).float()
            )

            # Processing with the 10c model
            in_field_10c = TensorField(
                features=features.to(device),
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            sinput_10c = in_field_10c.sparse()
            soutput_10c = model_pred_10c(sinput_10c)
            out_field_10c = soutput_10c.slice(in_field_10c).F
            pred_10c = (
                (out_field_10c > 0.5).float()
                if conf_10c.num_classes == 1
                else torch.argmax(out_field_10c, axis=1).float()
            )

            # Fusion des prédictions
            pred_final = np.zeros_like(pred_curbgutt.view(-1).cpu().numpy())
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 1)] = 1
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 4)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 5)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 6)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 7)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 8)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 9)] = 0
            pred_final[np.where(pred_10c.view(-1).cpu().numpy() == 0)] = 0
            pred_final[np.where(pred_curbgutt.view(-1).cpu().numpy() == 1)] = 1
            pred_final[np.where(pred_curbgutt.view(-1).cpu().numpy() == 2)] = 1

            # Sauvegarde du nuage de points avec la classification fusionnée
            test_tile = test_tiles[j]
            las = laspy.read(test_tile)
            las.classification = pred_final.astype(np.uint8)
            las.write(savepath + os.path.basename(test_tile))


def main() -> None:
    time1 = time.time()

    weight_path_curbgutt = "/mnt/d/pointcloud_data/iantsa/reanno_16c/12test/iantsa_reanno71_e500_vxl0002_intrgb_b1_newloss_invprob_3c_curbgutt_bestvallossweights.pth"
    weight_path_10c = "/mnt/d/pointcloud_data/iantsa/reanno_16c/12test/iantsa_reanno71_e400_vxl0003_intrgb_b1_newloss_invprob_10c_bestvallossweights.pth"
    config_file_curbgutt = (
        "/home/mtardif/src/MinkowskiEngine/fuze_training/curbgutt.yaml"
    )
    config_file_10c = (
        "/home/mtardif/src/MinkowskiEngine/fuze_training/10c.yaml"
    )
    pointclouds = "/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v13_8c/"
    save_path = "/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v14/pred_sol/"

    # Prediction, fusion, and save for each point cloud
    predict_and_fuse(
        weight_path_curbgutt,
        config_file_curbgutt,
        weight_path_10c,
        config_file_10c,
        pointclouds,
        save_path,
    )

    time2 = time.time()
    print(f"tps: {time2 - time1}")


if __name__ == "__main__":
    main()
