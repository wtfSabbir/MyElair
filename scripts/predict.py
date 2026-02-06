"""Predict pointclouds with dual models."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import numpy.typing as npt
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)
from model import *
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from utils import TestDataset, collate_custom_test, seed_everything


def load_config_and_models(weights: str, config: Path, device: torch.device) -> tuple:
    """Load configuration and initialize models.

    :param weights: Path to weights for the Mink model.
    :param config: Path to the configuration file (YAML format).
    :param device: Device to run the models on (CPU or GPU).
    :return: Tuple containing configuration and initialized models (conf, model1, model2).
    """
    conf = OmegaConf.merge(OmegaConf.load(config), OmegaConf.create())
    try:
        model_class = MODEL_REGISTRY[conf.model_name]
    except KeyError:
        raise ValueError(f"Modèle inconnu : {conf.model_name}")

    model_ = model_class(conf.num_features, conf.num_classes, conf)
    model_.load_state_dict(torch.load(weights))
    model_.to(device)

    return conf, model_


def load_pointcloud_paths(pointclouds: Path) -> list[Path]:
    """Load paths to point cloud files.

    :param pointclouds: Path to a JSON file or directory containing point clouds.
    :return: List of paths to point cloud files.
    """
    if Path(pointclouds).suffix == ".json":
        with Path(pointclouds).open(encoding="utf-8") as f:
            all_tiles = json.load(f)
        return [Path(pointclouds).parent / "pointclouds" / x["tile_name"] for x in all_tiles if x["split"] == "val"]
    return list(Path(pointclouds).glob("*.la[sz]"))


def create_dataloader(conf: ListConfig | DictConfig, pointcloud_paths: list[Path]) -> PyGDataLoader:
    """Create a DataLoader for point cloud data.

    :param conf: Configuration object.
    :param pointcloud_paths: List of paths to point cloud files.
    :return: DataLoader for the test dataset.
    """
    dataset = TestDataset(conf, pointcloud_paths)
    return PyGDataLoader(
        dataset,
        batch_size=conf.test_batch_size,
        collate_fn=collate_custom_test,
        pin_memory=False,
    )


def predict_base_model(model, batch: Batch, inv_voxel_size: float, device: torch.device) -> npt.NDArray[np.int64]:
    """Run ground/non-ground classification using the first model.

    :param model: The Mink model.
    :param batch: Batch of point cloud data.
    :param inv_voxel_size: Inverse of the voxel size for coordinate quantization.
    :param device: Device to run the model on.
    :return: Predicted ground/non-ground labels.
    """
    coords = torch.cat(
        [batch.batch.unsqueeze(1), torch.floor(batch.pos * inv_voxel_size)],
        dim=1,
    )
    features = batch.x.to(device)

    features = features.unsqueeze(1) if features.dim() == 1 else features

    field = TensorField(
        features=features,
        coordinates=coords.to(device),
        quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
    )
    out = model(field.sparse()).slice(field).F
    return torch.argmax(out, dim=1).cpu().numpy()


def save_predictions(  # noqa: PLR0913, PLR0917
    las: laspy.LasData,
    pred1: npt.NDArray[np.int64],
    indices_kept: npt.NDArray[np.int64],
    index_quantile_excluded: npt.NDArray[np.int64],
    savepath: Path,
    tile_path: Path,
) -> None:
    """Save predictions to a LAS/LAZ file.

    :param las: Input LAS file.
    :param pred1: Predictions from the first model.
    :param indices_kept: Indices of points kept after preprocessing.
    :param index_quantile_excluded: Indices of excluded points.
    :param savepath: Directory to save the output file.
    :param tile_path: Path to the input tile.
    """
    n_points = len(las.classification)
    full_pred1 = np.zeros(n_points + 1, dtype=np.uint8)

    full_pred1[indices_kept] = pred1
    full_pred1[index_quantile_excluded] = 1

    full_pred1 = full_pred1[:-1]

    if "eclair" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="eclair", type=np.uint8))

    las.eclair = full_pred1

    las.write(Path(savepath) / Path(tile_path).name)


def predict_model(weights: str, config: Path, pointclouds: Path, savepath: Path) -> None:
    """
    Predicts classifications for point clouds using two models and saves results to LAS/LAZ files.

    This function loads the Mink model, predict the pointclouds and saves
    the results with additional fields in LAS/LAZ files.

    :param weights: Path to the weights file for the Mink model).
    :param config: Path to the configuration file (YAML format).
    :param pointclouds: Path to a JSON file containing point cloud metadata or a directory with LAS/LAZ files.
    :param savepath: Directory where the output LAS/LAZ files will be saved.
    :raises FileNotFoundError: If the weights, config, or point cloud files cannot be found.
    :raises ValueError: If the JSON file format is invalid or required fields are missing.
    :raises ValueError: If the tile doesn't contain enough points to be predicted by the models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration and models
    conf, model1 = load_config_and_models(weights, config, device)
    seed_everything(conf.random_seed)

    # Load point cloud paths and create dataloader
    pointcloud_paths = load_pointcloud_paths(pointclouds)
    dataloader = create_dataloader(conf, pointcloud_paths)

    inv_voxel_size = 1.0 / conf.voxel_size

    # Process each batch
    with torch.no_grad():
        for j, batch in enumerate(tqdm(dataloader, desc="Predicting pointclouds")):
            # Predict with first model (ground/non-ground)
            try:
                pred1 = predict_base_model(model1, batch, inv_voxel_size, device)
            # Si la tuile ne contient pas assez de points pour être en entrée du modèle, on met tous les champs à zero
            except ValueError as e:
                if "Expected more than 1 value per channel" in str(e):
                    n_points = batch.pos.shape[0]
                    pred1 = np.zeros(n_points, dtype=np.uint8)
                else:
                    raise  # Si c'est une autre erreur, on ne masque pas !

            # Save results
            las = laspy.read(pointcloud_paths[j])
            save_predictions(
                las,
                pred1,
                batch.index_kept.cpu().numpy(),
                batch.index_quantile_excluded.cpu().numpy(),
                savepath,
                pointcloud_paths[j],
            )


def main() -> None:
    """Run the prediction."""
    parser = argparse.ArgumentParser(description="Dual-model point cloud prediction")
    parser.add_argument("--weight_path", type=str, help="Path to the model weights")
    parser.add_argument("--config_file", type=str, help="Config file used for training")
    parser.add_argument("--pointclouds", type=str, help="Path to input .json or folder with pointclouds to predict")
    parser.add_argument("--save_path", default="/preds/", type=str, help="Where to save .laz outputs")

    args = parser.parse_args()

    predict_model(
        args.weight_path,
        args.config_file,
        args.pointclouds,
        args.save_path,
    )


if __name__ == "__main__":
    main()
