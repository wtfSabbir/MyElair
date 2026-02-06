"""Predict pointclouds with dual models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import laspy
import numpy as np
import numpy.typing as npt
import torch
from MinkowskiEngine import MinkowskiAlgorithm, SparseTensorQuantizationMode, TensorField
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from eclair.plos.model.model import Binary_model, MinkUNet14C, MinkUNet34A
from eclair.plos.model.utils import collate_custom_test, seed_everything
from eclair.plos.predict.dataset import TestDataset
from eclair.plos.predict.plo_post_processing import PLOPostProcessing

if TYPE_CHECKING:
    from torch_geometric.data import Batch


def _check_classes(config: DictConfig) -> None:
    classes = set(config.classes)
    for model_classes_attr_name in ("classes_base", "classes_plo"):
        missing_classes = set(config[model_classes_attr_name]) - classes
        if missing_classes:
            msg = (
                f"Some classes referenced in the '{model_classes_attr_name}' section of "
                f"the config are missing in the 'classes' section: {missing_classes}"
            )
            raise ValueError(msg)


def load_config(config: Path) -> DictConfig:
    """Load configuration from a YAML file."""
    conf = OmegaConf.merge(OmegaConf.load(config), OmegaConf.create())
    if not isinstance(conf, DictConfig):
        msg = f"The configuration {config} is expected to be loaded as a dictionary, found: {type(conf)}"
        raise TypeError(msg)
    return conf


def load_config_and_models(
    weights1: str, weights2: str, config: Path, device: torch.device
) -> tuple[DictConfig, MinkUNet34A, MinkUNet14C]:
    """Load configuration and initialize models.

    :param weights1: Path to weights for the first model (basemodel, MinkUNet34A).
    :param weights2: Path to weights for the second model (PLO, MinkUNet14C).
    :param config: Path to the configuration file (YAML format).
    :param device: Device to run the models on (CPU or GPU).
    :return: Tuple containing configuration and initialized models (conf, model_base, model_plo).
    """
    conf = load_config(config)
    _check_classes(conf)

    model_base = MinkUNet34A(conf.num_features, len(conf.classes_base))
    model_base.load_state_dict(torch.load(weights1))
    model_base.to(device)

    model_plo = MinkUNet14C(conf.num_features, len(conf.classes_plo))
    model_plo.load_state_dict(torch.load(weights2))
    model_plo.to(device)

    return conf, model_base, model_plo


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


def create_dataloader(conf: DictConfig, pointcloud_paths: list[Path]) -> PyGDataLoader:
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


def predict_model(
    model: nn.Module,
    batch: Batch,
    voxel_size: float,
    device: torch.device,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.uint8]:
    """Run a classification using a Minkowski model.

    :param model: A classification model.
    :param batch: Batch of point cloud data.
    :param voxel_size: Voxel size for coordinate quantization.
    :param device: Device to run the model on.
    :param mask: Boolean mask indicating the elements to send to the model.
    :return: Predicted labels, which is a 1D array of the same length as ``batch.classification``.
    """

    def predict(features: torch.Tensor, batch_idx: torch.Tensor, pos: torch.Tensor) -> npt.NDArray[np.uint8]:
        coords = torch.cat(
            [batch_idx.unsqueeze(1), torch.floor(pos * (1.0 / voxel_size))],
            dim=1,
        )

        field = TensorField(
            features=features.to(device),
            coordinates=coords.to(device),
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
        )
        out = model(field.sparse()).slice(field).F
        if isinstance(model, Binary_model):
            return torch.gt(out, 0.5).squeeze().to(torch.uint8).cpu().numpy()
        return torch.argmax(out, dim=1).to(torch.uint8).cpu().numpy()

    if mask is None:
        return predict(features=batch.x, batch_idx=batch.batch, pos=batch.pos)
    pred = np.zeros_like(mask, dtype=np.uint8)
    if np.count_nonzero(mask) <= 1:  # Need at least 2 points to predict
        return pred
    pred[mask] = predict(features=batch.x[mask], batch_idx=batch.batch[mask], pos=batch.pos[mask])
    return pred


def format_predictions(  # noqa: PLR0917,PLR0913
    pred_base: npt.NDArray[np.uint8],
    pred_plo: npt.NDArray[np.uint8],
    indices_kept: npt.NDArray[np.int64],
    index_quantile_excluded: npt.NDArray[np.int64],
    n_points: int,
    unclassified_id: int,
    ground_id: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Create classification arrays of size ``n_points`` from ``pred_base`` and ``pred_plo``.

    :param pred_base: Predictions from the first model.
    :param pred_plo: Predictions from the second model.
    :param indices_kept: Indices of points kept after preprocessing.
    :param index_quantile_excluded: Indices of excluded points.
    :param n_points: Number of points in the point cloud.
    :param ground_id: ID of the ground class.
    :return: Tuple containing two arrays of labels of size ``n_points``.
    """
    # Adding one more point because of the virtual max point added when creating the batch:
    full_pred_base = np.zeros(n_points + 1, dtype=np.uint8)
    full_pred_base[indices_kept] = pred_base
    full_pred_base[index_quantile_excluded] = ground_id

    full_pred_plo = np.zeros(n_points + 1, dtype=np.uint8)
    full_pred_plo[indices_kept] = np.where(pred_base == ground_id, unclassified_id, pred_plo)

    return full_pred_base[:-1], full_pred_plo[:-1]


def fill_predictions(
    las: laspy.LasData,
    pred_base: npt.NDArray[np.uint8],
    pred_plo: npt.NDArray[np.uint8],
    fused_pred: npt.NDArray[np.uint8],
) -> None:
    """Fill a LAS/LAZ point cloud with the predictions.

    :param las: Input LAS file.
    :param pred_base: Predictions from the first model.
    :param pred_plo: Predictions from the second model.
    :param fused_pred: Merged predictions from the two models.
    """

    def create_missing_field(name: str) -> None:
        if name not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))

    create_missing_field("basemodel_v13")
    create_missing_field("plo_v13")
    create_missing_field("fuze")
    las.basemodel_v13 = pred_base
    las.plo_v13 = pred_plo
    las.fuze = fused_pred
    las.classification = fused_pred


def build_mapping(
    conf: DictConfig, model_classes_attr_name: Literal["classes_base", "classes_plo"]
) -> npt.NDArray[np.uint8]:
    """Build a mapping from class IDs to class names."""
    classes: list[str] = conf[model_classes_attr_name]
    return np.array([conf.classes[class_name] for class_name in classes], dtype=np.uint8)


def predict_dual_model(  # noqa: PLR0917,PLR0913,PLR0914
    weights1: str,
    weights2: str,
    config: Path,
    pointclouds: Path,
    savepath: Path,
    device_name: Literal["cuda", "cpu"] | None = "cuda",
) -> None:
    """
    Predicts classifications for point clouds using two models and saves results to LAS/LAZ files.

    This function loads two pre-trained models (MinkUNet34A for base classification and
    MinkUNet14C for PLO classification), processes point clouds from the specified input,
    applies a mask to focus on non-ground points for the second model, fuses the predictions, and saves
    the results with additional fields in LAS/LAZ files.

    :param weights1: Path to the weights file for the first model (MinkUNet34A).
    :param weights2: Path to the weights file for the second model (MinkUNet14C).
    :param config: Path to the configuration file (YAML format).
    :param pointclouds: Path to a JSON file containing point cloud metadata or a directory with LAS/LAZ files.
    :param savepath: Directory where the output LAS/LAZ files will be saved.
    :param device_name: Name of the device to run the models on (CPU or GPU).
        If None, the device will be selected automatically.
    :raises FileNotFoundError: If the weights, config, or point cloud files cannot be found.
    :raises ValueError: If the JSON file format is invalid or required fields are missing.
    :raises ValueError: If the tile doesn't contain enough points to be predicted by the models
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # Load configuration and models
    conf, model_base, model_plo = load_config_and_models(weights1, weights2, config, device)
    seed_everything(conf.random_seed)

    # Load point cloud paths and create dataloader
    pointcloud_paths = load_pointcloud_paths(pointclouds)
    dataloader = create_dataloader(conf, pointcloud_paths)
    postprocessing = PLOPostProcessing.from_config(conf)
    classes: dict[str, int] = conf.classes
    base_mapping = build_mapping(conf, "classes_base")
    plo_mapping = build_mapping(conf, "classes_plo")

    # Process each batch
    with torch.inference_mode():
        model_base.eval()
        model_plo.eval()
        for j, batch in enumerate(tqdm(dataloader, desc="Predicting pointclouds")):
            # Predict with first model (ground/non-ground)
            pred_base = predict_model(model_base, batch, conf.voxel_size, device)
            pred_base = base_mapping[pred_base]
            # Predict with second model (non-ground objects)
            mask_not_ground = pred_base != classes["ground"]
            pred_plo = predict_model(model_plo, batch, conf.voxel_size, device, mask=mask_not_ground)
            pred_plo = plo_mapping[pred_plo]

            # Save results
            las = laspy.read(pointcloud_paths[j])
            full_pred_base, full_pred_plo = format_predictions(
                pred_base,
                pred_plo,
                indices_kept=batch.index_kept.cpu().numpy(),
                index_quantile_excluded=batch.index_quantile_excluded.cpu().numpy(),
                n_points=len(las.classification),
                unclassified_id=classes["unclassified"],
                ground_id=classes["ground"],
            )
            fused_pred = postprocessing.execute(
                full_pred_base, full_pred_plo, coords=np.vstack((las.x, las.y, las.z)).T
            )
            fill_predictions(
                las,
                full_pred_base,
                full_pred_plo,
                fused_pred,
            )
            las.write(str(Path(savepath, pointcloud_paths[j].name)))


def main() -> None:
    """Run the prediction."""
    parser = argparse.ArgumentParser(description="Dual-model point cloud prediction")
    parser.add_argument("--weight_path1", type=str, help="Path to weights for model 1")
    parser.add_argument("--weight_path2", type=str, help="Path to weights for model 2")
    parser.add_argument("--config_file", type=str, help="Config file for model 2")
    parser.add_argument("--pointclouds", type=str, help="Path to input .json or folder with pointclouds")
    parser.add_argument("--save_path", default="/preds/", type=str, help="Where to save .laz outputs")

    args = parser.parse_args()

    predict_dual_model(
        args.weight_path1,
        args.weight_path2,
        args.config_file,
        args.pointclouds,
        args.save_path,
    )


if __name__ == "__main__":
    main()
