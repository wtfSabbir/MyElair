"""Fonctions de manipulation des nuages."""

import os
import random
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import torch
from torch_geometric.data import Batch, Data


def las2pyg(las: laspy.LasData, path: Path) -> Data:
    """
    Convert LAS data to a PyTorch Geometric Data object.

    :param las: The LAS data to be converted.
    :param path: The path to the LAS file.
    :return: The converted PyTorch Geometric Data object.

    Example
    --------
    >>> las_ = laspy.read("path/to/lasfile.las")
    >>> data = las2pyg(las_, Path("path/to/lasfile.las"))
    """
    gt_key: str = "classification" if "classification" in set(las.point_format.dimension_names) else "pred"
    return Data(
        xyz=torch.from_numpy(las.xyz.copy()),
        intensity=torch.from_numpy(las.intensity.astype(np.int64)),
        classification=torch.from_numpy(np.array(las[gt_key])).long(),
        return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
        number_of_returns=torch.from_numpy(np.asarray(las.number_of_returns)).long(),
        edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
        instance_id=(
            torch.from_numpy(np.asarray(las.instance).copy().astype(np.int64)).long()
            if hasattr(las, "instance")
            else torch.full((len(las.return_number),), fill_value=-1, dtype=torch.long)
        ),
        rgb=(
            torch.stack(
                [
                    torch.from_numpy(las.red.astype(np.int64)),
                    torch.from_numpy(las.green.astype(np.int64)),
                    torch.from_numpy(las.blue.astype(np.int64)),
                ],
                dim=-1,
            ).long()
            if hasattr(las, "red")
            else None
        ),
        filename=path,
    )


def collate_custom_test(batch: Batch) -> Any:  # noqa: ANN401
    """
    Create custom collate function for test data.

    :param batch: A list of items to be collated.
    :return: The collated batch.
    """
    batch = [item[:3] for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def seed_everything(seed: int) -> None:
    """
    Fix a random seed for Numpy, PyTorch, and CUDA to improve reproducibility of DL pipelines.

    :param seed: The seed to fix.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
