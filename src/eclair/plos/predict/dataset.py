"""Fonctions utilisees pour la prediction uniquement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import laspy
import torch

from eclair.plos.model.functional import compose_transforms_from_list
from eclair.plos.model.utils import las2pyg

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch_geometric.data import Data


class TestDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """TestDataset class."""

    def __init__(self, config: DictConfig, fnames: list[Path]) -> None:
        """
        Initialize the test dataset.

        :param config: Configuration object containing test parameters.
        :param fnames: List of paths to the test data files.
        """
        self.fnames: list[Path] = fnames
        self.transforms = compose_transforms_from_list(config.test_transforms)

    def __getitem__(self, index: int) -> Data:
        """
        Get an item from the dataset.

        :param index: Index of the item to retrieve.
        :return: The retrieved data item.
        """
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data = las2pyg(las, Path(tile_path))
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        return pyg_data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: The number of items in the dataset.
        """
        return len(self.fnames)
