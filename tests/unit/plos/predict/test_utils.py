from enum import IntEnum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import laspy
import numpy as np
import numpy.typing as npt
import pytest
from laspy import PackedPointRecord
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from eclair.plos.model.utils import las2pyg
from eclair.plos.predict.plo_post_processing import PLOPostProcessing
from eclair.plos.predict.dataset import TestDataset


class ClassId(IntEnum):
    UNCLASSIFIED = 0
    GROUND = 1
    TRUNK = 2
    VEGETATION = 3
    BUILDING = 4
    POLE = 5
    SIGN = 6
    BOLLARD = 7
    VERTICAL_OBJECT = 8


# Fixtures
@pytest.fixture(name="sample_config")
def fixture_sample_config() -> DictConfig:
    config = DictConfig({
        "test_transforms": [],
        "classes": {
            "unclassified": ClassId.UNCLASSIFIED,
            "ground": ClassId.GROUND,
            "trunk": ClassId.TRUNK,
            "vegetation": ClassId.VEGETATION,
            "building": ClassId.BUILDING,
            "pole": ClassId.POLE,
            "sign": ClassId.SIGN,
            "bollard": ClassId.BOLLARD,
            "vertical_object": ClassId.VERTICAL_OBJECT,
        },
        "dbscan": {"eps": 1.0, "min_samples": 2},
    })
    return config


@pytest.fixture(name="sample_las_data")
def fixture_sample_las_data() -> MagicMock:
    las_mock = MagicMock(spec=laspy.LasData)
    # Créer un tableau structuré avec les bons champs et formes
    points = np.zeros(10, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("classification", "u1")])
    points["x"] = np.random.rand(10) * 1000
    points["y"] = np.random.rand(10) * 1000
    points["z"] = np.random.rand(10) * 1000
    points["classification"] = np.random.randint(0, 5, 10).astype(np.uint8)

    las_mock.points = points
    las_mock.point_format.dimension_names = ["x", "y", "z", "classification"]

    # xyz correctement défini
    las_mock.xyz = np.vstack((points["x"], points["y"], points["z"])).T

    # Autres attributs avec les bonnes longueurs
    las_mock.intensity = np.random.randint(0, 256, 10).astype(np.uint16)
    las_mock.classification = points["classification"]
    las_mock.return_number = np.random.randint(1, 3, 10).astype(np.uint8)
    las_mock.number_of_returns = np.random.randint(1, 4, 10).astype(np.uint8)
    las_mock.edge_of_flight_line = np.random.randint(0, 2, 10).astype(np.uint8)
    las_mock.instance = np.arange(10).astype(np.int32)
    las_mock.red = np.random.randint(0, 256, 10).astype(np.uint16)
    las_mock.green = np.random.randint(0, 256, 10).astype(np.uint16)
    las_mock.blue = np.random.randint(0, 256, 10).astype(np.uint16)

    # Simule __getitem__ pour las["classification"], etc.
    def getitem(key: str) -> Any:
        return getattr(las_mock, key)

    las_mock.__getitem__.side_effect = getitem

    return las_mock


@pytest.fixture(name="sample_las_file")
def fixture_sample_las_file(tmp_path: Path, sample_las_data: Data) -> Path:
    las_path = tmp_path / "test.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.point_count = 10
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]

    las = laspy.LasData(header)
    las.points = PackedPointRecord.zeros(10, header.point_format)

    # Correction ici : shape 1D
    las.X = (sample_las_data.xyz[:, 0] / header.scales[0]).astype(np.int32)
    las.Y = (sample_las_data.xyz[:, 1] / header.scales[1]).astype(np.int32)
    las.Z = (sample_las_data.xyz[:, 2] / header.scales[2]).astype(np.int32)

    las.classification = sample_las_data.classification
    las.intensity = sample_las_data.intensity
    las.return_number = sample_las_data.return_number
    las.number_of_returns = sample_las_data.number_of_returns
    las.edge_of_flight_line = sample_las_data.edge_of_flight_line
    las.red = sample_las_data.red
    las.green = sample_las_data.green
    las.blue = sample_las_data.blue

    if hasattr(sample_las_data, "instance"):
        las.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.int32))
        las["instance"] = sample_las_data.instance

    las.write(las_path)
    return las_path


@pytest.fixture(name="sample_data")
def fixture_sample_data(sample_las_file: laspy.LasData) -> Data:
    return las2pyg(laspy.read(sample_las_file), sample_las_file)


@pytest.fixture(name="sample_predictions")
def fixture_sample_predictions(sample_data: Data) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    num_points = sample_data.xyz.shape[0]
    return (
        np.random.randint(0, 5, num_points, dtype=np.int32),  # full_pred_base
        np.random.randint(2, 9, num_points, dtype=np.int32),  # full_pred_plo
    )


# Tests pour TestDataset
def test_test_dataset_init(sample_config: DictConfig, tmp_path: Path) -> None:
    fnames = [tmp_path / "test.las"]
    dataset = TestDataset(sample_config, fnames)
    assert len(dataset.fnames) == 1
    assert dataset.fnames[0] == fnames[0]
    assert isinstance(dataset.transforms, Compose)


def test_test_dataset_getitem(sample_las_file: laspy.LasData, sample_config: DictConfig, sample_data: Data) -> None:
    # Use the path from sample_las_file, which already exists
    fnames = [sample_las_file]
    dataset = TestDataset(sample_config, fnames)
    data = dataset[0]

    assert isinstance(data, Data)
    assert data.xyz.shape == sample_data.xyz.shape  # Verify the shape matches the expected sample_data


def test_test_dataset_len(sample_config: DictConfig, tmp_path: Path) -> None:
    fnames = [tmp_path / f"test_{i}.las" for i in range(3)]
    dataset = TestDataset(sample_config, fnames)
    assert len(dataset) == 3


# Tests pour post_traitement_et_fusion
def test_postprocessing_and_fusion(
    sample_predictions: npt.NDArray[np.int32], sample_data: Data, sample_config: DictConfig
) -> None:
    full_pred_base, full_pred_plo = sample_predictions
    coords = sample_data.xyz.numpy()
    full_pred_plo_before = full_pred_plo.copy()
    postprocessing = PLOPostProcessing.from_config(sample_config)
    result = postprocessing.execute(full_pred_base, full_pred_plo, coords)
    assert np.all(full_pred_plo == full_pred_plo_before), (
        "`postprocessing_and_fusion` unexpectedly modified `full_pred_plo`"
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == full_pred_base.shape
    # Vérifier que les classes remappées respectent la config
    unique_values = np.unique(result)
    class_ids = set(sample_config.classes.values())
    assert all(val in class_ids or val == 0 for val in unique_values)


def test_postprocessing_and_fusion_no_vertical_objects(
    sample_predictions: npt.NDArray[np.int32], sample_data: Data, sample_config: DictConfig
) -> None:
    full_pred_base, full_pred_plo = sample_predictions
    full_pred_plo[:] = 0  # Pas d'objets verticaux
    coords = sample_data.xyz.numpy()
    postprocessing = PLOPostProcessing.from_config(sample_config)
    result = postprocessing.execute(full_pred_base, full_pred_plo, coords)
    assert np.array_equal(result, full_pred_base)  # Pas de changement si pas d'objets verticaux


# Tests pour _cluster_vertical_objects
def test_cluster_vertical_objects(sample_config: DictConfig) -> None:
    coords = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.1], [2.0, 2.0, 2.0]])
    postprocessing = PLOPostProcessing(classes=sample_config.classes, eps=0.5, min_samples=2)
    labels = postprocessing._cluster_vertical_objects(coords)
    assert len(labels) == 3
    assert np.all(labels[[0, 1]] == 0)  # Points proches forment un cluster
    assert labels[2] == -1  # Point isolé est bruit


@pytest.mark.parametrize(
    "cluster_labels, expected_pred",
    [
        (
            np.array([ClassId.POLE, ClassId.BOLLARD, ClassId.SIGN]),
            [ClassId.POLE, ClassId.POLE, ClassId.SIGN],  # potelet → poteau à cause du panneau
        ),
        (
            np.array([ClassId.POLE, ClassId.BOLLARD, ClassId.BOLLARD]),
            [ClassId.BOLLARD, ClassId.BOLLARD, ClassId.BOLLARD],  # poteau → potelet par majorité
        ),
    ],
)
def test_fix_pred_2_cluster(
    sample_config: DictConfig, cluster_labels: npt.NDArray[np.uint8], expected_pred: list[int]
) -> None:
    cluster_indices = np.array([0, 1, 2])

    postprocessing = PLOPostProcessing.from_config(sample_config)
    postprocessing._fix_pred_2_cluster(cluster_labels, cluster_indices, cluster_labels)
    assert np.all(cluster_labels[cluster_indices] == expected_pred)


@pytest.mark.parametrize(
    "base_set, is_fix_needed",
    [
        ({ClassId.TRUNK, ClassId.VEGETATION}, False),
        ({ClassId.TRUNK, ClassId.VERTICAL_OBJECT}, True),
        ({ClassId.BUILDING}, False),
        ({ClassId.BUILDING, ClassId.VERTICAL_OBJECT}, True),
        ({ClassId.VERTICAL_OBJECT}, True),
    ],
)
def test_is_fused_pred_fix_needed(sample_config: DictConfig, base_set: set[int], is_fix_needed: bool) -> None:
    postprocessing = PLOPostProcessing.from_config(sample_config)
    assert postprocessing._is_fused_pred_fix_needed(base_set) == is_fix_needed


@pytest.mark.parametrize(
    "cluster_labels, expected_pred",
    [
        pytest.param(
            [ClassId.UNCLASSIFIED, ClassId.GROUND, ClassId.BUILDING],
            [ClassId.UNCLASSIFIED, ClassId.UNCLASSIFIED, ClassId.UNCLASSIFIED],
            id="unaffected-classes",
        ),
        pytest.param(  # POLE and SIGN are the only two classes that are affected:
            [ClassId.UNCLASSIFIED, ClassId.POLE, ClassId.SIGN],
            [ClassId.UNCLASSIFIED, ClassId.POLE, ClassId.SIGN],
            id="affected-classes",
        ),
    ],
)
def test_fix_fused_pred_cluster(
    sample_config: DictConfig, cluster_labels: list[ClassId], expected_pred: list[ClassId]
) -> None:
    fused_pred = np.array([ClassId.UNCLASSIFIED, ClassId.UNCLASSIFIED, ClassId.UNCLASSIFIED])
    cluster_indices = np.array([0, 1, 2])
    postprocessing = PLOPostProcessing.from_config(sample_config)

    postprocessing._fix_fused_pred_cluster(fused_pred, cluster_indices, np.array(cluster_labels))
    assert np.all(fused_pred == expected_pred)


@pytest.mark.parametrize(
    "fused_pred,expected_result",
    [
        pytest.param(
            np.array([
                ClassId.VERTICAL_OBJECT,
                ClassId.BOLLARD,
                ClassId.BOLLARD,
                ClassId.VERTICAL_OBJECT,
                ClassId.BOLLARD,
            ]),
            np.array([ClassId.BOLLARD, ClassId.BOLLARD, ClassId.BOLLARD, ClassId.BOLLARD, ClassId.BOLLARD]),
            id="all-bollard",
        ),
        pytest.param(
            np.array([
                ClassId.VERTICAL_OBJECT,
                ClassId.POLE,
                ClassId.POLE,
                ClassId.VERTICAL_OBJECT,
                ClassId.VERTICAL_OBJECT,
            ]),
            np.array([ClassId.POLE, ClassId.POLE, ClassId.POLE, ClassId.POLE, ClassId.POLE]),
            id="all-pole",
        ),
    ],
)
def test_fusion_finale(
    sample_config: DictConfig, fused_pred: npt.NDArray[np.uint8], expected_result: npt.NDArray[np.uint8]
) -> None:
    coords = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.1], [0.2, 0.2, 1.2], [0.3, 0.3, 1.3], [0.4, 0.4, 1.4]])
    postprocessing = PLOPostProcessing.from_config(sample_config)
    result = postprocessing._final_fusion(fused_pred, coords)
    assert np.all(result == expected_result)
