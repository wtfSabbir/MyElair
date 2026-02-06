import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch.cuda
from geosat.cloud import PointCloud
from geosat.psanp.abstractions.pipelinemetadata import PipelineMetadata
from pytest import MonkeyPatch

from eclair.plos.psanp.classifiers import EclairClassifier

if platform.system() == "Windows":
    _TEST_DATA_PATH = Path("R:\6_gitlab\integration_continue\eclair")
else:
    _TEST_DATA_PATH = Path("/mnt/r/6_gitlab/integration_continue/eclair")
if not _TEST_DATA_PATH.is_dir():
    raise FileNotFoundError(
        f"The test data folder for eclair {_TEST_DATA_PATH} does not exist. Is the R&D NAS correctly mounted?"
    )
_BASE_CLASSIFICATION_CODES = {
    "unclassified": 0,
    "ground": 1,
    "vertical_object": 2,
    "trunk": 3,
    "vegetation": 4,
    "building": 5,
}
_ALL_CLASSIFICATION_CODES = {
    "unclassified": 0,
    "ground": 1,
    "trunk": 2,
    "vegetation": 3,
    "building": 4,
    "pole": 5,
    "sign": 6,
    "bollard": 7,
    "vertical_object": 8,
}


def test_eclair_classifier_supports_empty_cloud(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    classifier = EclairClassifier(stage_name="EclairBaseClassifier", device="cpu")
    cloud = PointCloud(
        [],
        fields={
            "classification": np.array([], dtype=np.uint8),
            "red": np.array([], dtype=np.uint16),
            "green": np.array([], dtype=np.uint16),
            "blue": np.array([], dtype=np.uint16),
        },
    )
    (output_cloud,) = classifier.execute_on(
        cloud, PipelineMetadata(source=".", classification_codes=_BASE_CLASSIFICATION_CODES)
    )
    assert len(output_cloud) == 0


def test_eclair_classifier_given_cuda_not_available_uses_cpu(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    with patch.object(torch.cuda, "is_available", return_value=False):
        classifier = EclairClassifier(stage_name="EclairBaseClassifier")
    assert classifier.device.type == "cpu"


def test_eclair_base_classifier_given_one_point(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    classifier = EclairClassifier(stage_name="EclairBaseClassifier", device="cpu")
    cloud = PointCloud(
        [[0.0, 0.0, 0.0]],
        fields={
            "classification": np.array([0], dtype=np.uint8),
            "red": np.array([0], dtype=np.uint16),
            "green": np.array([0], dtype=np.uint16),
            "blue": np.array([0], dtype=np.uint16),
        },
    )
    (output_cloud,) = classifier.execute_on(
        cloud, PipelineMetadata(source=".", classification_codes=_BASE_CLASSIFICATION_CODES)
    )
    assert len(output_cloud) == 1


def test_eclair_base_classifier_returns_expected_classification(monkeypatch: MonkeyPatch) -> None:
    gt_path = _TEST_DATA_PATH / "clouds" / "plo_gt"
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    classifier = EclairClassifier(stage_name="EclairBaseClassifier", device="cpu")
    metadata = PipelineMetadata(source=".", classification_codes=_BASE_CLASSIFICATION_CODES)
    cloud_paths = list(gt_path.glob("*.la[sz]"))
    assert len(cloud_paths) > 0
    for cloud_path in cloud_paths:
        gt_cloud = PointCloud.read_from_path(cloud_path)
        gt_field = gt_cloud.basemodel_v13.copy()
        gt_cloud.classification[:] = 0
        classifier.execute_on(gt_cloud, metadata)
        assert np.all(gt_cloud.classification == gt_field)


def test_eclair_classifier_remaps_classes(monkeypatch: MonkeyPatch) -> None:
    gt_path = _TEST_DATA_PATH / "clouds" / "plo_gt"
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    classifier = EclairClassifier(stage_name="EclairBaseClassifier", device="cpu")
    metadata_base = PipelineMetadata(source=".", classification_codes=_BASE_CLASSIFICATION_CODES)
    classification_codes_2 = {
        "unclassified": 2,
        "ground": 4,
        "vertical_object": 6,
        "trunk": 8,
        "vegetation": 10,
        "building": 12,
    }
    metadata_2 = PipelineMetadata(source=".", classification_codes=classification_codes_2)
    cloud_paths = list(gt_path.glob("*.la[sz]"))
    assert len(cloud_paths) > 0
    for cloud_path in cloud_paths:
        gt_cloud = PointCloud.read_from_path(cloud_path)
        gt_cloud.classification[:] = 0
        classifier.execute_on(gt_cloud, metadata_base)
        classification_base = gt_cloud.classification.copy()
        gt_cloud.classification[:] = 0
        classifier.execute_on(gt_cloud, metadata_2)
        classification_2 = gt_cloud.classification.copy()
        for class_name in _BASE_CLASSIFICATION_CODES:
            class_id_base = _BASE_CLASSIFICATION_CODES[class_name]
            class_id_2 = classification_codes_2[class_name]
            assert np.equal(classification_base == class_id_base, classification_2 == class_id_2).all()


def test_eclair_base_and_plo_classifier_return_expected_classification(monkeypatch: MonkeyPatch) -> None:
    gt_path = _TEST_DATA_PATH / "clouds" / "plo_gt"
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    base_classifier = EclairClassifier(stage_name="EclairBaseClassifier", device="cpu")
    plo_classifier = EclairClassifier(stage_name="EclairPLOClassifier", device="cpu")
    metadata = PipelineMetadata(source=".", classification_codes=_ALL_CLASSIFICATION_CODES)
    cloud_paths = list(gt_path.glob("*.la[sz]"))
    assert len(cloud_paths) > 0
    for cloud_path in cloud_paths:
        gt_cloud = PointCloud.read_from_path(cloud_path)
        gt_classification = gt_cloud.classification.copy()
        gt_cloud.classification[:] = 0
        base_classifier.execute_on(gt_cloud, metadata)
        plo_classifier.execute_on(gt_cloud, metadata)
        assert np.all(gt_cloud.classification == gt_classification)


def test_eclair_plo_classifier_supports_only_ground_cloud(monkeypatch: MonkeyPatch) -> None:
    """A non-regression test to verify that the EclairPLOClassifier does not raise a
    `ValueError` when classifying a point cloud with only ground.
    """
    monkeypatch.setenv("PSANP_BUNDLE_FOLDER", str(_TEST_DATA_PATH / "classifiers"))
    classifier = EclairClassifier(stage_name="EclairPLOClassifier", device="cpu")
    codes = _ALL_CLASSIFICATION_CODES
    cloud = PointCloud(
        [[0.0, 0.0, 0.0]],
        fields={
            "classification": np.array([codes["ground"]], dtype=np.uint8),
            "red": np.array([0], dtype=np.uint16),
            "green": np.array([0], dtype=np.uint16),
            "blue": np.array([0], dtype=np.uint16),
        },
    )
    (output_cloud,) = classifier.execute_on(cloud, PipelineMetadata(source=".", classification_codes=codes))
    assert len(output_cloud) == 1
