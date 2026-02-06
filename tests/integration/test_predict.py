from pathlib import Path

import numpy as np
import pytest
from geosat.cloud import PointCloud

from eclair.plos.predict.predict import predict_dual_model


_TEST_DATA_PATH = Path("/mnt/r/6_gitlab/integration_continue/eclair")


def test_predict_given_base_when_all_clouds(tmp_path: Path) -> None:
    input_clouds_path = _TEST_DATA_PATH / "clouds" / "plo_gt"
    predict_dual_model(
        str(_TEST_DATA_PATH / "classifiers" / "EclairBaseClassifier" / "weights.pth"),
        str(_TEST_DATA_PATH / "classifiers" / "EclairPLOClassifier" / "weights.pth"),
        Path("src/eclair/plos/predict/config.yaml"),
        input_clouds_path,
        tmp_path,
        device_name="cpu",
    )
    cloud_paths = list(input_clouds_path.glob("*.la[sz]"))
    assert len(cloud_paths) > 0
    for cloud_path in cloud_paths:
        gt_cloud = PointCloud.read_from_path(cloud_path)
        pred_cloud = PointCloud.read_from_path(tmp_path / cloud_path.name)
        assert len(gt_cloud) == len(pred_cloud)
        assert np.all(gt_cloud.classification == pred_cloud.classification)
