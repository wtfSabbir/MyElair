from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Importez le script à tester
from eclair.plos.predict.evaluate_mlflow import (
    apply_dbscan,
    calculate_object_metrics,
    evaluate,
    evaluate_single_file,
    load_laz,
    log_to_mlflow,
    main,
    remap_labels,
)


# Configuration pour les tests
@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Fichier de config evaluate_mlflow.json."""
    return {
        "class_names": {"2": "Tronc", "5": "Poteau", "6": "Panneau", "7": "Potelet"},
        "remap_dict": {"2": "5", "3": "6", "4": "7", "5": "2", "6": "3", "7": "4", "8": "4", "9": "4", "10": "4"},
        "mlflow_uri": "https://example.com/mlflow/",
        "mlflow_experiment_name": "Test_Experiment",
        "mlflow_run_name": "Test_Run",
        "mlflow_dataset_name": "Test_Dataset",
        "mlflow_folder_path": "/test/path",
    }


@pytest.fixture
def sample_las() -> Mock:
    """Objet laspy avec coords et classif."""
    las = Mock()
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([4.0, 5.0, 6.0])
    las.z = np.array([7.0, 8.0, 9.0])
    las.classification = np.array([2, 5, 6], dtype=np.uint8)
    return las


@pytest.fixture
def gt_path(tmp_path: Path) -> Path:
    return tmp_path / "gt.laz"


@pytest.fixture
def pred_path(tmp_path: Path) -> Path:
    return tmp_path / "gt_opt.laz"


# Test pour load_laz
@patch("laspy.read")
def test_load_laz(mock_read: Mock, sample_las: Mock) -> None:
    """Vérifie que les coordonnées et labels sont correctement extraits d’un fichier LAZ simulé."""
    mock_read.return_value = sample_las
    coords, labels = load_laz(Path("dummy.laz"))
    expected_coords = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    np.testing.assert_array_equal(coords, expected_coords)
    np.testing.assert_array_equal(labels, np.array([2, 5, 6], dtype=np.uint8))


# Test pour remap_labels
def test_remap_labels() -> None:
    labels = np.array([2, 3, 4, 5], dtype=np.uint8)
    remap_dict = {2: 5, 3: 6, 4: 7, 5: 2}
    result = remap_labels(labels, remap_dict)
    np.testing.assert_array_equal(result, np.array([5, 6, 7, 2], dtype=np.uint8))


# Test pour apply_dbscan
# @patch("sklearn.cluster.DBSCAN")
def test_apply_dbscan() -> None:
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])

    result = apply_dbscan(
        coords, eps=21, min_samples=1
    )  # prends un points sur 4 pour être rapide. dist([1.0, 2.0, 3.0],[13.0, 14.0, 15.0]) = 20.784
    np.testing.assert_array_equal(result, np.array([0, 0], dtype=np.int32))  # points dans le même cluster

    result2 = apply_dbscan(coords, eps=20, min_samples=1)
    np.testing.assert_array_equal(result2, np.array([0, 1], dtype=np.int32))  # points dans differents clusters

    # Si coords vide, retourne un array vide
    result3 = apply_dbscan(np.array([]), eps=20, min_samples=1)
    np.testing.assert_array_equal(result3, np.array([], dtype=np.int32))


# Test pour calculate_object_metrics
def test_calculate_object_metrics() -> None:
    gt_clusters = np.array([0, 0, 1, 1, -1], dtype=np.int32)
    pred_clusters = np.array([0, 0, 0, -1, -1], dtype=np.int32)
    gt_coords = np.array([[1.0, 1.0, 1.0], [1.1, 1.1, 1.1], [2.0, 2.0, 2.0], [2.1, 2.1, 2.1], [3.0, 3.0, 3.0]])
    pred_coords = np.array([[1.0, 1.0, 1.0], [1.1, 1.1, 1.1], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    tp, fp, fn = calculate_object_metrics(gt_clusters, pred_clusters, gt_coords, pred_coords)
    assert tp == 2  # Un cluster correspond (intersection suffisante)
    assert fp == 0  # Pas de cluster prédit sans correspondance (Clusters -1 non pris en compte car bruit)
    assert fn == 0  # Un cluster GT non détecté


# Test pour evaluate_single_file
@patch("eclair.plos.predict.evaluate_mlflow.load_laz")
@patch("eclair.plos.predict.evaluate_mlflow.apply_dbscan")
@patch("eclair.plos.predict.evaluate_mlflow.calculate_object_metrics")
def test_evaluate_single_file(
    mock_metrics: Mock, mock_dbscan: Mock, mock_load_laz: Mock, gt_path: Path, pred_path: Path
) -> None:
    # Ensure pred_path exists
    pred_path.touch()  # Create the file to ensure exists() returns True

    mock_load_laz.side_effect = [
        (np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), np.array([2, 5], dtype=np.uint8)),  # GT
        (np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), np.array([2, 5], dtype=np.uint8)),  # Pred
    ]

    mock_dbscan.side_effect = [
        np.array([0], dtype=np.int32),  # GT class 2
        np.array([0], dtype=np.int32),  # Pred class 2
        np.array([0], dtype=np.int32),  # GT class 5
        np.array([0], dtype=np.int32),  # Pred class 5
    ]

    mock_metrics.side_effect = lambda *args: (1, 0, 0)
    class_ids = [2, 5]
    remap_dict: dict[int, int] = {}  # Empty, as per your test

    tp, fp, fn = evaluate_single_file(gt_path, pred_path.parent, class_ids, remap_dict)

    assert tp[2] == 1
    assert fp[2] == 0
    assert fn[2] == 0
    assert tp[5] == 1
    assert fp[5] == 0
    assert fn[5] == 0


# Test pour evaluate
@patch("eclair.plos.predict.evaluate_mlflow.evaluate_single_file")
def test_evaluate(mock_evaluate_single_file: Mock, tmp_path: Path, sample_config: dict[str, Any]) -> None:
    mock_evaluate_single_file.return_value = ({2: 1, 5: 0}, {2: 0, 5: 1}, {2: 0, 5: 1})
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "file1.laz").touch()
    pred_dir = tmp_path / "pred"
    metrics = evaluate(str(gt_dir), str(pred_dir), [2, 5], sample_config["remap_dict"], sample_config["class_names"])
    print(metrics)
    assert metrics["Classe_2_precision"] == 1.0 / (1.0 + 1e-6)
    assert metrics["Classe_5_recall"] == 0.0 / (0.0 + 1.0 + 1e-6)


# Test pour log_to_mlflow
@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_metric")
@patch("mlflow.log_input")
@patch("mlflow.log_param")
@patch("mlflow.set_tags")
def test_log_to_mlflow(
    mock_set_tags: Mock,
    _mock_log_param: Mock,
    _mock_log_input: Mock,
    mock_log_metric: Mock,
    _mock_start_run: Mock,
    mock_set_experiment: Mock,
    mock_set_tracking_uri: Mock,
    tmp_path: Path,
    sample_config: dict[str, Any],
) -> None:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "file1.laz").touch()
    metrics = {"Tronc_precision": 0.8, "Tronc_recall": 0.9}
    tags = {"eval_type": "objects"}
    log_to_mlflow(str(gt_dir), metrics, sample_config, tags)
    mock_set_tracking_uri.assert_called_with(sample_config["mlflow_uri"])
    mock_set_experiment.assert_called_with(sample_config["mlflow_experiment_name"])
    mock_log_metric.assert_any_call("Tronc_precision", 0.8)
    mock_log_metric.assert_any_call("Tronc_recall", 0.9)
    mock_set_tags.assert_called_with(tags)


# Test pour main
@patch("eclair.plos.predict.evaluate_mlflow.evaluate")
@patch("eclair.plos.predict.evaluate_mlflow.log_to_mlflow")
@patch("argparse.ArgumentParser.parse_args")
@patch("pathlib.Path.open")
def test_main(
    mock_open: Mock,
    mock_parse_args: Mock,
    mock_log_to_mlflow: Mock,
    mock_evaluate: Mock,
    sample_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    mock_parse_args.return_value = Mock(
        gt_dir=str(tmp_path / "gt"),
        pred_dir=str(tmp_path / "pred"),
        config_path=str(tmp_path / "config.json"),
        log_mlflow=True,
    )
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(sample_config)
    mock_evaluate.return_value = {"Tronc_precision": 0.8}
    main()
    mock_evaluate.assert_called_once()  # Vérifie que la fonction evaluate a été appelée exactement une fois.
    mock_log_to_mlflow.assert_called_once()  # Vérifie que la fonction log_to_mlflow a été appelée exactement une fois, car log_mlflow=True
