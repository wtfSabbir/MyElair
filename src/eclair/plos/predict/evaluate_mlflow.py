"""Compare les champs classification de deux dossiers de nuages et log les métriques sur MLFLOW."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import laspy
import mlflow
import mlflow.data.pandas_dataset
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import DBSCAN


def load_laz(filepath: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    """
    Charge un fichier LAZ et extrait les coordonnées 3D ainsi que les classifications.

    :param filepath: Chemin vers le fichier LAZ à lire.
    :return: Un tuple contenant :
        - coords (np.ndarray) : tableau Nx3 des coordonnées (x, y, z),
        - labels (np.ndarray) : tableau des labels de classification associés à chaque point.
    """
    las = laspy.read(filepath)
    coords = np.vstack((las.x, las.y, las.z)).T
    labels = las.classification
    return coords, labels


def remap_labels(gt_labels: npt.NDArray[np.uint8], remap_dict: dict[int, int]) -> npt.NDArray[np.uint8]:
    """
    Remappe les labels d'un tableau selon un dictionnaire de correspondance.

    :param gt_labels: Tableau des labels d'origine.
    :param remap_dict: Dictionnaire {ancien_label: nouveau_label} pour la conversion.
    :return: Nouveau tableau numpy avec les labels remappés.
    """
    return np.array([remap_dict.get(int(label), label) for label in gt_labels])


def apply_dbscan(coords: npt.NDArray[np.float64], eps: float = 1.2, min_samples: int = 60) -> npt.NDArray[np.int32]:
    """
    Applique l'algorithme DBSCAN sur un sous-échantillon des coordonnées pour détecter les clusters.

    Les coordonnées sont réduites en prenant un point sur quatre pour accélérer le calcul.

    :param coords: Coordonnées 3D des points.
    :param eps: Distance maximale entre deux points pour être considérés dans le même cluster (paramètre DBSCAN).
    :param min_samples: Nombre minimum de points requis pour former un cluster (paramètre DBSCAN).
    :return: Tableau numpy des labels de clusters attribués par DBSCAN.
    """
    reduced_coords = coords[::4]
    if reduced_coords.shape[0] == 0:
        return np.array([], dtype=np.int32)
    return cast("npt.NDArray[np.int32]", DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_coords).labels_)


def calculate_object_metrics(
    gt_clusters: npt.NDArray[np.int32],
    pred_clusters: npt.NDArray[np.int32],
    gt_coords: npt.NDArray[np.float64],
    pred_coords: npt.NDArray[np.float64],
) -> tuple[int, int, int]:
    """
    Métriques entre objets annotés (ground truth) et prédits.

    Deux objets sont considérés comme correspondants si l'intersection de leurs points est supérieure ou égale à 5 %
    du nombre de points de l'objet ground truth.

    :param gt_clusters: Tableau d'étiquettes de cluster pour les objets au sol (ground truth).
    :param pred_clusters: Tableau d'étiquettes de cluster pour les objets prédits.
    :param gt_coords: Coordonnées 3D associées aux clusters ground truth.
    :param pred_coords: Coordonnées 3D associées aux clusters prédits.
    :return: Un triplet contenant :
        - True Positives (tp) : nombre d'objets correctement détectés,
        - False Positives (fp) : objets détectés mais sans correspondance dans le ground truth,
        - False Negatives (fn) : objets du ground truth sans correspondance parmi les prédictions.
    """
    gt_objects = defaultdict(set)
    pred_objects = defaultdict(set)

    for i, label in enumerate(gt_clusters):
        if label != -1:
            gt_objects[label].add(tuple(gt_coords[i]))

    for i, label in enumerate(pred_clusters):
        if label != -1:
            pred_objects[label].add(tuple(pred_coords[i]))

    gt_matched = set()
    pred_matched = set()

    for gt_id, gt_set in gt_objects.items():
        for pred_id, pred_set in pred_objects.items():
            intersection = gt_set & pred_set
            if len(intersection) >= 0.05 * len(gt_set):
                gt_matched.add(gt_id)
                pred_matched.add(pred_id)
                break

    tp = len(gt_matched)
    fp = len(pred_objects) - len(pred_matched)
    fn = len(gt_objects) - len(gt_matched)

    return tp, fp, fn


def evaluate_single_file(  # noqa: PLR0914
    gt_path: Path,
    pred_folder: Path,
    class_ids: list[int],
    remap_dict: dict[int, int],
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """
    TP, FP, FN pour un seul fichier de vérité terrain.

    :returns: Trois dictionnaires (tp, fp, fn) avec les compteurs par classe.
    """
    pred_path = pred_folder / f"{gt_path.stem}_opt.laz"
    if not pred_path.exists():
        return {}, {}, {}

    gt_coords, gt_labels = load_laz(gt_path)
    pred_coords, pred_labels = load_laz(pred_path)
    gt_labels = remap_labels(gt_labels, remap_dict)

    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for class_id in class_ids:
        gt_has = class_id in gt_labels
        pred_has = class_id in pred_labels

        if not gt_has and not pred_has:
            continue  # Pas d'info, on skip
        if not gt_has and pred_has:
            tp[class_id] += 0
            fn[class_id] += 0
            fp[class_id] += 1
            continue
        if gt_has and not pred_has:
            tp[class_id] += 0
            fp[class_id] += 0
            fn[class_id] += 1
            continue

        gt_mask = gt_labels == class_id
        pred_mask = pred_labels == class_id

        gt_class_clusters = apply_dbscan(gt_coords[gt_mask])
        pred_class_clusters = apply_dbscan(pred_coords[pred_mask])

        c_tp, c_fp, c_fn = calculate_object_metrics(
            gt_class_clusters,
            pred_class_clusters,
            gt_coords[gt_mask],
            pred_coords[pred_mask],
        )
        tp[class_id] += c_tp
        fp[class_id] += c_fp
        fn[class_id] += c_fn

    return tp, fp, fn


def evaluate(
    gt_folder: str,
    pred_folder: str,
    class_ids: list[int],
    remap_dict: dict[int, int],
    class_names: dict[int, str],
) -> dict[str, float]:
    """
    Métriques de précision et rappel moyennes pour plusieurs classes.

    Cette fonction parcourt tous les fichiers .laz du dossier de vérité terrain,
    appelle une fonction dédiée pour calculer les vrais positifs (TP), faux positifs (FP)
    et faux négatifs (FN) pour chaque fichier, puis agrège ces valeurs sur l'ensemble des fichiers.
    Enfin, elle calcule la précision et le rappel moyens par classe.

    :param gt_folder: Chemin vers le dossier contenant les fichiers ground truth (.laz).
    :param pred_folder: Chemin vers le dossier contenant les fichiers de prédiction (.laz).
    :param class_ids: Liste des identifiants des classes à évaluer.
    :param remap_dict: Dictionnaire permettant de remapper les étiquettes des ground truth.
    :param class_names: Dictionnaire associant un nom lisible à chaque identifiant de classe.

    :return: Dictionnaire des métriques moyennes, avec pour chaque classe deux clés :
             "<nom_classe>_precision" et "<nom_classe>_recall".
    """
    gt_files = sorted(Path(gt_folder).glob("*.laz"))
    global_tp: dict[int, int] = defaultdict(int)
    global_fp: dict[int, int] = defaultdict(int)
    global_fn: dict[int, int] = defaultdict(int)

    for gt_path in gt_files:
        tp, fp, fn = evaluate_single_file(gt_path, Path(pred_folder), class_ids, remap_dict)
        for class_id in class_ids:
            global_tp[class_id] += tp.get(class_id, 0)
            global_fp[class_id] += fp.get(class_id, 0)
            global_fn[class_id] += fn.get(class_id, 0)

    avg_metrics = {}
    for class_id in class_ids:
        tp_count = global_tp[class_id]
        fp_count = global_fp[class_id]
        fn_count = global_fn[class_id]

        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / (tp_count + fn_count + 1e-6)

        name = class_names.get(class_id, f"Classe_{class_id}")
        avg_metrics[f"{name}_precision"] = precision
        avg_metrics[f"{name}_recall"] = recall

    print("\n--- Résultats des métriques des objets ---")  # noqa: T201
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")  # noqa: T201

    return avg_metrics


def log_to_mlflow(
    gt_path: str, metrics: dict[str, float], config: dict[str, Any], tags: dict[str, str] | None = None
) -> None:
    """
    Enregistre les métriques et les informations d évaluation dans MLflow.

    Cette fonction configure un run MLflow, associe les métriques, paramètres et tags fournis,
    et logue un tableau issu des fichiers de vérité terrain pour traçabilité.

    :param gt_path: Chemin vers le dossier contenant les fichiers de vérité terrain (`.las` ou `.laz`).
    :param metrics: Dictionnaire des métriques à enregistrer (e.g., IoU, précision, rappel).
    :param config: Dictionnaire issu du fichier .json
    :param tags: Dictionnaire optionnel de tags à associer au run MLflow.
    """
    mlflow.set_tracking_uri(config["mlflow_uri"])
    mlflow.set_experiment(config.get("mlflow_experiment_name"))

    gt_files = Path(gt_path).glob("*.la[sz]")
    gt_filenames = [Path(f).name for f in gt_files]
    dataset_dataframe = pd.DataFrame(gt_filenames, columns=["filename"])
    mlflow_pandas_dataset = mlflow.data.pandas_dataset.from_pandas(
        dataset_dataframe,
        name=config.get("mlflow_dataset_name"),
    )

    with mlflow.start_run(run_name=config.get("mlflow_run_name")):
        if tags:
            mlflow.set_tags(tags)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_input(mlflow_pandas_dataset, context="eval")
        mlflow.log_param(
            "Folder path",
            config.get("mlflow_folder_path"),
        )
        artifact_paths = config.get("log_artifacts_files")
        if artifact_paths:
            for path in artifact_paths:
                artifact_path = Path(path)
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))


def main() -> None:
    """Applique le calcul des metriques en comparant les champs classifications de nuages de points."""
    parser = argparse.ArgumentParser(description="Évaluation des métriques d'objets sur des nuages de points LAZ")
    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Répertoire des fichiers LAZ de Ground Truth (GT)",
    )
    parser.add_argument("--pred_dir", required=True, help="Répertoire des fichiers LAZ de Prédiction")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Chemin vers le fichier de configuration des classes",
    )
    parser.add_argument(
        "--log_mlflow",
        action="store_true",
        default=False,
        help="Activer l'enregistrement des résultats dans MLflow",
    )
    args = parser.parse_args()

    with Path(args.config_path).open(encoding="utf-8") as f:
        config = json.load(f)

    # Cast des clés en int, car les JSON mettent les clés en string par défaut
    class_names = {int(k): v for k, v in config["class_names"].items()}
    class_ids = sorted(class_names.keys())
    remap_dict = {int(k): int(v) for k, v in config["remap_dict"].items()}

    object_metrics = evaluate(
        args.gt_dir,
        args.pred_dir,
        class_ids=class_ids,
        remap_dict=remap_dict,
        class_names=class_names,
    )

    if args.log_mlflow:
        log_to_mlflow(
            args.gt_dir,
            object_metrics,
            config,
            tags={"eval_type": "objects"},
        )


if __name__ == "__main__":
    main()
