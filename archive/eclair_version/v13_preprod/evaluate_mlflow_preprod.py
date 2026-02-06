import json
import os
import glob
import laspy
import numpy as np
from collections import defaultdict

import pandas
from sklearn.cluster import DBSCAN
import argparse
import mlflow


def load_laz(filepath):
    las = laspy.read(filepath)
    coords = np.vstack((las.x, las.y, las.z)).T
    labels = las.classification
    return coords, labels

def remap_labels(gt_labels,remap_dict):
    return np.array([remap_dict.get(label, label) for label in gt_labels])

def apply_dbscan(coords, eps=1.2, min_samples=60):
    reduced_coords = coords[::4]
    if reduced_coords.shape[0] == 0:
        print("[WARNING] Pas assez de points après réduction pour appliquer DBSCAN.")
        return np.array([])
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_coords).labels_
    return clusters

def calculate_object_metrics(gt_clusters, pred_clusters, gt_coords, pred_coords):
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

def evaluate(gt_folder, pred_folder, class_ids, remap_dict, class_names):
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.laz")))
    all_object_metrics = defaultdict(list)
    global_tp = defaultdict(int)
    global_fp = defaultdict(int)
    global_fn = defaultdict(int)

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        name_no_ext = os.path.splitext(filename)[0]
        pred_path = os.path.join(pred_folder, f"{name_no_ext}_opt.laz")

        if not os.path.exists(pred_path):
            print(f"[WARNING] Fichier prédiction non trouvé pour : {filename}")
            continue

        gt_coords, gt_labels = load_laz(gt_path)
        pred_coords, pred_labels = load_laz(pred_path)
        gt_labels = remap_labels(gt_labels, remap_dict)

        for class_id in class_ids:
            if class_id not in gt_labels and class_id not in pred_labels:
                all_object_metrics[f"{class_id}_precision"].append(np.nan)
                all_object_metrics[f"{class_id}_recall"].append(np.nan)
                continue
            if class_id not in gt_labels and class_id in pred_labels:
                all_object_metrics[f"{class_id}_precision"].append(0)
                all_object_metrics[f"{class_id}_recall"].append(np.nan)
                continue
            if class_id in gt_labels and class_id not in pred_labels:
                all_object_metrics[f"{class_id}_precision"].append(np.nan)
                all_object_metrics[f"{class_id}_recall"].append(0)
                continue

            gt_mask = gt_labels == class_id
            pred_mask = pred_labels == class_id
            gt_class_coords = gt_coords[gt_mask]
            pred_class_coords = pred_coords[pred_mask]
            gt_class_clusters = apply_dbscan(gt_class_coords)
            pred_class_clusters = apply_dbscan(pred_class_coords)

            tp, fp, fn = calculate_object_metrics(gt_class_clusters, pred_class_clusters,
                                                  gt_class_coords, pred_class_coords)
            global_tp[class_id] += tp
            global_fp[class_id] += fp
            global_fn[class_id] += fn

    avg_metrics = {}
    for class_id in class_ids:
        tp = global_tp[class_id]
        fp = global_fp[class_id]
        fn = global_fn[class_id]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        class_name = class_names.get(class_id, f"Classe_{class_id}")
        metric_key_precision = f"{class_name}_precision"
        metric_key_recall = f"{class_name}_recall"

        avg_metrics[metric_key_precision] = precision
        avg_metrics[metric_key_recall] = recall

    return avg_metrics

def log_to_mlflow(gt_path, metrics: dict, run_name, experiment_name="Eclair_PLO", tags=None):
    remote_server_uri = "https://gsvision.geo-sat.com/mlflow/"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    gt_files = glob.glob(gt_path + "/*.la[sz]")
    gt_filenames = [os.path.basename(f) for f in gt_files]
    dataset_dataframe = pandas.DataFrame(gt_filenames, columns=["filename"])
    mlflow_pandas_dataset = mlflow.data.from_pandas(
        dataset_dataframe, name="EvalPLOv1"
    )


    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_input(mlflow_pandas_dataset, context="eval")
        mlflow.log_param("Folder path", r"R:\9_temporary_files\for_temporary_storage_1_month\malo\Eclair_preprod\v13")



def main():
    parser = argparse.ArgumentParser(description="Évaluation des métriques d'objets sur des nuages de points LAZ")
    parser.add_argument("--gt_dir", required=True, help="Répertoire des fichiers LAZ de Ground Truth (GT)")
    parser.add_argument("--pred_dir", required=True, help="Répertoire des fichiers LAZ de Prédiction")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Chemin vers le fichier de configuration des classes"
    )
    parser.add_argument("--log_mlflow", action="store_true", default=False,
                        help="Activer l'enregistrement des résultats dans MLflow")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Cast des clés en int, car les JSON mettent les clés en string par défaut
    class_names = {int(k): v for k, v in config["class_names"].items()}
    class_ids = sorted(class_names.keys())
    remap_dict = {int(k): int(v) for k, v in config["remap_dict"].items()}

    object_metrics = evaluate(args.gt_dir, args.pred_dir, class_ids=class_ids, remap_dict=remap_dict,class_names=class_names)

    print("\n--- Résultats des métriques des objets ---")
    for k, v in object_metrics.items():
        print(f"{k}: {v:.4f}")

    if args.log_mlflow:
        log_to_mlflow(args.gt_dir, object_metrics, run_name="Eclair_v13_fuze", tags={"eval_type": "objects"})

if __name__ == "__main__":
    main()
