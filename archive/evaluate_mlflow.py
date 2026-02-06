import argparse
import glob
import json
import os
from typing import List, Dict, Tuple

import mlflow
import numpy as np
import numpy.typing as npt
import pandas
import seaborn as sns
from geosat.cloud import PointCloud
from matplotlib import pyplot as plt

from gssemantic.metrics import (
    fast_confusion,
    get_iou_per_class_from_confusion_matrix,
)
import MinkowskiEngine as ME
import torch  # If using PyTorch
from model import MinkUNet14C, Binary_model
import mlflow.pyfunc

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_classification_codes(
    gt_path: str, pred_path: str
) -> Tuple[List[str], List[int]]:
    """
    Verification of the classification codes for ground truth and prediction folder.
    A classification code file must be in the ground truth folder AND in the prediction folder.
    They must be exactly the same.

    :param gt_path: Path of the ground truth point clouds folder.
    :param pred_path: Path of the predicted point clouds folder.
    :raises ValueError:
        Classification codes are different for ground truth and predicted point clouds.
    :return keys: The keys (names of the classes).
    :return values: The values (ids of the classes).
    """

    # Loading the classification codes from the ground truth and prediction folders.
    with open(gt_path + "/classification_codes.json") as file_object:
        gt_classification_codes: Dict = json.load(file_object)
    with open(pred_path + "/classification_codes.json") as file_object:
        pred_classification_codes: Dict = json.load(file_object)

    # Check if the classification codes are the same. If not, return an error
    if gt_classification_codes != pred_classification_codes:
        raise ValueError(
            "Classification codes for ground truth and prediction are different"
        )

    # return of the keys and values as lists
    return list(gt_classification_codes.keys()), list(gt_classification_codes.values())


def evaluate(
    gt_path: str, pred_path: str, gt_ids: List[int]
) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.int64]]]:
    """
    For the given ground truth path and prediction path, compute for each point cloud the mean IoU of each class.

    :param gt_path: Path of the ground truth point clouds folder.
    :param pred_path: Path of the predicted point clouds folder.
    :param gt_ids: The values (ids of the classes).

    :raises FileNotFoundError:
        A point cloud in ground truth hasn't been found in the predicted folder.
        Be careful about the name and extension of the file (must be exactly the same).

    :return list_iou_per_classes:
        List that contains for each point cloud the array of IoU for each class.

    :return list_confusion_matrices:
        List that contains for each point cloud the confusion matrices.

    """

    # Loading the path of all .las or .laz files contained in the ground truth folder
    gt_files: List[str] = glob.glob(gt_path + "/*.la[sz]")

    # Creating an empty list that will contain the IoU values of each class for each pair of
    # (ground truth, predict) point cloud
    list_iou_per_classes: List[npt.NDArray[np.float64]] = []
    list_confusion_matrices_recall: List[npt.NDArray[np.int64]] = []
    list_confusion_matrices_precision: List[npt.NDArray[np.int64]] = []

    # Iterating through each point cloud contained in the ground truth folder.
    for gt_file in gt_files:

        # Loading the 2 corresponding point clouds (ground truth, predict).
        # Check if the same name exists in the pred folder; otherwise, return an error.
        gt_cloud: PointCloud = PointCloud.read_from_path(gt_file)
        try:
            pred_cloud: PointCloud = PointCloud.read_from_path(
                pred_path + "/" + os.path.basename(gt_file)
            )
        except FileNotFoundError as exception:
            raise FileNotFoundError(
                f"Ground truth and prediction files must have exactly the same name and extension. "
                f"No prediction files found for: {gt_file}."
            ) from exception

        # Retrieving the classification of each point from the ground truth and predict point clouds.
        gt_classification: npt.NDArray[np.int8] = gt_cloud.fields["classification"]
        pred_classification: npt.NDArray[np.int8] = pred_cloud.fields["classification"]

        print(gt_file)
        # Computation of the corresponding confusion matrix
        confusion_matrix_recall: npt.NDArray[np.int64] = fast_confusion(
            gt_classification,pred_classification,np.asarray(gt_ids)
        )

        confusion_matrix_precision: npt.NDArray[np.int64] = fast_confusion(
            pred_classification,gt_classification,np.asarray(gt_ids)
        )

        # Calculation the IoU per class from the confusion matrix.
        iou_per_class: npt.NDArray[np.float64] = (
            get_iou_per_class_from_confusion_matrix(confusion_matrix_recall)
        )

        # Adding the results (IoU per class) to the list of results
        list_iou_per_classes.append(iou_per_class)

        list_confusion_matrices_precision.append(confusion_matrix_precision)
        list_confusion_matrices_recall.append(confusion_matrix_recall)

    # Return of the list containing the IoU for each class and each point cloud
    return list_iou_per_classes, list_confusion_matrices_precision,list_confusion_matrices_recall


def get_results_from_evaluation(
    iou_results: List[npt.NDArray[np.float64]],
    confusion_matrices_precision: List[npt.NDArray[np.int64]],
    confusion_matrices_recall: List[npt.NDArray[np.int64]],
    gt_keys: List[str],
    pred_path: str,
) -> Tuple[Dict[str, float], npt.NDArray[int]]:
    """
    Function that calculates the mean IoU for each class across all point clouds,
    given the IoU for each class on each individual point cloud as input.

    :param iou_results: List containing the IoU per classe and per point cloud.
    :param confusion_matrices: List containing the confusion matrices for each point cloud.
    :param gt_keys: List of keys of the ground truth point clouds.
    :param pred_path: Path of the predicted point clouds folder.

    :return dict_mean_iou_per_class: Dictionary containing the mean IoU for each class across all point clouds.
    :return norm_confusion_matrix_img: Normalized confusion matrix as an array
    """

    normalized_sum_confusion_matrix_precision = np.divide(
        sum(confusion_matrices_precision), np.sum(sum(confusion_matrices_precision), axis=1, keepdims=True)
    )
    normalized_sum_confusion_matrix_recall = np.divide(
        sum(confusion_matrices_recall), np.sum(sum(confusion_matrices_recall), axis=1, keepdims=True)
    )

    dict_precision_per_class: Dict[str, float] = {}
    dict_recall_per_class: Dict[str, float] = {}
    dict_f1score_per_class: Dict[str, float] = {}

    for iter, gt_key in enumerate(gt_keys):
        precision_iter=normalized_sum_confusion_matrix_precision[iter,iter]
        recall_iter=normalized_sum_confusion_matrix_recall[iter,iter]
        dict_precision_per_class[gt_key + "_precision"] = precision_iter
        dict_recall_per_class[gt_key + "_recall"] = recall_iter
        dict_f1score_per_class[gt_key + "_f1score"] = 2*precision_iter*recall_iter/(precision_iter+recall_iter)


    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        normalized_sum_confusion_matrix_precision,
        annot=True,
        fmt=".2f",
        xticklabels=gt_keys,
        yticklabels=gt_keys,
        annot_kws={"size": 12}
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.savefig(pred_path + "/confusion_matrix_precision.png")
    norm_confusion_matrix_img_precision = plt.imread(pred_path + "/confusion_matrix_precision.png")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        normalized_sum_confusion_matrix_recall,
        annot=True,
        fmt=".2f",
        xticklabels=gt_keys,
        yticklabels=gt_keys,
        annot_kws={"size": 12}
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(pred_path + "/confusion_matrix_recall.png")
    norm_confusion_matrix_img_recall = plt.imread(pred_path + "/confusion_matrix_recall.png")

    # return of the list containing the mean IoU values of each class associated with its key value
    return dict_precision_per_class,dict_recall_per_class,dict_f1score_per_class, norm_confusion_matrix_img_precision, norm_confusion_matrix_img_recall

def model_2_mlflow(num_features, num_classes, weights) -> None:
    """
    Log the model directly to MLflow as a .pth file to avoid issues with serialization.

    :param num_features: Number of input features.
    :param num_classes: Number of output classes.
    :param weights: Path to the weights file (.pth)
    """

    # Instantiate the model
    model_pred = MinkUNet14C(13, num_classes)
    if num_classes == 1:
        model_pred = Binary_model(model_pred)

    # Load the model weights
    model_pred.load_state_dict(torch.load(weights))
    model_pred.eval()  # Set the model to evaluation mode

    # Save the model's state dict to a file (.pth)
    model_path = "model.pth"
    torch.save(model_pred.state_dict(), model_path)

    # Logging the model file (.pth) with MLflow
    try:
        # Log the model artifact (the .pth file) into MLflow
        mlflow.log_artifact(model_path)
        print(f"Model successfully logged to MLflow as {model_path}.")
    except Exception as e:
        print(f"Error while logging the model: {e}")

def save_mlflow(
    gt_path: str,
    model_name: str,
    dataset_name: str,
    precision_per_classes:Dict[str, float] ,
    recall_per_classes: Dict[str, float],
    f1score_per_classes: Dict[str, float],
    norm_confusion_matrix_precision: npt.NDArray[int],
    norm_confusion_matrix_recall : npt.NDArray[int]
) -> None:

    """
    Save the results and parameters on MLflow

    :param gt_path: path to the gt folder
    :param model_name: name of the model
    :param dataset_name: name of the dataset
    :param mean_iou_per_class: mean iou per class computed over the whole dataset
    :param norm_confusion_matrix: normalized confusion matrix computed over the whole dataset
    :return: None
    """

    gt_files: List[str] = glob.glob(gt_path + "/*.la[sz]")

    # normal
    dataset_dataframe = pandas.DataFrame(gt_files, columns=["filename"])

    # creation of the mlflow dataset. it can be named
    mlflow_pandas_dataset = mlflow.data.from_pandas(
        dataset_dataframe, name=dataset_name
    )

    # connection to the server
    remote_server_uri: str = "https://gsvision.geo-sat.com/mlflow/"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("gssemantic_evaluation")

    with mlflow.start_run(run_name=model_name):

        # Enregistrement de paramètres
        mlflow.log_param("model", model_name)
        mlflow.log_param("Dataset path", gt_path)
        mlflow.log_param("Preds path", "/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v9/preds/handcraft_vxl0009_05_081_avecsol/")
        # mlflow.log_param("Unclassified path", r"D:\pointcloud_data\viamapa_flai\predictions\unclass")
        # mlflow.log_param("Signs path", r"R:\9_temporary_files\for_temporary_storage_1_month\malo\pointcloud_data\ns2f_moreclasses\pred_eclair\binary_preds\sign_newaug_big_vxl003_b8_accf1_f4_tv0901")
        # mlflow.log_param("Merge path", r"R:\9_temporary_files\for_temporary_storage_1_month\malo\pointcloud_data\ns2f\eclair\pred_newpoles_newtrees")

        num_features=4
        num_classes=4
        weights="/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v9/preds/handcraft_vxl0009_05_081_avecsol/test_handcraft_extanded_3c_v9_05050505_0808081_vxl0009_bestvallossweights.pth"
        mlflow.log_param("Voxel size", 0.009)
        # mlflow.log_param("Nuages en entrainement", "Iantsa_reanno_61")

        mlflow.log_param("Loss", "MCTFL")
        # mlflow.log_param("Loss_precision", 1)
        # mlflow.log_param("Loss_recall", 0.5)

        mlflow.log_param("Normalisation", "norm50")

        mlflow.log_param("Batch size", 1)
        # mlflow.log_param("Training set path",r"R:\9_temporary_files\for_temporary_storage_1_month\viamapa\lidar_hd_dataset")
        mlflow.log_param("weights path", weights)
        # mlflow.log_param("weights path 10 classes", r"D:\pointcloud_data\iantsa\reanno_16c\12test\iantsa_reanno71_e400_vxl0003_intrgb_b1_newloss_invprob_10c_bestvallossweights.pth")

        # enregistrement du dataset
        mlflow.log_input(mlflow_pandas_dataset, context="eval")
        # enregistrement de metric
        mlflow.log_metrics(precision_per_classes)
        mlflow.log_metrics(recall_per_classes)
        mlflow.log_metrics(f1score_per_classes)

        # enregistrer une image
        mlflow.log_image(norm_confusion_matrix_precision, "confusion_matrix_precision.png")
        mlflow.log_image(norm_confusion_matrix_recall, "confusion_matrix_recall.png")
        # mlflow.log_text(r"R:\9_temporary_files\for_temporary_storage_1_month\malo\pointcloud_data\ns2f_moreclasses\pred_eclair\b8_vxl001_e300_input4_wfl_acc\ns2f_big_all_tv0505_input4_newaugallv2_vx001_batch8_wieghted_e300_bestvalaccweights.pth", "weight_path.txt")


        #Enregistrer fichiers trainn modèle, ...
        # mlflow.log_artifact("/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v5/predictions/yaml.txt")
        # mlflow.log_artifact("/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/all_set_v5/predictions/functional.txt")
        # mlflow.log_artifact(r"D:\pointcloud_data\iantsa\reanno_16c\11test\train_metrics.txt","train_metrics.txt")
        # mlflow.log_artifact(r"D:\pointcloud_data\train_ns2f_big\pred_rail\ns2f_rail_tv_vxl0004_b1_trainbigv4_denoised_200e_tv0605_bestvallossweights\yaml.txt","yaml.txt")
        model_2_mlflow(num_features, num_classes, weights)

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Point clouds classifications evaluation."
    )

    parser.add_argument(
        "--gt_path", type=str, help="path to the folder of the ground truth point clouds"
    )

    parser.add_argument(
        "--pred_path", type=str, help="path to the folder of the predicted point clouds"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model used for the prediction",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset on which the evaluation is performed",
    )

    args = parser.parse_args()

    # Checking the ground truth and prediction classification codes. If they are different, the execution stops.
    gt_keys, gt_values = check_classification_codes(args.gt_path, args.pred_path)

    # Calculating the IoU per class for each point clouds contained in the ground truth and prediction folders.
    iou_per_classes, confusion_matrices_precision,confusion_matrices_recall = evaluate(
        args.gt_path, args.pred_path, gt_values
    )

    # Calculating the mean iou per class over all the point clouds.
    precision_per_classes, recall_per_classes, f1score_per_classes, fig_conf_matrix_precision, fig_conf_matrix_recall = get_results_from_evaluation(
        iou_per_classes,confusion_matrices_precision, confusion_matrices_recall, gt_keys, args.pred_path
    )

    # Save of a .txt file containing the mean IoU of each class on the entire dataset.
    save_mlflow(
        args.gt_path,
        args.model_name,
        args.dataset_name,
        precision_per_classes,
        recall_per_classes,
        f1score_per_classes,
        fig_conf_matrix_precision,
        fig_conf_matrix_recall,
    )


if __name__ == "__main__":
    main()
