import gc
import json
import os
import random
import argparse
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)
from model import *
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from utils import FocalLoss, FocalTverskyLoss, FocalTverskyLoss_multiclass, TrainDataset, ValDataset

from eclair.plos.model.utils import collate_custom_test, seed_everything


def make_model(conf):
    try:
        model_class = MODEL_REGISTRY[conf.model_name]
    except KeyError:
        raise ValueError(f"Modèle inconnu : {conf.model_name}")
    model_ = model_class(conf.num_features, conf.num_classes, conf)
    if conf.num_classes == 1:
        model_ = Binary_model(model_)
    return model_


def make_data(conf, data_dir, label_json):
    with open(label_json, encoding="utf-8") as f:
        all_tiles = json.load(f)

    train_tiles = [os.path.join(data_dir, x["tile_name"]) for x in all_tiles if x["split"] == "train"]
    val_tiles = [os.path.join(data_dir, x["tile_name"]) for x in all_tiles if x["split"] == "val"]

    if not train_tiles:
        raise ValueError("No training data found.")
    if not val_tiles:
        raise ValueError("No validation data found.")

    dataset_train = TrainDataset(conf, train_tiles)
    dataloader_train = PyGDataLoader(dataset_train, batch_size=conf.batch_size, shuffle=True,
                                     collate_fn=collate_custom_test, pin_memory=conf.pin_memory,
                                     num_workers=conf.num_worker, drop_last=True)

    dataset_val = ValDataset(conf, val_tiles)
    dataloader_val = PyGDataLoader(dataset_val, batch_size=1,
                                   collate_fn=collate_custom_test, pin_memory=conf.pin_memory,
                                   num_workers=1)

    return dataloader_train, dataloader_val


def train_model(model, device, conf, train_data, val_data, save_dir) -> None:
    """
    Training the model. Save best weights for the validation loss and accuracy. Save the final weights too.

    :param model: model to train
    :param device: GPU if cuda is available or CPU if not
    :param conf: Dictionary from train.yaml
    :param train_data: Train Dataloader
    :param val_data: Val Dataloader
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    if conf.loss == "FocalTverskyLoss":
        focal_loss_fn = FocalTverskyLoss(alpha=conf.loss_weights[0], beta=conf.loss_weights[1], gamma=2, smooth=1).to(
            device
        )
    elif conf.loss == "FocalLoss":
        focal_loss_fn = FocalLoss(gamma=2, alpha=conf.loss_weights).to(device)
    elif conf.loss == "FocalTverskyLoss_multiclass":
        focal_loss_fn = FocalTverskyLoss_multiclass(
            gamma=2,
            smooth=1,
            per_class_alpha=conf.loss_weights[0],
            per_class_beta=conf.loss_weights[1],
            ignore_zero=False,
        ).to(device)
    else:
        raise ValueError(
            "Undefined loss. Choose either 'FocalTverskyLoss' for binary classification or FocalLoss in other cases"
        )

    nb_patience: int = 0
    f1_tresh: float = 0.0
    loss_val_tresh: float = 0.0
    acc_val_tresh: float = 0.0
    inv_voxel_size = 1.0 / conf.voxel_size  # Pré-calculé hors boucle pour encore plus de rapidité

    for epoch in range(conf.epochs):
        random.seed(epoch + 42)  # Changer la seed à chaque epoch

        running_loss: float = 0.0
        running_accuracy: float = 0.0
        running_loss_val: float = 0.0
        running_accuracy_val: float = 0.0

        model.train()

        with tqdm(train_data, desc="Training model") as tepoch:
            for step, batch in enumerate(tepoch):
                coords, features, batch_idx = (
                    torch.floor(batch.pos * inv_voxel_size),
                    batch.x,
                    batch.batch,
                )

                coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

                features = features.unsqueeze(1) if features.dim() == 1 else features

                in_field = TensorField(
                    features=features.to(device),
                    coordinates=coords.to(device),
                    quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
                )

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    # Convert to a sparse tensor
                    sinput = in_field.sparse()

                    # sparse model output
                    soutput = model(sinput)
                    # dense model output
                    out_field = soutput.slice(in_field).F

                    if conf.num_classes == 1:
                        out_field_max = (out_field > 0.5).float()
                    else:
                        out_field_max = torch.argmax(out_field, dim=1)

                    y_true = batch.classification.to(device)
                    loss_ = focal_loss_fn(out_field, y_true)

                scaler.scale(loss_).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)

                scaler.update()

                running_loss += loss_.item() * conf.batch_size

                correct_results_sum = (out_field_max.view(-1) == y_true).sum().item()

                acc = 100 * correct_results_sum / y_true.shape[0]
                running_accuracy += acc * conf.batch_size
                tepoch.set_postfix(loss=loss_.item(), accuracy=acc)

                # Compute the mean loss and accuracy for the current epoch
                del loss_, soutput, out_field, sinput, in_field
                torch.cuda.empty_cache()
                gc.collect()

            epoch_loss = running_loss / len(train_data.dataset)
            epoch_accuracy = running_accuracy / len(train_data.dataset)
            print(f"Epoch {epoch + 1}/{conf.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy}")
            model.eval()

            true_positives = 0
            false_positives = 0
            false_negatives = 0

            with tqdm(val_data, desc="Validate the model") as tepoch_val:
                with torch.no_grad():
                    for batch in tepoch_val:
                        coords, features, batch_idx = (
                            torch.floor(batch.pos * inv_voxel_size),
                            batch.x,
                            batch.batch,
                        )
                        coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

                        features = features.unsqueeze(1) if features.dim() == 1 else features

                        in_field = TensorField(
                            features=features.to(device),
                            coordinates=coords.to(device),
                            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                            minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
                        )

                        sinput = in_field.sparse()
                        soutput = model(sinput)
                        out_field = soutput.slice(in_field).F

                        if conf.num_classes == 1:
                            out_field_max = (out_field > 0.5).float()
                        else:
                            out_field_max = torch.argmax(out_field, dim=1)

                        # get the true labels
                        y_true = batch.classification.to(device)
                        loss_val = focal_loss_fn(out_field, y_true)
                        running_loss_val += loss_val.item()
                        correct_results_sum_val = (out_field_max.view(-1) == y_true).sum().item()

                        acc_val = 100 * correct_results_sum_val / y_true.shape[0]
                        running_accuracy_val += acc_val

                        tepoch_val.set_postfix(loss=loss_val.item(), accuracy=acc_val)

                        # Calcul du f1 score
                        if conf.num_classes == 1:
                            true_positives += ((out_field_max.view(-1) == 1) & (y_true == 1)).sum().item()
                            false_positives += ((out_field_max.view(-1) == 1) & (y_true == 0)).sum().item()
                            false_negatives += ((out_field_max.view(-1) == 0) & (y_true == 1)).sum().item()

        epoch_loss_val: float = running_loss_val / len(val_data.dataset)
        epoch_accuracy_val: float = running_accuracy_val / len(val_data.dataset)
        print(f"Epoch {epoch + 1}/{conf.epochs}, Val Loss: {epoch_loss_val:.4f}, Val Accuracy: {epoch_accuracy_val}")

        if epoch == 0:
            loss_val_tresh = epoch_loss_val
            acc_val_tresh = epoch_accuracy_val

        if conf.num_classes == 1:
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.0

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            if epoch == 0:
                f1_tresh = f1_score

            print(f"Validation F1 Score (class 1): {f1_score:.4f}, Precision: {precision}, Recall: {recall}")

            if f1_score >= f1_tresh:
                f1_tresh = f1_score
                print("new best model weights save for f1_score")
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"{conf.weights_name}_bestf1score.pth")
                )

        if epoch_loss_val < loss_val_tresh:
            loss_val_tresh = epoch_loss_val
            print("new best model weights save for loss")
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{conf.weights_name}_bestvallossweights.pth",
                             ))
            nb_patience = 0
        else:
            nb_patience += 1

        if nb_patience == conf.patience + 1:
            break

        if epoch_accuracy_val >= acc_val_tresh:
            acc_val_tresh = epoch_accuracy_val
            print("new best model weights save for acc")
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"{conf.weights_name}_bestvalaccweights.pth",
                             ))

    torch.save(model.state_dict(), os.path.join(save_dir, f"{conf.weights_name}_finalweights.pth"))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Minkowski model with CLI config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder with train/val pointclouds")
    parser.add_argument("--label_json", type=str, required=True, help="Path to label .json file")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save weights")
    return parser.parse_args()

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    base_conf = OmegaConf.load(args.config_yaml)
    conf = OmegaConf.merge(base_conf, OmegaConf.create({}))  # si besoin d’override CLI plus tard

    seed_everything(conf.random_seed)
    model = make_model(conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data, val_data = make_data(conf, args.data_dir, args.label_json)
    train_model(model, device, conf, train_data, val_data, args.save_dir)


if __name__ == "__main__":
    main()
