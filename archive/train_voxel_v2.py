import gc
import json
import os
from typing import List, Dict
from torch_geometric.loader import DataLoader as PyGDataLoader
from MinkowskiEngine import (
    MinkowskiAlgorithm,
    SparseTensorQuantizationMode,
    TensorField,
)

from tqdm import tqdm

from functional import TimedTransform
from utils import (
    FocalLoss,
    FocalTverskyLoss,
    seed_everything,
    TrainDataset,
    collate_custom_test,
    TestDataset,
    FocalTverskyLoss_multiclass, ValDataset,
)

import torch
from omegaconf import OmegaConf
from model_test import MinkUNet14C, Binary_model, MinkUNet101, MinkUNet50, MinkUNet34A, MinkUNet14, MinkUNet18, MinkUNet18B
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def make_model(conf) -> MinkUNet14C:
    """
    Instantiate and return the model. If num_class == 1, then a sigmoid is put at the output of the network

    :param conf: Dictionary from train.yaml
    :return: MinkUNet14C model with/without sigmoid
    """
    model_: MinkUNet14C = MinkUNet14C(conf.num_features, conf.num_classes)
    if conf.num_classes == 1:
        model_ = Binary_model(model_)  # add a sigmoid at the end
    print(model_)
    return model_

def make_data(conf) -> PyGDataLoader:
    """
    Read the .json file with the name of the pointclouds and their repartition in train/val.
    Apply the transformation (data-aug, normalization) and create the train and val dataloaders.

    :param conf: Dictionary from train.yaml
    :return: train and validation dataloader for the training and validation steps.
    """
    label_file = "data/labels_v13_horssol.json"
    # label_file = "/mnt/d/pointcloud_data/iantsa/reanno_16c/gt/labels_iantsa16c.json"
    with open(label_file, encoding="utf-8") as f:
        all_tiles: List[Dict] = json.load(f)

    train_tiles = [
        "data/pointclouds_horssol" + "/" + x["tile_name"]
        # "/mnt/d/pointcloud_data/iantsa/reanno_16c/gt/pointclouds" + "/" + x["tile_name"]
        for x in all_tiles
        if x["split"] == "train"
    ]

    if len(train_tiles) == 0:
        raise ValueError(
            "No training data. Please verify your labels.json file. Pointclouds should be in data/pointclouds"
        )

    val_tiles = [
        "data/pointclouds_horssol/" + "/" + x["tile_name"]
        # "/mnt/d/pointcloud_data/iantsa/reanno_16c/gt/pointclouds" + "/" + x["tile_name"]
        for x in all_tiles
        if x["split"] == "val"
    ]

    if len(val_tiles) == 0:
        raise ValueError(
            "No validation data. Please verify your labels.json file. Pointclouds should be in data/pointclouds"
        )

    dataset_train = TrainDataset(conf, train_tiles)
    dataloader_train = PyGDataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_custom_test,
        pin_memory=False,
        num_workers=2,
        drop_last=True,
    )

    dataset_val = ValDataset(conf, val_tiles)
    dataloader_val = PyGDataLoader(
        dataset_val,
        batch_size=1,
        collate_fn=collate_custom_test,
        pin_memory=False,
        num_workers=1,
    )

    return dataloader_train, dataloader_val


def train_model(model, device, conf, train_data, val_data) -> None:
    """
    Training the model. Save best weights for the validation loss and accuracy. Save the final weights too.

    :param model: model to train
    :param device: GPU if cuda is available or CPU if not
    :param conf: Dictionary from train.yaml
    :param train_data: Train Dataloader
    :param val_data: Val Dataloader
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.amp.GradScaler('cuda')
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    #
    # # Enchaîner les deux schedulers
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])


    if conf.loss == "FocalTverskyLoss":
        focal_loss_fn = FocalTverskyLoss(
            alpha=conf.loss_weights[0], beta=conf.loss_weights[1], gamma=2, smooth=1
        ).to(device)
    else:
        if conf.loss == "FocalLoss":
            focal_loss_fn = FocalLoss(gamma=2, alpha=conf.loss_weights).to(device)
        else:
            if conf.loss == "FocalTverskyLoss_multiclass":
                focal_loss_fn = FocalTverskyLoss_multiclass(gamma=2, smooth=1, per_class_alpha=conf.loss_weights[0], per_class_beta=conf.loss_weights[1], ignore_zero=False).to(device)
            else:
                raise ValueError(
                    "Undefined loss. Choose either 'FocalTverskyLoss' for binary classification or FocalLoss in other cases"
                )

    nb_patience: int = 0
    f1_tresh: float = 0.0
    loss_val_tresh: float = 0.0
    acc_val_tresh: float = 0.0
    gradient_accumulation_steps = 4
    inv_voxel_size = 1.0 / conf.voxel_size  # Pré-calculé hors boucle pour encore plus de rapidité

    for epoch in range(conf.epochs):

        seed_everything(epoch + 42)  # Changer la seed à chaque epoch

        running_loss: float = 0.0
        running_accuracy: float = 0.0
        running_loss_val: float = 0.0
        running_accuracy_val: float = 0.0

        model.train()

        with tqdm(train_data, desc="Training model") as tepoch:
            for step, batch in enumerate (tepoch):

                # ###VOXELS CARRE###
                # coords, features, batch_idx = (
                #     batch.pos / conf.voxel_size,
                #     batch.x,
                #     batch.batch,
                # )

                coords, features, batch_idx = (
                    torch.floor(batch.pos * inv_voxel_size),
                    batch.x,
                    batch.batch,
                )
                coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

                # coords = coords.to(device)
                #
                # batch.classification = batch.classification.to(device)
                #
                # # Trouver les voxels uniques et leurs indices
                # unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
                # num_voxels = unique_coords.shape[0]
                #
                # # Nombre de classes
                # num_classes = conf.num_classes
                # # Convertir les labels en one-hot
                #
                # assert batch.classification.min() >= 0, "Erreur: Les labels doivent être >= 0"
                # assert batch.classification.max() < num_classes, "Erreur: Les labels doivent être 0 ou 1"
                #
                # y_one_hot = torch.nn.functional.one_hot(batch.classification, num_classes=num_classes).float().to(
                #     device)
                # # Accumuler les votes par voxel
                # votes_per_voxel = torch.zeros(num_voxels, num_classes, device=device)
                # votes_per_voxel.scatter_add_(0, inverse_indices.view(-1, 1).expand(-1, num_classes), y_one_hot)
                #
                # # Obtenir la classe majoritaire pour chaque voxel
                # y_voxel_majority = votes_per_voxel.argmax(dim=1)  # Shape: [num_voxels]
                # # Attribuer à chaque point la classe majoritaire de son voxel
                # y_true_per_point = y_voxel_majority[inverse_indices]  # Shape: [num_points]

                ###VOXEL RECTANGULAIRE###
                # pos = batch.pos  # Shape: [N, 3]
                # voxel_size_tensor = torch.tensor(conf.voxel_size, dtype=pos.dtype, device=pos.device).view(1, -1)
                # coords_quantized = torch.floor(pos / voxel_size_tensor)
                # coords = torch.cat([batch.batch.unsqueeze(1), coords_quantized], dim=1)
                # features = batch.x

                in_field = TensorField(
                    features=features.to(device),
                    coordinates=coords.to(device,dtype=torch.float32),
                    quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
                )

                # optimizer.zero_grad()
                with torch.amp.autocast('cuda'):

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
                # loss_ = focal_loss_fn(out_field, y_true_per_point)
                # loss_ = focal_loss_fn(out_field, y_true)
                # loss_.backward()
                # optimizer.step()

                    loss_ = focal_loss_fn(out_field, y_true)
                    # loss_ = loss_ / gradient_accumulation_steps  # Normaliser la perte
                    loss_ = loss_ # Normaliser la perte
                scaler.scale(loss_).backward()

                # if (step + 1) % gradient_accumulation_steps == 0:

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad()


                # scheduler.step()
                # running_loss += loss_.item() * conf.batch_size*gradient_accumulation_steps
                running_loss += loss_.item() * conf.batch_size

                # correct_results_sum = (out_field_max.view(-1) == y_true_per_point).sum().item()
                correct_results_sum = (out_field_max.view(-1) == y_true).sum().item()

                acc = 100 * correct_results_sum / y_true.shape[0]
                running_accuracy += acc * conf.batch_size
                tepoch.set_postfix(loss=loss_.item(), accuracy=acc)

                if step % 10 == 0:  # Tous les 10 batchs
                    torch.cuda.empty_cache()
                    gc.collect()

            # Compute the mean loss and accuracy for the current epoch
            epoch_loss = running_loss / len(train_data.dataset)
            epoch_accuracy = running_accuracy / len(train_data.dataset)
            print(
                f"Epoch {epoch + 1}/{conf.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy}"
            )
            model.eval()

            true_positives = 0
            false_positives = 0
            false_negatives = 0

            with tqdm(val_data, desc="Validate the model") as tepoch_val:
                with torch.no_grad():

                    for batch in tepoch_val:

                        ###VOXELs CARRE###
                        # coords, features, batch_idx = (
                        #     batch.pos / conf.voxel_size,
                        #     batch.x,
                        #     batch.batch,
                        # )

                        coords, features, batch_idx = (
                            torch.floor(batch.pos * inv_voxel_size),
                            batch.x,
                            batch.batch,
                        )
                        coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

                        # coords = coords.to(device)
                        #
                        # batch.classification = batch.classification.to(device)
                        #
                        # # Trouver les voxels uniques et leurs indices
                        # unique_coords, inverse_indices = torch.unique(coords, dim=0, return_inverse=True)
                        # num_voxels = unique_coords.shape[0]
                        #
                        # # Nombre de classes
                        # num_classes = conf.num_classes
                        # # Convertir les labels en one-hot
                        #
                        # assert batch.classification.min() >= 0, "Erreur: Les labels doivent être >= 0"
                        # assert batch.classification.max() < num_classes, "Erreur: Les labels doivent être 0 ou 1"
                        #
                        # y_one_hot = torch.nn.functional.one_hot(batch.classification, num_classes=num_classes).float().to(
                        #     device)
                        # # Accumuler les votes par voxel
                        # votes_per_voxel = torch.zeros(num_voxels, num_classes, device=device)
                        # votes_per_voxel.scatter_add_(0, inverse_indices.view(-1, 1).expand(-1, num_classes), y_one_hot)
                        #
                        # # Obtenir la classe majoritaire pour chaque voxel
                        # y_voxel_majority = votes_per_voxel.argmax(dim=1)  # Shape: [num_voxels]
                        # # Attribuer à chaque point la classe majoritaire de son voxel
                        # y_true_per_point = y_voxel_majority[inverse_indices]  # Shape: [num_points]
                        ###VOXELS RECTANGULAIRE###
                        # pos = batch.pos  # Shape: [N, 3]
                        # voxel_size_tensor = torch.tensor(conf.voxel_size, dtype=pos.dtype, device=pos.device).view(1, -1)
                        # coords_quantized = torch.floor(pos / voxel_size_tensor)
                        # coords = torch.cat([batch.batch.unsqueeze(1), coords_quantized], dim=1)
                        # features = batch.x


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
                        # loss_val = focal_loss_fn(out_field, y_true_per_point)
                        loss_val = focal_loss_fn(out_field, y_true)
                        running_loss_val += loss_val.item()

                        # correct_results_sum_val = (
                        #     (out_field_max.view(-1) == y_true_per_point).sum().item()
                        # )
                        correct_results_sum_val = (
                            (out_field_max.view(-1) == y_true).sum().item()
                        )

                        acc_val = 100 * correct_results_sum_val / y_true.shape[0]
                        running_accuracy_val += acc_val

                        tepoch_val.set_postfix(loss=loss_val.item(), accuracy=acc_val)

                        # Calcul du f1 score
                        if conf.num_classes == 1:
                            true_positives += (
                                ((out_field_max.view(-1) == 1) & (y_true == 1)).sum().item()
                            )
                            false_positives += (
                                ((out_field_max.view(-1) == 1) & (y_true == 0)).sum().item()
                            )
                            false_negatives += (
                                ((out_field_max.view(-1) == 0) & (y_true == 1)).sum().item()
                        )

        epoch_loss_val: float = running_loss_val / len(val_data.dataset)
        epoch_accuracy_val: float = running_accuracy_val / len(val_data.dataset)

        print(
            f"Epoch {epoch + 1}/{conf.epochs}, Val Loss: {epoch_loss_val:.4f}, Val Accuracy: {epoch_accuracy_val}"
        )

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

            print(
                f"Validation F1 Score (class 1): {f1_score:.4f}, Precision: {precision}, Recall: {recall}"
            )

            if f1_score >= f1_tresh:
                f1_tresh = f1_score
                print("new best model weights save for f1_score")
                torch.save(
                    model.state_dict(),
                    f"model_weights/{conf.weights_name}_bestf1score.pth",
                )

        if epoch_loss_val < loss_val_tresh:
            loss_val_tresh = epoch_loss_val
            print("new best model weights save for loss")
            torch.save(
                model.state_dict(),
                f"/mnt/d/pointcloud_data/trainingset_eclair/photo_verticaux/weights/{conf.weights_name}_bestvallossweights.pth",
            )
            nb_patience = 0
        else:
            nb_patience += 1

        if nb_patience == conf.patience + 1:
            break

        if epoch_accuracy_val >= acc_val_tresh:
            acc_val_tresh = epoch_accuracy_val
            print("new best model weights save for acc")
            # torch.save(
            #     model.state_dict(),
            #     f"model_weights/{conf.weights_name}_bestvalaccweights.pth",
            # )

    # torch.save(
    #     model.state_dict(), f"model_weights/{conf.weights_name}_finalweights.pth"
    # )


def main() -> None:
    torch.backends.cudnn.benchmark = True

    from_cli = OmegaConf.create()
    base_conf = OmegaConf.load("configs/train_verti_3c.yaml")
    conf = OmegaConf.merge(base_conf, from_cli)

    # seed_everything(conf.random_seed)
    model = make_model(conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data, val_data = make_data(conf)

    # Charger les poids du modèle si un checkpoint existe
    checkpoint_path = f""

    if os.path.exists(checkpoint_path):
        print(f"🔄 Chargement des poids depuis {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("⚠️ Aucun checkpoint trouvé, démarrage d'un nouvel entraînement.")


    train_model(model, device, conf, train_data, val_data)


if __name__ == "__main__":
    main()
