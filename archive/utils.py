import os
import random
from typing import Sequence, Optional, List, Union


import laspy
from laspy import LasHeader, LasData

from functional_10c import compose_transforms_from_list
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path


def las2pyg(las: laspy.LasData, path: Path) -> Data:
    """
    Converts LAS data to a PyTorch Geometric Data object.

    :param las: The LAS data to be converted.
    :param path: The path to the LAS file.
    :return: The converted PyTorch Geometric Data object.

    Example
    --------
    >>> las_ = laspy.read('path/to/lasfile.las')
    >>> data = las2pyg(las_, Path('path/to/lasfile.las'))
    """
    gt_key: str = (
        "classification"
        if "classification" in set(las.point_format.dimension_names)
        else "pred"
    )
    return Data(
        xyz=torch.from_numpy(las.xyz.copy()),
        intensity=torch.from_numpy(las.intensity.astype(np.int64)),
        classification=torch.from_numpy(np.array(las[gt_key])).long(),
        return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
        number_of_returns=torch.from_numpy(np.asarray(las.number_of_returns)).long(),
        edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
        instance_id=(
            torch.from_numpy(np.asarray(las.instance).copy().astype(np.int64)).long()
            if hasattr(las, "instance")
            else torch.full((len(las.return_number),), fill_value=-1, dtype=torch.long)
        ),
        rgb=(
            torch.stack(
                [
                    torch.from_numpy(las.red.astype(np.int64)),
                    torch.from_numpy(las.green.astype(np.int64)),
                    torch.from_numpy(las.blue.astype(np.int64)),
                ],
                dim=-1,
            ).long()
            if hasattr(las, "red")
            else None
        ),
        filename=path
    )
    return data

def collate_custom_test(batch: List) -> torch.Tensor:
    """
    Custom collate function for test data.

    :param batch: A list of items to be collated.
    :return: The collated batch.
    """
    # item = [coords, feats, labels, (unique_map, inverse_map)]
    batch = [item[:3] for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path]):
        """
        Initializes the training dataset.

        :param config: Configuration object containing training parameters.
        :param fnames: List of paths to the training data files.
        """

        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.train_transforms)

    def __getitem__(self, index: int) -> Data:
        """
        Gets an item from the dataset.

        :param index: Index of the item to retrieve.
        :return: The retrieved data item.
        """

        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        return pyg_data

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        :return: The number of items in the dataset.
        :rtype: int
        """
        return len(self.fnames)


class TrainDataset_save(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path], save_augmented=True, output_dir="augmented_laz"):
        """
        Initializes the training dataset with an option to save augmented point clouds.

        :param config: Configuration object containing training parameters.
        :param fnames: List of paths to the training data files.
        :param save_augmented: If True, saves augmented data as .las files.
        :param output_dir: Directory where augmented files will be saved.
        """
        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.train_transforms)
        self.save_augmented = save_augmented
        self.output_dir = Path(output_dir)

        if self.save_augmented:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> Data:
        """
        Gets an item from the dataset, applies augmentation, and optionally saves it.

        :param index: Index of the item to retrieve.
        :return: The retrieved and transformed PyG Data object.
        """
        print(f"📂 Chargement du fichier {self.fnames[index]}")
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))

        if self.transforms:
            pyg_data = self.transforms(pyg_data)  # Appliquer les transformations


        # Optionnel : Sauvegarde du nuage augmenté
        if self.save_augmented:

            self.save_augmented_las(pyg_data, tile_path)

        return pyg_data

    def __len__(self) -> int:
        return len(self.fnames)

    def save_augmented_las(self, pyg_data: Data, original_path: Path):
        """
        Sauvegarde un nuage de points transformé au format LAS.

        :param pyg_data: Objet PyG Data contenant le nuage de points transformé.
        :param original_path: Chemin original du fichier LAS/LAZ.
        """
        output_path = self.output_dir / f"aug_{os.path.basename(original_path)}.laz"



        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.x = pyg_data.pos[:, 0].numpy().astype(np.float64)
        las.y = pyg_data.pos[:, 1].numpy().astype(np.float64)
        las.z = pyg_data.pos[:, 2].numpy().astype(np.float64)

        # las.intensity = pyg_data.intensity.numpy().astype(np.uint16)
        las.classification = pyg_data.classification.numpy().astype(np.uint8)

        if pyg_data.rgb is not None:
            if isinstance(pyg_data.rgb[:, 0], np.ndarray):
                las.red = pyg_data.rgb[:, 0].astype(np.uint16)
                las.green = pyg_data.rgb[:, 1].astype(np.uint16)
                las.blue = pyg_data.rgb[:, 2].astype(np.uint16)
            else:
                las.red = pyg_data.rgb[:, 0].numpy().astype(np.uint16)
                las.green = pyg_data.rgb[:, 1].numpy().astype(np.uint16)
                las.blue = pyg_data.rgb[:, 2].numpy().astype(np.uint16)

        las.write(str(output_path))
        print(f"✅ Nuage augmenté sauvegardé : {output_path}")


class TrainDataset_supprvoxel_save(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path], save_augmented=True, output_dir="augmented_laz", min_point_per_voxel=3):
        """
        Initializes the training dataset with an option to save augmented point clouds.

        :param config: Configuration object containing training parameters.
        :param fnames: List of paths to the training data files.
        :param save_augmented: If True, saves augmented data as .las files.
        :param output_dir: Directory where augmented files will be saved.
        """
        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.test_transforms)
        self.save_augmented = save_augmented
        self.output_dir = Path(output_dir)
        self.min_point_per_voxel=min_point_per_voxel

        if self.save_augmented:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> Data:
        """
        Gets an item from the dataset, applies augmentation, filters noise, and optionally saves it.

        :param index: Index of the item to retrieve.
        :return: The retrieved and transformed PyG Data object.
        """
        print(f"📂 Chargement du fichier {self.fnames[index]}")
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))

        # Appliquer les transformations
        if self.transforms:
            pyg_data = self.transforms(pyg_data)

        # Appliquer le filtrage des voxels sous-peuplés
        pyg_data = self.filter_sparse_voxels(pyg_data, min_points_per_voxel=self.min_point_per_voxel)

        # Optionnel : Sauvegarde du nuage augmenté avec le champ "removed_points"
        if self.save_augmented:
            self.save_augmented_las(pyg_data, tile_path)

        return pyg_data

    def __len__(self) -> int:
        return len(self.fnames)

    def filter_sparse_voxels(self, pyg_data: Data, min_points_per_voxel: int) -> Data:
        """
        Supprime les voxels contenant moins de `min_points_per_voxel` points et marque les points supprimés.
        """
        # Étape 1 : Quantification des coordonnées en voxels
        inv_voxel_size = 1.0 / 0.08  # Remplace 0.08 par ton voxel_size si besoin
        coords = torch.floor(pyg_data.pos * inv_voxel_size).long()  # Assure-toi que c'est bien en int64

        # Étape 2 : Comptage des points par voxel
        unique_coords, inverse_indices, counts = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)

        # Étape 3 : Identifier les voxels sous-peuplés
        mask_valid_voxels = counts >= min_points_per_voxel  # True si le voxel est valide

        # Calculer le nombre de points par voxel
        points_per_voxel = torch.bincount(inverse_indices, minlength=mask_valid_voxels.shape[0])

        # Créer un masque des points appartenant à des voxels sous-peuplés
        mask_points = points_per_voxel[inverse_indices] >= min_points_per_voxel

        # Debugging prints
        print(f"Total points avant filtrage: {pyg_data.pos.shape[0]}")
        print(f"Total voxels uniques: {unique_coords.shape[0]}")
        print(f"Voxels supprimés: {(~mask_valid_voxels).sum().item()}")
        print(f"Points supprimés: {(~mask_points).sum().item()}")
        print(f"Indices des points supprimés: {torch.where(~mask_points)[0][:20]}")  # Vérifie les premiers indices

        # Étape 5 : Créer `removed_points` (0 = conservé, 1 = supprimé)
        pyg_data.removed_points = (~mask_points).int()

        # # Supprimer les points qui appartiennent à des voxels sous-peuplés
        # pyg_data.pos = pyg_data.pos[mask_points]
        # pyg_data.x = pyg_data.x[mask_points] if pyg_data.x is not None else None
        # pyg_data.batch = pyg_data.batch[mask_points] if pyg_data.batch is not None else None
        # pyg_data.classification = pyg_data.classification[mask_points]
        # if pyg_data.rgb is not None:
        #     pyg_data.rgb = pyg_data.rgb[mask_points]

        return pyg_data

    def save_augmented_las(self, pyg_data: Data, original_path: Path):
        """
        Sauvegarde un nuage de points transformé au format LAS avec le champ 'removed_points'.

        :param pyg_data: Objet PyG Data contenant le nuage de points transformé.
        :param original_path: Chemin original du fichier LAS/LAZ.
        """
        output_path = self.output_dir / f"aug_{os.path.basename(original_path)}.laz"

        header = LasHeader(point_format=3, version="1.2")
        las = LasData(header)

        las.x = pyg_data.pos[:, 0].numpy().astype(np.float64)
        las.y = pyg_data.pos[:, 1].numpy().astype(np.float64)
        las.z = pyg_data.pos[:, 2].numpy().astype(np.float64)

        # Ajouter la classification
        las.classification = pyg_data.classification.numpy().astype(np.uint8)

        # Ajouter les couleurs si disponibles
        if pyg_data.rgb is not None:
            las.red = pyg_data.rgb[:, 0].numpy().astype(np.uint16)
            las.green = pyg_data.rgb[:, 1].numpy().astype(np.uint16)
            las.blue = pyg_data.rgb[:, 2].numpy().astype(np.uint16)

        # Ajouter le champ 'removed_points' pour visualisation
        las.add_extra_dim(
            laspy.ExtraBytesParams(name="removed_points", type=np.uint8)
        )
        las.removed_points = pyg_data.removed_points.numpy().astype(np.uint8)

        # Sauvegarder le fichier LAS
        las.write(str(output_path))
        print(f"✅ Nuage augmenté sauvegardé avec 'removed_points' : {output_path}")




#
# class TrainDataset_supprvoxel_save(torch.utils.data.Dataset):
#     def __init__(self, config, fnames: Sequence[Path], save_augmented=True, output_dir="augmented_laz", min_point_per_voxel=3):
#         """
#         Initialise le dataset en supprimant les points appartenant aux voxels sous-peuplés.
#         """
#         self.fnames: Sequence[Path] = fnames
#         self.transforms = compose_transforms_from_list(config.test_transforms)
#         self.save_augmented = save_augmented
#         self.output_dir = Path(output_dir)
#         self.min_point_per_voxel = min_point_per_voxel
#
#         if self.save_augmented:
#             self.output_dir.mkdir(parents=True, exist_ok=True)
#
#     def __getitem__(self, index: int) -> Data:
#         """
#         Charge un fichier LAS, supprime les points des voxels sous-peuplés, applique les transformations et sauvegarde.
#         """
#         print(f"📂 Chargement du fichier {self.fnames[index]}")
#         tile_path: Path = self.fnames[index]
#         las: laspy.LasData = laspy.read(tile_path)
#         pyg_data: Data = las2pyg(las, str(tile_path))
#
#         # Supprimer les points des voxels sous-peuplés AVANT l'entraînement
#         pyg_data = self.filter_sparse_voxels(pyg_data, self.min_point_per_voxel)
#
#         # Sauvegarde le fichier nettoyé
#         if self.save_augmented:
#             self.save_clean_las(pyg_data, tile_path)
#
#         return pyg_data
#
#     def __len__(self) -> int:
#         return len(self.fnames)
#
#     def filter_sparse_voxels(self, pyg_data: Data, min_points_per_voxel: int) -> Data:
#         """
#         Supprime les points des voxels contenant moins de `min_points_per_voxel` points.
#         """
#         # Étape 1 : Quantification des coordonnées en voxels
#         inv_voxel_size = 1.0 / 0.08
#         coords = torch.floor(pyg_data.pos * inv_voxel_size).long()
#
#         # Étape 2 : Comptage des points par voxel
#         unique_coords, inverse_indices, counts = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
#
#         # Étape 3 : Identifier les voxels sous-peuplés
#         mask_valid_voxels = counts >= min_points_per_voxel
#
#         # Étape 4 : Associer chaque point à son voxel et vérifier s'il doit être supprimé
#         mask_points = mask_valid_voxels[inverse_indices]
#
#         # Debugging
#         print(f"Total points avant filtrage: {pyg_data.pos.shape[0]}")
#         print(f"Total voxels uniques: {unique_coords.shape[0]}")
#         print(f"Voxels supprimés: {(~mask_valid_voxels).sum().item()}")
#         print(f"Points supprimés: {(~mask_points).sum().item()}")
#
#         # Étape 5 : Appliquer le filtrage (supprimer les points inutiles)
#         pyg_data.pos = pyg_data.pos[mask_points]
#         pyg_data.x = pyg_data.x[mask_points] if pyg_data.x is not None else None
#         pyg_data.batch = pyg_data.batch[mask_points] if pyg_data.batch is not None else None
#         pyg_data.classification = pyg_data.classification[mask_points]
#         if pyg_data.rgb is not None:
#             pyg_data.rgb = pyg_data.rgb[mask_points]
#
#         return pyg_data


class TrainDataset_supprvoxel_save_trainset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path], save_augmented=True, output_dir="augmented_laz", min_point_per_voxel=3):
        """
        Initialise le dataset en supprimant les points appartenant aux voxels sous-peuplés.
        """
        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.test_transforms)
        self.save_augmented = save_augmented
        self.output_dir = Path(output_dir)
        self.min_point_per_voxel = min_point_per_voxel

        if self.save_augmented:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> Data:
        """
        Charge un fichier LAS, supprime les points des voxels sous-peuplés, applique les transformations et sauvegarde.
        """
        print(f"📂 Chargement du fichier {self.fnames[index]}")
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))

        # Appliquer les transformations
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        # Supprimer les points des voxels sous-peuplés AVANT l'entraînement
        pyg_data = self.filter_sparse_voxels(pyg_data, self.min_point_per_voxel)

        # Sauvegarde le fichier nettoyé
        if self.save_augmented:
            self.save_clean_las(pyg_data, tile_path)

        return pyg_data

    def __len__(self) -> int:
        return len(self.fnames)

    def filter_sparse_voxels(self, pyg_data: Data, min_points_per_voxel: int) -> Data:
        """
        Identifie les points des voxels sous-peuplés et stocke leurs indices à supprimer.
        """
        inv_voxel_size = 1.0 / 0.08
        coords = torch.floor(pyg_data.pos * inv_voxel_size).long()

        unique_coords, inverse_indices, counts = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
        mask_valid_voxels = counts >= min_points_per_voxel

        # Utilisation de bincount pour un filtrage plus rapide
        points_per_voxel = torch.bincount(inverse_indices, minlength=mask_valid_voxels.shape[0])
        mask_points = points_per_voxel[inverse_indices] >= min_points_per_voxel

        # Stocker les indices des points conservés pour `save_clean_las()`
        pyg_data.index_kept = torch.where(mask_points)[0]

        return pyg_data

    def save_clean_las(self, pyg_data: Data, original_path: Path):
        """
        Sauvegarde un nuage nettoyé au format LAS en supprimant uniquement les points sous-peuplés,
        tout en conservant les autres informations intactes.
        """
        output_path = self.output_dir / f"{os.path.basename(original_path)}.laz"

        # Charger le fichier d'origine
        las = laspy.read(original_path)

        # Récupérer les indices des points à conserver
        indices_to_keep = pyg_data.index_kept.cpu().numpy() if hasattr(pyg_data, 'index_kept') else np.arange(len(las.x))

        # Filtrer les points dans le fichier original
        las.points = las.points[indices_to_keep]

        # Sauvegarder le fichier LAS nettoyé
        las.write(str(output_path))
        print(f"✅ Nuage nettoyé sauvegardé : {output_path}")





class ValDataset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path]):
        """
        Initializes the test dataset.

        :param config: Configuration object containing test parameters.
        :param fnames: List of paths to the test data files.
        """
        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.val_transforms)

    def __getitem__(self, index: int) -> Data:
        """
        Gets an item from the dataset.

        :param index: Index of the item to retrieve.
        :return: The retrieved data item.
        """
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        return pyg_data

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        :return: The number of items in the dataset.
        """
        return len(self.fnames)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path]):
        """
        Initializes the test dataset.

        :param config: Configuration object containing test parameters.
        :param fnames: List of paths to the test data files.
        """
        self.fnames: Sequence[Path] = fnames
        self.transforms = compose_transforms_from_list(config.test_transforms)

    def __getitem__(self, index: int) -> Data:
        """
        Gets an item from the dataset.

        :param index: Index of the item to retrieve.
        :return: The retrieved data item.
        """
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)
        pyg_data: Data = las2pyg(las, str(tile_path))
        if self.transforms:
            pyg_data = self.transforms(pyg_data)
        return pyg_data

    def __len__(self) -> int:
        """
        Gets the length of the dataset.

        :return: The number of items in the dataset.
        """
        return len(self.fnames)


def seed_everything(seed: int) -> None:
    """
    Fixes a random seed for Numpy, PyTorch, and CUDA to improve reproducibility of DL pipelines.

    :param seed: The seed to fix.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1,
        alpha: Optional[Union[float, int, List[float]]] = None,
        size_average: bool = True,
    ):
        """
        Initializes the Focal Loss.

        :param gamma: The gamma parameter for the Focal Loss, default is 1.
        :param alpha: The weights for the Focal Loss, default is None.
        :param size_average: Whether to average the loss, default is True.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None
        self.size_average = size_average

    def forward(self, inputs, target):
        """
        Computes the forward pass for the Focal Loss.

        :param inputs: The input tensor of shape (N, C, ...).
        :param target: The target tensor of shape (N, ...).
        :return: The computed focal loss.
        """
        # Ensure inputs are in the shape of (N, C) and targets are (N,)
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # (N, C, H*W)
            inputs = inputs.transpose(1, 2)  # (N, H*W, C)
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # (N*H*W, C)
        target = target.view(-1)

        # Compute log softmax of inputs and get log probabilities of the target class
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, target)
            logpt = logpt * at

        # Compute the focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Return the loss value
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: int = 1
    ):
        """
        Initializes the Focal Tversky Loss.

        :param alpha: The alpha parameter to penalize the false positives, default is 0.5.
        :param beta: The beta parameter to penalize the false negatives, default is 0.5.
        :param gamma: The gamma parameter to penalize , default is 1.0.
        :param smooth: The smoothing parameter to avoid division by zero, default is 1.0.
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Computes the forward pass for the Focal Tversky Loss.

        :param inputs: The input tensor.
        :param targets: The target tensor.
        :return: The computed Focal Tversky Loss.
        """
        # Ensure inputs and targets are flattened
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = torch.sum(inputs * targets)
        fp = torch.sum((1 - targets) * inputs)
        fn = torch.sum((1 - inputs) * targets)
        # Compute Tversky index
        Tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        # Compute Focal Tversky Loss
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class FocalTverskyLoss_multiclass(nn.Module):
    def __init__(
            self,
            gamma: float = 1.0,
            smooth: int = 1,
            per_class_alpha: torch.Tensor = None,
            per_class_beta: torch.Tensor = None,
            ignore_zero=False,
    ):
        """
        Initializes the Focal Tversky Loss for multi-class predictions with class-specific alpha and beta.

        :param alpha: Default alpha for penalizing false positives if per_class_alpha is not provided.
        :param beta: Default beta for penalizing false negatives if per_class_beta is not provided.
        :param gamma: The gamma parameter to penalize, default is 1.0.
        :param smooth: The smoothing parameter to avoid division by zero, default is 1.
        :param per_class_alpha: Tensor of shape [C] for per-class alpha values.
        :param per_class_beta: Tensor of shape [C] for per-class beta values.
        """
        super(FocalTverskyLoss_multiclass, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.per_class_alpha = torch.tensor(per_class_alpha, dtype=torch.float32)
        self.per_class_beta = torch.tensor(per_class_beta, dtype=torch.float32)
        self.ignore_zero=ignore_zero

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Computes the forward pass for the Focal Tversky Loss for multi-class predictions.

        :param inputs: The input tensor (logits or probabilities) of shape [B, C, ...].
        :param targets: The target tensor (one-hot encoded) of shape [B, C, ...].
        :return: The summed Focal Tversky Loss per class.
        """
        # Apply softmax to ensure probabilities for multi-class outputs
        inputs = torch.softmax(inputs, dim=1)  # Shape: [B, C, ...]

        # Convert targets to one-hot encoding if not already one-hot
        if targets.ndim == inputs.ndim - 1:
            targets = nn.functional.one_hot(targets, num_classes=inputs.size(1)).permute(0, -1, *range(1, targets.ndim))

        # Flatten inputs and targets across all dimensions except class dimension
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [B, C, N]
        targets = targets.view(targets.size(0), targets.size(1), -1)  # [B, C, N]

        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN) per class
        tp = torch.sum(inputs * targets, dim=2)  # Shape: [B, C]
        fp = torch.sum(inputs * (1 - targets), dim=2)  # Shape: [B, C]
        fn = torch.sum((1 - inputs) * targets, dim=2)  # Shape: [B, C]

        # Use class-specific alpha and beta
        alpha = self.per_class_alpha.to(inputs.device)  # Shape: [C]

        beta = self.per_class_beta.to(inputs.device)  # Shape: [C]

        # Reshape alpha and beta for broadcasting
        alpha = alpha.view(1, -1)  # Shape: [1, C]
        beta = beta.view(1, -1)

        # Match dimensions of tp, fp, and fn
        alpha = alpha.expand(tp.size())  # Ensure alpha matches the shape of tp, fp, and fn
        beta = beta.expand(tp.size())

        # Compute Tversky index per class
        Tversky = (tp + self.smooth) / (
                tp + alpha * fp + beta * fn + self.smooth
        )  # Shape: [B, C]

        # Compute Focal Tversky Loss per class
        FocalTversky = (1 - Tversky) ** self.gamma  # Shape: [B, C]

        if self.ignore_zero:
            # Exclude class 0 from the loss computation
            FocalTversky = FocalTversky[:, 1:]  # Ignore class 0 (column 0)

        # Sum losses over all classes
        loss = torch.sum(FocalTversky, dim=1)  # Sum over classes for each example in the batch
        loss = torch.mean(loss)  # Average over the batch

        return loss

