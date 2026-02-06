import os
import random
from typing import Sequence, Optional, List, Union
import laspy

from functional_v13_zmanaged import compose_transforms_from_list
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
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
        tile_path: Path = self.fnames[index]
        las: laspy.LasData = laspy.read(tile_path)

        # ➤ Downsampling : on garde un point sur 2
        mask = np.zeros(len(las.x), dtype=bool)
        mask[::2] = True  # Garde un point sur deux
        las.points = las.points[mask]  # Sous-échantillonnage avant las2pyg

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

