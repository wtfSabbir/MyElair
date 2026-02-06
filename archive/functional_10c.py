import random
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


def compose_transforms_from_list(input_transforms: List[str]) -> Compose:
    """
    Receive the transformation defined in the .yaml file, verify that they are available and compose them
    :param input_transforms: transformation listed in the .yaml file (either in train_transform or test_transform).
    :return: Compose the listed transformation together
    """
    available_transforms: Dict = {
        "Intensity_RGB_Variation": Intensity_RGB_Variation,
        "Scale": Scale,
        "Noise": Noise,
        "Rotate": Rotate,
        "Flip": Flip,
        "RandomCrop": RandomCrop,
        "RandomDelete": RandomDelete,
        "NormalizeCoordinates": NormalizeCoordinates,
        "NormalizeFeatures": NormalizeFeatures,
        "RemapClassification": RemapClassification,
    }

    transforms: List = []
    for transform in input_transforms:
        if transform["name"] in available_transforms:
            transform_fn = available_transforms[transform["name"]]
            transforms.append(transform_fn(**transform.get("params", {})))
        else:
            raise ValueError(
                f"Transform {transform['name']} not found."
                f" Available transforms are: {list(available_transforms.keys())}"
            )
    return Compose(transforms=transforms)


class BaseTransform:
    """
    Base class for all transformation classes.

    This abstract base class defines the interface for all transformations to be applied to
    the point cloud data. All derived classes must implement the __call__ method.

    Methods:
        __call__(data: Data) -> Data:
            Abstract method to be implemented by derived classes. Applies the transformation
            to the input data and returns the modified data.

        __repr__() -> str:
            Returns a string representation of the class instance.
    """

    def __call__(self, data: Data) -> Data:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class NormalizeCoordinates(BaseTransform):
    """
    Normalizes the coordinates of the point cloud data. for z axis, divide it by 50.

    This transformation centers the coordinates around the origin and scales them so that the
    furthest point from the origin is at a unit distance.

    Methods:
        __call__(data: Data) -> Data:
            Applies the normalization to the coordinates of the input data.
            The modified data object is returned with normalized coordinates in the 'pos' attribute.

    Example:
        >>> transform = NormalizeCoordinates()
        >>> data = transform(data)
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data: Data) -> Data:

        coordinates: torch.Tensor = data.xyz

        # Step 1: Subtract the minimum z-coordinate to center vertically (relative to ground)
        coordinates[:, 2] = coordinates[:, 2]/50

        # Step 2: Normalize the vertical scale (z-axis) based on the maximum height
        # max_z = torch.max(coordinates[:, 2])
        # if max_z > 0:  # Avoid division by zero
        #     coordinates[:, 2] /= max_z

        # Step 3: Normalize the horizontal coordinates (x, y) based on their range
        min_x, max_x = torch.min(coordinates[:, 0]), torch.max(coordinates[:, 0])
        min_y, max_y = torch.min(coordinates[:, 1]), torch.max(coordinates[:, 1])

        # Avoid division by zero for x and y axes
        if max_x - min_x > 0:
            coordinates[:, 0] = (coordinates[:, 0] - min_x) / (max_x - min_x)
        if max_y - min_y > 0:
            coordinates[:, 1] = (coordinates[:, 1] - min_y) / (max_y - min_y)

        # Step 4: Store the normalized coordinates in the 'pos' attribute
        data.pos = coordinates.float()

        del data.xyz
        return data



class NormalizeFeatures(BaseTransform):
    """
    Normalizes specific features of the point cloud data.

    This transformation processes various feature attributes of the input data, including intensity,
    return number, number of returns, coordinates, and colors. The specific features to be normalized
    are specified at the initialization of the class.

    Attributes:
        feature_names (List[str]): List of feature names to normalize. Supported features include
                                   "intensity", "return_number", "number_of_returns", "coordinates", and "colors".

    Methods:
        __call__(data: Data) -> Data:
            Applies the normalization to the specified features of the input data.
            The modified data object is returned with the normalized features concatenated in the 'x' attribute.

    Example:
        >>> transform = NormalizeFeatures(feature_names=["intensity", "colors"])
        >>> data = transform(data)

    Note:
        - For the "intensity" feature, values are normalized to the range [-0.5, 0.5].
        - For "return_number" and "number_of_returns", one-hot encoding is applied with a maximum of 5 classes.
        - For "colors", RGB values are scaled to the range [-0.5, 0.5].
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        features: List[torch.Tensor] = []
        if "intensity" in self.feature_names:
            intensity: npt.NDArray = (
                data.intensity / np.iinfo("uint16").max - 0.5
            ).reshape(-1, 1)
            features.append(intensity)
        if "return_number" in self.feature_names:
            n_classes: int = 5
            return_number: torch.Tensor = functional.one_hot(
                (data.return_number - 1).clamp(0, n_classes - 1).long(),
                num_classes=n_classes,
            )
            features.append(return_number)
        if "number_of_returns" in self.feature_names:
            n_classes: int = 5
            number_of_returns: torch.Tensor = functional.one_hot(
                (data.number_of_returns - 1).clamp(0, n_classes - 1).long(),
                num_classes=n_classes,
            )
            features.append(number_of_returns)
        if "coordinates" in self.feature_names:
            coordinates: torch.Tensor = data.pos
            features.append(coordinates)
        if "colors" in self.feature_names:
            rgb: npt.NDArray = (data.rgb / 255).reshape(-1, 3)
            rgb -= 0.5
            features.append(rgb)
        data.x = torch.cat(features, dim=-1).float()
        return data


class RemapClassification(BaseTransform):
    """
    Remaps the classification labels of the point cloud data.

    This transformation modifies the classification labels based on a provided mapping dictionary.

    Attributes:
        class_mapping (Dict[int, int]): A dictionary mapping old class labels to new class labels.

    Methods:
        __call__(data: Data) -> Data:
            Applies the remapping of classification labels to the input data.
            The modified data object is returned with updated classification labels.

    Example:
        >>> transform = RemapClassification(class_mapping={0: 1, 1: 2})
        >>> data = transform(data)
    """

    def __init__(self, class_mapping: Dict[int, int]):
        super().__init__()
        self.class_mapping: Dict[int, int] = class_mapping

    def __call__(self, data: Data) -> Data:
        if self.class_mapping:
            new_y: torch.Tensor = torch.zeros_like(data.classification)
            for map_from, map_to in self.class_mapping.items():
                new_y.masked_fill_(data.classification == map_from, map_to)
            data.classification = new_y
            data.y = new_y
        return data


class Intensity_RGB_Variation(BaseTransform):
    """
    Applies random variations to the intensity and RGB color features of the point cloud data.

    This transformation randomly adjusts the intensity and RGB color values within a specified range.

    Attributes: feature_names (List[str]): List of feature names to apply variations. Supported features include
    "intensity" and "colors".

    Methods:
        __call__(data: Data) -> Data:
            Applies random variations to the specified features of the input data.
            The modified data object is returned with varied intensity and RGB values.

    Example:
        >>> transform = Intensity_RGB_Variation(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        if "intensity" in self.feature_names:
            random_variation_intensity: npt.NDArray[np.float32] = np.random.uniform(
                0.9, 1.1, np.shape(data.intensity)
            )
            new_intensity: npt.NDArray[np.float32] = (
                data.intensity * random_variation_intensity
            )
            data.intensity = new_intensity

        if "colors" in self.feature_names:
            random_variation_rgb: npt.NDArray[np.float32] = np.random.uniform(
                0.9, 1.1, np.shape(data.rgb)
            )
            new_rgb: npt.NDArray[np.float32] = data.rgb * random_variation_rgb
            data.rgb = new_rgb
        return data


def _rotate2d(pts: npt.NDArray[np.float64], angle: float) -> npt.NDArray[float]:
    """
    Rotates 2D points around the origin by a given angle.

    This function applies a 2D rotation to the (x, y) coordinates of the input points. The rotation is
    performed counterclockwise by the specified angle.

    :param pts : A NumPy array of shape (N, 3) representing the point cloud data,where N is the
    number of points and each point has (x, y, z) coordinates.
    :param angle : The rotation angle in radians.
    :return pts : The rotated points as a NumPy array of the same shape as the input.

    Example
    --------
    >>> points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> rad = np.pi / 4  # 45 degrees
    >>> rotated_pts = _rotate2d(points, rad)
    >>> print(rotated_pts)
    array([[ 0.70710678,  0.70710678,  0.],
           [-0.70710678,  0.70710678,  0.]])
    """
    rotation_matrix: npt.NDArray[float] = np.asarray(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    pts[:, :2] = pts[:, :2] @ rotation_matrix
    return pts


class Rotate(BaseTransform):
    """
    Applies a random 2D rotation to the point cloud data around the Z-axis.

    This transformation rotates the 2D coordinates (x, y) of the point cloud data by a random angle.

    Attributes: feature_names (List[str]): List of feature names to apply the rotation. Supported features
    include "coordinates".

    Methods:
        __call__(data: Data) -> Data:
            Applies the random 2D rotation to the input data.
            The modified data object is returned with rotated coordinates.

    Example:
        >>> transform = Rotate(feature_names=["coordinates"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        coordinates: torch.Tensor = data.xyz
        data.xyz = _rotate2d(coordinates, 2 * np.pi * random.random())

        return data


class Flip(BaseTransform):
    """
    Applies random flips to the point cloud data along the X and Y axes.

    This transformation randomly flips the coordinates along the X and Y axes with a 50% probability for each axis.

    Attributes: feature_names (List[str]): List of feature names to apply the flips. Supported features include
    "coordinates".

    Methods:
        __call__(data: Data) -> Data:
            Applies the random flips to the input data.
            The modified data object is returned with flipped coordinates.

    Example:
        >>> transform = Flip(feature_names=["coordinates"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        if random.random() > 0.5:
            data.xyz[:, 0] = -data.xyz[:, 0]
        if random.random() > 0.5:
            data.xyz[:, 1] = -data.xyz[:, 1]
        return data


class Scale(BaseTransform):
    """
    Applies a random scaling to the coordinates of the point cloud data.

    This transformation scales the coordinates by a random factor within a specified range.

    Attributes: feature_names (List[str]): List of feature names to apply the scaling. Supported features include
    "coordinates".

    Methods:
        __call__(data: Data) -> Data:
            Applies the random scaling to the coordinates of the input data.
            The modified data object is returned with scaled coordinates.

    Example:
        >>> transform = Scale(feature_names=["coordinates"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        random_variation_scale: float = np.random.uniform(0.9, 1.1)
        data.xyz = data.xyz * random_variation_scale

        return data


class Noise(BaseTransform):
    """
    Applies random noise to the intensity and RGB color features of the point cloud data.

    This transformation adds random noise to the intensity and RGB color values within a specified range.

    Attributes: feature_names (List[str]): List of feature names to apply noise. Supported features include
    "intensity" and "colors".

    Methods:
        __call__(data: Data) -> Data:
            Applies random noise to the specified features of the input data.
            The modified data object is returned with noisy intensity and RGB values.

    Example:
        >>> transform = Noise(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:

        if "intensity" in self.feature_names:
            random_noise_int: npt.NDArray[np.float32] = np.random.uniform(
                -0.05, 0.05, np.shape(data.intensity)
            )
            new_intensity: npt.NDArray[np.float32] = (
                data.intensity + data.intensity * random_noise_int
            )
            data.intensity = new_intensity

        if "colors" in self.feature_names:
            random_noise_rgb: npt.NDArray[np.float32] = np.random.uniform(
                -0.05, 0.05, np.shape(data.rgb)
            )
            new_rgb: npt.NDArray[np.float32] = data.rgb + data.rgb * random_noise_rgb
            data.rgb = new_rgb

        return data


class RandomDelete(BaseTransform):
    """
    Randomly deletes a percentage of points from the point cloud data.

    This transformation removes a random subset of points and their associated features from the point cloud.

    Attributes: feature_names (List[str]): List of feature names to consider during deletion. Supported features
    include "intensity", "colors", "return_number", "number_of_returns", and "classification".

    Methods:
        __call__(data: Data) -> Data:
            Applies the random deletion to the input data.
            The modified data object is returned with a subset of the original points.

    Example:
        >>> transform = RandomDelete(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        nbpoints: int = int(np.shape(data.xyz[:, 0])[0])
        percent2delete: float = np.random.uniform(0.001, 0.05)
        indice2delete: npt.NDArray[int] = np.unique(
            np.random.randint(nbpoints, size=int(percent2delete * nbpoints))
        )

        new_x: np.ndarray = np.expand_dims(np.delete(data.xyz[:, 0], indice2delete), 1)
        new_y: np.ndarray = np.expand_dims(np.delete(data.xyz[:, 1], indice2delete), 1)
        new_z: np.ndarray = np.expand_dims(np.delete(data.xyz[:, 2], indice2delete), 1)

        data.xyz = np.concatenate((new_x, new_y, new_z), axis=1)

        if "intensity" in self.feature_names:
            data.intensity = np.delete(data.intensity, indice2delete)

        if "colors" in self.feature_names:
            new_r = np.expand_dims(np.delete(data.rgb[:, 0], indice2delete), 1)
            new_g = np.expand_dims(np.delete(data.rgb[:, 1], indice2delete), 1)
            new_b = np.expand_dims(np.delete(data.rgb[:, 2], indice2delete), 1)

            data.rgb = np.concatenate((new_r, new_g, new_b), axis=1)

        if "return_number" in self.feature_names:
            data.return_number = np.delete(data.return_number, indice2delete)

        if "number_of_returns" in self.feature_names:
            data.number_of_returns = np.delete(data.number_of_returns, indice2delete)

        classif: torch.Tensor = data.classification
        data.classification = np.delete(classif, indice2delete, axis=0)

        # Propagation du champ 'cluster' s'il existe
        if hasattr(data, "cluster"):
            data.cluster = np.delete(data.cluster, indice2delete)

        return data


class RandomCrop(BaseTransform):
    """
    Applies a random crop to the point cloud data.

    This transformation selects a random cuboid region of the point cloud and retains only the points within this
    region.

    Attributes: feature_names (List[str]): List of feature names to consider during cropping. Supported features
    include "intensity", "colors", "return_number", "number_of_returns", and "classification".

    Methods:
        __call__(data: Data) -> Data:
            Applies the random crop to the input data.
            The modified data object is returned with a subset of the original points within the crop region.

    Example:
        >>> transform = RandomCrop(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:

        # Déterminer les limites du cadre de découpage
        data_xyz_abs: np.ndarray = np.abs(data.xyz)

        min_coords: np.ndarray = np.min(data_xyz_abs, 0)
        max_coords: np.ndarray = np.max(data_xyz_abs, 0)
        crop_min: np.ndarray = min_coords + (max_coords - min_coords) * [
            np.random.uniform(0, 0.15) for _ in range(3)
        ]
        crop_max: np.ndarray = crop_min + (max_coords - min_coords) * [
            np.random.uniform(0.85, 1) for _ in range(3)
        ]

        # Sélectionner les points dans les limites du cadre
        mask: np.ndarray = (
            (data_xyz_abs[:, 0] >= crop_min[0])
            & (data_xyz_abs[:, 0] <= crop_max[0])
            & (data_xyz_abs[:, 1] >= crop_min[1])
            & (data_xyz_abs[:, 1] <= crop_max[1])
        )
        indices: np.ndarray = np.nonzero(mask)

        # Mettre à jour les données
        data.xyz = torch.tensor(data.xyz[indices])

        if "intensity" in self.feature_names:
            data.intensity = torch.tensor(data.intensity[indices])

        if "colors" in self.feature_names:
            data.rgb = torch.tensor(data.rgb[indices])

        if "return_number" in self.feature_names:
            data.return_number = torch.tensor(data.return_number[indices])

        if "number_of_returns" in self.feature_names:
            data.number_of_returns = torch.tensor(data.number_of_returns[indices])

        classif: torch.Tensor = data.classification
        data.classification = classif[indices]

        # Propagation du champ 'cluster' s'il existe
        if hasattr(data, "cluster"):
            data.cluster = data.cluster[indices]

        return data

