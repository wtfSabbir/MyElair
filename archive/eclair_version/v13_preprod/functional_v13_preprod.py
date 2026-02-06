import math
import random
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
import time
from collections import defaultdict
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F

class TimedTransform(BaseTransform):
    stats = defaultdict(lambda: {'total_time': 0.0, 'count': 0})

    def __init__(self, transform, name=None):
        super().__init__()
        self.transform = transform
        self.name = name or transform.__class__.__name__

    def __call__(self, data):
        start_time = time.time()
        data = self.transform(data)
        elapsed_time = time.time() - start_time

        TimedTransform.stats[self.name]['total_time'] += elapsed_time
        TimedTransform.stats[self.name]['count'] += 1

        # AJOUTE CETTE LIGNE IMMÉDIATEMENT POUR VOIR EN TEMPS RÉEL
        total = TimedTransform.stats[self.name]['total_time']
        count = TimedTransform.stats[self.name]['count']

        return data

    @staticmethod
    def print_stats():
        for name, stat in TimedTransform.stats.items():
            avg_time = stat['total_time'] / stat['count']
            print(f"{name:30s}: Total: {stat['total_time']:.2f}s | Moyenne: {avg_time:.4f}s/appel ({stat['count']} appels)")

def compose_transforms_from_list(input_transforms: List[str], timed: bool = True) -> Compose:
    """
    Receive the transformation defined in the .yaml file, verify that they are available and compose them.
    Optionally wrap transforms with TimedTransform to measure execution time.

    :param input_transforms: Transformations listed in the .yaml file.
    :param timed: Boolean flag to wrap transformations with TimedTransform.
    :return: Composed transformations (possibly timed).
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
        "NormalizeCoordinates_predict": NormalizeCoordinates_predict,
        "NormalizeFeatures": NormalizeFeatures,
        "RemapClassification": RemapClassification,
        'AddRandomPoints': AddRandomPoints,
        'ElasticDistortion': ElasticDistortion,
        'ShearTransform': ShearTransform,
        'StretchTransform': StretchTransform,
    }

    transforms: List = []
    for transform in input_transforms:
        if transform["name"] in available_transforms:
            transform_fn = available_transforms[transform["name"]](**transform.get("params", {}))
            if timed:
                transform_fn = TimedTransform(transform_fn, transform["name"])
            transforms.append(transform_fn)
        else:
            raise ValueError(
                f"Transform {transform['name']} not found. "
                f"Available transforms are: {list(available_transforms.keys())}"
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

    - Randomly adjusts intensity and RGB color values within a specified range.
    - Ensures that the RGB values remain within valid limits using np.clip.

    Attributes:
        feature_names (List[str]): List of feature names to apply variations. Supported features include
        "intensity" and "colors".

    Methods:
        __call__(data: Data) -> Data:
            Applies random variations to the specified features of the input data.
            The modified data object is returned with varied intensity and RGB values.

    Example:
        >>> transform = Intensity_RGB_Variation(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: List[str], rgb_max: int = 65535):
        """
        :param feature_names: List of features to apply variations ("intensity", "colors").
        :param rgb_max: Maximum RGB value (default 65535 for 16-bit, use 255 for 8-bit images).
        """
        super().__init__()
        self.feature_names: List[str] = feature_names
        self.rgb_max: int = rgb_max  # Définit la borne supérieure pour le clipping

    def __call__(self, data: Data) -> Data:
        if "intensity" in self.feature_names:
            random_variation_intensity = np.random.uniform(0.9, 1.1, np.shape(data.intensity))
            new_intensity = data.intensity * random_variation_intensity
            data.intensity = np.clip(new_intensity, 0, np.iinfo("uint16").max)  # Clip en 16-bit

        if "colors" in self.feature_names:
            random_variation_rgb = np.random.uniform(0.85, 1.15, np.shape(data.rgb))
            new_rgb = data.rgb * random_variation_rgb

            # Cliper les valeurs pour ne pas dépasser la borne max
            data.rgb = np.clip(new_rgb, 0, self.rgb_max)

        return data


class Rotate(BaseTransform):
    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names = feature_names

    def __call__(self, data: Data) -> Data:
        angle = 2 * math.pi * random.random()

        # Matrice de rotation directement sur GPU
        c, s = math.cos(angle), math.sin(angle)
        rotation_matrix = torch.tensor([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], device=data.xyz.device, dtype=data.xyz.dtype)

        data.xyz = torch.matmul(data.xyz, rotation_matrix)

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

    def __init__(self, feature_names: List[str], rgb_max: int = 65535):
        super().__init__()
        self.feature_names: List[str] = feature_names
        self.rgb_max: int = rgb_max
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

            data_rgb = np.clip(new_rgb, 0, self.rgb_max)
            data.rgb = data_rgb

        return data



class RandomDelete(BaseTransform):
    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names = feature_names

    def __call__(self, data: Data) -> Data:
        num_points = data.xyz.shape[0]
        device = data.xyz.device  # Récupérer l'emplacement du tenseur (GPU ou CPU)

        if num_points == 0:
            return data  # Rien à supprimer

        # Pourcentage de points à supprimer (entre 1% et 10%)
        percent_to_delete = torch.empty(1, device=device).uniform_(0.001, 0.1).item()
        num_to_delete = int(num_points * percent_to_delete)

        # Création d'un masque booléen (plus rapide que `randperm`)
        mask = torch.ones(num_points, dtype=torch.bool, device=device)
        mask[torch.randperm(num_points, device=device)[:num_to_delete]] = False

        # Vérifier que la suppression est appliquée uniformément
        data.xyz = data.xyz[mask]

        # Vérification et application du masque à tous les attributs
        if hasattr(data, "classification"):
            data.classification = data.classification[mask].clone()

        if hasattr(data, "intensity") and "intensity" in self.feature_names:
            data.intensity = data.intensity[mask]

        if hasattr(data, "rgb") and "colors" in self.feature_names:
            data.rgb = data.rgb[mask]

        if hasattr(data, "return_number") and "return_number" in self.feature_names:
            data.return_number = data.return_number[mask]

        if hasattr(data, "cluster"):
            data.cluster = data.cluster[mask]

        return data




class RandomCrop(BaseTransform):

    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names: List[str] = feature_names

    def __call__(self, data: Data) -> Data:
        coords = data.xyz
        device = coords.device

        min_coords, _ = torch.min(coords, dim=0)
        max_coords, _ = torch.max(coords, dim=0)

        # Définition du crop avec PyTorch
        crop_min = min_coords + (max_coords - min_coords) * torch.rand(3, device=device) * 0.1
        crop_max = crop_min + (max_coords - min_coords) * (0.9 + torch.rand(3, device=device) * 0.1)

        # Masque booléen (même longueur que coords)
        mask = (
            (coords[:, 0] >= crop_min[0]) & (coords[:, 0] <= crop_max[0]) &
            (coords[:, 1] >= crop_min[1]) & (coords[:, 1] <= crop_max[1]) &
            (coords[:, 2] >= crop_min[2]) & (coords[:, 2] <= crop_max[2])
        )

        # Application du même masque partout
        data.xyz = coords[mask]

        if "intensity" in self.feature_names and hasattr(data, "intensity"):
            data.intensity = data.intensity[mask]

        if "colors" in self.feature_names and hasattr(data, "rgb"):
            data.rgb = data.rgb[mask]

        if hasattr(data, "classification"):
            data.classification = data.classification[mask].clone()

        if hasattr(data, "cluster"):
            data.cluster = data.cluster[mask]

        return data


class NormalizeCoordinates(BaseTransform):
    """
    Ajuste les coordonnées XYZ en :
    - Supprimant les points hors de la plage Z.
    - Ajoutant un point artificiel (x_max, y_max, z_max) avec RGB et classification.
    """

    def __init__(self, max_range_xy=40.0, max_range_z=40.0):
        super().__init__()
        self.max_range_xy = max_range_xy  # Étendue en XY
        self.max_range_z = max_range_z  # Étendue en Z

    def __call__(self, data):

        minx=data.xyz[:, 0].min()
        miny=data.xyz[:, 1].min()
        minz=data.xyz[:, 2].min()

        # 2️⃣ Filtrer les points en dehors de la plage Z
        mask = data.xyz[:, 2] - minz <= self.max_range_z
        data.xyz = data.xyz[mask]

        # 3️⃣ Ajouter un point artificiel garantissant la même extension
        new_point = torch.tensor([[minx+self.max_range_xy, miny+self.max_range_xy, minz+self.max_range_z]], dtype=data.xyz.dtype, device=data.xyz.device)
        data.xyz = torch.cat([data.xyz, new_point], dim=0)


        # 4️⃣ Gestion des attributs supplémentaires :
        if hasattr(data, "rgb"):
            rgb = data.rgb[mask]
            new_rgb = torch.zeros(1, 3, dtype=rgb.dtype, device=rgb.device)
            data.rgb = torch.cat([rgb, new_rgb], dim=0)

        if hasattr(data, "classification"):
            classification = data.classification[mask]
            new_classification = torch.tensor([0], dtype=classification.dtype, device=classification.device)
            new_classification = new_classification.expand(1)
            data.classification = torch.cat([classification, new_classification], dim=0)

        # Mise à jour des coordonnées dans `data`

        data.pos = data.xyz.double()
        del data.xyz
        return data



class NormalizeCoordinates_predict(BaseTransform):
    """
    Ajuste les coordonnées XYZ en :
    - Supprimant les points hors de la plage Z.
    - Ajoutant un point artificiel (x_max, y_max, z_max) avec RGB et classification.
    - Sauvegardant les indices des points gardés et supprimés.
    """

    def __init__(self, max_range_xy=40.0, max_range_z=40.0):
        super().__init__()
        self.max_range_xy = max_range_xy  # Étendue en XY
        self.max_range_z = max_range_z  # Étendue en Z

    def __call__(self, data):
        coords = data.xyz.clone()

        minx=data.xyz[:, 0].min()
        miny=data.xyz[:, 1].min()
        minz=data.xyz[:, 2].min()

        # 1️⃣ Sauvegarde des indices initiaux
        index_original = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)

        x_max, y_max, z_max = self.max_range_xy, self.max_range_xy, self.max_range_z

        # 3️⃣ Filtrer les points en dehors de la plage Z
        mask = coords[:, 2] - minz <= z_max

        # Enregistrer les indices conservés et supprimés
        index_kept = index_original[mask]

        # 4️⃣ Mise à jour des coordonnées et des autres attributs
        coords = coords[mask]

        # 5️⃣ Ajouter un point artificiel garantissant la même extension
        new_point = torch.tensor([[minx+x_max, miny+y_max, minz+z_max]], dtype=coords.dtype, device=coords.device)
        coords = torch.cat([coords, new_point], dim=0)
        index_added = torch.tensor([coords.shape[0] - 1], dtype=torch.long,
                                   device=coords.device)  # Index du point ajouté

        # 6️⃣ Gestion des attributs supplémentaires :
        if hasattr(data, "rgb"):
            rgb = data.rgb[mask].clone()
            new_rgb = torch.zeros(1, 3, dtype=rgb.dtype, device=rgb.device)
            data.rgb = torch.cat([rgb, new_rgb], dim=0)

        if hasattr(data, "classification"):
            classification = data.classification[mask].clone()
            new_classification = torch.tensor([0], dtype=classification.dtype, device=classification.device)
            data.classification = torch.cat([classification, new_classification], dim=0)

        # 7️⃣ Sauvegarde des indices
        index_kept = torch.cat([index_kept, torch.tensor([index_added], dtype=torch.long, device=coords.device)])

        data.index_kept = index_kept

        # 8️⃣ Mise à jour des coordonnées dans `data`
        data.pos = coords.double()
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
        with torch.no_grad():

            features: List[torch.Tensor] = []
            if "intensity" in self.feature_names:
                features.append(torch.tensor(data.intensity))
            if "return_number" in self.feature_names:
                features.append(torch.tensor(data.return_number))
            if "number_of_returns" in self.feature_names:
                n_classes: int = 5
                number_of_returns: torch.Tensor = functional.one_hot(
                    (data.number_of_returns - 1).clamp(0, n_classes - 1).long(),
                    num_classes=n_classes,
                )
                features.append(torch.tensor(number_of_returns))
            if "coordinates" in self.feature_names:
                features.append(torch.tensor(data.pos))
            if "colors" in self.feature_names:
                features.append(data.rgb.clone()/65535)
            data.x = torch.cat(features, dim=-1).float()
        return data


class AddRandomPoints(BaseTransform):
    """
    Ajoute un nombre aléatoire de points dans le nuage de points.

    Le nombre de points ajoutés est choisi aléatoirement comme un pourcentage maximal du nombre total
    de points existants. Les nouveaux points reçoivent des coordonnées aléatoires dans une zone
    définie par une marge autour du nuage existant, leurs valeurs RGB sont générées aléatoirement,
    et leur classification est fixée à 0.
    """

    def __init__(self, max_percentage: float = 0.00001, margin: float = 0.01):
        """
        :param max_percentage: Pourcentage maximal du nombre de points existants pouvant être ajouté (entre 0 et 1).
        :param margin: Marge relative pour l'extension de la boîte englobante.
        """
        super().__init__()
        self.max_percentage = max_percentage
        self.margin = margin

    def __call__(self, data: Data) -> Data:
        with torch.no_grad():

            # On suppose que data.xyz contient les coordonnées existantes sous forme de tensor (N, 3)
            coords = data.xyz.clone()
            N_total = coords.shape[0]

            # Calculer le nombre maximum de points à ajouter et choisir un nombre aléatoire
            max_points_to_add = int(self.max_percentage * N_total)
            n_points = random.randint(0, max_points_to_add) if max_points_to_add > 0 else 0

            if n_points == 0:
                return data  # Aucun point à ajouter

            # Calculer les bornes (min, max) des coordonnées existantes
            min_coords, _ = torch.min(coords, dim=0)
            max_coords, _ = torch.max(coords, dim=0)
            range_coords = max_coords - min_coords

            # Étendre légèrement les bornes avec la marge
            # extra_min = min_coords - self.margin * range_coords
            # extra_max = max_coords + self.margin * range_coords

            # Générer n_points aléatoires dans la boîte définie par extra_min et extra_max
            new_points = torch.empty((n_points, 3), dtype=coords.dtype, device=coords.device)
            for i in range(3):
                new_points[:, i] = torch.rand(n_points, device=coords.device) * (max_coords[i] - min_coords[i]) + min_coords[i]

            # Concaténer ces nouveaux points aux existants
            data.xyz = torch.cat([coords, new_points], dim=0)

            # Générer des valeurs RGB aléatoires pour les nouveaux points (supposées dans [0, 1])
            if hasattr(data, "rgb"):
                rgb = data.rgb.clone()

                # new_rgb = torch.rand((n_points, 3), dtype=torch.float, device=rgb.device)
                new_rgb = torch.randint(0, 65535, (n_points, 3), dtype=rgb.dtype, device=rgb.device)
                data.rgb = torch.cat([rgb, new_rgb], dim=0)

            # if hasattr(data, "intensity"):
            #     intensity = data.intensity.clone()
            #     # Si intensity est 1D, on le convertit en 2D de forme (N, 1)
            #     if intensity.dim() == 1:
            #         intensity = intensity.unsqueeze(1)
            #     new_intensity = torch.rand((n_points, 1), dtype=torch.float, device=intensity.device)
            #     data.intensity = torch.cat([intensity, new_intensity], dim=0)

            # Affecter une classification à 0 pour les nouveaux points
            if hasattr(data, "classification"):
                classification = data.classification.clone()
                # On suppose que classification est de forme (N, 1) ou (N,)
                if classification.dim() == 1:
                    new_classification = torch.zeros(n_points, dtype=classification.dtype, device=classification.device)
                else:
                    new_classification = torch.zeros((n_points, classification.shape[1]), dtype=classification.dtype,
                                                     device=classification.device)
                data.classification = torch.cat([classification, new_classification], dim=0)
                # Optionnellement, on peut aussi mettre à jour data.y si c'est utilisé
                data.y = data.classification.clone()

        return data


class ElasticDistortion(BaseTransform):
    def __init__(self, sigma_range=(0.01, 0.3), scale_range=(0.01, 0.1)):
        super().__init__()
        self.sigma_range = sigma_range
        self.scale_range = scale_range
        self.kernel_size = 7  # Taille raisonnable pour un filtre gaussien
        self.sigma_kernel = 1.0  # Valeur fixe de sigma pour le noyau gaussien
        self.kernel = None  # Déclaré, mais sera déplacé sur le bon device plus tard

    @staticmethod
    def _gaussian_kernel1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype):
        """ Génère un noyau gaussien normalisé 1D sur le bon device et type. """
        coords = torch.arange(size, device=device, dtype=dtype) - size // 2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, -1)  # Format attendu par conv1d
        return kernel

    def __call__(self, data):
        device = data.xyz.device  # Récupérer où est stocké `data.xyz`
        dtype = torch.float64  # On impose le dtype float64

        # Déplacer le noyau sur le bon device et dtype s'il n'est pas encore initialisé
        if self.kernel is None or self.kernel.device != device or self.kernel.dtype != dtype:
            self.kernel = self._gaussian_kernel1d(self.kernel_size, self.sigma_kernel, device, dtype)

        coords = data.xyz  # On garde en float64

        # Générer sigma et scale aléatoirement en float64
        sigma = torch.empty(1, device=device, dtype=dtype).uniform_(*self.sigma_range).item()
        scale = torch.empty(1, device=device, dtype=dtype).uniform_(*self.scale_range).item()

        # Générer un champ aléatoire de déplacements en float64
        displacement = torch.randn_like(coords, device=device, dtype=dtype) * sigma

        # Appliquer un lissage avec la convolution 1D (dimension par dimension)
        for dim in range(3):
            displacement[:, dim] = F.conv1d(
                displacement[:, dim].unsqueeze(0).unsqueeze(0),
                self.kernel.to(dtype),
                padding=self.kernel.shape[-1] // 2
            ).squeeze()

        # Appliquer la déformation élastique
        coords += displacement * scale

        data.xyz = coords  # On garde le type float64
        return data



class ShearTransform(BaseTransform):
    """
    Applique un cisaillement aléatoire au nuage de points.

    Paramètres :
    - `shear_range` : Définie l'amplitude du cisaillement aléatoire appliqué.

    Exemple :
        shear = ShearTransform(shear_range=0.2)
        data = shear(data)
    """

    def __init__(self, shear_range:tuple = (0.001, 0.01)):
        super().__init__()
        self.shear_range = shear_range

    def __call__(self, data: Data) -> Data:
        with torch.no_grad():

            coords = data.xyz.clone()

            shear=random.uniform(*self.shear_range)
            # Matrice de cisaillement aléatoire
            shear_matrix = torch.eye(3,dtype=torch.float64)
            shear_matrix[0, 1] = random.uniform(-shear, shear)
            shear_matrix[1, 0] = random.uniform(-shear, shear)
            shear_matrix[2, 0] = random.uniform(-shear, shear)

            coords = coords @ shear_matrix.T
            data.xyz = coords
        return data


class StretchTransform(BaseTransform):
    """
    Applique un étirement ou compression aléatoire sur le nuage de points.

    - Chaque axe peut être étiré indépendamment.
    - Simule des objets plus larges ou plus fins.
    - L'intervalle de variation est lui-même aléatoire à chaque appel.

    Paramètres :
    - `stretch_range_bounds` : Intervalle pour générer aléatoirement `stretch_range` (ex: (0.7, 1.3))

    Exemple :
        stretch = StretchTransform(stretch_range_bounds=(0.7, 1.3))
        data = stretch(data)
    """

    def __init__(self, stretch_range_bounds: tuple = (0.85, 1.15)):
        super().__init__()
        self.stretch_range_bounds = stretch_range_bounds  # Intervalle possible des bornes de stretch_range

    def __call__(self, data: Data) -> Data:
        with torch.no_grad():

            coords = data.xyz.clone()

            # Générer des bornes aléatoires pour l'intervalle de stretch
            min_stretch = random.uniform(self.stretch_range_bounds[0], 1.0)  # Min entre 0.7 et 1.0
            max_stretch = random.uniform(1.0, self.stretch_range_bounds[1])  # Max entre 1.0 et 1.3

            # Générer un facteur d'étirement différent pour chaque axe (X, Y, Z) en utilisant l'intervalle aléatoire
            stretch_factors = torch.tensor([random.uniform(min_stretch, max_stretch) for _ in range(3)],
                                           dtype=coords.dtype, device=coords.device)

            coords = coords * stretch_factors
            data.xyz = coords

        return data

