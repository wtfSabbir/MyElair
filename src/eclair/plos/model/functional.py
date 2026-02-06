"""Contient les fonctions de preprocessing et de data-augmentation des nuages de points."""

import math
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


def compose_transforms_from_list(input_transforms: list[dict[str, Any]]) -> Compose:
    """
    Construit une composition de transformations à partir d'une liste de dictionnaires décrivant les transformations.

    Chaque élément de la liste `input_transforms` doit être un dictionnaire contenant :
      - `"name"` : le nom d'une transformation (doit correspondre à une clé dans `available_transforms`)
      - `"params"` (optionnel) : un dictionnaire de paramètres à passer au constructeur de la transformation

    Exemple d'entrée :
        [
            {"name": "Scale", "params": {"scale_range": (0.9, 1.1)}},
            {"name": "Flip", "params": {"treshold": 0.5}},
        ]

    :args: input_transforms (list[dict]): Liste de dictionnaires décrivant les transformations à appliquer.
    :returns: Compose: Une instance de `torch_geometric.transforms.Compose`
            contenant toutes les transformations enchaînées.
    :raise ValueError: Si une transformation demandée n'existe pas dans `available_transforms`.
    """
    available_transforms: dict[str, type[BaseTransform]] = {
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
        "AddRandomPoints": AddRandomPoints,
        "ElasticDistortion": ElasticDistortion,
        "ShearTransform": ShearTransform,
        "StretchTransform": StretchTransform,
    }

    transforms: list[BaseTransform] = []
    for transform in input_transforms:
        if transform["name"] in available_transforms:
            transform_fn = available_transforms[transform["name"]](**transform.get("params", {}))
            transforms.append(transform_fn)
        else:
            message = (
                f"Transform {transform['name']} not found. "
                f"Available transforms are: {list(available_transforms.keys())}"
            )
            raise ValueError(message)
    return Compose(transforms=transforms)


class BaseTransform:
    """
    Base class for all transformation classes.

    This abstract base class defines the interface for all transformations to be applied to
    the point cloud data. All derived classes must implement the __call__ method.

    Methods
    -------
        __call__(data: Data) -> Data:
            Abstract method to be implemented by derived classes. Applies the transformation
            to the input data and returns the modified data.

        __repr__() -> str:
            Returns a string representation of the class instance.
    """

    def __call__(self, data: Data) -> Data:
        """
        Méthode abstraite à implémenter dans les classes dérivées.

        Cette méthode doit appliquer la transformation sur un objet Data
        et retourner l'objet modifié.

        :param data: Objet Data à transformer.
        :return: Objet Data transformé.
        :raises NotImplementedError: Si la méthode n'est pas implémentée.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Représentation textuelle de l'instance.

        :return: Nom de la classe sous forme de chaîne de caractères.
        """
        return f"{self.__class__.__name__}()"


class RemapClassification(BaseTransform):
    """
    Remaps the classification labels of the point cloud data.

    This transformation modifies the classification labels based on a provided mapping dictionary.

    Attributes
    ----------
        class_mapping (Dict[int, int]): A dictionary mapping old class labels to new class labels.

    Methods
    -------
        __call__(data: Data) -> Data:
            Applies the remapping of classification labels to the input data.
            The modified data object is returned with updated classification labels.

    Example:
        >>> transform = RemapClassification(class_mapping={0: 1, 1: 2})
        >>> data = transform(data)
    """

    def __init__(self, class_mapping: dict[int, int]) -> None:
        super().__init__()
        self.class_mapping: dict[int, int] = class_mapping

    def __call__(self, data: Data) -> Data:
        """
        Applique une remapping des classes de classification selon `self.class_mapping`.

        Pour chaque paire (ancienne_classe, nouvelle_classe) dans `class_mapping`,
        les indices dans `data.classification` correspondant à l'ancienne classe sont remplacés par la nouvelle.

        Les attributs `data.classification` et `data.y` sont mis à jour avec les nouvelles classes.

        :param data: Objet Data contenant au moins un attribut `classification` (tensor).
        :return: L'objet Data avec la classification mise à jour selon la map.
        """
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

    Attributes
    ----------
        feature_names (List[str]): List of feature names to apply variations. Supported features include
        "intensity" and "colors".

    Methods
    -------
        __call__(data: Data) -> Data:
            Applies random variations to the specified features of the input data.
            The modified data object is returned with varied intensity and RGB values.

    Example:
        >>> transform = Intensity_RGB_Variation(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: list[str], rgb_max: int = 65535) -> None:
        """
        Variation de l'intensite et du RGB.

        :param feature_names: List of features to apply variations ("intensity", "colors").
        :param rgb_max: Maximum RGB value (default 65535 for 16-bit, use 255 for 8-bit images).
        """
        super().__init__()
        self.feature_names: list[str] = feature_names
        self.rgb_max: int = rgb_max  # Définit la borne supérieure pour le clipping

    def __call__(self, data: Data) -> Data:
        """
        Applique une variation aléatoire aux intensités et aux couleurs du nuage de points.

        - Pour l'intensité, chaque valeur est multipliée par un facteur aléatoire uniforme entre 0.9 et 1.1,
          puis clipée pour rester dans l'intervalle valide (16 bits non signé).
        - Pour les couleurs (RGB), chaque canal est multiplié par un facteur aléatoire uniforme entre 0.85 et 1.15,
          puis clipé pour ne pas dépasser la valeur maximale autorisée (`self.rgb_max`).

        Cette transformation agit uniquement sur les attributs présents dans `feature_names`.

        :param data: Objet Data contenant les attributs du nuage de points (intensity, rgb).
        :return: L'objet Data modifié avec des intensités et couleurs légèrement altérées.
        """
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
    """Applique une rotation aléatoire autour de l'axe Z aux coordonnées du nuage de points."""

    def __init__(self, feature_names: list[str]) -> None:
        super().__init__()
        self.feature_names = feature_names

    def __call__(self, data: Data) -> Data:
        """
        Applique une rotation aléatoire autour de l'axe Z aux coordonnées du nuage de points.

        L'angle de rotation est choisi uniformément entre 0 et 2π radians.
        La rotation est effectuée via une matrice de rotation 3D appliquée sur le tenseur des coordonnées.

        :param data: Objet Data contenant les coordonnées XYZ du nuage de points.
        :return: L'objet Data modifié avec les coordonnées tournées autour de l'axe Z.
        """
        angle = 2 * math.pi * random.random()

        # Matrice de rotation directement sur GPU
        c, s = math.cos(angle), math.sin(angle)
        rotation_matrix = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=data.xyz.device, dtype=data.xyz.dtype)

        data.xyz = torch.matmul(data.xyz, rotation_matrix)

        return data


class Flip(BaseTransform):
    """
    Applies random flips to the point cloud data along the X and Y axes.

    This transformation randomly flips the coordinates along the X and Y axes with a 50% probability for each axis.

    Attributes: feature_names (List[str]): List of feature names to apply the flips. Supported features include
    "coordinates".

    Methods
    -------
        __call__(data: Data) -> Data:
            Applies the random flips to the input data.
            The modified data object is returned with flipped coordinates.

    Example:
        >>> transform = Flip(feature_names=["coordinates"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: list[str], treshold: float = 0.5) -> None:
        super().__init__()
        self.feature_names: list[str] = feature_names
        self.treshold = treshold

    def __call__(self, data: Data) -> Data:
        """
        Applique une réflexion aléatoire sur l'axe Y des coordonnées du nuage de points.

        Avec une probabilité égale à 1 - `self.treshold`,
        les valeurs Y des coordonnées sont inversées (multipliées par -1),
        créant un effet miroir horizontal aléatoire.

        :param data: Objet Data contenant les coordonnées XYZ du nuage de points.
        :return: L'objet Data modifié avec les coordonnées Y éventuellement inversées.
        """
        if random.random() > self.treshold:
            data.xyz[:, 1] = -data.xyz[:, 1]
        return data


class Scale(BaseTransform):
    """
    Applies a random scaling to the coordinates of the point cloud data.

    This transformation scales the coordinates by a random factor within a specified range.

    Attributes: feature_names (List[str]): List of feature names to apply the scaling. Supported features include
    "coordinates".

    Methods
    -------
        __call__(data: Data) -> Data:
            Applies the random scaling to the coordinates of the input data.
            The modified data object is returned with scaled coordinates.

    Example:
        >>> transform = Scale(feature_names=["coordinates"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: list[str]) -> None:
        super().__init__()
        self.feature_names: list[str] = feature_names

    def __call__(self, data: Data) -> Data:
        """
        Applique une variation aléatoire uniforme sur l'échelle des coordonnées XYZ du nuage de points.

        La variation est un facteur multiplicatif choisi uniformément dans l'intervalle [0.9, 1.1],
        ce qui permet de légèrement agrandir ou réduire les coordonnées.

        :param data: Objet Data contenant les coordonnées XYZ du nuage de points.
        :return: L'objet Data modifié avec les coordonnées mises à l'échelle.
        """
        random_variation_scale: float = np.random.uniform(0.9, 1.1)
        data.xyz *= random_variation_scale

        return data


class Noise(BaseTransform):
    """
    Applies random noise to the intensity and RGB color features of the point cloud data.

    This transformation adds random noise to the intensity and RGB color values within a specified range.

    Attributes: feature_names (List[str]): List of feature names to apply noise. Supported features include
    "intensity" and "colors".

    Methods
    -------
        __call__(data: Data) -> Data:
            Applies random noise to the specified features of the input data.
            The modified data object is returned with noisy intensity and RGB values.

    Example:
        >>> transform = Noise(feature_names=["intensity", "colors"])
        >>> data = transform(data)
    """

    def __init__(self, feature_names: list[str], rgb_max: int = 65535) -> None:
        super().__init__()
        self.feature_names: list[str] = feature_names
        self.rgb_max: int = rgb_max

    def __call__(self, data: Data) -> Data:
        """
        Ajoute un bruit aléatoire multiplicatif aux attributs 'intensity' et 'colors' du point cloud.

        Le bruit est généré uniformément dans l'intervalle [-0.05, 0.05] et est appliqué
        en proportion des valeurs originales (bruit multiplicatif). Les valeurs des couleurs
        sont ensuite recadrées pour rester dans l'intervalle valide [0, rgb_max].

        :param data: Objet Data contenant les attributs du nuage de points.
        :return: L'objet Data modifié avec bruit ajouté aux intensités et/ou couleurs selon les
                 features présentes.
        """
        if "intensity" in self.feature_names:
            random_noise_int: npt.NDArray[np.float32] = np.random.uniform(-0.05, 0.05, np.shape(data.intensity)).astype(
                np.float32
            )
            new_intensity: npt.NDArray[np.float32] = data.intensity + data.intensity * random_noise_int
            data.intensity = new_intensity

        if "colors" in self.feature_names:
            random_noise_rgb: npt.NDArray[np.float32] = np.random.uniform(-0.05, 0.05, np.shape(data.rgb)).astype(
                np.float32
            )
            new_rgb: npt.NDArray[np.float32] = data.rgb + data.rgb * random_noise_rgb

            data_rgb = np.clip(new_rgb, 0, self.rgb_max)
            data.rgb = data_rgb

        return data


class RandomDelete(BaseTransform):
    """Supprime aleatoirement un pourcentage de points du nuage."""

    def __init__(self, feature_names: list[str]) -> None:
        super().__init__()
        self.feature_names = feature_names

    def __call__(self, data: Data) -> Data:
        """
        Supprime aléatoirement un pourcentage de points.

        Supprime aléatoirement un pourcentage de points (entre 0.1% et 10%) du nuage de points,
        en conservant les autres attributs associés (classification, intensité, couleurs, etc.)
        synchronisés avec la suppression.

        :param data: Objet contenant les données du nuage de points, avec au moins `xyz`.
                     Peut contenir d'autres attributs comme `classification`,
                     `intensity`, `rgb`, `return_number`.
        :return: Objet `data` avec un sous-ensemble aléatoire des points et attributs mis à jour.
        """
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

        return data


class RandomCrop(BaseTransform):
    """Applique un crop aléatoire sur le nuage de points."""

    def __init__(self, feature_names: list[str]) -> None:
        super().__init__()
        self.feature_names: list[str] = feature_names

    def __call__(self, data: Data) -> Data:
        """
        Applique un crop aléatoire sur le nuage de points.

        Applique un crop aléatoire sur le nuage de points en conservant une sous-partie
        des coordonnées dans une boîte définie aléatoirement entre 10% et 100% de l'étendue
        initiale sur chaque axe. Les autres attributs associés (intensité, couleur, classification)
        sont également filtrés en conséquence.

        :param data: Objet contenant les données du nuage de points, avec au moins `xyz`.
        :return: Objet `data` avec un sous-ensemble croppé des points et attributs associés.
        """
        coords = data.xyz
        device = coords.device

        min_coords, _ = torch.min(coords, dim=0)
        max_coords, _ = torch.max(coords, dim=0)

        # Définition du crop avec PyTorch
        crop_min = min_coords + (max_coords - min_coords) * torch.rand(3, device=device) * 0.1
        crop_max = crop_min + (max_coords - min_coords) * (0.9 + torch.rand(3, device=device) * 0.1)

        # Masque booléen (même longueur que coords)
        mask = (
            (coords[:, 0] >= crop_min[0])
            & (coords[:, 0] <= crop_max[0])
            & (coords[:, 1] >= crop_min[1])
            & (coords[:, 1] <= crop_max[1])
            & (coords[:, 2] >= crop_min[2])
            & (coords[:, 2] <= crop_max[2])
        )

        # Application du même masque partout
        data.xyz = coords[mask]

        if "intensity" in self.feature_names and hasattr(data, "intensity"):
            data.intensity = data.intensity[mask]

        if "colors" in self.feature_names and hasattr(data, "rgb"):
            data.rgb = data.rgb[mask]

        if hasattr(data, "classification"):
            data.classification = data.classification[mask].clone()

        return data


class NormalizeCoordinates(BaseTransform):
    """
    Ajuste les coordonnées XYZ.

    Filtre les points hors de la plage Z en utilisant un quantile pour le `minz`.
    Ajoute un point artificiel (x_max, y_max, z_max) avec RGB et classification.
    Si le calcul du quantile échoue en raison d'un tenseur trop grand, utilise le minimum Z brut.
    """

    def __init__(self, max_range_xy: float = 40.0, max_range_z: float = 40.0, z_quantile: float = 0.5) -> None:
        super().__init__()
        self.max_range_xy = max_range_xy  # Étendue en XY
        self.max_range_z = max_range_z  # Étendue en Z
        self.z_quantile = z_quantile  # Quantile pour filtrer le bruit

    def _compute_min_xyz_clean(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Minimum XYZ filtré en ignorant les points en dessous d'un quantile sur Z.

        Gère le cas où le calcul du quantile échoue sur des très gros tenseurs
        en retournant le minimum global.

        :param coords: Coordonnées XYZ (Tensor).
        :return: Minimum XYZ filtré (Tensor).
        :raises RuntimeError: Relancé sauf si c'est une erreur liée au calcul du quantile sur un grand tenseur.
        """
        try:
            z = coords[:, 2]
            z_thresh = torch.quantile(z, self.z_quantile / 100.0)  # Quantile utilisé pour ignorer le bruit
            min_xyz_clean = coords[z >= z_thresh].min(dim=0).values
        except RuntimeError as e:
            if "quantile() input tensor is too large" in str(e):
                min_xyz_clean = coords.min(dim=0).values
            else:
                raise e  # noqa: TRY201
        return min_xyz_clean

    def __call__(self, data: Data) -> Data:
        """
        Normalise les coordonnées.

        Filtre les points 3D en appliquant un seuil sur le quantile z pour ignorer le bruit vertical,
        définit une bounding box (bbox) autour du minimum filtré, conserve uniquement les points
        dans cette bbox, ajoute un point fictif pour garantir l'extension de la tuile, et met à jour
        les attributs RGB, classification, et les coordonnées.

        :param data: Objet contenant les données du nuage de points, avec au moins `xyz` (coordonnées).
        :return: Objet `data` modifié avec points filtrés et mis à jour.
        """
        coords = data.xyz.clone()

        # 1️⃣ Calcul du minimum filtré avec gestion d'erreur
        min_xyz_clean = self._compute_min_xyz_clean(coords)

        # 2️⃣ Définition de la bbox 40x40x40 à partir du `minz` propre
        max_xyz = min_xyz_clean + torch.tensor(
            [self.max_range_xy, self.max_range_xy, self.max_range_z], device=coords.device
        )
        mask_bbox = torch.all((coords >= min_xyz_clean) & (coords <= max_xyz), dim=1)

        # 3️⃣ Filtrage des points qui tombent dans la bbox
        coords = coords[mask_bbox]

        # 4️⃣ Ajout du point fictif (juste pour garantir la même extension de la tuile)
        fake_point = (
            min_xyz_clean + torch.tensor([self.max_range_xy, self.max_range_xy, self.max_range_z], device=coords.device)
        ).unsqueeze(0)
        coords = torch.cat([coords, fake_point], dim=0)

        # 5️⃣ Attributs RGB et classification
        if hasattr(data, "rgb"):
            rgb = data.rgb[mask_bbox].clone()
            rgb = torch.cat([rgb, torch.zeros(1, 3, dtype=rgb.dtype, device=rgb.device)], dim=0)
            data.rgb = rgb

        if hasattr(data, "classification"):
            cls = data.classification[mask_bbox].clone()
            cls = torch.cat([cls, torch.tensor([0], dtype=cls.dtype, device=cls.device)], dim=0)
            data.classification = cls

        # 6️⃣ Mise à jour des coordonnées dans `data`
        data.pos = coords.double()
        del data.xyz

        return data


class NormalizeCoordinates_predict(BaseTransform):
    """
    Normalise les coordonnées.

    Ajuste les coordonnées XYZ en :
    - Ignore le bruit Z bas lors du calcul du min.
    - Garde tous les points (alignement avec le fichier LAS).
    - Sélectionne ceux dans une bbox 40x40x40 autour d un min "propre".
    - Ajoute un point fictif pour garantir l extension.
    - Enregistre les indices utilisés pour prédiction.
    """

    def __init__(self, max_range_xy: float = 40.0, max_range_z: float = 40.0, z_quantile: float = 0.5) -> None:
        super().__init__()
        self.max_range_xy = max_range_xy
        self.max_range_z = max_range_z
        self.z_quantile = z_quantile  # pour ignorer le bruit vertical

    def _compute_min_xyz_clean_and_excluded(
        self, coords: torch.Tensor, index_original: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ignore les points situés en dessous d'un quantile sur l'altitude.

        Gère le cas où le calcul du quantile échoue sur des très gros tenseurs
        en retournant le minimum global.

        :param coords: Coordonnées XYZ (Tensor).
        :param index_original: Indices originaux des points.
        :return: Tuple (min_xyz_clean, index_quantile_excluded) où :
                 - min_xyz_clean est le minimum filtré sur Z,
                 - index_quantile_excluded sont les indices exclus du quantile.
        :raises RuntimeError: Relancé sauf si c'est une erreur liée au calcul du quantile sur un grand tenseur.
        """
        try:
            z = coords[:, 2]
            z_thresh = torch.quantile(z, self.z_quantile / 100.0)
            min_xyz_clean = coords[z >= z_thresh].min(dim=0).values

            mask_quantile = z < z_thresh
            index_quantile_excluded = index_original[mask_quantile]
        except RuntimeError as e:
            if "quantile() input tensor is too large" in str(e):
                min_xyz_clean = coords.min(dim=0).values
                index_quantile_excluded = torch.tensor([], dtype=torch.long, device=coords.device)
            else:
                raise
        return min_xyz_clean, index_quantile_excluded

    def __call__(self, data: Data) -> Data:
        """
        Normalise les coordonnées.

        Filtre et restreint les points d'un nuage en fonction d'un quantile sur l'altitude,
        construit une boîte englobante 3D d'une taille fixe,
        ajoute un point fictif et met à jour les attributs associés.

        Étapes principales :
        1. Calcul du min propre sur Z en excluant les points en dessous d'un quantile défini.
        2. Définition d'une boîte englobante (bbox) autour de ce min propre.
        3. Sélection des points dans cette bbox.
        4. Ajout d'un point fictif au bout de la bbox.
        5. Mise à jour des attributs RGB et classification en filtrant selon la bbox
           et en ajoutant une valeur nulle pour le point fictif.
        6. Mise à jour des indices des points conservés, incluant le point fictif.
        7. Remplacement des coordonnées dans `data` par celles filtrées.

        :param data: Objet Data contenant les coordonnées, éventuellement les attributs rgb et classification.
        :return: Objet Data modifié avec les points filtrés, le point fictif ajouté, et les attributs mis à jour.
        :raises RuntimeError: Relancée si une erreur autre que "quantile() input tensor is too large" survient.
        """
        coords = data.xyz.clone()
        index_original = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)

        min_xyz_clean, index_quantile_excluded = self._compute_min_xyz_clean_and_excluded(coords, index_original)
        data.index_quantile_excluded = index_quantile_excluded

        max_xyz = min_xyz_clean + torch.tensor(
            [self.max_range_xy, self.max_range_xy, self.max_range_z], device=coords.device
        )
        mask_bbox = torch.all((coords >= min_xyz_clean) & (coords <= max_xyz), dim=1)

        coords_kept = coords[mask_bbox]
        index_kept = index_original[mask_bbox]

        fake_point = (
            min_xyz_clean + torch.tensor([self.max_range_xy, self.max_range_xy, self.max_range_z], device=coords.device)
        ).unsqueeze(0)
        coords_final = torch.cat([coords_kept, fake_point], dim=0)

        if hasattr(data, "rgb"):
            rgb = data.rgb[mask_bbox].clone()
            rgb = torch.cat([rgb, torch.zeros(1, 3, dtype=rgb.dtype, device=rgb.device)], dim=0)
            data.rgb = rgb

        if hasattr(data, "classification"):
            cls = data.classification[mask_bbox].clone()
            cls = torch.cat([cls, torch.tensor([0], dtype=cls.dtype, device=cls.device)], dim=0)
            data.classification = cls

        fake_idx = torch.tensor([coords_final.shape[0] - 1], dtype=torch.long, device=coords.device)
        data.index_kept = torch.cat([index_kept, fake_idx], dim=0)

        data.pos = coords_final.double()
        del data.xyz

        return data


class NormalizeFeatures(BaseTransform):
    """
    Normalizes specific features of the point cloud data.

    This transformation processes various feature attributes of the input data, including intensity,
    return number, number of returns, coordinates, and colors. The specific features to be normalized
    are specified at the initialization of the class.

    Attributes
    ----------
        feature_names (List[str]): List of feature names to normalize. Supported features include
                                   "intensity", "return_number", "number_of_returns", "coordinates", and "colors".

    Methods
    -------
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

    def __init__(self, feature_names: list[str]) -> None:
        super().__init__()
        self.feature_names: list[str] = feature_names

    def __call__(self, data: Data) -> Data:
        """
        Construit et ajoute un tenseur de features à partir des attributs sélectionnés de l'objet Data.

        :param data: Objet Data contenant des attributs tels que intensity, return_number,
                     number_of_returns, pos, rgb selon la disponibilité.
        :return: Objet Data avec l'attribut `x` contenant la concaténation des features sélectionnées,
                 normalisées et converties en float.
        """
        with torch.no_grad():
            features: list[torch.Tensor] = []
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
                features.append(data.rgb.clone() / 65535)
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

    def __init__(self, max_percentage: float = 0.00001, margin: float = 0.01) -> None:
        """
        Initialise le transformateur d'ajout de points aléatoires.

        :param max_percentage: Pourcentage maximal du nombre de points existants pouvant être ajouté (entre 0 et 1).
        :param margin: Marge relative pour l'extension de la boîte englobante.
        """
        super().__init__()
        self.max_percentage = max_percentage
        self.margin = margin

    def __call__(self, data: Data) -> Data:
        """
        Ajoute aléatoirement des points supplémentaires dans le nuage de points existant.

        :param data: Objet Data contenant au minimum un tensor xyz (N x 3) avec les coordonnées.
                     Peut contenir aussi les attributs optionnels `rgb` et `classification`.
        :return: Objet Data mis à jour avec les nouveaux points ajoutés.
                 Les nouveaux points ont des coordonnées aléatoires dans la boîte englobante
                 du nuage initial, des couleurs RGB aléatoires (si `rgb` présent),
                 et une classification à zéro (si `classification` présent).
        """
        with torch.no_grad():
            # On suppose que data.xyz contient les coordonnées existantes sous forme de tensor (N, 3)
            coords = data.xyz.clone()
            n_total = coords.shape[0]

            # Calculer le nombre maximum de points à ajouter et choisir un nombre aléatoire
            max_points_to_add = int(self.max_percentage * n_total)
            n_points = random.randint(0, max_points_to_add) if max_points_to_add > 0 else 0

            if n_points == 0:
                return data  # Aucun point à ajouter

            # Calculer les bornes (min, max) des coordonnées existantes
            min_coords, _ = torch.min(coords, dim=0)
            max_coords, _ = torch.max(coords, dim=0)

            # Générer n_points aléatoires dans la boîte définie par extra_min et extra_max
            new_points = torch.empty((n_points, 3), dtype=coords.dtype, device=coords.device)
            for i in range(3):
                new_points[:, i] = (
                    torch.rand(n_points, device=coords.device) * (max_coords[i] - min_coords[i]) + min_coords[i]
                )

            # Concaténer ces nouveaux points aux existants
            data.xyz = torch.cat([coords, new_points], dim=0)

            # Générer des valeurs RGB aléatoires pour les nouveaux points (supposées dans [0, 1])
            if hasattr(data, "rgb"):
                rgb = data.rgb.clone()
                new_rgb = torch.randint(0, 65535, (n_points, 3), dtype=rgb.dtype, device=rgb.device)
                data.rgb = torch.cat([rgb, new_rgb], dim=0)

            # Affecter une classification à 0 pour les nouveaux points
            if hasattr(data, "classification"):
                classification = data.classification.clone()
                # On suppose que classification est de forme (N, 1) ou (N,)
                if classification.dim() == 1:
                    new_classification = torch.zeros(n_points, dtype=classification.dtype, device=classification.device)
                else:
                    new_classification = torch.zeros(
                        (n_points, classification.shape[1]), dtype=classification.dtype, device=classification.device
                    )
                data.classification = torch.cat([classification, new_classification], dim=0)
                # Optionnellement, on peut aussi mettre à jour data.y si c'est utilisé
                data.y = data.classification.clone()

        return data


class ElasticDistortion(BaseTransform):
    """Applique une distortion elastique aux nuages."""

    def __init__(
        self, sigma_range: tuple[float, float] = (0.01, 0.3), scale_range: tuple[float, float] = (0.01, 0.1)
    ) -> None:
        super().__init__()
        self.sigma_range = sigma_range
        self.scale_range = scale_range
        self.kernel_size = 7  # Taille raisonnable pour un filtre gaussien
        self.sigma_kernel = 1.0  # Valeur fixe de sigma pour le noyau gaussien
        self.kernel = torch.empty(0)  # Déclaré, mais sera déplacé sur le bon device plus tard

    @staticmethod
    def _gaussian_kernel1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Génère un noyau gaussien 1D normalisé pour une convolution, sur le bon device et avec le bon type.

        :param size: Taille du noyau (nombre impair recommandé).
        :param sigma: Écart-type de la gaussienne.
        :param device: Périphérique sur lequel créer le noyau (ex. 'cpu' ou 'cuda').
        :param dtype: Type de données du tenseur (ex. torch.float32, torch.float64).

        :return: Noyau gaussien 1D normalisé de forme (1, 1, size), prêt pour conv1d.
        """
        coords = torch.arange(size, device=device, dtype=dtype) - size // 2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # Format attendu par conv1d

    def __call__(self, data: Data) -> Data:
        """
        Applique une déformation élastique aléatoire au nuage de points.

        :param data: Objet `Data` contenant les coordonnées 3D du nuage (`data.xyz`).

        :return: Objet `Data` avec coordonnées modifiées via une déformation élastique.
        """
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
                padding=self.kernel.shape[-1] // 2,
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

    def __init__(self, shear_range: tuple[float, float] = (0.001, 0.01)) -> None:
        super().__init__()
        self.shear_range = shear_range

    def __call__(self, data: Data) -> Data:
        """
        Applique une transformation de cisaillement (shear) aléatoire sur le nuage de points.

        :param data: Objet `Data` contenant les coordonnées 3D du nuage (`data.xyz`).

        :return: Objet `Data` avec des coordonnées modifiées par cisaillement.
        """
        with torch.no_grad():
            coords = data.xyz.clone()

            shear = random.uniform(*self.shear_range)
            # Matrice de cisaillement aléatoire
            shear_matrix = torch.eye(3, dtype=torch.float64)
            shear_matrix[0, 1] = random.uniform(-shear, shear)
            shear_matrix[1, 0] = random.uniform(-shear, shear)
            shear_matrix[2, 0] = random.uniform(-shear, shear)

            coords @= shear_matrix.T
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

    def __init__(self, stretch_range_bounds: tuple[float, float] = (0.85, 1.15)) -> None:
        super().__init__()
        self.stretch_range_bounds = stretch_range_bounds  # Intervalle possible des bornes de stretch_range

    def __call__(self, data: Data) -> Data:
        """
        Applique un étirement aléatoire indépendant sur chaque axe (X, Y, Z) du nuage de points.

        :param data: Objet `Data` contenant les coordonnées 3D du nuage (`data.xyz`).

        :return: Objet `Data` avec les coordonnées étirées.
        """
        with torch.no_grad():
            coords = data.xyz.clone()

            # Générer des bornes aléatoires pour l'intervalle de stretch
            min_stretch = random.uniform(self.stretch_range_bounds[0], 1.0)  # Min entre 0.7 et 1.0
            max_stretch = random.uniform(1.0, self.stretch_range_bounds[1])  # Max entre 1.0 et 1.3

            # Générer un facteur d'étirement différent pour chaque axe (X, Y, Z) en utilisant l'intervalle aléatoire
            stretch_factors = torch.tensor(
                [random.uniform(min_stretch, max_stretch) for _ in range(3)], dtype=coords.dtype, device=coords.device
            )

            coords *= stretch_factors
            data.xyz = coords

        return data
