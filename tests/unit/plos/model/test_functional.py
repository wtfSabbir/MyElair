from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from eclair.plos.model.functional import (  # Adjust the import path based on your project structure
    AddRandomPoints,
    ElasticDistortion,
    Flip,
    Intensity_RGB_Variation,
    Noise,
    NormalizeCoordinates,
    NormalizeCoordinates_predict,
    NormalizeFeatures,
    RandomCrop,
    RandomDelete,
    RemapClassification,
    Rotate,
    Scale,
    ShearTransform,
    StretchTransform,
    compose_transforms_from_list,
)


# Fixture pour créer un objet Data de test
@pytest.fixture
def sample_data() -> Data:
    num_points = 10
    xyz = torch.randn(num_points, 3, dtype=torch.float64)
    pos = xyz.clone()
    intensity = torch.randint(0, 65535, (num_points, 1), dtype=torch.float32)
    rgb = torch.randint(0, 65535, (num_points, 3), dtype=torch.float32)
    return_number = torch.randint(0, 5, (num_points, 1), dtype=torch.float32)
    number_of_returns = torch.randint(0, 5, (num_points,), dtype=torch.float32)
    classification = torch.randint(0, 2, (num_points,), dtype=torch.long)
    return Data(
        xyz=xyz,
        intensity=intensity,
        rgb=rgb,
        classification=classification,
        return_number=return_number,
        number_of_returns=number_of_returns,
        pos=pos,
    )


@pytest.fixture
def sample_data_big() -> Data:
    num_points = 50000000
    xyz = torch.randn(num_points, 3, dtype=torch.float64)
    intensity = torch.randint(0, 65535, (num_points, 1), dtype=torch.float32)
    rgb = torch.randint(0, 65535, (num_points, 3), dtype=torch.float32)
    classification = torch.randint(0, 2, (num_points,), dtype=torch.long)
    return Data(xyz=xyz, intensity=intensity, rgb=rgb, classification=classification)


# Test de la fonction compose_transforms_from_list
def test_compose_transforms_from_list() -> None:
    # Test avec des transformations valides
    transforms_config = [
        {"name": "Scale", "params": {"feature_names": ["coordinates"]}},
        {"name": "Flip", "params": {"feature_names": ["coordinates"], "treshold": 0.5}},
    ]
    composed = compose_transforms_from_list(transforms_config)
    assert len(composed.transforms) == 2
    assert isinstance(composed.transforms[0], Scale)
    assert isinstance(composed.transforms[1], Flip)

    # Test avec une transformation inexistante
    with pytest.raises(ValueError, match="Transform NonExistentTransform not found"):
        compose_transforms_from_list([{"name": "NonExistentTransform"}])

    # Test avec plusieurs transformations inexistantes
    with pytest.raises(ValueError, match="Transform NonExistentTransform not found"):
        compose_transforms_from_list([{"name": "NonExistentTransform"}, {"name": "AnotherInvalidTransform"}])

    # Test avec un mélange de transformations valides et invalides
    mixed_config: list[dict[str, Any]] = [
        {"name": "Scale", "params": {"feature_names": ["coordinates"]}},
        {"name": "NonExistentTransform"},
    ]
    with pytest.raises(ValueError, match="Transform NonExistentTransform not found"):
        compose_transforms_from_list(mixed_config)

    # Test pour vérifier que le message d'erreur inclut les transformations disponibles
    try:
        compose_transforms_from_list([{"name": "NonExistentTransform"}])
    except ValueError as e:
        available_transforms = [
            "Intensity_RGB_Variation",
            "Scale",
            "Noise",
            "Rotate",
            "Flip",
            "RandomCrop",
            "RandomDelete",
            "NormalizeCoordinates",
            "NormalizeCoordinates_predict",
            "NormalizeFeatures",
            "RemapClassification",
            "AddRandomPoints",
            "ElasticDistortion",
            "ShearTransform",
            "StretchTransform",
        ]
        assert any(transform in str(e) for transform in available_transforms)


# Tests pour chaque transformation
def test_remap_classification(sample_data: Data) -> None:
    mapping = {0: 1, 1: 2}
    transform = RemapClassification(class_mapping=mapping)
    original_classification = sample_data.classification.clone()  # Store original values
    data = transform(sample_data.clone())

    # Check that elements originally 0 are now 1
    assert torch.all(data.classification[original_classification == 0] == 1)
    # Check that elements originally 1 are now 2
    assert torch.all(data.classification[original_classification == 1] == 2)
    # Check that data.classification equals data.y
    assert torch.equal(data.classification, data.y)


def test_intensity_rgb_variation(sample_data: Data) -> None:
    transform = Intensity_RGB_Variation(feature_names=["intensity", "colors"], rgb_max=65535)
    data = transform(sample_data.clone())
    assert data.intensity.shape == sample_data.intensity.shape
    assert data.rgb.shape == sample_data.rgb.shape
    assert torch.all(data.intensity >= 0) and torch.all(data.intensity <= 65535)
    assert torch.all(data.rgb >= 0) and torch.all(data.rgb <= 65535)


def test_rotate(sample_data: Data) -> None:
    transform = Rotate(feature_names=["coordinates"])
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    # Vérifier que les coordonnées ont changé (approximation due à l'aléatoire)
    assert not torch.allclose(data.xyz, sample_data.xyz, atol=1e-5)


def test_flip(sample_data: Data) -> None:
    transform = Flip(feature_names=["coordinates"], treshold=0.0)  # Force le flip
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    assert not torch.allclose(data.xyz[:, 1], sample_data.xyz[:, 1], atol=1e-5)


def test_scale(sample_data: Data) -> None:
    transform = Scale(feature_names=["coordinates"])
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    scale_factor = torch.norm(data.xyz) / torch.norm(sample_data.xyz)
    assert 0.9 <= scale_factor <= 1.1  # Vérifie que l'échelle est dans [0.9, 1.1]


def test_noise(sample_data: Data) -> None:
    transform = Noise(feature_names=["intensity", "colors"], rgb_max=65535)
    data = transform(sample_data.clone())
    assert data.intensity.shape == sample_data.intensity.shape
    assert data.rgb.shape == sample_data.rgb.shape
    assert torch.all(data.intensity >= 0) and torch.all(data.intensity <= 65535)
    assert torch.all(data.rgb >= 0) and torch.all(data.rgb <= 65535)


def test_random_delete(sample_data: Data) -> None:
    transform = RandomDelete(feature_names=["intensity", "colors", "return_number"])
    data = transform(sample_data.clone())
    assert data.xyz.shape[0] <= sample_data.xyz.shape[0]
    assert data.xyz.shape[0] >= int(sample_data.xyz.shape[0] * 0.9)  # Au moins 90% conservés


def test_random_crop(sample_data: Data) -> None:
    transform = RandomCrop(feature_names=["intensity", "colors"])
    data = transform(sample_data.clone())
    assert data.xyz.shape[0] <= sample_data.xyz.shape[0]
    assert data.xyz.shape[0] > 0


def test_normalize_coordinates(sample_data: Data) -> None:
    transform = NormalizeCoordinates(max_range_xy=40.0, max_range_z=40.0, z_quantile=50.0)
    data = transform(sample_data.clone())
    assert (
        data.pos.shape[0] == sample_data.xyz.shape[0] / 2 + 1
    )  # +1 pour le point fictif, /2 car on supprime 50% avec le z_quantile
    assert torch.all(data.pos[:, 2] >= torch.quantile(sample_data.xyz[:, 2], 0.5))


def test_normalize_coordinates_predict(sample_data: Data) -> None:
    transform = NormalizeCoordinates_predict(max_range_xy=40.0, max_range_z=40.0, z_quantile=50.0)
    data = transform(sample_data.clone())
    assert data.pos.shape[0] == sample_data.xyz.shape[0] / 2 + 1
    assert hasattr(data, "index_kept")
    assert hasattr(data, "index_quantile_excluded")


def test_NormalizeCoordinates_predict_big(sample_data_big: Data) -> None:
    transform = NormalizeCoordinates_predict(max_range_xy=40.0, max_range_z=40.0, z_quantile=50.0)
    data = transform(sample_data_big.clone())
    # Vérifiez le comportement lorsque l'erreur est interceptée
    assert torch.equal(data.index_quantile_excluded, torch.tensor([], dtype=torch.long, device=data.pos.device))
    assert torch.allclose(data.pos.min(dim=0).values, sample_data_big.xyz.min(dim=0).values)


def test_normalize_features(sample_data: Data) -> None:
    transform = NormalizeFeatures(
        feature_names=["intensity", "colors", "return_number", "number_of_returns", "coordinates"]
    )
    data = transform(sample_data.clone())

    assert hasattr(data, "x")  # Verify that the x attribute exists
    # Expected shape: [10, 13] (1 for intensity + 3 for rgb + 3 for coordinates + 1 for return_number + 5 for one-hot encoded number_of_returns)
    assert data.x.shape == (10, 1 + 3 + 3 + 1 + 5)
    assert data.x.dtype == torch.float32  # Verify the data type


def test_add_random_points(sample_data: Data) -> None:
    transform = AddRandomPoints(max_percentage=0.5, margin=0.1)
    data = transform(sample_data.clone())
    assert data.xyz.shape[0] >= sample_data.xyz.shape[0]
    assert data.xyz.shape[0] <= sample_data.xyz.shape[0] + int(0.5 * sample_data.xyz.shape[0])


def test_elastic_distortion(sample_data: Data) -> None:
    transform = ElasticDistortion(sigma_range=(0.01, 0.3), scale_range=(0.01, 0.1))
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    assert not torch.allclose(data.xyz, sample_data.xyz, atol=1e-5)


def test_shear_transform(sample_data: Data) -> None:
    transform = ShearTransform(shear_range=(0.001, 0.01))
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    assert not torch.allclose(data.xyz, sample_data.xyz, atol=1e-5)


def test_stretch_transform(sample_data: Data) -> None:
    transform = StretchTransform(stretch_range_bounds=(0.85, 1.15))
    data = transform(sample_data.clone())
    assert data.xyz.shape == sample_data.xyz.shape
    stretch_factor = torch.norm(data.xyz) / torch.norm(sample_data.xyz)
    assert 0.85 <= stretch_factor <= 1.15
