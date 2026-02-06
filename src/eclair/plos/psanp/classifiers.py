"""PSANP plugin for Eclair PLOs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from geosat.psanp.abstractions.classifier import Classifier
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing_extensions import Self, override

from eclair.plos.model.functional import compose_transforms_from_list
from eclair.plos.model.model import MODEL_REGISTRY, Binary_model
from eclair.plos.model.utils import collate_custom_test, seed_everything
from eclair.plos.predict.plo_post_processing import PLOPostProcessing
from eclair.plos.predict.predict import load_config, predict_model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from geosat.cloud import PointCloud
    from geosat.psanp.abstractions.option import Option
    from geosat.psanp.abstractions.pipelinemetadata import PipelineMetadata
    from omegaconf import DictConfig
    from torch_geometric.transforms import Compose

logger = getLogger(__name__)


@dataclass
class _EclairBundle:
    config: DictConfig
    model: nn.Module

    @classmethod
    def load(cls, bundle_name: str, device: torch.device) -> Self:
        """Load a bundle from the given name."""
        try:
            bundle_folder = os.environ["PSANP_BUNDLE_FOLDER"]
        except KeyError as e:
            msg = (
                "The environment variable PSANP_BUNDLE_FOLDER is not set. "
                "Please set it to the path to the folder containing the models."
            )
            raise FileNotFoundError(msg) from e
        bundle_path = Path(bundle_folder, bundle_name)
        if not bundle_path.is_dir():
            msg = f"The Eclair model folder {bundle_path} does not exist."
            raise FileNotFoundError(msg)
        conf = load_config(bundle_path / "model.yaml")
        model_class = MODEL_REGISTRY[conf.model_name]
        model: nn.Module = model_class(conf.num_features, conf.num_classes)
        if conf.num_classes == 1:
            model = Binary_model(model)
        state_dict = torch.load(bundle_path / "weights.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return cls(conf, model)


class PredictDataset(torch.utils.data.Dataset[Data]):
    """Dataset for point cloud prediction.

    :param cloud: Point cloud to launch the prediction on.
    :param transforms: Transformations to apply to the point cloud.
    """

    def __init__(self, cloud: PointCloud, transforms: Compose) -> None:
        self.cloud = cloud
        self.transforms = transforms

    @override
    def __getitem__(self, index: int) -> Data:
        if index != 0:
            msg = f"The dataset contains a single item (index 0), received {index}."
            raise IndexError(msg)
        data = Data(
            xyz=torch.from_numpy(self.cloud.coords.copy()),
            classification=torch.from_numpy(self.cloud.classification),
            rgb=torch.stack(
                [
                    torch.from_numpy(self.cloud.red << 8),
                    torch.from_numpy(self.cloud.green << 8),
                    torch.from_numpy(self.cloud.blue << 8),
                ],
                dim=-1,
            ).long(),
        )
        return self.transforms(data) if self.transforms else data

    def __len__(self) -> int:
        """Return the length of the dataset, which is always 1, as it contains a single point cloud."""
        return 1


class EclairClassifier(Classifier):
    """PSANP classifier that uses Eclair PLOs' dual-model to segment point clouds."""

    def __init__(self, stage_name: str | None, device: str | None = None) -> None:
        super().__init__(stage_name)
        if stage_name is None:
            msg = "The stage name must be specified to identify the weights and the configuration to load for Eclair."
            raise ValueError(msg)
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.bundle = _EclairBundle.load(stage_name, self.device)
        seed_everything(self.bundle.config.random_seed)

    @override
    def input_classes(self) -> set[str]:
        return set(self.bundle.config.input_classes)

    @override
    def output_classes(self) -> set[str]:
        return set(self.bundle.config.output_classes)

    @override
    def options(self) -> list[Option]:
        return []

    @override
    def execute_on(self, cloud: PointCloud, pipeline_metadata: PipelineMetadata) -> Iterable[PointCloud]:
        logger.info("Executing %s on point cloud %s…", self.stage_name, cloud.metadata.get("filepath"))
        if not len(cloud):
            return (cloud,)
        transforms = compose_transforms_from_list(self.bundle.config.test_transforms)
        dataset = PredictDataset(cloud, transforms)
        dataloader = PyGDataLoader(
            dataset,
            batch_size=self.bundle.config.test_batch_size,
            collate_fn=collate_custom_test,
            pin_memory=False,
        )
        remapping = np.array([
            pipeline_metadata.classification_codes[class_name] for class_name in self.bundle.config.output_classes
        ])
        try:
            input_classes_blacklist = self.bundle.config.input_classes_blacklist
        except AttributeError:
            input_classes_blacklist = []
        blacklist = [pipeline_metadata.classification_codes[class_name] for class_name in input_classes_blacklist]
        ground_id = pipeline_metadata.classification_codes["ground"]
        unclassified_id = pipeline_metadata.classification_codes["unclassified"]
        with torch.inference_mode():
            # Using `model.train()` instead of `model.eval()` to avoid artifacts
            # when predicting. See #!10 for more context
            self.bundle.model.train()
            for batch in dataloader:
                mask = ~np.isin(batch.classification, blacklist) if blacklist else None
                pred = predict_model(self.bundle.model, batch, self.bundle.config.voxel_size, self.device, mask)
                indices_kept = batch.index_kept.cpu().numpy()
                full_pred = np.full(len(cloud) + 1, unclassified_id, dtype=np.uint8)
                if mask is None:
                    full_pred[indices_kept] = remapping[pred]
                    index_quantile_excluded = batch.index_quantile_excluded.cpu().numpy()
                    full_pred[index_quantile_excluded] = ground_id
                    cloud.classification[:] = full_pred[:-1]
                else:
                    full_pred[indices_kept[mask]] = remapping[pred[mask]]
                    postprocessing = PLOPostProcessing(
                        pipeline_metadata.classification_codes,
                        eps=self.bundle.config.dbscan.eps,
                        min_samples=self.bundle.config.dbscan.min_samples,
                    )
                    fused_pred = postprocessing.execute(cloud.classification, full_pred[:-1], cloud.coords)
                    cloud.classification[:] = fused_pred
        return (cloud,)
