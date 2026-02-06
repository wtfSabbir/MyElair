"""GSVision plugin for Eclair PLOs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from geosat.gsvision.models import factory
from geosat.gsvision.models.model import GsvisionDirPath, GsvisionExistingFilePath, GsvisionPath, Model
from pydantic import Field
from typing_extensions import override

from .predict.predict import predict_dual_model as predict

if TYPE_CHECKING:
    from geosat.gsvision.models.progress_notifier import IProgressNotifier


class EclairPLOs(Model):
    """GSVision model that uses Eclair PLOs' dual-model to segment point clouds."""

    weight_path1: GsvisionExistingFilePath = Field(
        description="Path to the weights file for the first model (MinkUNet34A).",
    )

    weight_path2: GsvisionExistingFilePath = Field(
        description="Path to the weights file for the second model (MinkUNet14C).",
    )

    config_file: GsvisionExistingFilePath = Field(
        description="Path to the configuration file (YAML format).",
    )

    pointclouds: GsvisionPath = Field(
        description="Path to a JSON file containing point cloud metadata or a directory with LAS/LAZ files.",
    )

    save_path: GsvisionDirPath = Field(
        description="Directory where the output LAS/LAZ files will be saved.",
    )

    @override
    def run_model(self, progress_notifier: IProgressNotifier) -> None:
        progress_notifier.start(total_count=2)
        progress_notifier.update_state(0)
        predict(
            weights1=self.weight_path1,
            weights2=self.weight_path2,
            config=Path(self.config_file),
            pointclouds=Path(self.pointclouds),
            savepath=Path(self.save_path),
        )
        progress_notifier.update_state(1)


def initialize() -> None:
    """Register the Eclair PLOs model as a GSVision plugin."""
    factory.register(EclairPLOs.__name__, EclairPLOs)
