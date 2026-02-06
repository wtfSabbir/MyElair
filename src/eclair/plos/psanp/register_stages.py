"""The PSANP importlib plugin entrypoint."""

from geosat.psanp.abstractions.stages import register_stage

from eclair.plos.psanp.classifiers import EclairClassifier


def _register_stages() -> None:
    """Register the Eclair models as a PSANP plugin."""
    register_stage(EclairClassifier)


_register_stages()
