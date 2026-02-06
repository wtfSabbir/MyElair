from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
from geosat.cloud import PointCloud
from geosat.psanp.abstractions.pipelinemetadata import PipelineMetadata
from geosat.psanp.abstractions.stages import register_plugin_stages

from eclair.plos.psanp.classifiers import EclairClassifier


def add_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "bundle_names",
        nargs="+",
        help="Name of the PSANP bundle to check. Must be present in the "
        "`psanp_bundle_folder`, and the corresponding folder must contain the "
        "'weights.pth' and the 'model.yaml' files to check",
    )
    parser.add_argument(
        "--psanp_bundle_folder",
        "-p",
        help="Path to the folder that contains the PSANP bundles. "
        "Overrides the `PSANP_BUNDLE_FOLDER` environment variable",
    )
    parser.add_argument(
        "--cloud_path",
        "-c",
        help="Path to an optional LAS/LAZ point cloud to launch the classifier(s) on. "
        "The resulting clouds will be stored in the same folder, and named after "
        "the bundles",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to use for inference",
    )


def execute(args: Namespace) -> None:
    cloud_path: Path | None = Path(args.cloud_path) if args.cloud_path else None
    if cloud_path:
        print("Loading cloud from", args.cloud_path)
        cloud = PointCloud.read_from_path(cloud_path)
    else:
        cloud = PointCloud(
            [[0.0, 0.0, 0.0]],
            fields={
                "classification": np.array([0], dtype=np.uint8),
                "red": np.array([0], dtype=np.uint16),
                "green": np.array([0], dtype=np.uint16),
                "blue": np.array([0], dtype=np.uint16),
            },
        )
    print(f"{len(args.bundle_names)} bundle(s) to check:")
    for i, bundle_name in enumerate(args.bundle_names, 1):
        print(f"{i} / {len(args.bundle_names)}: Checking {bundle_name}")
        classifier = EclairClassifier(stage_name=bundle_name, device=args.device)
        all_classes = classifier.input_classes() | classifier.output_classes()
        classification_codes = {class_name: class_id for class_id, class_name in enumerate(sorted(all_classes))}
        print(f"Classification codes:\n{classification_codes}")
        metadata = PipelineMetadata(
            source=".",
            classification_codes=classification_codes,
        )
        cloud.classification[:] = 0
        (output_cloud,) = classifier.execute_on(cloud, metadata)
        if cloud_path:
            out_cloud_path = (cloud_path.parent / bundle_name).with_suffix(".laz")
            print(f"Writing {out_cloud_path}")
            output_cloud.save(out_cloud_path)
        else:
            assert len(output_cloud) == 1
        print(f"✅  {bundle_name}")


def main() -> None:
    parser = ArgumentParser(
        description="Verifies that Eclair classifier bundles (folder of `model.yaml` "
        "file + `weights.pth`) are valid for PSANP",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    if args.psanp_bundle_folder:
        os.environ["PSANP_BUNDLE_FOLDER"] = args.psanp_bundle_folder
    register_plugin_stages()
    execute(args)


if __name__ == "__main__":
    main()
