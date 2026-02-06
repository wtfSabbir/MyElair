# Changelog

## [0.2.1] - 2025-10-29

### Fixed

- Fixed artifacts at the ground / non-ground transition when using `EclairGroundClassifier` (!10).
- Fixed arbitrary class IDs when using `check_psanp_bundles` (!10).

## [0.2.0] - 2025-10-20

### Added

- Added support for binary models, like `EclairGroundClassifier`.
- Added a script `check_psanp_bundles` to verify that a model is compatible with PSANP.

## [0.1.5] - 2025-08-29

### Fixed

- Fixed a bug where launching the `EclairPLOClassifier` on a point cloud that only 
  contained ground caused a `ValueError`.
- The models are now in `.eval()` mode instead of `.train()` model when used during 
  prediction.

## [0.1.4] - 2025-08-28

### Fixed

- Fixed a bug where launching an inference with the GPU caused an `AssertionError`.

## [0.1.3] - 2025-08-26

### Fixed

- Fixed a bug where instantiating an `EclairClassifier` when CUDA was not available 
  raised a `RuntimeError`.

## [0.1.2] - 2025-08-13

### Added

- Implemented a *PSANP* plugin that registers an `EclairClassifier`.

## [0.1.1] - 2025-06-27

### Added

- Implemented an Eclair PLOs plugin for GSVision.

## [0.1.0] - 2025-06-18

### Added

- Packaged Eclair.
- Introduced a CI for validation and publishing.
