# Project Layout Rationale

This is a companion/overlay project, not a fork of the entire XR1 repository.

- `configs/`: versioned experiment configuration snapshots.
- `scripts/atlas/`: Atlas-specific environment and safe training launchers.
- `scripts/validation/`: data contract checks and dwell-aware step planning.
- `scripts/training/`: checkpoint inspection and resume wrapper.
- `src/`: project-owned logic, especially NERO deployment adaptation.
- `patches/xr1/`: modifications/additions required in upstream XR1.
- `references/`: supplied source snapshots for audit and comparison.
- `manifests/`: dataset, dependency, and upstream-file provenance.

The upstream XR1 repo remains the source of truth for VLM/DiT, Lightning Trainer, DataModule, model loader, and current NERO controller implementation.