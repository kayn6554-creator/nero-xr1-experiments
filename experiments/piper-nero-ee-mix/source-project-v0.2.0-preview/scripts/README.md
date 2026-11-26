# Scripts

- `atlas/check_environment.sh`: read-only environment inspection.
- `atlas/train_nero_piper_ee_sb.sh`: check-only by default; explicit opt-in required for training.
- `data/make_ee_sb_config_reference.py`: original data-index/config generator from the handoff archive; inspect before rerunning because it removes/recreates its mix root and refuses to overwrite an existing YAML.
- `validation/validate_mix_dataset.py`: checks source JSONs and reports raw/effective frames.
- `validation/validate_batch.py`: instantiates one NERO and one Piper JsonDataset item and checks shape/mask.
- `validation/plan_training_steps.py`: uses effective Dwell-filtered frames to plan steps.
- `training/inspect_checkpoint.py`: inspects checkpoint directories without loading large tensors.
- `training/resume_training.sh`: passes a full checkpoint path to the launcher.

- `deployment/inspect_deployment_artifact.py`: checks the Epoch 3 deployment config/weights and writes SHA256 provenance.
