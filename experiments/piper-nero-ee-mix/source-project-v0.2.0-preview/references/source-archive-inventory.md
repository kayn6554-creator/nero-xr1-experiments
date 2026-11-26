# Source archive inventory

Reviewed inputs:
- `piper_nero_ee_sb_handoff_20261125.tar.gz`
- `nero_piper_train_refs_20261125.tar.gz`
- `nero_piper_train_deploy_chain_20261125.tar.gz`

The third archive included 51 entries, including Python bytecode and duplicate helper copies. The project keeps the relevant text sources/configs and excludes `__pycache__` artifacts from its working scripts.

Key findings:
- `tools/train.py` is the actual Hydra/Lightning entry point.
- `scripts/train.sh` invokes `torch.distributed.run`.
- `mibot/utils/epoch_checkpoint.py` is an optional selected-epoch callback.
- `mibot/server/deploy.py` loads model weights for inference.
- `scripts/serve_openpi_policy.py` does not yet support `nero_ee_sb`.
- The NERO client repository/source was not part of the supplied archives.