# AI / Contributor Instructions — NERO × XR-1

This repository is a long-running robotics experiment archive and working overlay. A new AI session must read the project state before editing or proposing commands.

## Required reading order

1. `README.md` — current repository map and active workstreams.
2. `docs/AI-START-HERE.md` — onboarding path for a new AI session.
3. `docs/AI-HANDOFF-2026-11-26.md` — authoritative status, known facts, blockers and next actions.
4. The relevant workstream document:
   - M6 RGB-D: `docs/research-lines/M6-RGBD.md`
   - Piper + NERO EE mixed training: `docs/research-lines/PIPER-NERO-EE-MIX.md`
5. Only then inspect the relevant code and historical records.

Do not treat an old conversation summary, historical README paragraph, source snapshot, or experiment package as more authoritative than the latest dated handoff.

## Two active workstreams — keep them separate

### M6: NERO RGB-D perception extension

- M6 is the NERO-only, M5-derived line.
- Preserve the M5 joint-space action contract (`nero_joint_sb`), 30-step horizon, robot output shape `[30, 8]`, and Step Flag semantics.
- The research variable is the observation modality: add aligned front/wrist depth to the existing RGB inputs.
- M6 is not yet implemented end-to-end. The sensor capture/storage foundation exists; Dataset, collate, model fusion, inference transport and deployment still need work.
- The intended initialization is the M5 Epoch-3 model weights, after a strict compatibility check.
- Keep the currently deployed M5 service and client path unchanged.

### Piper + NERO EE-SB mixed training

- This is a parallel experiment, not an M6 subtask.
- Its objective is to use Piper V2 left-arm data to improve a policy that will ultimately run on NERO.
- It changes the training action/state representation to `nero_ee_sb`: a 60-dimensional container with 10 valid State dimensions and 9 valid Action dimensions.
- Piper samples must mask NERO-only elbow and Step Flag supervision.
- It currently starts from the base Xiaomi-Robotics-1-5B pretrained weights, not the M5 Epoch-3 checkpoint.
- Atlas environment and real sample checks are complete; the attempted smoke runs were blocked by shared NPU contention. Formal training has **not** run.
- Do not combine the Piper data mix with M6 RGB-D until both lines have been independently validated.

## Evidence labels

Use these distinctions in docs, code comments and reports:

- **Verified in repository**: directly supported by a committed file or test.
- **User-reported / experimentally observed**: supplied by the project owner, but not independently re-run in this session.
- **Historical snapshot**: true only at its recorded time.
- **Proposed**: a design recommendation, not an accepted implementation.
- **Unknown / blocked**: evidence is missing or a required environment is unavailable.

Never turn a planned experiment into a completed result. Never infer a current process, GPU/NPU allocation, port, checkpoint, or dataset state from a historical snapshot.

## Code and source-of-truth rules

- `code/snapshots/` preserves machine/source snapshots. Do not overwrite snapshots.
- `code/xr1-overlay/`, `code/inference-overlay/`, `code/collector/` and `code/plugins/` are working copies/overlays. Before updating one, compare it with the relevant snapshot and record the source version.
- `experiments/piper-nero-ee-mix/source-project-v0.2.0-preview/` preserves the companion project package as received.
- `experiments/piper-nero-ee-mix/atlas-export-20261126/` preserves the exact Atlas export package as received.
- The two Piper launchers are not interchangeable:
  - companion-project dynamic/check-only launcher: `source-project-v0.2.0-preview/scripts/atlas/train_nero_piper_ee_sb.sh`
  - exact Atlas export of `/mnt/panhe/train_nero_piper_ee_sb.sh`: `atlas-export-20261126/atlas_new/train_nero_piper_ee_sb.sh`
- The old `atlas-export-20261126/xr1_overlay/scripts/plan_json_epochs.py` counts raw JSON frames and does **not** apply Dwell Filter. Do not use it to calculate the current 18,246-step mixed-training plan.
- Use `source-project-v0.2.0-preview/scripts/validation/plan_training_steps.py` for the Dwell-aware plan, and compare its result with the real `JsonDataset(finite_epoch=True)` scan / actual DataLoader before changing data or batch topology.

## Shared hardware and robotics safety

- Never kill, stop, restart, or modify another person's process, container, tmux session, or service.
- Never infer that Atlas is free because no training process is visible. Check NPU allocation and process/container ownership at the time of use.
- The 2026-11-26 Minimax vLLM occupancy record is a historical user-reported blocker; do not attempt to stop that container.
- Do not start training unless the user explicitly requests it and the NPU allocation is confirmed available and authorized.
- Do not start a NERO Live run, bypass IK, disable client/follower limits, or replace a live model service without explicit user authorization and the documented staged checks.
- Keep M5 6713 and all existing production models untouched while building M6 or Piper EE-SB.
- Never upload datasets, model weights, personal videos, credentials, private keys, `.env` files, or raw process dumps to this repository.
- Do not put passwords or access tokens into shell commands, documentation, logs, or GitHub.

## Before committing

1. Confirm the change belongs to the correct workstream.
2. Preserve source provenance and original copyright headers.
3. Run the narrowest relevant tests and state exactly where they ran.
4. Do not claim Atlas/NPU, 113/CUDA, or real-robot validation from local CPU tests.
5. Update `docs/AI-HANDOFF-2026-11-26.md` or add a dated handoff when project status changes.
6. Update the root README only for stable navigation/current status, not every experiment log.

## When information is missing

First search the repository, the exact source snapshots, and the dated handoff. Ask the user only for facts that cannot be recovered from those sources or that require a live permission/decision. Do not ask the user to repeat the project's history.

## 2026-11-26 additional workstream rules

- M6 `DepthRecorder` leading-missing-frame fix lives only in `code/collector/nero_depth.py`; `code/snapshots/m6-context-20261126/` is immutable. Do not claim the fix is deployed to collector 102 until a backup, patch application and real short-episode RGB-D round-trip test are recorded.
- Before M6 Dataset work, inspect a real RGB-D episode and confirm the JSON episode-index join. The repository currently has code snapshots but no real RGB-D dataset sample; do not guess episode IDs from JSON filenames.
- Piper first run must use `experiments/piper-nero-ee-mix/preflight_atlas.sh` after the repository/package is available on Atlas. This script is read-only and not yet remotely validated; it does not authorize training or prove NPU availability.
- The companion-project launcher calls `sync_config()` before its check-only exit and may create/copy config files. Do not describe it as a strictly read-only preflight. For current Atlas source, use the exact Atlas-export launcher hash recorded in the Piper workstream.
- The current mixed-training plan expects 8 NPU workers. A one-device visibility setting changes sampler steps; do not use 18,246 steps unless the runtime JsonDataset and distributed topology still produce 6,082 steps/epoch.
