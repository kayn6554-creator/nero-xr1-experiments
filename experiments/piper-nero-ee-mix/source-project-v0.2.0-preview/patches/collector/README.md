# Collector integration: Step Flag without Step Prompt

The 2026-11-16 M5 client patch in the upstream repository requires both `step_prompt` and `step_file`. That is correct for the M5 joint model trained with step text.

The new mixed `nero_ee_sb` data config has:
- `step_prompt: false`
- Step Flag supervision enabled in action slot 8.

For this model the client must:
1. Require `step_file` when metadata `step_flag.enabled=true`.
2. Enable boundary pause and manual confirmation.
3. Parse `step_flag` and `step_boundary` response fields.
4. Keep `--step-prompt` disabled when metadata `step_prompt.enabled=false`.
5. Continue to require `--step-prompt` only when metadata declares it enabled.

The current GitHub `code/collector/nero_vla_client.py` is an earlier source snapshot; the complete 2026-11-16 deployed M5 patch is represented by hashes/tests, not a full committed source file. Do not overwrite the deployed client with the older snapshot.
