# Patch application checklist

The current Atlas XR1 repository already contains the `nero_ee_sb` changes described in this project.

For a clean compatible XR1 checkout:

1. Set `XR1_ROOT` to the repository.
2. Confirm `mibot/utils/io.py` and `mibot/data/datasets/json_dataset.py` are the expected baseline.
3. Back up both files.
4. Run the included `apply_ee_sb.py` from the XR1 repository root:

   ```bash
   cd "$XR1_ROOT"
   python /path/to/nero-piper-ee-mix/patches/xr1/apply_ee_sb.py
   ```

5. Copy the two added helper modules from `patches/xr1/overlay/` into the corresponding XR1 paths:
   - `mibot/utils/ef_align.py`
   - `mibot/utils/piper_kin.py`

6. Compare the resulting files with `references/patched-xr1/`.
7. Run the mixed dataset validator.

Do not run both the patch script and `nero_ee_sb.patch` on the same checkout.