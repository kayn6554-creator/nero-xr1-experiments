"""EF serving, 11-12 tweak (run from the xr1 dir on 113, after apply_ef_serve.py + apply_ef_log.py):
  * IK convergence 0.1 mm / 0.001 rad (0.057 deg) instead of 0.01 mm / 0.0001 rad: the model output
    is far noisier than that, and the tight tolerance made IK take 30-860 ms per chunk;
    override with NERO_EE_TOL_MM / NERO_EE_TOL_RAD.
  * the log line also says where the biggest joint jump was: jump_at=k<step>j<joint>
    (k0 = from the current state to the first target).
Backup: scripts/serve_openpi_policy.py.before_ef_fast.  Restart the EF service afterwards.
"""
import shutil
from pathlib import Path

PATH = "scripts/serve_openpi_policy.py"
EDITS = [
    ('    max_jump = float(os.environ.get("NERO_EE_MAX_JUMP_DEG", "15"))\n',
     '    max_jump = float(os.environ.get("NERO_EE_MAX_JUMP_DEG", "15"))\n'
     '    tol_pos = float(os.environ.get("NERO_EE_TOL_MM", "0.1")) / 1000\n'
     '    tol_rot = float(os.environ.get("NERO_EE_TOL_RAD", "0.001"))\n'),
    ('        q, ok, ep, er = K.ik(target, q_prev, q3_target=state8[2] + delta[k, 7])\n',
     '        q, ok, ep, er = K.ik(target, q_prev, q3_target=state8[2] + delta[k, 7],\n'
     '                             tol_pos=tol_pos, tol_rot=tol_rot, tol_q3=tol_rot)\n'),
    ('    worst = {"max_pos_err_mm": 0.0, "max_rot_err_deg": 0.0, "max_step_jump_deg": 0.0}\n',
     '    worst = {"max_pos_err_mm": 0.0, "max_rot_err_deg": 0.0, "max_step_jump_deg": 0.0}\n'
     '    jump_at = ""\n'),
    ('        worst["max_step_jump_deg"] = max(worst["max_step_jump_deg"], jump)\n',
     '        if jump >= worst["max_step_jump_deg"]:\n'
     '            jump_at = f"k{k}j{int(np.argmax(np.abs(q - q_prev))) + 1}"\n'
     '        worst["max_step_jump_deg"] = max(worst["max_step_jump_deg"], jump)\n'),
    ('    worst["ms"] = round(1000 * (time.perf_counter() - t0), 1)\n',
     '    worst["ms"] = round(1000 * (time.perf_counter() - t0), 1)\n'
     '    worst["jump_at"] = jump_at\n'),
    ("        f\"jump_deg={worst['max_step_jump_deg']} dj3_deg=",
     "        f\"jump_deg={worst['max_step_jump_deg']} jump_at={worst['jump_at']} dj3_deg="),
]
text = Path(PATH).read_text()
if "_ef_log" not in text:
    raise SystemExit("apply_ef_serve.py / apply_ef_log.py not applied yet, nothing written")
if "NERO_EE_TOL_MM" in text:
    raise SystemExit("already applied, nothing written")
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{PATH}: expected 1 match, found {n}:\n{old}")
    text = text.replace(old, new)
shutil.copy2(PATH, PATH + ".before_ef_fast")
Path(PATH).write_text(text)
print("edited", PATH, "(backup .before_ef_fast); restart the EF service")
