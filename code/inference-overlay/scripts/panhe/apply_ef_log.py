"""EF serving: write one line per inference to the server log (run from the xr1 dir on 113).

Line format (logger "xr1.ef", INFO, goes to serve.log via the existing tee):
  EF-IK t=<HH:MM:SS.mmm> ms=<IK time> pos_err_mm=<max> rot_err_deg=<max> jump_deg=<max step jump>
        dj3_deg=<j3 change over the chunk> grip_mm=<first>-><last> dpos_mm=<flange travel over the chunk>
and on a refused chunk:  EF-IK REFUSED <reason>
Match these to testbench runs by time.  Needs apply_ef_serve.py applied first.
Backup: scripts/serve_openpi_policy.py.before_ef_log.  Restart the EF service afterwards.
"""
import shutil
from pathlib import Path

PATH = "scripts/serve_openpi_policy.py"
EDITS = [
    ('        if not ok:\n'
     '            raise ValueError(f"EF IK failed at step {k}: pos err {ep * 1000:.2f} mm, rot err {np.degrees(er):.2f} deg")\n'
     '        if jump > max_jump:\n'
     '            raise ValueError(f"EF IK joint jump {jump:.1f} deg at step {k} > {max_jump} deg")\n',
     '        if not ok:\n'
     '            _ef_log.warning(f"EF-IK REFUSED IK failed at step {k}: pos err {ep * 1000:.2f} mm, rot err {np.degrees(er):.2f} deg")\n'
     '            raise ValueError(f"EF IK failed at step {k}: pos err {ep * 1000:.2f} mm, rot err {np.degrees(er):.2f} deg")\n'
     '        if jump > max_jump:\n'
     '            _ef_log.warning(f"EF-IK REFUSED joint jump {jump:.1f} deg at step {k}")\n'
     '            raise ValueError(f"EF IK joint jump {jump:.1f} deg at step {k} > {max_jump} deg")\n'),
    ('    worst["ms"] = round(1000 * (time.perf_counter() - t0), 1)\n'
     '    return out, worst\n',
     '    worst["ms"] = round(1000 * (time.perf_counter() - t0), 1)\n'
     '    _ef_log.info(\n'
     '        f"EF-IK t={time.strftime(\'%H:%M:%S\')}.{int(time.time() * 1000) % 1000:03d} ms={worst[\'ms\']} "\n'
     '        f"pos_err_mm={worst[\'max_pos_err_mm\']} rot_err_deg={worst[\'max_rot_err_deg\']} "\n'
     '        f"jump_deg={worst[\'max_step_jump_deg\']} dj3_deg={np.degrees(out[-1, 2] - state8[2]):.2f} "\n'
     '        f"grip_mm={1000 * out[0, 7]:.0f}->{1000 * out[-1, 7]:.0f} "\n'
     '        f"dpos_mm={1000 * np.linalg.norm(delta[-1, 0:3]):.0f}")\n'
     '    return out, worst\n'),
    ('\ndef nero_ee_to_joint(',
     '\n_ef_log = __import__("logging").getLogger("xr1.ef")\n\n\ndef nero_ee_to_joint('),
]
text = Path(PATH).read_text()
if "def nero_ee_to_joint" not in text:
    raise SystemExit("apply_ef_serve.py not applied yet, nothing written")
if "_ef_log" in text:
    raise SystemExit("already applied, nothing written")
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{PATH}: expected 1 match, found {n}:\n{old}")
    text = text.replace(old, new)
shutil.copy2(PATH, PATH + ".before_ef_log")
Path(PATH).write_text(text)
print("edited", PATH, "(backup .before_ef_log); restart the EF service")
