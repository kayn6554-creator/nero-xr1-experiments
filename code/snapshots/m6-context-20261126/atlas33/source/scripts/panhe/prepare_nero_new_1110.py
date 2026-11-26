#!/usr/bin/env python3
"""Convert the 11-10 NERO datasets (LeRobot v3) to XR-1 JSON for M3.

JSON schema is copied field by field from scripts/prepare_nero3_xr1.py (the
baseline converter); video transcoding/counting is copied from
scripts/panhe/prepare_nero_fail_prefix.py (imageio-ffmpeg, no system ffmpeg).
Differences from the baseline converter:
  * dataset list comes from --names (default: the five 11-10 datasets);
  * no normalization stats / yaml are written (M3 reuses the baseline stats);
  * instead it reports how much of the new state data falls outside the
    baseline q01/q99 (read from --baseline-config), per joint;
  * step_index must be 1..4 in every frame (step_prompt needs it).

Usage (from /mnt/panhe/xr1):
  .venv/bin/python scripts/panhe/prepare_nero_new_1110.py --source /mnt/panhe/lerobot \
      --output /mnt/panhe/dataset/xr1_nero_new_20261110 [--dry-run]
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
import yaml

NAMES = ('nero_noise_v1', 'nero_ircup_v1', 'nero_tds_v1', 'nero_ph_v1', 'nero_stemp_v1')
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def count_frames(path):
    """Decode the whole file and return (frames, (width, height))."""
    proc = subprocess.run([FFMPEG, '-hide_banner', '-nostdin', '-i', str(path), '-map', '0:v:0',
                           '-f', 'null', '-'], capture_output=True, text=True, check=True)
    frames = re.findall(r'frame=\s*(\d+)', proc.stderr)
    size = re.search(r'Video: h264.*?, (\d+)x(\d+)', proc.stderr)
    return int(frames[-1]), (int(size.group(1)), int(size.group(2))) if size else None


def transcode(job):
    src, dst, expected = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.writing.mp4')
    if not dst.exists():
        subprocess.run([FFMPEG, '-v', 'error', '-nostdin', '-y', '-threads', '2',
                        '-i', str(src), '-map', '0:v:0', '-an', '-c:v', 'libx264',
                        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-threads', '2', '-vsync', '0', str(tmp)], check=True)
        tmp.replace(dst)
    frames, size = count_frames(dst)
    assert frames == expected, (src, frames, expected)
    assert size == (640, 480), (src, size)
    return {'path': str(dst), 'frames': expected, 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='dir holding the LeRobot datasets')
    ap.add_argument('--output', required=True)
    ap.add_argument('--names', nargs='+', default=list(NAMES))
    ap.add_argument('--baseline-config', default='configs/data/nero3_m2_20261106.yaml')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true', help='check and count only, write nothing')
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.output)
    td = yaml.safe_load(Path(a.baseline_config).read_text())['data']['params']['train_datasets']
    q01, q99 = np.asarray(td['q01'], dtype=np.float64)[0, :7], np.asarray(td['q99'], dtype=np.float64)[0, :7]
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    manifest, videos, all_states = [], {}, []
    for name in a.names:
        root = source / name
        info = json.loads((root / 'meta/info.json').read_text())
        assert info['fps'] == 10 and info['robot_type'] == 'nero_follower'
        for key in ('action', 'observation.state'):
            assert info['features'][key]['names'] == [f'joint{i}.pos' for i in range(1, 8)] + ['gripper.pos']
        episodes = [e for p in sorted((root / 'meta/episodes').rglob('*.parquet')) for e in pq.read_table(p).to_pylist()]
        episodes.sort(key=lambda e: e['episode_index'])
        rows = [r for p in sorted((root / 'data').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
        rows.sort(key=lambda r: r['index'])
        assert len(rows) == info['total_frames'] and len(episodes) == info['total_episodes']
        assert [r['index'] for r in rows] == list(range(len(rows)))
        summary = {'name': name, 'episodes': len(episodes), 'frames': len(rows), 'outcomes': {},
                   'steps': {}, 'tasks': [], 'max_camera_gap_ms': 0, 'duplicate_camera_stamps': 0}
        ds_states = []
        for ep in episodes:
            ei, n = int(ep['episode_index']), int(ep['length'])
            d = rows[int(ep['dataset_from_index']):int(ep['dataset_to_index'])]
            assert len(d) == n and [r['frame_index'] for r in d] == list(range(n))
            assert all(r['episode_index'] == ei for r in d)
            assert np.allclose([r['timestamp'] for r in d], np.arange(n) / 10, atol=1e-4)
            state = np.asarray([r['observation.state'] for r in d], dtype=np.float64)
            action = np.asarray([r['action'] for r in d], dtype=np.float64)
            assert state.shape == action.shape == (n, 8)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert min(state[:, 7].min(), action[:, 7].min()) >= -1e-5, (name, ei, 'gripper < 0')
            assert max(state[:, 7].max(), action[:, 7].max()) <= .10001, (name, ei, 'gripper > 0.1')
            steps = [int(r['step_index']) for r in d]
            assert all(1 <= s <= 4 for s in steps), (name, ei, 'step_index outside 1..4', sorted(set(steps)))
            assert all(b >= a_ for a_, b in zip(steps, steps[1:])), (name, ei, 'step_index goes backwards')
            for s in set(steps):
                summary['steps'][s] = summary['steps'].get(s, 0) + steps.count(s)
            ds_states.append(state)
            stamps = np.asarray([r['camera.stamp_ms'] for r in d])
            gap = np.diff(stamps, axis=0)
            assert (gap >= 0).all(), (name, ei, 'backward camera timestamp')
            if len(gap):
                summary['max_camera_gap_ms'] = max(summary['max_camera_gap_ms'], float(gap.max()))
            summary['duplicate_camera_stamps'] += int((gap == 0).sum())
            outcome = str(ep.get('episode_success', 'unreviewed'))
            summary['outcomes'][outcome] = summary['outcomes'].get(outcome, 0) + 1
            images = {}
            for cam in ('front', 'wrist'):
                key = 'observation.images.' + cam
                stem = 'videos/' + key + '/'
                rel = Path(name) / info['video_path'].format(video_key=key, chunk_index=int(ep[stem + 'chunk_index']),
                                                             file_index=int(ep[stem + 'file_index']))
                start, end = float(ep[stem + 'from_timestamp']) * 10, float(ep[stem + 'to_timestamp']) * 10
                assert abs(start - round(start)) < 1e-3 and abs(end - start - n) < 1e-3, (name, ei, cam, start, end, n)
                src = source / rel
                assert src.is_file(), src
                dst = out / 'media' / rel
                previous = videos.get(str(src), (src, dst, 0))
                videos[str(src)] = (src, dst, max(previous[2], round(end)))
                images[cam] = [{'path': str(dst), 'start': round(start)}]
            tasks = ep['tasks']
            assert tasks and all(isinstance(t, str) and t.strip() for t in tasks)
            for t in tasks:
                if t not in summary['tasks']:
                    summary['tasks'].append(t)
            traj = {
                'time': f'{name}/episode_{ei:06d}', 'num_frames': n, 'fps': 10,
                'trajectory_type': 'success' if outcome == 'success' else 'ongoing',
                'instruction': {'general': [{'images': ['images.front', 'images.wrist'], 'conversations': [
                    {'from': 'human', 'value': '<image>\n<image>\n' + task},
                    {'from': 'gpt', 'value': ''}]} for task in tasks]},
                'images': images,
                'proprios': {'arm_joint': state[:, :7].tolist(), 'gripper_pos': state[:, 7:].tolist()},
                'actions': {'arm_joint': action[:, :7].tolist(), 'gripper_pos': action[:, 7:].tolist()},
                'source': {'dataset': name, 'episode_index': ei, 'episode_success': outcome,
                           'camera_stamp_ms': stamps.tolist(), 'step_index': steps,
                           'item_index': [r['item_index'] for r in d],
                           'action_semantics': 'recorded absolute command at t, no shift; joints radians, gripper physical 0..0.1'},
            }
            if not a.dry_run:
                dest = out / 'data' / name / f'episode_{ei:06d}.json'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(traj, ensure_ascii=False, separators=(',', ':')))
        s = np.concatenate(ds_states)[:, :7]
        summary['outside_baseline_q01_q99_pct'] = [round(float(x), 1) for x in 100 * ((s < q01) | (s > q99)).mean(0)]
        summary['state_min'] = [round(float(x), 3) for x in s.min(0)]
        summary['state_max'] = [round(float(x), 3) for x in s.max(0)]
        all_states.append(s)
        manifest.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    s = np.concatenate(all_states)
    print('baseline q01:', [round(float(x), 3) for x in q01])
    print('baseline q99:', [round(float(x), 3) for x in q99])
    print('all new data, % of frames outside baseline q01/q99 per joint:',
          [round(float(x), 1) for x in 100 * ((s < q01) | (s > q99)).mean(0)])
    if a.dry_run:
        print('DRY RUN: nothing written;', sum(x['episodes'] for x in manifest), 'episodes',
              len(s), 'frames', len(videos), 'videos')
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        probes = list(pool.map(transcode, videos.values()))
    report = {'datasets': manifest, 'episodes': sum(x['episodes'] for x in manifest), 'frames': len(s),
              'cameras': ['front', 'wrist'], 'action_layout': 'nero_joint', 'video_probes': probes}
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('COMPLETE', report['episodes'], 'episodes', report['frames'], 'frames', len(probes), 'videos', flush=True)


if __name__ == '__main__':
    main()