#!/usr/bin/env python3
"""Convert NERO failure recordings (LeRobot v3) to XR-1 JSON, keeping only the
correct steps BEFORE the failed step.

Based on scripts/prepare_nero3_xr1.py (same JSON schema, same H.264 transcode),
with these differences:
  * failed step k = last frame's step_index, cross-checked with
    _labels/<dataset>.json "failure_types"; a mismatch skips the episode;
  * frames with step_index >= k are dropped; F1 episodes are skipped entirely;
  * trajectory_type = "ongoing" (identical to "success" in the loss);
  * ffmpeg comes from imageio-ffmpeg (no system ffmpeg/ffprobe on the server);
  * no normalization stats are written here (M2 computes its own).

Usage:
  python scripts/panhe/prepare_nero_fail_prefix.py --source /mnt/panhe/lerobot \
      --output /mnt/panhe/dataset/xr1_nero_fail_prefix_20261106 [--dry-run]
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

NAMES = ('nero_weigh_fail_v1', 'nero_temp_fail_v1', 'nero_wind_fail_v1')
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def count_frames(path):
    """Decode the whole file and return (frames, width, height)."""
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
    ap.add_argument('--source', required=True, help='dir holding nero_*_fail_v1 and _labels')
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true', help='count only, write nothing')
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.output)
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    report, videos = {'datasets': [], 'skipped': []}, {}
    for name in NAMES:
        root = source / name
        info = json.loads((root / 'meta/info.json').read_text())
        assert info['fps'] == 10 and info['robot_type'] == 'nero_follower'
        for key in ('action', 'observation.state'):
            assert info['features'][key]['names'] == [f'joint{i}.pos' for i in range(1, 8)] + ['gripper.pos']
        labels = json.loads((source / '_labels' / f'{name}.json').read_text())['failure_types']
        episodes = [e for p in sorted((root / 'meta/episodes').rglob('*.parquet')) for e in pq.read_table(p).to_pylist()]
        episodes.sort(key=lambda e: e['episode_index'])
        rows = [r for p in sorted((root / 'data').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
        rows.sort(key=lambda r: r['index'])
        assert len(rows) == info['total_frames'] and len(episodes) == info['total_episodes']
        summary = {'name': name, 'episodes': len(episodes), 'kept': 0, 'kept_frames': 0,
                   'by_failure_step': {}, 'kept_steps': {}}
        for ep in episodes:
            ei, n = int(ep['episode_index']), int(ep['length'])
            d = rows[int(ep['dataset_from_index']):int(ep['dataset_to_index'])]
            assert len(d) == n and [r['frame_index'] for r in d] == list(range(n))
            steps = [int(r['step_index']) for r in d]
            assert all(b >= a for a, b in zip(steps, steps[1:])), (name, ei, 'step_index goes backwards')
            k = steps[-1]
            label = labels.get(str(ei))
            summary['by_failure_step'][k] = summary['by_failure_step'].get(k, 0) + 1
            if label is None or int(label) != k:
                report['skipped'].append({'dataset': name, 'episode': ei, 'last_step': k, 'label': label, 'why': 'label mismatch'})
                continue
            m = steps.index(k)  # first frame of the failed step
            if m == 0:
                report['skipped'].append({'dataset': name, 'episode': ei, 'last_step': k, 'label': label, 'why': 'F1, nothing before the failure'})
                continue
            d = d[:m]
            state = np.asarray([r['observation.state'] for r in d], dtype=np.float64)
            action = np.asarray([r['action'] for r in d], dtype=np.float64)
            assert state.shape == action.shape == (m, 8)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert min(state[:, 7].min(), action[:, 7].min()) >= -1e-5
            assert max(state[:, 7].max(), action[:, 7].max()) <= .10001
            images = {}
            for cam in ('front', 'wrist'):
                key = 'observation.images.' + cam
                stem = 'videos/' + key + '/'
                rel = Path(name) / info['video_path'].format(video_key=key, chunk_index=int(ep[stem + 'chunk_index']),
                                                             file_index=int(ep[stem + 'file_index']))
                start, end = float(ep[stem + 'from_timestamp']) * 10, float(ep[stem + 'to_timestamp']) * 10
                assert abs(start - round(start)) < 1e-3 and abs(end - start - n) < 1e-3
                src = source / rel
                assert src.is_file(), src
                dst = out / 'media' / rel
                previous = videos.get(str(src), (src, dst, 0))
                videos[str(src)] = (src, dst, max(previous[2], round(end)))
                images[cam] = [{'path': str(dst), 'start': round(start)}]
            tasks = ep['tasks']
            assert tasks and all(isinstance(t, str) and t.strip() for t in tasks)
            traj = {
                'time': f'{name}/episode_{ei:06d}', 'num_frames': m, 'fps': 10,
                'trajectory_type': 'ongoing',
                'instruction': {'general': [{'images': ['images.front', 'images.wrist'], 'conversations': [
                    {'from': 'human', 'value': '<image>\n<image>\n' + task},
                    {'from': 'gpt', 'value': ''}]} for task in tasks]},
                'images': images,
                'proprios': {'arm_joint': state[:, :7].tolist(), 'gripper_pos': state[:, 7:].tolist()},
                'actions': {'arm_joint': action[:, :7].tolist(), 'gripper_pos': action[:, 7:].tolist()},
                'source': {'dataset': name, 'episode_index': ei, 'episode_success': str(ep.get('episode_success', 'failure')),
                           'failure_step': k, 'original_num_frames': n,
                           'camera_stamp_ms': [r['camera.stamp_ms'] for r in d],
                           'step_index': steps[:m], 'item_index': [r['item_index'] for r in d],
                           'action_semantics': 'recorded absolute command at t, no shift; joints radians, gripper physical 0..0.1; '
                                               'failure prefix: steps 1..k-1 only'},
            }
            summary['kept'] += 1
            summary['kept_frames'] += m
            for s in set(steps[:m]):
                summary['kept_steps'][s] = summary['kept_steps'].get(s, 0) + steps[:m].count(s)
            if not a.dry_run:
                dest = out / 'data' / name / f'episode_{ei:06d}.json'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(traj, ensure_ascii=False, separators=(',', ':')))
        report['datasets'].append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print('skipped:', len(report['skipped']), 'non-F1 mismatches:',
          [s for s in report['skipped'] if s['why'] == 'label mismatch'])
    if a.dry_run:
        print('DRY RUN: nothing written')
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        report['video_probes'] = list(pool.map(transcode, videos.values()))
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('COMPLETE', sum(x['kept'] for x in report['datasets']), 'episodes',
          sum(x['kept_frames'] for x in report['datasets']), 'frames', len(report['video_probes']), 'videos', flush=True)


if __name__ == '__main__':
    main()
