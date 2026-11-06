import json, glob, sys, collections
import numpy as np
D = 180 / np.pi
speed = collections.defaultdict(list)
runs = collections.defaultdict(list)   # (task, step, thr) -> static run lengths
for f in glob.glob(sys.argv[1] + '/*/*.json'):
    t = json.load(open(f)); task = t['source']['dataset'].split('_')[1]
    q = np.asarray(t['proprios']['arm_joint']); s = np.asarray(t['source']['step_index'])
    v = np.r_[np.inf, np.abs(np.diff(q, axis=0)).max(1) * D * 10]   # deg/s, fastest joint
    for k in (1, 2, 3, 4):
        speed[(task, k)] += list(v[(s == k)][1:])
    for thr in (1, 2, 4):
        mask = (v < thr)
        i = 0
        while i < len(s):
            if mask[i]:
                j = i
                while j < len(s) and mask[j] and s[j] == s[i]: j += 1
                runs[(task, int(s[i]), thr)].append(j - i); i = j
            else:
                i += 1
for (task, k), v in sorted(speed.items()):
    v = np.asarray(v); line = f'{task:6s} step{k} frames {len(v):6d} speed p10/p50 {np.percentile(v,10):5.1f}/{np.percentile(v,50):5.1f} deg/s |'
    for thr in (1, 2, 4):
        r = np.asarray(runs.get((task, k, thr), [0]))
        line += f' <{thr}: static {r.sum()/len(v):4.0%}, runs>=1s {int((r>=10).sum()):4d}, cut@20 {int(np.maximum(r-20,0).sum()):5d} |'
    print(line)
