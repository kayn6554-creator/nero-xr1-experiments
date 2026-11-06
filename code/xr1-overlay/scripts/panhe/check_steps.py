import json, glob, collections, statistics, sys
root = sys.argv[1]
for ds in sorted(glob.glob(root + '/*')):
    files = sorted(glob.glob(ds + '/*.json'))
    vals, first, last = collections.Counter(), collections.Counter(), collections.Counter()
    back = jump = 0
    lens, tasks = collections.defaultdict(list), collections.Counter()
    for f in files:
        t = json.load(open(f)); s = t['source']['step_index']
        vals.update(s); first[s[0]] += 1; last[s[-1]] += 1
        back += any(b < a for a, b in zip(s, s[1:]))
        jump += any(b - a > 1 for a, b in zip(s, s[1:]))
        for k in set(s): lens[k].append(s.count(k))
        tasks[t['instruction']['general'][0]['conversations'][0]['value']] += 1
    print(ds.split('/')[-1], 'episodes', len(files), 'values', dict(sorted(vals.items())))
    print('  first', dict(first), 'last', dict(last), 'backward', back, 'skip', jump)
    print('  median frames/step', {k: statistics.median(v) for k, v in sorted(lens.items())})
    print('  tasks', dict(tasks))
