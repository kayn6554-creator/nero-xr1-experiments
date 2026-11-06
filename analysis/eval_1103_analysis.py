import json, glob, os, collections, numpy as np
recs=[]
for t in ('weigh','task2','task3'):
    for l in open(f'vla_eval/{t}.jsonl'):
        r=json.loads(l); recs.append(r)
print('records',len(recs))
C=collections.Counter
print('exec/slow', C((r['task'],r['exec_steps'],r['slow']) for r in recs))
print('items', C((r['task'],r['item']) for r in recs).most_common(12))
print('end', C((r['task'],r['end_reason'][:18]) for r in recs))
def first_fail(r):
    for k in '1234':
        if r['steps'][k]!='done': return k if r['steps'][k]=='fail' else 'na@'+k
    return 'ok'
print('firstfail', sorted(C((r['task'],first_fail(r)) for r in recs).items()))
rows=[]
for r in recs:
    d=f"vla_runs/{r['run']}/live"
    if not os.path.exists(d+'/chunks.npz'): rows.append((r,None,None)); continue
    z=np.load(d+'/chunks.npz',allow_pickle=True); s=json.load(open(d+'/summary.json'))
    rows.append((r,z,s))
print('with npz',sum(z is not None for _,z,_ in rows))
D=np.pi/180
stat=collections.defaultdict(list)
for r,z,s in rows:
    if z is None: continue
    st,ac=z['state'],z['actions']
    if ac.dtype==object: continue
    n=min(len(st),len(ac))
    mv=np.abs(ac[:n,-1,:7]-st[:n,:7]).max(1)/D   # deg moved by end of chunk
    stall=(mv<1.0)
    g=ac[:n,:,7]
    oor=((g<-1e-4)|(g>0.1001)).any(1)
    stat['task'].append(r['task']); stat['ff'].append(first_fail(r))
    stat['nchunk'].append(n); stat['stall'].append(int(stall.sum())); stat['oor'].append(int(oor.sum()))
    stat['sync_max'].append(float(np.max(z['sync_ms']))); stat['sync_med'].append(float(np.median(z['sync_ms'])))
    stat['sync_bad'].append(int((z['sync_ms']>50).sum())); stat['lat'].append(float(np.median(z['latency'])))
    stat['slow'].append(s['args']['slow']); stat['exec'].append(s['args']['exec_steps'])
    # gripper: min predicted vs following actual state
    pairs=[(g[i].min(), st[i+1,7]) for i in range(n-1) if g[i].min()<0.03]
    stat['close_pairs'].append(pairs)
    stat['gmin'].append(float(g.min())); stat['gmax'].append(float(g.max()))
import pandas as pd
df=pd.DataFrame({k:v for k,v in stat.items() if k!='close_pairs'})
print(df.groupby('task')[['nchunk','stall','oor','sync_bad','sync_max','lat']].agg(['mean','sum','max']).round(2).to_string())
print('runs with stall chunks by task/firstfail'); print(df.assign(hs=df.stall>0).groupby(['task','ff']).hs.agg(['sum','count']).to_string())
print('runs with oor by task/ff'); print(df.assign(h=df.oor>0).groupby(['task','ff']).h.agg(['sum','count']).to_string())
print('slow/exec actual', C(zip(df.task,df.slow,df.exec)))
allsync=np.concatenate([np.load(f"vla_runs/{r['run']}/live/chunks.npz")['sync_ms'] for r,z,s in rows if z is not None])
print('sync_ms all chunks: n',len(allsync),'pct>20',(allsync>20).mean().round(3),'pct>50',(allsync>50).mean().round(3),'pct>200',(allsync>200).mean().round(3),'p50',np.median(allsync),'p90',np.percentile(allsync,90).round(1),'max',allsync.max())
cp=[p for ps in stat['close_pairs'] for p in ps]
cp=np.array(cp); print('close chunks',len(cp),'pred min median',np.median(cp[:,0]).round(4),'actual next state gripper median',np.median(cp[:,1]).round(4), 'pct actual>0.02',(cp[:,1]>0.02).mean().round(2))
print('gmin by task', df.groupby('task').gmin.describe().round(4).to_string())
