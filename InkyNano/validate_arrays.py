"""Independent CPU check of saved causal demand arrays and source grouping."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def norm(x):return x/np.maximum(x.sum(axis=-1,keepdims=True),1e-12)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--routing',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 details={};shape_expect={'generated':{0:16,32:80,64:48,128:32},
                          'reference_forced':{0:16,32:16,64:16,128:16}}
 for arm in ('generated','reference_forced'):
  base=a.routing/arm;roster={r['root']:r for r in map(json.loads,(base/'roster.jsonl').read_text().splitlines())}
  if len(roster)!=16 or len({r['family'] for r in roster.values()})!=16:raise ValueError('bad roster')
  if len({r['root'] for r in roster.values() if r['split']=='train'})!=12:raise ValueError('wrong split')
  details[arm]={}
  for h,n in shape_expect[arm].items():
   path=base/f'demand_H{h}.npz'
   with np.load(path,allow_pickle=False) as d:
    x,y,c=d['x'],d['y'],d['visit_count'];root=d['root'];start,end=d['start'],d['end']
    if x.shape!=(n,24,256) or y.shape!=(n,24,128) or c.shape!=(n,24):raise ValueError('wrong shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite demand')
    if not np.allclose(y.sum(-1),1,atol=1e-5):raise ValueError('unnormalized demand')
    expected=(end-start)[:,None]*np.array([3 if 6<=l<18 else 1 for l in range(24)])[None,:]
    if not np.array_equal(c,expected):raise ValueError('visit count mismatch')
    for r in roster:
     order=sorted(np.flatnonzero(root==r),key=lambda i:int(start[i]))
     if not order or int(start[order[0]])!=roster[r]['prompt_len']:raise ValueError('prompt boundary')
     first=order[0];prompt=x[first,:,:128]
     history=prompt*(roster[r]['prompt_len']*np.array([3 if 6<=l<18 else 1 for l in range(24)])[:,None])
     cursor=roster[r]['prompt_len']
     for i in order:
      if int(start[i])!=cursor or int(end[i])>len(roster[r]['token_ids']):raise ValueError('window gap')
      if not np.allclose(x[i,:,:128],prompt,atol=1e-5):raise ValueError('prompt features changed')
      if not np.allclose(x[i,:,128:],norm(history),atol=1e-5):raise ValueError('future leaked into features')
      history=history+y[i]*c[i,:,None];cursor=int(end[i])
     if cursor!=len(roster[r]['token_ids']):raise ValueError('uncovered continuation')
    details[arm][str(h)]={'n_samples':n,'x_shape':list(x.shape),'y_shape':list(y.shape),
                           'sha256':sha(path),'features_prefix_reconstructed':True}
 a.out.write_text(json.dumps({'status':'verified','arms':details},indent=2)+'\n')
 print(json.dumps({'status':'verified','sample_counts':{arm:{h:x['n_samples'] for h,x in v.items()} for arm,v in details.items()}}))
if __name__=='__main__':main()
