"""LOGRANK-BPD-07: compositional memory retrieval; development only."""
import argparse, itertools, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOTS={16:[1701,1709,1721],32:[1801,1811,1823]}

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bank(k,d):
    if not d:
        yield ('const',(),(1,)); yield ('const',(),(-1,)); return
    for ix in itertools.combinations(range(k),d):
        for s in (1,-1): yield ('parity',ix,(s,)+(1,)*(d-1))
        if d>1:
            for signs in itertools.product((1,-1),repeat=d): yield ('and',ix,signs)

def evaluate(x,p):
    op,ix,signs=p
    if op=='const': return np.full(len(x),signs[0],dtype=np.int8)
    lit=x[:,ix]*np.asarray(signs,dtype=np.int8)
    return np.prod(lit,axis=1,dtype=np.int8) if op=='parity' else np.min(lit,axis=1)

def fit(x,y,k):
    best=np.full((3,k),-len(x)-1,dtype=int); ties=np.zeros_like(best); chosen=[[None]*k for _ in range(3)]; counts=[0]*3; yf=y.astype(np.float32)
    for d in range(4):
        pool=list(bank(k,d))
        for cap in range(max(1,d),4): counts[cap-1]+=len(pool)
        for start in range(0,len(pool),1024):
            chunk=pool[start:start+1024]; feat=np.stack([evaluate(x,p) for p in chunk],1).astype(np.float32); scores=(feat.T@yf).astype(int)
            for cap in range(max(1,d),4):
                for j in range(k):
                    vals=scores[:,j]; m=int(vals.max()); n=int((vals==m).sum())
                    if m>best[cap-1,j]: best[cap-1,j]=m; ties[cap-1,j]=n; chosen[cap-1][j]=chunk[int(vals.argmax())]
                    elif m==best[cap-1,j]: ties[cap-1,j]+=n
    return chosen,best,ties,counts

def recurrent(x,programs,codes):
    by_code=dict(zip(codes,programs)); out=np.empty((len(x),len(programs)),dtype=np.int8); reads=[]
    for j,code in enumerate(codes):
        op,ix,signs=by_code[code]; a=np.ones(len(x),dtype=np.int8); reads.append(len(ix))
        if op=='const': a.fill(signs[0])
        for i,s in zip(ix,signs):
            # One selected rank-one coordinate component per pass; one-bit accumulator.
            v=x[:,i]*s; a=a*v if op=='parity' else np.minimum(a,v)
        out[:,j]=a
    assert np.array_equal(out,np.stack([evaluate(x,p) for p in programs],1))
    return out,reads

def static(x,y,z,k,d,seed,steps):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed+30000); xx=torch.from_numpy(x).float(); yy=(torch.from_numpy(y).float()+1)/2; zz=torch.from_numpy(z).float()
    v=torch.nn.Parameter(torch.randn(k,d)/k**.5); u=torch.nn.Parameter(torch.randn(d,k)/d**.5); b=torch.nn.Parameter(torch.zeros(k)); opt=torch.optim.Adam([v,u,b],lr=.03)
    for _ in range(steps):
        i=torch.randint(len(x),(128,),generator=g); loss=F.binary_cross_entropy_with_logits(xx[i]@v@u+b,yy[i]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.where(zz@v@u+b>=0,1,-1).numpy().astype(np.int8),(v@u).numpy().tolist(),b.numpy().tolist()

def unpack(a,k): return 2*((np.array(a,dtype=np.uint64)[:,None]>>np.arange(k,dtype=np.uint64))&1).astype(np.int8)-1

def run(steps):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); results=[]
    for k,roots in ROOTS.items():
        for root in roots:
            start=time.monotonic(); rng=np.random.default_rng(root); codes=rng.permutation(k).tolist(); classes=[('parity',1),('parity',2),('parity',3),('and',2),('and',3)]; truth=[]; used=set()
            for j in range(k):
                op,d=classes[j%len(classes)]
                while True:
                    ix=tuple(sorted(rng.choice(k,d,replace=False).tolist())); signs=tuple(rng.choice([-1,1],d).tolist())
                    if op=='parity': signs=(int(np.prod(signs)),)+(1,)*(d-1)
                    p=(op,ix,signs)
                    if p not in used: used.add(p); truth.append(p); break
            rng.shuffle(truth); ids=list(dict.fromkeys(rng.integers(0,1<<k,size=2048,dtype=np.uint64).tolist()))[:1280]; assert len(ids)==1280
            train_ids,test_ids=ids[:256],ids[256:]; assert not(set(train_ids)&set(test_ids)); x,z=unpack(train_ids,k),unpack(test_ids,k); y=np.stack([evaluate(x,p) for p in truth],1); t=np.stack([evaluate(z,p) for p in truth],1); chosen,best,ties,counts=fit(x,y,k); arms=[]
            for cap,programs in enumerate(chosen,1):
                pred,reads=recurrent(z,programs,codes); cert=[p==g for p,g in zip(programs,truth)]; correct=pred==t
                arms.append(dict(arm=f'depth{cap}',correct=int(correct.sum()),total=correct.size,exact_rows=int(correct.all(1).sum()),certified_queries=sum(cert),certificates=cert,programs=programs,train_correct=((best[cap-1]+len(x))//2).tolist(),ties=ties[cap-1].tolist(),candidate_count=counts[cap-1],reads=reads,predictions=''.join('1' if a else '0' for a in (pred>0).flat)))
            for d in (k//2,k):
                pred,w,b=static(x,y,z,k,d,root,steps); ok=pred==t; arms.append(dict(arm='rank_half' if d<k else 'rank_full',correct=int(ok.sum()),total=ok.size,exact_rows=int(ok.all(1).sum()),weight=w,bias=b,predictions=''.join('1' if a else '0' for a in (pred>0).flat)))
            r=dict(k=k,root=root,bit_cap=k.bit_length(),query_codes=codes,true_programs=truth,train_ids=train_ids,heldout_ids=test_ids,labels=''.join('1' if a else '0' for a in (t>0).flat),arms=arms,seconds=time.monotonic()-start); results.append(r)
            print(k,root,[(a['arm'],a['correct'],a.get('certified_queries')) for a in arms],round(r['seconds'],2),flush=True)
    return dict(evidence='fresh development; fixed randomized address; dictionary fit; native BPD not executed',steps=steps,roots=ROOTS,source_sha256=digest(__file__),results=results)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=350); p.add_argument('--out',default='receipt.json'); a=p.parse_args(); Path(a.out).write_text(json.dumps(run(a.steps),indent=2)+'\n')
