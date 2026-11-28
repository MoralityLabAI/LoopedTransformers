"""Shared-trace multi-frame BP and bounded likelihood repair."""
import json,time,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
from base13 import Frame,world,dot,run,bp_moment
from bp import bp
H=Path(__file__).resolve().parent
RS=(8,21,45); NOISE=(.15,.35); CAL=range(300,304); DEV=range(310,318); TEMPS=(1.,2.,4.,8.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def parity(x):return np.bitwise_count(np.asarray(x,dtype=np.uint64))&1

def frames(w,rows,obs):
    pool=[w['pool'][j] for j in sorted(range(48),key=lambda j:(w['costs'][j],j))]
    unique=list(dict.fromkeys(rows));votes=Counter()
    for a,b in zip(rows,obs):votes[a]+=1-2*b
    rng=np.random.default_rng(w['seed']+14000)
    orders=[unique,sorted(unique,key=lambda a:-abs(votes[a]))]
    for _ in range(6):orders.append([unique[j] for j in rng.permutation(len(unique))])
    fs=[Frame(a+pool) for a in orders];R=len(Frame(rows).rows)
    assert all(len(f.rows)==w['d'] and all(f.coord(a)<1<<R for a in rows) for f in fs)
    return fs,R

def moments(f,mu,qs,R):
    return np.array([0. if f.coord(q)>=1<<R else math.prod(mu[i] for i in range(R) if f.coord(q)>>i&1) for q in qs])

def candidates(f,canon,mu,R,u,random=False,seed=0):
    base=sum(int(mu[i]<0)<<i for i in range(R));u=min(u,R)
    ix=np.random.default_rng(seed).choice(R,u,replace=False) if random else np.argsort(np.abs(mu[:R]),kind='stable')[:u]
    n=1<<u;local=np.full(n,base,dtype=np.uint64);z=np.arange(n,dtype=np.uint64)
    for j,i in enumerate(ix):local^=((z>>np.uint64(j))&np.uint64(1))<<np.uint64(i)
    result=np.zeros(n,dtype=np.uint64)
    for j,q in enumerate(canon.rows[:R]):result|=parity(local&np.uint64(f.coord(q))).astype(np.uint64)<<np.uint64(j)
    return result

def likelihood(states,canon,rows,obs,qs,p,R):
    cnt=Counter(zip(rows,obs));masks=np.array([canon.coord(q) for q,b in cnt],dtype=np.uint64);yy=np.array([b for q,b in cnt],dtype=np.uint8);mult=np.array(list(cnt.values()),dtype=np.int64)
    err=np.empty(len(states),dtype=np.int64);peak=0
    for j in range(0,len(states),256):
        x=states[j:j+256,None]&masks;bad=parity(x)!=yy;err[j:j+256]=bad@mult;peak=max(peak,x.nbytes+bad.nbytes)
    logw=(err-err.min())*math.log(p/(1-p));weights=np.exp(logw);weights/=weights.sum()
    out=[0. if canon.coord(q)>=1<<R else float(1-2*(parity(states&np.uint64(canon.coord(q)))@weights)) for q in qs]
    return np.array(out),dict(states=len(states),min_errors=int(err.min()),ess=float(1/(weights@weights)),scratch_lower_bytes=int(peak+err.nbytes+weights.nbytes+states.nbytes))

def metrics(prob,w,gate=None):
    wt=np.array(w['weights']);y=np.array([dot(q,w['hidden']) for q in w['targets']]);ok=(prob>.5+1e-12)==y;take=np.maximum(prob,1-prob)>=.9-1e-12
    if gate is not None:take&=gate
    return dict(brier=float(wt@(prob-y)**2),accuracy=float(wt@ok),coverage=float(wt@take),covered_correct=float(wt@(take&ok)))

def decode(w,rows,obs,p):
    qs=w['targets'];rec={};start=time.perf_counter();m,it=bp_moment(w,rows,obs,p,'induced');prob=(1-m(qs))/2
    def put(name,mu,sec,**meta):
        pr=np.clip((1-mu)/2,0,1);rec[name]=dict(prob=pr.tolist(),seconds=sec,**metrics(pr,w),**meta)
    put('fixed_induced',1-2*prob,time.perf_counter()-start,iterations=it)
    start=time.perf_counter();fs,R=frames(w,rows,obs);canon=fs[0];ms=[];mus=[];times=[];its=[]
    for f in fs:
        a=time.perf_counter();mu,it=bp([f.coord(q) for q in rows],obs,p,w['d']);mus.append(mu);ms.append(moments(f,mu,qs,R));its.append(it);times.append(time.perf_counter()-a)
    setup=time.perf_counter()-start-sum(times);ms=np.array(ms)
    for n in (1,4,8):put('observed' if n==1 else f'ensemble{n}',ms[:n].mean(0),setup+sum(times[:n]),iterations=sum(its[:n]),frame_count=n,frame_mask_payload_bytes=n*w['d']*8)
    for n,u,rand,name in [(1,8,False,'repair1'),(4,6,False,'repair4'),(8,8,False,'repair8'),(8,8,True,'random_repair8')]:
        a=time.perf_counter();raw=[candidates(f,canon,mu,R,u,rand,w['seed']+i+17000) for i,(f,mu) in enumerate(zip(fs[:n],mus[:n]))];states=np.unique(np.concatenate(raw));moment,meta=likelihood(states,canon,rows,obs,qs,p,R);put(name,moment,setup+sum(times[:n])+time.perf_counter()-a,**meta,proposals=sum(len(x) for x in raw),frame_count=n,free_bits=u,frame_mask_payload_bytes=n*w['d']*8,state_ids=states.tolist())
    if w['d']==11:
        a=time.perf_counter();mu,meta=likelihood(np.arange(1<<R,dtype=np.uint64),canon,rows,obs,qs,p,R);put('exact',mu,time.perf_counter()-a,**meta)
    return rec,ms.tolist(),R,[f.rows for f in fs]

def temper(pr,t):
    pr=np.clip(pr,1e-12,1-1e-12);z=np.log(pr/(1-pr))/t;return 1/(1+np.exp(-z))
def main():
    out=[]
    def write(temps=None):(H/'receipt.json').write_text(json.dumps(dict(evidence='development; fixed shared trace',source_sha256=sha(__file__),base_sha256=sha(H/'base13.py'),bp_sha256=sha(H/'bp.py'),protocol_sha256=sha(H/'PROTOCOL.md'),temperatures=temps,worlds=out),separators=(',',':'))+'\n')
    for split,ids in [('cal',CAL),('dev',DEV)]:
        for r in RS:
            for i in ids:
                w=world(r,i);w['split']=split;w['runs']=[]
                for p in NOISE:
                    trace,_=run(w,p,'induced_fixed');rows=w['initial']+[t['query'] for t in trace['trace']];obs=trace['initial_obs']+[t['obs'] for t in trace['trace']];res,ms,R,fs=decode(w,rows,obs,p);w['runs'].append(dict(noise=p,trace=trace,decoders=res,frame_moments=ms,rank=R,frames=fs))
                out.append(w);print(split,r,w['seed'],[(x['noise'],round(x['decoders']['repair8']['brier'],4)) for x in w['runs']],flush=True);write()
    temps={}
    for p in NOISE:
        cal=[(w,x) for w in out if w['split']=='cal' for x in w['runs'] if x['noise']==p];loss=[float(np.mean([metrics(temper(np.array(x['decoders']['repair8']['prob']),t),w)['brier'] for w,x in cal])) for t in TEMPS];temps[str(p)]=dict(temperature=TEMPS[int(np.argmin(loss))],cal_losses=loss)
    for w in out:
        for x in w['runs']:
            pr=temper(np.array(x['decoders']['repair8']['prob']),temps[str(x['noise'])]['temperature']);ms=np.array(x['frame_moments']);gate=(np.all(ms>=.4-1e-12,axis=0)&(pr<.5))|(np.all(ms<=-.4+1e-12,axis=0)&(pr>.5));sec=x['decoders']['repair8']['seconds'];x['decoders']['repair_calibrated']=dict(prob=pr.tolist(),**metrics(pr,w),seconds=sec);x['decoders']['repair_guard']=dict(prob=pr.tolist(),**metrics(pr,w,gate),seconds=sec)
    write(temps)
if __name__=='__main__':main()
