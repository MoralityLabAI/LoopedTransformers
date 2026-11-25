"""Train-only supervised Bellman approximation; disjoint synthetic DNF structures."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from lbf20d_dp import Problem, independent, minimal, bits, synthetic_audit

N=8
WIDTH=64
SEEDS=(107,109,113)
ARMS=("direct1","recur3","untied3")
STEPS=400
BATCH=128
DIM=N*N+5*N+1

def signature(n,masks):
    # Necessary isomorphism invariants: rejecting every repeated signature is
    # conservative and guarantees no isomorphic mask family across cohorts.
    return (n,tuple(sorted(m.bit_count() for m in masks)),
            tuple(sorted(sum(bool(m&(1<<g)) for m in masks) for g in range(n))),
            tuple(sorted((a&b).bit_count() for i,a in enumerate(masks) for b in masks[i+1:])))

def tasks():
    rng=random.Random(2026112509)
    used=set()
    result={}
    for name,n,count in (("train",6,96),("dev6",6,32),("stress8",8,12)):
        xs=[]
        attempts=0
        while len(xs)<count:
            attempts+=1
            if attempts>100000: raise RuntimeError("generator exhausted")
            masks=minimal(sum(1<<g for g in rng.sample(range(n),rng.randint(2,min(5,n))))
                          for _ in range(rng.randint(4,8)))
            if len(masks)<2 or (sum(1<<g for g in range(n) if any(m&(1<<g) for m in masks)) != (1<<n)-1): continue
            sig=signature(n,masks)
            if sig in used: continue
            used.add(sig)
            a=[rng.choice((.12,.22,.35,.65,.78,.88)) for _ in range(n)]
            b=[1-p if rng.random()<.8 else p for p in a]
            w=rng.choice((.25,.4,.6,.75))
            xs.append(Problem(masks,[a,b],[w,1-w]))
        result[name]=xs
    return result

def feature(p,s,f):
    x=[float(bool(m&(1<<g))) for m in sorted(p.masks) for g in range(N)]
    x += [0.]*(N*N-len(x))
    for c in p.components:
        x.extend(c)
        x.extend([0.]*(N-p.n))
    x.append(p.weights[0])
    x.extend(float(bool(s&(1<<g))) for g in range(N))
    x.extend(float(bool(f&(1<<g))) for g in range(N))
    x.extend(float(g<p.n) for g in range(N))
    assert len(x)==DIM
    return x

def dataset(p,sample=None,rng=None):
    p.solve()
    keys=sorted(p.q)
    if sample is not None and len(keys)>sample:
        keys=sorted(rng.sample(keys,sample-1)+[(0,0)]) if (0,0) not in keys else sorted(set(rng.sample(keys,sample-1)+[(0,0)]))
    x=[]; y=[]; valid=[]
    for s,f in keys:
        qs=p.q[s,f]; v=min(qs.values())
        x.append(feature(p,s,f))
        y.append([qs[g]-v if g in qs else 0. for g in range(N)])
        valid.append([g in qs for g in range(N)])
    return keys,torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),torch.tensor(valid,dtype=torch.bool)

class Net(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.entry=nn.Linear(DIM,WIDTH)
        self.blocks=nn.ModuleList(nn.Linear(WIDTH,WIDTH) for _ in range(3 if arm=="untied3" else 1))
        self.head=nn.Linear(WIDTH,N)
    def forward(self,x):
        h=torch.tanh(self.entry(x))
        for k in range(1 if self.arm=="direct1" else 3):
            h=torch.tanh(h+self.blocks[k if self.arm=="untied3" else 0](h))
        return self.head(h)

def analytic(p):
    opt,_=p.solve()
    ind=independent(p)
    return {"joint_opt":opt,"independent_dp":p.expected(lambda s,f:ind.solve(s,f)[1]),
            "posterior_risk":p.expected(lambda s,f:p.choose_risk(s,f)),
            "marginal_risk":p.expected(lambda s,f:p.choose_risk(s,f,prior=ind))}

@torch.no_grad()
def evaluate(model,items):
    rows=[]
    for i,(p,(keys,x,y,m),base) in enumerate(items):
        pred=model(x).masked_fill(~m,1e9)
        act=pred.argmin(-1)
        chosen=y.gather(1,act[:,None]).squeeze(1)
        policy=dict(zip(keys,act.tolist()))
        value=p.expected(lambda s,f:policy[s,f])
        regret=value-base["joint_opt"]
        assert regret>=-1e-7
        rows.append({"root":i,"expected_queries":value,"regret":regret,
                     "state_optimal_action_rate":float((chosen<1e-6).float().mean()),
                     "mean_local_regret":float(chosen.mean()),"states":len(keys),**base})
    return rows

def main(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    audit=synthetic_audit()
    print("DP_AUDIT",json.dumps(audit),flush=True)
    cohorts=tasks()
    rng=random.Random(2026112510)
    d=[dataset(p,128,rng) for p in cohorts["train"]]
    x=torch.cat([a[1] for a in d]); y=torch.cat([a[2] for a in d]); m=torch.cat([a[3] for a in d])
    held={k:[(p,dataset(p),analytic(p)) for p in cohorts[k]] for k in ("dev6","stress8")}
    sources={k:[{"masks":list(p.masks),"components":p.components,"weights":p.weights}
                 for p in ps] for k,ps in cohorts.items()}
    srcbytes=(json.dumps(sources,sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(srcbytes)
    fits=[]
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model=Net(arm)
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            gen=torch.Generator().manual_seed(seed+7919)
            start=time.perf_counter()
            for step in range(STEPS):
                idx=torch.randint(len(x),(BATCH,),generator=gen)
                pred=model(x[idx]); valid=m[idx]
                loss=((pred-y[idx]).square()*valid).sum()/valid.sum()
                opt.zero_grad(); loss.backward(); opt.step()
            seconds=time.perf_counter()-start
            ckpt=out/f"{arm}_{seed}.pt"
            torch.save(model.state_dict(),ckpt)
            rec={"architecture":arm,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
                 "passes":1 if arm=="direct1" else 3,"steps":STEPS,"batch_size":BATCH,
                 "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                 "fit_seconds":seconds,"final_batch_mse":float(loss.detach()),"cohorts":{}}
            for k,items in held.items():
                rows=evaluate(model,items)
                rec["cohorts"][k]={"roots":len(rows),"mean_expected_queries":statistics.mean(r["expected_queries"] for r in rows),
                   "mean_regret":statistics.mean(r["regret"] for r in rows),
                   "mean_optimal_action_rate":statistics.mean(r["state_optimal_action_rate"] for r in rows),
                   "mean_local_regret":statistics.mean(r["mean_local_regret"] for r in rows),"rows":rows}
            fits.append(rec)
            print("FIT",arm,seed,rec["parameters"],{k:round(v["mean_regret"],5) for k,v in rec["cohorts"].items()},flush=True)
    agg={}
    for k,items in held.items():
        agg[k]={"roots":len(items),"analytic":{a:statistics.mean(b[a] for _,_,b in items) for a in items[0][2]},
                "learned":{a:{"mean_expected_queries":statistics.mean(r["cohorts"][k]["mean_expected_queries"] for r in fits if r["architecture"]==a),
                             "mean_regret":statistics.mean(r["cohorts"][k]["mean_regret"] for r in fits if r["architecture"]==a),
                             "seed_regret":[r["cohorts"][k]["mean_regret"] for r in fits if r["architecture"]==a]}
                           for a in ARMS}}
    summary={"schema":"lbf20d.correlated-policy.v1","scope":"development_only","train_tasks":96,"train_states":len(x),
             "task_sha256":hashlib.sha256(srcbytes).hexdigest(),"audit":audit,"cohorts":agg,"fits":fits}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("NEURAL_RESULTS",json.dumps({k:v for k,v in summary.items() if k!="fits"}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
