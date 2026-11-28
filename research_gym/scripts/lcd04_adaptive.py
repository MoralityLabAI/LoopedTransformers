"""LCD-04: prefix-only adaptive depth, with exact depth-multiset random control."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import math
import torch
from research_gym.scripts import lcd01_huginn as base
from research_gym.scripts import lcd02_spectral as spectral
from research_gym.scripts import lcd03_anderson as aa

SCHEMA="lcd04-v1"
CAL_SEED,TEST_SEED=2026112509,2026112510
SHORT,LONG,BUDGET=8,24,16
DEPTHS=(8,16,24,32)
FEATURES=("relative_residual","growth","alignment")
QUANTILES=(.25,.375,.5,.625,.75)

def feature_vector(h6,h7,h8):
    a=(h7.float()-h6.float()).flatten()
    b=(h8.float()-h7.float()).flatten()
    na=a.norm()
    nb=b.norm()
    eps=1e-20
    vals=torch.stack((nb/h8.float().norm().clamp_min(eps),
                      nb/na.clamp_min(eps),
                      (a@b)/(na*nb).clamp_min(eps))).tolist()
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("nonfinite prefix feature")
    return dict(zip(FEATURES,vals))

def route(features,cfg):
    if cfg["feature"]=="always_short":
        return SHORT
    v=features[cfg["feature"]]
    long=v>cfg["threshold"] if cfg["direction"]==1 else v<cfg["threshold"]
    return LONG if long else SHORT

def pool(rows):
    import numpy as np
    cfgs=[dict(feature="always_short")]
    for k in FEATURES:
        for q in QUANTILES:
            t=float(np.quantile([r["features"][k] for r in rows],q))
            for sign in (-1,1):
                cfgs.append(dict(feature=k,threshold=t,direction=sign))
    return list({base.key(c):c for c in cfgs}.values())

def policy_rows(rows,cfg):
    out=[]
    for r in rows:
        depth=route(r["features"],cfg)
        out.append(dict(r["scores"][str(depth)],id=r["id"],family=r["family"],
                        depth=depth,tokens=r["tokens"]))
    return out

def select(rows):
    trials=[]
    for cfg in pool(rows):
        rr=policy_rows(rows,cfg)
        loops=sum(r["depth"] for r in rr)/len(rr)
        token_work=sum(r["depth"]*r["tokens"] for r in rr)/sum(r["tokens"] for r in rr)
        s=base.summarize(rr)
        trials.append(dict(config=cfg,mean_loops=loops,token_weighted_loops=token_work,summary=s))
    eligible=[r for r in trials if r["mean_loops"]<=BUDGET and r["token_weighted_loops"]<=BUDGET]
    chosen=min(eligible,key=lambda r:(-r["summary"]["accuracy"],r["summary"]["nll"],
                                     r["token_weighted_loops"],base.key(r["config"])))
    return chosen["config"],trials

def random_schedule(rows,seed=314159):
    # Exactly match adaptive depth counts within family and 64-token length bucket.
    # This comparator uses aggregate routes, never correctness or logits.
    groups={}
    for r in rows:
        groups.setdefault((r["family"],r["tokens"]//64),[]).append(r)
    out={}
    for k,rr in sorted(groups.items()):
        rr=sorted(rr,key=lambda r:r["id"])
        ds=[r["depth"] for r in rr]
        random.Random(seed+1009*k[0]+k[1]).shuffle(ds)
        out.update({r["id"]:d for r,d in zip(rr,ds)})
    return out

def smoke():
    a=torch.tensor([1.,2.])
    f=feature_vector(a,a+1,a+2)
    assert abs(f["alignment"]-1.)<1e-6
    assert abs(f["growth"]-1.)<1e-6
    f=feature_vector(a,a+1,a)
    assert abs(f["alignment"]+1.)<1e-6
    assert route(f,dict(feature="alignment",threshold=0.,direction=-1))==LONG
    rows=[]
    for fam in (1,2,3):
        for i in range(8):
            rows.append(dict(id=f"{fam}-{i}",family=fam,tokens=100,
                features=dict(relative_residual=float(i),growth=float(i),alignment=float(i)),
                scores={"8":dict(correct=i<4,nll=0. if i<4 else 2.),
                        "24":dict(correct=True,nll=0.)}))
    p,_=select(rows)
    rr=policy_rows(rows,p)
    assert sum(r["depth"] for r in rr)==16*len(rr)
    assert all(r["correct"] for r in rr)
    s=random_schedule(rr)
    for fam in (1,2,3):
        assert sorted(r["depth"] for r in rr if r["family"]==fam)==sorted(
            s[r["id"]] for r in rr if r["family"]==fam)
    assert s==random_schedule(rr)
    print("LCD-04 prefix routing and exact depth matching smoke PASS")

@torch.inference_mode()
def admission(model,tok,cal):
    receipts=[]
    for it in cal[::16]:
        ids=spectral.input_ids(tok,it)
        seed=spectral.seed(it)
        h=base.trajectory(model,ids,seed,max(DEPTHS),DEPTHS)
        for d in DEPTHS:
            p=base.trajectory(model,ids,seed,d,())[d]
            z=base.read(model,p)
            torch.manual_seed(seed)
            native=model(input_ids=ids,num_steps=d,use_cache=False).logits[:,-1].float()
            prefix=aa.errors(p,h[d])
            logits=aa.errors(z,native)
            top=bool((z.argmax(-1)==native.argmax(-1)).all())
            receipts.append(dict(id=it["id"],depth=d,prefix=prefix,logits=logits,
                                 top1_equal=top,passed=top and aa.within(prefix) and aa.within(logits)))
    return all(r["passed"] for r in receipts),receipts

@torch.inference_mode()
def calibrate(model,tok,labels,cand,roots):
    rr=[]
    for i,it in enumerate(roots):
        ids=spectral.input_ids(tok,it)
        ss=base.trajectory(model,ids,spectral.seed(it),LONG,(6,7,8))
        feats=feature_vector(ss[6],ss[7],ss[8])
        target=labels.index(it["answer"])
        scores={str(d):base.row(base.read(model,ss[d]),cand,target) for d in (SHORT,LONG)}
        rr.append(dict(id=it["id"],family=it["family"],tokens=ids.shape[-1],features=feats,scores=scores))
        if (i+1)%8==0:
            print("LCD-04 calibration",i+1,len(roots),flush=True)
    return rr

@torch.inference_mode()
def forward(model,ids,it,depth=None,policy=None):
    if (depth is None)==(policy is None):
        raise ValueError("specify exactly one route")
    e,h,idx=aa.initialize(model,ids,spectral.seed(it))
    history=[]
    feats=None
    stop=LONG if policy is not None else depth
    for t in range(1,stop+1):
        h,idx=aa.step(model,e,h,idx,t-1)
        if policy is not None and 6<=t<=8:
            history.append(h.clone())
        if policy is not None and t==SHORT:
            feats=feature_vector(*history)
            history.clear()
            stop=route(feats,policy)
        if t==stop:
            break
    return base.read(model,h),t,feats

@torch.inference_mode()
def measure(model,tok,labels,cand,it,depth=None,policy=None):
    ids=spectral.input_ids(tok,it)
    base.sync()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    z,d,feat=forward(model,ids,it,depth,policy)
    base.sync()
    elapsed=(time.perf_counter()-start)*1000
    peak=torch.cuda.max_memory_allocated()
    r=base.row(z,cand,labels.index(it["answer"]))
    r.update(id=it["id"],family=it["family"],tokens=ids.shape[-1],depth=d,
             latency_ms=elapsed,peak_bytes=peak,coda_passes=1,features=feat)
    return r

def cost(rows):
    return dict(mean_loops=sum(r["depth"] for r in rows)/len(rows),
                token_weighted_loops=sum(r["depth"]*r["tokens"] for r in rows)/sum(r["tokens"] for r in rows),
                total_forward_ms=sum(r["latency_ms"] for r in rows),
                peak_bytes=max(r["peak_bytes"] for r in rows))

def main(a):
    model,tok,labels,cand,meta=spectral.load()
    meta.update(schema=SCHEMA,code_sha=hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
                for m in (base,spectral,aa))+Path(__file__).read_bytes()).hexdigest(),
                cal_seed=CAL_SEED,budget=BUDGET,depths=list(DEPTHS))
    cal=base.items(CAL_SEED,16)
    meta["cal_sha"]=base.digest(cal)
    ok,receipt=admission(model,tok,cal)
    if not ok:
        base.write(a.out,dict(meta=meta,status="BLOCKED_PARITY",admission=receipt))
        raise RuntimeError("native parity failed")
    if a.stage=="calibrate":
        start=time.perf_counter()
        rows=calibrate(model,tok,labels,cand,cal)
        cfg,trials=select(rows)
        base.write(a.out,dict(meta=meta,status="FROZEN",admission=receipt,policy=cfg,
                   freeze_sha=base.digest(cfg),calibration_elapsed_s=time.perf_counter()-start,
                   rows=rows,trials=trials))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or f["freeze_sha"]!=base.digest(f["policy"]):
        raise ValueError("freeze mismatch")
    cfg,_=select(f["rows"])
    if cfg!=f["policy"] or {r["id"] for r in f["rows"]}!={r["id"] for r in cal}:
        raise ValueError("selection mismatch")
    test=base.items(TEST_SEED,64)
    if len({r["id"] for r in test})!=len(test) or {r["id"] for r in test}&{r["id"] for r in cal}:
        raise ValueError("non-disjoint roots")
    meta["test_sha"]=base.digest(test)
    rows={"adaptive":[],"random":[]}
    rows.update({f"fixed{d}":[] for d in DEPTHS})
    # First pass computes routes without exposing labels or future states to the policy.
    # This pass is untimed for comparisons; rerun adaptive with frozen route parity below.
    planned=[]
    for it in test:
        ids=spectral.input_ids(tok,it)
        ss=base.trajectory(model,ids,spectral.seed(it),SHORT,(6,7,8))
        fts=feature_vector(ss[6],ss[7],ss[8])
        planned.append(dict(id=it["id"],family=it["family"],tokens=ids.shape[-1],depth=route(fts,cfg)))
    assigned=random_schedule(planned)
    plan={r["id"]:r["depth"] for r in planned}
    # Every timed arm gets a new same-seed forward. Random allocation has no answer access.
    for i,it in enumerate(test):
        names=list(rows)
        random.Random(spectral.seed(it)).shuffle(names)
        for name in names:
            if name=="adaptive":
                r=measure(model,tok,labels,cand,it,policy=cfg)
                if r["depth"]!=plan[it["id"]]:
                    raise RuntimeError("adaptive route replay mismatch")
            else:
                d=assigned[it["id"]] if name=="random" else int(name[5:])
                r=measure(model,tok,labels,cand,it,depth=d)
            rows[name].append(r)
        if (i+1)%8==0:
            print("LCD-04 test",i+1,len(test),flush=True)
            base.write(a.out,dict(meta=meta,status="IN_PROGRESS",policy=cfg,admission=receipt,rows=rows))
    s={k:base.summarize(v) for k,v in rows.items()}
    costs={k:cost(v) for k,v in rows.items()}
    comparisons={k:dict(paired=base.paired(rows["adaptive"],v),
                        latency_ratio=costs["adaptive"]["total_forward_ms"]/costs[k]["total_forward_ms"])
                 for k,v in rows.items() if k!="adaptive"}
    base.write(a.out,dict(meta=meta,status="PILOT_COMPLETE",admission=receipt,policy=cfg,
                freeze_sha=base.digest(cfg),routing_audit_extra_loops=len(test)*SHORT,
                summaries=s,costs=costs,comparisons=comparisons,rows=rows,
                test_budget_pass=costs["adaptive"]["mean_loops"]<=BUDGET and
                                 costs["adaptive"]["token_weighted_loops"]<=BUDGET))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","calibrate","test"),required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.out is None or a.out.exists() or (a.stage=="test" and a.freeze is None):
            p.error("fresh output and test freeze required")
        main(a)
