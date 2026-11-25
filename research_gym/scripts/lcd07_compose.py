"""LCD-07: prospective factorial composition of individually screened methods.

Prior held-out receipts screen eligibility; new roots test composition.
No retuning, no combining failed component pilots.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
import torch
from research_gym.scripts import lcd01_huginn as base
from research_gym.scripts import lcd02_spectral as spectral
from research_gym.scripts import lcd03_anderson as aa
from research_gym.scripts import lcd04_adaptive as adaptive
from research_gym.scripts import lcd05_precision as precision

SCHEMA="lcd07-v1"
TEST_SEED=2026112519
DEPTH=16

def read_pair(root,folder,freeze_name,schema):
    fp=root/folder/freeze_name
    tp=root/folder/"test.json"
    f=json.loads(fp.read_text(encoding="utf-8"))
    t=json.loads(tp.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or t["status"]!="PILOT_COMPLETE":
        raise ValueError(folder+" incomplete")
    if f["meta"]["schema"]!=schema or t["meta"]["schema"]!=schema:
        raise ValueError(folder+" schema mismatch")
    if any(t["meta"].get(k)!=v for k,v in f["meta"].items()):
        raise ValueError(folder+" environment/source mismatch")
    key="frozen" if "frozen" in f else "policy" if "policy" in f else "arms"
    if f["freeze_sha"]!=base.digest(f[key]) or t["freeze_sha"]!=f["freeze_sha"]:
        raise ValueError(folder+" freeze mismatch")
    return f,t,dict(freeze_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
                    test_sha256=hashlib.sha256(tp.read_bytes()).hexdigest())

def screens(s,a,p,d):
    sc=s["comparisons"]
    ac=a["comparisons"]["12"]
    pc=p["comparisons"]
    dc=d["comparisons"]
    gates={
        "spectral":dict(selected_scalar=sc["scalar"]["delta"]["delta"]>0,
                        norm_scalar=sc["matched_scalar"]["delta"]["delta"]>0,
                        norm_random=sc["matched_random"]["delta"]["delta"]>0,
                        latency=sc["loopcd"]["pca_latency_ratio"]<=1.10),
        "anderson":dict(base=ac["base16"]["paired"]["delta"]>0,
                        scalar=ac["scalar12"]["paired"]["delta"]>0,
                        norm_scalar=ac["matched_scalar12"]["paired"]["delta"]>0,
                        norm_random=ac["matched_random12"]["paired"]["delta"]>0,
                        latency=ac["base16"]["latency_ratio"]<=1.10),
        "precision":dict(recovery=pc["bf16"]["paired"]["delta"]>=-.02,
                        uniform=pc["q4"]["paired"]["delta"]>0,
                        random=pc["random_matched"]["paired"]["delta"]>0,
                        payload=p["costs"]["contrast"]["payload_ratio"]<=.40),
        "adaptive":dict(random=dc["random"]["paired"]["delta"]>0,
                        fixed=dc["fixed16"]["paired"]["delta"]>0,
                        budget=bool(d["test_budget_pass"]),
                        latency=dc["fixed16"]["latency_ratio"]<=1.)}
    return {k:dict(passed=all(v.values()),checks=v) for k,v in gates.items()}

def arms_for(gates):
    names=[k for k in ("spectral","anderson","precision") if gates[k]["passed"]]
    arms={"base16":dict(depth=16),"base32":dict(depth=32),"loopcd16":dict(depth=16,scalar=True)}
    for n in range(1,len(names)+1):
        for subset in itertools.combinations(names,n):
            arms["+".join(subset)]=dict(depth=16,**{k:True for k in subset})
    if gates["adaptive"]["passed"]:
        arms["adaptive"]=dict(adaptive=True)
    return names,arms

def smoke():
    gates={k:dict(passed=True) for k in ("spectral","anderson","precision","adaptive")}
    names,arms=arms_for(gates)
    assert len(names)==3 and len(arms)==11
    assert arms["spectral+anderson+precision"]==dict(depth=16,spectral=True,anderson=True,precision=True)
    gates["anderson"]["passed"]=False
    names,arms=arms_for(gates)
    assert len(names)==2 and len(arms)==7
    assert all(not c.get("anderson",False) for c in arms.values())
    print("LCD-07 eligibility and factorial enumeration smoke PASS")

@torch.inference_mode()
def forward(model,ids,it,arm,cfg,us,up):
    if arm.get("adaptive"):
        z,depth,features=adaptive.forward(model,ids,it,policy=cfg["adaptive"])
        return z,dict(loops=depth,codec_roundtrips=0,state_bytes=None,basis_bytes=0,
                      intervention_norms={},features=features)
    e,h,idx=aa.initialize(model,ids,spectral.seed(it))
    depth=arm["depth"]
    history=[]
    ref=None
    ncodec=0
    payload=None
    norms={}
    def roundtrip(x):
        nonlocal ncodec,payload
        packed=precision.pack(x,4,up)
        payload=precision.payload_bytes(packed)
        ncodec+=1
        return precision.unpack(packed,up,dtype=x.dtype)
    for t in range(1,depth+1):
        h,idx=aa.step(model,e,h,idx,t-1)
        if arm.get("precision"):
            h=roundtrip(h)
        if t==6 and (arm.get("spectral") or arm.get("scalar")):
            ref=h.clone()
        if arm.get("anderson"):
            c=cfg["anderson"]
            if c["at"]-c["window"]<=t<=c["at"]:
                history.append(h.clone())
            if t==c["at"]:
                g,rec=aa.bounded_correction(history,c["ridge"],c["dose"])
                norms["anderson"]=float(aa.norm(g))
                h=(h.float()+g).to(h.dtype)
                history.clear()
                del g
                if arm.get("precision"):
                    h=roundtrip(h)
        if t==depth and (arm.get("spectral") or arm.get("scalar")):
            d=h.float()-ref.float()
            if arm.get("spectral"):
                g=spectral.guide(d,us,torch.tensor(cfg["spectral"]["lam"],device=h.device))
            else:
                g=.5*d
            norms["terminal"]=float(aa.norm(g))
            h=(h.float()+g).to(h.dtype)
    z=base.read(model,h)
    return z,dict(loops=depth,codec_roundtrips=ncodec,state_bytes=payload if payload is not None else precision.bytes_of(h),
                  basis_bytes=sum(precision.bytes_of(u) for u in (us,up) if u is not None),
                  intervention_norms=norms,features=None)

def main(args):
    root=args.results
    pairs={
        "spectral":read_pair(root,"lcd02","fit.json",spectral.SCHEMA),
        "anderson":read_pair(root,"lcd03","cal.json",aa.SCHEMA),
        "adaptive":read_pair(root,"lcd04","cal.json",adaptive.SCHEMA),
        "precision":read_pair(root,"lcd05","fit.json",precision.SCHEMA)}
    gates=screens(pairs["spectral"][1],pairs["anderson"][1],pairs["precision"][1],pairs["adaptive"][1])
    components,arms=arms_for(gates)
    provenance={k:v[2] for k,v in pairs.items()}
    meta=dict(schema=SCHEMA,gates=gates,components=components,provenance=provenance,
              code_sha=hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
               for m in (base,spectral,aa,adaptive,precision))+Path(__file__).read_bytes()).hexdigest())
    if len(components)<2:
        base.write(args.out,dict(meta=meta,status="SKIPPED_INSUFFICIENT_COMPONENTS",arms=arms))
        print("Composition skipped: fewer than two methods cleared their independent screens")
        return
    model,tok,labels,cand,env=spectral.load()
    for f,t,_ in pairs.values():
        for k in ("model","revision","dtype","torch","transformers","device"):
            if f["meta"][k]!=env[k]:
                raise ValueError("prior native environment mismatch: "+k)
    meta["environment"]=env
    sf=pairs["spectral"][0]["frozen"]
    pf=pairs["precision"][0]["frozen"]
    cfg=dict(spectral=sf["arms"]["pca"],anderson=pairs["anderson"][0]["arms"]["aa12"],
             precision=pf["arms"]["contrast"],adaptive=pairs["adaptive"][0]["policy"])
    us0=torch.tensor(sf["bases"]["pca"],dtype=torch.float32)[:,:cfg["spectral"]["rank"]].contiguous()
    up0=torch.tensor(pf["bases"]["contrast"],dtype=torch.float32)[:,:cfg["precision"]["rank"]].contiguous()
    if any(not torch.isfinite(u).all() or not torch.allclose(u.T@u,torch.eye(u.shape[1]),atol=1e-4)
           for u in (us0,up0)):
        raise ValueError("invalid frozen basis")
    # Admission uses prior calibration roots only. No LCD-07 outcomes until frozen.
    ok,parity=aa.admit(model,tok,base.items(aa.CAL_SEED,16))
    if not ok:
        base.write(args.out,dict(meta=meta,status="BLOCKED_NATIVE_MAP",admission=parity))
        raise RuntimeError("native map admission failed")
    seal=dict(arms=arms,config=cfg,provenance=provenance)
    meta["freeze_sha"]=base.digest(seal)
    test=base.items(TEST_SEED,64)
    ids={r["id"] for r in test}
    if len(ids)!=len(test):
        raise ValueError("duplicate test roots")
    prior_ids=set()
    # Check all earlier synthetic splits, including LCD-01.
    for seed,n in ((base.CAL_SEED,16),(base.TEST_SEED,48),
                   (spectral.FIT_SEED,16),(spectral.SELECT_SEED,16),(spectral.TEST_SEED,64),
                   (aa.CAL_SEED,16),(aa.TEST_SEED,64),(adaptive.CAL_SEED,16),(adaptive.TEST_SEED,64),
                   (precision.FIT_SEED,16),(precision.SELECT_SEED,16),(precision.TEST_SEED,64)):
        prior_ids.update(r["id"] for r in base.items(seed,n))
    if ids&prior_ids:
        raise ValueError("test overlaps earlier splits")
    meta["test_sha"]=base.digest(test)
    rows={k:[] for k in arms}
    base.write(args.out,dict(meta=meta,status="SEALED",seal=seal,admission=parity,rows=rows))
    for i,it in enumerate(test):
        ids_=spectral.input_ids(tok,it)
        names=list(arms)
        random.Random(spectral.seed(it)).shuffle(names)
        for k in names:
            arm=arms[k]
            us=us0.to("cuda") if arm.get("spectral") else None
            up=up0.to("cuda") if arm.get("precision") else None
            base.sync()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            z,info=forward(model,ids_,it,arm,cfg,us,up)
            base.sync()
            ms=(time.perf_counter()-start)*1000
            peak=torch.cuda.max_memory_allocated()
            r=base.row(z,cand,labels.index(it["answer"]))
            r.update(id=it["id"],family=it["family"],tokens=ids_.shape[-1],latency_ms=ms,
                     peak_bytes=peak,coda_passes=1,**info)
            rows[k].append(r)
            del z,us,up
        if (i+1)%8==0:
            base.write(args.out,dict(meta=meta,status="IN_PROGRESS",seal=seal,admission=parity,rows=rows))
            print("LCD-07",i+1,len(test),flush=True)
    primary="+".join(components)
    sums={k:base.summarize(v) for k,v in rows.items()}
    costs={}
    for k,rr in rows.items():
        ms=sum(r["latency_ms"] for r in rr)
        costs[k]=dict(total_ms=ms,correct_per_second=sum(r["correct"] for r in rr)*1000/ms,
                      peak_bytes=max(r["peak_bytes"] for r in rr),
                      mean_loops=sum(r["loops"] for r in rr)/len(rr),
                      token_weighted_loops=sum(r["loops"]*r["tokens"] for r in rr)/sum(r["tokens"] for r in rr),
                      codec_roundtrips=sum(r["codec_roundtrips"] for r in rr))
    comp={k:dict(paired=base.paired(rows[primary],v),
                latency_ratio=costs[primary]["total_ms"]/costs[k]["total_ms"])
          for k,v in rows.items() if k!=primary}
    base.write(args.out,dict(meta=meta,status="PILOT_COMPLETE",seal=seal,admission=parity,primary=primary,
                summaries=sums,costs=costs,comparisons=comp,rows=rows))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","test"),required=True)
    p.add_argument("--results",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    elif a.results is None or a.out is None or a.out.exists():
        p.error("results root and fresh output required")
    else:
        main(a)
