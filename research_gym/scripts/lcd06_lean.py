"""LCD-06B: frozen recurrence-guided ranking with independent Lean certification."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch
from torch import nn
from research_gym.scripts import lcd01_huginn as base

SCHEMA="lcd06b-v1"
CAL_SEED,TEST_SEED=2026112516,2026112517
DEPTH,REF,BUDGET=8,4,32

def guided_logits(z,ref,legal,dose):
    d=torch.where(legal,z.float()-ref.float(),0.)
    return (z.float()+dose*d).masked_fill(~legal,-1e4)

class Guided(nn.Module):
    def __init__(self,model,method,dose=0.):
        super().__init__()
        self.model=model
        self.cfg=model.cfg
        self.method=method
        self.dose=dose
        self.readout_calls=0
        self.forward_calls=0

    def forward(self,b,fast_steps=None,glob=None):
        T=fast_steps or self.cfg.fast_steps
        self.forward_calls+=1
        states=[]
        hook=None
        if self.method in ("hidden","random"):
            def grab(_,args):
                states.append(args[0].clone())
            hook=self.model.readout.register_forward_pre_hook(grab)
        try:
            outs=self.model(b,fast_steps=T,glob=glob)
        finally:
            if hook is not None:
                hook.remove()
        self.readout_calls+=len(outs)
        if self.method=="base":
            return outs
        ri=[i for i,o in enumerate(outs) if o["steps"]==REF]
        if len(ri)!=1 or outs[-1]["steps"]!=DEPTH:
            raise ValueError("required readout ticks absent")
        i=ri[0]
        final=dict(outs[-1])
        if self.method=="logits":
            final["logits"]=guided_logits(final["logits"],outs[i]["logits"],b["legal"],self.dose)
        else:
            if len(states)!=len(outs):
                raise RuntimeError("readout trace mismatch")
            h=states[-1]
            d=(h.float()-states[i].float())*b["tok_mask"][...,None]
            if self.method=="random":
                # Deterministic per proof-state inputs; padded tokens have zero correction.
                raw=b["x"].detach().cpu().contiguous().float().numpy().tobytes()
                seed=int(hashlib.sha256(raw).hexdigest()[:8],16)
                d=base.random_direction(d,seed)
            h=(h.float()+self.dose*d).to(h.dtype)
            lg,_=self.model.readout(h,b["main"],b["legal"])
            self.readout_calls+=1
            final["logits"]=lg.float().masked_fill(~b["legal"],-1e4)
        # Keep atom and halt heads identical to the ordinary terminal readout.
        return outs[:-1]+[final]

def smoke():
    a=torch.tensor([[1.,2.,-10000.,3.]])
    b=torch.tensor([[2.,0.,-10000.,1.]])
    legal=torch.tensor([[True,True,False,True]])
    z=guided_logits(a,b,legal,.5)
    assert torch.equal(z,torch.tensor([[.5,3.,-10000.,4.]]))
    assert torch.equal(guided_logits(a,b,legal,0.),a)
    print("LCD-06B legal-mask smoke PASS")

def dataset_exclusions(name):
    from lcgrt import config,fragment as F
    from lcgrt.dataset import iter_records
    root=config.DATA_ROOT/name
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    seen=set()
    splits=sorted({s["split"] for s in manifest["shards"]})
    if "train" not in splits or "val" not in splits:
        raise ValueError("training/validation corpus missing")
    for shard in manifest["shards"]:
        p=root/shard["split"]/shard["shard"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=shard["sha256"]:
            raise ValueError("dataset shard hash mismatch")
    for split in splits:
        for r in iter_records(name,split,certified_only=False):
            kw={k:tuple(v) if isinstance(v,list) else v for k,v in (r.get("gen_kwargs") or {}).items()}
            thm=F.generate(r["seed"],r["depth"],r["family"],arity3=r["arity3"],tid=r["tid"],**kw)
            if F.stmt_sha(thm)!=r["stmt_sha256"]:
                raise ValueError("dataset statement regeneration mismatch")
            seen.add(F.canonical_hash(thm))
    return seen,hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()

def roots(seed,n,seen,split):
    from lcgrt import fragment as F
    rng=random.Random(seed)
    families=list(F.TRAIN_FAMILIES)
    rows=[]
    for j in range(n):
        fam=families[j%len(families)]
        depth=2+(j//len(families))%5
        for _ in range(1000):
            s=rng.randrange(1<<31)
            tid=f"lcd06b_{split}_{j:04d}"
            thm=F.generate(s,depth,fam,arity3=False,tid=tid)
            h=F.canonical_hash(thm)
            if h in seen:
                continue
            seen.add(h)
            rows.append((tid,F.statement(thm),dict(family=fam,depth=depth,seed=s,
                        arity3=False,stmt_sha256=F.stmt_sha(thm),canonical_hash=h)))
            break
        else:
            raise RuntimeError("could not generate novel theorem")
    return rows

def certify_rows(rows):
    from lcgrt import fragment as F
    from lcgrt.certify import certify,decl_name
    items=[]
    for r in rows:
        r["certified"]=False
        if r["success"]:
            t=F.generate(r["seed"],r["depth"],r["family"],arity3=r["arity3"],tid=r["tid"])
            if F.stmt_sha(t)!=r["stmt_sha256"]:
                raise ValueError("proof statement mismatch")
            items.append((r["tid"],t,r["script"]))
    if not items:
        return dict(batch=None,individual=[],passed=0)
    def clean(rec):
        return {k:rec[k] for k in ("source_sha256","lean_version","returncode","timed_out","status","passed")}
    result=certify(items)
    individual=[]
    good=set()
    if result["returncode"]==0 and not result["timed_out"]:
        good={tid for tid,_,_ in items if result["status"].get(decl_name(tid))=="passed"}
    else:
        # A failing declaration must not discard unrelated valid proofs.
        # Require independently successful standalone compilation.
        for item in items:
            r=certify([item])
            individual.append(clean(r))
            if r["returncode"]==0 and not r["timed_out"] and r["status"].get(decl_name(item[0]))=="passed":
                good.add(item[0])
    for r in rows:
        r["certified"]=r["tid"] in good
    return dict(batch=clean(result),individual=individual,passed=len(good))

def evaluate(model,items,arms,device,out,meta):
    from lcgrt.search import Scorer,run_searches
    from lcgrt.lean_repl import LeanRepl
    reports={}
    for name,c in arms.items():
        wrapper=Guided(model,c["method"],c.get("dose",0.)).eval()
        scorer=Scorer(wrapper,device,fast_steps=c.get("depth",DEPTH),halt_threshold=None,use_glob=False)
        start=time.perf_counter()
        with LeanRepl(imports=()) as repl:
            rows=run_searches(repl,items,"best",scorer,budget_fn=lambda _:BUDGET,width=1,seed=42)
            calls=repl.calls
            lean_version=repl.version
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        search_seconds=time.perf_counter()-start
        start=time.perf_counter()
        cert=certify_rows(rows)
        reports[name]=dict(accuracy=sum(r["certified"] for r in rows)/len(rows),
            raw_closure=sum(r["success"] for r in rows),lean_calls=calls,
            model_forwards=scorer.forwards,model_fast_steps=sum(scorer.steps_used),
            readout_calls=wrapper.readout_calls,search_seconds=search_seconds,
            certificate_seconds=time.perf_counter()-start,lean_version=lean_version,
            certificate=cert,rows=rows)
        print("LCD-06B",name,reports[name]["accuracy"],flush=True)
        base.write(out,dict(meta=meta,status="IN_PROGRESS",arms=arms,reports=reports))
    return reports

def configs():
    return {base.key(c):c for c in [dict(method=m,dose=w) for m in ("logits","hidden") for w in (.25,.5,1.)]}

def choose(reports):
    cs=configs()
    selected={}
    for m in ("logits","hidden"):
        options=[k for k,c in cs.items() if c["method"]==m]
        k=min(options,key=lambda k:(-reports[k]["accuracy"],
                                  sum(r["expansions"] for r in reports[k]["rows"]),k))
        selected[m]=cs[k]
    return dict(base8=dict(method="base",depth=8),base16=dict(method="base",depth=16),
                logits=selected["logits"],hidden=selected["hidden"],
                random=dict(method="random",dose=selected["hidden"]["dose"]))

def paired(a,b):
    import numpy as np
    bb={r["tid"]:r for r in b}
    if len(a)!=len(bb) or {r["tid"] for r in a}!=set(bb):
        raise ValueError("unpaired Lean roots")
    delta=np.array([int(r["certified"])-int(bb[r["tid"]]["certified"]) for r in a])
    rng=np.random.default_rng(2026112518)
    reps=rng.choice(delta,(5000,len(delta)),replace=True).mean(1)
    return dict(delta=float(delta.mean()),ci95=np.quantile(reps,[.025,.975]).tolist(),
                wrong_to_right=int((delta==1).sum()),right_to_wrong=int((delta==-1).sum()))

def main(a):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research"/"lean_cgrt_hrm400"))
    from lcgrt import config
    from lcgrt.model import ProofNet,ModelConfig
    from lcgrt.train import load_verified,sha256
    if not (a.ckpt/"complete.json").exists():
        raise ValueError("complete checkpoint required")
    which,ck=load_verified(a.ckpt,a.device)
    if ck is None or ck["identity"]["data"]!=a.data:
        raise ValueError("verified checkpoint with matching data identity required")
    model=ProofNet(ModelConfig(**ck["config"])).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()
    if model.cfg.arch!="hrm" or REF%model.cfg.slow_every or DEPTH%model.cfg.slow_every:
        raise ValueError("HRM checkpoint with required ticks needed")
    seen,manifest_sha=dataset_exclusions(a.data)
    cal=roots(CAL_SEED,24,seen,"cal")
    meta=dict(schema=SCHEMA,checkpoint_sha=sha256(a.ckpt/(which+".pt")),model=model.cfg.to_json(),
              data=a.data,manifest_sha=manifest_sha,cal_sha=base.digest(cal),budget=BUDGET,
              torch=torch.__version__,device=a.device,lean_exe_sha=sha256(config.LEAN_EXE),
              repl_exe_sha=sha256(config.REPL_EXE),code_sha=hashlib.sha256(Path(__file__).read_bytes()+
                  Path(base.__file__).read_bytes()).hexdigest())
    if a.stage=="calibrate":
        reports=evaluate(model,cal,configs(),a.device,a.out,meta)
        arms=choose(reports)
        base.write(a.out,dict(meta=meta,status="FROZEN",arms=arms,freeze_sha=base.digest(arms),reports=reports))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or f["freeze_sha"]!=base.digest(f["arms"]) or f["arms"]!=choose(f["reports"]):
        raise ValueError("freeze mismatch")
    test=roots(TEST_SEED,96,seen,"test")
    meta["test_sha"]=base.digest(test)
    arms=f["arms"]
    reports=evaluate(model,test,arms,a.device,a.out,meta)
    comparisons={m:{k:paired(reports[m]["rows"],v["rows"]) for k,v in reports.items() if k!=m}
                 for m in ("logits","hidden")}
    base.write(a.out,dict(meta=meta,status="PILOT_COMPLETE",freeze_sha=f["freeze_sha"],arms=arms,
                reports=reports,comparisons=comparisons))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","calibrate","test"),required=True)
    p.add_argument("--ckpt",type=Path)
    p.add_argument("--data",default="lcgrt-v1")
    p.add_argument("--device",default="cpu")
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.ckpt is None or a.out is None or a.out.exists() or (a.stage=="test" and a.freeze is None):
            p.error("verified checkpoint, fresh output, test freeze required")
        main(a)
