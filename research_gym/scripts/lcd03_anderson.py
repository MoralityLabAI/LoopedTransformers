"""LCD-03: bounded Anderson trajectory extrapolation in native Huginn.

One intervention at step 12 then four native transitions, or one at step 16
before readout. Every primary arm pays exactly 16 native transitions, one coda.
A stationarity/resume admission check precedes calibration and held-out runs.
"""
from __future__ import annotations
import argparse
import hashlib
import random
import time
from pathlib import Path
import json
import math
import torch
from research_gym.scripts import lcd01_huginn as base
from research_gym.scripts import lcd02_spectral as spectral

SCHEMA="lcd03-v1"
CAL_SEED, TEST_SEED=2026112507,2026112508
DEPTH=16
WINDOWS=(3,5)
RIDGES=(.001,.01)
DOSES=(.25,.5,1.)
INTERVENTIONS=(12,16)
MAX_GAIN=2.
MAX_L1=8.

def norm(x):
    return torch.linalg.vector_norm(x.float())

def mix(outputs,residuals,ridge):
    """Constrained regularized Anderson coefficients; trace-normalized Gram."""
    if not 2<=len(outputs)==len(residuals) or ridge<=0:
        raise ValueError("invalid history")
    r=torch.stack([x.float().reshape(-1) for x in residuals])
    if not torch.isfinite(r).all():
        raise ValueError("nonfinite residual")
    gram=(r@r.T).double()
    n=len(outputs)
    scale=float(gram.diag().mean())
    if scale<=1e-20:
        return outputs[-1].float(),dict(alpha=[0.]*(n-1)+[1.],condition=1.,
                                        l1=1.,fallback="zero_residual")
    a=gram/scale+ridge*torch.eye(n,device=gram.device,dtype=gram.dtype)
    one=torch.ones(n,device=gram.device,dtype=gram.dtype)
    cond=float(torch.linalg.cond(a))
    v=torch.linalg.solve(a,one)
    alpha=v/v.sum()
    l1=float(alpha.abs().sum())
    if not torch.isfinite(alpha).all() or not math.isfinite(cond) or l1>MAX_L1:
        return outputs[-1].float(),dict(alpha=[0.]*(n-1)+[1.],condition=cond if math.isfinite(cond) else None,
                                        l1=l1 if math.isfinite(l1) else None,fallback="ill_conditioned")
    y=sum((x.float()*alpha[i].float() for i,x in enumerate(outputs)),torch.zeros_like(outputs[-1],dtype=torch.float32))
    return y,dict(alpha=alpha.tolist(),condition=cond,l1=l1,fallback=None)

def bounded_correction(history,ridge,dose):
    outputs=history[1:]
    residuals=[b.float()-a.float() for a,b in zip(history[:-1],history[1:])]
    y,rec=mix(outputs,residuals,ridge)
    d=y-outputs[-1].float()
    budget=MAX_GAIN*norm(residuals[-1])
    raw=dose*d
    limit=torch.minimum(torch.ones((),device=raw.device),budget/norm(raw).clamp_min(1e-20))
    g=raw*limit
    rec.update(raw_norm=float(norm(raw)),correction_norm=float(norm(g)),
               latest_residual=float(norm(residuals[-1])),clipped=bool(limit<1))
    return g,rec

def matched_direction(axis,g):
    n=norm(axis)
    if float(n)<=1e-20:
        return torch.zeros_like(axis),bool(norm(g)>1e-20)
    return axis.float()*(norm(g)/n),False

def sham_direction(g,seed):
    gen=torch.Generator(device=g.device).manual_seed(seed)
    v=torch.randn(g.shape,device=g.device,generator=gen,dtype=torch.float32)
    # A single full-state Euclidean norm, not a separate norm at each position.
    q=g.float()
    v=v-(v*q).sum()/q.square().sum().clamp_min(1e-20)*q
    return v*(norm(q)/norm(v).clamp_min(1e-20))

def smoke():
    # An affine contraction has a fixed point at one. Unregularized AA(2)
    # extrapolates exactly in one dimension; regularization gives a near solution.
    h=[torch.tensor([0.])]
    for _ in range(2):
        h.append(.5*h[-1]+.5)
    y,rec=mix(h[1:],[b-a for a,b in zip(h[:-1],h[1:])],1e-7)
    assert abs(float(y)-1.)<1e-4
    assert abs(sum(rec["alpha"])-1.)<1e-6
    g,rec=bounded_correction(h,1e-7,1.)
    assert norm(g)<=MAX_GAIN*norm(h[-1]-h[-2])+1e-6
    zero=[torch.zeros(4) for _ in range(4)]
    z,r=bounded_correction(zero,.01,1.)
    assert torch.equal(z,zero[-1]) and r["fallback"]=="zero_residual"
    # Near-identical residuals, zero ridge prohibited, regularized solve stable.
    y,r=mix([torch.ones(4),torch.ones(4)*2],[torch.ones(4),torch.ones(4)],.001)
    assert torch.isfinite(y).all() and abs(sum(r["alpha"])-1.)<1e-6
    d=torch.arange(1.,33.).reshape(1,4,8)
    q=sham_direction(d,4)
    assert torch.allclose(norm(q),norm(d),rtol=1e-5)
    assert abs(float((q*d).sum()))<.001
    q,bad=matched_direction(d*.1,d)
    assert not bad and torch.allclose(norm(q),norm(d))
    q,bad=matched_direction(torch.zeros_like(d),d)
    assert bad and not bool(q.any())
    print("LCD-03 Anderson tensor smoke PASS")

def clone_idx(idx):
    return idx.clone() if isinstance(idx,torch.Tensor) else idx

def initialize(model,ids,seed):
    torch.manual_seed(seed)
    e,idx=model.embed_inputs(ids)
    return e,model.initialize_state(e,scale=1.),idx

def step(model,e,h,idx,t):
    y,idx,_=model.iterate_one_step(e,h,block_idx=idx,current_step=t)
    return y,idx

def errors(x,y):
    d=x.float()-y.float()
    a=float(d.abs().max())
    b=float(norm(d)/norm(y).clamp_min(1e-20))
    return dict(max_abs=a if math.isfinite(a) else None,relative=b if math.isfinite(b) else None)

def within(r):
    return r["relative"] is not None and r["max_abs"] is not None and r["max_abs"]<=.125 and r["relative"]<=.005

@torch.inference_mode()
def admit(model,tok,roots):
    result=[]
    ok=True
    # One root per family. Compare complete hidden tensors at real native
    # indices, not final-token vectors; also compare outputs after a transplant.
    for it in roots[::len(roots)//3]:
        ids=spectral.input_ids(tok,it)
        seed=spectral.seed(it)
        e,h,idx=initialize(model,ids,seed)
        saved={0:(h.clone(),clone_idx(idx))}
        for t in range(DEPTH):
            h,idx=step(model,e,h,idx,t)
            if t+1 in (6,8,12,16):
                saved[t+1]=(h.clone(),clone_idx(idx))
        z=base.read(model,h)
        torch.manual_seed(seed)
        native=model(input_ids=ids,num_steps=DEPTH,use_cache=False).logits[:,-1].float()
        parity=errors(z,native)
        top1=bool((z.argmax(-1)==native.argmax(-1)).all())
        ss=base.trajectory(model,ids,seed,32,(DEPTH,))
        prefix=errors(ss[DEPTH],saved[DEPTH][0])
        z32=base.read(model,ss[32])
        torch.manual_seed(seed)
        n32=model(input_ids=ids,num_steps=32,use_cache=False).logits[:,-1].float()
        parity32=errors(z32,n32)
        top32=bool((z32.argmax(-1)==n32.argmax(-1)).all())
        hs,ix=saved[8]
        resumed=hs.clone()
        j=clone_idx(ix)
        for t in range(8,16):
            resumed,j=step(model,e,resumed,j,t)
        resume=errors(resumed,saved[16][0])
        # Same h, same embedding, different real step/index pairs, and RNG
        # changes. If the map changes, do not label the method fixed-point AA.
        hs,ix=saved[6]
        torch.manual_seed(seed+1)
        ref,_=step(model,e,hs.clone(),clone_idx(ix),6)
        comparisons=[]
        for t in (6,8,12,16):
            torch.manual_seed(seed+2)
            v,_=step(model,e,hs.clone(),clone_idx(saved[t][1]),t)
            comparisons.append(dict(step=t,**errors(v,ref)))
        passed=(top1 and top32 and within(parity) and within(parity32) and within(prefix)
                and within(resume) and all(within(r) for r in comparisons))
        result.append(dict(id=it["id"],parity=parity,top1_equal=top1,parity32=parity32,
                           top1_equal32=top32,prefix16=prefix,resume=resume,
                           same_input_map=comparisons,passed=passed))
        ok=ok and passed
    return ok,result

def configs():
    out=[]
    for t in INTERVENTIONS:
        for w in WINDOWS:
            for r in RIDGES:
                for dose in DOSES:
                    out.append(dict(method="anderson",at=t,window=w,ridge=r,dose=dose))
        for dose in DOSES:
            out.append(dict(method="scalar",at=t,dose=dose))
    return out

@torch.inference_mode()
def forward(model,ids,it,c,diagnose=False):
    method=c["method"]
    depth=32 if method=="deep" else DEPTH
    seed=spectral.seed(it)
    e,h,idx=initialize(model,ids,seed)
    ref=None
    source=c.get("source",c)
    at=source.get("at",0)
    window=source.get("window",0)
    ref_at=source.get("ref",at-4)
    history=[]
    if window and at-window==0:
        history=[h.clone()]
    rec=None
    for t in range(1,depth+1):
        h,idx=step(model,e,h,idx,t-1)
        if method in ("scalar","matched_scalar") and t==ref_at:
            ref=h.clone()
        if window and at-window<=t<=at:
            history.append(h.clone())
        if t==at:
            if method=="scalar":
                g=c["dose"]*(h.float()-ref.float())
                rec=dict(correction_norm=float(norm(g)))
            else:
                g,rec=bounded_correction(history,source["ridge"],source["dose"])
                if method=="matched_scalar":
                    g,bad=matched_direction(h.float()-ref.float(),g)
                    rec["norm_match_degenerate"]=bad
                elif method=="matched_random":
                    g=sham_direction(g,seed^0xCAFE)
                rec["applied_norm"]=float(norm(g))
            guided=(h.float()+g).to(h.dtype)
            rec["cast_norm"]=float(norm(guided.float()-h.float()))
            h=guided
            del g,guided
            history.clear()
            ref=None
    z=base.read(model,h)
    # Diagnostic residual measured at *actual* final state using the next
    # native map. Charged separately; not available to any inference arm.
    residual=None
    if diagnose:
        return z,rec,(e,h,idx,depth)
    return z,rec,None

def candidate_key(c):
    return base.key(c)

@torch.inference_mode()
def evaluate(model,tok,labels,cand,roots,arms,meta,out,admission,timed=False):
    rows={name:[] for name in arms}
    for i,it in enumerate(roots):
        ids=spectral.input_ids(tok,it)
        target=labels.index(it["answer"])
        names=list(arms)
        random.Random(spectral.seed(it)).shuffle(names)
        for name in names:
            c=arms[name]
            base.sync()
            if timed:
                torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            z,rec,diag=forward(model,ids,it,c,diagnose=timed)
            base.sync()
            elapsed=(time.perf_counter()-start)*1000
            peak=torch.cuda.max_memory_allocated() if timed else None
            residual=None
            if timed:
                e,h,idx,depth=diag
                next_h,_=step(model,e,h.clone(),clone_idx(idx),depth)
                residual=float(norm(next_h.float()-h.float())/norm(h).clamp_min(1e-20))
                if not math.isfinite(residual):
                    raise ValueError("nonfinite native diagnostic residual")
                del e,h,idx,next_h
            r=base.row(z,cand,target)
            r.update(id=it["id"],family=it["family"],target=target,seed=spectral.seed(it),
                     loops=32 if c["method"]=="deep" else DEPTH,coda_passes=1,
                     diagnostic_extra_loops=int(timed),tokens=ids.shape[-1],latency_ms=elapsed,
                     peak_bytes=peak,final_relative_residual=residual,intervention=rec)
            rows[name].append(r)
            del z,diag
        if (i+1)%8==0:
            print("LCD-03",i+1,len(roots),flush=True)
            base.write(out,dict(meta=meta,status="IN_PROGRESS",admission=admission,arms=arms,rows=rows))
    return rows,{k:base.summarize(v) for k,v in rows.items()}

def choose(arms,summaries):
    def best(method,at=None):
        keys=[k for k,c in arms.items() if c["method"]==method and (at is None or c["at"]==at)]
        return arms[min(keys,key=lambda k:(-summaries[k]["accuracy"],summaries[k]["nll"],k))]
    out={"base16":dict(method="base"),"base32":dict(method="deep"),
         "loopcd16":dict(method="scalar",at=16,ref=6,dose=.5)}
    for at in INTERVENTIONS:
        a=best("anderson",at)
        out[f"aa{at}"]=a
        out[f"scalar{at}"]=best("scalar",at)
        out[f"matched_scalar{at}"]=dict(method="matched_scalar",source=a)
        out[f"matched_random{at}"]=dict(method="matched_random",source=a)
    return out

def main(a):
    model,tok,labels,cand,meta=spectral.load()
    meta.update(schema=SCHEMA,code_sha=hashlib.sha256(Path(__file__).read_bytes()+
                Path(spectral.__file__).read_bytes()+Path(base.__file__).read_bytes()).hexdigest(),
                cal_seed=CAL_SEED,depth=DEPTH,interventions=list(INTERVENTIONS))
    cal=base.items(CAL_SEED,16)
    meta["cal_sha"]=base.digest(cal)
    ok,receipt=admit(model,tok,cal)
    if not ok:
        base.write(a.out,dict(meta=meta,status="BLOCKED_NATIVE_MAP",admission=receipt))
        raise RuntimeError("stationarity or resume admission failed; retain receipt")
    if a.stage=="calibrate":
        arms={candidate_key(c):c for c in configs()}
        rows,summaries=evaluate(model,tok,labels,cand,cal,arms,meta,a.out,receipt)
        frozen=choose(arms,summaries)
        base.write(a.out,dict(meta=meta,status="FROZEN",admission=receipt,arms=frozen,
                               freeze_sha=base.digest(frozen),rows=rows,summaries=summaries))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or f["freeze_sha"]!=base.digest(f["arms"]):
        raise ValueError("invalid freeze or environment mismatch")
    arms=f["arms"]
    if arms!=choose({candidate_key(c):c for c in configs()},f["summaries"]):
        raise ValueError("freeze does not match recorded calibration selection")
    test=base.items(TEST_SEED,64)
    if {r["id"] for r in cal}&{r["id"] for r in test}:
        raise ValueError("overlapping roots")
    if len({r["id"] for r in test})!=len(test):
        raise ValueError("duplicate test roots")
    meta["test_sha"]=base.digest(test)
    rows,summaries=evaluate(model,tok,labels,cand,test,arms,meta,a.out,receipt,timed=True)
    comp={}
    for at in INTERVENTIONS:
        r=rows[f"aa{at}"]
        comp[str(at)]={}
        for name in ("base16","base32","loopcd16",f"scalar{at}",f"matched_scalar{at}",f"matched_random{at}"):
            b=rows[name]
            comp[str(at)][name]=dict(paired=base.paired(r,b),
                latency_ratio=None if name.startswith("matched") else sum(x["latency_ms"] for x in r)/sum(x["latency_ms"] for x in b),
                residual_delta=sum(x["final_relative_residual"]-y["final_relative_residual"] for x,y in zip(r,b))/len(r))
    base.write(a.out,dict(meta=meta,status="PILOT_COMPLETE",admission=receipt,arms=arms,
                           freeze_sha=f["freeze_sha"],rows=rows,summaries=summaries,comparisons=comp))

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
            p.error("need fresh --out; test also needs --freeze")
        main(a)
