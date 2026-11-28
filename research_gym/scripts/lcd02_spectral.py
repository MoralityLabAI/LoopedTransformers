"""LCD-02: calibration-fitted low-rank recurrent extrapolation.

Native Huginn pilot; frozen model, 16 steps, reference at step 6.
Run --stage smoke on CPU; fit and test require the existing CUDA environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from research_gym.scripts import lcd01_huginn as base

SCHEMA = "lcd02-v1"
FIT_SEED, SELECT_SEED, TEST_SEED = 2026112504, 2026112505, 2026112506
DEPTH, REF, OMEGA, EPS = 16, 6, .5, .25
RANKS, RIDGES = (4, 8, 16), (.1, 1.)
SCALAR_DOSES = (0., .25, .5, .75, 1.)
RANDOM_SEED = 271828

def norm(x):
    return torch.linalg.vector_norm(x.float())

def get_basis(samples, rank, random_seed=None):
    """Uncentered contrast SVD of one final-token vector per calibration root."""
    x = samples.detach().cpu().double()
    if x.ndim != 2 or rank > min(x.shape):
        raise ValueError("rank exceeds sample matrix dimensions")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite contrast")
    if random_seed is None:
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        if s[rank-1] <= s[0]*1e-8:
            raise ValueError("insufficient empirical rank")
        return vh[:rank].T.float()
    g = torch.Generator().manual_seed(random_seed)
    q, _ = torch.linalg.qr(torch.randn(x.shape[1],rank,generator=g,dtype=torch.float64))
    return q.float()

def projected(d, u, coeff):
    d = d.float()
    u = u.to(device=d.device,dtype=d.dtype)
    coeff = coeff.to(device=d.device,dtype=d.dtype)
    return ((d @ u)*coeff) @ u.T

def cap(g, d, max_ratio=1.):
    # Batch-one native pilot: global per-prompt norm across all sequence positions.
    return g * torch.minimum(torch.ones((),device=g.device),
                              max_ratio*norm(d)/norm(g).clamp_min(1e-20))

def guide(d, u, lam):
    return cap(OMEGA*d.float()+projected(d,u,lam), d)

def scalar_match(d, g):
    return d.float() * (norm(g)/norm(d).clamp_min(1e-20))

def perturb(h,g):
    return (h.float()+g).to(h.dtype)

def margin(z,target):
    # Margin of correct option against logsumexp over the other nine.
    other = torch.cat((z[:target], z[target+1:]))
    return z[target]-other.logsumexp(0)

def fit_ridge(features, target, ridge):
    x = features.double().cpu()
    y = target.double().cpu()
    if x.ndim != 2 or y.shape != (x.shape[0],) or ridge <= 0:
        raise ValueError("invalid ridge input")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("nonfinite ridge input")
    eye = torch.eye(x.shape[1],dtype=x.dtype)
    lam = torch.linalg.solve(x.T@x/len(x)+ridge*eye,x.T@y/len(x))
    # Effective spectral multipliers stay in [0,1]; cap full correction too.
    return lam.clamp(-OMEGA,1.-OMEGA).float()

def smoke():
    gen=torch.Generator().manual_seed(4)
    # Known rank-three population: recover subspace up to a rotation.
    q,_=torch.linalg.qr(torch.randn(32,3,generator=gen))
    x=torch.randn(48,3,generator=gen)@q.T
    u=get_basis(x,3)
    v=get_basis(x,3,RANDOM_SEED)
    assert torch.allclose(u.T@u,torch.eye(3),atol=1e-5)
    assert torch.allclose(u@u.T,q@q.T,atol=1e-5)
    assert torch.allclose(v.T@v,torch.eye(3),atol=1e-5)
    assert torch.equal(v,get_basis(x,3,RANDOM_SEED))
    d=torch.randn(1,6,32,generator=gen)
    g=guide(d,u,torch.tensor([.5,-.5,.25]))
    assert norm(g) <= norm(d)+1e-5
    assert torch.allclose(norm(g),norm(scalar_match(d,g)),atol=1e-5)
    assert torch.equal(guide(torch.zeros_like(d),u,torch.ones(3)),torch.zeros_like(d))
    assert torch.allclose(projected(d,u,torch.ones(3)),d@u@u.T,atol=1e-5)
    cfg=dict(method="pca",rank=3,lam=[.5,-.5,.25])
    rb=dict(method="random",rank=3,lam=[.2,.1,-.3])
    bb=dict(pca=u,random=v)
    assert torch.allclose(norm(correction(d,dict(method="matched_random",source=cfg,random=rb),bb)),norm(g),atol=1e-5)
    f=torch.randn(128,3,generator=gen)
    truth=torch.tensor([.2,-.3,.1])
    lam=fit_ridge(f,f@truth,1e-6)
    assert torch.allclose(lam,truth,atol=1e-4)
    assert fit_ridge(f,100*f[:,0],.1).abs().max()<=.5
    # Strong synthetic contrast signal with irrelevant high-variance directions:
    # checks numerical estimator, not model accuracy.
    assert margin(torch.tensor([2.,0.,0.]),0)>0
    print("LCD-02 spectral tensor smoke PASS")

def seed(it):
    return int(it["id"][:8],16)

def data(include_test=False):
    # Fit U and coefficients, select hyperparameters, and test on disjoint fresh roots.
    fit=base.items(FIT_SEED,16)
    select=base.items(SELECT_SEED,16)
    test=base.items(TEST_SEED,64) if include_test else []
    sets=[{r["id"] for r in x} for x in (fit,select,test)]
    if any(len(s)!=len(x) for s,x in zip(sets,(fit,select,test))) or any(
        sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError("duplicate or overlapping roots")
    return fit,select,test

def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(base.__file__).read_bytes()).hexdigest()

def load():
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast
    import transformers
    from research_gym.scripts.rmp_huginn_depth_sweep import load_huginn,answer_tokens
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for native stages")
    torch.backends.cuda.matmul.allow_tf32=False
    local=Path(snapshot_download(base.MODEL,revision=base.REV,local_files_only=True))
    tok=PreTrainedTokenizerFast(tokenizer_file=str(local/"tokenizer.json"),
                               bos_token="<|begin_text|>",add_bos_token=True)
    labels, ids=answer_tokens(tok,"varchain")
    if len(set(ids))!=len(labels):
        raise ValueError("ambiguous token ids")
    model=load_huginn(local,torch.device("cuda")).eval()
    meta=dict(schema=SCHEMA,code_sha=code_sha(),model=base.MODEL,revision=base.REV,
              torch=torch.__version__,transformers=transformers.__version__,
              device=torch.cuda.get_device_name(),dtype=str(next(model.parameters()).dtype),
              depth=DEPTH,ref=REF,eps=EPS)
    return model,tok,labels,torch.tensor(ids,device="cuda"),meta

def input_ids(tok,it):
    ids=torch.tensor([tok.encode(it["prompt"])],device="cuda")
    if int(ids[0,0])!=65504:
        raise ValueError("BOS mismatch")
    return ids

def state(model,ids,it,depth=DEPTH,need_contrast=True):
    keep_ref=depth==DEPTH and need_contrast
    ss=base.trajectory(model,ids,seed(it),depth,(REF,) if keep_ref else ())
    h=ss[depth]
    return h,h.float()-ss[REF].float() if keep_ref else None

@torch.inference_mode()
def parity(model,tok,fit):
    receipt=[]
    for it in fit[::16]:
        ids=input_ids(tok,it)
        for depth in (DEPTH,32):
            h,_=state(model,ids,it,depth)
            z=base.read(model,h)
            torch.manual_seed(seed(it))
            native=model(input_ids=ids,num_steps=depth,use_cache=False).logits[:,-1].float()
            err=float((z-native).abs().max())
            receipt.append(dict(id=it["id"],depth=depth,max_abs=err,
                                top1_equal=bool((z.argmax(-1)==native.argmax(-1)).all())))
            if not torch.isfinite(z).all() or not torch.isfinite(native).all() or err>.125 or not receipt[-1]["top1_equal"]:
                return False,receipt
    return True,receipt

@torch.inference_mode()
def fit(model,tok,labels,cand,roots):
    begin=time.perf_counter()
    vectors=[]
    for it in roots:
        h,d=state(model,input_ids(tok,it),it)
        vectors.append(d[0,-1].cpu())
    samples=torch.stack(vectors)
    bases={k:get_basis(samples,max(RANKS),None if k=="pca" else RANDOM_SEED)
           for k in ("pca","random")}
    features={k:[] for k in bases}
    targets=[]
    for i,it in enumerate(roots):
        h,d=state(model,input_ids(tok,it),it)
        target=labels.index(it["answer"])
        scalar=OMEGA*d
        z=base.read(model,perturb(h,scalar))[0,cand]
        m=margin(z,target)
        targets.append(float((-m).clamp(0.,4.)))
        for k,u0 in bases.items():
            u=u0.to(d.device)
            a=d@u
            vals=[]
            # Central finite differences around scalar LoopCD. Each mode incurs two coda passes.
            for j in range(u.shape[1]):
                dj=a[...,j:j+1]*u[:,j]
                plus=base.read(model,perturb(h,scalar+EPS*dj))[0,cand]
                minus=base.read(model,perturb(h,scalar-EPS*dj))[0,cand]
                vals.append(float((margin(plus,target)-margin(minus,target))/(2*EPS)))
            features[k].append(vals)
        if (i+1)%8==0:
            print("spectral fit",i+1,len(roots),flush=True)
    candidates=[]
    for k,u in bases.items():
        for rank in RANKS:
            for ridge in RIDGES:
                lam=fit_ridge(torch.tensor(features[k])[:,:rank],torch.tensor(targets),ridge)
                candidates.append(dict(method=k,rank=rank,ridge=ridge,lam=lam.tolist()))
    for w in SCALAR_DOSES:
        candidates.append(dict(method="scalar",dose=w))
    # Fixed label-free projected contrast, with explicit same-rank random projection.
    candidates.extend(dict(method=k+"_projected",rank=8) for k in bases)
    return {k:u.tolist() for k,u in bases.items()},candidates,dict(
        features=features,targets=targets,fit_elapsed_s=time.perf_counter()-begin,
        fit_roots=len(roots),fit_coda_passes=len(roots)*(1+4*max(RANKS)),
        singular_values=torch.linalg.svdvals(samples.double()).tolist())

def correction(d,c,bases):
    m=c["method"]
    if m=="scalar":
        return c["dose"]*d
    if m in ("pca_projected","random_projected"):
        u=bases[m.split("_")[0]][:,:c["rank"]]
        return cap(projected(d,u,torch.full((c["rank"],),OMEGA)),d)
    if m=="matched_scalar":
        return scalar_match(d,correction(d,c["source"],bases))
    if m=="matched_random":
        g=correction(d,c["random"],bases)
        target=norm(correction(d,c["source"],bases))
        return g*(target/norm(g).clamp_min(1e-20))
    u=bases[m][:,:c["rank"]]
    return guide(d,u,torch.tensor(c["lam"]))

@torch.inference_mode()
def evaluate(model,tok,labels,cand,roots,candidates,basis_lists,timed=False):
    bases={k:torch.tensor(u,device="cuda") for k,u in basis_lists.items()}
    rows={base.key(c):[] for c in candidates}
    for i,it in enumerate(roots):
        ids=input_ids(tok,it)
        target=labels.index(it["answer"])
        order=list(candidates)
        random.Random(seed(it)).shuffle(order)
        if not timed:
            h,d=state(model,ids,it)
        for c in order:
            depth=32 if c["method"]=="deep" else DEPTH
            unguided=depth==32 or (c["method"]=="scalar" and c["dose"]==0.)
            if timed:
                base.sync()
                torch.cuda.reset_peak_memory_stats()
                t0=time.perf_counter()
                h,d=state(model,ids,it,depth,need_contrast=not unguided)
            g=None if unguided else correction(d,c,bases)
            z=base.read(model,h if g is None else perturb(h,g))
            if timed:
                base.sync()
                latency=(time.perf_counter()-t0)*1000
                peak=torch.cuda.max_memory_allocated()
            else:
                latency=peak=None
            r=base.row(z,cand,target)
            r.update(id=it["id"],family=it["family"],target=target,depth=depth,seed=seed(it),
                     tokens=ids.shape[-1],latency_ms=latency,peak_bytes=peak,
                     correction_norm=0. if g is None else float(norm(g)),
                     cast_correction_norm=0. if g is None else float(norm(perturb(h,g).float()-h.float())))
            rows[base.key(c)].append(r)
            del z,g
            if timed:
                del h,d
        if (i+1)%16==0:
            print("spectral eval",i+1,len(roots),flush=True)
    summaries={k:base.summarize(r) for k,r in rows.items()}
    return rows,summaries

def select_arms(candidates,summaries):
    selected={}
    for method in ("pca","random","scalar"):
        cs=[c for c in candidates if c["method"]==method]
        selected[method]=min(cs,key=lambda c:(-summaries[base.key(c)]["accuracy"],
                                             summaries[base.key(c)]["nll"],base.key(c)))
    p=selected["pca"]
    selected["random_rank_matched"]=next(c for c in candidates if c["method"]=="random"
                                         and c["rank"]==p["rank"] and c["ridge"]==p["ridge"])
    selected["matched_scalar"]=dict(method="matched_scalar",source=p)
    selected["matched_random"]=dict(method="matched_random",source=p,random=selected["random_rank_matched"])
    selected["base"]=dict(method="scalar",dose=0.)
    selected["loopcd"]=dict(method="scalar",dose=OMEGA)
    selected["deep"]=dict(method="deep")
    for m in ("pca_projected","random_projected"):
        selected[m]=next(c for c in candidates if c["method"]==m)
    return selected

def unique_configs(arms):
    return list({base.key(c):c for c in arms.values()}.values())

def main(a):
    model,tok,labels,cand,meta=load()
    fit_roots,sel_roots,test_roots=data(a.stage=="test")
    meta.update(fit_sha=base.digest(fit_roots),select_sha=base.digest(sel_roots))
    ok,receipt=parity(model,tok,fit_roots)
    if not ok:
        base.write(a.out,dict(meta=meta,status="BLOCKED_PARITY",parity=receipt))
        raise RuntimeError("native parity failed")
    if a.stage=="fit":
        bases,candidates,fit_receipt=fit(model,tok,labels,cand,fit_roots)
        start=time.perf_counter()
        rows,summaries=evaluate(model,tok,labels,cand,sel_roots,candidates,bases)
        arms=select_arms(candidates,summaries)
        fit_receipt["selection_elapsed_s"]=time.perf_counter()-start
        fit_receipt["selection_coda_passes"]=len(sel_roots)*len(candidates)
        frozen=dict(bases=bases,arms=arms)
        base.write(a.out,dict(meta=meta,status="FROZEN",parity=receipt,frozen=frozen,
                               freeze_sha=base.digest(frozen),fit=fit_receipt,
                               selection_rows=rows,selection_summaries=summaries))
        return
    f=json.loads(a.freeze.read_text(encoding="utf-8"))
    if f["status"]!="FROZEN" or f["meta"]!=meta or base.digest(f["frozen"])!=f["freeze_sha"]:
        raise ValueError("invalid freeze or changed source/environment")
    frozen=f["frozen"]
    arms=frozen["arms"]
    rows,summaries=evaluate(model,tok,labels,cand,test_roots,unique_configs(arms),frozen["bases"],timed=True)
    primary=rows[base.key(arms["pca"])]
    comp={}
    for name,c in arms.items():
        rr=rows[base.key(c)]
        comp[name]=dict(delta=base.paired(primary,rr),
                        pca_latency_ratio=(None if name in ("matched_scalar","matched_random") else
                            sum(r["latency_ms"] for r in primary)/sum(r["latency_ms"] for r in rr)),
                        peak_bytes=max(r["peak_bytes"] for r in rr))
    # Matched-norm arms compute the source correction to know its norm; they are diagnostic,
    # not fair latency baselines. Native forward cost excludes fit amortization.
    base.write(a.out,dict(meta=dict(meta,test_sha=base.digest(test_roots)),status="PILOT_COMPLETE",
                           parity=receipt,freeze_sha=f["freeze_sha"],arms=arms,
                           summaries=summaries,comparisons=comp,rows=rows))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("smoke","fit","test"),required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--freeze",type=Path)
    a=p.parse_args()
    if a.stage=="smoke":
        smoke()
    else:
        if a.out is None or a.out.exists() or (a.stage=="test" and a.freeze is None):
            p.error("need fresh --out; test also needs --freeze")
        main(a)
