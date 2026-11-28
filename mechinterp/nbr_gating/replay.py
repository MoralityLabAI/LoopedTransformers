"""NBR-02A: retrospective activation-gating coordinate replay.

A signed permutation plus block-Hadamard is an orthogonal transform Q.
Use Q on a, rows of W^T and J. A full reconstruction is invariant;
compare equally sparse partial reconstructions. No checkpoint inference.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mechinterp.neuron_basis import native as N

SEED=20261125021
DRAWS=8
WIDTH=256
KS=(16,32,64,128,256)
EPS=1e-20
ROSTER_SHA="763ea63c8565245928a7dd59f7b56b7916fad6e2f8d3126075a801c56a36f224"


def hadamard(x,width):
    """Normalized Walsh-Hadamard along the first axis in fixed blocks."""
    x=np.array(x,dtype=np.float64,copy=True)
    n=x.shape[0]
    if width<1 or width&(width-1) or n%width:
        raise ValueError("invalid block width")
    rest=x.shape[1:]
    y=x.reshape(n//width,width,*rest)
    step=1
    while step<width:
        v=y.reshape(n//width,width//(2*step),2,step,*rest)
        a=v[:,:,0].copy(); b=v[:,:,1].copy()
        v[:,:,0]=a+b;v[:,:,1]=a-b
        step*=2
    return x/np.sqrt(width)


def rotations(n,width=WIDTH,draws=DRAWS):
    rng=np.random.default_rng(SEED)
    return [(rng.permutation(n),rng.choice(np.array([-1.,1.]),n),width)
            for _ in range(draws)]


def transform(x,rot):
    perm,sign,width=rot
    return hadamard(x[perm]*sign.reshape((-1,)+(1,)*(x.ndim-1)),width)


def loss(full,approx):
    en=float(full@full); err=float((full-approx)@(full-approx))
    return None if en+err<=EPS else err/(en+err)


def summarize_selection(a,w,j,rank,ks):
    # w is [n, hidden]; j is [n, digits].
    score=np.abs(a)*np.linalg.norm(w if rank=="physical" else j,axis=1)
    order=np.argsort(-score,kind="stable")
    p=a@w;t=a@j
    total=float(a@a)
    result={}
    for k in ks:
        ix=order[:k]
        pp=a[ix]@w[ix];tt=a[ix]@j[ix]
        result[str(k)]={"physical_loss":loss(p,pp),
            "transported_loss":loss(t,tt),
            "physical_error_l2":float(np.linalg.norm(p-pp)),
            "transported_error_l2":float(np.linalg.norm(t-tt)),
            "retained_coefficient_energy":None if total<=EPS else float(a[ix]@a[ix]/total)}
    return result


def compare(a,w,j,rots,rot_ws,seed,ks=KS):
    a=np.asarray(a,dtype=np.float64);j=np.asarray(j,dtype=np.float64)
    if len(a)!=len(w) or len(a)!=len(j) or not all(np.isfinite(x).all() for x in (a,w,j)):
        raise ValueError("bad receipt arrays")
    p=a@w;t=a@j; ae=float(a@a)
    out={}
    native={r:summarize_selection(a,w,j,r,ks) for r in ("physical","transported")}
    controls={r:[] for r in native}
    max_parity=0.
    for rot,rw in zip(rots,rot_ws):
        ra=transform(a,rot);rj=transform(j,rot)
        checks=[np.linalg.norm(ra@rw-p)/max(np.linalg.norm(p),1.),
                np.linalg.norm(ra@rj-t)/max(np.linalg.norm(t),1.),
                abs(ra@ra-ae)/max(ae,1.)]
        max_parity=max(max_parity,*map(float,checks))
        if max(checks)>1e-9:
            raise ArithmeticError("full-write coordinate parity failed")
        for rank in native:
            controls[rank].append(summarize_selection(ra,rw,rj,rank,ks))
    for rank in native:
        out[rank]={}
        for k,vals in native[rank].items():
            sub={}
            for name,val in vals.items():
                ctl=[r[k][name] for r in controls[rank] if r[k][name] is not None]
                avg=float(np.mean(ctl)) if ctl else None
                sub[name]={"native":val,"rotated_mean":avg,
                    "rotated_draws":ctl,
                    "advantage":None if val is None or avg is None else avg-val}
            out[rank][k]=sub
    rng=np.random.default_rng(seed)
    shuffled=[]
    # Shuffle ranking only; the actual native coefficients and write columns
    # remain fixed. Uniform random k-selection is the corresponding null.
    k=min(64,len(a))
    for _ in range(DRAWS):
        ix=rng.permutation(len(a))[:k]
        shuffled.append({"physical_loss":loss(p,a[ix]@w[ix]),
                         "transported_loss":loss(t,a[ix]@j[ix])})
    return {"rankings":out,"shuffled_k64":shuffled,
            "full_parity_max":max_parity,
            "physical_norm":float(np.linalg.norm(p)),
            "transported_norm":float(np.linalg.norm(t)),
            "coefficient_energy":ae,
            "physical_cancellation_ratio":float(np.linalg.norm(p)/max(np.sum(np.abs(a)*np.linalg.norm(w,axis=1)),1e-20)),
            "transported_cancellation_ratio":float(np.linalg.norm(t)/max(np.sum(np.abs(a)*np.linalg.norm(j,axis=1)),1e-20))}


def self_test():
    rng=np.random.default_rng(5)
    n=32;a=rng.normal(size=n);w=rng.normal(size=(n,16));j=rng.normal(size=(n,9))
    rs=rotations(n,8,3);rw=[transform(w,r) for r in rs]
    for r in rs:
        q=transform(np.eye(n),r)
        np.testing.assert_allclose(q@q.T,np.eye(n),atol=1e-12)
    x=compare(a,w,j,rs,rw,SEED,(4,n))
    assert x["full_parity_max"]<1e-12
    for rank in x["rankings"].values():
        for m in ("physical_loss","transported_loss"):
            assert rank[str(n)][m]["native"]<1e-20
            assert rank[str(n)][m]["rotated_mean"]<1e-20
    # Sparse planted coefficients; rotation spreads the selected coordinate.
    a=np.zeros(n);a[0]=1
    x=compare(a,w,j,rs,rw,SEED,(1,))
    assert x["rankings"]["physical"]["1"]["physical_loss"]["native"]==0.
    assert x["rankings"]["physical"]["1"]["physical_loss"]["rotated_mean"]>0
    print("NBR-02A coordinate parity and sparse positive control PASS")


def boot(v):
    if not v:return {"n":0,"mean":None,"ci95":None}
    v=np.asarray(v,dtype=np.float64)
    rng=np.random.default_rng(SEED)
    draws=v[rng.integers(len(v),size=(2000,len(v)))].mean(axis=1)
    return {"n":len(v),"mean":float(v.mean()),
            "ci95":np.quantile(draws,[.025,.975]).tolist()}


def summarize(rows):
    keys=rows[0]["calls"].keys()
    result={}
    for k in keys:
        result[k]={}
        for rank in ("physical","transported"):
            result[k][rank]={}
            for n in KS:
                res={}
                for name in ("physical_loss","transported_loss","retained_coefficient_energy"):
                    s=[b["calls"][k]["rankings"][rank][str(n)][name] for b in rows]
                    res[name]={m:boot([v[m] for v in s if v[m] is not None])
                               for m in ("native","rotated_mean","advantage")}
                result[k][rank][str(n)]=res
        for m in ("physical_loss","transported_loss"):
            vs=[b["calls"][k]["rankings"]["physical"]["64"][m]["native"] for b in rows]
            ctl=[[v[m] for v in b["calls"][k]["shuffled_k64"] if v[m] is not None] for b in rows]
            result[k]["shuffled_"+m]=boot([float(np.mean(c))-v for c,v in zip(ctl,vs) if c and v is not None])
    classes={}
    for s in (3,6,12):
        key=str(s)
        wrong=[b for b in rows if b["site_outcomes"][key]["wrong"]]
        classes[key]={"n":len(rows),"wrong":len(wrong),
                      "recovered_after":sum(b["site_outcomes"][key]["stable_after"] for b in wrong),
                      "not_recovered_after":sum(not b["site_outcomes"][key]["stable_after"] for b in wrong),
                      "exact_at_step":sum(b["site_outcomes"][key]["exact"] for b in rows),
                      "exact_at_16":sum(b["exact_at_16"] for b in rows)}
    return {"calls":result,"class_coverage":classes}


def run(roster,receipts,out):
    if out.exists():raise FileExistsError(out)
    if N.sha(roster)!=ROSTER_SHA:raise ValueError("roster hash")
    ro={r["id"]:r for r in json.loads(roster.read_text())}
    report=json.loads((receipts/"report.json").read_text())
    if report["checkpoint_sha256"]!=N.EXPECTED_SHA or report["roster_sha256"]!=ROSTER_SHA:
        raise ValueError("parent receipt identity mismatch")
    if len(report["boards"])!=144 or {b["id"] for b in report["boards"]}!=set(ro):
        raise ValueError("parent board coverage")
    torch.set_num_threads(4)
    model,digest=N.native_model().load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if digest!=N.EXPECTED_SHA:raise ValueError("checkpoint hash")
    w=model.layers[0].down_proj.weight.detach().numpy().T.astype(np.float64)
    del model
    rs=rotations(len(w));rw=[transform(w,r) for r in rs]
    # Also report column norm quantiles, independent of board labels.
    quant=lambda x:np.quantile(np.linalg.norm(x,axis=1),[0,.1,.5,.9,1]).tolist()
    rows=[]
    for idx,b in enumerate(report["boards"]):
        r=ro[b["id"]];p=receipts/b["receipt"]
        if N.sha(p)!=b["sha256"] or b["split"]!=r["split"]:
            raise ValueError("altered board receipt")
        blank=[i for i,c in enumerate(r["question"]) if c in ".0"]
        site=blank[len(blank)//2];assert b["site"]==site
        truth=np.array(list(r["answer"]),dtype=int)
        with np.load(p,allow_pickle=False) as z:
            pred=z["pred"]
            if pred.shape!=(16,81):raise ValueError("prediction shape")
            calls={}
            for step in (3,6,12):
                for role in ("L0","H0"):
                    k=f"s{step}_{role}"
                    calls[f"{step}/{role}"]=compare(z[k+"_a"],w,z[k+"_j"],rs,rw,SEED+idx)
            outcomes={str(s):{"wrong":bool(pred[s-1,site]!=truth[site]),
                "stable_after":bool(np.all(pred[s:,site]==truth[site])),
                "exact":bool(np.all(pred[s-1,blank]==truth[blank]))}
                for s in (3,6,12)}
            rows.append({"id":b["id"],"split":b["split"],"stratum":r["stratum"],
                "site_outcomes":outcomes,"exact_at_16":bool(np.all(pred[-1,blank]==truth[blank])),
                "calls":calls})
        if (idx+1)%24==0:print("NBR-02A",idx+1,"boards",flush=True)
    families={}
    for split in ("cal","eval","rep"):
        group=[b for b in rows if b["split"]==split]
        families[split]=summarize(group)
        families[split]["strata"]={s:summarize([b for b in group if b["stratum"]==s])
                                    for s in ("singles","stalled")}
    result={"schema":"NBR/gated-sparse-replay-v1","exploratory_only":True,
        "roster_sha256":ROSTER_SHA,"checkpoint_sha256":digest,
        "parent_report_sha256":N.sha(receipts/"report.json"),
        "seed":SEED,"draws":DRAWS,"block_width":WIDTH,"ks":list(KS),
        "native_column_norm_quantiles":quant(w),
        "rotated_column_norm_quantiles":[quant(v) for v in rw],
        "max_parity_error":max(c["full_parity_max"] for b in rows for c in b["calls"].values()),
        "families":families,"boards":rows}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,allow_nan=False,separators=(",",":"))+"\n")
    primary={s:{"class_coverage":f["class_coverage"],
        "physical":f["calls"]["3/L0"]["physical"]["64"]["physical_loss"],
        "transported":f["calls"]["3/L0"]["physical"]["64"]["transported_loss"]}
        for s,f in families.items()}
    print("NBR-02A primary",json.dumps(primary))
    print("NBR-02A max full parity",result["max_parity_error"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    for k in ("roster","receipts","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.roster and a.receipts and a.out:run(a.roster,a.receipts,a.out)
    else:p.error("supply --self-test or --roster --receipts --out")

if __name__=="__main__":main()
