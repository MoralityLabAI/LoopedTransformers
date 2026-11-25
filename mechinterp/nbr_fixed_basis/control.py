"""NBR-01B: shared physical direction vs native neuron identity.

Reanalyze native.py receipts without new model inference. Same coefficient
vector is used at both ACTs for each random mixture; output direction
covariance matches that of unit-normalized native output columns in
expectation. Pilot reanalysis is exploratory. Prospective test needs a
fresh source manifest and separate registration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mechinterp.neuron_basis import native as N

SEED=202611251031
DRAWS=8
MIX_PER_DRAW=256

def cosine(a,b):
    u,va=N.unit_rows(a);v,vb=N.unit_rows(b)
    ok=va&vb
    return (float((u[ok]*v[ok]).sum(1).mean()),int(ok.sum())) if ok.any() else (None,0)

def prepare(w,head):
    norms=np.linalg.norm(w,axis=0)
    if np.any(norms<=N.EPS):
        raise ValueError("zero decoder column")
    unit=w/norms[None,:]
    static=unit.T@head.T
    rng=np.random.default_rng(SEED)
    # A coefficient vector is fixed across every ACT and board. Scaling by
    # actual physical norm cancels in cosine, but records correct geometry.
    coeffs=[]
    for _ in range(DRAWS):
        c=rng.normal(size=(MIX_PER_DRAW,w.shape[1]))/np.sqrt(w.shape[1])
        physical=c@unit.T
        n=np.linalg.norm(physical,axis=1)
        if np.any(n<=N.EPS):
            raise ValueError("degenerate mixed physical write")
        coeffs.append(c/n[:,None])
    return norms,static,coeffs

def compare(j1,j2,norms,static,coeffs):
    a=j1/norms[:,None];b=j2/norms[:,None]
    out={}
    for mode in ("raw","residual"):
        x,y=(a,b) if mode=="raw" else (N.residualize(a,static),N.residualize(b,static))
        same,n=cosine(x,y)
        control=[];coverage=[]
        for c in coeffs:
            ma,mb=c@a,c@b
            if mode=="residual":
                ms=c@static
                ma,mb=N.residualize(ma,ms),N.residualize(mb,ms)
            q,k=cosine(ma,mb)
            if q is not None:
                control.append(q);coverage.append(k)
        out[mode]={"native":same,"native_n":n,
                   "fixed_mix":float(np.mean(control)) if control else None,
                   "mix_draws":control,"mix_valid_counts":coverage,
                   "advantage":None if same is None or not control else same-float(np.mean(control))}
    return out

def self_test():
    rng=np.random.default_rng(9)
    w=rng.normal(size=(16,64));head=rng.normal(size=(9,16))
    head-=head.mean(0)
    g=rng.normal(size=(16,9));g-=g.mean(1,keepdims=True)
    norm,static,c=prepare(w,head)
    j=w.T@g
    same=compare(j,j,norm,static,c)
    for v in same.values():
        assert abs(v["native"]-1)<1e-10 and abs(v["fixed_mix"]-1)<1e-10
        assert abs(v["advantage"])<1e-10
    q=rng.normal(size=(16,9));q-=q.mean(1,keepdims=True)
    for cf in c:
        np.testing.assert_allclose(cf@(j/norm[:,None]),(cf@(w/norm[None,:]).T)@g)
        np.testing.assert_allclose(cf@((w.T@q)/norm[:,None]),(cf@(w/norm[None,:]).T)@q)
    print("NBR-01B fixed basis toy PASS")

def run(receipts,out):
    if out.exists():
        raise FileExistsError(out)
    r=json.loads((receipts/"report.json").read_text())
    if r["checkpoint_sha256"]!=N.EXPECTED_SHA:
        raise ValueError("checkpoint mismatch")
    torch.set_num_threads(4)
    m,digest=N.native_model().load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    assert digest==N.EXPECTED_SHA
    head=m.lm_head.weight[2:11].detach().numpy().astype(np.float64)
    head-=head.mean(0)
    geometry={}
    for li,layer in enumerate(m.layers):
        w=layer.down_proj.weight.detach().numpy().astype(np.float64)
        geometry[li]=prepare(w,head)
    boards=[]
    for row in r["boards"]:
        p=receipts/row["receipt"]
        if N.sha(p)!=row["sha256"]:
            raise ValueError("altered native receipt")
        with np.load(p,allow_pickle=False) as z:
            pred=z["pred"]
            stats={}
            for s,t in ((3,6),(6,12)):
                for role in ("L","H"):
                    for li,g in geometry.items():
                        key=f"{role}{li}"
                        stats[f"{s}-{t}/{key}"]=compare(
                            z[f"s{s}_{key}_j"].astype(np.float64),
                            z[f"s{t}_{key}_j"].astype(np.float64),*g)
            boards.append({"id":row["id"],"split":row["split"],
                "act3_to_6_digit_changes":int((pred[2]!=pred[5]).sum()),
                "stats":stats})
    families={}
    for s in ("cal","eval","rep"):
        rows=[b for b in boards if b["split"]==s]
        families[s]={}
        for key in boards[0]["stats"]:
            families[s][key]={}
            for mode in ("raw","residual"):
                vs=[b["stats"][key][mode] for b in rows]
                families[s][key][mode]={k:N.bootstrap_mean([v[k] for v in vs if v[k] is not None])
                    for k in ("native","fixed_mix","advantage")}
    report={"schema":"NBR/fixed-basis-engineering-v1","engineering_only":True,
        "checkpoint_sha256":digest,"parent_report_sha256":N.sha(receipts/"report.json"),
        "seed":SEED,"draws":DRAWS,"mix_per_draw":MIX_PER_DRAW,
        "boards":boards,"families":families}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("NBR-01B primary:",json.dumps({s:v["3-6/L0"] for s,v in families.items()}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--receipts",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.receipts and a.out:run(a.receipts,a.out)
    else:p.error("supply --self-test or --receipts and --out")

if __name__=="__main__":main()
