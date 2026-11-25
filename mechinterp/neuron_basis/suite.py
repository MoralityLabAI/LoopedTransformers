"""NBR-02/03/04/06/08 engineering suite; NBR-05 cross-check via checkpoint flag.

Fixed calibration/eval/rep boards, no evaluation-based direction selection.
Real causal edits at the native down_proj output, before RMSNorm; local
Jacobian and finite differences, then signed future deduction outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from mechinterp.neuron_basis import native as N

SEED=2026112508
DOSES=(0.02,0.25)
BUDGET=64
K_PRUNE=8


def tail(model, h, l, x, site, gradient=True):
    """Actual L0 last-L write, same-cell local Jacobian, all board write samples."""
    slot=model.puzzle_emb_len+site
    data={}
    def pre(_m,args):
        data["a"]=args[0].detach()[0,slot].cpu().numpy()
    def post(_m,args,out):
        data["samples"]=out.detach()[0,model.puzzle_emb_len:].cpu().numpy()
        data["write"]=out.detach()[0,slot].cpu().numpy()
        if not gradient:
            return
        delta=torch.zeros_like(out[0,slot],requires_grad=True)
        data["delta"]=delta
        mask=out.new_zeros(1,out.shape[1],1); mask[:,slot]=1
        return out+mask*delta.view(1,1,-1)
    mod=model.layers[0].down_proj
    hp=mod.register_forward_pre_hook(pre); hh=mod.register_forward_hook(post)
    try:
        with torch.set_grad_enabled(gradient):
            l=model.run_l_level(l,h+x)
        # The module is also used in the H commit. Remove L hooks first.
    finally:
        hp.remove(); hh.remove()
    with torch.set_grad_enabled(gradient):
        h=model.run_l_level(h,l)
        y=model.digit_logits(h)[0,site]
        data["y"]=y.detach().cpu().numpy()
        if gradient:
            c=y-y.mean()
            data["g"]=np.stack([torch.autograd.grad(c[i],data["delta"],
                retain_graph=i+1<len(c))[0].detach().cpu().numpy()
                for i in range(len(c))],axis=1)
            del data["delta"]
    return h.detach(),l.detach(),data


@torch.no_grad()
def collect(model,row,trm):
    q=row["question"]; blanks=np.flatnonzero(np.array(list(q))==".")
    site=int(blanks[len(blanks)//2])
    x=model.encode_inputs(trm.encode_questions([q],"cpu"))
    h,l=model.init_states(1)
    preds=[]; anchor=None; d=None
    for s in range(1,17):
        if s==6:
            refh,refl=model.act_step(h,l,x)
            for hc in range(model.h_cycles):
                for lc in range(model.l_cycles):
                    if hc==model.h_cycles-1 and lc==model.l_cycles-1:
                        anchor=(h.clone(),l.clone(),x)
                        h,l,d=tail(model,h,l,x,site)
                    else:
                        l=model.run_l_level(l,h+x)
                if hc!=model.h_cycles-1:
                    h=model.run_l_level(h,l)
            torch.testing.assert_close(h,refh,atol=0,rtol=0)
            torch.testing.assert_close(l,refl,atol=0,rtol=0)
        else:
            h,l=model.act_step(h,l,x)
        preds.append(model.digit_logits(h)[0].argmax(-1).cpu().numpy()+1)
    pred=np.stack(preds)
    y=d["y"]; win,runner=np.argsort(-y,kind="stable")[:2]
    target=np.array(list(row["answer"]),dtype=int)
    return {**row,"site":site,"blanks":blanks,"target":target,"pred":pred,
            "anchor":anchor,"d":d,"contrast":d["g"][:,win]-d["g"][:,runner],
            "eligible":bool(pred[5,site]!=target[site]),
            "recovered":bool(np.all(pred[6:,site]==target[site])),
            "failed":bool(np.any(pred[-1,blanks]!=target[blanks]))}


def fit_bases(cal,w):
    """All learned directions fit on calibration MLP writes only."""
    rng=np.random.default_rng(SEED)
    d,m=w.shape
    b=min(BUDGET,d,m)
    samples=np.concatenate([r["d"]["samples"] for r in cal]).astype(np.float64)
    mu=samples.mean(0)
    centered=samples-mu
    _,_,vh=np.linalg.svd(centered,full_matrices=False)
    pca=vh[:b].T
    rand=rng.normal(size=(d,b)); rand/=np.linalg.norm(rand,axis=0)
    ids=np.sort(rng.choice(m,b,replace=False))
    nw=w[:,ids]; norms=np.linalg.norm(nw,axis=0)
    if np.any(norms<=N.EPS):
        raise ArithmeticError("zero native direction in fixed candidate set")
    native=nw/norms
    # SAE on the actual residual write; unit decoder columns, L1 sparse code.
    torch.manual_seed(SEED)
    scale=max(float(np.sqrt(np.mean(centered**2))),1e-8)
    z=torch.tensor(centered/scale,dtype=torch.float32)
    enc=nn.Linear(d,b)
    dec=nn.Linear(b,d,bias=False)
    opt=torch.optim.Adam([*enc.parameters(),*dec.parameters()],lr=0.003)
    gen=torch.Generator().manual_seed(SEED)
    losses=[]
    for _ in range(120):
        idx=torch.randint(len(z),(min(128,len(z)),),generator=gen)
        batch=z[idx]; a=torch.relu(enc(batch))
        unit=dec.weight/dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8)
        recon=a@unit.T
        loss=(recon-batch).square().mean()+0.02*a.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        sae=(dec.weight/dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8)).numpy().copy()
        a=torch.relu(enc(z)); recon=a@torch.tensor(sae).T
        mse=float((recon-z).square().mean())
        density=float((a>0).float().mean())
    return {"dirs":{"native":native,"activation":native,"static":native,
                    "pca":pca,"random":rand,"sae":sae},
            "ids":ids,"norms":norms,"mu":mu,"scale":scale,"enc":enc,
            "fit":{"budget":b,"sae_loss_first":losses[0],"sae_loss_last":losses[-1],
                   "sae_cal_normalized_mse":mse,"sae_cal_density":density,
                   "n_cal_tokens":len(z)}}


def features(r,bases,head):
    z=r["d"]["write"]-bases["mu"]
    a=r["d"]["a"][bases["ids"]]*bases["norms"]
    g=r["contrast"]
    with torch.no_grad():
        sa=torch.relu(bases["enc"](torch.tensor(z/bases["scale"],dtype=torch.float32))).numpy()*bases["scale"]
    y=r["d"]["y"]; win,runner=np.argsort(-y,kind="stable")[:2]
    scores={}
    for arm,dirs in bases["dirs"].items():
        coeff=a if arm in ("native","activation","static") else sa if arm=="sae" else z@dirs
        effect=(head[win]-head[runner])@dirs if arm=="static" else g@dirs
        scores[arm]=coeff if arm=="activation" else coeff*effect
    return scores


def fit_detector(cal,arms,label,eligible):
    rs=[r for r in cal if eligible(r)]
    if not rs:
        return {a:None for a in arms}
    y=[r[label] for r in rs]
    return {a:N.pick(y,np.stack([r["features"][a] for r in rs])) for a in arms}


def evaluate_detector(rows,selected,label,eligible):
    rs=[r for r in rows if eligible(r)]
    y=np.array([r[label] for r in rs],dtype=bool)
    out={"n":len(y),"positive":int(y.sum()),"arms":{}}
    scores={}
    for arm,s in selected.items():
        if s is None or not len(rs):
            out["arms"][arm]=None; continue
        z=np.array([r["features"][arm][s["neuron"]]*s["sign"] for r in rs])
        scores[arm]=z
        a=N.auc(y,z)
        out["arms"][arm]=None if a is None else float(a)
    rng=np.random.default_rng(SEED)
    out["paired_vs_native"]={}
    if "native" in scores and y.any() and not y.all():
        inds=rng.integers(len(y),size=(2000,len(y)))
        inds=[i for i in inds if y[i].any() and not y[i].all()]
        for arm,z in scores.items():
            if arm=="native": continue
            ds=[float(N.auc(y[i],scores["native"][i])-N.auc(y[i],z[i])) for i in inds]
            out["paired_vs_native"][arm]={"delta":out["arms"]["native"]-out["arms"][arm],
                "ci95":np.quantile(ds,[.025,.975]).tolist() if ds else None,
                "discarded":2000-len(ds)}
    return out


def schedule(pred, target, blanks):
    good=pred[:,blanks]==target[blanks]
    # Count backwards from ACT16 until first incorrect.
    stable=np.cumprod(good[::-1],axis=0).sum(0)
    return 17-stable


@torch.no_grad()
def interventions(model,r,bases,selected,prune_ids,random_prune):
    """Same-forward batch, equal residual-write norm for signed local edits.

    Pruning is an independent equal-count intervention, not equal-norm.
    """
    h,l,x=r["anchor"]
    slot=model.puzzle_emb_len+r["site"]
    variants=[{"arm":"clean"},{"arm":"sham"}]
    dirs=[np.zeros(model.hidden_size),np.zeros(model.hidden_size)]
    # Calibration-selected feature IDs. No eval labels in edit construction.
    for arm,s in selected.items():
        if s is None: continue
        v=bases["dirs"][arm][:,s["neuron"]]
        for dose in DOSES:
            for sign in (-1,1):
                variants.append({"arm":arm,"dose":dose,"sign":sign})
                dirs.append(v*dose*sign)
    variants += [{"arm":"prune"},{"arm":"prune_random"}]
    dirs += [np.zeros(model.hidden_size),np.zeros(model.hidden_size)]
    n=len(variants)
    h,l,x=[z.repeat(n,1,1) for z in (h,l,x)]
    shifts=torch.tensor(np.stack(dirs),dtype=h.dtype)
    energy=[0.,0.]
    mod=model.layers[0].down_proj
    def shift(_m,_a,out):
        out=out.clone(); out[:,slot]+=shifts
        return out
    def prune(_m,args):
        a=args[0].clone()
        for j,ids in enumerate((prune_ids,random_prune)):
            row=n-2+j
            ix=torch.as_tensor(ids)
            removed=a[row,:,ix].clone()
            w=mod.weight[:,ix]
            energy[j]+=float((removed@w.T).square().sum())
            a[row,:,ix]=0
        return (a,)
    hk=mod.register_forward_hook(shift)
    try:
        l=model.run_l_level(l,h+x)
    finally:
        hk.remove()
    h=model.run_l_level(h,l)
    logits=model.digit_logits(h)[:,r["site"]].cpu().numpy()
    ps=[np.broadcast_to(r["pred"][:5],(n,5,81)).copy(),
        (model.digit_logits(h).argmax(-1).cpu().numpy()+1)[:,None,:]]
    hk=mod.register_forward_pre_hook(prune)
    try:
        for _ in range(7,17):
            h,l=model.act_step(h,l,x)
            ps.append((model.digit_logits(h).argmax(-1).cpu().numpy()+1)[:,None,:])
    finally:
        hk.remove()
    pred=np.concatenate(ps,axis=1)
    if not np.array_equal(pred[0],pred[1]) or not np.array_equal(logits[0],logits[1]):
        raise ArithmeticError("same-batch sham differs")
    base=schedule(pred[0],r["target"],r["blanks"])
    out=[]
    for k,v in enumerate(variants):
        t=schedule(pred[k],r["target"],r["blanks"])
        measured=logits[k]-logits[0]; measured-=measured.mean()
        predicted=np.array(dirs[k])@r["d"]["g"]
        out.append({**v,"vdd":int(np.abs(t-base).sum()),
                    "signed_schedule_improvement":int((base-t).sum()),
                    "final_blank_correct":int((t<17).sum()),
                    "exact":bool(np.all(t<17)),
                    "local_logit_delta":measured.tolist(),
                    "linear_prediction":predicted.tolist(),
                    "linear_error":float(np.linalg.norm(measured-predicted)),
                    "write_l2":float(np.linalg.norm(dirs[k]))})
    return {"variants":out,"removed_write_energy":energy,
            "batch_clean_matches_single":bool(np.array_equal(pred[0],r["pred"])),
            "predictions":pred}


def graph_test(model,r,src_ids,dst_ids):
    """L0 -> L1 same-token edge Jacobian; compare with finite difference.

    Edges are local; no claim about full recurrent circuit identification.
    """
    if len(model.layers)<2:
        return None
    h,l,x=r["anchor"]; slot=model.puzzle_emb_len+r["site"]
    left=model.layers[0].down_proj; right=model.layers[1].down_proj
    src=np.asarray(src_ids); dst=np.asarray(dst_ids)
    def one(edit=None,grad=False):
        saved={}
        def pre(_m,args):
            a=args[0]; off=a.new_zeros(a.shape[-1])
            if grad:
                off=torch.zeros_like(off,requires_grad=True)
                saved["delta"]=off
            if edit is not None:
                off[int(edit[0])]=float(edit[1])
            mask=a.new_zeros(1,a.shape[1],1);mask[:,slot]=1
            return (a+mask*off.view(1,1,-1),)
        def observe(_m,args):
            saved["dst"]=args[0][0,slot,torch.as_tensor(dst)]
        handles=[left.register_forward_pre_hook(pre),right.register_forward_pre_hook(observe)]
        try:
            with torch.set_grad_enabled(grad):
                model.run_l_level(l,h+x)
                v=saved["dst"]
                if grad:
                    js=[torch.autograd.grad(v[i],saved["delta"],retain_graph=i+1<len(v))[0]
                        [torch.as_tensor(src)].detach().numpy() for i in range(len(v))]
                    return np.stack(js)
                return v.detach().numpy()
        finally:
            for hk in handles: hk.remove()
    jac=one(grad=True)
    eps=0.002
    fd=np.stack([(one((i,eps))-one((i,-eps)))/(2*eps) for i in src],axis=1)
    a=jac.ravel(); b=fd.ravel()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return {"source":src.tolist(),"target":dst.tolist(),
            "jacobian":jac.tolist(),"finite_difference":fd.tolist(),
            "cosine":float(a@b/denom) if denom>1e-12 else None,
            "relative_error":float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))}


def run(roster,out,checkpoint,toy=False):
    rows=N.load_roster(roster)
    if set(r["split"] for r in rows)!={"cal","eval","rep"}:
        raise ValueError("all three splits required")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4 if not toy else 1);torch.manual_seed(SEED)
    trm=N.native_model()
    if toy:
        model=trm.PureSudokuTRM(hidden_size=16,num_heads=2,l_layers=2,h_cycles=1,l_cycles=2)
        digest="random-toy"
    else:
        model,digest=trm.load_trm_checkpoint(checkpoint,"cpu",dtype=torch.float32)
    model.float().eval().requires_grad_(False)
    w=model.layers[0].down_proj.weight.detach().numpy().astype(np.float64)
    head=model.lm_head.weight[2:11].detach().numpy()
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        data=[collect(model,r,trm) for r in rows]
        cal=[r for r in data if r["split"]=="cal"]
        bases=fit_bases(cal,w)
        for r in data:
            r["features"]=features(r,bases,head)
        arms=list(bases["dirs"])
        recovery=fit_detector(cal,arms,"recovered",lambda r:r["eligible"])
        failure=fit_detector(cal,arms,"failed",lambda r:True)
        # Fallback for class-degenerate engineering roster: unsupervised
        # calibration write-score magnitude, explicitly distinct from detector.
        selected={}
        for arm in arms:
            s=recovery[arm]
            if s is None:
                i=int(np.argmax(np.mean([np.abs(r["features"][arm]) for r in cal],axis=0)))
                selected[arm]={"neuron":i,"sign":1,"source":"cal_unsupervised"}
            else:
                selected[arm]={**s,"source":"cal_recovery"}
        importance=np.mean([np.abs(r["d"]["a"]*(w.T@r["contrast"])) for r in cal],axis=0)
        prune_ids=np.argsort(-importance,kind="stable")[:K_PRUNE]
        norm=np.linalg.norm(w,axis=0)
        rng=np.random.default_rng(SEED)
        available=set(range(w.shape[1]))-set(prune_ids)
        random_prune=[]
        for i in prune_ids:
            near=sorted(available,key=lambda j:(abs(norm[j]-norm[i]),j))[:8]
            j=int(rng.choice(near));available.remove(j);random_prune.append(j)
        # Graph targets fixed by index, selected before any eval board.
        dst_ids=np.sort(rng.choice(model.layers[1].down_proj.in_features,4,replace=False))
        src_ids=prune_ids[:4]
        report={"schema":"NBR/suite-engineering-v1","engineering_only":True,
                "checkpoint":checkpoint,"checkpoint_sha256":digest,"toy":toy,
                "platform":platform.platform(),"torch":torch.__version__,
                "numpy":np.__version__,"seed":SEED,"roster_sha256":N.sha(roster),
                "fit":bases["fit"],"native_ids":bases["ids"].tolist(),
                "directions_selected":selected,"recovery_selected":recovery,
                "failure_selected":failure,"prune_ids":prune_ids.tolist(),
                "random_prune_ids":random_prune,"families":{}}
        # Save the learned basis and detector choices, independent of held-out edits.
        arrays={"mu":bases["mu"],"sae_scale":np.array(bases["scale"]),
            "sae_encoder_weight":bases["enc"].weight.detach().numpy(),
            "sae_encoder_bias":bases["enc"].bias.detach().numpy()}
        arrays.update({f"dir_{a}":v for a,v in bases["dirs"].items()})
        np.savez_compressed(out/"basis.npz",**arrays)
        report["basis_sha256"]=N.sha(out/"basis.npz")
        for split in ("eval","rep"):
            rs=[r for r in data if r["split"]==split]
            fam={"recovery":evaluate_detector(rs,recovery,"recovered",lambda r:r["eligible"]),
                 "failure":evaluate_detector(rs,failure,"failed",lambda r:True),"boards":[]}
            for idx,r in enumerate(rs):
                iv=interventions(model,r,bases,selected,prune_ids,random_prune)
                graph=graph_test(model,r,src_ids,dst_ids)
                p=out/f"{split}_{idx:03d}.npz"
                np.savez_compressed(p,pred=iv.pop("predictions"),clean=r["pred"],
                    activation=r["d"]["a"],jacobian=r["d"]["g"],write=r["d"]["write"],
                    **{f"score_{a}":v for a,v in r["features"].items()})
                fam["boards"].append({"id":r["id"],"site":r["site"],"sha256":N.sha(p),
                    "receipt":p.name,"eligible":r["eligible"],"recovered":r["recovered"],
                    "failed":r["failed"],"causal":iv,"graph":graph})
                print("NBR suite",split,r["id"],flush=True)
            report["families"][split]=fam
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roster",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--checkpoint",choices=("gaoxin492","yagizdevre"),default="gaoxin492")
    p.add_argument("--toy",action="store_true")
    a=p.parse_args();run(a.roster,a.out,a.checkpoint,a.toy)


if __name__=="__main__":
    main()
