"""ROC-05B: frozen-query recurrent midpoint, physical source/input rule interventions.

Scientific registration: configs/roc05b_matched_rule_causal_v1.json,
commit ef18aef009e72c0b0e75de9331d6135db7977d98 BEFORE these measurements.
Only exposed ROC05A development records. ROC05 reserved tables ungenerated;
the original ROC02 family 6/7 source files are NEVER accessed.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from research_gym.analysis.roc03b_query_table_memory import QueryConditionedTableMemory
from research_gym.analysis.roc02_tied_train import rows,tensors
from research_gym.analysis.roc05_permutation_source import swapped_row_counterfactual

SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
RADII=(.05,.10,.20)
ROUTES=("full_legal_unmatched","input_soft","input_induced_hidden",
        "supervised_observed","supervised_dark","pca_observed",
        "random_observed","isotropic")
N=256
WIDTH=128
RANK=7
SRC={"low_affine4":"train_low4.jsonl",
     "high_permutation512":"train_high512.jsonl","development":"development.jsonl"}

def sha(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked_files(source_dir,config):
    d={}
    for key,name in SRC.items():
        file=source_dir/name
        if sha(file)!=config["parent"]["exact_source_sha256"][
            {"low_affine4":"train_low4","high_permutation512":"train_high512","development":"development"}[key]
        ]:
            raise RuntimeError("source SHA mismatch: "+name)
        d[key]=rows(file)
    if len(d["development"])!=N or len(d["low_affine4"])!=1024 or len(d["high_permutation512"])!=1024:
        raise AssertionError("wrong source cardinality")
    if len({x["rule_table_id"] for x in d["development"]})!=N:
        raise AssertionError("dev table identity duplicate")
    if any(x["family"]!="roc05_dev_unseen" for x in d["development"]):
        raise AssertionError("unexpected exposed family")
    return d

def load_model(seed,arm,ckdir,cfg,source_dir):
    path=ckdir/f"roc05_{arm}_S{seed}_step1920.pt"
    actual=sha(path)
    expected=cfg["parent"]["frozen_1920step_checkpoint_sha256"][str(seed)][arm]
    if actual!=expected:
        raise RuntimeError("original first-run frozen model SHA mismatch "+path.name)
    v=torch.load(path,map_location="cpu",weights_only=True)
    if (v["seed"],v["arm"],v["optimizer_steps"])!=(seed,arm,1920):
        raise RuntimeError("original model identity mismatch")
    for name,record in SRC.items():
        if v["source_sha256"][name]!=sha(source_dir/record):
            raise RuntimeError("original model source file hash mismatch "+name)
    m=QueryConditionedTableMemory()
    m.load_state_dict(v["state_dict"],strict=True)
    m.eval()
    return m,actual

def query_ids(x,mask):
    query_mask=x[:,:,3]*mask
    if not torch.all(query_mask.sum(dim=1)==1):
        raise AssertionError("exactly one visible query")
    cue=(x[:,:,71:79]*query_mask[:,:,None]).sum(dim=1)
    if not torch.all(cue.sum(dim=1)==1):
        raise AssertionError("invalid final onehot query")
    return cue.argmax(dim=1)

@torch.no_grad()
def midpoint(model,x,mask):
    query_mask=x[:,:,3]*mask
    memory_mask=mask*(1-query_mask)
    h=x.new_zeros((x.shape[0],WIDTH))
    for t in range(x.shape[1]):
        h=h+memory_mask[:,t,None]*(.25*model.event_embed(x[:,t,:])+model.delta(h))
    for _ in range(4):
        h=h+model.delta(h)
    return h

@torch.no_grad()
def suffix_logits(model,h,q):
    h=h.float()
    for _ in range(4):
        h=h+model.delta(h)
    tables=model.table_head(h).reshape(-1,8,8)
    idx=torch.arange(len(h),device=h.device)
    return tables[idx,q].double()

def observer_bases(h,y):
    x=h.double()
    x=x-x.mean(dim=0)
    z=F.one_hot(y,8).double()
    z=z-z.mean(dim=0)
    cross=x.T@z
    _,s,_=torch.linalg.svd(cross,full_matrices=False)
    if float(s[6]) < 1e-9:
        raise AssertionError("supervised 7th singular value vanishes")
    sup=torch.linalg.svd(cross,full_matrices=False).U[:,:7]
    pca=torch.linalg.svd(x,full_matrices=False).Vh.T[:,:7]
    eye=torch.eye(7,dtype=torch.float64)
    for u in (sup,pca):
        if float(torch.linalg.norm(u.T@u-eye))>1e-8:
            raise AssertionError("observer basis not orthonormal")
    return {"supervised":sup,"pca":pca},{"supervised_s7":float(s[6]),
        "supervised_s8":float(s[7])}

def suffix_tangent(model,h):
    z=h.detach().double().clone()
    w1=model.block_in.weight.detach().double()
    b1=model.block_in.bias.detach().double()
    w2=model.block_out.weight.detach().double()
    b2=model.block_out.bias.detach().double()
    I=torch.eye(WIDTH,dtype=torch.float64)
    T=I.clone()
    for _ in range(4):
        a=torch.tanh(w1@z+b1)
        T=(I+.25*((w2*(1-a*a)[None,:])@w1))@T
        z=z+.25*(w2@a+b2)
    return T

def geom_metric(T,R,U):
    D=T-torch.eye(WIDTH,dtype=torch.float64)
    dn=float(torch.linalg.vector_norm(D))
    if dn<1e-12:
        return None
    UD=U.T@D
    A=UD@U
    DU=D@U
    aa=float(torch.sum(A*A))
    bb=max(0.,float(torch.sum(UD*UD))-aa)
    cc=max(0.,float(torch.sum(DU*DU))-aa)
    g=R@T
    h=float((g*g).sum())
    cap=float(((g@U)**2).sum()/h) if h>1e-15 else None
    return {"a":math.sqrt(aa)/dn,"b":math.sqrt(bb)/dn,
            "c":math.sqrt(cc)/dn,"s":cap,
            "A_frob":math.sqrt(aa),"B_frob":math.sqrt(bb),
            "C_frob":math.sqrt(cc),"D_frob":dn,
            "H_trace":h}

def keyed_seed(*args):
    token="|".join(map(str,args))
    value=hashlib.sha256(token.encode()).digest()
    return int.from_bytes(value[:8],"big")%(2**63-1)

def rand_basis(cksha,shadow,random_seed):
    g=torch.Generator(device="cpu").manual_seed(
        keyed_seed(cksha,"roc05b",shadow,7,random_seed))
    mat=torch.randn(WIDTH,7,dtype=torch.float64,generator=g)
    return torch.linalg.qr(mat,mode="reduced")[0]

def empirical(x,a,low):
    if x is None or len(a)!=128 or any(v is None for v in a):
        return None
    return (1+sum(v<=x if low else v>=x for v in a))/129.

def holm(p):
    valid=sorted((v,i) for i,v in enumerate(p) if v is not None)
    q=[None]*len(p);largest=0.
    for place,(v,i) in enumerate(valid):
        largest=max(largest,min(1.,(6-place)*v))
        q[i]=largest
    return q

def state_cards(model,h,q,bases,cksha,shadow):
    T=suffix_tangent(model,h)
    R=model.table_head.weight.detach().double().reshape(8,8,128)[int(q)]
    randoms=[geom_metric(T,R,rand_basis(cksha,shadow,j))
             for j in range(941001,941129)]
    if any(x is None for x in randoms):
        raise AssertionError("invalid random null dynamics")
    keys=("a","b","c","s")
    arr={k:sorted(r[k] for r in randoms) for k in keys}
    med={k:(v[63]+v[64])/2 if len(v)==128 and all(x is not None for x in v) else None
         for k,v in arr.items()}
    cards=[]
    for name,U in bases.items():
        m=geom_metric(T,R,U)
        if m is None:
            raise AssertionError("undefined computed-change state")
        p=[empirical(m["a"],arr["a"],True),
           empirical(m["b"],arr["b"],True),
           empirical(m["c"],arr["c"],True),
           empirical(m["b"],arr["b"],False),
           empirical(m["s"],arr["s"],False),
           empirical(m["s"],arr["s"],True)]
        qv=holm(p)
        ra=m["a"]/med["a"] if med["a"] else None
        rb=m["b"]/med["b"] if med["b"] else None
        rc=m["c"]/med["c"] if med["c"] else None
        low_s=m["s"]/med["s"] if med["s"] and m["s"] is not None else None
        high_s=(m["s"]-med["s"])/(1-med["s"]) if med["s"] is not None and 0<med["s"]<1 and m["s"] is not None else None
        def ok(i,cond):
            return qv[i] is not None and qv[i]<=.05 and cond
        near=lambda v:v is not None and v<=.2
        tags=[]
        if ok(1,near(rb)):tags.append("closed_quotient")
        if ok(2,near(rc)):tags.append("invariant_subspace")
        if ok(0,near(ra)) and ok(1,near(rb)):
            if ok(4,high_s is not None and high_s>=.8):tags.append("latched_memory")
            if ok(5,near(low_s)):tags.append("irrelevant")
        if ok(3,rb is not None and rb>=1.2) and ok(0,near(ra)) and ok(2,near(rc)):
            tags.append("output_register")
        cards.append({"basis":name,"rank":7,"shadow_index":shadow,
              "metric":m,"null_medians":med,"ratios":{"a":ra,"b":rb,"c":rc,
                       "s_low":low_s,"s_high":high_s},
              "one_sided_p":p,"holm_q":qv,"tags":tags or ["unclassified"],
              "null_distinct":128,"calibration_only":True})
    return cards

def normalize(v,r):
    n=torch.linalg.vector_norm(v,dim=1)
    good=torch.isfinite(n)&(n>1e-12)
    shifted=r*v/n.clamp_min(1e-12)[:,None]
    return shifted,good

@torch.no_grad()
def soft_input_states(model,x,changed,mask,h0,hfull):
    n=len(x)
    dx=changed-x
    norms=torch.linalg.vector_norm((hfull.double()-h0.double()),dim=1)
    bracket={}
    for r in RADII:
        bracket[r]=[torch.zeros(n,dtype=torch.float64),
                    torch.ones(n,dtype=torch.float64),
                    torch.zeros(n,dtype=torch.bool)]
    for j in range(1,65):
        t=j/64
        h=midpoint(model,x+t*dx,mask)
        dis=torch.linalg.vector_norm(h.double()-h0.double(),dim=1)
        for r in RADII:
            lo,hi,found=bracket[r]
            fresh=(~found)&(dis>=r)
            lo[fresh]=(j-1)/64
            hi[fresh]=t
            found |= fresh
    out={}
    for r in RADII:
        lo,hi,found=bracket[r]
        for _ in range(24):
            mid=(lo+hi)/2
            h=midpoint(model,x+mid[:,None,None].float()*dx,mask)
            dis=torch.linalg.vector_norm(h.double()-h0.double(),dim=1)
            crossed=dis>=r
            hi=torch.where(crossed,mid,hi)
            lo=torch.where(crossed,lo,mid)
        h=midpoint(model,x+hi[:,None,None].float()*dx,mask).double()
        norm=torch.linalg.vector_norm(h-h0.double(),dim=1)
        err=(norm-r).abs()
        good=found&(norms>=r)&(err<=2e-5)&torch.isfinite(err)
        out[r]={"mid":h,"eligible":good,"actual_l2":norm,
                "max_radius_error":float(err[good].max()) if bool(good.any()) else None,
                "ineligible_count":int((~good).sum())}
    return out

def seeded_random_vectors(cksha,record_ids,r,U):
    observed=[];isotropic=[]
    for identity in record_ids:
        a=torch.Generator(device="cpu").manual_seed(keyed_seed(cksha,identity,r,"rand_observed"))
        b=torch.Generator(device="cpu").manual_seed(keyed_seed(cksha,identity,r,"rand_isotropic"))
        observed.append(U@torch.randn(7,dtype=torch.float64,generator=a))
        isotropic.append(torch.randn(128,dtype=torch.float64,generator=b))
    return torch.stack(observed),torch.stack(isotropic)

def target_gain(logits,base,y,yp):
    a=logits.gather(1,yp[:,None])[:,0]-logits.gather(1,y[:,None])[:,0]
    b=base.gather(1,yp[:,None])[:,0]-base.gather(1,y[:,None])[:,0]
    return a-b

def evaluate_model(seed,arm,source,checkpoint,cfg,out):
    model,cksha=load_model(seed,arm,checkpoint,cfg,source)
    cases=checked_files(source,cfg)
    train=tensors(cases[arm])
    dev=tensors(cases["development"])
    changed=[swapped_row_counterfactual(r) for r in cases["development"]]
    altered=tensors(changed)
    if not torch.equal(altered[1],dev[1]) or not torch.equal(altered[2],(dev[2]+4)%8):
        # Permutation row swap changes the queried output; it generally is
        # NOT a fixed +4 label shift. Recheck exact row-wise difference below.
        if not torch.equal(altered[1],dev[1]) or bool(torch.any(altered[2]==dev[2])):
            raise AssertionError("legal counterfactual label/schedule invalid")
    with torch.no_grad():
        mid_train=midpoint(model,train[0],train[1])
        h0=midpoint(model,dev[0],dev[1])
        hfull=midpoint(model,altered[0],altered[1])
        q=query_ids(dev[0],dev[1])
        original=suffix_logits(model,h0,q)
        orig_native=model(dev[0],dev[1])[0].double()
        err=float((original-orig_native).abs().max())
        if err>1e-5:
            raise AssertionError(f"native midpoint/suffix parity failed {err}")
        legal=suffix_logits(model,hfull,q)
        basis,sv=observer_bases(mid_train,train[2])
        cards=[]
        for idx in (0,1):
            cards.extend(state_cards(model,h0[idx],q[idx],basis,cksha,idx))
        soft=soft_input_states(model,dev[0],altered[0],dev[1],h0,hfull)
        full_delta=(hfull-h0).double()
        full_norm=torch.linalg.vector_norm(full_delta,dim=1)
        project=(full_delta@basis["supervised"])@basis["supervised"].T
        dark=full_delta-project
        pca=(full_delta@basis["pca"])@basis["pca"].T
        y=dev[2];yp=altered[2]
        base_correct=int((original.argmax(-1)==y).sum())
        legal_correct=int((legal.argmax(-1)==yp).sum())
        records=[]
        primary={}
        for r in RADII:
            observed_rand,isotropic=seeded_random_vectors(
                cksha,[x["id"] for x in cases["development"]],r,basis["supervised"])
            directions={"input_induced_hidden":full_delta,
                "supervised_observed":project,"supervised_dark":dark,
                "pca_observed":pca,"random_observed":observed_rand,"isotropic":isotropic}
            logit_by_arm={
                "full_legal_unmatched":(legal,torch.ones(N,dtype=torch.bool),full_norm),
                "input_soft":(suffix_logits(model,soft[r]["mid"],q),
                              soft[r]["eligible"],soft[r]["actual_l2"])
            }
            for name,vector in directions.items():
                step,valid=normalize(vector,r)
                moved=h0.double()+step
                outcome=suffix_logits(model,moved,q)
                actual=torch.linalg.vector_norm(moved-h0.double(),dim=1)
                valid=valid&((actual-r).abs()<=2e-5)
                logit_by_arm[name]=(outcome,valid,actual)
            for name,(outcome,valid,actual) in logit_by_arm.items():
                gains=target_gain(outcome,original,y,yp)
                for i in range(N):
                    active=bool(valid[i])
                    records.append({"seed":seed,"model_arm":arm,"case_index":i,
                       "source_id":cases["development"][i]["id"],
                       "source_family":"roc05_dev_unseen",
                       "original_label":int(y[i]),"legal_changed_label":int(yp[i]),
                       "query":int(q[i]),"radius":r,"route":name,
                       "valid":int(active),"midpoint_l2":float(actual[i]),
                       "margin_gain":float(gains[i]) if active else None,
                       "new_target_argmax":int(outcome[i].argmax()) if active else None,
                       "original_pred":int(original[i].argmax()),
                       "legal_full_pred":int(legal[i].argmax()),
                       "soft_off_manifold":int(name=="input_soft"),
                       "full_legal_natural_dose":int(name=="full_legal_unmatched")})
            if r==.2:
                observed_gain=target_gain(logit_by_arm["supervised_observed"][0],original,y,yp)
                soft_gain=target_gain(logit_by_arm["input_soft"][0],original,y,yp)
                pairmask=(logit_by_arm["supervised_observed"][1] & logit_by_arm["input_soft"][1])
                paired=int(pairmask.sum())
                primary={"paired_eligible":paired,"eligibility_fraction":paired/N,
                   "primary_rank7_gain":float(observed_gain[pairmask].mean()) if paired else None,
                   "primary_soft_input_gain":float(soft_gain[pairmask].mean()) if paired else None,
                   "primary_observed_minus_soft":float((observed_gain[pairmask]-soft_gain[pairmask]).mean()) if paired else None}
        out.mkdir(parents=True,exist_ok=True)
        path=out/f"roc05b_S{seed}_{arm}.csv.gz"
        with path.open("wb") as sink:
            with gzip.GzipFile(filename="",mode="wb",fileobj=sink,
                               compresslevel=6,mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding="utf-8",newline="") as stream:
                    fields=list(records[0])
                    writer=csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader();writer.writerows(records)
        card={"study":"ROC05B-v1-matched-hidden-input-causal-rule-intervention",
              "seed":seed,"arm":arm,"checkpoint_sha256":cksha,
              "source_dev_sha256":sha(source/"development.jsonl"),
              "model_training_updates_during_causal_analysis":0,
              "number_of_cases":N,"per_case_row_count":len(records),
              "per_case_csv_gzip_sha256":sha(path),
              "native_baseline_max_error":err,"original_correct":base_correct,
              "full_legal_changed_correct":legal_correct,"basis_singulars":sv,
              "shadow_cards":cards,"primary":primary,
              "nulls_per_shadow":128,"sealed_test_evaluated":False}
        (out/f"roc05b_S{seed}_{arm}_receipt.json").write_text(
            json.dumps(card,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"seed":seed,"arm":arm,"original_correct":base_correct,
                          "legal_correct":legal_correct,"primary":primary,
                          "mean_shadow_b_ratio":{z["basis"]:z["ratios"]["b"] for z in cards[:2]},
                          "per_case_rows":len(records)},sort_keys=True),flush=True)
    return card

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("configs/roc05b_matched_rule_causal_v1.json"))
    p.add_argument("--outdir",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    cfg=json.loads(args.config.read_text())
    if cfg["study"]!="ROC-05B-v1-matched-hidden-input-causal-rule-intervention":
        raise AssertionError("wrong frozen protocol")
    for arm in ARMS:
        evaluate_model(args.seed,arm,args.source_dir,args.checkpoint_dir,cfg,args.outdir)

if __name__=="__main__":
    main()
