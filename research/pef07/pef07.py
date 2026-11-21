"""PEF-07: preregistered 12-bit coalition and additive-threshold transfer.

Protocol committed separately at a3b441ce before any PEF-07 training.
Full 16-cell old-bit intervention groups stay context-disjoint. Both
trained task models, random-initialization baselines and a new h14
linear readout are evaluated on the same sealed heldout contexts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from itertools import combinations
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from lsa.mini_transformer_probe import MiniAttentionMlpResidualBlock

D, L, STEPS, BATCH = 32, 12, 450, 128
SEEDS = (103, 131, 167)
TASKS = ("coalition", "threshold")
EXTRA = (0, 1)
COUNTS = {"train": 160, "probe": 48, "heldout": 48}
SALT = b"pef07-v1-quartet:"
PAIRS = tuple(combinations(range(4), 2))
SPECIAL = ((0,1),(2,3))
CEILINGS = {"coalition":480, "threshold":528}


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()


def model_digest(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(bytes(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def source_split():
    contexts=[tuple((n>>i)&1 for i in range(8)) for n in range(256)]
    contexts.sort(key=lambda c:(hashlib.sha256(SALT+bytes(c)).digest(),bytes(c)))
    catalog={name:[] for name in COUNTS}
    ownership={}
    offset=0
    for name,count in COUNTS.items():
        for ctx in contexts[offset:offset+count]:
            ownership["".join(map(str,ctx))]=name
            catalog[name].extend(
                tuple((old>>i)&1 for i in range(4))+ctx for old in range(16)
            )
        offset+=count
    if offset!=256 or len(set.union(*(set(v) for v in catalog.values())))!=4096:
        raise AssertionError("invalid global context partition")
    for name,rows in catalog.items():
        if len(rows)!=COUNTS[name]*16:
            raise AssertionError("context group count")
        for i in range(0,len(rows),16):
            group=rows[i:i+16]
            if len({r[4:] for r in group})!=1 or len({r[:4] for r in group})!=16:
                raise AssertionError("old-bit counterfactuals crossed source partitions")
    manifest={
        "campaign":"PEF-07","salt":SALT.decode(),
        "context_count_by_split":COUNTS,
        "history_count_by_split":{k:len(v) for k,v in catalog.items()},
        "context_assignment":ownership,
        "contexts_in_sha256_order":["".join(map(str,c)) for c in contexts],
        "focal_positions":[0,1,2,3],
        "context_positions":list(range(4,12)),
        "source_contexts_are_disjoint":True,
    }
    manifest["sha256"]=hashlib.sha256(encoded(manifest)).hexdigest()
    return {k:tuple(v) for k,v in catalog.items()},manifest


def as_tensor(rows):
    return torch.tensor(rows,dtype=torch.long)


def target(bits,task):
    if bits.ndim!=2 or bits.shape[1]!=12:
        raise ValueError("expected 12 chronological bits")
    a,b,c,d=(bits[:,i] for i in range(4))
    if task=="coalition":
        return ((a*b+c*d)%2).long()
    if task=="threshold":
        return ((a+b+c+d)>=2).long()
    raise ValueError("unknown target")


class Looper(nn.Module):
    def __init__(self):
        super().__init__()
        self.bit_embedding=nn.Embedding(2,D)
        self.role=nn.Parameter(torch.randn(2,D)*0.02)
        self.memory_init=nn.Parameter(torch.zeros(D))
        self.block=MiniAttentionMlpResidualBlock(
            D,heads=4,mlp_ratio=2,alpha=1.0,beta=0.2
        )
        self.head=nn.Linear(D,2)

    def step(self,memory,embedding):
        tokens=torch.stack((embedding+self.role[0],memory+self.role[1]),dim=1)
        return self.block(tokens)[:,1]

    def memory(self,bits,continuation=()):
        if bits.ndim!=2 or bits.shape[1]!=L:
            raise ValueError("expected 12 chronological bits")
        h=self.memory_init[None,:].expand(len(bits),-1)
        for j in range(L):
            h=self.step(h,self.bit_embedding(bits[:,j]))
        for b in continuation:
            if b not in (0,1):
                raise ValueError("continuation must be binary")
            injected=torch.full((len(bits),),b,dtype=torch.long,device=bits.device)
            h=self.step(h,self.bit_embedding(injected))
        return h

    def logits(self,bits):
        return self.head(self.memory(bits))


def fit_ridge(states, labels):
    x=states.double()
    avg=x.mean(0)
    std=x.std(0,unbiased=False).clamp_min(1e-4)
    phi=torch.cat(((x-avg)/std,torch.ones((len(x),1),dtype=torch.float64)),1)
    penalty=torch.eye(phi.shape[1],dtype=torch.float64)
    penalty[-1,-1]=0
    beta=torch.linalg.solve(phi.T@phi+penalty,phi.T@labels.double())
    return avg,std,beta


def apply_ridge(states,fit):
    avg,std,beta=fit
    x=(states.double()-avg)/std
    phi=torch.cat((x,torch.ones((len(x),1),dtype=torch.float64)),1)
    return ((phi@beta)>=0.5).long()


def ridge_digest(fit):
    h=hashlib.sha256()
    for t in fit:
        h.update(bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def evaluate_classes(pred,truth):
    correct=int((pred==truth).sum())
    out={"correct":correct,"total":len(truth),"accuracy":correct/len(truth)}
    recalls=[]
    for c in (0,1):
        mask=truth==c
        value=float((pred[mask]==c).float().mean())
        out[f"class_{c}_count"]=int(mask.sum())
        out[f"class_{c}_recall"]=value
        recalls.append(value)
    out["balanced_accuracy"]=sum(recalls)/2
    return out


@torch.no_grad()
def hidden(model,bits,extra=()):
    return torch.cat([
        model.memory(bits[i:i+192],extra)
        for i in range(0,len(bits),192)
    ],0)


def cohort_interventions(bits,state,logits,task):
    if len(bits)%16 or state.shape!=(len(bits),D) or logits.shape!=(len(bits),2):
        raise ValueError("incomplete 16-state context blocks")
    n=len(bits)//16
    for i in range(0,len(bits),16):
        group=bits[i:i+16].tolist()
        if any(v[4:]!=group[0][4:] for v in group) or {
            tuple(v[:4]) for v in group
        }!={tuple((mask>>j)&1 for j in range(4)) for mask in range(16)}:
            raise AssertionError("counterfactual source closure failed")
    h=state.reshape(n,16,D)
    z=logits.reshape(n,16,2)
    scalar=z[:,:,1]-z[:,:,0]
    pred=z.argmax(2)
    truth=target(bits,task).reshape(n,16)
    special=[]
    crossed=[]
    rows=[]
    for i,j in PAIRS:
        m=(1<<i)|(1<<j)
        dz=scalar[:,m]-scalar[:,1<<i]-scalar[:,1<<j]+scalar[:,0]
        dh=h[:,m]-h[:,1<<i]-h[:,1<<j]+h[:,0]
        pair_only=(pred[:,1<<i]==pred[:,0])&(pred[:,1<<j]==pred[:,0])&(
            pred[:,m]!=pred[:,0])
        truth_pair=(truth[:,1<<i]==truth[:,0])&(
            truth[:,1<<j]==truth[:,0])&(truth[:,m]!=truth[:,0])
        additive=(z[:,1<<i]+z[:,1<<j]-z[:,0]).argmax(1)
        same=pair_only&(additive==pred[:,m])
        different=pair_only&(additive!=pred[:,m])
        avg=float(dz.abs().mean())
        (special if (i,j) in SPECIAL else crossed).append(avg)
        row={
            "pair":[i,j],"context_count":n,
            "teacher_pair_only":int(truth_pair.sum()),
            "native_pair_only":int(pair_only.sum()),
            "additive_surrogate_compatible":int(same.sum()),
            "additive_surrogate_mismatch":int(different.sum()),
            "mean_abs_four_corner_logit":avg,
            "mean_signed_four_corner_logit":float(dz.mean()),
            "mean_four_corner_hidden_l2":float(dh.norm(dim=1).mean()),
        }
        expected=(i,j) in (SPECIAL if task=="coalition" else PAIRS)
        if row["teacher_pair_only"]!=(n if expected else 0):
            raise AssertionError("independent teacher pair signature incorrect")
        if row["native_pair_only"]!=row["additive_surrogate_compatible"]+row["additive_surrogate_mismatch"]:
            raise AssertionError("additive control cases do not partition")
        rows.append(row)
    a=sum(special)/2
    b=sum(crossed)/4
    return {
        "contexts":n,"all_counterfactuals_are_natural_model_states":True,
        "pairs":rows,"intended_abs_mixed":a,"cross_abs_mixed":b,
        "intended_to_cross_ratio":a/(b+1e-6),
        "native_pair_only_events":sum(r["native_pair_only"] for r in rows),
        "surrogate_compatible_events":sum(r["additive_surrogate_compatible"] for r in rows),
        "surrogate_mismatch_events":sum(r["additive_surrogate_mismatch"] for r in rows),
        "teacher_additive_score_has_zero_mixed_logit":task=="threshold",
    }


def frozen_file(root,seed,task):
    return root/f"seed{seed}.{task}.frozen.pt"


def train_model(seed,task,pool,root,steps=STEPS):
    if seed not in SEEDS or task not in TASKS or steps<1:
        raise ValueError("unregistered task/seed/step count")
    torch.manual_seed(seed+(10000 if task=="threshold" else 0))
    model=Looper().train()
    start=model_digest(model)
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0)
    gen=torch.Generator().manual_seed(seed*7919+59+(100000 if task=="threshold" else 0))
    initial=last=None
    for k in range(steps):
        inp=pool[torch.randint(len(pool),(BATCH,),generator=gen)]
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(inp),target(inp,task))
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite train loss")
        if k==0:
            initial=float(loss.detach())
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(gn):
            raise AssertionError("nonfinite gradient")
        opt.step()
        last=float(loss.detach())
    model.eval()
    finish=model_digest(model)
    path=frozen_file(root,seed,task)
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                "seed":seed,"task":task,"steps":steps,
                "initial_weights_sha256":start,
                "frozen_weights_sha256":finish},path)
    receipt={
        "seed":seed,"task":task,"steps":steps,
        "file":path.name,
        "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "initial_weights_sha256":start,"frozen_weights_sha256":finish,
        "train_loss_initial":initial,"train_loss_final":last,
    }
    return receipt


def load_checkpoint(root,rec):
    path=root/rec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=rec["checkpoint_sha256"]:
        raise AssertionError("checkpoint SHA256 altered")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if (payload["seed"],payload["task"],payload["steps"])!=(
        rec["seed"],rec["task"],rec["steps"]):
        raise AssertionError("checkpoint metadata mismatch")
    model=Looper()
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    if model_digest(model)!=rec["frozen_weights_sha256"]:
        raise AssertionError("checkpoint weights mismatch")
    return model


@torch.no_grad()
def score_checkpoint(root,rec,catalog):
    seed,task=rec["seed"],rec["task"]
    model=load_checkpoint(root,rec)
    model_sha=model_digest(model)
    probe=as_tensor(catalog["probe"])
    heldout=as_tensor(catalog["heldout"])
    truth=target(heldout,task)
    h12=hidden(model,heldout)
    z12=model.head(h12)
    native=evaluate_classes(z12.argmax(1),truth)
    p14=hidden(model,probe,EXTRA)
    h14=hidden(model,heldout,EXTRA)
    ridge=fit_ridge(p14,target(probe,task))
    later=evaluate_classes(apply_ridge(h14,ridge),truth)
    original_head_later=evaluate_classes(model.head(h14).argmax(1),truth)
    torch.manual_seed(seed+30000+(10000 if task=="threshold" else 0))
    random=Looper().eval()
    random_sha=model_digest(random)
    r12=hidden(random,heldout)
    random_native=evaluate_classes(random.head(r12).argmax(1),truth)
    rp=hidden(random,probe,EXTRA)
    re=hidden(random,heldout,EXTRA)
    random_ridge=fit_ridge(rp,target(probe,task))
    random_later=evaluate_classes(apply_ridge(re,random_ridge),truth)
    effects=cohort_interventions(heldout,h12,z12,task)
    if model_digest(model)!=model_sha or model_digest(random)!=random_sha:
        raise AssertionError("frozen model modified by downstream scoring")
    if hashlib.sha256(frozen_file(root,seed,task).read_bytes()).hexdigest()!=rec["checkpoint_sha256"]:
        raise AssertionError("checkpoint bytes changed through evaluation")
    result={
        "seed":seed,"task":task,"checkpoint_sha256":rec["checkpoint_sha256"],
        "model_sha256":model_sha,"trained_h12_native_head":native,
        "frozen_h14_downstream_ridge":later,
        "native_head_after_h14_out_of_training_horizon":original_head_later,
        "downstream_probe_sha256":ridge_digest(ridge),
        "random_initializer_sha256":random_sha,
        "random_h12_native_head":random_native,
        "random_h14_downstream_ridge":random_later,
        "context_only_exact_ceiling":CEILINGS[task],
        "context_only_exact_denominator":768,
        "probe_histories":len(probe),"heldout_histories":len(heldout),
        "focal_pair_effects":effects,
        "all_original_weights_frozen_before_probe":True,
    }
    (root/f"seed{seed}.{task}.evaluation.json").write_bytes(encoded(result))
    return result


def run(root,seeds=SEEDS,tasks=TASKS,steps=STEPS):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True,exist_ok=True)
    catalog,manifest=source_split()
    (root/"source_split.json").write_bytes(encoded(manifest))
    pool=as_tensor(catalog["train"])
    sealed=[]
    for seed in seeds:
        for task in tasks:
            rec=train_model(seed,task,pool,root,steps)
            rec["source_sha256"]=manifest["sha256"]
            sealed.append(rec)
            print("PEF07_SEALED",seed,task,rec["checkpoint_sha256"],flush=True)
    ledger={
        "campaign":"PEF-07",
        "all_six_trunks_sealed_before_scoring":True,
        "protocol":"research/pef07/PRE_REGISTRATION.md",
        "seed_order":list(seeds),"tasks":list(tasks),
        "registered_steps":STEPS,"actual_steps":steps,
        "source_sha256":manifest["sha256"],
        "cpu_torch":str(torch.__version__),
        "cpu_capability":torch.backends.cpu.get_cpu_capability(),
        "onednn_max_cpu_isa":os.environ.get("DNNL_MAX_CPU_ISA"),
        "processor":platform.processor(),
        "checkpoints":sealed,
    }
    (root/"sealed_training_ledger.json").write_bytes(encoded(ledger))
    rows=[score_checkpoint(root,rec,catalog) for rec in sealed]
    registered=tuple(seeds)==SEEDS and tuple(tasks)==TASKS and steps==STEPS
    primary=[]
    mechanistic=[]
    for seed in seeds:
        coal=next((x for x in rows if x["seed"]==seed and x["task"]=="coalition"),None)
        add=next((x for x in rows if x["seed"]==seed and x["task"]=="threshold"),None)
        if coal is None or add is None:
            continue
        success=(coal["trained_h12_native_head"]["correct"]>=615
                 and coal["frozen_h14_downstream_ridge"]["correct"]>=576)
        a=coal["focal_pair_effects"]["intended_to_cross_ratio"]
        b=add["focal_pair_effects"]["intended_to_cross_ratio"]
        mechanism=(a>1.25 and a-b>0.20)
        primary.append({"seed":seed,"pass":success})
        mechanistic.append({"seed":seed,"coalition_ratio":a,"threshold_ratio":b,
                            "difference":a-b,"pass":mechanism})
    n=sum(x["pass"] for x in primary)
    k=sum(x["pass"] for x in mechanistic)
    decision=lambda score:("criterion_met" if score>=2 else "criterion_not_met")\
        if registered else "nonregistered_debug_smoke"
    report={
        "campaign":"PEF-07","full_registered_run":registered,
        "primary_decision":decision(n),"diagnostic_decision":decision(k),
        "primary_seeds_met":n,"diagnostic_seeds_met":k,
        "seed_order":list(seeds),"task_order":list(tasks),"steps":steps,
        "source_sha256":manifest["sha256"],
        "context_only_ceiling_correct":CEILINGS,
        "train_history_count":2560,"probe_history_count":768,
        "heldout_history_count":768,
        "all_models_sealed_before_any_probe_or_test":True,
        "all_checkpoint_hashes_checked":True,
        "primary_per_seed":primary,"diagnostic_per_seed":mechanistic,
        "models":rows,
    }
    (root/"summary.json").write_bytes(encoded(report))
    print(json.dumps({
        "campaign":"PEF-07","registered":registered,
        "primary":report["primary_decision"],
        "interaction":report["diagnostic_decision"],
        "models":[{
            "seed":v["seed"],"task":v["task"],
            "h12":v["trained_h12_native_head"]["correct"],
            "h14":v["frozen_h14_downstream_ridge"]["correct"],
            "random_h14":v["random_h14_downstream_ridge"]["correct"],
            "pair_ratio":v["focal_pair_effects"]["intended_to_cross_ratio"],
        } for v in rows],
    },indent=2,sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--task",choices=TASKS)
    p.add_argument("--debug-steps",type=int,default=STEPS)
    args=p.parse_args()
    run(args.out,
        (args.seed,) if args.seed else SEEDS,
        (args.task,) if args.task else TASKS,
        args.debug_steps)


if __name__=="__main__":
    main()
