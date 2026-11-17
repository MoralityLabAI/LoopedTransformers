"""LOOPSCOPE-04 / LGA-01: supervised weight-tied TRM activation-gradient alignment.

Source-attributed MIT architecture mirror:
NagaSatyaPhanindraVallabhaneni/tiny-recursive-model @ f0406195,
also in MoralityLabAI/Compactification-Theory
research/levin_sieve_v1/pretrained_trm/upstream_arch.py.
Copyright (c) 2026 Naga Satya Phanindra Vallabhaneni, MIT License.

The scientific protocol was committed before any LGA training/test data.
The correct target is used only for supervised training and heldout diagnostics.
A head-only Bayes decision is target-BLIND; its retrospective utility is
scored only after the decision has been frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import struct

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "LGA-01/weight-tied-TRM-supervised-gradient-alignment-v1"
ROOT=Path(__file__).resolve().parent
SEEDS=(103,211,307)
ARMS=("final_only","deep_020")
STEPS=200
SNAPSHOTS=(0,50,100,200)
TRAIN_N=512
VAL_N=128
TEST_N=256
N_TOKENS=81
WIDTH=48
N_LOOPS=4
BOOT=2000
BOOT_SEED=41041
GOALS=tuple((81*j)//8 for j in range(8))
NEUTRAL=.15
ACTION_COST=.10
TRAIN_SEED=31041
VAL_SEED=31042
TEST_SEED=31043


class ReasoningBlock(nn.Module):
    def __init__(self,d_model=48,n_tokens=81,d_tok=162,d_ff=128):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.tok_mix=nn.Sequential(nn.Linear(n_tokens,d_tok),nn.GELU(),
                                   nn.Linear(d_tok,n_tokens))
        self.ln2=nn.LayerNorm(d_model)
        self.chan_mix=nn.Sequential(nn.Linear(d_model,d_ff),nn.GELU(),
                                    nn.Linear(d_ff,d_model))

    def forward(self,z):
        z=self.ln1(z+self.tok_mix(z.transpose(1,2)).transpose(1,2))
        return self.ln2(z+self.chan_mix(z))


class TinyRecursiveModel(nn.Module):
    def __init__(self,d_model=48,n_steps=4):
        super().__init__()
        self.n_steps=n_steps
        self.token_emb=nn.Embedding(10,d_model)
        self.pos_emb=nn.Embedding(81,d_model)
        self.block=ReasoningBlock(d_model)
        self.head=nn.Linear(d_model,10)

    def initial_state(self,x):
        pos=torch.arange(81,device=x.device).unsqueeze(0).expand(x.shape[0],-1)
        return self.token_emb(x)+self.pos_emb(pos)

    def reason_states(self,x):
        state=self.initial_state(x)
        states=[state]
        for _ in range(self.n_steps):
            state=self.block(state)
            states.append(state)
        return states

    def forward(self,x):
        return self.head(self.reason_states(x)[-1])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def derived_seed(*parts):
    data="LGA01|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8],"big") & ((1<<63)-1)


def fixed_data(n,seed):
    gen=torch.Generator().manual_seed(seed)
    x=torch.randint(1,10,(n,N_TOKENS),generator=gen)
    y=x.roll(shifts=1,dims=1)
    return x,y


def tensor_int64_sha(x):
    """Endian-fixed tensor provenance independent of optional NumPy import."""
    vals=x.detach().contiguous().reshape(-1).tolist()
    return sha(struct.pack("<"+"q"*len(vals),*vals))


def frozen_inputs(smoke=False):
    if smoke:
        counts=(64,32,32)
    else:
        counts=(TRAIN_N,VAL_N,TEST_N)
    data=[fixed_data(n,seed) for n,seed in zip(counts,
                                                 (TRAIN_SEED,VAL_SEED,TEST_SEED))]
    fingerprints={}
    for name,(x,y) in zip(("train","val","test"),data):
        fingerprints[name]={
            "n_sequences":len(x),
            "x_sha256":tensor_int64_sha(x),
            "y_sha256":tensor_int64_sha(y),
            "source_seed":{"train":TRAIN_SEED,"val":VAL_SEED,"test":TEST_SEED}[name]
        }
    observed={tuple(row.tolist()) for row in data[0][0]}
    for x,_ in data[1:]:
        if any(tuple(row.tolist()) in observed for row in x):
            raise ValueError("train/val/test sequence duplication")
        observed.update(tuple(row.tolist()) for row in x)
    return data,fingerprints


def avg_ce(logits,y):
    return F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))


def per_token_ce(logits,y):
    return F.cross_entropy(logits.reshape(-1,10),y.reshape(-1),
                           reduction="none").reshape(-1,N_TOKENS)


def cos_flat(a,b):
    a=a.reshape(-1).double();b=b.reshape(-1).double()
    denom=float(a.norm()*b.norm())
    return float((a@b)/denom) if denom>1e-18 else None


def model_state(model):
    return {k:v.detach().clone().cpu() for k,v in model.state_dict().items()}


def train_one(seed,arm,x,y,updates=STEPS,check_steps=SNAPSHOTS):
    torch.manual_seed(seed)
    model=TinyRecursiveModel(WIDTH,N_LOOPS)
    if sum(p.numel() for p in model.parameters())!=44001:
        raise ValueError("model does not match source 44,001 parameters")
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    gen=torch.Generator().manual_seed(123456+seed)
    history={}
    weights={0:model_state(model)}
    for step in range(1,updates+1):
        model.train()
        ids=torch.randint(len(x),(16,),generator=gen)
        states=model.reason_states(x[ids])
        logits=[model.head(h) for h in states]
        loss=avg_ce(logits[4],y[ids])
        if arm=="deep_020":
            loss=loss+(0.2/3.)*sum(avg_ce(logits[k],y[ids]) for k in (1,2,3))
        elif arm!="final_only":
            raise ValueError("unregistered supervision arm")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step in check_steps:
            grad=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])
            before=torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        optimizer.step()
        if step in check_steps:
            after=torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            delta=after-before
            history[str(step)]={
                "train_minibatch_loss_before_update":float(loss.detach()),
                "optimizer_update_cos_negative_BPTT_grad":cos_flat(delta,-grad),
                "optimizer_gradient_l2":float(grad.norm()),
                "optimizer_update_l2":float(delta.norm()),
                "linear_predicted_batch_loss_decrease":float(-(grad.double()@delta.double())),
            }
            weights[step]=model_state(model)
    return model,weights,history


def head_rowspace(model):
    matrix=model.head.weight.detach().double()
    _,s,vh=torch.linalg.svd(matrix,full_matrices=False)
    rank=int((s>1e-10*float(s.max())).sum())
    return vh[:rank].T.contiguous(),rank


def policy_decision(values,weights):
    """BPD-OUT06 exact action formula for a head-only proposed delta."""
    if len(values)!=9 or len(weights)!=9:
        raise ValueError("nine candidate effects and distribution required")
    pplus=sum(p for p,d in zip(weights,values) if d>NEUTRAL)
    pminus=sum(p for p,d in zip(weights,values) if d< -NEUTRAL)
    pneutral=1-pplus-pminus
    estimates={1:2*pplus-1-ACTION_COST,
               -1:2*pminus-1-ACTION_COST,
               0:pneutral}
    best=max(estimates.values())
    # Favor abstention on ties.
    return (0 if estimates[0]>=best-1e-12 else
            1 if estimates[1]>=best-1e-12 else -1),{
                "p_plus":pplus,"p_minus":pminus,"p_neutral":pneutral,
                "expected_utility":estimates}


def reward(action,truth):
    if action not in (-1,0,1) or truth not in (-1,0,1):
        raise ValueError("ternary decision expected")
    base=(1 if action==0 else -1) if truth==0 else (
        0 if action==0 else 1 if action==truth else -1)
    return base-ACTION_COST*int(action!=0)


def finite_class(effect):
    return 1 if effect>NEUTRAL else -1 if effect< -NEUTRAL else 0


def head_only_bayes(model,h,delta,targets,rank_basis):
    """Classwise head-only, after proposed delta is computed; label-free action.

    Native finite output mediation over the remaining loop is a separate test.
    """
    with torch.no_grad():
        b,n,d=h.shape
        clean=model.head(h)
        plus=model.head(h+delta)
        minus=model.head(h-delta)
        q=torch.softmax(clean[...,1:10].double(),dim=-1)
        uniform=[1/9.]*9
        result=[]
        for i in range(b):
            local=[]
            for j in range(n):
                logits=clean[i,j].tolist()
                p=q[i,j].tolist()
                effect=[]
                for digit in range(1,10):
                    rivals=[a for a in range(10) if a!=digit]
                    competitor=max(rivals,key=lambda a:logits[a])
                    effect.append(float((plus[i,j,digit]-plus[i,j,competitor])-
                                        (minus[i,j,digit]-minus[i,j,competitor])))
                selected,_=policy_decision(effect,p)
                control,_=policy_decision(effect,uniform)
                actual=finite_class(effect[int(targets[i,j])-1])
                local.append({
                    "bayes_action":selected,"uniform_action":control,
                    "true_class":actual,
                    "bayes_utility":reward(selected,actual),
                    "uniform_utility":reward(control,actual),
                    "always_abstain_utility":reward(0,actual),
                    "act":int(selected!=0),
                    "mass_sign_correct":int(selected==actual and actual!=0)
                })
            result.append({
                "bayes_utility":sum(z["bayes_utility"] for z in local)/n,
                "uniform_utility":sum(z["uniform_utility"] for z in local)/n,
                "always_abstain_utility":sum(z["always_abstain_utility"] for z in local)/n,
                "action_rate":sum(z["act"] for z in local)/n,
                "correct_nonzero":sum(z["mass_sign_correct"] for z in local)/n,
            })
    return result


def profile(model,x,y):
    """All local-head gradient statistics, with sequence-cluster raw receipts."""
    model.eval()
    basis,rank=head_rowspace(model)
    per_layer=[[] for _ in range(4)]
    curves={"cross_entropy":[[] for _ in range(5)],
            "accuracy":[[] for _ in range(5)]}
    decisions=[[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0,len(x),32):
            xb=x[start:start+32];yb=y[start:start+32]
            states=model.reason_states(xb)
            readouts=[model.head(h) for h in states]
            losses=[per_token_ce(z,yb) for z in readouts]
            for k in range(5):
                p=readouts[k].argmax(-1).eq(yb).float()
                curves["cross_entropy"][k].extend(losses[k].mean(-1).tolist())
                curves["accuracy"][k].extend(p.mean(-1).tolist())
            for k in range(4):
                h=states[k]
                delta=states[k+1]-h
                prob=torch.softmax(readouts[k],dim=-1)
                onehot=F.one_hot(yb,num_classes=10).float()
                local_grad=(prob-onehot)@model.head.weight
                dot=-(local_grad*delta).sum(-1)
                gradnorm=local_grad.norm(dim=-1)
                deltanorm=delta.norm(dim=-1)
                cosine=dot/(gradnorm*deltanorm).clamp_min(1e-12)
                actual=losses[k]-losses[k+1]
                visible=(delta.double()@basis).square().sum(-1)/(
                    delta.double().square().sum(-1).clamp_min(1e-18))
                for i in range(len(xb)):
                    per_layer[k].append({
                        "cosine":float(cosine[i].mean()),
                        "predicted_CE_gain":float(dot[i].mean()),
                        "actual_CE_gain":float(actual[i].mean()),
                        "positive_predicted_fraction":float((dot[i]>0).float().mean()),
                        "actual_CE_decrease_fraction":float((actual[i]>0).float().mean()),
                        "mean_delta_norm":float(deltanorm[i].mean()),
                        "mean_gradient_norm":float(gradnorm[i].mean()),
                        "head_visible_squared_fraction":float(visible[i].mean()),
                    })
                # This is a head-only diagnostic, distinct from native future readout.
                decisions[k].extend(head_only_bayes(model,h,delta,yb,basis))
    if any(len(z)!=len(x) for z in per_layer+decisions):
        raise ValueError("sample-level trajectory receipt missing")
    summary={
        "n_independent_sequences":len(x),
        "measured_head_rank":rank,
        "per_loop":{
            str(k):{
                "mean":{name:sum(v[name] for v in per_layer[k])/len(x)
                        for name in per_layer[k][0]},
                "per_sequence":per_layer[k],
                "head_only_bayes":{
                    name:sum(v[name] for v in decisions[k])/len(x)
                    for name in decisions[k][0]
                },
            } for k in range(4)
        },
        "readout":{
            str(k):{
                "mean_CE":sum(curves["cross_entropy"][k])/len(x),
                "mean_accuracy":sum(curves["accuracy"][k])/len(x),
                "per_sequence_CE":curves["cross_entropy"][k],
                "per_sequence_accuracy":curves["accuracy"][k]
            } for k in range(5)
        }
    }
    return summary


def bootstrap(values,seed,draws=BOOT):
    rng=random.Random(seed)
    n=len(values)
    data=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                for _ in range(draws))
    return [data[int(.025*draws)],data[int(.975*draws)-1]]


def _mean(x):
    return sum(x)/len(x) if x else None


def intervention(model,x,y,model_seed,arm,checkpoint=STEPS,max_sequences=64):
    """Full native remaining-loop counterfactual, not a local probe or hook.

    Each one-token edit has exactly its proposed update's L2 norm.
    """
    model.eval()
    output={}
    with torch.no_grad():
        for k in (1,2):
            cases=[]
            for sample_idx in range(min(max_sequences,len(x))):
                xb=x[sample_idx:sample_idx+1]
                states=model.reason_states(xb)
                state=states[k][0]
                next_delta=(states[k+1]-states[k])[0]
                variants=[]
                for j,tok in enumerate(GOALS):
                    delta=next_delta[tok]
                    gen=torch.Generator().manual_seed(derived_seed(
                        TEST_SEED,model_seed,arm,checkpoint,sample_idx,k,j))
                    random_unit=torch.randn(delta.shape,generator=gen,dtype=delta.dtype)
                    random_unit=random_unit*(
                        delta.norm()/random_unit.norm().clamp_min(1e-12))
                    for tag,edit in (
                        ("clean",None),("plus",delta),("minus",-delta),
                        ("random",random_unit)):
                        h=state.clone()
                        if edit is not None:
                            h[tok]+=edit
                        variants.append((j,tok,tag,h,
                                         float(delta.norm()),float(random_unit.norm())))
                batch=torch.stack([v[3] for v in variants])
                z=batch
                for _ in range(4-k):
                    z=model.block(z)
                out=model.head(z)
                per_goal={}
                for i,(j,tok,tag,_,dose,sham_dose) in enumerate(variants):
                    logits=out[i,tok]
                    target=int(y[sample_idx,tok])
                    if tag=="clean":
                        rival=logits.clone();rival[target]=-float("inf")
                        index=int(rival.argmax())
                        per_goal[j]={"goal_token":tok,"truth_digit":target,
                                     "rival":index,"dose":dose,
                                     "sham_dose":sham_dose,"arms":{}}
                    # Arm order is clean, plus, minus, random.
                    competitor=per_goal[j]["rival"]
                    lp=F.cross_entropy(logits.view(1,10),
                                       torch.tensor([target])).item()
                    margin=float(logits[target]-logits[competitor])
                    per_goal[j]["arms"][tag]={"margin":margin,"CE":lp}
                for v in per_goal.values():
                    if abs(v["dose"]-v["sham_dose"])>1e-6:
                        raise ArithmeticError("native random physical-dose mismatch")
                    clean=v["arms"]["clean"]
                    v["contrasts"]={
                        mode:{
                            "margin_change":v["arms"][mode]["margin"]-clean["margin"],
                            "CE_change":v["arms"][mode]["CE"]-clean["CE"]
                        } for mode in ("plus","minus","random")
                    }
                cases.append({"sample":sample_idx,"goals":list(per_goal.values())})
            summary={}
            for mode in ("plus","minus","random"):
                summary[mode]={
                    name:_mean([_mean([g["contrasts"][mode][name] for g in row["goals"]])
                                for row in cases])
                    for name in ("margin_change","CE_change")
                }
            output[str(k)]={"n_puzzles":len(cases),"n_goals":len(cases)*len(GOALS),
                            "arms_mean":summary,"cases":cases}
    return output


def run(smoke=False,output_dir=None):
    torch.set_num_threads(2)
    data,rosters=frozen_inputs(smoke)
    (train_x,train_y),(val_x,val_y),(test_x,test_y)=data
    seeds=(103,) if smoke else SEEDS
    snapshots=(0,5,10) if smoke else SNAPSHOTS
    updates=snapshots[-1]
    models={}
    outcome={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "torch":torch.__version__,
        "source":"MIT 44k original TRM mirror, independent weights",
        "task":"81 IID 1..9 digits; target cyclic roll right by 1",
        "source_splits":rosters,
        "seeds":list(seeds),"training_arms":list(ARMS),
        "steps":updates,"checkpoints":list(snapshots),
        "results":{},
        "inference_limits":"Local head CE gradients use ground truth for diagnosis, head-only Bayes uses computed next update and cannot save compute."
    }
    for seed in seeds:
        outcome["results"][str(seed)]={}
        for arm in ARMS:
            model,checkpoints,opt=train_one(seed,arm,train_x,train_y,updates,check_steps=snapshots)
            item={"optimizer_ledger":opt,"checkpoints":{}}
            for step in snapshots:
                model.load_state_dict(checkpoints[step],strict=True)
                prof=profile(model,test_x,test_y)
                item["checkpoints"][str(step)]=prof
                print("LGA01_CHECKPOINT",seed,arm,step,"accuracies",
                      [round(prof["readout"][str(k)]["mean_accuracy"],3)
                       for k in range(5)],"mean_cos",
                      [round(prof["per_loop"][str(k)]["mean"]["cosine"],3)
                       for k in range(4)],flush=True)
            final=item["checkpoints"][str(updates)]
            initial=item["checkpoints"]["0"]
            admitted=final["readout"]["4"]["mean_accuracy"]>.2
            depth=[]
            for k in range(4):
                vals=[p["cosine"] for p in final["per_loop"][str(k)]["per_sequence"]]
                lower,upper=bootstrap(vals,derived_seed(
                    BOOT_SEED,seed,arm,k),draws=200 if smoke else BOOT)
                improved=final["per_loop"][str(k)]["mean"]["cosine"]>\
                         initial["per_loop"][str(k)]["mean"]["cosine"]
                depth.append({"k":k,"mean":_mean(vals),
                              "bootstrap95":[lower,upper],
                              "lower_above_zero":bool(lower>0),
                              "higher_than_initial":bool(improved)})
            item["registered_diagnostics"]={
                "test_final_acc":final["readout"]["4"]["mean_accuracy"],
                "task_admitted":admitted,
                "per_loop_gradient_alignment":depth,
                "seed_positive_descent":bool(admitted and sum(
                    z["lower_above_zero"] for z in depth)>=2)
            }
            # Causal endpoint runs from same final registered weights.
            model.load_state_dict(checkpoints[updates],strict=True)
            item["native_remaining_loop_intervention"]=intervention(
                model,test_x,test_y,seed,arm,checkpoint=updates,
                max_sequences=8 if smoke else 64)
            outcome["results"][str(seed)][arm]=item
            print("LGA01_CAUSAL_COMPLETE",seed,arm,flush=True)
    votes=[
        outcome["results"][str(seed)]["final_only"]["registered_diagnostics"]
        for seed in seeds
    ]
    good=sum(x["seed_positive_descent"] and sum(
        z["higher_than_initial"] and z["lower_above_zero"]
        for z in x["per_loop_gradient_alignment"])>=2 for x in votes)
    outcome["registered_study_positive"]=bool(not smoke and good>=2)
    outcome["registered_study_status"]=( "PILOT_ONLY" if smoke else
        "POSITIVE" if outcome["registered_study_positive"] else "FAIL_OR_NOT_ADMITTED")
    if output_dir:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        (output_dir/("smoke.json" if smoke else "full.json")).write_text(
            json.dumps(outcome,sort_keys=True,indent=2)+"\n")
    print("LGA01_COMPLETE",outcome["registered_study_status"],flush=True)
    return outcome


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    report=run(smoke=args.smoke,output_dir=args.out)
    print(json.dumps({
        "mode":report["mode"],"status":report["registered_study_status"],
        "split_sha":report["source_splits"],
        "out":str(args.out)
    },sort_keys=True))


if __name__=="__main__":
    main()
