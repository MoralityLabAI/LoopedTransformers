"""LGA-04: frozen paired critics on fresh cyclic-copy test, native causal edits."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path
import torch
from torch.nn import functional as F
from mechinterp.loopscope04.gradient_flow import fixed_data,per_token_ce,head_rowspace,tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen
from mechinterp.loopscope05.terminal_sae_test import make_model,make_sae,terminal_gradients,active_components
from mechinterp.loopscope06.eval_value import gradient_value
from mechinterp.loopscope07.train_sobolev import load_pair,protocol_sha,TEACHER_SEEDS,OLD_SOURCE,TRAIN_SEED,VAL_SEED,SCHEMA as FIT_SCHEMA

SCHEMA="LGA-04/new-heldout-paired-sobolev-value-native-causal-v1"
TEST_SEED=31054
SMOKE_SEED=61054
N_TEST=256
N_CAUSAL=64
BOOT=2000
DEPTHS=(0,1,2,3)
CAUSAL_DEPTHS=(1,2)
GOALS=(0,10,20,30,40,50,60,70)
EDIT_ARMS=("sobolev_down","valueonly_down","original_down","native","random","sae_active","sae_remainder","head_null","oracle_down")
CAUSAL_CONTRASTS=(("sobolev_down","valueonly_down"),("sobolev_down","random"),
                  ("sobolev_down","original_down"),("sae_active","sae_remainder"),
                  ("head_null","random"))


def seed(*parts):
    return int.from_bytes(hashlib.sha256(("LGA04|"+"|".join(map(str,parts))).encode()).digest()[:8],"big")&((1<<63)-1)


def fresh_test(smoke=False):
    n=8 if smoke else N_TEST
    source_seed=SMOKE_SEED if smoke else TEST_SEED
    x,y=fixed_data(n,source_seed)
    prior=set()
    for s,num in (*OLD_SOURCE,(TRAIN_SEED,256),(VAL_SEED,64)):
        old,_=fixed_data(num,s)
        prior.update(tuple(row) for row in old.tolist())
    here=[tuple(row) for row in x.tolist()]
    if len(set(here))!=len(here) or any(row in prior for row in here):
        raise ValueError("new heldout source repeats original training or calibration")
    return x,y,{"seed":source_seed,"n_sequences":n,
                "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
                "independent_from_original_and_current_calibration":True}


def cosine(a,b):
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)


def bootstrap(vals,root,draws=BOOT):
    if len(vals)<2 or any(not math.isfinite(v) for v in vals):
        raise ValueError("independent full-sequence bootstrap needs finite values")
    rng=random.Random(root);n=len(vals)
    samples=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return [samples[int(.025*draws)],samples[int(.975*draws)-1]]


def profiles(teacher,nets,mu,std,scale,x,y):
    """One row per independent 81-token sequence at each of four depths."""
    all_rows={str(k):[] for k in DEPTHS}
    for start in range(0,len(x),8):
        xb=x[start:start+8];yb=y[start:start+8]
        states,oracle=terminal_gradients(teacher,xb,yb)
        with torch.no_grad():
            target_ce=per_token_ce(teacher.head(states[-1]),yb)
        for k in DEPTHS:
            h=states[k].detach()
            target_grad=oracle[k]
            out=teacher.head(h).detach()
            local=(torch.softmax(out,-1)-F.one_hot(yb,10).float())@teacher.head.weight
            native=states[k+1]-h
            estimates={}
            for name,net in nets.items():
                pred,g=gradient_value(net,h,k,mu,std)
                estimates[name]={
                    "cos":cosine(g,target_grad),
                    "grad_NMSE":((g-target_grad)/scale).square().mean(-1),
                    "CE_MSE":(pred-target_ce).square(),
                    "native_negative_cos":cosine(native,-g)
                }
            for i in range(len(xb)):
                row={"local_cos":float(cosine(local[i],target_grad[i]).mean()),
                     "native_terminal_cos":float(cosine(native[i],-target_grad[i]).mean())}
                for name,v in estimates.items():
                    for metric,vals in v.items():
                        row[f"{name}_{metric}"]=float(vals[i].mean())
                row["gain_over_valueonly"]=row["value_plus_gradient_cos"]-row["value_only_continue_cos"]
                row["gain_over_original"]=row["value_plus_gradient_cos"]-row["original_cos"]
                all_rows[str(k)].append(row)
    if any(len(a)!=len(x) for a in all_rows.values()):
        raise ValueError("registered whole-sequence profiles incomplete")
    return all_rows


def summarize_profiles(rows,sid,smoke=False):
    output={};n_pass=0
    for k in DEPTHS:
        items=rows[str(k)]
        means={name:sum(row[name] for row in items)/len(items) for name in items[0]}
        ci={}
        for field in ("gain_over_valueonly","gain_over_original"):
            ci[field]=bootstrap([x[field] for x in items],seed(44054,sid,k,field),
                                draws=100 if smoke else BOOT)
        ok=bool(ci["gain_over_valueonly"][0]>0 and ci["gain_over_original"][0]>0 and
                means["value_plus_gradient_grad_NMSE"]<means["value_only_continue_grad_NMSE"])
        output[str(k)]={"rows":items,"n_whole_sequences":len(items),
                        "means":means,"bootstrap95":ci,"registered_depth_pass":ok}
        n_pass+=int(ok)
    return output,n_pass


def normalized(v,r):
    norm=float(v.norm())
    if norm<=1e-8 or r<=1e-8:
        return None
    edit=(v*(r/norm)).detach().float()
    if abs(float(edit.norm())-r)>1e-5:
        raise ArithmeticError("matched physical intervention L2 dose mismatch")
    return edit


def edit_sha(edit):
    vals=edit.detach().float().contiguous().reshape(-1).tolist()
    return digest(struct.pack("<"+"f"*len(vals),*vals))


def native_interventions(teacher,nets,mu,std,saes,x,y,sid,smoke=False):
    """Original h_k edit direction selection is label-blind except oracle arm."""
    count=min(4 if smoke else N_CAUSAL,len(x))
    basis,_=head_rowspace(teacher)
    output={}
    for k in CAUSAL_DEPTHS:
        sae,saemu,saestd=saes[k]
        cases=[]
        for i in range(count):
            xb=x[i:i+1];yb=y[i:i+1]
            with torch.no_grad():states=teacher.reason_states(xb)
            h=states[k][0].detach()
            v=states[k+1][0]-h
            estimates={}
            for name,net in nets.items():
                _,g=gradient_value(net,h.unsqueeze(0),k,mu,std)
                estimates[name]=g[0]
            parts=active_components(sae,saemu,saestd,h,v)
            null=v.double()-(v.double()@basis)@basis.T
            plans=[];schedule=[]
            for j,tok in enumerate(GOALS):
                r=.25*float(v[tok].norm())
                g=torch.Generator().manual_seed(seed("random",sid,k,i,j))
                rand=torch.randn(48,generator=g,dtype=torch.float32)
                dirs={
                    "sobolev_down":-estimates["value_plus_gradient"][tok],
                    "valueonly_down":-estimates["value_only_continue"][tok],
                    "original_down":-estimates["original"][tok],
                    "native":v[tok],"random":rand,
                    "sae_active":parts["active"][tok],
                    "sae_remainder":parts["remainder"][tok],
                    "head_null":null[tok].float()
                }
                eligible=bool(parts["admitted"][tok]) and r>1e-8 and all(
                    float(a.norm())>1e-8 for a in dirs.values())
                plan={"slot":j,"token":tok,"radius":r,
                      "preoutcome_admitted":eligible,"arms":{}}
                if eligible:
                    for arm,z in dirs.items():
                        edit=normalized(z,r)
                        if edit is None:raise ArithmeticError("source admitted direction vanished")
                        schedule.append((j,tok,arm,edit))
                else:plan["reason"]="source-degenerate-SAE-or-critic-direction"
                plans.append(plan)
            # Teacher labels are used ONLY for oracle and retrospective scoring.
            _,oracle_all=terminal_gradients(teacher,xb,yb)
            oracle=oracle_all[k][0]
            for j,tok in enumerate(GOALS):
                if plans[j]["preoutcome_admitted"]:
                    edit=normalized(-oracle[tok],plans[j]["radius"])
                    if edit is not None:
                        schedule.append((j,tok,"oracle_down",edit))
                    else:plans[j]["oracle_unavailable"]=True
            changes=[h]
            for j,tok,arm,edit in schedule:
                altered=h.clone();altered[tok]+=edit
                changes.append(altered)
            z=torch.stack(changes)
            with torch.no_grad():
                for _ in range(4-k):z=teacher.block(z)
                full_logits=teacher.head(z)
                CE=per_token_ce(full_logits,yb.expand(len(z),-1))
            clean=full_logits[0];baseline=CE[0]
            with torch.no_grad():reference=teacher.head(states[-1])[0]
            if float((clean-reference).abs().max())>1e-5:
                raise ArithmeticError("native source clean/zero patch recurrence failed")
            for ix,(j,tok,arm,edit) in enumerate(schedule,1):
                truth=int(yb[0,tok])
                opponent=clean[tok].clone();opponent[truth]=-float("inf")
                rival=int(opponent.argmax())
                readout=full_logits[ix,tok]
                plans[j]["arms"][arm]={
                    "dose":float(edit.norm()),"edit_sha256":edit_sha(edit),
                    "truth_digit":truth,"clean_rival":rival,
                    "all_final_token_CE":CE[ix].tolist(),
                    "clean_all_final_token_CE":baseline.tolist(),
                    "final_global_CE":float(CE[ix].mean()),
                    "clean_global_CE":float(baseline.mean()),
                    "global_CE_change":float(CE[ix].mean()-baseline.mean()),
                    "goal_logits":readout.tolist(),"clean_goal_logits":clean[tok].tolist(),
                    "goal_CE_change":float(CE[ix,tok]-baseline[tok]),
                    "goal_margin_change":float(
                        (readout[truth]-readout[rival])-(clean[tok,truth]-clean[tok,rival])),
                    "predicted_sobolev_global_CE_change":float(
                        estimates["value_plus_gradient"][tok].double().dot(edit.double())/81),
                    "predicted_terminal_global_CE_change":float(
                        oracle[tok].double().dot(edit.double())/81),
                }
            cases.append({"test_sequence_index":i,"goal_plan":plans})
        output[str(k)]=cases
    return output


def summarize_native(cases,sid,smoke=False):
    results={}
    for k in CAUSAL_DEPTHS:
        raws=cases[str(k)]
        rows=[]
        for record in raws:
            eligible=[g for g in record["goal_plan"] if g["preoutcome_admitted"]]
            contrast={}
            for a,b in CAUSAL_CONTRASTS:
                key=f"{a}_minus_{b}"
                values=[g["arms"][a]["global_CE_change"]-
                        g["arms"][b]["global_CE_change"] for g in eligible]
                contrast[key]=sum(values)/len(values) if values else 0.
            rows.append({"test_sequence_index":record["test_sequence_index"],
                         "n_original_goals":8,"n_admitted_goals":len(eligible),
                         "paired_global_CE":contrast})
        comps={}
        for a,b in CAUSAL_CONTRASTS:
            key=f"{a}_minus_{b}"
            values=[r["paired_global_CE"][key] for r in rows]
            ci=bootstrap(values,seed("causal",44054,sid,k,key),
                         draws=100 if smoke else BOOT)
            comps[key]={"mean":sum(values)/len(values),"bootstrap95":ci,
                        "direction":"POSITIVE" if ci[0]>0 else "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"}
        results[str(k)]={"n_original_sequences":len(rows),
                         "n_original_goal_slots":8*len(rows),
                         "n_admitted_goal_slots":sum(r["n_admitted_goals"] for r in rows),
                         "comparisons":comps,"source_rows":rows,"raw_cases":raws}
    return results


def evaluate(source_bundle,source_meta,prior_bundle,prior_meta,
             paired_bundle,paired_meta,out,smoke=False):
    torch.set_num_threads(2)
    old_raw=Path(source_bundle).read_bytes()
    meta_raw=Path(source_meta).read_bytes()
    prior_raw=Path(prior_bundle).read_bytes()
    prior_meta_raw=Path(prior_meta).read_bytes()
    pair_raw=Path(paired_bundle).read_bytes()
    pair_meta_raw=Path(paired_meta).read_bytes()
    _,source=load_frozen(source_bundle,source_meta)
    pairing=json.loads(pair_meta_raw)
    if (pairing["schema"]!=FIT_SCHEMA or pairing["protocol_sha256"]!=protocol_sha() or
        pairing["old_teacher_SAE_bundle_sha256"]!=digest(old_raw) or
        pairing["old_teacher_SAE_meta_sha256"]!=digest(meta_raw) or
        pairing["old_LGA03_critic_bundle_sha256"]!=digest(prior_raw) or
        pairing["old_LGA03_critic_receipt_sha256"]!=digest(prior_meta_raw) or
        pairing["frozen_bundle_sha256"]!=digest(pair_raw) or
        pairing["test_data_opened"] is not False):
        raise ValueError("calibration-only frozen critics changed before heldout")
    x,y,manifest=fresh_test(smoke)
    output={};passed_count=0
    for sid in pairing["original_teacher_seeds"]:
        teacher=make_model(source,sid)
        models,original,mu,std,scale=load_pair(paired_bundle,paired_meta,sid)
        nets={**models,"original":original}
        rows=profiles(teacher,nets,mu,std,scale,x,y)
        gr,n_positive=summarize_profiles(rows,sid,smoke)
        saes={k:make_sae(source,sid,k) for k in CAUSAL_DEPTHS}
        source_native=native_interventions(teacher,nets,mu,std,saes,x,y,sid,smoke)
        native=summarize_native(source_native,sid,smoke)
        admitted=bool(n_positive>=2)
        passed_count+=int(admitted)
        output[str(sid)]={"gradient_profiles":gr,
                          "positive_gradient_depths":n_positive,
                          "registered_seed_pass":admitted,"native_effects":native}
        print("LGA04_EVALUATED",sid,"matched_cos",[
            round(gr[str(k)]["means"]["value_plus_gradient_cos"],4) for k in DEPTHS],
            "valueonly_cos",[
            round(gr[str(k)]["means"]["value_only_continue_cos"],4) for k in DEPTHS],
            "passed",admitted,flush=True)
    verdict="PILOT_ONLY" if smoke else "POSITIVE" if passed_count>=2 else "FAIL"
    record={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "source_bundle_sha256":digest(old_raw),
        "source_meta_sha256":digest(meta_raw),
        "original_critic_bundle_sha256":digest(prior_raw),
        "original_critic_meta_sha256":digest(prior_meta_raw),
        "frozen_matched_critic_bundle_sha256":digest(pair_raw),
        "frozen_matched_critic_meta_sha256":digest(pair_meta_raw),
        "new_test":manifest,
        "fresh_source":{"x":x.tolist(),"y":y.tolist()},
        "teacher_seeds":pairing["original_teacher_seeds"],
        "n_test_sequences":len(x),
        "registered_scientific_verdict":verdict,
        "n_registered_seed_passes":passed_count,
        "results":output,
        "limitations":"External critics fitted to old terminal gradients; teacher weights frozen. Test labels used only for oracle derivative and post-edit native scoring."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print("LGA04_HELDOUT_COMPLETE",verdict,passed_count,flush=True)
    return record


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","prior-bundle","prior-meta",
                 "paired-bundle","paired-meta","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    evaluate(a.source_bundle,a.source_meta,a.prior_bundle,a.prior_meta,
             a.paired_bundle,a.paired_meta,a.out,a.smoke)


if __name__=="__main__":
    main()
