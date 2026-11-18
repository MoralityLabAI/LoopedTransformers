"""LGA-05 Stage B: source-frozen potential gradients, energy monotonicity and native causality.

Correct new test labels enter only retrospective local/terminal gradients
and original final CE scoring, AFTER source physical intervention plans.
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
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import (
    fixed_data,tensor_int64_sha,per_token_ce,head_rowspace
)
from mechinterp.loopscope05.fit_and_freeze import digest,load_frozen
from mechinterp.loopscope05.terminal_sae_test import (
    make_model,make_sae,active_components,terminal_gradients
)
from mechinterp.loopscope08.fit_potential import (
    load_potentials,potential_gradient,protocol_sha,OLD_SOURCES,
    TRAIN_SEED,VAL_SEED,TEACHER_SEEDS,SCHEMA as CAL_SCHEMA
)

SCHEMA="LGA-05/native-recurrent-energy-and-causal-fresh-test-v1"
TEST_SEED=31057
SMOKE_SEED=61057
N_TEST=256
N_CAUSAL=64
GOALS=(0,10,20,30,40,50,60,70)
BOOT=2000
DEPTHS=(0,1,2,3)
CAUSAL_DEPTHS=(1,2)
ARMS=("shared_down","conditioned_down","initial_down","radial_down",
      "native","sae_active","sae_remainder","head_null","random","oracle_down")
CONTRASTS=(("shared_down","random"),("shared_down","conditioned_down"),
           ("shared_down","native"),("sae_active","sae_remainder"),
           ("head_null","random"))


def seed(*parts):
    key="LGA05|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"big")&((1<<63)-1)


def new_test(smoke=False):
    test_seed=SMOKE_SEED if smoke else TEST_SEED
    n=8 if smoke else N_TEST
    x,y=fixed_data(n,test_seed)
    seen=set()
    for s,c in (*OLD_SOURCES,(TRAIN_SEED,256),(VAL_SEED,64)):
        prev,_=fixed_data(c,s)
        seen.update(tuple(row) for row in prev.tolist())
    source=[tuple(row) for row in x.tolist()]
    if len(set(source))!=n or any(r in seen for r in source):
        raise ValueError("new test strings duplicate old calibration/test sources")
    return x,y,{
        "seed":test_seed,"n_sequences":n,"x_sha256":tensor_int64_sha(x),
        "y_sha256":tensor_int64_sha(y),
        "independent_source_no_overlap":True
    }


def cosine(a,b):
    return (a*b).sum(dim=-1)/(
        a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-12)


def bootstrap(values,key,draws=BOOT):
    n=len(values)
    if n<2 or any(not math.isfinite(float(z)) for z in values):
        raise ValueError("requires finite independent whole-string clusters")
    rng=random.Random(key)
    samples=sorted(sum(values[rng.randrange(n)] for i in range(n))/n
                   for _ in range(draws))
    return [samples[int(.025*draws)],samples[int(.975*draws)-1]]


def radial_gradient(h,mu,std):
    return (h-mu)/(std*std)


def radial_energy(h,mu,std):
    return ((h-mu)/std).square().sum(dim=(1,2))*.5


def potential_drop(net,h,hnext,embed,mu,std):
    with torch.no_grad():
        before=net(h,embed,mu,std).sum(-1)
        after=net(hnext,embed,mu,std).sum(-1)
    return (before-after)/81


def profile(teacher,models,untrained,mu,std,x,y):
    """One result per whole independent sequence for each true recurrent pass."""
    records={str(k):[] for k in DEPTHS}
    for start in range(0,len(x),8):
        bx=x[start:start+8];by=y[start:start+8]
        states,terminal=terminal_gradients(teacher,bx,by)
        for k in DEPTHS:
            h=states[k]
            next_h=states[k+1]
            v=next_h-h
            _,g_shared=potential_gradient(models["shared"],h,0,mu,std)
            _,g_cond=potential_gradient(models["conditioned"],h,k,mu,std)
            _,g_init=potential_gradient(untrained,h,0,mu,std)
            g_rad=radial_gradient(h,mu,std)
            with torch.no_grad():
                logits=teacher.head(h)
                local=(torch.softmax(logits,-1)-
                       F.one_hot(by,10).float())@teacher.head.weight
            drop_shared=potential_drop(models["shared"],h,next_h,0,mu,std)
            drop_cond=potential_drop(models["conditioned"],h,next_h,k,mu,std)
            drop_init=potential_drop(untrained,h,next_h,0,mu,std)
            drop_rad=(radial_energy(h,mu,std)-
                      radial_energy(next_h,mu,std))/81
            cr=cosine(v,-g_rad)
            cs=cosine(v,-g_shared)
            ci=cosine(v,-g_init)
            cc=cosine(v,-g_cond)
            cterm=cosine(v,-terminal[k])
            clocal=cosine(v,-local)
            for i in range(len(bx)):
                records[str(k)].append({
                    "shared_cosine":float(cs[i].mean()),
                    "conditioned_cosine":float(cc[i].mean()),
                    "initial_cosine":float(ci[i].mean()),
                    "radial_cosine":float(cr[i].mean()),
                    "gain_shared_over_initial":float((cs[i]-ci[i]).mean()),
                    "gain_shared_over_radial":float((cs[i]-cr[i]).mean()),
                    "shared_energy_drop":float(drop_shared[i]),
                    "conditioned_energy_drop":float(drop_cond[i]),
                    "initial_energy_drop":float(drop_init[i]),
                    "radial_energy_drop":float(drop_rad[i]),
                    "shared_positive_drop":float(drop_shared[i]>0),
                    "conditioned_positive_drop":float(drop_cond[i]>0),
                    "native_terminal_gradient_cosine":float(cterm[i].mean()),
                    "native_current_head_gradient_cosine":float(clocal[i].mean()),
                })
    if any(len(rows)!=len(x) for rows in records.values()):
        raise ValueError("per-sequence four-loop source population incomplete")
    return records


def summarize_profiles(raw,sid,smoke=False):
    out={}
    npositive=0
    for k in DEPTHS:
        rows=raw[str(k)]
        means={name:sum(x[name] for x in rows)/len(rows) for name in rows[0]}
        intervals={}
        for field in ("gain_shared_over_initial","gain_shared_over_radial",
                      "shared_energy_drop","shared_positive_drop"):
            intervals[field]=bootstrap(
                [z[field] for z in rows],seed(45057,sid,k,field),
                draws=100 if smoke else BOOT)
        passed=(intervals["gain_shared_over_initial"][0]>0 and
                intervals["gain_shared_over_radial"][0]>0 and
                intervals["shared_energy_drop"][0]>0 and
                intervals["shared_positive_drop"][0]>.50)
        npositive+=int(passed)
        out[str(k)]={"means":means,"bootstrap95":intervals,
                      "n_independent_sequences":len(rows),
                      "registered_depth_pass":bool(passed),
                      "per_sequence":rows}
    n=len(raw["0"])
    all_monotonic=sum(int(all(raw[str(k)][i]["shared_positive_drop"]==1.
                              for k in DEPTHS)) for i in range(n))/n
    return out,npositive,all_monotonic


def normalize_direction(v,r):
    norm=float(v.norm())
    if norm<=1e-8 or r<=1e-8:return None
    edit=(v*(r/norm)).detach().float()
    if abs(float(edit.norm())-r)>1e-5:
        raise ArithmeticError("native edits have differing physical L2 dose")
    return edit


def physical_sha(z):
    values=z.detach().float().reshape(-1).tolist()
    return digest(struct.pack("<"+"f"*len(values),*values))


def native(teacher,models,initial,mu,std,saes,x,y,sid,smoke=False):
    """Five predeclared contrasts with full true native recurrent continuation."""
    amount=min(4 if smoke else N_CAUSAL,len(x))
    head_basis,_=head_rowspace(teacher)
    output={}
    for k in CAUSAL_DEPTHS:
        sae,saemu,saestd=saes[k]
        source_cases=[]
        for i in range(amount):
            bx=x[i:i+1];by=y[i:i+1]
            with torch.no_grad():
                states=teacher.reason_states(bx)
            h=states[k][0].detach()
            update=states[k+1][0]-h
            _,g_shared=potential_gradient(models["shared"],h.unsqueeze(0),0,mu,std)
            _,g_cond=potential_gradient(models["conditioned"],h.unsqueeze(0),k,mu,std)
            _,g_init=potential_gradient(initial,h.unsqueeze(0),0,mu,std)
            gradient_rad=radial_gradient(h,mu,std)
            parts=active_components(sae,saemu,saestd,h,update)
            hn=update.double()-(update.double()@head_basis)@head_basis.T
            candidates=[]
            sources=[]
            for j,tok in enumerate(GOALS):
                radius=.25*float(update[tok].norm())
                generator=torch.Generator().manual_seed(seed("isotropic",sid,k,i,j))
                rand=torch.randn(48,generator=generator,dtype=torch.float32)
                directions={
                    "shared_down":-g_shared[0,tok],
                    "conditioned_down":-g_cond[0,tok],
                    "initial_down":-g_init[0,tok],
                    "radial_down":-gradient_rad[tok],
                    "native":update[tok],
                    "sae_active":parts["active"][tok],
                    "sae_remainder":parts["remainder"][tok],
                    "head_null":hn[tok].float(),
                    "random":rand,
                }
                admitted=bool(parts["admitted"][tok]) and radius>1e-8 and all(
                    float(value.norm())>1e-8 for value in directions.values())
                cell={"token":tok,"slot":j,"dose":radius,
                      "source_admitted":admitted,"arms":{}}
                if admitted:
                    for name,v in directions.items():
                        edit=normalize_direction(v,radius)
                        if edit is None:raise ArithmeticError("source-admitted physical arm vanished")
                        candidates.append((j,tok,name,edit))
                else:cell["reason"]="degenerate_SAE_headnull_or_scalar_gradient"
                sources.append(cell)
            # Oracle labels revealed only after frozen label-blind arm choices.
            _,future=terminal_gradients(teacher,bx,by)
            oracle=future[k][0]
            for j,tok in enumerate(GOALS):
                if sources[j]["source_admitted"]:
                    edit=normalize_direction(-oracle[tok],sources[j]["dose"])
                    if edit is not None:candidates.append((j,tok,"oracle_down",edit))
                    else:sources[j]["oracle_available"]=False
            batch=[h]
            for j,tok,name,edit in candidates:
                tmp=h.clone();tmp[tok]+=edit
                batch.append(tmp)
            z=torch.stack(batch)
            with torch.no_grad():
                for _ in range(4-k):z=teacher.block(z)
                logits=teacher.head(z)
                losses=per_token_ce(logits,by.expand(len(z),-1))
                clean=teacher.head(states[-1])[0]
                closure=float((logits[0]-clean).abs().max())
            if closure>1e-5:
                raise ArithmeticError("source native clean 4-k replay changed")
            baseline=losses[0]
            clean_logits=logits[0]
            for index,(j,tok,name,edit) in enumerate(candidates,1):
                truth=int(by[0,tok])
                opponent=clean_logits[tok].clone()
                opponent[truth]=-float("inf")
                rival=int(opponent.argmax())
                final=logits[index,tok]
                edited=h.clone()
                edited[tok]+=edit
                with torch.no_grad():
                    e0=float(models["shared"](h.unsqueeze(0),0,mu,std).sum())
                    e1=float(models["shared"](edited.unsqueeze(0),0,mu,std).sum())
                sources[j]["arms"][name]={
                    "edit_L2":float(edit.norm()),
                    "edit_sha256":physical_sha(edit),
                    "truth_digit":truth,"clean_rival":rival,
                    "final_goal_logits":final.tolist(),
                    "clean_goal_logits":clean_logits[tok].tolist(),
                    "all_final_token_CE":losses[index].tolist(),
                    "clean_all_final_token_CE":baseline.tolist(),
                    "full_sequence_CE":float(losses[index].mean()),
                    "clean_full_sequence_CE":float(baseline.mean()),
                    "global_CE_change":float(losses[index].mean()-baseline.mean()),
                    "goal_CE_change":float(losses[index,tok]-baseline[tok]),
                    "goal_margin_change":float(
                        (final[truth]-final[rival])-
                        (clean_logits[tok,truth]-clean_logits[tok,rival])),
                    "shared_E_before":e0,"shared_E_after":e1,
                    "physical_edit_E_decrease":(e0-e1)/81,
                    "true_terminal_linear_global_CE_prediction":float(
                        oracle[tok].double().dot(edit.double())/81),
                    "learned_shared_linear_E_prediction":float(
                        g_shared[0,tok].double().dot(edit.double())/81)
                }
            source_cases.append({"test_sequence_index":i,"goals":sources})
        output[str(k)]=source_cases
    return output


def causal_summary(raw,sid,smoke=False):
    result={}
    for k in CAUSAL_DEPTHS:
        rows=[]
        for x in raw[str(k)]:
            admitted=[z for z in x["goals"] if z["source_admitted"]]
            cells={}
            for a,b in CONTRASTS:
                name=f"{a}_minus_{b}"
                differences=[v["arms"][a]["global_CE_change"]-
                             v["arms"][b]["global_CE_change"] for v in admitted]
                cells[name]=sum(differences)/len(differences) if differences else 0.
            rows.append({"test_sequence_index":x["test_sequence_index"],
                         "n_original_goals":8,
                         "n_admitted_goals":len(admitted),
                         "paired_contrasts":cells})
        summary={}
        for a,b in CONTRASTS:
            name=f"{a}_minus_{b}"
            values=[z["paired_contrasts"][name] for z in rows]
            ci=bootstrap(values,seed("causal",45057,sid,k,name),
                         draws=100 if smoke else BOOT)
            summary[name]={"mean":sum(values)/len(values),"bootstrap95":ci,
                           "nominal_direction":"POSITIVE" if ci[0]>0 else
                           "NEGATIVE" if ci[1]<0 else "INCONCLUSIVE"}
        result[str(k)]={
            "n_independent_source_sequences":len(rows),
            "n_original_goals":8*len(rows),
            "n_admitted_goal_cases":sum(z["n_admitted_goals"] for z in rows),
            "per_sequence":rows,"contrasts":summary,"raw_cases":raw[str(k)]
        }
    return result


def curl_witness(model,x,sid,smoke=False):
    """Finite-ε antisymmetric Jacobian bilinear on the 81x48 state space."""
    source_count=min(4 if smoke else N_CAUSAL,len(x))
    double_model=copy.deepcopy(model).double().eval()
    cells={}
    eps=0.001
    with torch.no_grad():
        for k in CAUSAL_DEPTHS:
            entries=[]
            for i in range(source_count):
                states=model.reason_states(x[i:i+1])
                h=states[k].double().detach()
                g=torch.Generator().manual_seed(seed("curl",sid,k,i))
                u=torch.randn(h.shape,generator=g,dtype=torch.float64)
                w=torch.randn(h.shape,generator=g,dtype=torch.float64)
                u=u/u.norm()
                w=w-(w*u).sum()*u
                w=w/w.norm()
                batch=torch.cat((h+eps*u,h-eps*u,h+eps*w,h-eps*w),dim=0)
                out=double_model.block(batch)-batch
                ju=(out[0]-out[1])/(2*eps)
                jw=(out[2]-out[3])/(2*eps)
                a=float((u*jw).sum())
                b=float((w*ju).sum())
                entries.append({"source_index":i,
                                "u_dot_J_w":a,"w_dot_J_u":b,
                                "skew_difference":a-b,
                                "skew_ratio":abs(a-b)/(abs(a)+abs(b)+1e-8)})
            cells[str(k)]={"n_source_sequences":len(entries),
                           "mean_abs_skew_ratio":sum(z["skew_ratio"] for z in entries)/len(entries),
                           "per_sequence":entries}
    return cells


def evaluate(source_bundle,source_meta,potential_bundle,potential_meta,out,smoke=False):
    torch.set_num_threads(2)
    frozen_raw=Path(source_bundle).read_bytes()
    frozen_meta=Path(source_meta).read_bytes()
    potential_raw=Path(potential_bundle).read_bytes()
    potential_meta_raw=Path(potential_meta).read_bytes()
    frozen_manifest,frozen=load_frozen(source_bundle,source_meta)
    meta=json.loads(potential_meta_raw)
    if (meta["schema"]!=CAL_SCHEMA or meta["protocol_sha256"]!=protocol_sha() or
        meta["original_teacher_source_bundle_sha256"]!=digest(frozen_raw) or
        meta["original_teacher_source_manifest_sha256"]!=digest(frozen_meta) or
        meta["source_potential_bundle_sha256"]!=digest(potential_raw) or
        meta["calibration_splits"]["new_test_generated"] is not False or
        meta["test_labels_read"] is not False):
        raise ValueError("new test opened before potential source freeze")
    if not smoke and (meta["mode"]!="full" or frozen_manifest["mode"]!="full"):
        raise ValueError("scientific study requires original full source models")
    x,y,test=new_test(smoke)
    participants={}
    npositive=0
    for sid in meta["teacher_seeds"]:
        teacher=make_model(frozen,sid)
        models,initial,mu,std,scale=load_potentials(potential_bundle,potential_meta,sid)
        r=profile(teacher,models,initial,mu,std,x,y)
        summary,n_depth,whole_monotone=summarize_profiles(r,sid,smoke)
        saes={k:make_sae(frozen,sid,k) for k in CAUSAL_DEPTHS}
        edits=native(teacher,models,initial,mu,std,saes,x,y,sid,smoke)
        contrasts=causal_summary(edits,sid,smoke)
        curl=curl_witness(teacher,x,sid,smoke)
        passed=n_depth>=3
        npositive+=int(passed)
        participants[str(sid)]={
            "trajectory_by_depth":summary,
            "n_positive_depths":n_depth,
            "registered_teacher_pass":passed,
            "fraction_all_four_energy_decrease":whole_monotone,
            "native_causal":contrasts,
            "integrability_descriptive":curl,
        }
        print("LGA05_EVALUATED",sid,"native_shared_gradient_cos",[
            round(summary[str(k)]["means"]["shared_cosine"],4) for k in DEPTHS],
            "native_energy_drop",[
            round(summary[str(k)]["means"]["shared_energy_drop"],4) for k in DEPTHS],
            "all_four_monotone",round(whole_monotone,4),
            "passed",passed,flush=True)
    verdict="PILOT_ONLY" if smoke else "POSITIVE" if npositive>=2 else "FAIL"
    receipt={
        "schema":SCHEMA,"mode":"smoke" if smoke else "full",
        "protocol_sha256":protocol_sha(),
        "original_source_bundle_sha256":digest(frozen_raw),
        "original_source_manifest_sha256":digest(frozen_meta),
        "frozen_potential_bundle_sha256":digest(potential_raw),
        "frozen_potential_manifest_sha256":digest(potential_meta_raw),
        "new_test":test,"test_source":{"x":x.tolist(),"y":y.tolist()},
        "teacher_seeds":meta["teacher_seeds"],
        "n_independent_new_sequences":len(x),
        "registered_scientific_status":verdict,
        "n_registered_teacher_passes":npositive,
        "results":participants,
        "limits":"External fitted scalar energy, not an explicitly computed potential in original teacher. Curl witness descriptive, no preregistered threshold."
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("LGA05_ORIGINAL_TEST_DONE",verdict,npositive,flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    for name in ("source-bundle","source-meta","potential-bundle","potential-meta","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    evaluate(a.source_bundle,a.source_meta,a.potential_bundle,a.potential_meta,a.out,a.smoke)


if __name__=="__main__":
    main()
