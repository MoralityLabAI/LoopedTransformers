"""LGA-08: frozen native k2->k4 pair-fiber future-gradient-matched donor study.

Prospectively registered in PROTOCOL.md @538f01ad. This executable opens
reserved seed 31062 ONLY after the original model and code-byte pretest lock.
All scored edit arms and every rejected source-goal case remain in the first
immutable development receipt. Three original teachers are exposed roots.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import torch
import torch.nn.functional as F

from mechinterp.loopscope04.gradient_flow import fixed_data,tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import load_frozen
from mechinterp.loopscope05.terminal_sae_test import make_model
from mechinterp.loopscope08.fit_potential import load_potentials,OLD_SOURCES
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.energy_fiber import source_energy_gradient
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import (
    BASE_COUNT,RESERVED_SEED,GOALS,TEACHERS,SHAMS,ARMS,stable_seed,plan_pair
)
from mechinterp.loopscope11.source_lock import verify_lock,sha_file,ORIGINAL

SCHEMA="LGA08/first-native-pair-gradient-matched-development-v1"
PREVIOUS=(*OLD_SOURCES,(31055,256),(31056,64),(31057,256),
          (31058,256),(31060,64),(31061,64))
BOOT=2000
PAIRS_PER_BASE=len(GOALS)
PAIR_ELEMS=2*48
FULL_ELEMS=81*48
EDIT_BYTES=PAIR_ELEMS*4
SOURCE_BYTES=(PAIR_ELEMS*3+FULL_ELEMS)*4


def raw_f32(t):
    x=t.detach().cpu().contiguous().float().reshape(-1).tolist()
    return struct.pack("<"+"f"*len(x),*x)


def boot(vals,sid,comparison):
    if len(vals)!=BASE_COUNT or any(not math.isfinite(float(x)) for x in vals):
        raise ValueError("invalid prospective whole-base bootstrap population")
    rng=random.Random(stable_seed("bootstrap",sid,comparison))
    v=sorted(sum(vals[rng.randrange(BASE_COUNT)] for _ in range(BASE_COUNT))/BASE_COUNT
             for _ in range(BOOT))
    return [v[int(.025*BOOT)],v[int(.975*BOOT)-1]]


def median8(a):
    a=sorted(float(x) for x in a)
    if len(a)!=8:
        raise ValueError("registered eight-match comparison changed")
    return (a[3]+a[4])/2


def new_roster():
    """One-time 31062 PyTorch generator, AFTER exact frozen source lock."""
    x,y=fixed_data(BASE_COUNT,RESERVED_SEED)
    old=set()
    for seed,size in PREVIOUS:
        v,_=fixed_data(size,seed)
        old.update(tuple(r) for r in v.tolist())
    seen=set();roster=[]
    for i in range(BASE_COUNT):
        r=tuple(x[i].tolist())
        if r in old or r in seen:
            raise ValueError("new 31062 source overlaps old LGA IID source")
        seen.add(r)
        donors=[]
        for j in GOALS:
            d,info=legal_predecessor_donor(x[i],j)
            dd=tuple(d.tolist())
            if dd in old:
                raise ValueError("legal donor overlaps old source")
            donors.append({"goal":j,"donor_x":d.tolist(),**info})
        roster.append({"base_index":i,"x":list(r),"y":y[i].tolist(),"donors":donors})
    return x,y,{
        "seed":RESERVED_SEED,"n_independent_bases":BASE_COUNT,
        "ordered_base_indices":list(range(BASE_COUNT)),
        "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
        "sources":roster,
    }


def score_batch(model,energy,mu,std,head,rec_h,don_h,donor_info, *,
                sid,base_id,edit_file,source_file,
                edit_digest,source_digest,edit_index,source_index,
                clean_rec4,clean_don4):
    """Freeze all source/model directions before computing any edited correctness."""
    egrad=source_energy_gradient(energy,rec_h,mu,std).detach()
    plans=[]
    for j,goal in enumerate(GOALS):
        p=plan_pair(
            model,energy,rec_h,don_h[j:j+1],mu,std,
            goal=goal,
            old_class=int(donor_info[j]["original_goal_label"]),
            donor_class=int(donor_info[j]["donor_goal_label"]),
            teacher_seed=sid,base_index=base_id,
            head_projector=head.projector,cached_egrad=egrad
        )
        plans.append(p)

    count=1+PAIRS_PER_BASE*len(ARMS)
    physical=rec_h.detach().expand(count,-1,-1).clone()
    for j,goal in enumerate(GOALS):
        p=(goal-1)%81
        for a,arm in enumerate(ARMS):
            idx=1+j*len(ARMS)+a
            if arm=="full_donor_state":
                physical[idx]=don_h[j]  # exact naturally reached whole donor
            else:
                edit=plans[j]["directions"][arm].float()
                physical[idx,p]+=edit[0]
                physical[idx,goal]+=edit[1]

    with torch.no_grad():
        readouts=model.head(physical)
        e_before=energy(physical,0,mu,std).double().sum(-1)
        z=physical
        for _ in range(2):
            z=model.block(z)
        logits=model.head(z)
        preds=logits.argmax(-1)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(e_before).all()):
        raise ArithmeticError("nonfinite native pair model inference")
    if float((logits[0]-clean_rec4[0]).abs().max())>1e-4:
        raise ArithmeticError("clean native k2->4 closure mismatch")
    records=[]
    for j,goal in enumerate(GOALS):
        p=(goal-1)%81
        source=plans[j]
        f=source["fiber"]
        truth=int(donor_info[j]["donor_goal_label"])
        old_target=int(donor_info[j]["original_goal_label"])
        end_logits=logits[0,goal].detach().double()
        baseline_competitor=max((c for c in range(10) if c!=truth),
                                key=lambda c:float(end_logits[c]))
        source_raw_parts=(
            raw_f32(f.donor),raw_f32(f.egrad),
            raw_f32(source["future_gradient"]),
            raw_f32((don_h[j]-rec_h[0]).detach())
        )
        source_raw=b"".join(source_raw_parts)
        if len(source_raw)!=SOURCE_BYTES:
            raise ArithmeticError("source state/gradient byte contract differs")
        source_file.write(source_raw)
        source_digest.update(source_raw)
        old_clean_correct=bool(int(clean_rec4[0,goal].argmax())==old_target)
        don_clean_correct=bool(int(clean_don4[j,goal].argmax())==truth)
        rec={
            "base_index":base_id,"goal_slot":j,"goal":goal,
            "pair_positions":[p,goal],
            "donor_target":truth,"original_target":old_target,
            "source_index":source_index,
            "pair_delta_sha256":hashlib.sha256(source_raw_parts[0]).hexdigest(),
            "energy_pair_gradient_sha256":hashlib.sha256(source_raw_parts[1]).hexdigest(),
            "future_margin_pair_gradient_sha256":hashlib.sha256(source_raw_parts[2]).hexdigest(),
            "full_donor_state_delta_sha256":hashlib.sha256(source_raw_parts[3]).hexdigest(),
            "admitted":bool(source["eligible"]),"exclusion_reason":source["reason"],
            "head_rank":int(f.current_head_rank),"fiber_dimension":int(f.dimension),
            "energy_row_active":bool(f.energy_active),"natural_pair_radius":float(f.natural_radius),
            "projected_pair_fraction":float(f.tangent_fraction),
            "head_only_pair_fraction":float(f.head_fraction),
            "tangent_head_residual":float(f.head_residual),
            "tangent_energy_residual":float(f.tangent_residual),
            "clean_future_margin":source["clean_future_margin"],
            "selected_signed_derivative":float(source["selected_signed_derivative"]),
            "clean_recipient_correct":old_clean_correct,
            "clean_donor_correct":don_clean_correct,
            "joint_clean_competence":bool(old_clean_correct and don_clean_correct),
            "clean_goal_logits":end_logits.tolist(),
            "clean_donor_all_logits":clean_don4[j].tolist(),
            "clean_predicted_classes":preds[0].tolist(),
            "arms":{},
        }
        source_index+=1
        for a,arm in enumerate(ARMS):
            idx=1+j*len(ARMS)+a
            if arm=="full_donor_state":
                # Complete 81-token donor edit has its own natural large dose;
                # pair binary ledger holds only its two registered positions.
                whole_edit=(physical[idx]-physical[0]).detach()
                actual=whole_edit[[p,goal]]
                physical_radius=float(whole_edit.norm())
            else:
                actual=(physical[idx,[p,goal]]-physical[0,[p,goal]]).detach()
                physical_radius=float(actual.norm())
            raw=raw_f32(actual)
            if len(raw)!=EDIT_BYTES:
                raise ArithmeticError("edited physical pair must be 96 float32 values")
            edit_file.write(raw)
            edit_digest.update(raw)
            out_logit=logits[idx,goal].detach().double()
            head_change=float((readouts[idx]-readouts[0]).norm())
            signed=float((source["future_gradient"]*actual.double()).sum())
            data={
                "edit_index":edit_index,"pair_edit_sha256":hashlib.sha256(raw).hexdigest(),
                "actual_pair_l2":float(actual.norm()),
                "actual_full_l2":physical_radius,
                "signed_future_margin_derivative":signed,
                "current_head_alltoken_l2":head_change,
                "finite_shared_E_change":float(e_before[idx]-e_before[0]),
                "goal_logits":out_logit.tolist(),
                "predicted_classes":preds[idx].tolist(),
                "donor_correct":bool(int(preds[idx,goal])==truth),
                "donor_goal_CE":float(-F.log_softmax(out_logit,dim=-1)[truth]),
                "donor_margin_clean_rival":float(out_logit[truth]-out_logit[baseline_competitor]),
                "clean_rival":baseline_competitor,
            }
            if arm=="full_donor_state":
                data["all_final_logits"]=logits[idx].tolist()
                rec["full_donor_max_logit_error"]=float(
                    (logits[idx]-clean_don4[j]).abs().max())
            rec["arms"][arm]=data
            edit_index+=1
        records.append(rec)
    return records,edit_index,source_index


def summarize(cases,sid):
    n=BASE_COUNT*len(GOALS)
    if len(cases)!=n:
        raise ValueError("original frozen donor-goal population missing cases")
    by={"random":{i:[] for i in range(BASE_COUNT)},
        "gradient":{i:[] for i in range(BASE_COUNT)}}
    admitted=joint=pairpositive=fullpositive=0
    max_full_error=0.
    counts={a:0 for a in ARMS}
    for row in cases:
        goal=row["goal"];idx=row["base_index"]
        if idx not in by["random"] or goal not in GOALS:
            raise ValueError("source-goal index changed")
        sel=float(row["arms"]["pair_energy_tangent"]["donor_correct"])
        r=median8([row["arms"][f"pair_random_k_{i}"]["donor_correct"]
                   for i in range(SHAMS)])
        g=median8([row["arms"][f"pair_gradient_match_{i}"]["donor_correct"]
                   for i in range(SHAMS)])
        by["random"][idx].append(sel-r)
        by["gradient"][idx].append(sel-g)
        admitted+=int(row["admitted"])
        is_competent=int(row["joint_clean_competence"])
        joint+=is_competent
        pairpositive+=is_competent*int(row["arms"]["pair_natural"]["donor_correct"])
        fullpositive+=is_competent*int(row["arms"]["full_donor_state"]["donor_correct"])
        max_full_error=max(max_full_error,float(row["full_donor_max_logit_error"]))
        for a in ARMS:
            counts[a]+=int(row["arms"][a]["donor_correct"])
    stats={}
    for comp in ("gradient","random"):
        if any(len(by[comp][i])!=len(GOALS) for i in range(BASE_COUNT)):
            raise ValueError("one independent source lost registered goals")
        base=[sum(by[comp][i])/len(GOALS) for i in range(BASE_COUNT)]
        stats[comp]={
          "paired_ITT_effect":sum(base)/BASE_COUNT,
          "whole_base_bootstrap95":boot(base,sid,comp),
          "base_level_paired_effects":base,
          "n_independent_base_sources":BASE_COUNT,
          "bootstrap_draws":BOOT,
        }
    good=bool(
      joint/n>=.80 and max_full_error<=1e-4 and
      fullpositive/max(joint,1)>=.95 and pairpositive/max(joint,1)>=.90 and
      stats["gradient"]["paired_ITT_effect"]>=.10 and
      stats["gradient"]["whole_base_bootstrap95"][0]>0 and
      stats["random"]["paired_ITT_effect"]>=.15 and
      stats["random"]["whole_base_bootstrap95"][0]>0
    )
    return {
      "n_original_source_goals":n,"n_admitted":admitted,
      "n_jointly_competent":joint,"joint_competence_fraction":joint/n,
      "full_donor_correct_among_competent":fullpositive/max(1,joint),
      "natural_pair_correct_among_competent":pairpositive/max(1,joint),
      "max_full_donor_parity_error":max_full_error,
      "arm_correct_counts":counts,
      "contrasts":stats,
      "registered_strong_teacher_gate":good,
    }


def run(teacher_bundle,teacher_meta,potential_bundle,potential_meta,lock,
        out,edit_path,source_path):
    if sys.byteorder!="little":
        raise RuntimeError("registered binary float32 source requires little-endian")
    torch.set_num_threads(2)
    frozen=verify_lock(lock,teacher_bundle,teacher_meta,potential_bundle,potential_meta)
    # The first and ONLY prospective 31062 source read follows exact lock verify.
    x,y,roster=new_roster()
    _,teacher_source=load_frozen(teacher_bundle,teacher_meta)
    edit_path=Path(edit_path);source_path=Path(source_path);out=Path(out)
    for p in (edit_path,source_path,out):
        p.parent.mkdir(parents=True,exist_ok=True)
    results={}
    edit_index=source_index=0
    edith=hashlib.sha256();sourceh=hashlib.sha256()
    with gzip.open(edit_path,"wb",compresslevel=5) as edit_file, gzip.open(
            source_path,"wb",compresslevel=5) as source_file:
        for sid in TEACHERS:
            model=make_model(teacher_source,sid).eval().requires_grad_(False)
            potential_models,_,mu,std,_=load_potentials(
                potential_bundle,potential_meta,sid)
            energy=potential_models["shared"].eval().requires_grad_(False)
            head=full_head_kernel(model.head.weight)
            root={
                "head_weight":model.head.weight.detach().double().tolist(),
                "head_projector":head.projector.tolist(),
                "cases":[],
            }
            for base in range(BASE_COUNT):
                rec=x[base:base+1]
                donors_info=roster["sources"][base]["donors"]
                dons=torch.tensor([d["donor_x"] for d in donors_info],dtype=torch.int64)
                with torch.no_grad():
                    rec_states=model.reason_states(rec)
                    donor_states=model.reason_states(dons)
                    rec4=model.head(rec_states[4])
                    don4=model.head(donor_states[4])
                cases,edit_index,source_index=score_batch(
                    model,energy,mu,std,head,rec_states[2],donor_states[2],
                    donors_info,sid=sid,base_id=base,edit_file=edit_file,
                    source_file=source_file,edit_digest=edith,source_digest=sourceh,
                    edit_index=edit_index,source_index=source_index,
                    clean_rec4=rec4,clean_don4=don4)
                root["cases"].extend(cases)
            root["summary"]=summarize(root["cases"],sid)
            results[str(sid)]=root
            print("LGA08_FIRST_TEACHER",sid,
                  "signed_match_effect",root["summary"]["contrasts"]["gradient"]["paired_ITT_effect"],
                  "random_effect",root["summary"]["contrasts"]["random"]["paired_ITT_effect"],
                  "strict_gate",root["summary"]["registered_strong_teacher_gate"],flush=True)
    passing=sum(int(results[str(s)]["summary"]["registered_strong_teacher_gate"])
                for s in TEACHERS)
    verdict=("DEVELOPMENT_STRONG_INCREMENTAL_PASS" if passing>=2 else
             "DEVELOPMENT_STRONG_INCREMENTAL_FAIL")
    report={
        "schema":SCHEMA,"evidence_class":"EXPOSED_OLD_TEACHERS_DEVELOPMENT_ONLY",
        "original_artifact_sha256":ORIGINAL,
        "first_pretest_lock_sha256":sha_file(lock),
        "protocol_sha256":frozen["protocol_sha256"],
        "implementation_sha256":frozen["implementation_sha256"],
        "source":roster,"teachers":results,"teacher_seeds":list(TEACHERS),
        "n_independent_bases":BASE_COUNT,"goals":list(GOALS),
        "arms":list(ARMS),"n_arms":len(ARMS),
        "source_vectors":{
            "source_stride_bytes":SOURCE_BYTES,"n_records":source_index,
            "raw_sha256":sourceh.hexdigest(),"gzip_sha256":sha_file(source_path)
        },
        "physical_pair_edits":{
            "stride_bytes":EDIT_BYTES,"n_records":edit_index,
            "raw_sha256":edith.hexdigest(),"gzip_sha256":sha_file(edit_path)
        },
        "n_teachers_passing_strong_incremental":passing,
        "first_registered_development_verdict":verdict,
        "torch_version":torch.__version__,
    }
    out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("LGA08_FIRST_NATIVE_VERDICT",verdict,"SHA256",sha_file(out),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
                 "source-lock","out","edits","sources"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.teacher_bundle,a.teacher_meta,a.potential_bundle,a.potential_meta,
        a.source_lock,a.out,a.edits,a.sources)


if __name__=="__main__":
    main()
