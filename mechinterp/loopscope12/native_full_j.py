"""LGA-09 first source-locked native 75D pair donor vs full-Jacobian matched shams.

No 31063 generation before source_lock.verify. First prospective protocol first
commit 5719f8d2; model roots are frozen/exposed development checkpoints.
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
from mechinterp.loopscope12.full_jacobian import (
    NEW_SEED,N_BASE,TEACHERS,GOALS,ARMS,SHAMS,plan_full_j,stable_seed
)
from mechinterp.loopscope12.source_lock import verify,ORIGINAL,file_sha

SCHEMA="LGA09/first-native-full-ten-logit-jacobian-pair-development-v1"
BOOT=2000
EDIT_BYTES=2*48*4
SOURCE_BYTES=(3*2*48+81*48+10*2*48)*4
PREVIOUS=(*OLD_SOURCES,(31055,256),(31056,64),(31057,256),
          (31058,256),(31060,64),(31061,64),(31062,64))


def raw_f32(t):
    x=t.detach().cpu().contiguous().float().reshape(-1).tolist()
    return struct.pack("<"+"f"*len(x),*x)


def bootstrap(v,sid,contrast):
    if len(v)!=N_BASE or any(not math.isfinite(float(x)) for x in v):
        raise ValueError("invalid registered independent base-source bootstrap")
    rng=random.Random(stable_seed("bootstrap",sid,contrast))
    z=sorted(sum(v[rng.randrange(N_BASE)] for _ in range(N_BASE))/N_BASE
             for _ in range(BOOT))
    return [z[int(.025*BOOT)],z[int(.975*BOOT)-1]]


def median8(x):
    x=sorted(float(v) for v in x)
    if len(x)!=8:
        raise ValueError("exact eight source-frozen shams required")
    return (x[3]+x[4])/2.


def new_source():
    """First 31063 generation allowed ONLY after exact immutable lock verified."""
    x,y=fixed_data(N_BASE,NEW_SEED)
    prev=set()
    for seed,count in PREVIOUS:
        old,_=fixed_data(count,seed)
        prev.update(tuple(r) for r in old.tolist())
    strings=[tuple(r) for r in x.tolist()]
    if len(set(strings))!=N_BASE or any(r in prev for r in strings):
        raise ValueError("first 31063 source duplicates earlier generated LGA sources")
    seen_donors=set();roster=[]
    for i in range(N_BASE):
        donors=[]
        for goal in GOALS:
            d,info=legal_predecessor_donor(x[i],goal)
            key=tuple(d.tolist())
            if key in prev or key in seen_donors:
                raise ValueError("first legal donor duplicates old/new source")
            seen_donors.add(key)
            donors.append({"goal":goal,"donor_x":d.tolist(),**info})
        roster.append({"base_index":i,"x":x[i].tolist(),"y":y[i].tolist(),
                       "donors":donors})
    return x,y,{
        "seed":NEW_SEED,"n_independent_bases":N_BASE,
        "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
        "source_base_ids":list(range(N_BASE)),
        "cases":roster
    }


def score_base(model,energy,mu,std,head,hrec,hdon,donor_info,
               rec4,don4,*,sid,base,edit_writer,source_writer,
               edit_hash,source_hash,edit_index,source_index):
    egrad=source_energy_gradient(energy,hrec,mu,std).detach()
    # Entire source/model-only plan fixed before any edited task outcome.
    plans=[]
    for j,goal in enumerate(GOALS):
        plans.append(plan_full_j(
            model,energy,hrec,hdon[j:j+1],mu,std,
            goal=goal,
            original_class=int(donor_info[j]["original_goal_label"]),
            donor_class=int(donor_info[j]["donor_goal_label"]),
            teacher_seed=sid,base_id=base,head_projector=head.projector,
            cached_egrad=egrad
        ))
    count=1+len(GOALS)*len(ARMS)
    physical=hrec.detach().expand(count,-1,-1).clone()
    for j,goal in enumerate(GOALS):
        p=(goal-1)%81
        for a,arm in enumerate(ARMS):
            idx=1+j*len(ARMS)+a
            if arm=="full_donor_state":
                physical[idx]=hdon[j]
            else:
                edit=plans[j]["directions"][arm].float()
                physical[idx,p]+=edit[0]
                physical[idx,goal]+=edit[1]
    with torch.no_grad():
        current=model.head(physical)
        E=energy(physical,0,mu,std).double().sum(-1)
        future=physical
        for _ in range(2):
            future=model.block(future)
        final=model.head(future)
        pred=final.argmax(-1)
    if not bool(torch.isfinite(final).all() and torch.isfinite(E).all()):
        raise ArithmeticError("native two-block 10-logit inference nonfinite")
    if float((final[0]-rec4[0]).abs().max())>1e-4:
        raise ArithmeticError("clean original native h2-to-h4 output mismatch")
    records=[]
    for j,goal in enumerate(GOALS):
        p=(goal-1)%81
        plan=plans[j];parent=plan["pair_plan"];fiber=parent["fiber"]
        grad=parent["future_gradient"]
        if plan["future"] is not None:
            J=plan["future"].full_jacobian.detach().double()
            raw_singular=plan["future"].singular_values.tolist()
        else:
            J=torch.zeros((10,2,48),dtype=torch.float64)
            raw_singular=[]
        src_parts=[
            raw_f32(fiber.donor),
            raw_f32(fiber.egrad),
            raw_f32(grad),
            raw_f32((hdon[j]-hrec[0]).detach()),
            raw_f32(J),
        ]
        source_raw=b"".join(src_parts)
        if len(source_raw)!=SOURCE_BYTES:
            raise ArithmeticError("registered LGA09 original source vector size invalid")
        source_writer.write(source_raw)
        source_hash.update(source_raw)
        truth=int(donor_info[j]["donor_goal_label"])
        old=int(donor_info[j]["original_goal_label"])
        clean_logits=final[0,goal].detach().double()
        rival=max((k for k in range(10) if k!=truth),
                  key=lambda k:float(clean_logits[k]))
        clean_rec=bool(int(rec4[0,goal].argmax())==old)
        clean_don=bool(int(don4[j,goal].argmax())==truth)
        c={
            "base_index":base,"goal_slot":j,"goal":goal,"pair_positions":[p,goal],
            "donor_target":truth,"original_target":old,
            "source_index":source_index,
            "source_sha256":{
                "pair_delta":hashlib.sha256(src_parts[0]).hexdigest(),
                "energy_pair_gradient":hashlib.sha256(src_parts[1]).hexdigest(),
                "signed_future_margin_gradient":hashlib.sha256(src_parts[2]).hexdigest(),
                "full_donor_delta":hashlib.sha256(src_parts[3]).hexdigest(),
                "full_ten_logit_jacobian":hashlib.sha256(src_parts[4]).hexdigest(),
            },
            "admitted":bool(plan["eligible"]),"reason":plan["reason"],
            "pair_radius":float(plan["natural_pair_radius"]),
            "current_head_rank":int(fiber.current_head_rank),
            "pair_fiber_dimension":int(fiber.dimension),
            "energy_row_active":bool(fiber.energy_active),
            "full_future_jacobian_rank":plan["rank_full_j"],
            "future_dark_dimension":plan["future_null_dimension"],
            "future_jacobian_singular_values":raw_singular,
            "projected_pair_fraction":float(fiber.tangent_fraction),
            "selected_margin_derivative":float(parent["selected_signed_derivative"]),
            "clean_recipient_correct":clean_rec,"clean_donor_correct":clean_don,
            "joint_clean_competence":bool(clean_rec and clean_don),
            "clean_goal_logits":clean_logits.tolist(),
            "clean_all_predicted_classes":pred[0].tolist(),
            "clean_donor_all_logits":don4[j].tolist(),
            "arms":{},
        }
        source_index+=1
        for a,arm in enumerate(ARMS):
            idx=1+j*len(ARMS)+a
            if arm=="full_donor_state":
                actual_full=(physical[idx]-physical[0]).detach()
                actual=actual_full[[p,goal]]
                full_norm=float(actual_full.norm())
            else:
                actual=(physical[idx,[p,goal]]-physical[0,[p,goal]]).detach()
                full_norm=float(actual.norm())
            raw=raw_f32(actual)
            if len(raw)!=EDIT_BYTES:
                raise ArithmeticError("original 96D physical pair edit missing")
            edit_writer.write(raw);edit_hash.update(raw)
            digit_logits=final[idx,goal].detach().double()
            # Full original ten digit first-order Jacobian response to
            # the ACTUAL f32 pair edit, not just the FP64 proposed vector.
            first_order=torch.einsum("cij,ij->c",J,actual.double()).tolist()
            record={
                "edit_index":edit_index,
                "edit_sha256":hashlib.sha256(raw).hexdigest(),
                "actual_pair_l2":float(actual.norm()),
                "actual_full_l2":full_norm,
                "ten_logit_linear_response":first_order,
                "signed_margin_linear_response":float((grad*actual.double()).sum()),
                "current_alltoken_head_l2":float((current[idx]-current[0]).norm()),
                "finite_shared_E_change":float(E[idx]-E[0]),
                "goal_logits":digit_logits.tolist(),
                "all_predicted_classes":pred[idx].tolist(),
                "donor_correct":bool(int(pred[idx,goal])==truth),
                "donor_goal_CE":float(-F.log_softmax(digit_logits,dim=-1)[truth]),
                "donor_goal_margin_clean_rival":float(digit_logits[truth]-digit_logits[rival]),
                "clean_rival":rival
            }
            if arm=="full_donor_state":
                record["all_final_logits"]=final[idx].tolist()
                c["full_donor_max_logit_error"]=float((final[idx]-don4[j]).abs().max())
            c["arms"][arm]=record
            edit_index+=1
        records.append(c)
    return records,edit_index,source_index


def summary(cases,sid):
    if len(cases)!=N_BASE*len(GOALS):
        raise ValueError("preregistered 512 case teacher roster changed")
    by={k:{i:[] for i in range(N_BASE)} for k in ("full_j","margin","random")}
    counts={a:0 for a in ARMS}
    admitted=joint=pairgood=fullgood=0
    maxparity=0.
    for c in cases:
        i=int(c["base_index"]);j=int(c["goal"])
        if i not in by["full_j"] or j not in GOALS:
            raise ValueError("case source-goal identity outside contract")
        selected=float(c["arms"]["pair_donor_k"]["donor_correct"])
        for key,prefix in (("full_j","full_j_sham_"),
                           ("margin","margin_sham_"),
                           ("random","random_k_sham_")):
            median=median8([c["arms"][prefix+str(k)]["donor_correct"]
                            for k in range(SHAMS)])
            by[key][i].append(selected-median)
        admitted+=int(c["admitted"])
        competent=int(c["joint_clean_competence"]);joint+=competent
        pairgood+=competent*int(c["arms"]["natural_pair"]["donor_correct"])
        fullgood+=competent*int(c["arms"]["full_donor_state"]["donor_correct"])
        maxparity=max(maxparity,float(c["full_donor_max_logit_error"]))
        for a in ARMS:
            counts[a]+=int(c["arms"][a]["donor_correct"])
    contrasts={}
    for kind in ("full_j","margin","random"):
        if any(len(by[kind][i])!=len(GOALS) for i in range(N_BASE)):
            raise ValueError("lost independent source goal")
        values=[sum(by[kind][i])/len(GOALS) for i in range(N_BASE)]
        contrasts[kind]={
            "paired_ITT_effect":sum(values)/N_BASE,
            "whole_base_bootstrap95":bootstrap(values,sid,kind),
            "base_level_effects":values,
            "bootstrap_draws":BOOT,
            "n_independent_bases":N_BASE,
        }
    competent_rate=joint/len(cases)
    fullrate=fullgood/max(1,joint)
    pairrate=pairgood/max(1,joint)
    pass_teacher=bool(
        admitted/len(cases)>=.95 and competent_rate>=.80 and
        fullrate>=.95 and pairrate>=.90 and maxparity<=1e-4 and
        contrasts["full_j"]["paired_ITT_effect"]>=.10 and
        contrasts["full_j"]["whole_base_bootstrap95"][0]>0 and
        contrasts["margin"]["paired_ITT_effect"]>=.10 and
        contrasts["margin"]["whole_base_bootstrap95"][0]>0 and
        contrasts["random"]["paired_ITT_effect"]>=.15 and
        contrasts["random"]["whole_base_bootstrap95"][0]>0
    )
    return {
        "n_original_source_goals":len(cases),
        "n_admitted":admitted,"n_joint_clean_competent":joint,
        "joint_competence_rate":competent_rate,
        "natural_pair_correct_among_competent":pairrate,
        "full_donor_correct_among_competent":fullrate,
        "max_original_full_donor_head_parity_error":maxparity,
        "arm_correct_counts":counts,"contrasts":contrasts,
        "teacher_strong_full_J_gate":pass_teacher
    }


def run(teacher_bundle,teacher_meta,potential_bundle,potential_meta,
        lock,out,edit_path,source_path):
    if sys.byteorder!="little":
        raise RuntimeError("registered float32 source requires little-endian")
    torch.set_num_threads(2)
    fixed=verify(lock,teacher_bundle,teacher_meta,potential_bundle,potential_meta)
    # Only after all pretrained model/observer/protocol/executable SHA checks:
    x,y,roster=new_source()
    _,trained=load_frozen(teacher_bundle,teacher_meta)
    outputs={}
    edith=hashlib.sha256();sourceh=hashlib.sha256()
    edit_index=source_index=0
    for path in (out,edit_path,source_path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(edit_path,"wb",compresslevel=5) as edit_writer, gzip.open(
            source_path,"wb",compresslevel=5) as source_writer:
        for sid in TEACHERS:
            model=make_model(trained,sid).eval().requires_grad_(False)
            potentials,_,mu,std,_=load_potentials(
                potential_bundle,potential_meta,sid)
            E=potentials["shared"].eval().requires_grad_(False)
            head=full_head_kernel(model.head.weight)
            root={"head_weight":model.head.weight.detach().double().tolist(),
                  "head_projector":head.projector.tolist(),"cases":[]}
            for i in range(N_BASE):
                rec=x[i:i+1]
                source=roster["cases"][i]["donors"]
                donors=torch.tensor([z["donor_x"] for z in source],dtype=torch.int64)
                with torch.no_grad():
                    rec_states=model.reason_states(rec)
                    don_states=model.reason_states(donors)
                    rec4=model.head(rec_states[4])
                    don4=model.head(don_states[4])
                cases,edit_index,source_index=score_base(
                    model,E,mu,std,head,rec_states[2],don_states[2],source,
                    rec4,don4,sid=sid,base=i,
                    edit_writer=edit_writer,source_writer=source_writer,
                    edit_hash=edith,source_hash=sourceh,
                    edit_index=edit_index,source_index=source_index)
                root["cases"].extend(cases)
            root["summary"]=summary(root["cases"],sid)
            outputs[str(sid)]=root
            contrasts=root["summary"]["contrasts"]
            print("LGA09_FIRST_TEACHER",sid,"full_J",contrasts["full_j"]["paired_ITT_effect"],
                  "margin",contrasts["margin"]["paired_ITT_effect"],
                  "random",contrasts["random"]["paired_ITT_effect"],
                  "strong_gate",root["summary"]["teacher_strong_full_J_gate"],flush=True)
    passed=sum(int(outputs[str(sid)]["summary"]["teacher_strong_full_J_gate"])
               for sid in TEACHERS)
    verdict=("DEVELOPMENT_STRONG_FULL_J_PASS" if passed>=2
             else "DEVELOPMENT_STRONG_FULL_J_FAIL")
    report={
        "schema":SCHEMA,"evidence_class":"EXPOSED_OLD_TEACHER_DEVELOPMENT_ONLY",
        "original_artifact_sha256":ORIGINAL,
        "source_lock_sha256":file_sha(lock),
        "protocol_sha256":fixed["protocol_sha256"],
        "implementation_sha256":fixed["implementation_sha256"],
        "source":roster,"teachers":outputs,
        "teacher_seeds":list(TEACHERS),"n_base":N_BASE,
        "goals":list(GOALS),"arms":list(ARMS),"n_arms":len(ARMS),
        "physical_pair_edits":{
            "stride_bytes":EDIT_BYTES,"n_records":edit_index,
            "raw_sha256":edith.hexdigest(),"gzip_sha256":file_sha(edit_path)
        },
        "source_jacobians":{
            "stride_bytes":SOURCE_BYTES,"n_records":source_index,
            "raw_sha256":sourceh.hexdigest(),"gzip_sha256":file_sha(source_path)
        },
        "n_old_teachers_strong_full_J":passed,
        "first_registered_development_verdict":verdict,
        "torch_version":torch.__version__,
    }
    out=Path(out)
    out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("LGA09_FIRST_NATIVE_VERDICT",verdict,"SHA256",file_sha(out),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for key in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
                "source-lock","out","edits","sources"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    run(a.teacher_bundle,a.teacher_meta,a.potential_bundle,a.potential_meta,
        a.source_lock,a.out,a.edits,a.sources)


if __name__=="__main__":
    main()
