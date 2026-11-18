"""LGA-07C: first source-locked whole-state legal-donor native development.

Protocol LGA07C_GLOBAL_FIBER_PROTOCOL.md committed at 2ee61b2 before code.
Fresh 31061 is opened once, only after a separately uploaded source lock.
No training/selection on old teachers or science-confirmation claims.
"""
from __future__ import annotations
from array import array
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

from mechinterp.loopscope04.gradient_flow import fixed_data, tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import load_frozen, digest
from mechinterp.loopscope05.terminal_sae_test import make_model
from mechinterp.loopscope08.fit_potential import load_potentials, OLD_SOURCES
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import (
    ARMS, SHAMS, GOALS, BASE_COUNT, RESERVED, ORIGINAL_SEEDS,
    full_head_kernel, label_blind_global_plan, stable_seed,
)
from mechinterp.loopscope10.source_lock_global import verify_source_lock, ORIGINAL_SHAS, sha_file

SCHEMA="LGA07C/global-81token-fiber-first-development-v1"
PREVIOUS=(*OLD_SOURCES,(31055,256),(31056,64),(31057,256),
          (31058,256),(31060,64))
BOOT=2000
EDIT_WIDTH=81*48
EDIT_BYTES=EDIT_WIDTH*4


def pack_float32(t):
    if sys.byteorder!="little":
        raise RuntimeError("registered raw float32 physical vectors require little endian")
    t=t.detach().cpu().contiguous().float()
    return array("f",t.reshape(-1).tolist()).tobytes()


def bootstrap(values, seed):
    if len(values)!=BASE_COUNT or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("invalid 64 independent base-ID measurements")
    rng=random.Random(int(seed))
    draws=sorted(sum(values[rng.randrange(BASE_COUNT)] for _ in range(BASE_COUNT))/BASE_COUNT
                 for _ in range(BOOT))
    return [draws[int(.025*BOOT)],draws[int(.975*BOOT)-1]]


def make_roster():
    """First new source generation, only after verify_source_lock."""
    x,y=fixed_data(BASE_COUNT,RESERVED)
    seen=set()
    for seed,n in PREVIOUS:
        old,_=fixed_data(n,seed)
        seen.update(tuple(z) for z in old.tolist())
    originals=[tuple(z) for z in x.tolist()]
    if len(set(originals))!=BASE_COUNT or any(z in seen for z in originals):
        raise ValueError("new 31061 source contains a previously generated IID source")
    roster=[]
    all_donors=set()
    for i in range(BASE_COUNT):
        donors=[]
        for goal in GOALS:
            d,info=legal_predecessor_donor(x[i],goal)
            donor_key=tuple(d.tolist())
            if donor_key in seen or donor_key in all_donors:
                raise ValueError("legal donor source collision")
            all_donors.add(donor_key)
            donors.append({"goal":goal,"x":d.tolist(),**info})
        roster.append({"base_index":i,"x":x[i].tolist(),
                       "y":y[i].tolist(),"donors":donors})
    return x,y,{"seed":RESERVED,"n_base":BASE_COUNT,
                "x_sha256":tensor_int64_sha(x),
                "y_sha256":tensor_int64_sha(y),
                "cases":roster}


def vec_bytes_and_meta(t, writer, running_digest, ordinal):
    raw=pack_float32(t)
    if len(raw)!=EDIT_BYTES:
        raise ArithmeticError("wrong native whole-state physical edit byte count")
    writer.write(raw)
    running_digest.update(raw)
    return {"vector_index":ordinal,"vector_sha256":hashlib.sha256(raw).hexdigest()}


def prepare_one_case(model, energy, mu, std, head_fiber,
                     recipient_h, donor_h, *, goal, teacher_seed, base_index):
    """Physical source-only donor/shams. No correctness labels accepted."""
    return label_blind_global_plan(
        model,energy,recipient_h,donor_h,mu,std,
        goal=goal,head_fiber=head_fiber,
        teacher_seed=teacher_seed,base_index=base_index)


def score_cases_for_base(model,energy,mu,std,head_fiber,
                         rec_state, donor_states, donor_roster,
                         clean_rec_logits, clean_donor_logits,
                         vec_writer, source_writer, vec_digest, source_digest,
                         vector_ordinal, source_ordinal, *, sid,base_index):
    """Score complete natural donor and every preplanned physically matched edit."""
    plans=[]
    for j,goal in enumerate(GOALS):
        plans.append(prepare_one_case(
            model,energy,mu,std,head_fiber,
            rec_state,donor_states[j:j+1],
            goal=goal,teacher_seed=sid,base_index=base_index))
    clean_batch=rec_state.detach()
    n_arms=len(ARMS)
    # All source plans frozen BEFORE any task target/clean competence scoring.
    physical=clean_batch.expand(1+len(GOALS)*n_arms,-1,-1).clone()
    for j,goal in enumerate(GOALS):
        for a,name in enumerate(ARMS):
            index=1+j*n_arms+a
            if name=="full_donor_state":
                physical[index]=donor_states[j]  # exact reachable state, no subtract/add drift
            else:
                physical[index] += plans[j]["directions"][name].float()
    with torch.no_grad():
        current_head=model.head(physical)
        source_energy=energy(physical,0,mu,std).double().sum(-1)
        final_state=physical
        for _ in range(2):
            final_state=model.block(final_state)
        final_logits=model.head(final_state)
        final_predictions=final_logits.argmax(-1)
    if not bool(torch.isfinite(final_logits).all()):
        raise ArithmeticError("nonfinite full native donor readout")
    clean_native=final_logits[0]
    if float((clean_native-clean_rec_logits).abs().max())>1e-4:
        raise ArithmeticError("zero-edit native 2-block closure failed")
    cases=[]
    for j,goal in enumerate(GOALS):
        plan=plans[j]
        fiber=plan["fiber"]
        truth=int(donor_roster[j]["donor_goal_label"])
        original_target=int(donor_roster[j]["original_goal_label"])
        clean_rival=max((c for c in range(10) if c!=truth),
                        key=lambda c:float(clean_native[goal,c]))
        # Archive exact ORIGINAL reached donor difference and energy gradient.
        delta_raw=pack_float32(fiber.delta)
        gradient_raw=pack_float32(fiber.gradient)
        if len(delta_raw)!=EDIT_BYTES or len(gradient_raw)!=EDIT_BYTES:
            raise ArithmeticError("wrong source-only vector size")
        source_writer.write(delta_raw+gradient_raw)
        source_digest.update(delta_raw+gradient_raw)
        idx=1+j*n_arms
        donor_parity=float((final_logits[idx+ARMS.index("full_donor_state")]-
                            clean_donor_logits[j]).abs().max())
        rec_correct=int(clean_rec_logits[0,goal].argmax())==original_target
        don_correct=int(clean_donor_logits[j,goal].argmax())==truth
        record={
            "base_index":base_index,"slot":j,"goal":goal,
            "original_target":original_target,"donor_target":truth,
            "admitted":bool(plan["eligible"]),
            "jointly_clean_competent":bool(rec_correct and don_correct),
            "clean_recipient_correct":bool(rec_correct),
            "clean_donor_correct":bool(don_correct),
            "radius_natural_full_state":float(plan["radius"]),
            "head_rank":int(plan["head_rank"]),
            "tangent_dimension":int(plan["tangent_dimension"]),
            "energy_row_active":bool(plan["energy_row_active"]),
            "head_only_fraction":float(plan["head_fraction"]),
            "energy_tangent_fraction":float(plan["tangent_fraction"]),
            "head_residual":float(plan["head_residual"]),
            "tangent_derivative":float(plan["tangent_gradient_dot"]),
            "source_vector_index":source_ordinal,
            "source_delta_sha256":hashlib.sha256(delta_raw).hexdigest(),
            "source_energy_gradient_sha256":hashlib.sha256(gradient_raw).hexdigest(),
            "full_donor_max_logit_error":donor_parity,
            "clean_goal_logits":clean_native[goal].tolist(),
            "clean_recipient_predicted_classes":final_predictions[0].tolist(),
            "clean_donor_goal_logits":clean_donor_logits[j,goal].tolist(),
            "clean_donor_all_logits":clean_donor_logits[j].tolist(),
            "arms":{},
        }
        source_ordinal+=1
        for a,name in enumerate(ARMS):
            batch_index=idx+a
            # The actual applied f32 physical perturbation can differ slightly
            # from the planned FP64 vector due to round-to-FP32 addition.
            applied=(physical[batch_index]-physical[0]).detach().cpu()
            binary=vec_bytes_and_meta(applied,vec_writer,vec_digest,vector_ordinal)
            vector_ordinal+=1
            goal_logits=final_logits[batch_index,goal].detach().double()
            val=goal_logits.tolist()
            ce=float(-F.log_softmax(goal_logits,dim=-1)[truth])
            radius=float(torch.linalg.vector_norm(applied))
            datum={
                **binary,"applied_l2":radius,
                "goal_logits":val,
                "predicted_classes":final_predictions[batch_index].tolist(),
                "donor_correct":bool(int(final_predictions[batch_index,goal])==truth),
                "donor_goal_CE":ce,
                "donor_margin_clean_rival":float(goal_logits[truth]-goal_logits[clean_rival]),
                "clean_rival":int(clean_rival),
                "current_head_alltoken_l2":float(torch.linalg.vector_norm(
                    current_head[batch_index]-current_head[0])),
                "finite_initial_E_change":float(source_energy[batch_index]-source_energy[0]),
                "firstorder_E_dot":float(torch.sum(
                    fiber.gradient * applied.double())),
            }
            if name=="full_donor_state":
                datum["all_final_logits"]=final_logits[batch_index].tolist()
            record["arms"][name]=datum
        cases.append(record)
    return cases,vector_ordinal,source_ordinal


def median8(v):
    x=sorted(float(z) for z in v)
    if len(x)!=8:
        raise ValueError("exactly eight preregistered shams required")
    return (x[3]+x[4])/2


def summarize(cases,sid):
    if len(cases)!=BASE_COUNT*len(GOALS):
        raise ValueError("original 512 source-goal cases missing")
    groups={i:[] for i in range(BASE_COUNT)}
    selected=shams=admitted=joint=donor_full_correct=0
    max_parity=0.
    for c in cases:
        i=int(c["base_index"])
        k=int(c["goal"])
        if k not in GOALS or i not in groups:
            raise ValueError("registered goal/source identity changed")
        a=int(c["arms"]["global_energy_tangent"]["donor_correct"])
        b=median8([int(c["arms"][f"same_k_sham_{n}"]["donor_correct"])
                   for n in range(SHAMS)])
        groups[i].append(a-b)
        selected+=a;shams+=b;admitted+=int(c["admitted"])
        competent=int(c["jointly_clean_competent"])
        joint+=competent
        donor_full_correct+=competent*int(c["arms"]["full_donor_state"]["donor_correct"])
        max_parity=max(max_parity,float(c["full_donor_max_logit_error"]))
    source_effects=[]
    for i in range(BASE_COUNT):
        if len(groups[i])!=len(GOALS):
            raise ValueError("new source lost one or more registered goals")
        source_effects.append(sum(groups[i])/len(GOALS))
    effect=sum(source_effects)/len(source_effects)
    ci=bootstrap(source_effects,stable_seed("bootstrap",sid))
    eligible_comp=joint/len(cases)
    natural_donor_success=donor_full_correct/joint if joint else 0.
    passed=bool(
        eligible_comp>=.8 and max_parity<=1e-4 and
        natural_donor_success>=.95 and effect>=.15 and ci[0]>0)
    return {
        "n_base_clusters":BASE_COUNT,"n_goal_pairs":len(cases),
        "n_admitted":admitted,"n_joint_competent":joint,
        "joint_competence_fraction":eligible_comp,
        "full_donor_correct_among_competent":natural_donor_success,
        "max_full_donor_alllogit_error":max_parity,
        "selected_ITT_correct_rate":selected/len(cases),
        "median_sham_ITT_correct_rate":shams/len(cases),
        "primary_paired_ITT_effect":effect,"whole_base_bootstrap95":ci,
        "bootstrap_repetitions":BOOT,
        "development_teacher_gate":passed,
        "base_level_effects":source_effects,
    }


def run(teacher_bundle,teacher_meta,potential_bundle,potential_meta,
        source_lock,out,vec_out,source_out):
    if sys.byteorder!="little":
        raise RuntimeError("source machine must be little-endian")
    torch.set_num_threads(2)
    lock=verify_source_lock(source_lock,teacher_bundle,teacher_meta,
                            potential_bundle,potential_meta)
    # Reserved 31061 opened ONLY after byte/implementation/protocol source lock.
    bases,targets,roster=make_roster()
    _,frozen=load_frozen(teacher_bundle,teacher_meta)
    results={}
    output_vectors=Path(vec_out);output_vectors.parent.mkdir(parents=True,exist_ok=True)
    output_sources=Path(source_out);output_sources.parent.mkdir(parents=True,exist_ok=True)
    vec_digest=hashlib.sha256();source_digest=hashlib.sha256()
    vector_ordinal=source_ordinal=0
    with gzip.open(output_vectors,"wb",compresslevel=3) as vec_f, gzip.open(
            output_sources,"wb",compresslevel=6) as source_f:
        for sid in ORIGINAL_SEEDS:
            teacher=make_model(frozen,sid).eval().requires_grad_(False)
            models,_,mu,std,_=load_potentials(potential_bundle,potential_meta,sid)
            potential=models["shared"].eval().requires_grad_(False)
            head=full_head_kernel(teacher.head.weight)
            root={"head_weight":teacher.head.weight.detach().double().tolist(),
                  "head_projector":head.projector.tolist(),"cases":[]}
            for i in range(BASE_COUNT):
                rec=bases[i:i+1]
                donor_spec=roster["cases"][i]["donors"]
                dons=torch.tensor([x["x"] for x in donor_spec],dtype=torch.int64)
                with torch.no_grad():
                    recip_states=teacher.reason_states(rec)
                    donor_states=teacher.reason_states(dons)
                    clean_rec_logits=teacher.head(recip_states[4])
                    clean_donor_logits=teacher.head(donor_states[4])
                cases,vector_ordinal,source_ordinal=score_cases_for_base(
                    teacher,potential,mu,std,head,
                    recip_states[2],donor_states[2],donor_spec,
                    clean_rec_logits,clean_donor_logits,
                    vec_f,source_f,vec_digest,source_digest,
                    vector_ordinal,source_ordinal,sid=sid,base_index=i)
                root["cases"].extend(cases)
            root["summary"]=summarize(root["cases"],sid)
            results[str(sid)]=root
            print("LGA07C_ORIGINAL_TEACHER",sid,
                  root["summary"]["primary_paired_ITT_effect"],
                  root["summary"]["full_donor_correct_among_competent"],
                  root["summary"]["development_teacher_gate"],flush=True)
    npass=sum(int(results[str(sid)]["summary"]["development_teacher_gate"])
              for sid in ORIGINAL_SEEDS)
    verdict=("DEVELOPMENT_ENGINEERING_POSITIVE" if npass>=2
             else "DEVELOPMENT_NULL_OR_INELIGIBLE")
    report={
        "schema":SCHEMA,"evidence_class":"EXPOSED_FROZEN_TEACHERS_DEVELOPMENT_ONLY",
        "frozen_original_shas":ORIGINAL_SHAS,
        "pretest_source_lock_sha256":sha_file(source_lock),
        "protocol_shas":lock["protocol_shas"],
        "code_shas":lock["code_shas"],
        "source":roster,"teachers":results,
        "teacher_seeds":list(ORIGINAL_SEEDS),
        "n_base":BASE_COUNT,"n_goal_per_base":len(GOALS),
        "n_arms_per_case":len(ARMS),"arms":list(ARMS),
        "binary_vectors":{"dtype":"little-endian float32","state_shape":[81,48],
            "n_records":vector_ordinal,"stride_bytes":EDIT_BYTES,
            "raw_sha256":vec_digest.hexdigest(),
            "gzip_sha256":sha_file(output_vectors)},
        "binary_sources":{"dtype":"little-endian float32","state_shape":[2,81,48],
            "n_records":source_ordinal,"stride_bytes":2*EDIT_BYTES,
            "raw_sha256":source_digest.hexdigest(),
            "gzip_sha256":sha_file(output_sources)},
        "n_teachers_meeting_development_gate":npass,
        "registered_first_development_verdict":verdict,
        "torch_version":torch.__version__,
    }
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("LGA07C_FIRST_SOURCE_LOCKED_VERDICT",verdict,
          "source",roster["x_sha256"],"full",sha_file(out),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for a in ("teacher-bundle","teacher-meta","potential-bundle","potential-meta",
              "source-lock","out","vec-out","source-out"):
        p.add_argument("--"+a,required=True,type=Path)
    x=p.parse_args()
    run(x.teacher_bundle,x.teacher_meta,x.potential_bundle,x.potential_meta,
        x.source_lock,x.out,x.vec_out,x.source_out)


if __name__=="__main__":
    main()
