"""LOOPSCOPE-10B: source-locked legal-donor native recurrent development evaluation.

First frozen execution contract: LGA07B_FIRST_NATIVE_RELEASE.md. New 31060
source is opened ONLY after verifying the uploaded Stage-A pretest lock.
All trained teacher/potential weights stay unchanged. No science confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

import torch
import torch.nn.functional as F

from mechinterp.loopscope04.gradient_flow import fixed_data, tensor_int64_sha
from mechinterp.loopscope05.fit_and_freeze import load_frozen, digest
from mechinterp.loopscope05.terminal_sae_test import make_model
from mechinterp.loopscope08.fit_potential import load_potentials, OLD_SOURCES
from mechinterp.loopscope10.donor_source import (
    GOALS, RESERVED_SEED, SOURCE_COUNT, legal_predecessor_donor,
    stable_seed, label_blind_donor_plan
)
from mechinterp.loopscope10.source_lock_donor import verify_lock, CHECKSUMS, file_sha

SCHEMA = "LGA07B/first-native-reachable-donor-development-v1"
ARMS = ("projected_donor", "inverse_projected_donor", "unprojected_donor",
        "full_donor_token", *(f"same_k_sham_{i}" for i in range(8)))
DEPTHS = (2, 6)
ENDPOINT = {2:4, 6:12}
BOOT = 2000
PREVIOUS = (*OLD_SOURCES, (31055,256), (31056,64), (31057,256), (31058,256))


def edit_sha(z):
    vals = z.detach().to(dtype=torch.float32,device="cpu").reshape(-1).tolist()
    return digest(struct.pack("<" + "f"*len(vals), *vals))


def bootstrap(values, seed, draws=BOOT):
    n = len(values)
    if n != SOURCE_COUNT or any(not math.isfinite(float(z)) for z in values):
        raise ValueError("invalid independent base-cluster bootstrap source")
    rng = random.Random(seed)
    sorted_means = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                          for _ in range(draws))
    return [sorted_means[int(.025*draws)],
            sorted_means[int(.975*draws)-1]]


def source_roster():
    """FIRST source generation, called after source_lock verify only."""
    x,y = fixed_data(SOURCE_COUNT, RESERVED_SEED)
    original = set()
    for previous_seed, count in PREVIOUS:
        prev,_ = fixed_data(count,previous_seed)
        original.update(tuple(map(int,row)) for row in prev.tolist())
    rows = [tuple(map(int,row)) for row in x.tolist()]
    if len(set(rows)) != SOURCE_COUNT or any(row in original for row in rows):
        raise ValueError("reserved new source reused previous IID source")
    records = []
    for i in range(SOURCE_COUNT):
        donor_records = []
        for goal in GOALS:
            d, info = legal_predecessor_donor(x[i],goal)
            if tuple(map(int,d.tolist())) in original:
                raise ValueError("new legal donor collides with historical source")
            donor_records.append({"goal":goal,"donor_x":d.tolist(),**info})
        records.append({"base_index":i,"x":x[i].tolist(),"y":y[i].tolist(),
                        "donors":donor_records})
    return x,y,{
        "seed":RESERVED_SEED,"n_independent_base_sequences":SOURCE_COUNT,
        "x_sha256":tensor_int64_sha(x),"y_sha256":tensor_int64_sha(y),
        "paired_base_ids":list(range(SOURCE_COUNT)),
        "cases":records,
    }


@torch.no_grad()
def native_states(model,x):
    h=model.initial_state(x)
    states=[h]
    for _ in range(12):
        h=model.block(h)
        states.append(h)
    if len(states)!=13 or any(not bool(torch.isfinite(z).all()) for z in states):
        raise ArithmeticError("true native 12-loop trajectory nonfinite")
    return states


def score_batch(model,potential,mu,std,recipient,plans,donor_targets,
                *,depth,horizon):
    """Batch actual native continuations after all source-only directions freeze."""
    if horizon - depth not in (2,6):
        raise ValueError("invalid frozen native horizon")
    full_names = ("clean",)
    chosen=[]
    rows=[]
    for idx, goal in enumerate(GOALS):
        plan=plans[idx]
        donor_truth=int(donor_targets[idx])
        row={"goal":int(goal),"admitted":bool(plan["eligible"]),
             "reason":None if plan["eligible"] else plan.get("reason"),
             "donor_goal_target":donor_truth,
             "fiber_dimension":int(plan["fiber"].null_dimension),
             "fiber_joint_rank":int(plan["fiber"].rank_joint),
             "fiber_head_rank":int(plan["fiber"].rank_current),
             "fiber_constraint_residual":float(plan["fiber"].null_residual),
             "fiber_projector_error":float(plan["fiber"].projector_error),
             "radius":float(plan["radius"]),"arms":{}}
        rows.append(row)
        for arm in ARMS:
            if plan["eligible"]:
                edit=plan["directions"][arm].detach().float().cpu()
            elif arm=="full_donor_token":
                # Unscreened naturally reached donor token, diagnostic control.
                edit=plan["_source_donor_token"].detach().float().cpu()
            else:
                edit=torch.zeros(recipient.shape[-1],dtype=torch.float32)
            chosen.append((idx,goal,arm,edit))
    original=recipient[0].detach().clone()
    batch=original[None].expand(len(chosen)+1,-1,-1).clone()
    for j,(_,goal,_,edit) in enumerate(chosen,1):
        batch[j,goal] += edit.to(device=batch.device,dtype=batch.dtype)
    # Current E and head changes before true native continuation.
    with torch.no_grad():
        energy_before=potential(batch,0,mu,std).double().sum(-1)
        head_before=model.head(batch)
        h=batch
        for _ in range(horizon-depth):
            h=model.block(h)
        final_logits=model.head(h)
        final_energy=potential(h,0,mu,std).double().sum(-1)
        predicted=final_logits.argmax(-1)
    if not bool(torch.isfinite(final_logits).all()):
        raise ArithmeticError("nonfinite native final head")
    clean_logits=final_logits[0]
    clean_pred=predicted[0].tolist()
    for idx,goal,arm,edit in chosen:
        pass
    for j,(idx,goal,arm,edit) in enumerate(chosen,1):
        row=rows[idx]
        truth=int(row["donor_goal_target"])
        gl=final_logits[j,goal].detach().double().cpu()
        cl=clean_logits[goal].detach().double().cpu()
        clean_rival=int(torch.cat((cl[:truth],cl[truth+1:])).argmax())
        if clean_rival>=truth:clean_rival+=1
        applied=(batch[j,goal]-batch[0,goal]).detach().double().cpu()
        radius=float(row["radius"])
        data={
            "goal_logits":[float(z) for z in gl],
            "predicted_classes":predicted[j].tolist(),
            "donor_target_correct":int(predicted[j,goal])==truth,
            "donor_goal_CE":float(-F.log_softmax(gl,dim=-1)[truth]),
            "donor_goal_margin":float(gl[truth]-gl[clean_rival]),
            "clean_competitor_class":clean_rival,
            "edit_vector_float32":edit.tolist(),
            "edit_sha256":edit_sha(edit),
            "edit_l2":float(edit.norm()),
            "realized_edit_l2":float(applied.norm()),
            "finite_current_head_l2":float(
                (head_before[j,goal]-head_before[0,goal]).norm()),
            "finite_shared_E_change":float(energy_before[j]-energy_before[0]),
            "final_shared_E_change":float(final_energy[j]-final_energy[0]),
        }
        row["arms"][arm]=data
    for idx,goal in enumerate(GOALS):
        row=rows[idx]
        cl=clean_logits[goal].detach().double().cpu()
        truth=int(row["donor_goal_target"])
        row["clean"]={
            "goal_logits":cl.tolist(),
            "predicted_classes":clean_pred,
            "donor_target_correct":int(predicted[0,goal])==truth,
            "donor_goal_CE":float(-F.log_softmax(cl,dim=-1)[truth]),
        }
        if plan_is_good := row["admitted"]:
            # Every selected/sham edit is same physical L2 by construction.
            for arm in ("projected_donor",*(f"same_k_sham_{i}" for i in range(8))):
                if abs(row["arms"][arm]["edit_l2"]-float(row["radius"]))>1e-5:
                    raise ArithmeticError("dose mismatch in admitted same-K edits")
    return rows


def median_binary(values):
    ordered=sorted(float(v) for v in values)
    return (ordered[3]+ordered[4])/2.


def summarize(cases,sid,depth):
    if len(cases)!=SOURCE_COUNT*len(GOALS):
        raise ValueError("number of original ITT source-goal pairs changed")
    by_base={i:[] for i in range(SOURCE_COUNT)}
    selected_rate=sham_rate=0.
    admissible=joint_competent=full_joint=0
    for row in cases:
        sham=[float(row["arms"][f"same_k_sham_{j}"]["donor_target_correct"])
              for j in range(8)]
        selected=float(row["arms"]["projected_donor"]["donor_target_correct"])
        control=median_binary(sham)
        by_base[row["base_index"]].append(selected-control)
        selected_rate+=selected
        sham_rate+=control
        admissible+=int(row["admitted"])
        eligible=int(row["clean_recipient_correct"] and row["clean_donor_correct"])
        joint_competent+=eligible
        full_joint+=eligible*int(row["arms"]["full_donor_token"]["donor_target_correct"])
    for i in range(SOURCE_COUNT):
        if len(by_base[i])!=len(GOALS):
            raise ValueError("goal roster changed within a source cluster")
    source_means=[sum(by_base[i])/len(GOALS) for i in range(SOURCE_COUNT)]
    ci=bootstrap(source_means,stable_seed("bootstrap",sid,depth))
    effect=sum(source_means)/SOURCE_COUNT
    donor_success=full_joint/joint_competent if joint_competent else 0.
    teacher_gate=bool(depth==2 and effect>=.15 and ci[0]>0 and
                      joint_competent/(SOURCE_COUNT*len(GOALS))>=.8 and
                      donor_success>=.95)
    return {
        "n_independent_base_sources":SOURCE_COUNT,
        "n_original_goals":len(cases),
        "n_admitted":admissible,
        "n_jointly_competent":joint_competent,
        "competence_fraction":joint_competent/len(cases),
        "selected_ITT_rate":selected_rate/len(cases),
        "sham_median_ITT_rate":sham_rate/len(cases),
        "ITT_paired_effect":effect,
        "whole_base_bootstrap95":ci,
        "full_donor_token_correct_among_jointly_competent":donor_success,
        "teacher_engineering_gate":teacher_gate,
        "base_level_paired_effects":source_means,
        "bootstrap_repetitions":BOOT,
    }


def run(teacher_bundle,teacher_meta,potential_bundle,potential_meta,source_lock,out):
    torch.set_num_threads(2)
    locked=verify_lock(source_lock,teacher_bundle,teacher_meta,
                       potential_bundle,potential_meta)
    # This is the sole code location that opens previously untouched 31060.
    bases,targets,source_manifest=source_roster()
    _, bundle=load_frozen(teacher_bundle,teacher_meta)
    results={}
    for sid in (103,211,307):
        model=make_model(bundle,sid).eval().requires_grad_(False)
        potentials,_,mu,std,_=load_potentials(potential_bundle,potential_meta,sid)
        energy=potentials["shared"].eval().requires_grad_(False)
        teacher_results={str(k):{"cases":[]} for k in DEPTHS}
        for i in range(SOURCE_COUNT):
            recipient=bases[i:i+1]
            goal_records=source_manifest["cases"][i]["donors"]
            donors=torch.tensor([z["donor_x"] for z in goal_records],dtype=torch.int64)
            with torch.no_grad():
                recipient_states=native_states(model,recipient)
                donor_states=native_states(model,donors)
                recipient4=model.head(recipient_states[4])[0]
                donor4=model.head(donor_states[4])
            for depth in DEPTHS:
                current=recipient_states[depth]
                plans=[]
                for j,goal in enumerate(GOALS):
                    plan=label_blind_donor_plan(
                        model,energy,current,donor_states[depth][j:j+1],
                        mu,std,goal=goal,
                        seed=stable_seed("sham",sid,i,goal,depth))
                    if not plan["eligible"]:
                        plan["_source_donor_token"]=(donor_states[depth][j,goal]-
                                                      current[0,goal]).detach()
                    plans.append(plan)
                donor_truths=[z["donor_goal_label"] for z in goal_records]
                rows=score_batch(model,energy,mu,std,current,plans,donor_truths,
                                 depth=depth,horizon=ENDPOINT[depth])
                for j,row in enumerate(rows):
                    goal=GOALS[j]
                    clean_rec=int(recipient4[goal].argmax())==int(targets[i,goal])
                    clean_don=int(donor4[j,goal].argmax())==int(donor_truths[j])
                    row.update({
                        "base_index":i,"source_goal_slot":j,"goal":goal,
                        "original_goal_target":int(targets[i,goal]),
                        "donor_goal_target":int(donor_truths[j]),
                        "clean_recipient_correct":bool(clean_rec),
                        "clean_donor_correct":bool(clean_don),
                        "clean_recipient4_goal_logits":recipient4[goal].tolist(),
                        "clean_donor4_goal_logits":donor4[j,goal].tolist(),
                        "source_donor_goal_delta_l2":float(
                            (donor_states[depth][j,goal]-current[0,goal]).norm()),
                    })
                    teacher_results[str(depth)]["cases"].append(row)
        for depth in DEPTHS:
            case_list=teacher_results[str(depth)]["cases"]
            teacher_results[str(depth)]["summary"]=summarize(case_list,sid,depth)
        results[str(sid)]=teacher_results
        print("LGA07B_DEVELOPMENT_TEACHER",sid,
              teacher_results["2"]["summary"]["ITT_paired_effect"],
              teacher_results["2"]["summary"]["teacher_engineering_gate"],flush=True)
    successes=sum(results[str(s)]["2"]["summary"]["teacher_engineering_gate"]
                  for s in (103,211,307))
    report={
        "schema":SCHEMA,
        "evidence_class":"EXPOSED_TEACHERS_DEVELOPMENT_ONLY",
        "source_lock_sha256":file_sha(source_lock),
        "original_artifact_shas":CHECKSUMS,
        "protocol_sha256":locked["protocol_sha256"],
        "implementation_sha256":locked["implementation_sha256"],
        "test_source":source_manifest,
        "teacher_seeds":[103,211,307],"native_depths":[2,6],
        "n_independent_base_sources":SOURCE_COUNT,
        "n_cases_per_teacher":SOURCE_COUNT*len(GOALS),
        "arms":list(ARMS),
        "teachers":results,
        "n_teachers_passing_engineering_gate":successes,
        "registered_development_verdict":
            "DEVELOPMENT_ENGINEERING_POSITIVE" if successes>=2 else
            "DEVELOPMENT_NULL_OR_INELIGIBLE",
        "torch_version":torch.__version__,
    }
    path=Path(out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,separators=(",",":")))
    print("LGA07B_FIRST_NATIVE_DEVELOPMENT",
          report["registered_development_verdict"],flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ("teacher-bundle","teacher-meta","potential-bundle",
                 "potential-meta","source-lock","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    run(a.teacher_bundle,a.teacher_meta,a.potential_bundle,a.potential_meta,
        a.source_lock,a.out)


if __name__=="__main__":
    main()
