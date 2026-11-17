"""LS-SAE06 independent stdlib-only replay of all 81 Sudoku digit logits.

Never imports Torch, NumPy, checkpoint files or original neural runner. Rebuilds
native hidden-mode power receipts, centered task effects, Sudoku validity,
per-cell truth margins and original puzzle-ID bootstrap from raw numerical data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as FAMILY
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD

SCHEMA="LS-SAE06/complete-81-sudoku-causal-task-transfer-v1"
PROTO="3b1bec465b89c552b93cd413593045bf61d4cec0"
CHECKPOINT="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
NPZ="073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25"
MANIFEST="cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a"
FIT="83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec"
ARMS=("state_power","logit_power","sae","random")
RADII=("0.05","0.20")
HORIZONS=("ACT3","ACT6")


def close(a,b,atol=2e-5,rtol=2e-5):
    if a is None or b is None:return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))


def vector_norm(x):
    return math.sqrt(sum(v*v for v in x))


def dot(a,b):
    if len(a)!=len(b):
        raise ValueError("mismatched physical vectors")
    return sum(x*y for x,y in zip(a,b))


def centered(M):
    return [[x-sum(row)/9 for x in row] for row in M]


def matrix_norm(M):
    return math.sqrt(sum(x*x for row in M for x in row))


def difference(A,B):
    if len(A)!=len(B) or any(len(a)!=len(b) for a,b in zip(A,B)):
        raise ValueError("two matrix shapes differ")
    return [[x-y for x,y in zip(a,b)] for a,b in zip(A,B)]


def check_logits(x):
    if len(x)!=81 or any(len(row)!=9 for row in x):
        raise ValueError("original native board logits must have 81 cells by 9 digits")
    if not all(math.isfinite(v) for row in x for v in row):
        raise ValueError("nonfinite native digit head output")


def predicted(logits):
    return [1+max(range(9),key=lambda digit:row[digit]) for row in logits]


def groups(grid):
    all_units=[]
    for r in range(9):
        all_units.append(grid[9*r:9*r+9])
        all_units.append([grid[9*c+r] for c in range(9)])
    for br in (0,3,6):
        for bc in (0,3,6):
            all_units.append([grid[9*(br+dr)+bc+dc] for dr in range(3) for dc in range(3)])
    return all_units


def valid_complete(grid,puzzle):
    return (len(grid)==81 and all(1<=x<=9 for x in grid)
            and all(not given or given==grid[i] for i,given in enumerate(puzzle))
            and all(set(g)==set(range(1,10)) for g in groups(grid)))


def verify_task_metrics(raw,pred,puzzle,target,goal,clean):
    blank=[i for i,x in enumerate(puzzle) if x==0]
    givens=[i for i,x in enumerate(puzzle) if x!=0]
    changed=[i for i in range(81) if pred[i]!=clean[i]]
    want={
      "predicted_digits_81":pred,
      "blank_correct":sum(pred[i]==target[i] for i in blank),
      "blank_total":len(blank),
      "board_exact":pred==target,
      "sudoku_valid_and_clue_consistent":valid_complete(pred,puzzle),
      "unit_conflicts":sum(9-len(set(unit)) for unit in groups(pred)),
      "clue_disagreements":sum(pred[i]!=puzzle[i] for i in givens),
      "all_81_prediction_changes":len(changed),
      "blank_prediction_changes":sum(i in blank for i in changed),
      "goal_prediction_changed":pred[goal]!=clean[goal],
      "other_cell_prediction_changes":sum(i!=goal for i in changed),
      "blank_correct_to_wrong":sum(clean[i]==target[i] and pred[i]!=target[i] for i in blank),
      "blank_wrong_to_correct":sum(clean[i]!=target[i] and pred[i]==target[i] for i in blank)
    }
    if raw!=want:
        diff=[key for key in want if raw.get(key)!=want[key]]
        raise ValueError("Sudoku exact correctness or validity replay disagreement: "+",".join(diff))


def verify_margins(raw,original,edit,puzzle,target):
    wanted=[i for i,v in enumerate(puzzle) if v==0]
    if raw["blank_indices"]!=wanted:
        raise ValueError("truth margin scored at given or missing goal positions")
    rival=[]
    clean=[]
    affected=[]
    for i in wanted:
        truth=target[i]-1
        other=[(-float("inf") if j==truth else original[i][j]) for j in range(9)]
        r=max(range(9),key=lambda j:other[j])
        rival.append(r)
        clean.append(original[i][truth]-original[i][r])
        affected.append(edit[i][truth]-edit[i][r])
    if raw["clean_rival_zero_based"]!=rival:
        raise ValueError("changed strongest clean nontruth margin rival")
    if any(not close(x,y) for key,vals in (
        ("clean",zip(raw["clean_margins"],clean)),
        ("edit",zip(raw["edited_margins"],affected))) for x,y in vals):
        raise ValueError("per-blank original true digit margins differ")
    mean=sum(abs(a-b) for a,b in zip(clean,affected))/len(wanted)
    if not close(mean,raw["mean_abs_margin_change"]):
        raise ValueError("mean absolute correct digit margin changed")


def check_power(progress,unit):
    if unit is None:
        if progress and all(x["status"]=="eligible" for x in progress) and len(progress)==4:
            raise ValueError("completed JtJ source-search direction missing")
        return
    if len(progress)!=4:
        raise ValueError("successful power source needs four frozen updates")
    for i,v in enumerate(progress):
        u=v["input_direction_512"]
        w=v["reverse_gradient_512"]
        if (v["step"]!=i or v["status"]!="eligible"
            or len(u)!=512 or len(w)!=512
            or any(not math.isfinite(x) for x in u+w)
            or not close(vector_norm(u),1,atol=2e-6)
            or not close(vector_norm(w),v["reverse_gradient_norm"])
            or not close(dot(u,w),v["jvp_dot_vjp"],atol=.001,rtol=.001)
            or not close(v["squared_gain"],v["output_gain"]**2,atol=.001,rtol=.001)
            or not close(v["jvp_dot_vjp"],v["squared_gain"],atol=.001,rtol=.001)):
            raise ValueError("four-step frozen JtJ JVP/VJP numerical receipt mismatch")
        next_unit=progress[i+1]["input_direction_512"] if i<3 else unit
        if len(next_unit)!=512 or any(not close(w[j]/vector_norm(w),next_unit[j],atol=2e-5)
                                     for j in range(512)):
            raise ValueError("non-normalized or skipped four-step matrix-free update")


def check_geometry(geometry,goal):
    if geometry is None:
        return None
    v=geometry["direction_512"]
    if len(v)!=512 or any(not math.isfinite(x) for x in v) or not close(vector_norm(v),1,atol=2e-6):
        raise ValueError("unmatched input source vector norm")
    n=geometry["all97_hidden_tangent_norms"]
    z=geometry["centered_digit_logit_jvp_81x9"]
    if len(n)!=97 or any(not math.isfinite(x) or x<0 for x in n):
        raise ValueError("missing 97 token tangent propagation norms")
    check_logits(z)
    if (not close(vector_norm(n),geometry["full_hidden_jvp_gain"])
        or not close(n[16+goal],geometry["goal_hidden_jvp_gain"])
        or not close(matrix_norm(z),geometry["centered_digit_logit_jvp_gain"])
        or any(abs(sum(row)/9)>2e-5 for row in z)):
        raise ValueError("full-state output or centered-logit JVP numerical bookkeeping changed")
    return v


def replay_pair(task,clean,variants,puzzle,target,goal,r,geom,horizon):
    if task is None:
        if variants is not None:
            raise ValueError("unsigned native edits lacked associated task metrics")
        return
    if variants is None or set(variants)!={"plus","minus"}:
        raise ValueError("full signed native logits missing")
    p=variants["plus"];m=variants["minus"]
    check_logits(p);check_logits(m)
    clean_pred=predicted(clean)
    pc=predicted(p);mc=predicted(m)
    for name,vec,pred in (("clean",clean,clean_pred),("plus",p,pc),("minus",m,mc)):
        verify_task_metrics(task[name],pred,puzzle,target,goal,clean_pred)
    delta_p=centered(difference(p,clean))
    delta_m=centered(difference(m,clean))
    C=(matrix_norm(delta_p)+matrix_norm(delta_m))/2
    sec=matrix_norm(centered(difference(p,m)))/2
    flip=(task["plus"]["blank_prediction_changes"]+task["minus"]["blank_prediction_changes"])/2
    correct=(task["plus"]["blank_correct"]+task["minus"]["blank_correct"])/2-task["clean"]["blank_correct"]
    if (not close(C,task["mean_centered_logit_effect_unscaled"])
        or not close(sec,task["centered_logit_sym_secant_unscaled"])
        or not close(C/r,task["centered_logit_effect_per_radius"])
        or not close(sec/r,task["centered_logit_symmetric_secant_per_radius"])
        or not close(flip,task["symmetric_blank_prediction_flip_score"])
        or not close(correct,task["symmetric_blank_correctness_gain"])):
        raise ValueError("finite signed output changed, logit gauge or radius mismatch")
    verify_margins(task["plus_margins"],clean,p,puzzle,target)
    verify_margins(task["minus_margins"],clean,m,puzzle,target)
    if horizon=="ACT3" and r==.05:
        if geom is None:
            raise ValueError("missing centered native logit JVP for finite secant")
        z=geom["centered_digit_logit_jvp_81x9"]
        v=[[x/(2*r) for x in row] for row in centered(difference(p,m))]
        err=matrix_norm(difference(v,z))
        rel=err/(1+matrix_norm(z))
        if (not close(err,task["centered_logit_secant_minus_jvp_fro"])
            or not close(rel,task["centered_logit_secant_minus_jvp_relative"])):
            raise ValueError("full81 finite centered-logit secant/JVP discrepancy changed")


def verify_case(case,source,kind,i):
    if (case["seed"]!=source["seed"] or case["puzzle_index"]!=i
        or case["source_calibration_locked"] is not True
        or case["visible_goal_position"]!=FAMILY.goal_from_visible_mask(source["x"])):
        raise ValueError("source causal intervention site used task labels")
    goal=case["visible_goal_position"]
    meta=case["direction_metadata"]
    geometry=meta["geometry"]
    if set(geometry)!=set(ARMS) or set(case["source_direction_status"])!=set(ARMS):
        raise ValueError("source geometry arm set modified")
    selected={name:check_geometry(geometry[name],goal) for name in ARMS}
    for name in ARMS:
        if case["source_direction_status"][name]!=(selected[name] is not None):
            raise ValueError("status and physical direction membership disagree")
    check_power(meta["state_power_transcript"],selected["state_power"])
    check_power(meta["logit_power_transcript"],selected["logit_power"])
    if selected["random"] is None:
        raise ValueError("source-normalized random control was lost")
    if set(case["native_81x9_logits"])!=set(HORIZONS) or set(case["task_score"])!=set(HORIZONS):
        raise ValueError("ACT3/ACT6 true full-board output record missing")
    for horizon in HORIZONS:
        variant=case["native_81x9_logits"][horizon]
        check_logits(variant["clean"])
        if set(case["task_score"][horizon])!=set(ARMS):
            raise ValueError("task metrics arms do not match source geometry")
        for arm in ARMS:
            if selected[arm] is None and variant.get(arm) is not None:
                raise ValueError("unavailable source direction has output edits")
            scores=case["task_score"][horizon][arm]
            if set(scores)!=set(RADII):
                raise ValueError("unregistered edit radius scored")
            for label in RADII:
                raw=variant.get(arm,{}).get(label)
                replay_pair(scores[label],variant["clean"],raw,
                            source["x"],source["target"],goal,float(label),geometry[arm],horizon)


def summarize(rows,family):
    event=0
    diffs=[]
    corr=[]
    for row in rows:
        a=row["task_score"]["ACT6"]["state_power"]["0.20"]
        b=row["task_score"]["ACT6"]["random"]["0.20"]
        fa=a["symmetric_blank_prediction_flip_score"] if a is not None else 0
        fb=b["symmetric_blank_prediction_flip_score"]
        diff=fa-fb
        event+=int(diff>=1)
        diffs.append(diff)
        corr.append((a["symmetric_blank_correctness_gain"] if a else 0)-b["symmetric_blank_correctness_gain"])
    rng=random.Random(202626130+len(family))
    means=[]
    for _ in range(2000):
        idx=[rng.randrange(len(rows)) for _ in rows]
        means.append(sum(diffs[i] for i in idx)/len(idx))
    means.sort()
    return {
      "n_puzzles":len(rows),"power_flip_events_at_least_one":event,
      "paired_flip_differences":diffs,
      "mean_power_minus_random_flip_score":sum(diffs)/len(diffs),
      "bootstrap95":[means[50],means[1949]],"bootstrap_draws":2000,
      "mean_power_minus_random_blank_correctness_gain":sum(corr)/len(corr),
      "registered_success":bool(len(rows)==8 and event>=6 and means[50]>0)
    }


def audit(report):
    if (report["schema"]!=SCHEMA
        or report["original_preregistration_commit"]!=PROTO
        or report["source_checkpoint_file_sha256"]!=CHECKPOINT
        or report["source_calibration_npz_file_sha256"]!=NPZ
        or report["source_calibration_manifest_sha256"]!=MANIFEST
        or report["source_calibration_fit_sha256"]!=FIT
        or report["horizons"]!=[3,6] or report["radii"]!=[.05,.2]
        or report["arms"]!=list(ARMS)
        or report["state_power_updates"]!=4 or report["logit_power_updates"]!=4
        or report["mode"] not in ("pilot","full")
        or report["full_protocol"]!=(report["mode"]=="full")):
        raise ValueError("original LS-SAE06 model, calibration, run protocol or budgets altered")
    source=FAMILY.manifest()
    if report["all_fresh_roster_sha256"]!=source:
        raise ValueError("fresh noncanonical solutions overlap or changed SHA")
    groups=("pilot_30","pilot_34") if report["mode"]=="pilot" else ("test_30","test_34")
    if set(report["families"])!=set(groups):
        raise ValueError("wrong pilot/full source rosters used")
    for kind in groups:
        original=FAMILY.roster(kind)
        group=report["families"][kind]
        if group["source_roster"]!=original or len(group["per_puzzle"])!=len(original):
            raise ValueError("original source Sudoku roster changed after test")
        for i,(case,row) in enumerate(zip(group["per_puzzle"],original)):
            verify_case(case,row,kind,i)
        regenerated=summarize(group["per_puzzle"],kind.replace("pilot_","test_"))
        for key,val in regenerated.items():
            old=group["statistics"][key]
            if isinstance(val,list):
                if len(val)!=len(old) or any(not close(x,y) for x,y in zip(val,old)):
                    raise ValueError("original raw paired/bootstrap values changed "+key)
            elif isinstance(val,float):
                if not close(val,old):raise ValueError("original registered contrast changed "+key)
            elif val!=old:
                raise ValueError("preregistered full gate or coverage count changed "+key)
    both=report["full_protocol"] and all(
        report["families"][k]["statistics"]["registered_success"] for k in ("test_30","test_34"))
    if both!=report["registered_success_both_families"]:
        raise ValueError("overall two-family verified outcome modified")
    return {
      "schema":"LS-SAE06/independent-stdlib-full81-sudoku-receipt-replay-v1",
      "original_preheldout_model_and_SAE_source_SHA_checked":True,
      "all_source_and_fresh_solution_rosters_checked":True,
      "both_four_step_matched_budget_JtJ_transcripts_replayed":True,
      "all81_logits_centered_separate_radius_signs_and_Sudoku_truth_checked":True,
      "actual_complete_grid_clue_and_unit_validity_replayed":True,
      "all2000_cluster_bootstrap_and_original_scientific_gates_replayed":True,
      "full_protocol":report["full_protocol"],
      "n_fresh_puzzles":sum(len(f["per_puzzle"]) for f in report["families"].values()),
      "scientific_gate_both_families":both,
      "limitation":"Independent original numerical transcript replay, without independent model weight execution or formal FP32 guarantees."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    b=args.input.read_bytes()
    result=audit(json.loads(b))
    result["raw_original_json_sha256"]=hashlib.sha256(b).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
