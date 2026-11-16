"""Independent stdlib receipt audit of LS-SAE05 matrix-free native power-JVP test."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.native_singular_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_trm_v1 import families as SOURCE

PROTO="36009cb071ab9d283cfae86d40f7e46cfbc608a6"
MODEL_SHA="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
FIT_SHA="83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec"
MANIFEST_SHA="cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a"
NPZ_SHA="073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25"
SCHEMA="LS-SAE05/native-512D-matrix-free-singular-power-v1"


def close(a,b,atol=2e-5,rtol=5e-4):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b))


def dot(a,b):
    if len(a)!=len(b):raise ValueError("vector shape")
    return sum(v*w for v,w in zip(a,b))


def norm(a):
    return math.sqrt(dot(a,a))


def check_vector(info,goal_pos,readout,gradient,y,rival):
    if info is None:return
    u=info["direction_512"]
    goal=info["goal_tangent_512"]
    tokens=info["all97_token_norms"]
    if (len(u)!=512 or len(goal)!=512 or len(tokens)!=97 or
        any(not math.isfinite(x) for x in u+goal+tokens) or
        any(x<0 for x in tokens)):
        raise ValueError("invalid original native tangent shapes/values")
    if not close(norm(u),1,atol=2e-6):
        raise ValueError("physical selected source direction not unit norm")
    full=norm(tokens);local=norm(goal)
    others=math.sqrt(max(0,full*full-local*local))
    if (not close(full,info["full_jvp_gain"]) or
        not close(local,info["goal_jvp_gain"]) or
        not close(local,tokens[16+goal_pos]) or
        not close(others,info["other_tokens_gain"])):
        raise ValueError("97 token JVP norm partition changed")
    frac=local*local/(full*full) if full>1e-12 else None
    if not close(frac,info["goal_energy_fraction"]):
        raise ValueError("native goal-token tangent energy fraction modified")
    slope=dot(goal,readout)
    reverse=dot(gradient,u)
    if (not close(slope,info["task_margin_slope"]) or
        not close(reverse,info["reverse_mode_slope"]) or
        abs(slope-reverse)>max(.02,.05*abs(slope))+2e-5):
        raise ValueError("full-state forward-JVP and reverse-mode margin mismatch")
    if "finite_margin" in info:
        finite=info["finite_margin"]
        p=finite["plus_margin"];m=finite["minus_margin"]
        sec=(p-m)/.1
        if (not close(sec,finite["finite_secant"]) or
            not close(sec-slope,finite["secant_minus_jvp"])):
            raise ValueError("symmetric finite physical margin response changed")


def check_case(case,puzzle,family,idx):
    if case["seed"]!=puzzle["seed"] or case["goal_position"]!=FAM.median_blank(puzzle["x"]):
        raise ValueError("selected target depends on non-visible Sudoku labels")
    if case["true_digit_zero_based"]!=puzzle["target"][case["goal_position"]]-1:
        raise ValueError("heldout answer mismatched")
    logits=case["clean_logits9"]
    y=case["true_digit_zero_based"]
    if len(logits)!=9 or not all(math.isfinite(x) for x in logits):
        raise ValueError("clean goal logits corrupted")
    alt=logits.copy();alt[y]=-float("inf")
    rival=max(range(9),key=lambda i:alt[i])
    if rival!=case["rival_digit_zero_based"]:
        raise ValueError("non-clean heldout task rival")
    readout=case["clean_readout_row_512"]
    grad=case["clean_gradient_512"]
    if len(readout)!=512 or len(grad)!=512 or any(not math.isfinite(x) for x in readout+grad):
        raise ValueError("full source digit-head/gradient absent")
    arms=case["directions"]
    if set(arms)!={"power","sae","task_head"}|{f"random_{j}" for j in range(8)}:
        raise ValueError("fixed source and random JVP arms changed")
    for name,data in arms.items():
        check_vector(data,case["goal_position"],readout,grad,y,rival)
    for name in ("power","sae","task_head","random_0"):
        if arms[name] is not None and "finite_margin" not in arms[name]:
            raise ValueError("required ±0.05 physical response missing")
    if len(case["power_iterations"])>4:
        raise ValueError("power updates exceed registered budget")
    if arms["power"] is None and len(case["power_iterations"])==4 and all(v["status"]=="eligible" for v in case["power_iterations"]):
        raise ValueError("successful four-step power trial missing scored direction")
    if arms["power"] is not None:
        if len(case["power_iterations"])!=4:
            raise ValueError("unregistered short power search scored")
        progress=case["power_iterations"]
        for step,j in enumerate(progress):
            u=j["input_direction_512"]
            w=j["reverse_gradient_512"]
            if (j["step"]!=step or j["status"]!="eligible"
                or len(u)!=512 or len(w)!=512
                or not close(norm(u),1,atol=2e-6)
                or not close(norm(w),j["reverse_gradient_norm"])
                or not close(dot(u,w),j["jvp_dot_vjp"],atol=1e-4)
                or not close(j["jvp_dot_vjp"],j["squared_gain"],atol=1e-3,rtol=1e-3)
                or not close(j["output_gain"]**2,j["squared_gain"],atol=1e-4)):
                raise ValueError("power iteration JVP/VJP numerical certificate changed")
            next_u=[v/norm(w) for v in w]
            given=progress[step+1]["input_direction_512"] if step<3 else arms["power"]["direction_512"]
            if any(not close(x,y,atol=1e-5) for x,y in zip(next_u,given)):
                raise ValueError("power iteration not equal to normalized JtJu recurrence")
    rand=[arms[f"random_{j}"]["full_jvp_gain"] for j in range(8)]
    best=max(rand)
    selected=arms["power"]["full_jvp_gain"] if arms["power"] else None
    d=selected-best if selected is not None else None
    event=bool(selected is not None and selected>=.25 and selected>=1.2*best)
    if (not close(best,case["best_random_full_jvp_gain"]) or
        not close(selected,case["power_full_jvp_gain"]) or
        not close(d,case["gain_difference"]) or
        event!=case["event"]):
        raise ValueError("preregistered event/gain contrast changed")
    return d if d is not None else -best


def score(rows,family):
    events=sum(bool(x["event"]) for x in rows)
    admitted=sum(x["power_full_jvp_gain"] is not None for x in rows)
    pairs=[x["gain_difference"] if x["gain_difference"] is not None
           else -x["best_random_full_jvp_gain"] for x in rows]
    rng=random.Random(202625130+len(family))
    values=[]
    for _ in range(2000):
        ids=[rng.randrange(len(pairs)) for _ in pairs]
        values.append(sum(pairs[i] for i in ids)/len(pairs))
    values.sort()
    ci=[values[50],values[1949]]
    return {"n_puzzles":len(rows),"eligible_power":admitted,
            "power_gain_events":events,
            "mean_power_minus_best_random_gain":sum(pairs)/len(pairs),
            "puzzle_bootstrap95":ci,"bootstrap_draws":2000,
            "registered_success":len(rows)==8 and events>=6 and ci[0]>0}


def audit(r):
    if (r["schema"]!=SCHEMA or r["protocol_commit"]!=PROTO or
        r["source_checkpoint_sha256"]!=MODEL_SHA or
        r["source_calibration_npz_sha256"]!=NPZ_SHA or
        r["source_calibration_manifest_sha256"]!=MANIFEST_SHA or
        r["source_fit_fingerprint_sha256"]!=FIT_SHA or
        r["power_updates"]!=4 or r["random_gain_controls"]!=8 or
        r["finite_margin_radius"]!=.05 or
        r["mode"] not in ("pilot","full") or
        r["full_protocol"]!=(r["mode"]=="full")):
        raise ValueError("source-locked model/calibration/experiment contract drifted")
    expected=FAM.manifest()
    if expected!=r["all_roster_sha256"]:
        raise ValueError("fresh Sudoku full-grid manifest differs")
    key=("pilot_30","pilot_34") if r["mode"]=="pilot" else ("test_30","test_34")
    if set(r["evaluated_families"])!=set(key):
        raise ValueError("unregistered heldout cohort evaluated")
    for family in key:
        rosters=FAM.roster(family)
        rows=r["evaluated_families"][family]["per_puzzle"]
        if len(rows)!=len(rosters):
            raise ValueError("full or pilot puzzle count differs")
        for i,(case,puzzle) in enumerate(zip(rows,rosters)):
            if case["puzzle_index"]!=i:
                raise ValueError("puzzle result order changed")
            check_case(case,puzzle,family,i)
        replay=score(rows,family.replace("pilot_","test_"))
        old=r["evaluated_families"][family]["statistics"]
        for name,val in replay.items():
            observed=old[name]
            if isinstance(val,list):
                if len(val)!=len(observed) or any(not close(x,y) for x,y in zip(val,observed)):
                    raise ValueError("puzzle-cluster bootstrap bounds modified")
            elif isinstance(val,float):
                if not close(val,observed):
                    raise ValueError("original paired gain estimate changed")
            elif val!=observed:
                raise ValueError("registered scientific gate changed "+name)
    both=r["full_protocol"] and all(x["statistics"]["registered_success"]
                                    for x in r["evaluated_families"].values())
    if r["registered_success_both_families"]!=both:
        raise ValueError("both-family decision modified")
    return {"schema":"LS-SAE05/independent-stdlib-replay-v1",
            "source_checkpoint_and_original_preheldout_calibration_checked":True,
            "fresh_noncanonical_sudoku_manifests_checked":True,
            "matrix_free_JtJ_power_recurrence_replayed":True,
            "full_JVP_readout_and_VJP_numeric_certificates_checked":True,
            "eight_fixed_random_gain_controls_and_2000_bootstrap_replayed":True,
            "puzzles":sum(len(v["per_puzzle"]) for v in r["evaluated_families"].values()),
            "full_protocol":r["full_protocol"],
            "registered_success_both_families":both,
            "scope":"Independent replay of saved numerical evidence, not a separate pretrained network forward pass."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    b=a.input.read_bytes()
    result=audit(json.loads(b))
    result["source_receipt_sha256"]=hashlib.sha256(b).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
