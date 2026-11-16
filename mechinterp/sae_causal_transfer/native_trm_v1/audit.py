"""LS-SAE02 independent stdlib audit of archived native Torch observations.

Replays rosters, goal selection, physical edit geometry receipts, full logit
tables, autograd first-order dot products, random rank tests and per-puzzle
bootstrap. This module never imports Torch, numpy or model weights.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from mechinterp.sae_causal_transfer.native_trm_v1 import families as FAM

ORIGINAL_SHA="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
PROTOCOL_SHA="3e5e29df0886c3754c2e5b9a07050953444f860b"
DEPTHS=("0","1","2")


def close(a,b,eps=1e-7):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=eps*max(1,abs(a),abs(b))


def norm(xs):
    return math.sqrt(sum(x*x for x in xs))


def dot(x,y):
    if len(x)!=len(y):
        raise ValueError("different vector dimensions")
    return sum(a*b for a,b in zip(x,y))


def cosine(x,y):
    denominator=norm(x)*norm(y)
    return abs(dot(x,y))/denominator if denominator>1e-12 else None


def verify_rosters(receipt):
    if set(receipt["source_rosters"])!=set(FAM.BASES):
        raise ValueError("new Sudoku cohort names drifted")
    seen_input=set();seen_complete=set()
    counts=FAM.PILOT if receipt["mode"]=="pilot" else FAM.COUNTS
    for family,seedbase in FAM.BASES.items():
        rows=receipt["source_rosters"][family]
        if len(rows)!=counts[family]:
            raise ValueError("registered split cardinality changed")
        if FAM.fingerprint(rows)!=receipt["source_roster_sha256"][family]:
            raise ValueError("fresh Sudoku manifest SHA mismatch")
        for i,row in enumerate(rows):
            if row["seed"]!=seedbase+i or row["holes"]!=FAM.HOLES[family]:
                raise ValueError("wrong puzzle seed or hole count")
            x,t=row["x"],row["target"]
            if not FAM.valid_solution(t) or FAM.rectangles(t)!=row["rectangles"] or row["rectangles"]<=0:
                raise ValueError("invalid or canonical solution")
            if (len(x)!=81 or x.count(0)!=row["holes"] or
                any(a and a!=b for a,b in zip(x,t)) or FAM.count_solutions(x)!=1):
                raise ValueError("incorrect fresh puzzle or solution count")
            if tuple(x) in seen_input or tuple(t) in seen_complete:
                raise ValueError("duplicated complete Sudoku solution or input")
            seen_input.add(tuple(x));seen_complete.add(tuple(t))


def verify_geometry(goal):
    geometry=goal["geometry"]
    active=geometry["active_ids"];protected=geometry["protected_ids"]
    inactive=geometry["inactive_ids"]
    selected=goal["feature"]
    if (len(active)!=4 or sorted(set(active+inactive))!=list(range(96)) or
        len(inactive)!=92 or len(protected)!=3 or selected not in active or
        set(protected)!=(set(active)-{selected})):
        raise ValueError("SAE support identity incorrect")
    code=geometry["clean_code"]
    if len(code)!=96 or any(not math.isfinite(z) for z in code):
        raise ValueError("nonfinite original clean feature code")
    ordered=sorted(code,reverse=True)
    if ordered[0]<=0 or ordered[0]-ordered[1]<=1e-8 or code[selected]!=ordered[0]:
        raise ValueError("target not strict naturally active top-one")
    edits=goal["physical_edit_48"]
    if set(edits)!={"kill","baseline","best"} or any(len(v)!=48 for v in edits.values()):
        raise ValueError("unexpected 48D physical edit vectors")
    dose=norm(edits["kill"])
    if any(not close(norm(v),dose,1e-8) for v in edits.values()):
        raise ValueError("physical dose is not matched")
    if not close(dose,geometry["physical_norm"]) or dose>6*geometry["native_decoder_norm"]*(1+1e-7):
        raise ValueError("native decoder dose cap or selected norm changed")
    shift={name:geometry["all96_"+name+"_preactivation_shift"]
           for name in ("kill","baseline","best")}
    if any(len(v)!=96 for v in shift.values()):
        raise ValueError("incomplete 96-feature affine shift")
    if not close(shift["kill"][selected],code[selected]+.02,2e-6):
        raise ValueError("target feature kill preactivation differs")
    if any(abs(shift["kill"][j])>2e-6 for j in protected):
        raise ValueError("other three active encoder rows moved")
    if any(abs(shift[name][j])>2e-6 for j in active for name in ("baseline","best")):
        raise ValueError("four active rows not tangent-sham protected")
    bdist=norm([shift["kill"][j]-shift["baseline"][j] for j in inactive])
    odist=norm([shift["kill"][j]-shift["best"][j] for j in inactive])
    if (not close(bdist,geometry["inactive_residual_baseline"],2e-6)
        or not close(odist,geometry["inactive_residual_best"],2e-6)):
        raise ValueError("inactive preactivation mismatch changed")
    if geometry["inactive_residual_floor"]>odist+2e-6 or odist>bdist+2e-6:
        raise ValueError("least-squares floor or sham ordering violated")
    if geometry["tangent_dimension"]!=44 or not 0<=geometry["inactive_encoder_rank"]<=48:
        raise ValueError("tangent/inactive rank mismatch")
    choices=geometry["candidates"]
    if len(choices)!=131 or choices[0]["candidate"]!="baseline" or not choices[0]["accepted"]:
        raise ValueError("baseline or candidate budget absent")
    if choices[1]["candidate"]!="least_squares_positive" or choices[2]["candidate"]!="least_squares_negative":
        raise ValueError("least-squares candidate order changed")
    for idx,q in enumerate(choices[3:]):
        if q["candidate"]!=f"random_{idx:03d}":
            raise ValueError("128 sham candidate enumeration differs")
    accepted=[v for v in choices if v["accepted"]]
    if not accepted or any(v["epsilon"] is None or v["epsilon"]<0 for v in accepted):
        raise ValueError("sham candidates missing inactive residual")
    if (not close(odist,min(v["epsilon"] for v in accepted),2e-6)
        or geometry["selected_candidate"] not in [v["candidate"] for v in accepted]):
        raise ValueError("best sham was not selected by smallest residual")
    if not 1<=geometry["baseline_seed_attempts"]<=4:
        raise ValueError("initial four-try baseline seed count changed")


def verify_depth(go,depth):
    v=go["depths"][depth]
    if v["depth"]!=int(depth):
        raise ValueError("loop depth receipt changed")
    full=v["all_arm_logits_4x9"]
    if len(full)!=4 or any(len(q)!=9 or not all(math.isfinite(z) for z in q) for q in full):
        raise ValueError("native four-arm digit logits are incomplete")
    if full[0]!=v["clean_logits_9"]:
        raise ValueError("clean logits duplicate record disagrees")
    y=v["true_digit_zero_based"]
    rival=v["clean_rival_digit_zero_based"]
    if not all(isinstance(t,int) and 0<=t<9 for t in (y,rival)) or rival==y:
        raise ValueError("target and competitor digit invalid")
    c=max((i for i in range(9) if i!=y),key=lambda i:full[0][i])
    if rival!=c:
        raise ValueError("clean non-target rival was changed by editing")
    pred=[max(range(9),key=lambda i:row[i]) for row in full]
    if pred!=v["predicted_digits_by_arm"] or pred[0]!=v["predicted_clean_digit_zero_based"]:
        raise ValueError("clean or intervention predictions changed")
    margins=[row[y]-row[rival] for row in full]
    for name,idx in (("clean",0),("kill",1),("baseline",2),("best",3)):
        if not close(margins[idx],v["margins"][name]):
            raise ValueError("raw task margin not reproducible: "+name)
    d=margins[3]-margins[1]
    baseline=margins[2]-margins[1]
    if not close(d,v["kill_minus_best_margin_loss"]) or not close(baseline,v["kill_minus_baseline_margin_loss"]):
        raise ValueError("paired causal effect differs from four-arm logits")
    gradient=v["gradient_48"]
    kill=go["physical_edit_48"]["kill"]
    sham=go["physical_edit_48"]["best"]
    if len(gradient)!=48 or not all(math.isfinite(a) for a in gradient):
        raise ValueError("native autograd vector corrupted")
    estimate=dot(gradient,[a-b for a,b in zip(kill,sham)])
    if not close(estimate,v["predicted_first_order_gradient_effect"],3e-6):
        raise ValueError("gradient projection arithmetic changed")
    if (not close(d-estimate,v["nonlinear_residual"],3e-6)
        or not close(abs(d-estimate)/(1+abs(d)),v["relative_nonlinear_residual"],3e-6)):
        raise ValueError("native nonlinear gradient discrepancy changed")
    winner=pred[0]==y
    if v["clean_correct"]!=winner:
        raise ValueError("clean correctness changed")
    for arm,idx in (("kill",1),("baseline",2),("best",3)):
        if v[arm+"_clean_correct_to_wrong"]!=(winner and pred[idx]!=y):
            raise ValueError("clean-correct-to-wrong flip changed")
    ranks=v["random_rank1_controls"]
    sae=cosine(gradient,kill)
    task=cosine(gradient,v["task_only_direction_48"])
    if not close(sae,ranks["sae_alignment"]) or not close(task,ranks["task_only_alignment"]):
        raise ValueError("rank-one SAE or task-only head alignment changed")
    observed=ranks["random_alignments"]
    if sae is None:
        if observed or ranks["rank_p"] is not None:
            raise ValueError("zero-gradient abstention altered")
    else:
        if len(observed)!=128 or any(z is None or not math.isfinite(z) or not 0<=z<=1+1e-7 for z in observed):
            raise ValueError("128 rank-one null magnitude controls missing")
        p=(1+sum(z>=sae-1e-15 for z in observed))/129
        if not close(p,ranks["rank_p"]):
            raise ValueError("rank-based empirical p-value differs")


def score_replay(rows,family):
    sums=[];counts=[];fk=fs=0
    for r in rows:
        valid=[g for g in r["goals"] if g["eligible"]]
        vals=[g["depths"]["2"]["kill_minus_best_margin_loss"] for g in valid]
        sums.append(sum(vals));counts.append(len(vals))
        fk+=sum(g["depths"]["2"]["kill_clean_correct_to_wrong"] for g in valid)
        fs+=sum(g["depths"]["2"]["best_clean_correct_to_wrong"] for g in valid)
    n=sum(counts);clusters=sum(c>0 for c in counts)
    rng=random.Random(202619130+len(family))
    draws=[]
    for _ in range(2000):
        ids=[rng.randrange(len(rows)) for _ in rows]
        denominator=sum(counts[i] for i in ids)
        if denominator:
            draws.append(sum(sums[i] for i in ids)/denominator)
    draws.sort()
    ci=[draws[50],draws[1949]] if len(draws)==2000 else None
    good=(len(rows)==16 and n>=12 and clusters>=8 and len(draws)==2000
          and ci is not None and ci[0]>0 and fk>fs)
    return {"eligible_goals":n,"eligible_puzzles":clusters,
            "selected_goals":len(rows)*2,"paired_mean_depth2":sum(sums)/n if n else None,
            "puzzle_bootstrap95":ci,"valid_bootstrap_resamples":len(draws),
            "kill_clean_correct_to_wrong":fk,"best_clean_correct_to_wrong":fs,
            "registered_success":bool(good)}


def audit(report):
    if (report["schema"]!="LS-SAE02/native-TRM-actual-autograd-depth-gradient-v1"
        or report["checkpoint_tag"]!="gaoxin492"
        or report["checkpoint_file_sha256"]!=ORIGINAL_SHA
        or report["preregistration_commit"]!=PROTOCOL_SHA):
        raise ValueError("unrecognized source-pinned experiment")
    if report["mode"] not in ("pilot","full") or report["full_protocol"]!=(report["mode"]=="full"):
        raise ValueError("protocol and pilot/full tag inconsistent")
    fit=report["fit_receipt"]
    if (fit["checkpoint_file_sha256"]!=ORIGINAL_SHA
        or fit["source_fit_roster_sha256"]!=report["source_roster_sha256"]["fit_30"]
        or (fit["sae_seed"],fit["sampling_seed"],fit["optimizer_steps"],fit["batch_size"])!=(71,20261971,200,256)
        or (fit["pca_rank"],fit["sae_width"],fit["topk"])!=(48,96,4)):
        raise ValueError("fit-only SAE training budget/fingerprint drift")
    verify_rosters(report)
    if (report["depth_grid"]!=[0,1,2] or report["random_rank1_observer_budget"]!=128
        or set(report["test"])!={"test_30","test_34"}):
        raise ValueError("changed depth or heldout observer grid")
    for family in ("test_30","test_34"):
        raw=report["test"][family]["per_puzzle"]
        roster=report["source_rosters"][family]
        if len(raw)!=len(roster):
            raise ValueError("heldout puzzle count differs")
        for i,(case,puzzle) in enumerate(zip(raw,roster)):
            positions=FAM.visible_goals(puzzle["x"])
            if (case["seed"]!=puzzle["seed"] or case["puzzle_index"]!=i
                or case["goal_positions"]!=list(positions) or len(case["goals"])!=2):
                raise ValueError("goal mask selected from answer or puzzle changed")
            for slot,g in enumerate(case["goals"]):
                if (g["position"]!=positions[slot] or
                    g["case_key"]!=f"LS-SAE02|{family}|{puzzle['seed']}|{slot}"):
                    raise ValueError("frozen goal identity changed")
                if g["eligible"]:
                    if g["reason"]!="eligible":
                        raise ValueError("accepted geometry has rejection reason")
                    verify_geometry(g)
                    if set(g["depths"])!=set(DEPTHS):
                        raise ValueError("registered 0/1/2 ACT grid modified")
                    for depth in DEPTHS:
                        if g["depths"][depth]["true_digit_zero_based"]!=puzzle["target"][g["position"]]-1:
                            raise ValueError("native task label differs from target solution")
                        verify_depth(g,depth)
                elif g["reason"]=="eligible" or "depths" in g or "physical_edit_48" in g:
                    raise ValueError("ineligible goal scored task outcome")
        original=report["test"][family]["statistics"]
        replay=score_replay(raw,family)
        for key,val in replay.items():
            prior=original[key]
            if isinstance(val,list):
                if not isinstance(prior,list) or len(val)!=len(prior) or any(not close(a,b) for a,b in zip(val,prior)):
                    raise ValueError("clustered bootstrap CI differs")
            elif isinstance(val,float) or val is None:
                if not close(val,prior):
                    raise ValueError("registered raw mean differs: "+key)
            elif val!=prior:
                raise ValueError("registered success or coverage differs: "+key)
        # Recompute every depth-specific gradient statistic from the raw goals.
        for depth in DEPTHS:
            arr=[g["depths"][depth] for r in raw for g in r["goals"] if g["eligible"]]
            summary=original["by_depth"][depth]
            for key,column in (
                ("mean_kill_minus_best","kill_minus_best_margin_loss"),
                ("mean_gradient_prediction","predicted_first_order_gradient_effect"),
                ("mean_relative_nonlinear_residual","relative_nonlinear_residual")
            ):
                val=sum(v[column] for v in arr)/len(arr) if arr else None
                if not close(val,summary[key]):
                    raise ValueError("depth-specific observed gradient summary changed")
            ps=[v["random_rank1_controls"]["rank_p"] for v in arr
                if v["random_rank1_controls"]["rank_p"] is not None]
            ps.sort()
            med=(ps[len(ps)//2] if len(ps)%2 else (ps[len(ps)//2-1]+ps[len(ps)//2])/2) if ps else None
            if not close(med,summary["median_rank_p"]):
                raise ValueError("random-observer-rank depth median changed")
    both=report["full_protocol"] and all(report["test"][f]["statistics"]["registered_success"]
                                         for f in ("test_30","test_34"))
    if report["registered_success_both_families"]!=both:
        raise ValueError("two-family scientific gate modified")
    return {"schema":"LS-SAE02/independent-stdlib-replay-v1",
            "checkpoint_and_calibration_rosters_checked":True,
            "fresh_exact_unique_solutions_checked":True,
            "all96_geometry_and_frozen_goal_eligibility_checked":True,
            "all_arm_logits_and_native_gradient_arithmetic_checked":True,
            "128_random_rank_one_controls_and_cluster_bootstrap_replayed":True,
            "full_protocol":report["full_protocol"],
            "two_family_scientific_gate":both,
            "n_goal_cases":sum(len(r["goals"]) for f in report["test"].values() for r in f["per_puzzle"]),
            "scope":"Independent replay of saved numerical receipts; Torch model and gradient computation remains source-pinned in native runner."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    b=args.input.read_bytes()
    report=audit(json.loads(b))
    report["full_raw_sha256"]=hashlib.sha256(b).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
