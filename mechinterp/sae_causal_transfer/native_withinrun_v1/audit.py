"""LS-SAE11 independent standard-library receipt and exact Sudoku audit.

Recomputes original source lock, external source hashes and unique witnesses,
full81 output digits, strict margin bounds, paired bootstrap and declared
hidden-state norm identities. No new PyTorch model execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as A8
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as F

SCHEMA="LS-SAE11/within-forward-original-and-solver-hard-native-v1"
SOURCE_SCHEMA="LS-SAE11/source-only-Champagne-20-v1"
PROTOCOL="e94a4439db78eacdb4988ecd167bcfb62a91230f"
ORIGINAL_SHA="69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
MODEL_SHA=A8.CHECKPOINT
MODES={"pilot":("pilot_champagne",),"reference":("test_52","test_54"),
       "hard":("test_champagne_a","test_champagne_b")}
RADII=("0.20","2.00","8.00")
HORIZONS=("ACT3","ACT6","ACT12")
ARMS=("state_power","logit_power","sae","random")
TOL=2e-4

UNITS=[]
for r in range(9):UNITS.append([9*r+c for c in range(9)])
for c in range(9):UNITS.append([9*r+c for r in range(9)])
for br in (0,3,6):
    for bc in (0,3,6):
        UNITS.append([9*(br+dr)+bc+dc for dr in range(3) for dc in range(3)])


def close(a,b,atol=2e-5,rtol=2e-5):
    return (math.isfinite(float(a)) and math.isfinite(float(b))
            and abs(a-b)<=atol+rtol*max(abs(a),abs(b)))


def complete_logits(rows):
    if not isinstance(rows,list) or len(rows)!=81:
        raise ValueError("all81 signed numerical logits absent")
    for row in rows:
        if (not isinstance(row,list) or len(row)!=9 or any(
                not isinstance(x,(int,float)) or not math.isfinite(x) for x in row)):
            raise ValueError("invalid 9-digit row of native logits")


def digits(rows):
    complete_logits(rows)
    return [1+max(range(9),key=lambda j:row[j]) for row in rows]


def digit_score(rows,puzzle,target,site,clean_prediction):
    preds=digits(rows)
    missing=[i for i,x in enumerate(puzzle) if not x]
    given=[i for i,x in enumerate(puzzle) if x]
    changed=[i for i in range(81) if preds[i]!=clean_prediction[i]]
    conflicts=sum(len(unit)-len({preds[i] for i in unit}) for unit in UNITS)
    valid=bool(conflicts==0 and all(preds[i]==puzzle[i] for i in given))
    return {
      "predicted_digits_81":preds,
      "blank_correct":sum(preds[i]==target[i] for i in missing),
      "blank_total":len(missing),
      "board_exact":preds==target,
      "sudoku_valid_and_clue_consistent":valid,
      "unit_conflicts":conflicts,
      "clue_disagreements":sum(preds[i]!=puzzle[i] for i in given),
      "all_81_prediction_changes":len(changed),
      "blank_prediction_changes":sum(i in missing for i in changed),
      "goal_prediction_changed":preds[site]!=clean_prediction[site],
      "other_cell_prediction_changes":sum(i!=site for i in changed),
      "blank_correct_to_wrong":sum(clean_prediction[i]==target[i] and preds[i]!=target[i] for i in missing),
      "blank_wrong_to_correct":sum(clean_prediction[i]!=target[i] and preds[i]==target[i] for i in missing)
    }


def certificate(clean,edited,puzzle):
    missing=[i for i,v in enumerate(puzzle) if v==0]
    winners=[max(range(9),key=lambda j:clean[i][j]) for i in range(81)]
    margins=[]
    strict=[]
    for i in missing:
        p=winners[i]
        margin=min(clean[i][p]-clean[i][j] for j in range(9) if j!=p)
        margins.append(float(margin))
        strict.append(all(edited[i][p]-edited[i][j]>0 for j in range(9) if j!=p))
    epsilon=max(abs(clean[i][j]-edited[i][j]) for i in range(81) for j in range(9))
    actual=sum(winners[i]!=max(range(9),key=lambda j:edited[i][j]) for i in missing)
    m=min(margins)
    return {
      "minimum_clean_blank_winner_margin":m,
      "all81_absolute_epsilon":epsilon,
      "strict_global_margin_certificate":m>2*epsilon,
      "strict_all_blank_pairwise_certificate":all(strict),
      "strict_pairwise_certified_blank_count":sum(strict),
      "actual_blank_digit_flips":actual
    }


def centered_fro(clean,edited):
    sumsq=0.
    for i in range(81):
        row=[edited[i][j]-clean[i][j] for j in range(9)]
        mean=sum(row)/9
        sumsq+=sum((v-mean)**2 for v in row)
    return math.sqrt(sumsq)


def check_transport(g,clean,edit,source,dose,sigma_min,sigma_max):
    blank=[i for i,v in enumerate(source["x"]) if v==0]
    expected=certificate(clean,edit,source["x"])
    if set(expected)-set(g):
        raise ValueError("finite margin witness missing fields")
    for key,val in expected.items():
        got=g[key]
        if type(val) is bool or type(val) is int:
            if got!=val:raise ValueError("incorrect source-blind output certificate: "+key)
        elif not close(got,val,atol=2e-5):
            raise ValueError("margins/epsilon drift from raw native logits: "+key)
    required=("h_global_l2","l_global_l2","h_gain","l_gain",
              "h_source_site_l2","h_non_site_l2","h_board_l2",
              "h_blank_l2","h_given_l2","h_readout_l2","h_null_l2",
              "readout_fraction","centered_logit_fro",
              "centered_readout_reconstruction_error")
    if any(key not in g or not math.isfinite(g[key]) or g[key]<0 for key in required):
        raise ValueError("nonfinite or missing native state transport")
    if not close(g["h_gain"],g["h_global_l2"]/dose) or not close(
            g["l_gain"],g["l_global_l2"]/dose):
        raise ValueError("source dose transport gain altered")
    for whole,a,b in (("h_global_l2","h_source_site_l2","h_non_site_l2"),
                       ("h_board_l2","h_blank_l2","h_given_l2"),
                       ("h_board_l2","h_readout_l2","h_null_l2")):
        if abs(g[whole]**2-g[a]**2-g[b]**2)>2e-4*(1+g[whole]**2):
            raise ValueError("native Pythagorean norm identity failed")
    expected_fraction=(g["h_readout_l2"]/g["h_board_l2"]
                       if g["h_board_l2"]>1e-12 else 0.)
    if not close(g["readout_fraction"],expected_fraction) or g["readout_fraction"]>1+1e-5:
        raise ValueError("centered 9-digit readout fraction invalid")
    expected_fro=centered_fro(clean,edit)
    if not close(g["centered_logit_fro"],expected_fro,atol=5e-5):
        raise ValueError("same-run centered native output Frobenius wrong")
    if g["centered_readout_reconstruction_error"]>TOL:
        raise ValueError("registered same-run 9-digit linear head identity exceeded tolerance")
    # Singular spectrum of the centered output head gives an independent bound.
    if expected_fro+TOL < sigma_min*g["h_readout_l2"] or (
            expected_fro-TOL > sigma_max*g["h_readout_l2"]):
        raise ValueError("centered-digit-head readout singular-value inequality failed")
    classes={
      "contracted":g["h_gain"]<0.1,
      "persistent_output_silent":g["h_gain"]>=1 and expected["actual_blank_digit_flips"]==0,
      "readout_small":g["h_board_l2"]>1e-8 and g["readout_fraction"]<0.1
    }
    if g["classification"]!=classes:
        raise ValueError("original descriptive state-transport labels altered")


def signed_grid(native,transport,scores,source,site,directions,smin,smax):
    if set(native)!=set(HORIZONS) or set(transport)!=set(HORIZONS) or set(scores)!=set(HORIZONS):
        raise ValueError("LS-SAE11 ACT3/6/12 horizon coverage incomplete")
    for horizon in HORIZONS:
        raw=native[horizon]; geometry=transport[horizon]; task=scores[horizon]
        status={a for a,ok in directions.items() if ok}
        if set(raw)!={"clean"}|status or set(geometry)!={"arms"} or set(task)!={"arms","clean"}:
            raise ValueError("original source arm grid, status or clean result changed")
        clean=raw["clean"]
        clean_pred=digits(clean)
        if task["clean"]!=digit_score(clean,source["x"],source["target"],site,clean_pred):
            raise ValueError("clean exact Sudoku score differs from model output")
        if set(geometry["arms"])!=status or set(task["arms"])!=status:
            raise ValueError("native transport/null arms omitted")
        for arm in ARMS:
            if not directions[arm]:continue
            if set(raw[arm])!=set(RADII) or set(geometry["arms"][arm])!=set(RADII) or set(task["arms"][arm])!=set(RADII):
                raise ValueError("physical signed radius grid incomplete")
            for rad in RADII:
                for group in (raw[arm][rad],geometry["arms"][arm][rad],task["arms"][arm][rad]):
                    if set(group)!={"plus","minus"}:raise ValueError("missing plus/minus edit")
                for sign in ("plus","minus"):
                    edit=raw[arm][rad][sign]
                    if task["arms"][arm][rad][sign]!=digit_score(
                        edit,source["x"],source["target"],site,clean_pred):
                        raise ValueError("edited Sudoku correctness, clue or flip tally changed")
                    check_transport(geometry["arms"][arm][rad][sign],
                                    clean,edit,source,float(rad),smin,smax)


def source_audit(source):
    if (source["schema"]!=SOURCE_SCHEMA or source["registration_commit"]!=PROTOCOL
        or source["external_repo_commit"]!=F.COMMIT
        or source["source_file_hashes"]!=[list(x) for x in F.SOURCE_FILES]
        or source["pilot_cases"]!=4 or source["full_cases"]!=16
        or source["model_inference_executed"] is not False):
        raise ValueError("registered external Sudoku source altered")
    F.admitted(source["rosters"])
    if {k:F.fingerprint(v) for k,v in source["rosters"].items()}!=source["fingerprints"]:
        raise ValueError("external source-roster hash differs from source stage")
    all_rows=[x for k in F.SOURCE_GROUPS for x in source["rosters"][k]]
    for i,row in enumerate(all_rows):
        name,git_sha=F.SOURCE_FILES[i]
        if (row["source_index"]!=i or row["source_file"]!=name
            or row["source_git_blob_sha1"]!=git_sha or row["source_repository_commit"]!=F.COMMIT
            or row["seed"]!=int(name.removesuffix(".txt")) or row["holes"]!=row["x"].count(0)):
            raise ValueError("exact public fixed source file index or hash changed")
        raw=bytes(48+v for v in row["x"])
        if F.git_blob_sha(raw)!=git_sha or hashlib.sha256(raw).hexdigest()!=row["source_raw_sha256"]:
            raise ValueError("original source file content differs from pinned public blob")
        exact=F.solve_unique(row["x"])
        if (exact["solutions"]!=1 or exact["target"]!=row["target"]
            or exact["search_nodes"]!=row["solver_search_nodes"]):
            raise ValueError("registered exact unique Sudoku solution or solver receipt changed")
        if F.S7.visible_structure(row["x"])!=row["visible_structural_difficulty"]:
            raise ValueError("source-only naked+hidden single logic changed")
    return True


def score_family(cases,family):
    if family not in ("test_champagne_a","test_champagne_b") or len(cases)!=8:
        raise ValueError("registered 8-puzzle full hard family required")
    diffs=[];inexact=0;events=0
    for case in cases:
        row=case["native_task_scores"]["ACT6"]
        blank=row["clean"]["blank_correct"]
        inexact+=int(blank<row["clean"]["blank_total"])
        def gain(arm):
            signed=row["arms"].get(arm,{}).get("8.00")
            if signed is None:return 0.
            return (signed["plus"]["blank_correct"]+signed["minus"]["blank_correct"])/2-blank
        diff=gain("state_power")-gain("random")
        diffs.append(float(diff));events+=int(diff>=1)
    rng=random.Random(202611170+len(family))
    boot=sorted(sum(diffs[rng.randrange(8)] for _ in range(8))/8 for _ in range(2000))
    return {
      "n_puzzles":8,"clean_ACT6_inexact":inexact,
      "paired_state_minus_random_blank_correctness_gain":diffs,
      "events_D_at_least_one":events,
      "mean_paired_gain":sum(diffs)/8,
      "puzzle_bootstrap95":[boot[50],boot[1949]],
      "bootstrap_draws":2000,
      "registered_success":bool(inexact>=4 and events>=6 and boot[50]>0)
    }


def audit(report,source,original,source_sha):
    A8.audit(original);source_audit(source)
    mode=report["mode"]
    if (mode not in MODES or report["schema"]!=SCHEMA
        or report["preregistration_commit"]!=PROTOCOL
        or report["source_ls_sae08_sha256"]!=ORIGINAL_SHA
        or report["source_original_checkpoint_sha256"]!=MODEL_SHA
        or report["external_source_receipt_sha256"]!=source_sha
        or report["external_source_roster_fingerprints"]!=source["fingerprints"]
        or report["horizons"]!=list(HORIZONS) or report["physical_radii"]!=list(RADII)
        or report["source_direction_arm_order"]!=list(ARMS)
        or report["same_run_centered_readout_tolerance"]!=TOL
        or set(report["families"])!=set(MODES[mode])):
        raise ValueError("source-registered same-forward experiment altered")
    singular=report["readout_nonzero_singular_values"]
    rank=report["readout_centered_rank"]
    if (not 1<=rank<=8 or len(singular)!=rank or any(
         not math.isfinite(x) or x<=0 for x in singular)
         or any(singular[j+1]>singular[j] for j in range(rank-1))):
        raise ValueError("centered nine-digit rank/singular declaration invalid")
    smin,smax=singular[-1],singular[0]
    for group in MODES[mode]:
        expected=(original["families"][group]["source_roster"] if mode=="reference"
                 else source["rosters"][group])
        found=report["families"][group]
        if found["source_roster"]!=expected or len(found["per_puzzle"])!=len(expected):
            raise ValueError("source roster or original denominator changed")
        cases=found["per_puzzle"]
        for index,(puzzle,case) in enumerate(zip(expected,cases)):
            donor_family,donor_i=(group,index) if mode=="reference" else (
                ("test_52",index) if mode=="pilot" and index<2 else
                ("test_54",index-2) if mode=="pilot" else
                ("test_52",index) if group=="test_champagne_a" else
                ("test_54",index))
            donor=original["families"][donor_family]["per_puzzle"][donor_i]
            status=donor["source_direction_status"]
            blanks=[i for i,x in enumerate(puzzle["x"]) if x==0]
            site=blanks[len(blanks)//2]
            if (case["source_seed"]!=puzzle["seed"]
                or case["source_file"]!=puzzle.get("source_file")
                or case["source_site"]!=site
                or case["donor_family"]!=donor_family
                or case["donor_index"]!=donor_i
                or case["source_direction_status"]!=status
                or not math.isfinite(case["source_ACT1_zH_norm"])
                or case["source_ACT1_zH_norm"]<=0):
                raise ValueError("source, donor or physical intervention site altered")
            signed_grid(case["native_81x9_logits"],case["same_run_transport"],
                        case["native_task_scores"],puzzle,site,status,smin,smax)
            dr=case["cross_run_original_drift"]
            if mode!="reference" and dr is not None:
                raise ValueError("external dataset cannot have original reference logit drift")
            if mode=="reference":
                if set(dr)!=set(HORIZONS):
                    raise ValueError("reference diagnostic drift ACT horizon missing")
                for horizon in HORIZONS:
                    old=donor["native_81x9_logits"][horizon]
                    new=case["native_81x9_logits"][horizon]
                    for arm in (["clean"]+[a for a in ARMS if status[a]]):
                        if arm=="clean":
                            pairs=[(dr[horizon]["clean"],old["clean"],new["clean"])]
                        else:
                            pairs=[(dr[horizon]["arms"][arm][rad][sign],
                                    old[arm][rad][sign],new[arm][rad][sign])
                                   for rad in RADII for sign in ("plus","minus")]
                        for rec,ref,now in pairs:
                            complete_logits(ref);complete_logits(now)
                            err=max(abs(ref[i][j]-now[i][j]) for i in range(81) for j in range(9))
                            changes=sum(a!=b for a,b in zip(digits(ref),digits(now)))
                            if not close(rec["max_absolute_logit_drift"],err,atol=2e-5) or rec["all81_argmax_changes"]!=changes:
                                raise ValueError("LS-SAE08 archived difference diagnostic modified")
        if mode=="hard" and found["statistics"]!=score_family(cases,group):
            raise ValueError("original prospective hard paired effect, bootstrap or gate altered")
        if mode!="hard" and "statistics" in found:
            raise ValueError("engineering/reference mechanism given a hard-science gate")
    expected_success=mode=="hard" and all(
        report["families"][k]["statistics"]["registered_success"] for k in MODES["hard"])
    if report["registered_hard_scientific_success"]!=expected_success:
        raise ValueError("registered two-family hard outcome differs")
    return {
      "schema":"LS-SAE11/independent-source-exact-native81-margin-and-signed-audit-v1",
      "original_LS_SAE08_pretrained_and_source_receipts_replayed":True,
      "external_exact_unique_puzzles_and_model_blind_hardness_replayed":True,
      "all_source_donor_doses_signs_and_ACT3_6_12_replayed":True,
      "all_finite_81x9_logits_predicted_digits_and_Sudoku_validity_recomputed":True,
      "all_strict_blank_margin_certificates_and_transport_norm_identities_checked":True,
      "all_original_2000_cluster_bootstraps_and_primary_gates_replayed":True,
      "mode":mode,"puzzles":sum(len(x["per_puzzle"]) for x in report["families"].values()),
      "registered_hard_scientific_success":expected_success,
      "limitation":"Independent stdlib recomputation from model-saved logits/source. Hidden state full 97x512 matrices are not re-executed or independently reconstructed.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    b=a.original.read_bytes()
    if hashlib.sha256(b).hexdigest()!=ORIGINAL_SHA:
        raise ValueError("original LS-SAE08 source SHA mismatch")
    sr=a.source.read_bytes()
    result=audit(json.loads(a.input.read_text()),json.loads(sr),json.loads(b),
                 hashlib.sha256(sr).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
