"""Independent NEC-03E raw outcomes and attenuation-only direct-dose audit.

No intervention, feature-selection or autoencoder implementation is imported.
All 16-state labels, heldout memberships, physical gains and summary
counts are checked against source-locked records with stdlib arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from mechinterp.neuron_communities.audit_hosted import (
    SOURCE_SHA, raw_sha, read_json, verify_registered_partition,
)
from mechinterp.neuron_communities.audit_nec03 import expected_target, correct_count

PAIRS=(("native","sham"),("native","sae"),("sae","sae_sham"))


def check_site(site: dict) -> dict:
    """Reconstruct allowable attenuation and exact equal output dose."""
    required=("full_left_dose","full_right_dose","shared_achievable_dose",
              "realized_left_dose","realized_right_dose","left_gain",
              "right_gain","relative_dose_mismatch","matched_within_1pct",
              "shared_nonzero","retained_fraction_of_full_left_dose")
    for key in required:
        if key not in site or len(site[key])!=16:
            raise AssertionError("incomplete local site-dose provenance: "+key)
    ratios=[]
    matched=0
    active=0
    for j in range(16):
        a=float(site["full_left_dose"][j])
        b=float(site["full_right_dose"][j])
        t=float(site["shared_achievable_dose"][j])
        ga=float(site["left_gain"][j])
        gb=float(site["right_gain"][j])
        r1=float(site["realized_left_dose"][j])
        r2=float(site["realized_right_dose"][j])
        mismatch=abs(r1-r2)/max(t,1e-6)
        vals=(a,b,t,ga,gb,r1,r2,mismatch)
        if not all(math.isfinite(x) and x>=0 for x in vals):
            raise AssertionError("nonfinite or negative dose")
        if max(ga,gb)>1+1e-6:
            raise AssertionError("attenuation-only policy amplified a donor")
        if abs(t-min(a,b))>1e-4:
            raise AssertionError("target dose differs from pair's achievable minimum")
        if abs(ga*a-t)>1e-4 or abs(gb*b-t)>1e-4:
            raise AssertionError("dose gain was not computed on original donors")
        if abs(r1-t)>1e-4 or abs(r2-t)>1e-4:
            raise AssertionError("output dose differs from frozen shared target")
        if abs(mismatch-float(site["relative_dose_mismatch"][j]))>1e-6:
            raise AssertionError("dose-match error incorrectly saved")
        if bool(site["matched_within_1pct"][j])!=(mismatch<=0.01+1e-7):
            raise AssertionError("false physical-matching flag")
        on=(t>1e-6)
        if bool(site["shared_nonzero"][j])!=on:
            raise AssertionError("physical activity flag incorrect")
        ratio=t/a if a>1e-6 else 0.
        if abs(ratio-float(site["retained_fraction_of_full_left_dose"][j]))>1e-4:
            raise AssertionError("physical signal-retention ratio incorrect")
        ratios.append(ratio)
        matched+=int(mismatch<=0.01+1e-7)
        active+=int(on)
    return {"n":16,"matched":matched,"active":active,"ratios":ratios,
            "common_dose":sum(site["shared_achievable_dose"])}


def check_one(root: Path, seed: int, task: str, source: dict):
    folder=root/"nec03e"/f"seed{seed}_{task}"
    result=read_json(folder/"pairwise_report.json")
    if (result["schema"]!="NEC-03E/attenuation-only-two-way-matched-physical-dose/v1"
        or (result["seed"],result["task"])!=(seed,task)
        or result["source_sha256"]!=SOURCE_SHA):
        raise AssertionError("wrong task/schema/source")
    if (not result["dose_policies"]["only_attenuation_no_amplification"]
        or result["dose_policies"]["pair_order"]!=[list(x) for x in PAIRS]
        or result["dose_policies"]["controls_never_scale_by_outcomes"] is not True):
        raise AssertionError("physical-dose policy modified")
    if (result["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY"
        or result["no_prospective_significance_inferred"] is not True):
        raise AssertionError("illegitimate prospective claim")
    previous=root/"nec03d"/f"seed{seed}_{task}"
    frozen=(previous/"prepare.json").read_bytes()
    if hashlib.sha256(frozen).hexdigest()!=result["source_only_sae_prepare_sha256"]:
        raise AssertionError("source-only SAE selection changed")
    prep=json.loads(frozen)
    if raw_sha(previous/prep["sae"]["sae_checkpoint_file"])!=result["sae_checkpoint_sha256"]:
        raise AssertionError("learned SAE checkpoint changed")
    original=root/"analysis"/f"seed{seed}_{task}"/"discovery.json"
    graph=read_json(original)
    if result["original_source_candidate_sha256"]!=raw_sha(original):
        raise AssertionError("frozen source candidate graph changed")
    if result["model_identity"]!=graph["identity"] or prep["model_identity"]!=graph["identity"]:
        raise AssertionError("frozen trunk identity differs")
    roster=[c["id"] for c in graph["candidates"]]
    if result["candidates"]!=len(roster) or result["comparisons"]!=3*len(roster):
        raise AssertionError("wrong comparison family")
    path=folder/result["raw_filename"]
    if raw_sha(path)!=result["raw_sha256"]:
        raise AssertionError("raw paired record modified")
    lines=path.read_text().splitlines()
    if len(lines)!=result["raw_records"]:
        raise AssertionError("raw record count mismatch")
    if len(lines)!=48*len(roster)*3:
        raise AssertionError("incomplete paired context family")
    assignment=source["context_assignment"]
    counts=defaultdict(lambda:{"clean":0,"hubs":0,"left":0,"right":0})
    site_scores=defaultdict(lambda:{"n":0,"matched":0,"active":0,"ratios":[],"dose_sum":0.0})
    seen=set()
    for line in lines:
        rec=json.loads(line)
        key=(rec["candidate_id"],tuple(rec["comparison"]))
        context="".join(map(str,rec["context_bits"]))
        if key[0] not in roster or key[1] not in PAIRS:
            raise AssertionError("unknown selected comparison")
        if assignment.get(context)!="heldout":
            raise AssertionError("raw result outside frozen heldout contexts")
        if (key,context) in seen:
            raise AssertionError("repeated candidate/pair/context")
        seen.add((key,context))
        if (rec["seed"],rec["task"])!=(seed,task):
            raise AssertionError("wrong model source label")
        if rec["focal_mask_order"]!=list(range(16)):
            raise AssertionError("counterfactual history order differs")
        labels=[expected_target(i,task) for i in range(16)]
        if rec["labels"]!=labels:
            raise AssertionError("teacher labels disagree")
        for arm,field in (("clean","clean_logits"),("hubs","hubs_logits"),
                          ("left","left_logits"),("right","right_logits")):
            counts[key][arm]+=16-correct_count(rec[field],labels)
        if not rec["site_doses"]:
            raise AssertionError("candidate has no active recurrent visits")
        for str_visit,samples in rec["site_doses"].items():
            v=int(str_visit)
            if v not in range(12):
                raise AssertionError("visit outside tied recurrence")
            x=check_site(samples)
            total=site_scores[key]
            total["n"]+=x["n"]
            total["matched"]+=x["matched"]
            total["active"]+=x["active"]
            total["ratios"].extend(x["ratios"])
            total["dose_sum"]+=x["common_dose"]
    heldout=set(x for x,owner in assignment.items() if owner=="heldout")
    for candidate in roster:
        for pair in PAIRS:
            if {ctx for (key,ctx) in seen if key==(candidate,pair)}!=heldout:
                raise AssertionError("candidate/pair has missing source groups")
    for r in result["results"]:
        key=(r["candidate_id"],tuple(r["pair"]))
        if r["candidate_id"] not in roster or tuple(r["pair"]) not in PAIRS:
            raise AssertionError("unknown result pair")
        x=counts[key]
        if [r["model_clean_errors_of_768"],r["hubs_only_errors_of_768"],
            r["left_errors_of_768"],r["right_errors_of_768"]] != [
                x["clean"],x["hubs"],x["left"],x["right"]]:
            raise AssertionError("reported class errors differ from raw")
        if r["left_minus_right_correct_predictions"]!=x["right"]-x["left"]:
            raise AssertionError("reported pairwise correctness contrast differs")
        if r["left_conditional_rescues_vs_hubs"]!=x["hubs"]-x["left"]:
            raise AssertionError("left recovery count mismatch")
        if r["right_conditional_rescues_vs_hubs"]!=x["hubs"]-x["right"]:
            raise AssertionError("right recovery count mismatch")
        raw=site_scores[key]
        summary=r["achievable_dose"]
        ratios=sorted(raw["ratios"])
        median=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
        matching=raw["matched"]/raw["n"]
        active=raw["active"]/raw["n"]
        if (raw["n"]!=summary["site_sample_count"] or
            abs(matching-summary["fraction_matched_within_1pct"])>1e-7 or
            abs(active-summary["fraction_with_positive_shared_dose"])>1e-7 or
            abs(median-summary["median_fraction_full_left_dose_preserved"])>1e-6):
            raise AssertionError("local physical-dose summary differs from raw")
        feasible=matching>=0.99 and active>=0.8 and median>=0.25
        if feasible != summary["candidate_physically_comparable_and_signal_sufficient"]:
            raise AssertionError("feasibility gate incorrectly applied")
    return {
        "seed":seed,"task":task,
        "candidate_roster_sha256":result["original_source_candidate_sha256"],
        "source_sae_sha256":result["sae_checkpoint_sha256"],
        "raw_pairwise_context_sha256":result["raw_sha256"],
        "candidate_count":len(roster),
        "pair_count":len(roster)*3,
        "candidate_pair_context_records":len(lines),
        "feasible_comparisons":sum(
            bool(row["achievable_dose"]["candidate_physically_comparable_and_signal_sufficient"])
            for row in result["results"]),
        "physical_equalization_verified":True,
        "all_48_source_contexts_per_pair_verified":True,
        "outcome_errors_recomputed_from_raw_16state_logits":True,
    }


def audit(root: Path,arms) -> dict:
    src=read_json(root/"pef07"/"source_split.json")
    verify_registered_partition(src)
    if src["sha256"]!=SOURCE_SHA:
        raise AssertionError("source changed")
    evidence=[check_one(root,seed,task,src) for seed,task in arms]
    result={
        "schema":"NEC-03E/independent-physical-dose-and-outcome-audit/v1",
        "evidence_class":"posthoc-development-only",
        "source_manifest_sha256":SOURCE_SHA,
        "pairs":len(PAIRS),
        "results":evidence,
        "total_candidate_pair_context_records":sum(x["candidate_pair_context_records"] for x in evidence),
        "feasible_comparisons":sum(x["feasible_comparisons"] for x in evidence),
        "independent_context_groups_per_model":48,
        "familywise_significance_inferred":False,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (root/"nec03e_audit.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(103,131,167),default=103)
    p.add_argument("--tasks",nargs="+",choices=("coalition","threshold"),
                   default=["coalition","threshold"])
    a=p.parse_args(argv)
    result=audit(a.root,[(a.seed,t) for t in a.tasks])
    print(json.dumps({"arms":len(result["results"]),
                      "candidate_pair_context_records":result["total_candidate_pair_context_records"],
                      "feasible":result["feasible_comparisons"]},sort_keys=True))


if __name__=="__main__":
    main()
