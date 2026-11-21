"""Independent stdlib NEC-03D audit of frozen source, logits, and physical doses.

This deliberately does not import the SAE, donor builder or result-summary
functions. It checks the exposed source partition, the raw outcome transcript
and exact paired correctness counts, keeping all dose mismatches.
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
from research.pef07.pef07 import source_split

EXPECTED_TASKS=("coalition","threshold")
MODES=("hubs","native","sham","sae","sae_sham")


def expected_target(mask: int, task: str) -> int:
    a,b,c,d=((mask>>i)&1 for i in range(4))
    return (a*b+c*d)%2 if task=="coalition" else int(a+b+c+d>=2)


def correct_count(logits, labels) -> int:
    if len(logits)!=len(labels) or len(logits)!=16:
        raise AssertionError("incomplete 16-state source context")
    count=0
    for pair,label in zip(logits,labels):
        if len(pair)!=2 or not all(math.isfinite(float(x)) for x in pair):
            raise AssertionError("nonfinite paired logits")
        if int(label) not in (0,1):
            raise AssertionError("invalid binary label")
        count+=int(int(pair[1]>pair[0])==label)
    return count


def verify_doses(raw_row, *, max_gain: float, tolerance: float=0.05) -> dict:
    sites=raw_row["local_physical_doses"]
    if not sites:
        raise AssertionError("candidate has no recurrent intervention visits")
    totals={name:{"samples":0,"matched":0,"saturated":0,"relative_error_sum":0.0}
            for name in ("sham","sae","sae_sham")}
    for visit, parts in sites.items():
        visit=int(visit)
        if not 0<=visit<12:
            raise AssertionError("visit outside frozen recurrent horizon")
        target=parts["native_direct_dose"]
        if len(target)!=16:
            raise AssertionError("native physical dose has wrong sample count")
        for name in totals:
            row=parts[name]
            if (len(row["target"])!=16 or len(row["realized"])!=16
                or len(row["relative_error"])!=16
                or len(row["matched_under_5pct"])!=16 or
                len(row["gain"])!=16 or len(row["saturated"])!=16):
                raise AssertionError("dose receipts have missing samples")
            for i in range(16):
                t=float(target[i])
                desired=float(row["target"][i])
                actual=float(row["realized"][i])
                g=float(row["gain"][i])
                delta=abs(actual-t)/max(t,1e-6)
                if not all(math.isfinite(x) for x in (t,desired,actual,g,delta)):
                    raise AssertionError("nonfinite physical intervention dose")
                if t<0 or actual<0 or g<-1e-8 or g>max_gain+1e-6:
                    raise AssertionError("physical dose or gain outside allowed range")
                if abs(t-desired)>1e-5 or abs(delta-row["relative_error"][i])>1e-4:
                    raise AssertionError("dose-matching sample mismatch")
                if bool(row["matched_under_5pct"][i]) != (delta<=tolerance+1e-6):
                    raise AssertionError("dose match threshold incorrectly reported")
                if bool(row["saturated"][i]) != (g>=max_gain-1e-6):
                    raise AssertionError("gain clipping incorrectly reported")
                totals[name]["samples"]+=1
                totals[name]["matched"]+=int(delta<=tolerance+1e-6)
                totals[name]["saturated"]+=int(g>=max_gain-1e-6)
                totals[name]["relative_error_sum"]+=delta
    return totals


def check_one(root: Path, seed: int, task: str, source: dict):
    folder=root/"nec03d"/f"seed{seed}_{task}"
    original=root/"analysis"/f"seed{seed}_{task}"
    frozen=(folder/"prepare.json").read_bytes()
    digest=hashlib.sha256(frozen).hexdigest()
    if digest!=(folder/"prepare.sha256").read_text().strip():
        raise AssertionError("source-only SAE manifest changed")
    prepared=json.loads(frozen)
    result=read_json(folder/"matched_graft_report.json")
    old=read_json(original/"discovery.json")
    if (prepared["seed"],prepared["task"])!=(seed,task):
        raise AssertionError("prepared artifact from wrong task/seed")
    if prepared["status"]!="source-only-sae-and-features-frozen":
        raise AssertionError("SAE/source feature roster not sealed")
    if (prepared["source_sha256"],result["source_sha256"])!=(SOURCE_SHA,SOURCE_SHA):
        raise AssertionError("changed source distribution")
    if result["prepare_sha256"]!=digest:
        raise AssertionError("evaluation used different source-only SAE")
    if (prepared["original_candidate_roster_sha256"]!=
            result["original_candidate_roster_sha256"] or
        prepared["original_candidate_roster_sha256"]!=raw_sha(original/"discovery.json")):
        raise AssertionError("candidate source changed")
    if result["model_identity"]!=prepared["model_identity"] or result["model_identity"]!=old["identity"]:
        raise AssertionError("model identity differs")
    sae=prepared["sae"]
    if raw_sha(folder/sae["sae_checkpoint_file"])!=sae["sae_checkpoint_sha256"]:
        raise AssertionError("SAE bytes changed after freeze")
    src=prepared["raw_source_contexts"]
    expected=source["contexts_in_sha256_order"]
    if src["train_context_ids"]!=expected[:80] or src["train_dev_context_ids"]!=expected[80:160]:
        raise AssertionError("feature selection/fit cohort differs from train-only source")
    if src["heldout_context_ids_used"]!=[]:
        raise AssertionError("SAE fit saw heldout contexts")
    if prepared["source_contexts"]!={"fit":80,"selection":80,"heldout":0}:
        raise AssertionError("different source-only allocation")
    if len(prepared["sae_rankings"])!=len(old["candidates"]):
        raise AssertionError("candidate-count changed")
    if [x["candidate_id"] for x in prepared["sae_rankings"]] != [
            x["id"] for x in old["candidates"]]:
        raise AssertionError("source ranked different native candidates")
    if result["independent_context_groups"]!=48 or result["correlated_focal_cases_per_context"]!=16:
        raise AssertionError("incorrect root-level outcome cohort")
    if result["familywise_significance_claim"] or result["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY":
        raise AssertionError("invalid inference evidence class")
    if set(result["candidates"][0]["errors_of_768_by_mode"])!=set(MODES) if result["candidates"] else False:
        raise AssertionError("missing named intervention")
    raw_path=folder/result["raw_filename"]
    if raw_sha(raw_path)!=result["raw_sha256"]:
        raise AssertionError("raw paired logits are not immutable")
    lines=raw_path.read_text().splitlines()
    if len(lines)!=result["raw_records"] or len(lines)!=48*len(old["candidates"]):
        raise AssertionError("incomplete source-context/candidate grid")
    seen=set()
    heldout=set(expected[208:])
    counts={x["candidate_id"]:{mode:0 for mode in MODES}
            for x in result["candidates"]}
    clean_counts={x["candidate_id"]:0 for x in result["candidates"]}
    dose_total={x["candidate_id"]:{
        k:{"samples":0,"matched":0,"saturated":0,"relative_error_sum":0.0}
        for k in ("sham","sae","sae_sham")}
        for x in result["candidates"]}
    for line in lines:
        row=json.loads(line)
        cid=row["candidate_id"]
        ctx="".join(map(str,row["context_bits"]))
        if cid not in counts or ctx not in heldout or (cid,ctx) in seen:
            raise AssertionError("invalid or repeated candidate/context")
        seen.add((cid,ctx))
        if (row["seed"],row["task"])!=(seed,task):
            raise AssertionError("raw logit task/seed mismatch")
        if row["counterfactual_mask_order"]!=list(range(16)):
            raise AssertionError("wrong focal quartet order")
        expected_labels=[expected_target(k,task) for k in range(16)]
        if row["labels"]!=expected_labels:
            raise AssertionError("source teacher labels not preserved")
        clean_counts[cid]+=16-correct_count(row["clean_logits"],expected_labels)
        if set(row["intervention_logits"])!=set(MODES) or set(row["metrics"])!=set(MODES):
            raise AssertionError("missing causal intervention condition")
        for mode in MODES:
            logits=row["intervention_logits"][mode]
            hits=correct_count(logits,expected_labels)
            counts[cid][mode]+=16-hits
            original_hits=correct_count(row["clean_logits"],expected_labels)
            measured=float(row["metrics"][mode]["correctness_drop"])
            if abs(measured-(hits*-1+original_hits)/16)>1e-8:
                raise AssertionError("raw case accuracy effect mismatch")
        doses=verify_doses(row,max_gain=prepared["physical_dose_gain_cap"])
        for name in ("sham","sae","sae_sham"):
            for key in ("samples","matched","saturated","relative_error_sum"):
                dose_total[cid][name][key]+=doses[name][key]
    if any({ctx for cand,ctx in seen if cand==cid}!=heldout for cid in counts):
        raise AssertionError("missing heldout context on a selected community")
    valid=0
    for item in result["candidates"]:
        cid=item["candidate_id"]
        if item["errors_of_768_by_mode"]!=counts[cid]:
            raise AssertionError("reported outcome counts differ from raw predictions")
        if item["clean_errors_of_768"]!=clean_counts[cid]:
            raise AssertionError("clean score changed or reported incorrectly")
        for mode in ("native","sham","sae","sae_sham"):
            if item["conditional_rescues_vs_hubs_errors_of_768"][mode]!=(
                  counts[cid]["hubs"]-counts[cid][mode]):
                raise AssertionError("rescue count mismatches raw outcomes")
        for mode in ("sham","sae","sae_sham"):
            d=dose_total[cid][mode]
            report=item["physical_dose_audit"][mode]
            if d["samples"] != report["site_sample_count"]:
                raise AssertionError("dose cohort count changed")
            pct=d["matched"]/d["samples"] if d["samples"] else 0
            if abs(pct-report["fraction_matched_within_5pct"])>1e-8:
                raise AssertionError("matched dose fraction altered")
            if bool(report["dose_failure_if_under_90pct_matched"])!=(pct<0.90):
                raise AssertionError("physical dose gate misreported")
        valid+=int(item["all_dose_controls_meet_90pct_5pct_tolerance"])
    return {"task":task,"seed":seed,
            "prepared_sha256":digest,"sae_checkpoint_sha256":sae["sae_checkpoint_sha256"],
            "raw_logits_sha256":result["raw_sha256"],"candidate_count":len(counts),
            "candidate_context_records":len(lines),
            "all_controls_matched_candidates":valid,
            "dose_failed_candidates":len(counts)-valid,
            "model_clean_errors":clean_counts,
            "outcome_errors_by_candidate":counts,
            "heldout_contexts_checked":48,
            "source_training_contexts_checked":160,
            "all_cases_and_local_dose_receipts_recomputed":True}


def audit(root: Path, arms) -> dict:
    source=read_json(root/"pef07/source_split.json")
    verify_registered_partition(source)
    assert source["sha256"]==SOURCE_SHA
    rows=[check_one(root,seed,task,source) for seed,task in arms]
    report={
        "schema":"NEC-03D/independent-stdlib-audit/v1",
        "evidence_class":"posthoc-development-only",
        "source":SOURCE_SHA,
        "arms":rows,
        "total_communities":sum(x["candidate_count"] for x in rows),
        "total_candidate_context_rows":sum(x["candidate_context_records"] for x in rows),
        "all_raw_cases_and_local_doses_verified":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY"}
    (root/"nec03d_audit.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(103,131,167),default=103)
    p.add_argument("--tasks",nargs="+",choices=EXPECTED_TASKS,default=list(EXPECTED_TASKS))
    args=p.parse_args(argv)
    report=audit(args.root,[(args.seed,t) for t in args.tasks])
    print(json.dumps({"arms":len(report["arms"]),
                      "candidate_count":report["total_communities"],
                      "candidate_context_records":report["total_candidate_context_rows"],
                      "dose_failed_candidates":sum(x["dose_failed_candidates"] for x in report["arms"])},
                      sort_keys=True))


if __name__=="__main__":
    main()
