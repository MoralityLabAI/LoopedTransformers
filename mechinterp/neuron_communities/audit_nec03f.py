"""NEC-03F independent raw causal outcome, routing-policy and dose audit.

Uses stdlib arithmetic and frozen source files. The TopK policy can be
checked from the recorded active feature indices and the independently
stored source-only permutation without invoking the SAE implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from mechinterp.neuron_communities.audit_hosted import (
    SOURCE_SHA,raw_sha,read_json,verify_registered_partition,
)
from mechinterp.neuron_communities.audit_nec03 import expected_target,correct_count
from mechinterp.neuron_communities.audit_nec03e import check_site

PAIRS=(("native","sham"),("native","topk"),("topk","topk_alt"))


def check_policy(rows,rank_by_visit):
    """Check source-rank top/bottom active route for all 16 focal examples."""
    if not rows:
        raise AssertionError("source active feature trace missing")
    for visit,data in rows.items():
        v=int(visit)
        if v<0 or v>=12:
            raise AssertionError("TopK route contains impossible visit")
        ranking=rank_by_visit[v]
        if len(ranking)!=128 or set(ranking)!=set(range(128)):
            raise AssertionError("source feature ranking not permutation")
        order={f:k for k,f in enumerate(ranking)}
        if (len(data["top_source_ranked_active"])!=16 or
            len(data["bottom_source_ranked_active"])!=16 or
            len(data["all_active_features"])!=16):
            raise AssertionError("incomplete input-conditioned routing ledger")
        n=data["selected_features_per_arm_per_visit"]
        if not 1<=n<=8:
            raise AssertionError("wrong active-feature budget")
        for i in range(16):
            active=data["all_active_features"][i]
            if len(active)!=16 or len(set(active))!=16:
                raise AssertionError("TopK16 active feature contract violated")
            if any(j<0 or j>=128 for j in active):
                raise AssertionError("active feature outside TopK vocabulary")
            ranked=sorted(active,key=lambda j:order[j])
            first=ranked[:n]
            last=list(reversed(ranked[-n:]))
            if data["top_source_ranked_active"][i]!=first:
                raise AssertionError("TopK top active selection differs from source rank")
            if data["bottom_source_ranked_active"][i]!=last:
                raise AssertionError("TopK alternate policy differs from source rank")
            if set(first)&set(last):
                raise AssertionError("selected active-policy control shares features")


def check_one(root,seed,task,src):
    folder=root/"nec03f"/f"seed{seed}_{task}"
    prepared=(folder/"prepare.json").read_bytes()
    prepare_sha=hashlib.sha256(prepared).hexdigest()
    if prepare_sha!=(folder/"prepare.sha256").read_text().strip():
        raise AssertionError("TopK source training/frozen policy manifest mutated")
    prep=json.loads(prepared)
    r=read_json(folder/"report.json")
    if (prep["schema"]!="NEC-03F/source-TopK16-dynamic-active-feature-pairwise-dose/v1"
        or r["schema"]!="NEC-03F/source-TopK16-dynamic-active-feature-pairwise-dose/v1"):
        raise AssertionError("wrong source policy schema")
    if (prep["seed"],prep["task"])!=(seed,task) or (r["seed"],r["task"])!=(seed,task):
        raise AssertionError("source task or model differs")
    if prep["source_manifest_sha256"]!=r["source_manifest_sha256"]!=SOURCE_SHA:
        raise AssertionError("original source hash changed")
    if r["source_topk_prepare_sha256"]!=prepare_sha:
        raise AssertionError("heldout evaluated a different feature policy")
    if prep["train_only_contexts"]!={"fit":80,"rank":80,"heldout":0}:
        raise AssertionError("TopK fit exposed heldout contexts")
    contexts=src["contexts_in_sha256_order"]
    if prep["train_context_ids"]!=contexts[:80] or prep["rank_context_ids"]!=contexts[80:160]:
        raise AssertionError("source-only train/development context membership wrong")
    if prep["topk_training"]["target_active_features_per_input"]!=16:
        raise AssertionError("SAE feature L0 constraint changed")
    if prep["topk_training"]["heldout_contexts_seen"]!=0:
        raise AssertionError("TopK trained on heldout")
    if raw_sha(folder/prep["topk_checkpoint_file"])!=prep["topk_checkpoint_sha256"]:
        raise AssertionError("TopK source checkpoint bytes changed")
    if r["topk_checkpoint_sha256"]!=prep["topk_checkpoint_sha256"]:
        raise AssertionError("TopK checkpoint changed after freeze")
    native=root/"analysis"/f"seed{seed}_{task}"/"discovery.json"
    d=read_json(native)
    if (r["original_native_discovery_sha256"]!=raw_sha(native)
        or prep["original_discovery_sha256"]!=raw_sha(native)):
        raise AssertionError("native selection roster differs")
    if d["identity"]!=prep["frozen_model_identity"]!=r["frozen_model_identity"]:
        raise AssertionError("model identity differs")
    frozen={c["id"]:c for c in d["candidates"]}
    if [x["id"] for x in prep["native_community_source_candidates"]]!=list(frozen):
        raise AssertionError("candidate roster order changed")
    if r["candidate_count"]!=len(frozen) or r["comparison_count"]!=len(frozen)*3:
        raise AssertionError("candidate comparison family changed")
    if r["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY" or not r["no_claim_of_out_of_family_generalization"]:
        raise AssertionError("development evidence class violated")
    raw=folder/r["raw_file"]
    if raw_sha(raw)!=r["raw_sha256"]:
        raise AssertionError("raw causal results changed")
    lines=raw.read_text().splitlines()
    if len(lines)!=r["raw_records"] or len(lines)!=48*len(frozen)*3:
        raise AssertionError("missing candidate/pair/source context cases")
    counts=defaultdict(lambda:{"clean":0,"hubs":0,"left":0,"right":0})
    doses=defaultdict(lambda:{"n":0,"matched":0,"active":0,"ratios":[]})
    seen=set()
    for line in lines:
        data=json.loads(line)
        pair=tuple(data["pair"])
        cid=data["candidate_id"]
        ctx="".join(map(str,data["context_bits"]))
        key=(cid,pair)
        if cid not in frozen or pair not in PAIRS:
            raise AssertionError("unknown candidate/source pair")
        if src["context_assignment"].get(ctx)!="heldout" or (key,ctx) in seen:
            raise AssertionError("heldout source duplicated or contaminated")
        seen.add((key,ctx))
        if data["mask_order"]!=list(range(16)) or (data["seed"],data["task"])!=(seed,task):
            raise AssertionError("wrong source outcome quartet")
        labels=[expected_target(i,task) for i in range(16)]
        if data["labels"]!=labels:
            raise AssertionError("wrong source Boolean teacher")
        for arm,field in (("clean","clean_logits"),("hubs","hubs_logits"),
                          ("left","left_logits"),("right","right_logits")):
            counts[key][arm]+=16-correct_count(data[field],labels)
        if set(data["site_doses"])!=set(data["site_active_policy"]):
            raise AssertionError("dose/policy visit families disagree")
        check_policy(data["site_active_policy"],prep["source_feature_rank_per_visit"])
        for site in data["site_doses"].values():
            x=check_site(site)
            d0=doses[key]
            d0["n"]+=x["n"]
            d0["matched"]+=x["matched"]
            d0["active"]+=x["active"]
            d0["ratios"].extend(x["ratios"])
    heldout={ctx for ctx,owner in src["context_assignment"].items() if owner=="heldout"}
    for cid in frozen:
        for p in PAIRS:
            if {ctx for (k,ctx) in seen if k==(cid,p)}!=heldout:
                raise AssertionError("missing independent heldout context per pair")
    for row in r["results"]:
        key=(row["candidate_id"],tuple(row["pair"]))
        expected=counts[key]
        if [row["clean_errors_of_768"],row["hubs_errors_of_768"],
            row["left_errors_of_768"],row["right_errors_of_768"]] != [
                expected["clean"],expected["hubs"],expected["left"],expected["right"]]:
            raise AssertionError("reported raw class counts mismatch")
        if row["left_minus_right_correct_count"]!=expected["right"]-expected["left"]:
            raise AssertionError("pairwise correct count wrong")
        raw_dose=doses[key]
        report=row["physical_dose"]
        if not raw_dose["n"]:
            raise AssertionError("missing dose metadata")
        activity=raw_dose["active"]/raw_dose["n"]
        matched=raw_dose["matched"]/raw_dose["n"]
        ratios=sorted(raw_dose["ratios"])
        median=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
        if (raw_dose["n"]!=report["site_sample_count"] or
            abs(activity-report["fraction_with_positive_shared_dose"])>1e-7 or
            abs(matched-report["fraction_matched_within_1pct"])>1e-7 or
            abs(median-report["median_fraction_full_left_dose_preserved"])>1e-6):
            raise AssertionError("result dose feasibility differs from raw samples")
        viable=(matched>=0.99 and activity>=0.80 and median>=0.25)
        if viable!=report["candidate_physically_comparable_and_signal_sufficient"]:
            raise AssertionError("signal gate erroneously reported")
    feasible=sum(x["physical_dose"]["candidate_physically_comparable_and_signal_sufficient"]
                 for x in r["results"])
    if r["numerically_dose_and_signal_feasible_pairs"]!=feasible:
        raise AssertionError("physical feasibility count tampered")
    return {
        "seed":seed,"task":task,
        "source_prepare_sha256":prepare_sha,
        "topk_checkpoint_sha256":prep["topk_checkpoint_sha256"],
        "native_roster_sha256":r["original_native_discovery_sha256"],
        "raw_paired_logits_sha256":r["raw_sha256"],
        "candidate_count":len(frozen),
        "candidate_pair_context_records":len(lines),
        "matched_and_sufficient":feasible,
        "source_policy_recomputed_from_active_code_indices":True,
        "paired_teacher_labels_and_correctness_recomputed":True,
        "all_local_doses_and_gains_verified":True,
    }


def audit(root,arms):
    src=read_json(root/"pef07"/"source_split.json")
    verify_registered_partition(src)
    assert src["sha256"]==SOURCE_SHA
    results=[check_one(root,s,t,src) for s,t in arms]
    report={
        "schema":"NEC-03F/independent-source-topk-route-dose-audit/v1",
        "status":"development-only",
        "arms":results,
        "source_sha256":SOURCE_SHA,
        "total_candidate_pair_context_records":sum(x["candidate_pair_context_records"] for x in results),
        "total_signal_feasible_pairs":sum(x["matched_and_sufficient"] for x in results),
        "no_prospective_significance_inferred":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
    }
    (root/"nec03f_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=103,choices=(103,131,167))
    ap.add_argument("--tasks",nargs="+",default=["coalition","threshold"],
                    choices=("coalition","threshold"))
    a=ap.parse_args(argv)
    res=audit(a.root,[(a.seed,t) for t in a.tasks])
    print(json.dumps({"arms":len(res["arms"]),
                      "rows":res["total_candidate_pair_context_records"],
                      "feasible_pairs":res["total_signal_feasible_pairs"]},sort_keys=True))


if __name__=="__main__":
    main()
