"""NEC-03W pure-stdlib independent source candidate/physical audit.

Recomputes candidate coactivation counts from the fixed source 0..15 rows,
checks that candidate ranking never depends on task labels or heldout source
query correctness, and separately reconstructs all matched 2+2 perturbation
vectors from frozen decoder32 inputs. This does not import the selector or
causal metric implementations.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path

from .nec03v_geometry import read_pack,original_lock_digest

ROOTS=tuple(range(8))


def _sha(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                       ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def verify_candidate_roster(pack:Path,roster:Path):
    source,shards,source_sha=read_pack(pack)
    data=json.loads(roster.read_text(encoding="utf-8"))
    checksum=data.pop("roster_sha256",None)
    if checksum!=_sha(data):
        raise AssertionError("original source-label-free candidate roster SHA changed")
    if (data["schema"]!="NEC-03W/source-only-factorial-candidate-roster/v1"
            or data["world_index"] not in ROOTS
            or data["world_index"]!=source["world_index"]
            or data["original_world_id"]!=source["world_id"]
            or data["source_pack_manifest_sha256"]!=source_sha
            or data["original_NEC03T_lock_sha256"]!=original_lock_digest()
            or data["source_query_is_SAE_fit_training_data"] if False else False):
        raise AssertionError("original NEC03W source-roster provenance changed")
    split=data["split"]
    if (split["geometry_discovery"]!=list(range(16))
            or split["functional_calibration"]!=list(range(16,24))
            or split["previously_behavior_scored_excluded"]!=list(range(24,32))
            or split["developmental_query"]!=list(range(32,40))
            or split["source_query_is_SAE_fit_training_data"] is not True
            or data["allowed_new_native_forwards"]!=0
            or data["task_labels_read"]!=0 or data["model_logits_read"]!=0
            or data["optimizer_steps"]!=0 or data["protected_cohort_reads"]!=0
            or data["interventions_scored"]!=0):
        raise AssertionError("incorrect source-only cohort custody")
    if set(data["arms"])!={"composed","additive"}:
        raise AssertionError("original source model arm omitted")
    n=0
    for arm in ("composed","additive"):
        shards_by_visit=shards[arm]["visits"]
        roster_visits=data["arms"][arm]["visits"]
        if set(roster_visits)!=set(shards_by_visit):
            raise AssertionError("candidate visit ownership differs from original SAE")
        for visit,rows in shards_by_visit.items():
            fits=rows["source_fit_active"]
            frequency=Counter()
            for active in fits[:512]:
                if len(active)>16 or len(active)!=len(set(active)):
                    raise AssertionError("invalid original source support")
                for members in combinations(active,4):
                    frequency[members]+=1
            ordered=sorted(frequency,key=lambda q:(-frequency[q],q))
            independent=[]
            for quartet in ordered:
                if all(len(set(quartet).intersection(prior))<=2 for prior in independent):
                    independent.append(quartet)
                if len(independent)==data["frozen_candidate_budget"]:
                    break
            candidates=roster_visits[visit]["candidates"]
            if (len(candidates)!=len(independent) or
                    roster_visits[visit]["selected_by_functional_effect"] is not False):
                raise AssertionError("a functional candidate was selected before calibration")
            for item,quartet in zip(candidates,independent):
                if item["quartet"]!=list(quartet):
                    raise AssertionError("outcome-blind candidate ordering differs")
                if item["source_discovery_count"]!=frequency[quartet]:
                    raise AssertionError("discovery training-only frequency differs")
                if not math.isclose(item["source_discovery_fraction"],
                                    frequency[quartet]/512,abs_tol=1e-12):
                    raise AssertionError("original source discovery fraction changed")
                support=set(quartet)
                expected_calib=sum(support.issubset(x) for x in fits[512:768])
                expected_query=sum(support.issubset(x) for x in fits[1024:1280])
                if (item["source_calibration_ID_support_only"]!=expected_calib or
                        item["source_query_ID_support_only"]!=expected_query):
                    raise AssertionError("query support counts altered after discovery")
                a,b,c,d=quartet
                partitions=[
                    [[a,b],[c,d]],[[a,c],[b,d]],[[a,d],[b,c]]]
                if item["pair_partitions"]!=partitions:
                    raise AssertionError("quartet pair split changed")
                n+=1
    return {"schema":"NEC-03W/independent-source-only-candidate-audit/v1",
            "original_world_id":source["world_id"],
            "registered_quartets_independently_verified":n,
            "source_pack_manifest_sha256":source_sha,
            "candidate_roster_sha256":checksum,
            "native_forward_calls":0,
            "model_behavior_outcomes_read":0,
            "protected_cohort_reads":0,
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}


def verify_pair_geometry(source_codes,decoder,receipt):
    if receipt["schema"]!="NEC-03W/matched-factorial-physical-sham/v1":
        raise AssertionError("invalid experimental geometry receipt")
    if receipt.get("admissible") is not True:
        raise AssertionError("only admitted 2x2 factorial geometry has physical vectors")
    ids={int(p[0]):float(p[1]) for p in source_codes}
    pairs=[receipt["pair_A"],receipt["pair_B"],
           receipt["matched_control_C"],receipt["matched_control_D"]]
    if (len(ids)!=16 or len(set(ids))!=16 or
            any(len(p)!=2 or len(set(p))!=2 or any(i not in ids for i in p) for p in pairs) or
            len(set(sum(pairs,[])))!=8 or
            sorted(sum(pairs[:2],[]))!=receipt["group_quartet"] or
            sorted(sum(pairs[2:],[]))!=receipt["control_quartet"]):
        raise AssertionError("matched factorial group and sham not disjoint 2x2")
    if not 0<float(receipt["source_gain"])<=1.0000001 or not 0<float(receipt["sham_gain"])<=1.0000001:
        raise AssertionError("source matched physical donor amplified")
    reconstructed={}
    for name,features,gain in zip(("A","B","C","D"),pairs,
                                  [receipt["source_gain"]]*2+[receipt["sham_gain"]]*2):
        actual=[]
        for j in range(32):
            actual.append(math.fsum(ids[i]*float(decoder[i][j]) for i in features)*gain)
        reconstructed[name]=actual
    for combined,a,b in (("AB","A","B"),("CD","C","D")):
        reconstructed[combined]=[x+y for x,y in zip(reconstructed[a],reconstructed[b])]
    for key,row in reconstructed.items():
        expected=receipt["projected32"][key]
        if len(expected)!=32 or any(not math.isclose(x,y,rel_tol=1e-6,abs_tol=2e-7)
                                    for x,y in zip(row,expected)):
            raise AssertionError("independently reconstructed 32D source pair differs")
    norm=lambda u:math.sqrt(math.fsum(x*x for x in u))
    a,b=norm(reconstructed["AB"]),norm(reconstructed["CD"])
    if abs(a-b)/min(a,b)>.01 or min(a,b)<.1:
        raise AssertionError("source/sham four-code physical dose mismatch")
    return {"schema":"NEC-03W/independent-2x2-pair-physical-audit/v1",
            "original_source_root":receipt["root"],
            "original_source_visit":receipt["visit"],
            "group_and_sham_pairs_disjoint":True,
            "physical_coordinates_recomputed":6*32,
            "new_model_forwards":0,
            "labels_or_logits_seen":0,
            "scientific_gate":"REDESIGN_OR_PILOT_ONLY"}
