"""NEC-03I-B: independent stdlib audit of four disjoint active-feature shams.

Rechecks frozen original world/checkpoint/SAE source identities, 48-context
16-state labels and correct predictions, deterministic feature routing,
local matching of direct physical dose and root-level effects.
No optimizer, neural-model import, confirmation or reserved context access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from mechinterp.neuron_communities.nec03g_worlds import (
    context_catalog,registry,worlds,
)
from mechinterp.neuron_communities.nec03i_trace_audit import (
    clean_hits,file_sha,verify_physical_dose,
)

SCHEMA="NEC-03I-B/4-hash-active-rank-shams/v1"
SALT=b"NEC03IB-hash-active-shams-v1"
PANEL_VARIANTS=range(4)


def verify_route(active: dict, rankings: list, seed: int, variant: int):
    for name,record in active.items():
        visit=int(name)
        rank=rankings[visit]
        if len(rank)!=128 or set(rank)!=set(range(128)):
            raise AssertionError("source feature rank is not full permutation")
        source_order={f:i for i,f in enumerate(rank)}
        hash_rank=sorted(range(128),key=lambda feature: sha256(
            SALT+seed.to_bytes(8,"big")+
            bytes((visit,variant,feature))).digest())
        hash_order={f:i for i,f in enumerate(hash_rank)}
        top=record["source_top_active"]
        alternative=record["hash_selected_disjoint_active"]
        all_active=record["all_active_feature_indices"]
        n=record["features_per_site"]
        if not (len(top)==len(alternative)==len(all_active)):
            raise AssertionError("wrong route shape")
        if len(top)!=16 or not 1<=n<=8:
            raise AssertionError("expected 16 focal histories and capped source sites")
        for index in range(16):
            active_codes=all_active[index]
            if len(active_codes)!=16 or len(set(active_codes))!=16:
                raise AssertionError("TopK active support incomplete")
            if top[index]!=sorted(active_codes,key=lambda f:source_order[f])[:n]:
                raise AssertionError("source top-ranked active codes changed")
            pool=[f for f in active_codes if f not in set(top[index])]
            expected=sorted(pool,key=lambda f:hash_order[f])[:n]
            if alternative[index]!=expected:
                raise AssertionError("hash control selected using changed rank or output")
            if set(top[index]) & set(alternative[index]):
                raise AssertionError("hash control is not disjoint from source-selected TopK")


def audit_one(root: Path,panel: Path,lock_path: Path) -> dict:
    w=json.loads((root/"world_spec.json").read_text())
    index=w["world_index"]
    if index not in range(8,16) or w["cohort"]!="development" or w!=worlds()[index]:
        raise AssertionError("wrong original NEC-03H development population")
    source=json.loads((root/"context_source_manifest.json").read_text())
    _,catalog=context_catalog(w)
    if source!=catalog:
        raise AssertionError("source assignment changed")
    meta=json.loads((root/"root_summary.json").read_text())
    original=json.loads((root/"composed.source_freeze.json").read_text())
    report=json.loads((panel/"source_blind_control_panel.json").read_text())
    sham_lock=Path(__file__).with_name("NEC03IB_SHAM_LOCK.json")
    lock_cfg=json.loads(sham_lock.read_text())
    if (lock_cfg["schema"]!="NEC-03I-B/active-hash-sham-lock/v1"
        or report.get("control_panel_lock_sha256")!=file_sha(sham_lock)
        or lock_cfg["original_run_id"]!=44878411694
        or lock_cfg["variants"]!=list(PANEL_VARIANTS)
        or lock_cfg["world_indices"]!=list(range(8,16))):
        raise AssertionError("new hash sham population was not frozen before evaluation")
    if (report["schema"]!=SCHEMA or report["world_id"]!=w["world_id"]
        or report["world_sha256"]!=w["world_sha256"]
        or report["world_index"]!=index
        or report["v2_lock_sha256"]!=file_sha(lock_path)
        or report["old_source_report_sha256"]!=file_sha(root/"composed.report.json")
        or report["source_freeze_sha256"]!=file_sha(root/"composed.source_freeze.json")
        or report["original_ITT_signed_effect"]!=meta["primary_effect"]["root_primary_effect"]
        or report["original_primary_eligible"]!=meta["primary_effect"]["planned_world_valid"]
        or report["original_source_active_top_vs_bottom_replayed_exact_contexts"]!=48
        or report["reserved_contexts_scored"]!=0
        or report["confirmation_worlds_trained_or_scored"]!=0):
        raise AssertionError("source identity, old baseline or cohort were altered")
    if (report["native_model_sha256"]!=original["model_sha256"]
        or report["source_sae_checkpoint_sha256"]!=original["source_sae"]["source_sae_sha256"]
        or not report["original_model_and_source_ranks_immutable"]
        or not report["controls_are_active_and_disjoint_from_original_top"]):
        raise AssertionError("frozen source SAE/native source changed")
    if not report["posthoc_exploratory_only"] or report["FSI_01_gate"]!="REDESIGN_OR_PILOT_ONLY":
        raise AssertionError("unsupported evidence class")
    if len(report["comparisons"])!=4 or report["hash_scramble_variants"]!=list(PANEL_VARIANTS):
        raise AssertionError("unexpected null hypothesis family")
    truth=w["truth_table_16"]
    labels=[(truth>>i)&1 for i in range(16)]
    seed=w["trunk_seed"]
    outputs=[]
    for variant in PANEL_VARIANTS:
        item=report["comparisons"][variant]
        if item["variant"]!=variant or not item["original_source_active_rank_excluded_from_control"]:
            raise AssertionError("control arm changed")
        rank_sha=sha256(SALT+seed.to_bytes(8,"big")+bytes([variant])).hexdigest()
        if item["label_blind_hash_rank_sha256"]!=rank_sha:
            raise AssertionError("fixed control hash salt changed")
        path=panel/item["raw_filename"]
        if file_sha(path)!=item["raw_context_sha256"]:
            raise AssertionError("raw control panel file tampered")
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        if len(rows)!=48 or item["context_rows"]!=48:
            raise AssertionError("wrong root-level context count")
        seen=set()
        error_count=Counter()
        matched=positive=total=0
        retained=[]
        for row in rows:
            ctx=row["context_index"]
            if ctx not in range(48) or ctx in seen:
                raise AssertionError("unexpected/repeated source context")
            seen.add(ctx)
            if (row["world_id"]!=w["world_id"] or row["variant"]!=variant
                or row["source_context_bits"] != [
                    int(v) for v in source["contexts_ordered"][160+ctx]]
                or row["focal_mask_order"]!=list(range(16))
                or row["labels"]!=labels):
                raise AssertionError("original source truth mask changed")
            for field,store in (("clean_logits","clean"),
                                ("hubs_logits","hubs"),
                                ("source_logits","source"),
                                ("control_logits","sham")):
                correct=clean_hits(row[field],labels)
                error_count[store]+=16-sum(correct)
            doses=row["local_direct_doses"]
            routes=row["active_feature_indices"]
            if set(doses)!=set(routes):
                raise AssertionError("different physical and ranked source sites")
            verify_route(routes,original["source_ranking_per_visit"],seed,variant)
            for receipt in doses.values():
                for direct,ratio,is_active in verify_physical_dose(receipt):
                    total+=1
                    positive+=int(is_active)
                    retained.append(ratio)
                matched+=sum(receipt["matched_within_1pct"])
        if seen!=set(range(48)):
            raise AssertionError("incomplete source contexts")
        if error_count!={k:item[v] for k,v in (
            ("clean","pairwise_clean_errors"),
            ("hubs","hubs_only_errors"),
            ("source","source_ranked_errors"),
            ("sham","hash_control_errors"))}:
            raise AssertionError("reported donor correctness changed")
        if item["net_source_ranked_correct_gain"]!=(
                error_count["sham"]-error_count["source"]):
            raise AssertionError("signed null-control difference corrupted")
        dose=item["dose"]
        ratios=sorted(retained)
        med=(ratios[(len(ratios)-1)//2]+ratios[len(ratios)//2])/2
        if (dose["site_sample_count"]!=total
            or abs(dose["fraction_with_positive_shared_dose"]-positive/total)>1e-6
            or abs(dose["fraction_matched_within_1pct"]-matched/total)>1e-6
            or abs(dose["median_fraction_full_left_dose_preserved"]-med)>1e-6):
            raise AssertionError("matched physical signal summary corrupted")
        eligible=(matched/total>=.99 and positive/total>=.8 and med>=.25)
        if eligible!=dose["candidate_physically_comparable_and_signal_sufficient"]:
            raise AssertionError("null physical signal gate mislabeled")
        outputs.append({"variant":variant,
                        "correct_gain":item["net_source_ranked_correct_gain"],
                        "physically_eligible":eligible,
                        "median_retained":med,
                        "positive_dose_fraction":positive/total,
                        "source_errors":error_count["source"],
                        "hash_control_errors":error_count["sham"],
                        "raw_sha256":item["raw_context_sha256"]})
    return {
        "world_id":w["world_id"],
        "world_index":index,
        "original_v2_primary_effect":meta["primary_effect"]["root_primary_effect"],
        "original_v2_primary_eligible":meta["primary_effect"]["planned_world_valid"],
        "source_site_kind":meta["source_site_kinds"]["composed"],
        "original_source_rank_and_frozen_checkpoint_reused":True,
        "total_new_paired_context_rows":4*48,
        "confirmation_worlds_scored":0,
        "controls":outputs,
    }


def aggregate(roots:Path,panels:Path,lock:Path,out:Path):
    worlds_data=[]
    for index in range(8,16):
        base=roots/f"nec03h-v2-development-root-{index}"
        panel=panels/f"nec03ib-v2-root-{index}"
        worlds_data.append(audit_one(base,panel,lock))
    if sum(x["total_new_paired_context_rows"] for x in worlds_data)!=1536:
        raise AssertionError("missing source-verified null panel")
    out.mkdir(parents=True,exist_ok=True)
    report={
        "schema":"NEC-03I-B/independent-eight-world-control-panel-audit/v1",
        "evidence_class":"read-only original models; posthoc new donor interventions",
        "worlds":worlds_data,
        "distinct_worlds":8,
        "rank_controls_per_world":4,
        "new_paired_context_records":8*4*48,
        "source_sae_checkpoint_changes":0,
        "native_model_retraining_steps":0,
        "confirmation_worlds_scored":0,
        "reserved_contexts_scored":0,
        "root_effect_statistics_are_descriptive":True,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "full_original_registry_sha256":registry()["registry_sha256"],
    }
    for variant in PANEL_VARIANTS:
        eligible=[w for w in worlds_data
                  if w["controls"][variant]["physically_eligible"]]
        report[f"variant_{variant}_feasible_worlds"]=len(eligible)
        report[f"variant_{variant}_signed_correct_gains"]=[
            w["controls"][variant]["correct_gain"] for w in worlds_data]
    (out/"nec03ib_control_panel_audit.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"worlds":8,"records":report["new_paired_context_records"],
                      "feasible_per_variant":[report[f"variant_{i}_feasible_worlds"] for i in PANEL_VARIANTS],
                      "confirmation_worlds_scored":0},sort_keys=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--roots",type=Path,required=True)
    p.add_argument("--panels",type=Path,required=True)
    p.add_argument("--lock",type=Path,
                   default=Path(__file__).with_name("NEC03H_V2_LOCK.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    aggregate(a.roots,a.panels,a.lock,a.out)


if __name__=="__main__":
    main()
