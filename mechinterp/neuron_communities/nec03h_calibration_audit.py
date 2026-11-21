"""NEC-03H pure stdlib audit of original 450 and new-from-seed 900 roots.

Every old dev-000..007 root was already opened in NEC-03G. Root unit and
world specs must agree with the unchanged 128-world registry. The new
checkpoints and source-neuron selections are validated before comparison.
This module performs no model training or intervention itself.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from mechinterp.neuron_communities.nec03g_worlds import (
    context_catalog, digest, worlds, registry, encoded,
)

H_SCHEMA="NEC-03H/source-site-fallback-900steps-development-v2"


def sha_file(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def audit_calibration(folder: Path, original_folder: Path) -> dict:
    current=json.loads((folder/"calibration.json").read_text())
    old=json.loads((original_folder/"root_summary.json").read_text())
    index=current["index"]
    if index not in range(8):
        raise AssertionError("calibration attempted unopened world")
    world=worlds()[index]
    registered=registry()
    _,source=context_catalog(world)
    if (current["schema"]!=H_SCHEMA
        or current["world_id"]!=world["world_id"]
        or current["world_sha256"]!=world["world_sha256"]
        or current["registered_world_population_sha256"]!=registered["registry_sha256"]
        or current["world_source_sha256"]!=source["sha256"]):
        raise AssertionError("calibration world source/registry changed")
    if json.loads((folder/"world_spec.json").read_text())!=world:
        raise AssertionError("new world spec differs from generator")
    if json.loads((folder/"context_source_manifest.json").read_text())!=source:
        raise AssertionError("new world split differs from generator")
    if old["world_sha256"]!=world["world_sha256"] or old["source_sha256"]!=source["sha256"]:
        raise AssertionError("old 450 receipt belongs to a different world")
    if (old["trunk_steps"]!=450 or old["sae_steps"]!=480
        or not old["full_registered_development_recipe"]):
        raise AssertionError("old baseline not full NEC-03G")
    if current["training_steps"]!=900 or not current["full_calibration_recipe"]:
        raise AssertionError("calibration is not the frozen 900-step recipe")
    if (current["confirmation_worlds_trained_or_scored"]!=0
        or current["reserved_contexts_evaluated"]!=0
        or current["new_development_worlds_evaluated"]!=0
        or not current["source_candidate_fallback_frozen_before_dev_readout"]):
        raise AssertionError("confirmation, reserved or late source selection opened")
    arms={}
    for arm in ("composed","additive"):
        entry=current["arms"][arm]
        train=entry["trained"]
        if (train["world_sha256"]!=world["world_sha256"]
            or train["steps"]!=900 or train["teacher"]!=arm
            or not train["source_training_only"]
            or not train["source_native_weights_frozen_before_evaluation"]):
            raise AssertionError("new 900 native-model provenance failed")
        if sha_file(folder/train["checkpoint_name"])!=train["checkpoint_sha256"]:
            raise AssertionError("new native-model checkpoint hash changed")
        if (train["training_seed"]!=world["trunk_seed"]+
            (100000 if arm=="additive" else 0)):
            raise AssertionError("new checkpoint seed differs")
        freeze=folder/f"{arm}.selection.json"
        if sha_file(freeze)!=entry["source_selection_sha256"]:
            raise AssertionError("source-site selection freeze altered")
        source_selection=json.loads(freeze.read_text())
        if (source_selection["world_id"]!=world["world_id"]
            or source_selection["model_sha256"]!=train["final_model_sha256"]
            or source_selection["source_sha256"]!=source["sha256"]):
            raise AssertionError("selection bound to wrong checkpoint")
        info=source_selection["selection"]
        if (info["source_contexts"]!=80
            or info["development_or_reserved_contexts_exposed_for_selection"]
            or info["source_selection_uses_labels"]):
            raise AssertionError("selection exposed test labels or wrong source")
        cand=info["candidate"]
        if entry["candidate_found"]!=(cand is not None):
            raise AssertionError("candidate status differs from source freeze")
        if entry["old_graph_community_found"]!=info["graph_has_qualifying_community"]:
            raise AssertionError("source graph status differs")
        if info["fallback_used"] and cand:
            fallback=info["fallback"]
            if fallback["scheduled_visits"]!=world["focal_positions"][2:4]:
                raise AssertionError("fallback site schedule changed")
            if len(cand["nodes"])!=4 or len(set(cand["nodes"]))!=4:
                raise AssertionError("fallback site cardinality differs")
            if set(cand["nodes"])&set(info["graph"]["hubs"]):
                raise AssertionError("fallback has source hub overlaps")
            if sorted(int(x.split(":")[0]) for x in cand["nodes"]) != sorted(
                    [v for v in fallback["scheduled_visits"] for _ in range(2)]):
                raise AssertionError("fallback sites not source focal visits")
            for x in cand["nodes"]:
                if info["graph"]["frequency"].get(x,0)<2:
                    raise AssertionError("fallback source frequency insufficient")
        if info["graph_has_qualifying_community"] and info["fallback_used"]:
            raise AssertionError("fallback used despite qualified source graph")
        if entry["source_selection_kind"] not in ("graph","fallback","none"):
            raise AssertionError("unknown source-selection kind")
        if entry["source_selection_kind"]=="fallback" and (not info["fallback_used"] or cand is None):
            raise AssertionError("incorrect fallback label")
        old_score=old["trunk_model_clean_scores"][arm]
        new_score=entry["clean"]
        if (old_score["total"]!=768 or new_score["total"]!=768
            or old_score["correct"] not in range(769)
            or new_score["correct"] not in range(769)):
            raise AssertionError("model clean score outside fixed 768-case population")
        if abs(new_score["accuracy"]-new_score["correct"]/768)>1e-10:
            raise AssertionError("new score fraction wrong")
        arms[arm]={
            "original_450_correct":old_score["correct"],
            "new_from_seed_900_correct":new_score["correct"],
            "delta_correct":new_score["correct"]-old_score["correct"],
            "old_clean_competent":old_score["accuracy"]>=.9,
            "new_clean_competent":new_score["accuracy"]>=.9,
            "source_graph_found":info["graph_has_qualifying_community"],
            "native_site_status":entry["source_selection_kind"],
            "checkpoint_sha256":train["checkpoint_sha256"],
            "source_site_sha256":entry["source_selection_sha256"],
        }
    receipt={
        "schema":H_SCHEMA,
        "world_id":world["world_id"],
        "original_450_root_sha256":sha_file(original_folder/"root_summary.json"),
        "retrained_900_calibration_sha256":sha_file(folder/"calibration.json"),
        "registry_sha256":registered["registry_sha256"],
        "source_manifest_sha256":source["sha256"],
        "checkpoint_byte_hashes_verified":True,
        "world_used_for_v2_protocol_calibration":True,
        "full_900_steps_from_original_seed":True,
        "new_development_cohort_opened":False,
        "confirmation_worlds_scored":0,
        "arms":arms,
    }
    (folder/"calibration_audit.json").write_bytes(encoded(receipt))
    return receipt


def aggregate(root: Path,baseline_root: Path,out:Path):
    by_index={}
    for index in range(8):
        folder=root/f"nec03h-calibration-{index}"
        old=baseline_root/f"nec03g-development-root-{index}"
        record=audit_calibration(folder,old)
        by_index[index]=record
    rows=[]
    for index in range(8):
        row=by_index[index]
        result={"world_id":row["world_id"]}
        for arm in ("composed","additive"):
            result[arm]=row["arms"][arm]
        rows.append(result)
    out.mkdir(parents=True,exist_ok=True)
    report={
        "schema":H_SCHEMA,
        "phase":"locked-development-competence-calibration-comparison",
        "original_450_world_count":8,"new_from_seed_900_world_count":8,
        "world_ids":[r["world_id"] for r in rows],
        "registry_sha256":registry()["registry_sha256"],
        "development_calibration_only":True,
        "confirmation_worlds_trained_or_scored":0,
        "new_development_worlds_scored":0,
        "source_fallback_algorithm_recorded_before_fresh_development_cohort":True,
        "models_above_clean_90pct_450":sum(
            row[arm]["old_clean_competent"] for row in rows
            for arm in ("composed","additive")),
        "models_above_clean_90pct_900":sum(
            row[arm]["new_clean_competent"] for row in rows
            for arm in ("composed","additive")),
        "worlds_both_models_competent_450":sum(
            all(row[a]["old_clean_competent"] for a in ("composed","additive"))
            for row in rows),
        "worlds_both_models_competent_900":sum(
            all(row[a]["new_clean_competent"] for a in ("composed","additive"))
            for row in rows),
        "models_using_fallback_900":sum(
            row[arm]["native_site_status"]=="fallback"
            for row in rows for arm in ("composed","additive")),
        "models_without_any_source_site_900":sum(
            row[arm]["native_site_status"]=="none"
            for row in rows for arm in ("composed","additive")),
        "worlds":rows,
        "confirmation_hard_stop":"confirmation cohort requires fresh separate preregistration and positive independent EvalMath gate",
    }
    (out/"opened_world_calibration_450_vs_900.json").write_bytes(encoded(report))
    print(json.dumps({
        "worlds":len(rows),
        "competent_models_450":report["models_above_clean_90pct_450"],
        "competent_models_900":report["models_above_clean_90pct_900"],
        "fallback_models":report["models_using_fallback_900"],
        "no_source_models":report["models_without_any_source_site_900"],
        "confirmation":0},sort_keys=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    aggregate(a.root,a.original,a.out)


if __name__=="__main__":
    main()
