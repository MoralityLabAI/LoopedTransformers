"""LBF-11 preregistered multi-shortest exposure audit over native train roots.

Audits only immutable, independently replayed train-root packets and the
native shortest-action labels derived *after* frozen behavior model visitation.
No calibration, ordinary-test or OOD roots are read. Stop before model training
if source does not achieve the preregistered action-set intervention strength.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import sys

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))

from ppt_gym.data import canonical,sha256_file
from ppt_gym.native_train import _load_teacher
from ppt_gym.lbf08_repair import (
    _load_manifest, _load_visited, COLLECT_SCHEMA, frame_schedule,
)
from ppt_gym.live_suite import load_suite

SCHEMA="lbf11.native_loss_source_exposure.v1"
VISIT_MIN_FRACTION=.25
MULTI_ROOTS_MIN=12
MIN_MULTI_TRAIN_UPDATES=50


def gate(suite:Path,teacher:Path,collection:Path,
         behavior:Path,replay_receipt:Path,out:Path)->dict:
    if out.exists():
        raise ValueError("LBF-11 source exposure audit must write a new directory")
    manifest=json.loads((suite/"manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("structural_alt") is not True
        or manifest.get("equal_alt_p",0.0)!=0.0
        or manifest.get("source_family")!="lbf10_independent_equal_cost_diamonds.v1"
        or manifest.get("seed")!=20261203):
        raise ValueError("LBF-11 source suite is not the preregistered native equal-cost generator")
    source_rows,teacher_meta=_load_teacher(teacher)
    train,provenance=load_suite(suite,"train",48)
    if (len(train)!=48 or len(source_rows)!=teacher_meta["train_examples"]
        or teacher_meta["reference_proofs_compiled"]!=48
        or teacher_meta["backend"]["actual_lean_process"] is not True
        or teacher_meta["query_roots_exposed_to_fit"]!=0
        or teacher_meta["source"]["suite_manifest_sha256"]!=provenance["suite_manifest_sha256"]):
        raise ValueError("LBF-11 source teacher lacks 48 native kernel/standalone proofs")
    meta=_load_manifest(collection,COLLECT_SCHEMA)
    frames,replay_meta=_load_visited(collection,teacher,suite,behavior,replay_receipt)
    if (meta["native_backend"]["actual_lean_process"] is not True
        or meta["behavior_root_episodes"]!=48
        or meta["source_reference_compiler_passed"] is not True
        or meta["query_roots_exposed_to_collection"]!=0
        or meta["oracle_assisted_not_model_success"] is not True
        or meta["locked_test_opened"] is not False):
        raise ValueError("LBF-11 visited source lacks native replay provenance")
    if len(frames)!=meta["model_visitation_frames"]:
        raise ValueError("visited train frame count does not match frozen native manifest")

    for row in source_rows:
        if (row["schema"]!="lsa02b.native_shortest_tactic_frame.v1"
            or row["evidence"]!="lean_kernel_verified_reference_path"
            or row.get("split", "train") != "train"):
            # Original native teacher frame schema has no split field.
            raise ValueError("non-train or unsupported native source teacher")
    multi_source=sum(sum(frame["labels"])>=2 for frame in source_rows)
    visited_multi=[f for f in frames if f["n_optimal"]>=2]
    multis=len(visited_multi)
    roots_with_multi=len({x["lineage"] for x in visited_multi})
    eligible=len(frames)
    if eligible<1:
        raise ValueError("LBF-11 has zero independently replayed model-visited witnesses")
    if multis!=meta["multiple_shortest_action_frames"]:
        raise ValueError("native multi-shortest visit count differs from frozen source")
    for f in visited_multi:
        if (f["split"]!="train"
            or len(set(f["candidates"]))!=len(f["candidates"])
            or sum(f["labels"])!=f["n_optimal"]
            or sum(f["single_labels"])!=1):
            raise ValueError("invalid multi-optimal native train frame")
        one=f["single_labels"].index(1)
        if f["labels"][one] != 1:
            raise ValueError("single-action target is not a member of native shortest set")

    by_root=defaultdict(list)
    for row in frames:
        by_root[row["lineage"]].append(row)
    root_fraction=sum(sum(x["n_optimal"]>=2 for x in rows)/len(rows)
                      for rows in by_root.values())/len(by_root)
    schedules={}
    for seed in (107,109):
        draws=frame_schedule(source_rows,frames,seed,720)
        vis=defaultdict(list)
        for row in frames:
            vis[row["lineage"]].append(row)
        selected=[vis[row["lineage"]][row["visit_index"]]
                  for row in draws if row["use_model_visited"]]
        multi_updates=sum(row["n_optimal"]>=2 for row in selected)
        schedules[str(seed)]={
            "total_steps":720,
            "model_visited_updates":len(selected),
            "multi_positive_visited_updates":multi_updates,
            "draw_schedule_sha256":hashlib.sha256(
                "".join(canonical(row)+"\n" for row in draws).encode()).hexdigest(),
        }
    exposure=multis/eligible
    enough=(exposure>=VISIT_MIN_FRACTION and
            roots_with_multi>=MULTI_ROOTS_MIN and
            len(by_root)>=12 and
            all(row["multi_positive_visited_updates"]>=MIN_MULTI_TRAIN_UPDATES
                for row in schedules.values()))
    result={
        "schema":SCHEMA,
        "suite_manifest_sha256":sha256_file(suite/"manifest.json"),
        "native_teacher_manifest_sha256":sha256_file(teacher/"manifest.json"),
        "collection_manifest_sha256":sha256_file(collection/"manifest.json"),
        "independent_replay_sha256":sha256_file(replay_receipt),
        "behavior_checkpoint_sha256":sha256_file(behavior),
        "source_structural_alt":True,
        "train_roots":48,
        "source_reference_frames":len(source_rows),
        "source_frames_with_multi_optimal_actions":multi_source,
        "model_visited_frames":eligible,
        "model_visited_multi_optimal_frames":multis,
        "model_visited_multi_fraction":exposure,
        "model_visited_root_weighted_multi_fraction":root_fraction,
        "model_visited_roots_with_witnesses":len(by_root),
        "model_visited_roots_with_multi_optimal_witnesses":roots_with_multi,
        "minimum_multiframe_fraction":VISIT_MIN_FRACTION,
        "minimum_multi_optimal_roots":MULTI_ROOTS_MIN,
        "minimum_multi_positive_training_steps":MIN_MULTI_TRAIN_UPDATES,
        "seed_draws":schedules,
        "source_native_verified":True,
        "source_independent_replay_verified":True,
        "labels_used_only_for_trainer_exposure_gate":True,
        "source_scoring_split":"train",
        "ordinary_test_opened":False,
        "ood_family_opened":False,
        "scientific_eligible":False,
        "gate":"MULTI_OPTIMAL_SOURCE_SUFFICIENT" if enough else "INSUFFICIENT_MULTI_OPTIMAL_EXPOSURE",
    }
    out.mkdir(parents=True)
    (out/"manifest.json").write_text(canonical(result)+"\n",encoding="utf-8")
    if not enough:
        raise ValueError("INSUFFICIENT_MULTI_OPTIMAL_EXPOSURE: registered native model-visited action-set gate failed")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("suite","teacher","collection","behavior","replay-receipt","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    result=gate(args.suite,args.teacher,args.collection,args.behavior,
                args.replay_receipt,args.out)
    print(canonical(result))


if __name__=="__main__":
    main()
