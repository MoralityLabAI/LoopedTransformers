"""LBF-08 root-paired native proof-completion audit.

Read-only comparator. Every counted closure has real Lean, standalone compiler
and exact source-episode replay provenance. Roots are resampled, not seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))

from ppt_gym.cli import checkpoint_load
from ppt_gym.data import canonical, sha256_file
from ppt_gym.lbf08_repair import (
    ARMS, COLLECT_SCHEMA, FIT_SCHEMA, REPLAY_SCHEMA,
)

SEEDS=(107,109)
SCHEMA="lbf08.repaired_policy_native_paired_v1"


def _json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path:Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def root_bootstrap(vals:list[float])->list[float]:
    if len(vals)<2:
        raise ValueError("too few independent theorem-root effects")
    rng=random.Random(20261123)
    draws=2000
    means=sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals)
                 for _ in range(draws))
    return [means[int(draws*.025)],means[int(draws*.975)-1]]


def compare(teacher:Path, repair:Path, replay:Path, runs:Path, out:Path)->dict:
    if out.exists():
        raise ValueError("LBF-08 report destination must be fresh")
    t=_json(teacher/"manifest.json")
    rep=_json(repair/"manifest.json")
    rp=_json(replay/"replay.json")
    teacher_sha=sha256_file(teacher/"manifest.json")
    repair_sha=sha256_file(repair/"manifest.json")
    replay_sha=sha256_file(replay/"replay.json")
    if not (
        t["schema"]=="lsa02b.native_teacher.v1"
        and t["source_compiler_passed"] is True
        and t["backend"]["actual_lean_process"] is True
        and t["independent_lineages"]==48
        and t["query_roots_exposed_to_fit"]==0
        and rep["schema"]==COLLECT_SCHEMA
        and rep["source_split"]=="train"
        and rep["native_backend"]["actual_lean_process"] is True
        and rep["native_teacher_manifest_sha256"]==teacher_sha
        and rep["source_reference_compiler_passed"] is True
        and rep["visited_witnessed_lineages"]>=12
        and rep["behavior_root_episodes"]==48
        and rep["query_roots_exposed_to_collection"]==0
        and rep["locked_test_opened"] is False
        and rp["schema"]==REPLAY_SCHEMA
        and rp["collection_manifest_sha256"]==repair_sha
        and rp["native_replay_passed"] is True
        and rp["behavior_episodes_replayed"]==48
        and rp["teacher_forced_full_proofs_replayed"]==rep["teacher_forced_standalone_certificates"]
    ):
        raise ValueError("LBF-08 train-source native proof/repair provenance failed")
    for fname,sha in t["artifacts"].items():
        if Path(fname).name!=fname or sha256_file(teacher/fname)!=sha:
            raise ValueError("native teacher artifact hash changed")
    for fname,sha in rep["artifacts"].items():
        if Path(fname).name!=fname or sha256_file(repair/fname)!=sha:
            raise ValueError("model visitation/repair artifact hash changed")
    cells={}
    initial={}
    draws={}
    root_set=None
    for seed in SEEDS:
        initial[seed]=[]
        draws[seed]=[]
        for arm in ARMS:
            key=f"{seed}-{arm}"
            directory=runs/key
            f=_json(directory/"fit"/"manifest.json")
            native_dir=directory/"calibration"
            native=_json(native_dir/"manifest.json")
            native_replay=_json(directory/"replay"/"replay.json")
            model,ck=checkpoint_load(directory/"fit"/"checkpoint.pt")
            if not (
                f["schema"]==FIT_SCHEMA
                and f["seed"]==seed and f["arm"]==arm
                and f["train_order"]=="ab"
                and f["updates"]==720 and f["requested_steps"]==720
                and f["parameter_count"]==57475
                and f["source_roots"]==48
                and f["native_source"] is True
                and f["calibration_updates"]==0
                and f["locked_test_opened"] is False
                and f["native_teacher_manifest_sha256"]==teacher_sha
                and f["repair_source_manifest_sha256"]==repair_sha
                and f["independent_replay_sha256"]==replay_sha
                and f["checkpoint_sha256"]==sha256_file(directory/"fit"/"checkpoint.pt")
                and f["optimizer_steps_sha256"]==sha256_file(directory/"fit"/"optimizer_steps.jsonl")
                and ck.get("lbf08_training_arm")==arm
                and ck.get("lbf08_repair_source_manifest_sha256")==repair_sha
                and ck.get("lbf08_independent_replay_sha256")==replay_sha
                and ck.get("native_teacher_sha256")==teacher_sha
                and ck.get("seed")==seed and ck.get("train_steps")==720
                and ck.get("source_manifest_sha256")==t["source"]["suite_manifest_sha256"]
                and model.parameter_count()==57475
                and native["schema"]=="lsa02b.native_block_order.v1"
                and native["split"]=="calibration"
                and native["paired_roots"]==32
                and native["status"]=="complete"
                and native["backend"]["actual_lean_process"] is True
                and native["checkpoint_sha256"]==f["checkpoint_sha256"]
                and native["training_provenance"]["training_source_verified"] is True
                and native["training_provenance"]["source_lineages_disjoint"] is True
                and native["training_provenance"]["training_lineages"]==48
                and native["training_provenance"]["native_teacher_manifest_sha256"]==teacher_sha
                and native["standalone_reference_certificates"]==32
                and native["query_label_training_updates"]==0
                and native["architecture_promotion_eligible"] is False
                and native_replay["replay_passed"] is True
                and native_replay["independent_replay_episodes"]==96
                and native_replay["session_manifest_sha256"]==sha256_file(native_dir/"manifest.json")
            ):
                raise ValueError("LBF-08 matched native checkpoint/receipt mismatch: "+key)
            for fname,sha in native["artifacts"].items():
                if Path(fname).name!=fname or sha256_file(native_dir/fname)!=sha:
                    raise ValueError("native calibration receipt artifact tampered: "+key)
            rows=_rows(native_dir/"paired_roots.jsonl")
            ids={x["root"] for x in rows}
            if len(rows)!=32 or len(ids)!=32 or len({x["lineage"] for x in rows})!=32:
                raise ValueError("native pair root identities duplicate/incomplete")
            if root_set is None:
                root_set=ids
            elif root_set != ids:
                raise ValueError("native evaluation root identities differ between models")
            initial[seed].append(f["initial_weights_sha256"])
            draws[seed].append(f["registered_reference_visitor_draw_schedule_sha256"])
            cells[key]={
                "proofs":native["scores"]["ab"]["standalone_verified_closures"],
                "tactic_checks":native["scores"]["ab"]["tactic_checks"],
                "model_calls":native["scores"]["ab"]["model_calls"],
                "public_auto8_proofs":native["scores"]["auto8"]["standalone_verified_closures"],
                "native_reference_budget_reachable":native["reference_reachable_within_budget"],
                "source_only_updates":f["reference_path_updates"],
                "model_visited_updates":f["model_visited_updates"],
                "multi_positive_visited_updates":f["multi_positive_visited_updates"],
                "checkpoint_sha256":f["checkpoint_sha256"],
                "last_32_loss":f["mean_last_32_loss"],
                "_roots":{x["root"]:int(x["ab_certified"]) for x in rows},
            }
    for seed in SEEDS:
        if len(set(initial[seed]))!=1 or len(set(draws[seed]))!=1:
            raise ValueError("same-seed initial weights or source/visit draw roster differ")
    if len({x["native_reference_budget_reachable"] for x in cells.values()})!=1:
        raise ValueError("structural Lean attainable ceiling changed between arms")
    if len({x["public_auto8_proofs"] for x in cells.values()})!=1:
        raise ValueError("public auto8 baseline changed between arms")
    if any(cells[f"{seed}-visited_one"]["model_visited_updates"] !=
           cells[f"{seed}-visited_set"]["model_visited_updates"] for seed in SEEDS):
        raise ValueError("one/set matched visitation training exposure differs")
    paired={}
    for comparison,first,second in (
        ("visited_set_minus_source_only","visited_set","source_only"),
        ("visited_set_minus_visited_one","visited_set","visited_one"),
        ("visited_one_minus_source_only","visited_one","source_only"),
    ):
        vals=[]
        for root in sorted(root_set):
            value=sum(cells[f"{seed}-{first}"]["_roots"][root] -
                      cells[f"{seed}-{second}"]["_roots"][root]
                      for seed in SEEDS)/len(SEEDS)
            vals.append(value)
        paired[comparison]={
            "mean":sum(vals)/len(vals),
            "descriptive_95_root_bootstrap":root_bootstrap(vals),
            "wins":sum(v>0 for v in vals),
            "losses":sum(v<0 for v in vals),
            "ties":sum(v==0 for v in vals),
        }
    out.mkdir(parents=True)
    result={
        "schema":SCHEMA,
        "registered_primary_threshold":0.05,
        "registered_action_set_threshold":0.025,
        "primary_gain_screen":paired["visited_set_minus_source_only"]["mean"]>=.05,
        "action_set_gain_screen":paired["visited_set_minus_visited_one"]["mean"]>=.025,
        "train_lineages":48,
        "native_train_frames":t["train_examples"],
        "real_visited_lineages":rep["visited_witnessed_lineages"],
        "witnessed_model_visits":rep["model_visitation_frames"],
        "censored_model_visits":rep["censored_visits"],
        "native_attempt_status_counts":rep["native_attempt_status_counts"],
        "behavior_model_autonomous_closures":rep["model_autonomous_kernel_closures"],
        "teacher_forced_standalone_recoveries":rep["teacher_forced_standalone_certificates"],
        "source_manifest_sha256":teacher_sha,
        "model_visit_collection_sha256":repair_sha,
        "independent_collection_replay_sha256":replay_sha,
        "calibration_unique_roots":32,"seeds":list(SEEDS),"arms":list(ARMS),
        "matched_initial_weights_and_draws":True,
        "native_episode_replay_verified":True,
        "cells":{k:{f:v for f,v in x.items() if f!="_roots"}
                 for k,x in sorted(cells.items())},
        "root_paired_effects":paired,
        "competence_gate":"BLOCKED_INCOMPETENT_CHECKPOINT",
        "native_candidate_uniform_baseline_available":False,
        "locked_test_opened":False,
        "architecture_promotion_eligible":False,"scientific_eligible":False,
        "claim":"prospective developmental source-only vs model-visited repair and one-vs-set action labels",
    }
    _write= lambda p,x:p.write_text(canonical(x)+"\n",encoding="utf-8")
    _write(out/"results.json",result)
    lines=[
        "# LBF-08: native Lean model-visited repair and action-set study",
        "",
        "Development-only paired CPU experiment on 32 distinct calibration theorem roots and two matched seeds, with real Lean 4.19 and independent standalone certificates. Locked ordinary/OOD roots remain closed.",
        "",
        "| Seed | Reference-only | Visited single | Visited full action set | Auto8 | Native reachable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        source,one,aset=[cells[f"{seed}-{arm}"] for arm in ARMS]
        lines.append(
            f"| {seed} | {source['proofs']}/32 | {one['proofs']}/32 | {aset['proofs']}/32 | {source['public_auto8_proofs']}/32 | {source['native_reference_budget_reachable']}/32 |"
        )
    lines += ["","Root-paired proof-completion differences:"]
    for k,v in paired.items():
        lines.append(
            f"- {k}: {v['mean']:+.5f}, descriptive root bootstrap {v['descriptive_95_root_bootstrap']}, wins/losses/ties {v['wins']}/{v['losses']}/{v['ties']}"
        )
    lines += ["",f"Native train roots: 48; real model-visited root lineages: {rep['visited_witnessed_lineages']}; witnessed visited frames: {rep['model_visitation_frames']}.",
              f"Original behavior-model autonomous proof closures: {rep['model_autonomous_kernel_closures']}. Teacher-forced full native/standalone repair certificates: {rep['teacher_forced_standalone_certificates']}. Those teacher-forced proofs count only as source evidence.",
              f"Registered +0.05 primary development screen: {result['primary_gain_screen']}; +0.025 action-set screen: {result['action_set_gain_screen']}.",
              "No full proof-competence promotion: 40% trained close gate, exact native uniform baseline and FSI-01 prospective evidence remain outstanding.",
              ""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("teacher","repair","replay","runs","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    report=compare(args.teacher,args.repair,args.replay,args.runs,args.out)
    print("LBF08_CELLS",canonical(report["cells"]))
    print("LBF08_EFFECTS",canonical(report["root_paired_effects"]))
    print("LBF08_DECISION",report["primary_gain_screen"],
          report["action_set_gain_screen"],report["competence_gate"])


if __name__=="__main__":
    main()
