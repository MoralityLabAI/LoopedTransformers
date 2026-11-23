"""LBF-06 PP-channel shams: paired kernel-certified development comparison.

Only real Lean episode certificates and independently replayed traces count as
proof completion. Two training seeds reuse the same independent theorem roots.
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
from ppt_gym.data import canonical,sha256_file

SCHEMA="lbf06.pretty_context_causal_screen.v1"
VIEWS=("hybrid","hybrid_nopp","hybrid_shuffle")
SEEDS=(107,109)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def interval(vals):
    rng=random.Random(20261123)
    draws=2000
    samples=sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals)
                   for _ in range(draws))
    return [samples[int(draws*.025)],samples[int(draws*.975)-1]]


def compare(teacher:Path,runs:Path,out:Path)->dict:
    if out.exists():
        raise ValueError("report path must be fresh")
    src=_load(teacher/"manifest.json")
    sh=sha256_file(teacher/"manifest.json")
    if (not src["source_compiler_passed"] or src["backend"]["actual_lean_process"] is not True
        or src["independent_lineages"] != 48 or src["query_roots_exposed_to_fit"] != 0):
        raise ValueError("native 48-lineage train provenance failed")
    cells={}
    initial={}
    common=None
    for seed in SEEDS:
        initial[seed]=[]
        for view in VIEWS:
            key=f"{seed}-{view}"
            d=runs/key
            fit=_load(d/"fit"/"manifest.json")
            native=_load(d/"calibration"/"manifest.json")
            replay=_load(d/"replay"/"replay.json")
            for filename, expected in native["artifacts"].items():
                if Path(filename).name!=filename or sha256_file(d/"calibration"/filename)!=expected:
                    raise ValueError(key+": native artifact hash mismatch")
            if not (
                fit["input_view"] == view
                and fit["seed"] == seed
                and fit["optimizer_steps"] == 720
                and fit["unique_parameter_count"] == 57475
                and fit["train_order"] == "ab"
                and fit["native_teacher_manifest_sha256"] == sh
                and fit["training_source_native"] is True
                and fit["optimizer_calls_on_calibration_or_test"] == 0
                and fit["view_input_over_budget"] == 0
                and native["input_view"] == view
                and native["checkpoint_sha256"] == fit["checkpoint_sha256"]
                and native["training_provenance"]["native_teacher_manifest_sha256"] == sh
                and native["training_provenance"]["training_source_verified"] is True
                and native["training_provenance"]["source_lineages_disjoint"] is True
                and native["training_provenance"]["training_lineages"] == 48
                and native["backend"]["actual_lean_process"] is True
                and native["status"] == "complete"
                and native["split"] == "calibration"
                and native["paired_roots"] == 32
                and native["standalone_reference_certificates"] == 32
                and native["query_label_training_updates"] == 0
                and replay["replay_passed"] is True
                and replay["independent_replay_episodes"] == 96
                and replay["session_manifest_sha256"] == sha256_file(d/"calibration"/"manifest.json")
            ):
                raise ValueError(key+": fit, native, or independent replay failed")
            rows=_lines(d/"calibration"/"paired_roots.jsonl")
            ids={x["root"] for x in rows}
            if len(ids)!=32 or len({x["lineage"] for x in rows})!=32:
                raise ValueError(key+": repeated or missing theorem roots")
            if common is None:common=ids
            elif common!=ids:raise ValueError(key+": theorem roster changed")
            initial[seed].append(fit["initial_weights_sha256"])
            cells[key]={
                "proofs":native["scores"]["ab"]["standalone_verified_closures"],
                "checks":native["scores"]["ab"]["tactic_checks"],
                "reference_reachable":native["reference_reachable_within_budget"],
                "auto8":native["scores"]["auto8"]["standalone_verified_closures"],
                "checkpoint_sha256":fit["checkpoint_sha256"],
                "last_loss":fit["last_training_loss"],
                "root_scores":{row["root"]:int(row["ab_certified"]) for row in rows},
            }
    if any(len(set(vals))!=1 for vals in initial.values()):
        raise ValueError("same-seed initialization differs between PP arms")
    if len({x["reference_reachable"] for x in cells.values()})!=1:
        raise ValueError("native reference proof ceiling changed")
    effects={}
    for sham in ("hybrid_nopp","hybrid_shuffle"):
        vals=[]
        for root in sorted(common):
            value=sum(
                cells[f"{seed}-hybrid"]["root_scores"][root] -
                cells[f"{seed}-{sham}"]["root_scores"][root] for seed in SEEDS
            )/len(SEEDS)
            vals.append(value)
        effects["hybrid_minus_"+sham]={
            "mean":sum(vals)/len(vals),
            "descriptive_root_bootstrap_95":interval(vals),
            "wins":sum(x>0 for x in vals),
            "losses":sum(x<0 for x in vals),
            "ties":sum(x==0 for x in vals),
        }
    out.mkdir(parents=True)
    result={
        "schema":SCHEMA,"source_lean_verified":True,
        "teacher_manifest_sha256":sh,"train_lineages":48,
        "train_frames":src["train_examples"],"calibration_unique_roots":32,
        "views":list(VIEWS),"seeds":list(SEEDS),
        "matched_initial_weights_per_seed":True,"same_cpu_worker":True,
        "all_episodes_replayed":True,
        "cells":{key:{k:v for k,v in cell.items() if k!="root_scores"}
                 for key,cell in sorted(cells.items())},
        "paired_effects":effects,
        "development_pp_effect_threshold":.05,
        "effect_exceeds_floor_against_both_shams":all(x["mean"]>=.05 for x in effects.values()),
        "native_candidate_uniform_baseline_measured":False,
        "competence_gate":"BLOCKED_INCOMPETENT_CHECKPOINT",
        "locked_test_opened":False,"scientific_eligible":False,
        "interpretation":"Within-source development causal-control screen; unique roots=32, not 64. PP shams do not assert a general mechanism.",
    }
    (out/"results.json").write_text(canonical(result)+"\n",encoding="utf-8")
    lines=["# LBF-06 readable-context control (development)", "",
           "Native Lean proof completion on 32 unique shared calibration roots, eight check budget. All source proofs independently compiled and all scored episodes model-free replayed.",
           "",
           "| Seed | Original hybrid | Byte-matched blank PP | Character-shuffled PP | Public auto8 | Reference reachable |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        a,b,c=[cells[f"{seed}-{v}"] for v in VIEWS]
        lines.append(f"| {seed} | {a['proofs']}/32 | {b['proofs']}/32 | {c['proofs']}/32 | {a['auto8']}/32 | {a['reference_reachable']}/32 |")
    lines += ["", "Root-paired effect means and descriptive intervals:"]
    for key, effect in effects.items():
        lines.append(f"- {key}: {effect['mean']:+.5f}, 95% descriptive root-bootstrap {effect['descriptive_root_bootstrap_95']}, wins/losses/ties {effect['wins']}/{effect['losses']}/{effect['ties']}")
    lines += ["",f"Prespecified +.05 advantage over **both** PP shams: {'PASS' if result['effect_exceeds_floor_against_both_shams'] else 'MISS'}.",
              "Model competence remains blocked, and locked test roots remain unopened.",
              ""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    res=compare(a.teacher,a.runs,a.out)
    print("LBF06_CELLS",canonical(res["cells"]))
    print("LBF06_EFFECTS",canonical(res["paired_effects"]))
    print("LBF06_DECISION",res["effect_exceeds_floor_against_both_shams"],res["competence_gate"])


if __name__=="__main__":
    main()
