"""LBF-05 immutable native root-paired raw / typed / hybrid comparison.

No optimizer updates and no Lean execution. Checks input provenance, original
standalone certificates, model-free replay, per-seed weights and root pairing.
Root resampling (rather than seed-as-extra-sample) handles repeated theorems.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

LCGRT = Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))
from ppt_gym.data import canonical, sha256_file

SEEDS = (107,109)
VIEWS = ("raw","typed","hybrid")
SCHEMA = "lbf05.matched_native_views.v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _interval(values: list[float], n: int = 2000) -> list[float]:
    if not values:
        raise ValueError("zero independent root differences")
    rng = random.Random(20261123)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(n))
    return [means[int(n*0.025)], means[int(n*0.975)-1]]


def compare(teacher: Path, runs: Path, out: Path, audit: Path) -> dict:
    if out.exists():
        raise ValueError("comparison output must be fresh")
    source = _json(teacher/"manifest.json")
    source_hash = sha256_file(teacher/"manifest.json")
    a = _json(audit/"manifest.json")
    if (not source["source_compiler_passed"]
        or source["backend"]["actual_lean_process"] is not True
        or source["independent_lineages"] != 48
        or source["query_roots_exposed_to_fit"] != 0
        or a["teacher_manifest_sha256"] != source_hash
        or a["training_frames"] != source["train_examples"]
        or a["independent_lineages"] != 48):
        raise ValueError("source training provenance or label-blind audit mismatch")

    outcomes = {}
    initial = {}
    all_roots = None
    for seed in SEEDS:
        initial[seed] = {}
        for view in VIEWS:
            key = f"{seed}-{view}"
            directory = runs/key
            f = _json(directory/"fit"/"manifest.json")
            n = _json(directory/"calibration"/"manifest.json")
            r = _json(directory/"replay"/"replay.json")
            root_rows = _rows(directory/"calibration"/"paired_roots.jsonl")
            for filename, expected in n["artifacts"].items():
                if (Path(filename).name != filename
                    or sha256_file(directory/"calibration"/filename) != expected):
                    raise ValueError(key+": native receipt bytes changed: "+filename)
            if (
                f["input_view"] != view or n["input_view"] != view
                or f["seed"] != seed
                or f["train_order"] != "ab" or f["optimizer_steps"] != 720
                or f["native_teacher_manifest_sha256"] != source_hash
                or n["checkpoint_sha256"] != f["checkpoint_sha256"]
                or n["training_provenance"]["native_teacher_manifest_sha256"] != source_hash
                or n["training_provenance"]["source_lineages_disjoint"] is not True
                or n["training_provenance"]["training_lineages"] != 48
                or n["training_provenance"]["training_source_verified"] is not True
                or n["backend"]["actual_lean_process"] is not True
                or n["status"] != "complete"
                or n["split"] != "calibration" or n["paired_roots"] != 32
                or n["standalone_reference_certificates"] != 32
                or n["query_label_training_updates"] != 0
                or n["architecture_promotion_eligible"] is not False
                or r["replay_passed"] is not True
                or r["independent_replay_episodes"] != 96
                or r["session_manifest_sha256"] != sha256_file(directory/"calibration"/"manifest.json")
            ):
                raise ValueError(key+": checkpoint / native proof / replay manifest mismatch")
            if len(root_rows)!=32 or len({x["root"] for x in root_rows})!=32:
                raise ValueError(key+": duplicate or incomplete independent roots")
            roots = {x["root"] for x in root_rows}
            if all_roots is None:
                all_roots = roots
            elif roots != all_roots:
                raise ValueError(key+": paired root fingerprints differ")
            initial[seed][view] = f["initial_weights_sha256"]
            outcomes[key] = {
                "fit": {
                    "checkpoint_sha256":f["checkpoint_sha256"],
                    "source_manifest_sha256":f["source_manifest_sha256"],
                    "parameter_count":f["unique_parameter_count"],
                    "loss_first":f["initial_training_loss"],
                    "loss_last":f["last_training_loss"],
                    "mean_visible_input_bytes": f["mean_view_public_bytes"],
                    "over_input_budget": f["view_input_over_budget"],
                    "contradictory_training_collisions": f["conflicting_training_view_collisions"],
                },
                "native": {
                    "proofs":n["scores"]["ab"]["standalone_verified_closures"],
                    "rate":n["scores"]["ab"]["standalone_verified_closures"]/32,
                    "tactic_checks":n["scores"]["ab"]["tactic_checks"],
                    "model_calls":n["scores"]["ab"]["model_calls"],
                    "reference_budget_reachable":n["reference_reachable_within_budget"],
                    "auto8_proofs":n["scores"]["auto8"]["standalone_verified_closures"],
                },
                "root_results":{x["root"]:int(x["ab_certified"]) for x in root_rows},
            }
    if not all(len(set(initial[seed].values())) == 1 for seed in SEEDS):
        raise ValueError("same-seed initial weights differ between input arms")
    if len({outcomes[key]["fit"]["parameter_count"] for key in outcomes})!=1:
        raise ValueError("parameter count differs between matched arms")
    if len({outcomes[key]["fit"]["source_manifest_sha256"] for key in outcomes})!=1:
        raise ValueError("source differ between matched arms")
    if len({outcomes[key]["native"]["reference_budget_reachable"] for key in outcomes}) != 1:
        raise ValueError("reference budget changed between matched arms")
    for view in ("typed","hybrid"):
        for seed in SEEDS:
            if outcomes[f"{seed}-{view}"]["fit"]["over_input_budget"] != 0:
                raise ValueError(f"unbounded input encoding {seed}-{view}")

    paired = {}
    for alternate in ("typed","hybrid"):
        differences=[]
        per_root={}
        for root in sorted(all_roots):
            effects=[
                outcomes[f"{seed}-{alternate}"]["root_results"][root] -
                outcomes[f"{seed}-raw"]["root_results"][root] for seed in SEEDS
            ]
            value=sum(effects)/len(effects)
            per_root[root] = {"seed_deltas":effects,"mean":value}
            differences.append(value)
        paired[alternate+"_minus_raw"]={
            "mean":sum(differences)/len(differences),
            "descriptive_root_bootstrap_95":_interval(differences),
            "root_wins":sum(v>0 for v in differences),
            "root_losses":sum(v<0 for v in differences),
            "root_ties":sum(v==0 for v in differences),
            "per_root":per_root,
        }

    def compact(key):
        row=outcomes[key]
        return {**row["native"],**row["fit"]}
    result = {
        "schema": SCHEMA, "train_lineages":48,"native_training_frames":source["train_examples"],
        "calibration_unique_roots":32,"seeds":list(SEEDS),"views":list(VIEWS),
        "same_initial_weights_per_seed":True,
        "same_cpu_runner":True,"source_lean_compiled":True,
        "all_native_episodes_replayed":True,
        "input_audit_sha256":sha256_file(audit/"manifest.json"),
        "source_manifest_sha256":source_hash,
        "cells":{key:compact(key) for key in sorted(outcomes)},
        "paired":paired,
        "registered_hybrid_gain_threshold":0.05,
        "hybrid_gain_clears_development_floor":paired["hybrid_minus_raw"]["mean"]>=0.05,
        "candidate_uniform_native_baseline_measured":False,
        "competence_gate":"BLOCKED_INCOMPETENT_CHECKPOINT",
        "locked_test_opened":False,"architecture_promotion_eligible":False,
        "scientific_eligible":False,
        "notes":("This comparison is a development screen. Two matched seeds share the same "
                 "32 roots and do not increase the independent-root denominator. Raw vs "
                 "typed has prior exposure; any hybrid effect is an initial pilot, "
                 "and separate preregistered novel roots are needed for confirmation."),
    }
    out.mkdir(parents=True)
    (out/"matched.json").write_text(canonical(result)+"\n",encoding="utf-8")
    head=[
        "# LBF-05 matched native Lean public-input comparison",
        "",
        "**Status:** development-only CPU experiment. Locked/held-out splits remain sealed.",
        "",
        f"Train: 48 native-compiled roots, {source['train_examples']} frames. Calibration: 32 independent roots shared across two training seeds.",
        "",
        "| Seed | Raw | Typed | Hybrid | Auto8 | Native budget reachable |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        nr=lambda v:outcomes[f"{seed}-{v}"]["native"]
        head.append(f"| {seed} | {nr('raw')['proofs']}/32 | {nr('typed')['proofs']}/32 | {nr('hybrid')['proofs']}/32 | {nr('raw')['auto8_proofs']}/32 | {nr('raw')['reference_budget_reachable']}/32 |")
    head += [
        "",
        f"Hybrid minus raw mean: **{paired['hybrid_minus_raw']['mean']:+.4f}**; root bootstrap descriptive 95% interval **{paired['hybrid_minus_raw']['descriptive_root_bootstrap_95']}**.",
        f"Typed minus raw mean: **{paired['typed_minus_raw']['mean']:+.4f}**; root bootstrap descriptive 95% interval **{paired['typed_minus_raw']['descriptive_root_bootstrap_95']}**.",
        "",
        f"Registered hybrid +0.05 development screen: **{'PASS' if result['hybrid_gain_clears_development_floor'] else 'MISS'}**.",
        "",
        "Model checkpoint competence: **BLOCKED_INCOMPETENT_CHECKPOINT**. Native candidate-uniform baseline absent; no sealed test evaluation.",
        "Two seeds are paired repeats on the same theorem roots. No significance claim.",
        "",
    ]
    (out/"RESULTS.md").write_text("\n".join(head),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=compare(args.teacher,args.runs,args.out,args.audit)
    print(canonical({k:v for k,v in result.items() if k!="paired"}))
    print("LBF05_PAIRED",canonical({key:{k:v for k,v in row.items() if k!='per_root'}
                                           for key,row in result["paired"].items()}))


if __name__=="__main__":
    main()
