"""LBF-07 native Lean root-paired candidate-binding and position sham comparison.

All proof credit comes from original native kernel plus standalone certification.
32 theorem identities are the independent units, repeated across two seeds.
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

SCHEMA="lbf07.rule_binding_causal_screen.v1"
VIEWS=("hybrid_rule_intact","hybrid_rule_rebind","hybrid_rule_order")
SEEDS=(107,109)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def interval(vals):
    rng=random.Random(20261123)
    n=2000
    means=sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals)
                 for _ in range(n))
    return [means[int(n*.025)],means[int(n*.975)-1]]


def compare(teacher:Path,runs:Path,out:Path,audit:Path)->dict:
    if out.exists():
        raise ValueError("report output must be fresh")
    src=load(teacher/"manifest.json")
    teacher_sha=sha256_file(teacher/"manifest.json")
    a=load(audit/"manifest.json")
    if not (
        src["source_compiler_passed"] is True
        and src["backend"]["actual_lean_process"] is True
        and src["independent_lineages"]==48
        and src["query_roots_exposed_to_fit"]==0
        and a["schema"]=="lbf07.public_rule_association_audit.v1"
        and a["teacher_manifest_sha256"]==teacher_sha
        and a["train_frames"]==src["train_examples"]
        and a["unique_train_roots"]==48
        and a["exact_visible_bytes_and_unicode_histogram"] is True
        and a["intact_matches_existing_blank_pp"] is True
        and a["labels_consumed_to_choose_representation"] is False
    ):
        raise ValueError("train source or label-blind input intervention audit failed")
    cells={}
    same_start={}
    roots=None
    for seed in SEEDS:
        same_start[seed]=[]
        for view in VIEWS:
            key=f"{seed}-{view}"
            d=runs/key
            f=load(d/"fit"/"manifest.json")
            n=load(d/"calibration"/"manifest.json")
            replay=load(d/"replay"/"replay.json")
            for fname,digest in n["artifacts"].items():
                if Path(fname).name!=fname or sha256_file(d/"calibration"/fname)!=digest:
                    raise ValueError(key+": native certificate/episode artifact hash mismatch")
            if not (
                f["input_view"]==view and f["seed"]==seed
                and f["train_order"]=="ab"
                and f["optimizer_steps"]==720
                and f["training_source_native"] is True
                and f["native_teacher_manifest_sha256"]==teacher_sha
                and f["unique_parameter_count"]==57475
                and f["optimizer_calls_on_calibration_or_test"]==0
                and f["view_input_over_budget"]==0
                and n["input_view"]==view
                and n["checkpoint_sha256"]==f["checkpoint_sha256"]
                and n["training_provenance"]["native_teacher_manifest_sha256"]==teacher_sha
                and n["training_provenance"]["training_source_verified"] is True
                and n["training_provenance"]["source_lineages_disjoint"] is True
                and n["training_provenance"]["training_lineages"]==48
                and n["backend"]["actual_lean_process"] is True
                and n["status"]=="complete" and n["split"]=="calibration"
                and n["paired_roots"]==32 and n["standalone_reference_certificates"]==32
                and n["query_label_training_updates"]==0
                and n["architecture_promotion_eligible"] is False
                and replay["replay_passed"] is True
                and replay["independent_replay_episodes"]==96
                and replay["session_manifest_sha256"]==sha256_file(d/"calibration"/"manifest.json")
            ):
                raise ValueError(key+": checkpoint, root or native replay provenance mismatch")
            rows=lines(d/"calibration"/"paired_roots.jsonl")
            ids={r["root"] for r in rows}
            if len(rows)!=32 or len(ids)!=32 or len({r["lineage"] for r in rows})!=32:
                raise ValueError(key+": duplicate or incomplete root lineage")
            if roots is None:
                roots=ids
            elif roots!=ids:
                raise ValueError(key+": paired native root identities differ")
            same_start[seed].append(f["initial_weights_sha256"])
            cells[key]={
                "proofs":n["scores"]["ab"]["standalone_verified_closures"],
                "checks":n["scores"]["ab"]["tactic_checks"],
                "model_calls":n["scores"]["ab"]["model_calls"],
                "auto8":n["scores"]["auto8"]["standalone_verified_closures"],
                "reference_reachable":n["reference_reachable_within_budget"],
                "checkpoint_sha256":f["checkpoint_sha256"],
                "mean_public_bytes":f["mean_view_public_bytes"],
                "last_loss":f["last_training_loss"],
                "scores":{r["root"]:int(r["ab_certified"]) for r in rows},
            }
    if any(len(set(s))!=1 for s in same_start.values()):
        raise ValueError("same-seed initial parameter fingerprints differ")
    if len({x["reference_reachable"] for x in cells.values()})!=1:
        raise ValueError("native reference ceiling differs among matched arms")
    differences={}
    for comparison in ("intact_minus_rebind","intact_minus_order","association_specific"):
        values=[]
        for root in sorted(roots):
            if comparison=="intact_minus_rebind":
                first,second="hybrid_rule_intact","hybrid_rule_rebind"
            elif comparison=="intact_minus_order":
                first,second="hybrid_rule_intact","hybrid_rule_order"
            else:
                # (I - R) - (I - O) = (O - R)
                first,second="hybrid_rule_order","hybrid_rule_rebind"
            paired=sum(cells[f"{seed}-{first}"]["scores"][root]
                       -cells[f"{seed}-{second}"]["scores"][root]
                       for seed in SEEDS)/len(SEEDS)
            values.append(paired)
        differences[comparison]={
            "mean":sum(values)/len(values),
            "descriptive_root_bootstrap_95":interval(values),
            "wins":sum(x>0 for x in values),
            "losses":sum(x<0 for x in values),
            "ties":sum(x==0 for x in values),
        }
    edge= differences["intact_minus_rebind"]["mean"]>=.05
    specific=differences["association_specific"]["mean"]>=.05
    exposure=a["intervention_exposure_sufficient"]
    result={
        "schema":SCHEMA,"source_lean_verified":True,
        "teacher_manifest_sha256":teacher_sha,
        "public_audit_manifest_sha256":sha256_file(audit/"manifest.json"),
        "train_lineages":48,"train_frames":src["train_examples"],
        "calibration_unique_roots":32,"views":list(VIEWS),"seeds":list(SEEDS),
        "same_seed_initial_weights":True,"same_worker":True,
        "native_episodes_independently_replayed":True,
        "cells":{key:{k:v for k,v in cell.items() if k!="scores"}
                 for key,cell in sorted(cells.items())},
        "paired_effects":differences,
        "exposure_root_weighted":a["root_weighted_candidate_binding_change_fraction"],
        "exposure_sufficient":exposure,
        "edge_gain_threshold":.05,
        "specificity_threshold":.05,
        "edge_gain_pass":edge,"specificity_pass":specific,
        "association_screen_pass":bool(edge and specific and exposure),
        "native_candidate_uniform_baseline_measured":False,
        "competence_gate":"BLOCKED_INCOMPETENT_CHECKPOINT",
        "locked_test_opened":False,"scientific_eligible":False,
        "interpretation":"A candidate-binding vs position-only developmental test; two correlated seeds on the same 32 roots. Rebound inputs deliberately distort public facts.",
    }
    out.mkdir(parents=True)
    (out/"results.json").write_text(canonical(result)+"\n",encoding="utf-8")
    md=[
        "# LBF-07 typed rule association experiment",
        "",
        "32 unique independent calibration roots shared across two CPU training seeds, same eight native tactic checks and independently compiled complete Lean proofs. Sealed test/OOD unchanged.",
        "",
        "| Seed | Intact association | Rebound payloads | Complete-rule position placebo | Public auto8 | Native reachable |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        original,rebound,order=[cells[f"{seed}-{v}"] for v in VIEWS]
        md.append(f"| {seed} | {original['proofs']}/32 | {rebound['proofs']}/32 | {order['proofs']}/32 | {original['auto8']}/32 | {original['reference_reachable']}/32 |")
    md.extend(["", "## Root-paired differences (descriptive 95% bootstrap)"])
    for name,x in differences.items():
        md.append(f"- {name}: {x['mean']:+.5f}, interval {x['descriptive_root_bootstrap_95']}, wins/losses/ties {x['wins']}/{x['losses']}/{x['ties']}")
    md.extend([
        "",
        f"Training candidate-binding perturbation exposure: {a['root_weighted_candidate_binding_change_fraction']:.1%}, >=50% threshold: {exposure}",
        f"Registered +0.05 edge and specificity development screen: {'PASS' if result['association_screen_pass'] else 'MISS'}",
        "Model competence blocked; native candidate-uniform baseline absent. Locked ordinary and OOD test splits unopened.",
        "",
    ])
    (out/"RESULTS.md").write_text("\n".join(md),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=compare(a.teacher,a.runs,a.out,a.audit)
    print("LBF07_CELLS",canonical(r["cells"]))
    print("LBF07_EFFECTS",canonical(r["paired_effects"]))
    print("LBF07_SCREEN",r["association_screen_pass"],r["exposure_sufficient"],r["competence_gate"])


if __name__=="__main__":
    main()
