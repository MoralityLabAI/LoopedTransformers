"""LBF-09 native action-set proof comparison with hard multi-optimal source gate.

The LBF-08 read-only comparator verifies full train-source, replay, checkpoint,
root-level proof and independent compiler provenance. This wrapper checks the
preregistered *new* equal-cost Lean source and actual multi-positive exposure
before interpreting one-positive versus full-shortest-set effects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))
from ppt_gym.data import canonical,sha256_file
from lbf08_compare import compare as audited_native_compare

SCHEMA="lbf09.multi_shortest_native_action_set.v1"


def compare(suite:Path,teacher:Path,collection:Path,replay:Path,
            exposure:Path,runs:Path,out:Path)->dict:
    if out.exists():
        raise ValueError("LBF-09 output report directory must be fresh")
    source=json.loads((suite/"manifest.json").read_text(encoding="utf-8"))
    gate=json.loads((exposure/"manifest.json").read_text(encoding="utf-8"))
    if not (
        source.get("seed")==20261130
        and source.get("equal_alt_p")==1.0
        and source.get("source_family")=="lbf09_true_equal_cost_premise_twins.v1"
        and gate.get("schema")=="lbf09.native_multi_optimal_source_exposure.v1"
        and gate.get("gate")=="MULTI_OPTIMAL_SOURCE_SUFFICIENT"
        and gate.get("source_native_verified") is True
        and gate.get("source_independent_replay_verified") is True
        and gate.get("suite_manifest_sha256")==sha256_file(suite/"manifest.json")
        and gate.get("native_teacher_manifest_sha256")==sha256_file(teacher/"manifest.json")
        and gate.get("collection_manifest_sha256")==sha256_file(collection/"manifest.json")
        and gate.get("independent_replay_sha256")==sha256_file(replay/"replay.json")
        and gate.get("model_visited_multi_fraction",0)>=.25
        and gate.get("model_visited_roots_with_multi_optimal_witnesses",0)>=12
        and all(gate.get("seed_draws",{}).get(str(seed),{}).get("multi_positive_visited_updates",0)>=50
                for seed in (107,109))
    ):
        raise ValueError("LBF-09 source multi-shortest native exposure gate has not passed")
    out.mkdir(parents=True)
    evidence=audited_native_compare(teacher,collection,replay/"replay.json",
                                    runs,out/"lbf08_native_audit")
    if not (
        evidence["schema"]=="lbf08.repaired_policy_native_paired_v1"
        and evidence["calibration_unique_roots"]==32
        and evidence["train_lineages"]==48
        and evidence["native_episode_replay_verified"] is True
        and evidence["locked_test_opened"] is False
        and evidence["matched_initial_weights_and_draws"] is True
    ):
        raise ValueError("LBF-09 inherited native proof evidence integrity gate failed")
    cells=evidence["cells"]
    for seed in (107,109):
        one=cells[f"{seed}-visited_one"]
        full=cells[f"{seed}-visited_set"]
        source_only=cells[f"{seed}-source_only"]
        if (one["model_visited_updates"]!=full["model_visited_updates"]
            or one["multi_positive_visited_updates"]!=0
            or full["multi_positive_visited_updates"]!=
               gate["seed_draws"][str(seed)]["multi_positive_visited_updates"]
            or full["multi_positive_visited_updates"]<50
            or source_only["model_visited_updates"]!=0):
            raise ValueError("LBF-09 real action-set optimizer exposure does not match frozen draw protocol")
    primary=evidence["root_paired_effects"]["visited_set_minus_visited_one"]
    secondary=evidence["root_paired_effects"]["visited_set_minus_source_only"]
    decision=primary["mean"]>=.05
    secondary_pass=secondary["mean"]>=.05
    result={
        "schema":SCHEMA,
        "registered_primary_action_set_delta":.05,
        "registered_secondary_visited_delta":.05,
        "primary_action_set_development_pass":decision,
        "secondary_visited_development_pass":secondary_pass,
        "multi_optimal_training_source_gate_pass":True,
        "source_only_training_roots":48,
        "train_only_native_source_frames":evidence["native_train_frames"],
        "model_visited_witness_frames":gate["model_visited_frames"],
        "model_visited_multi_optimal_frames":gate["model_visited_multi_optimal_frames"],
        "model_visited_multi_optimal_fraction":gate["model_visited_multi_fraction"],
        "model_visited_roots_with_multi_optimal":gate["model_visited_roots_with_multi_optimal_witnesses"],
        "per_seed_real_multi_positive_visited_training_updates":{
            str(seed):cells[f"{seed}-visited_set"]["multi_positive_visited_updates"]
            for seed in (107,109)},
        "same_source_trained_model_parameters":True,
        "unique_calibration_roots":32,
        "seeds":[107,109],
        "arms":["source_only","visited_one","visited_set"],
        "native_checkpoint_cells":cells,
        "root_paired_effects":evidence["root_paired_effects"],
        "teacher_manifest_sha256":sha256_file(teacher/"manifest.json"),
        "source_suite_manifest_sha256":sha256_file(suite/"manifest.json"),
        "exposure_gate_manifest_sha256":sha256_file(exposure/"manifest.json"),
        "lbf08_complete_native_audit_sha256":sha256_file(out/"lbf08_native_audit"/"results.json"),
        "locked_test_opened":False,
        "scientific_eligible":False,
        "competence_gate":"BLOCKED_INCOMPETENT_CHECKPOINT",
        "interpretation":"Development-only true multi-shortest native action-set comparison; 32 roots, two paired seeds. All proof successes independently kernel/standalone certified.",
    }
    (out/"results.json").write_text(canonical(result)+"\n",encoding="utf-8")
    report=[
        "# LBF-09: genuine native multi-shortest proof branches and action-set training",
        "",
        "Only development calibration roots were opened. Source: Lean 4.19 real theorem server, 48 independently compiled train roots, two matched seeds, all proof completions standalone compiled and model-free replayed.",
        "",
        f"Source exposure gate PASSED: {gate['model_visited_multi_optimal_frames']}/{gate['model_visited_frames']} witnessed model-visited frames have >=2 shortest public tactics ({gate['model_visited_multi_fraction']:.1%}); {gate['model_visited_roots_with_multi_optimal_witnesses']} independent train roots carry multi-optimal visits.",
        "",
        "| Seed | Reference only | Visited one optimal action | Visited full action set | Public auto8 | Native reference attainable | Multi-positive update count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (107,109):
        sc,one,full=[cells[f"{seed}-{arm}"] for arm in ("source_only","visited_one","visited_set")]
        report.append(f"| {seed} | {sc['proofs']}/32 | {one['proofs']}/32 | {full['proofs']}/32 | {sc['public_auto8_proofs']}/32 | {sc['native_reference_budget_reachable']}/32 | {full['multi_positive_visited_updates']} |")
    report += ["", "## Paired root-level effects, descriptive 95% bootstrap"]
    for key, effect in evidence["root_paired_effects"].items():
        report.append(f"- {key}: {effect['mean']:+.5f}, 95% interval {effect['descriptive_95_root_bootstrap']}; wins/losses/ties {effect['wins']}/{effect['losses']}/{effect['ties']}")
    report.extend(["",
        f"Registered +.05 full-action-set over one-action development screen: {'PASS' if decision else 'MISS'}.",
        f"Registered +.05 full-action-set over reference-only secondary screen: {'PASS' if secondary_pass else 'MISS'}.",
        "",
        "Model competence remains blocked; an exact independent candidate-uniform native baseline and an FSI-01 powered confirmation are absent. No sealed ordinary/OOD evaluations occurred.",
        "",
    ])
    (out/"RESULTS.md").write_text("\n".join(report),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("suite","teacher","collection","replay","exposure","runs","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    result=compare(a.suite,a.teacher,a.collection,a.replay,a.exposure,a.runs,a.out)
    print("LBF09_NATIVE_CELLS",canonical(result["native_checkpoint_cells"]))
    print("LBF09_PAIRED_EFFECTS",canonical(result["root_paired_effects"]))
    print("LBF09_DECISION",result["primary_action_set_development_pass"],
          result["secondary_visited_development_pass"],result["competence_gate"])


if __name__=="__main__":
    main()
