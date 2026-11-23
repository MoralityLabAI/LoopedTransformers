"""LBF-10 exact native uniform control and root-paired transport comparison."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path:
    sys.path.insert(0,str(LCGRT))
from ppt_gym.data import canonical,digest,sha256_file
from ppt_gym.live_suite import load_suite
from ppt_gym.lbf10_source import FAMILY
from ppt_gym.lbf10_uniform import SCHEMA as UNIFORM_SCHEMA, coefficients
from lbf08_compare import compare as native_compare, root_bootstrap

SEEDS=(107,109)
ARMS=("source_only","visited_one","visited_set")
SCHEMA="lbf10.native_structural_uniform_transport.v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def audited_uniform(suite, directory, expected):
    m=read(directory/"manifest.json")
    if not (m["schema"]==UNIFORM_SCHEMA and m["source"]["suite_manifest_sha256"]==sha256_file(suite/"manifest.json")
            and m["split"]=="calibration" and m["roots"]==32 and m["check_budget"]==8
            and m["backend"]["actual_lean_process"] is True
            and m["native_public_menu_graph_audit_passed"] is True
            and m["standalone_reference_certificates"]==32 and m["model_calls"]==0
            and m["query_label_training_updates"]==0 and m["locked_test_opened"] is False):
        raise ValueError("uniform native baseline manifest gate failed")
    for name,sha in m["artifacts"].items():
        if Path(name).name!=name or sha256_file(directory/name)!=sha:
            raise ValueError("uniform baseline receipt changed")
    rr=rows(directory/"roots.jsonl")
    if len(rr)!=32 or {x["root"]:x["lineage"] for x in rr}!=expected:
        raise ValueError("uniform root roster changed")
    result={}
    for r in rr:
        a=r["menu_audit"]
        if not (a["public_menu_exact"] and a["static_horn_context"]
                and a["representative_goals"]==len(r["table"])
                and a["audited_actions"]==sum(len(v) for v in r["table"].values())
                and r["reference_kernel"]=={"ok":True,"axioms":[]}
                and digest(r["table"])==r["table_sha256"]):
            raise ValueError("uniform menu/graph evidence invalid")
        p=coefficients(r["table"],8)[r["goal"]]
        mass=sum(p,Fraction(0))
        if ([str(x) for x in p]!=r["exact_step_mass"] or str(mass)!=r["probability"]
                or (mass>0)!=(r["native_shortest_steps"]<=8)):
            raise ValueError("exact rational uniform recomputation mismatch")
        result[r["root"]]=mass
    mean=sum(result.values(),Fraction(0))/32
    if str(mean)!=m["expected_completion_exact"] or abs(float(mean)-m["expected_completion"])>1e-12:
        raise ValueError("uniform mean changed")
    return m,result


def compare(suite,legacy_suite,teacher,collection,replay,exposure,runs,transport,
            uniform_structural,uniform_legacy,out):
    if out.exists():
        raise ValueError("fresh LBF-10 report required")
    source=read(suite/"manifest.json")
    legacy=read(legacy_suite/"manifest.json")
    gate=read(exposure/"manifest.json")
    if not (source["seed"]==20261201 and source.get("structural_alt") is True
            and source.get("source_family")==FAMILY and not source.get("equal_alt_p",0)
            and legacy["seed"]==20261202 and not legacy.get("structural_alt",False)
            and not legacy.get("equal_alt_p",0)
            and gate["schema"]=="lbf10.native_structural_source_exposure.v1"
            and gate["gate"]=="MULTI_OPTIMAL_SOURCE_SUFFICIENT"
            and gate["source_native_verified"] and gate["source_independent_replay_verified"]
            and gate["suite_manifest_sha256"]==sha256_file(suite/"manifest.json")
            and gate["native_teacher_manifest_sha256"]==sha256_file(teacher/"manifest.json")
            and gate["collection_manifest_sha256"]==sha256_file(collection/"manifest.json")
            and gate["independent_replay_sha256"]==sha256_file(replay/"replay.json")
            and gate["model_visited_multi_fraction"]>=.25
            and gate["model_visited_roots_with_multi_optimal_witnesses"]>=12):
        raise ValueError("registered structural source/exposure not verified")
    train,_=load_suite(suite,"train",64)
    train_set={c.lineage for c in train}
    out.mkdir(parents=True)
    reports={}
    roster_sets=[]
    checkpoint_hashes={}
    for name,query,run_dir,uniform in (
        ("structural",suite,runs,uniform_structural),
        ("legacy",legacy_suite,transport,uniform_legacy),
    ):
        cases,src=load_suite(query,"calibration",32)
        expected={c.root:c.lineage for c in cases}
        if len(expected)!=32 or set(expected.values()) & train_set:
            raise ValueError("query root roster not independent of structural train source")
        roster_sets.append(set(expected.values()))
        u,uniform_probs=audited_uniform(query,uniform,expected)
        audit=native_compare(teacher,collection,replay/"replay.json",run_dir,out/(name+"_native_audit"))
        if not (audit["calibration_unique_roots"]==32 and audit["native_episode_replay_verified"]
                and audit["matched_initial_weights_and_draws"] and not audit["locked_test_opened"]):
            raise ValueError("native paired comparison receipt failed")
        cells=audit["cells"]
        for seed in SEEDS:
            one=cells[f"{seed}-visited_one"]; full=cells[f"{seed}-visited_set"]
            if (full["multi_positive_visited_updates"]<50
                    or full["multi_positive_visited_updates"]!=gate["seed_draws"][str(seed)]["multi_positive_visited_updates"]
                    or full["model_visited_updates"]!=one["model_visited_updates"]
                    or one["multi_positive_visited_updates"]!=0
                    or cells[f"{seed}-source_only"]["model_visited_updates"]!=0):
                raise ValueError("actual full-set training exposure differs from preregistration")
        for key,cell in cells.items():
            n=read(run_dir/key/"calibration"/"manifest.json")
            rr=rows(run_dir/key/"calibration"/"paired_roots.jsonl")
            if not (n["source"]["suite_manifest_sha256"]==src["suite_manifest_sha256"]
                    and n["check_budget"]==8 and n["depth"]==2
                    and {r["root"]:r["lineage"] for r in rr}==expected
                    and all(n["backend"][f]==u["backend"][f]
                            for f in ("lean_version","repl_bin_sha256","lean_bin_sha256"))
                    and cell["native_reference_budget_reachable"]==u["reference_reachable_within_budget"]):
                raise ValueError("native learner/uniform cohort custody mismatch")
            if key in checkpoint_hashes and checkpoint_hashes[key]!=cell["checkpoint_sha256"]:
                raise ValueError("transport refit or checkpoint substitution")
            checkpoint_hashes[key]=cell["checkpoint_sha256"]
            exact_rate=Fraction(cell["proofs"],32)
            exact_uniform=Fraction(u["expected_completion_exact"])
            cell["close_rate"]=float(exact_rate)
            cell["minus_uniform"]=float(exact_rate-exact_uniform)
            cell["descriptive_95_root_bootstrap_minus_uniform"]=root_bootstrap([
                float(Fraction(int(r["ab_certified"]))-uniform_probs[r["root"]])
                for r in sorted(rr,key=lambda x:x["root"])])
            cell["numerical_competence_gate"]=bool(exact_rate>=Fraction(2,5)
                                                  and exact_rate-exact_uniform>=Fraction(1,10))
        delta=audit["root_paired_effects"]["visited_set_minus_visited_one"]
        reports[name]={
            "suite_manifest_sha256":src["suite_manifest_sha256"],
            "uniform_baseline_manifest_sha256":sha256_file(uniform/"manifest.json"),
            "uniform_expected_completion":u["expected_completion"],
            "uniform_expected_completion_exact":u["expected_completion_exact"],
            "native_uniform_menu_audit_passed":True,
            "uniform_audited_goals":u["audited_goals"],
            "uniform_audited_actions":u["audited_public_actions"],
            "native_reference_budget_reachable":u["reference_reachable_within_budget"],
            "cells":cells,"root_paired_effects":audit["root_paired_effects"],
            "registered_action_set_development_pass":delta["mean"]>=.05,
        }
    if roster_sets[0] & roster_sets[1]:
        raise ValueError("structural and transport calibration lineages overlap")
    result={"schema":SCHEMA,"seeds":list(SEEDS),"roots_per_cohort":32,
            "exposure":gate,"source_suite_manifest_sha256":sha256_file(suite/"manifest.json"),
            "teacher_manifest_sha256":sha256_file(teacher/"manifest.json"),
            "cohorts":reports,
            "structural_replication_pass":reports["structural"]["registered_action_set_development_pass"],
            "legacy_transport_pass":reports["legacy"]["registered_action_set_development_pass"],
            "same_checkpoints_transport":True,"locked_test_opened":False,
            "architecture_promotion_eligible":False,"scientific_eligible":False,
            "interpretation":"Development-only; numerical competence is descriptive. Independent powered FSI-01 confirmation and matched sham remain required."}
    (out/"results.json").write_text(canonical(result)+"\n",encoding="utf-8")
    report=["# LBF-10 structural action sets and legacy transport", "",
            "Two matched training seeds, 32 independent development roots per cohort. Exact bounded native public-menu uniform expectation; no locked test scoring.",
            "", f"Multi-optimal visited frames: {gate['model_visited_multi_optimal_frames']}/{gate['model_visited_frames']}.",
            "", "| Cohort | Seed | Reference only | Visited single | Visited set | Uniform expected | Auto8 | Native reachable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in reports.items():
        for seed in SEEDS:
            a,b,c=[r["cells"][f"{seed}-{arm}"] for arm in ARMS]
            report.append(f"| {name} | {seed} | {a['proofs']}/32 | {b['proofs']}/32 | {c['proofs']}/32 | {r['uniform_expected_completion']:.3%} | {a['public_auto8_proofs']}/32 | {r['native_reference_budget_reachable']}/32 |")
    for name,r in reports.items():
        report.extend(["",f"## {name} root-paired effects",""])
        for key,delta in r["root_paired_effects"].items():
            report.append(f"- {key}: {delta['mean']:+.5f}, descriptive 95% {delta['descriptive_95_root_bootstrap']}, wins/losses/ties {delta['wins']}/{delta['losses']}/{delta['ties']}.")
        report.append(f"- Registered +.05 action-set screen: {'PASS' if r['registered_action_set_development_pass'] else 'MISS'}.")
    report.extend(["","No scientific promotion; FSI-01 independent confirmation remains outstanding.",""])
    (out/"RESULTS.md").write_text("\n".join(report),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("suite","legacy-suite","teacher","collection","replay","exposure","runs","transport","uniform-structural","uniform-legacy","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    r=compare(a.suite,a.legacy_suite,a.teacher,a.collection,a.replay,a.exposure,a.runs,
              a.transport,a.uniform_structural,a.uniform_legacy,a.out)
    for name,c in r["cohorts"].items():
        print("LBF10_COHORT",name,canonical(c),flush=True)
    print("LBF10_DECISION",r["structural_replication_pass"],r["legacy_transport_pass"],flush=True)


if __name__=="__main__":
    main()
