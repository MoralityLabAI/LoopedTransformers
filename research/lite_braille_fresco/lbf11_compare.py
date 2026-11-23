"""LBF-11 source-locked native proof factorial, with rational uniform baseline."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

LCGRT=Path(__file__).resolve().parents[1]/"lean_cgrt_hrm400"
if str(LCGRT) not in sys.path: sys.path.insert(0,str(LCGRT))

from ppt_gym.cli import checkpoint_load
from ppt_gym.data import canonical,sha256_file
from ppt_gym.lbf08_repair import frame_schedule
from ppt_gym.lbf11_loss import (
    ARMS,TERMS,SCHEMA,OBJECTIVE_SCHEMA,verified_source,_sha_rows,
)
from ppt_gym import lbf11_loss
from ppt_gym.live_suite import load_suite
from ppt_gym.native_train import _model_fingerprint
from lbf10_compare import audited_uniform
from lbf08_compare import root_bootstrap

SEEDS=(107,109)


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def compare(suite,teacher,collection,behavior,replay,exposure,runs,uniform,diagnostic,out):
    if out.exists(): raise ValueError("report path must be new")
    refs,source,visits,repair,gate=verified_source(suite,teacher,collection,behavior,replay,exposure)
    replay_sha=sha256_file(replay)
    source_sha=sha256_file(teacher/"manifest.json")
    repair_sha=sha256_file(collection/"manifest.json")
    exp_sha=sha256_file(exposure/"manifest.json")
    suite_sha=sha256_file(suite/"manifest.json")
    impl_sha=sha256_file(Path(lbf11_loss.__file__))
    d=read(diagnostic/"manifest.json")
    if not (d["schema"]=="lbf11.frozen_train_only_diagnostic.v1"
            and d["source_kind"]=="fresh_behavior" and d["teacher_manifest_sha256"]==source_sha
            and d["collection_manifest_sha256"]==repair_sha and d["replay_sha256"]==replay_sha
            and d["train_only"] and d["optimizer_updates"]==0
            and d["calibration_outcomes_read"]==0 and not d["locked_test_opened"]
            and d["models"]["fresh_behavior"]["checkpoint_sha256"]==sha256_file(behavior)
            and d["models"]["fresh_behavior"]["frames"]==gate["model_visited_multi_optimal_frames"]
            and d["models"]["fresh_behavior"]["frame_metrics_sha256"]==sha256_file(diagnostic/"fresh_behavior.jsonl")):
        raise ValueError("fresh frozen diagnostic missing or changed")
    cases,src=load_suite(suite,"calibration",32)
    expected={c.root:c.lineage for c in cases}
    train,_=load_suite(suite,"train",64)
    if len(expected)!=32 or set(expected.values()) & {c.lineage for c in train}:
        raise ValueError("calibration lineage roster changed")
    u,probs=audited_uniform(suite,uniform,expected)
    uniform_mean=Fraction(u["expected_completion_exact"])
    sr,vr=defaultdict(list),defaultdict(list)
    for f in refs: sr[f["lineage"]].append(f)
    for f in visits: vr[f["lineage"]].append(f)
    repeat=read(runs/"determinism.json")
    if not (repeat.get("same_weights") is True and repeat.get("same_optimizer_trace") is True
            and repeat.get("seed")==107 and repeat.get("arm")=="b1_r1"):
        raise ValueError("deterministic endpoint repeat failed")
    cells={}; scores={}
    for seed in SEEDS:
        schedule=frame_schedule(refs,visits,seed,720)
        seed_init=set(); visitor_packets=set()
        for arm in ARMS:
            key=f"{seed}-{arm}"; folder=runs/key; fit=folder/"fit"
            fm=read(fit/"manifest.json"); ckpt=fit/"checkpoint.pt"
            model,ck=checkpoint_load(ckpt)
            if not (
                fm["schema"]==SCHEMA and fm["objective_schema"]==OBJECTIVE_SCHEMA
                and fm["objective_impl_sha256"]==impl_sha
                and fm["seed"]==seed and fm["arm"]==arm and fm["updates"]==720
                and fm["parameter_count"]==57475 and model.parameter_count()==57475
                and fm["source_roots"]==48 and fm["train_order"]=="ab" and fm["depth"]==2
                and fm["input_view"]=="hybrid_rule_intact" and fm["threads"]==1 and fm["learning_rate"]==.001
                and fm["calibration_updates"]==0 and not fm["locked_test_opened"]
                and fm["source_suite_manifest_sha256"]==suite_sha
                and fm["native_teacher_manifest_sha256"]==source_sha
                and fm["repair_source_manifest_sha256"]==repair_sha
                and fm["independent_replay_sha256"]==replay_sha
                and fm["exposure_manifest_sha256"]==exp_sha
                and fm["checkpoint_sha256"]==sha256_file(ckpt)
                and fm["optimizer_steps_sha256"]==sha256_file(fit/"optimizer_steps.jsonl")
                and fm["final_weights_sha256"]==_model_fingerprint(model)
                and ck["lbf11_loss_arm"]==arm and ck["lbf11_objective_schema"]==OBJECTIVE_SCHEMA
                and ck["lbf11_impl_sha256"]==impl_sha and ck["lbf11_exposure_sha256"]==exp_sha
                and ck["lbf11_packet_roster_sha256"]==fm["packet_roster_sha256"]
                and ck["native_teacher_sha256"]==source_sha and ck["source_manifest_sha256"]==suite_sha
                and ck["lbf08_repair_source_manifest_sha256"]==repair_sha
                and ck["lbf08_independent_replay_sha256"]==replay_sha
                and ck["seed"]==seed and ck["train_steps"]==720 and ck["train_order"]=="ab"
                and fm["draw_schedule_sha256"]==_sha_rows(schedule)==gate["seed_draws"][str(seed)]["draw_schedule_sha256"]
            ): raise ValueError(key+": loss checkpoint/source provenance mismatch")
            draw=rows(fit/"optimizer_steps.jsonl")
            if len(draw)!=720: raise ValueError(key+": missing optimizer records")
            packets=[]; counts=defaultdict(int)
            for row,log in zip(schedule,draw):
                use=bool(arm!="source_only" and row["use_model_visited"])
                example=vr[row["lineage"]][row["visit_index"]] if use else sr[row["lineage"]][row["source_index"]]
                packet={**row,"chosen_source":"model_visited" if use else "native_reference",
                        "packet_sha256":hashlib.sha256(canonical([example["goal"],example["candidates"],
                                                                 example["checks_remaining"]]).encode()).hexdigest(),
                        "native_positive_count":sum(example["labels"]),"candidate_count":len(example["labels"])}
                a,b=TERMS[arm] if use else ("bs","rs")
                if (any(log.get(k)!=v for k,v in packet.items()) or log["outcome_target"]!=a
                        or log["rank_target"]!=b or not math.isfinite(log["loss"])
                        or not math.isfinite(log["preclip_gradient_norm"])):
                    raise ValueError(key+": actual packets or targets differ from registered draws")
                packets.append(packet)
                if use:
                    counts["visited"]+=1
                    if packet["native_positive_count"]>1:
                        counts["multi"]+=1
                        counts["outcome_multi"]+=int(a=="bs")
                        counts["rank_multi"]+=int(b in ("rs","rb"))
            plan=gate["seed_draws"][str(seed)]
            if not (fm["packet_roster_sha256"]==_sha_rows(packets)
                    and counts["visited"]==fm["model_visited_updates"]
                    and counts["multi"]==fm["native_multi_positive_visited_updates"]
                    and counts["outcome_multi"]==fm["outcome_multi_positive_visited_updates"]
                    and counts["rank_multi"]==fm["rank_multi_positive_visited_updates"]
                    and (arm=="source_only" or (counts["visited"]==plan["model_visited_updates"]
                         and counts["multi"]==plan["multi_positive_visited_updates"]>=50))):
                raise ValueError(key+": multi-positive treatment exposure differs")
            seed_init.add(fm["initial_weights_sha256"])
            if arm!="source_only": visitor_packets.add(fm["packet_roster_sha256"])

            native_dir=folder/"calibration"; n=read(native_dir/"manifest.json")
            rp=read(folder/"replay"/"replay.json")
            provenance=n["training_provenance"]
            if not (n["schema"]=="lsa02b.native_block_order.v1" and n["status"]=="complete"
                    and n["split"]=="calibration" and n["paired_roots"]==32
                    and n["source"]["suite_manifest_sha256"]==suite_sha and n["check_budget"]==8 and n["depth"]==2
                    and n["backend"]["actual_lean_process"] is True
                    and all(n["backend"][k]==u["backend"][k] for k in ("lean_version","repl_bin_sha256","lean_bin_sha256"))
                    and n["checkpoint_sha256"]==fm["checkpoint_sha256"]
                    and provenance["training_source_verified"] and provenance["source_lineages_disjoint"]
                    and provenance["training_lineages"]==48 and provenance["native_teacher_manifest_sha256"]==source_sha
                    and n["standalone_reference_certificates"]==32
                    and n["reference_reachable_within_budget"]==u["reference_reachable_within_budget"]
                    and n["query_label_training_updates"]==0 and n["architecture_promotion_eligible"] is False
                    and rp["replay_passed"] and rp["independent_replay_episodes"]==96
                    and rp["session_manifest_sha256"]==sha256_file(native_dir/"manifest.json")):
                raise ValueError(key+": native calibration/replay provenance mismatch")
            for filename,sha in n["artifacts"].items():
                if Path(filename).name!=filename or sha256_file(native_dir/filename)!=sha:
                    raise ValueError(key+": native receipt changed")
            rr=rows(native_dir/"paired_roots.jsonl")
            if len(rr)!=32 or {x["root"]:x["lineage"] for x in rr}!=expected:
                raise ValueError(key+": root roster mismatch")
            outcomes={r["root"]:int(r["ab_certified"]) for r in rr}
            scores[key]=outcomes; total=sum(outcomes.values())
            if total!=n["scores"]["ab"]["standalone_verified_closures"]:
                raise ValueError(key+": proof count mismatch")
            rate=Fraction(total,32)
            cells[key]={
                "proofs":total,"close_rate":float(rate),"minus_uniform":float(rate-uniform_mean),
                "numerical_competence_gate":rate>=Fraction(2,5) and rate-uniform_mean>=Fraction(1,10),
                "descriptive_95_minus_uniform":root_bootstrap([float(Fraction(outcomes[r])-probs[r]) for r in sorted(expected)]),
                "tactic_checks":n["scores"]["ab"]["tactic_checks"],"model_calls":n["scores"]["ab"]["model_calls"],
                "auto8":n["scores"]["auto8"]["standalone_verified_closures"],
                "visited_updates":counts["visited"],"native_multi_updates":counts["multi"],
                "outcome_multi_updates":counts["outcome_multi"],"rank_multi_updates":counts["rank_multi"],
                "mean_preclip_gradient_norm":fm["mean_preclip_gradient_norm"],
                "clipped_updates":fm["clipped_updates"],"checkpoint_sha256":fm["checkpoint_sha256"],
            }
        if len(seed_init)!=1 or len(visitor_packets)!=1:
            raise ValueError("matched seed initialization or visited packet roster changed")
    if len({v["auto8"] for v in cells.values()})!=1:
        raise ValueError("public auto8 baseline changed")
    # Two seed contrasts are repeated within each independent theorem.
    contrasts={
        "outcome_main":{"bs_r1":.5,"b1_r1":-.5,"bs_rs":.5,"b1_rs":-.5},
        "ranking_main":{"b1_rs":.5,"b1_r1":-.5,"bs_rs":.5,"bs_r1":-.5},
        "interaction":{"bs_rs":1,"bs_r1":-1,"b1_rs":-1,"b1_r1":1},
        "balanced_rescue":{"bs_balanced":1,"bs_rs":-1},
        "full_minus_single":{"bs_rs":1,"b1_r1":-1},
    }
    for arm in ARMS:
        if arm!="source_only": contrasts[arm+"_minus_source"]={arm:1,"source_only":-1}
    effects={}
    for name,terms in contrasts.items():
        vals=[sum(sum(weight*scores[f"{seed}-{arm}"][root] for arm,weight in terms.items())
                  for seed in SEEDS)/2 for root in sorted(expected)]
        effects[name]={"mean":sum(vals)/32,"descriptive_95_root_bootstrap":root_bootstrap(vals),
                       "wins":sum(v>0 for v in vals),"losses":sum(v<0 for v in vals),"ties":sum(v==0 for v in vals)}
    result={"schema":"lbf11.native_loss_factorial_results.v1","roots":32,"seeds":list(SEEDS),
            "arms":list(ARMS),"source_exposure":gate,"uniform_expected":float(uniform_mean),
            "uniform_expected_exact":str(uniform_mean),"reference_reachable":u["reference_reachable_within_budget"],
            "uniform_audited_goals":u["audited_goals"],"uniform_audited_actions":u["audited_public_actions"],
            "cells":cells,"effects":effects,
            "outcome_adverse_screen":effects["outcome_main"]["mean"]<=-.05,
            "ranking_adverse_screen":effects["ranking_main"]["mean"]<=-.05,
            "balanced_rescue_screen":effects["balanced_rescue"]["mean"]>=.05,
            "matched_source_packet_rosters":True,"same_worker_repeat":True,"native_replay_verified":True,
            "source_manifest_sha256":source_sha,"diagnostic_manifest_sha256":sha256_file(diagnostic/"manifest.json"),
            "uniform_manifest_sha256":sha256_file(uniform/"manifest.json"),
            "locked_test_opened":False,"scientific_eligible":False,
            "architecture_promotion_eligible":False}
    out.mkdir(parents=True)
    (out/"results.json").write_text(canonical(result)+"\n",encoding="utf-8")
    report=["# LBF-11 native outcome/ranking factorial","",
            "Development-only. The twelve checkpoints share 32 independent calibration roots.","",
            f"Exact uniform expectation: {float(uniform_mean):.3%}. Native attainable {result['reference_reachable']}/32.","",
            "| Seed | Reference | B1 R1 | BS R1 | B1 RS | BS RS | BS balanced |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        report.append("| "+str(seed)+" | "+" | ".join(str(cells[f"{seed}-{a}"]["proofs"])+"/32" for a in ARMS)+" |")
    report.extend(["","## Root-paired effects",""])
    for k,v in effects.items():
        report.append(f"- {k}: {v['mean']:+.6f}; descriptive 95% {v['descriptive_95_root_bootstrap']}; wins/losses/ties {v['wins']}/{v['losses']}/{v['ties']}.")
    report.extend(["","No sealed evaluation, multiplicity-adjusted inference or scientific promotion.",""])
    (out/"RESULTS.md").write_text("\n".join(report),encoding="utf-8")
    print("LBF11_RESULTS",canonical(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("suite","teacher","collection","behavior","replay","exposure","runs","uniform","diagnostic","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    compare(a.suite,a.teacher,a.collection,a.behavior,a.replay,a.exposure,a.runs,a.uniform,a.diagnostic,a.out)


if __name__=="__main__": main()
