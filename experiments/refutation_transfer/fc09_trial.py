"""FC-09: CPU-time-matched recurrent proof actor versus symbolic AND/OR A*.

The per-case AND/OR time allowance is the model's actually measured
rule-scoring forward time, a SINGLE scalar, with no access to neural
scores, decoded action lists, gold or source reference proofs. A separate
source-only generator revision is mounted after training has finished.
All accepted source rule traces are independently replayed and the model's
positive proofs are kernel-compiled in a subsequent CI step.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fc03_compare as fc03
import fc04_adaptive as fc04
import fc06_joint as fc06

PREREG_SHA="37ceea8c6e5310fb9fb6fb43fd2aab1e9f3210a4"
TRAIN_SOURCE_SHA="315119ead7264fbfbf5e58ee1b7b5c8b512d475d"
TRAIN_DATA_SHA="680318ea334bba0777a048d3299a9e490207faa3a8ead549393616790a3761d5"
TEST_SOURCE_SHA="0b045fade18ca725ad9adc536ad48d6f58935fb3"
TEST_DATA_SHA="89719bf1af2927fbd957f6735d15e4ec30598696b173a3a365bcd39bbfd5440d"
SEEDS=(17,29)
METHODS=("learned_joint","andor_cpu_matched","hmax_24","fc08_24",
         "random_legal","goal_first","backward_slice","unary_bfs")
STATIC=("andor_cpu_matched","hmax_24","fc08_24",
        "backward_slice","unary_bfs")


def _sha_rows(rows,fc02):
    return hashlib.sha256(b"".join(fc02.canonical_bytes(r)+b"\n" for r in rows)).hexdigest()


def _load_train(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc09 import train_data as fc09
    from research.refutation_bench.formal_curriculum import generator as fc02
    data=fc09.generate_train_dev()
    fc09.audit_train_dev(data)
    h=_sha_rows(data,fc02)
    if h!=TRAIN_DATA_SHA:
        raise AssertionError("FC-09 training source digest changed: "+h)
    train=[r for r in data if r["split"]=="train"]
    dev=[r for r in data if r["split"]=="dev"]
    if (len(train),len(dev))!=(768,192):
        raise AssertionError("train/dev source count changed")
    return train,dev,h


def train(root:Path,out:Path):
    """No FC-09 test generator or symbolic control is accessible in this job."""
    torch.set_num_threads(2)
    train,dev,digest=_load_train(root)
    fc02train,fc02dev,_=fc03.load_source_training(root)
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.certificates import verify_countermodel
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        "experiment":"FC09",
        "prereg_sha":PREREG_SHA,
        "phase":"training_only_before_test_generator_checkout",
        "train_source_commit":TRAIN_SOURCE_SHA,
        "train_dataset_sha256":digest,
        "fc02_data_sha256":fc03.FC02_DATA_SHA,
        "test_generator_absent_in_training":True,
        "train_cases":len(train),"dev_cases":len(dev),
        "symbolic_time_budget_is_derived_only_from_frozen_neural_runtime":True,
        "model_seeds":list(SEEDS),
        "models":[],
    }
    for seed in SEEDS:
        gate,meta=fc03._train_one(fc02train,"tied4_width24",seed)
        threshold=fc04.calibrate_threshold(gate,fc02dev,verify_countermodel)
        actor,actor_info=fc06._train_actor(gate,train,seed)
        name=f"fc09_s{seed}"
        gatefile=out/(name+"_gate.pt")
        actorfile=out/(name+"_actor.pt")
        torch.save(gate.state_dict(),gatefile)
        torch.save(actor.state_dict(),actorfile)
        item={
            "tag":name,"seed":seed,
            "theta":threshold["selected_threshold"],
            "theta_grid":threshold["all_grid_scores"],
            "gate_parameters":sum(p.numel() for p in gate.parameters()),
            "gate_sha256":hashlib.sha256(gatefile.read_bytes()).hexdigest(),
            "actor_sha256":hashlib.sha256(actorfile.read_bytes()).hexdigest(),
            "gate_train_seconds":meta["train_seconds"],
            **actor_info,
        }
        manifest["models"].append(item)
        print("FC09_FROZEN",json.dumps({key:item[key] for key in (
            "tag","theta","train_prefix_states","train_first128_accuracy",
            "actor_parameters","gate_sha256","actor_sha256")},sort_keys=True),flush=True)
    file=out/"FC09_train_manifest.json"
    file.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("FC09_TRAIN_RECEIPT",file)


def _load_test(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc09 import test_data as fc09
    from research.refutation_bench.fc09 import controls
    from research.refutation_bench.fc08 import controls as old_controls
    from research.refutation_bench.fc05 import proofs
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    cases=fc09.generate_tests()
    fc09.audit_tests(cases)
    digest=_sha_rows(cases,fc02)
    if digest!=TEST_DATA_SHA:
        raise AssertionError("frozen test data digest mismatch: "+digest)
    return cases,controls,old_controls,proofs,verify_countermodel,digest


def _symbolic_result(task:dict,pre:dict,method:str,
                     suggestion:dict,proofs)->dict:
    """Exact replay of original rule IDs; no shortest proof oracle repairs."""
    import time
    result={
        "case_id":task["case_id"],"method":method,
        "outcome":pre["outcome"],"preproof_outcome":pre["outcome"],
        "preproof_reason":pre["reason"],"stop_passes":pre["passes"],
        "stop_model_ms":pre["model_forward_ms"],
        "negative_checker_ms":pre["checker_ms"],
        "negative_checker_calls":pre["checker_calls"],
        "negative_countermodel":pre["certificate"] if pre["outcome"]=="refuted_certified" else None,
        "positive_trace":None,"proposal_trace":None,
        "positive_checked":False,"positive_checker_ms":0.,"positive_checker_calls":0,
        "actor_calls":0,"actor_recurrent_cells":0,"actor_forward_ms":0.,
        "rule_actions":0,"proof_failure_reason":None,
        "symbolic_search_ms":0.,"symbolic_states_popped":0,
        "symbolic_allowed_ms":None,"symbolic_budget_overrun_ms":0.,
        "symbolic_timed_out":False,
    }
    if pre["outcome"]!="established_provisional":
        return result
    start=time.perf_counter()
    valid=proofs.verify_trace(task,suggestion["trace"],budget=5)
    verify_ms=1000*(time.perf_counter()-start)
    result.update({
      "proposal_trace":suggestion["trace"],
      "positive_trace":suggestion["trace"] if valid["verified"] else None,
      "positive_checked":bool(valid["verified"]),
      "positive_checker_calls":1,
      "positive_checker_ms":round(verify_ms,6),
      "rule_actions":len(suggestion["trace"]),
      "outcome":"established_certified" if valid["verified"] else "abstain",
      "proof_failure_reason":None if valid["verified"] else valid["reason"],
      "symbolic_search_ms":suggestion.get("search_ms",suggestion.get("symbolic_search_ms",0.)),
      "symbolic_states_popped":suggestion.get("states_popped",0),
      "symbolic_allowed_ms":suggestion.get("allowed_ms"),
      "symbolic_budget_overrun_ms":suggestion.get("budget_overrun_ms",0.),
      "symbolic_timed_out":suggestion.get("timed_out",False),
    })
    return result


def _score(tasks,outputs):
    base=fc06._metrics(tasks,outputs)
    graph_ms=sum(r.get("symbolic_search_ms",0.) for r in outputs)
    base["symbolic_search_ms_per_case"]=round(graph_ms/len(outputs),6)
    base["total_cpu_wall_ms_per_case"]=round(
        base["model_and_checker_ms_per_case"]+graph_ms/len(outputs),6)
    base["symbolic_pops"]=sum(r.get("symbolic_states_popped",0) for r in outputs)
    base["symbolic_timeouts"]=sum(bool(r.get("symbolic_timed_out")) for r in outputs)
    base["symbolic_overrun_ms_total"]=round(sum(r.get(
        "symbolic_budget_overrun_ms",0.) for r in outputs),6)
    eligible=[r for r in outputs if r["preproof_outcome"]=="established_provisional"]
    allowed=sum(r.get("symbolic_allowed_ms") or 0 for r in eligible)
    actual=sum(r.get("symbolic_search_ms",0.) for r in eligible)
    base["symbolic_allowed_ms_total"]=round(allowed,6)
    base["symbolic_actual_over_allowed_ratio"]=round(actual/allowed,6) if allowed>0 else None
    base["mean_candidate_actions"]=round(sum(r["rule_actions"] for r in outputs)/len(outputs),6)
    base["failure_reasons"]=dict(Counter(
        r.get("proof_failure_reason") or "none" for r in eligible
        if r["outcome"]=="abstain"))
    base["model_rule_forward_ms_per_case"]=round(
        sum(r["actor_forward_ms"] for r in outputs)/len(outputs),6)
    return base


def _paired(tasks,neural,control):
    model_only=control_only=both=0
    for task,p,q in zip(tasks,neural,control):
        if not(task["case_id"]==p["case_id"]==q["case_id"]):
            raise AssertionError("paired case reference mismatch")
        if task["label"]!="established":
            continue
        p_ok=p["outcome"]=="established_certified"
        q_ok=q["outcome"]=="established_certified"
        model_only+=int(p_ok and not q_ok)
        control_only+=int(q_ok and not p_ok)
        both+=int(p_ok and q_ok)
    return {"learned_only":model_only,"symbolic_only":control_only,
            "both_certified":both,"net_neural":model_only-control_only}


def evaluate(root:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifestfile=weights/"FC09_train_manifest.json"
    manifest=json.loads(manifestfile.read_text())
    if (manifest.get("prereg_sha")!=PREREG_SHA or
        manifest.get("train_dataset_sha256")!=TRAIN_DATA_SHA or
        manifest.get("test_generator_absent_in_training") is not True):
        raise AssertionError("invalid frozen train manifest or data boundary")
    if {x["seed"] for x in manifest["models"]}!=set(SEEDS):
        raise AssertionError("unexpected model seed or arm")
    cases,controls,old_controls,proofs,verify,source_hash=_load_test(root)
    out.mkdir(parents=True,exist_ok=True)
    accepted=[]
    scores=[]
    contrasts=[]
    for meta in manifest["models"]:
        seed=meta["seed"];tag=meta["tag"]
        gatepath=weights/(tag+"_gate.pt")
        actorpath=weights/(tag+"_actor.pt")
        if hashlib.sha256(gatepath.read_bytes()).hexdigest()!=meta["gate_sha256"]:
            raise AssertionError("gate checkpoint tampered")
        if hashlib.sha256(actorpath.read_bytes()).hexdigest()!=meta["actor_sha256"]:
            raise AssertionError("actor checkpoint tampered")
        gate=fc03.make_model("tied4_width24")
        gate.load_state_dict(torch.load(gatepath,weights_only=True,map_location="cpu"))
        gate.eval()
        actor=fc06.JointActor()
        actor.load_state_dict(torch.load(actorpath,weights_only=True,map_location="cpu"))
        actor.eval()
        theta=meta["theta"]
        for split in ("test_same","test_deep"):
            tasks=[x for x in cases if x["split"]==split]
            # Common gate cached once across all policies. Gold is not an input.
            pre=[fc04.run_one(gate,t,theta,"adaptive",verify) for t in tasks]
            learned=[fc06.evaluate_condition(gate,actor,t,theta,"learned_joint",
                            proofs,verify,p) for t,p in zip(tasks,pre)]
            outcomes={"learned_joint":learned}
            for method in METHODS[1:]:
                output=[]
                for task,g,neural in zip(tasks,pre,learned):
                    if g["outcome"]!="established_provisional":
                        suggested={"trace":[],"search_ms":0.,
                                   "states_popped":0,"allowed_ms":None}
                    elif method=="andor_cpu_matched":
                        # The only learned-model datum supplied to the
                        # symbolic planner is its measured rule-scoring time.
                        allowed=float(neural["actor_forward_ms"])
                        suggested=controls.hmax_search(
                            task,budget=5,max_popped=None,
                            max_runtime_ms=allowed)
                    elif method=="hmax_24":
                        suggested=controls.hmax_search(
                            task,budget=5,max_popped=24)
                    elif method=="fc08_24":
                        suggested=old_controls.baseline_trace(task,"bounded_best_first")
                    else:
                        suggested=old_controls.baseline_trace(task,method)
                    output.append(_symbolic_result(task,g,method,suggested,proofs))
                outcomes[method]=output
            for method,pred in outcomes.items():
                metrics=_score(tasks,pred)
                groups={}
                for family in sorted({t["family_id"] for t in tasks}):
                    indices=[i for i,t in enumerate(tasks) if t["family_id"]==family]
                    groups[family]=_score([tasks[i] for i in indices],
                                          [pred[i] for i in indices])
                file=out/(f"{tag}_{split}_{method}.jsonl")
                file.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in pred))
                metrics.update(seed=seed,tag=tag,split=split,method=method,
                    threshold=theta,by_family=groups,
                    response_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                scores.append(metrics)
                if method=="learned_joint":
                    accepted.extend((t,r["positive_trace"]) for t,r in zip(tasks,pred)
                                    if r["outcome"]=="established_certified")
                print("FC09_SCORE",json.dumps({key:metrics[key] for key in (
                    "seed","split","method","eligible_positive",
                    "certified_positive","certified_negative","abstentions",
                    "total_recurrent_cells","symbolic_pops",
                    "total_cpu_wall_ms_per_case")},sort_keys=True),flush=True)
            for method in METHODS[1:]:
                contrasts.append({"seed":seed,"split":split,"against":method,
                                  **_paired(tasks,learned,outcomes[method])})
    leanfile=out/"FC09_model_proofs.lean"
    leanfile.write_text(proofs.lean_document(accepted),encoding="utf8")
    primary=[]
    for seed in SEEDS:
        for split in ("test_same","test_deep"):
            def score(method):
                return next(s for s in scores if s["seed"]==seed and
                            s["split"]==split and s["method"]==method)
            model=score("learned_joint")
            matched=score("andor_cpu_matched")
            model_cost=(model["total_cpu_wall_ms_per_case"]*192)/max(1,model["certified_positive"])
            symbolic_cost=(matched["total_cpu_wall_ms_per_case"]*192)/max(1,matched["certified_positive"])
            primary.append({
                "seed":seed,"split":split,
                "neural_positive":model["certified_positive"],
                "cpu_matched_symbolic_positive":matched["certified_positive"],
                "learner_beats_symbolic":model["certified_positive"]>matched["certified_positive"],
                "neural_ms_per_positive":round(model_cost,6),
                "symbolic_ms_per_positive":round(symbolic_cost,6),
                "neural_cheaper_per_certificate":model_cost<symbolic_cost,
                "exact_no_false_certificate":model["false_formal_certificates"]==0,
                "symbolic_budget_overrun_ms_total":matched["symbolic_overrun_ms_total"],
                "symbolic_actual_to_allowed_time_ratio":matched["symbolic_actual_over_allowed_ratio"],
            })
    report={
        "experiment":"FC09_first_preregistered_CPU_matched_ANDOR",
        "preregistration_sha":PREREG_SHA,
        "train_source_commit":TRAIN_SOURCE_SHA,
        "train_data_sha256":TRAIN_DATA_SHA,
        "test_source_commit":TEST_SOURCE_SHA,
        "test_data_sha256":source_hash,
        "train_manifest_sha256":hashlib.sha256(manifestfile.read_bytes()).hexdigest(),
        "first_frozen_test":True,
        "positive_kernel_theorems":len(accepted),
        "model_lean_source_sha256":hashlib.sha256(leanfile.read_bytes()).hexdigest(),
        "learning_criterion_passed":all(p["learner_beats_symbolic"] and
                     p["exact_no_false_certificate"] for p in primary),
        "cpu_efficiency_criterion_passed":all(p["neural_cheaper_per_certificate"]
                     for p in primary),
        "main":primary,"scores":scores,"paired":contrasts,
        "scope":"12-atom public Horn graphs; exact oracle always solves formal entailments",
    }
    (out/"FC09_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC09_CRITERIA",json.dumps({
        "learning":report["learning_criterion_passed"],
        "compute":report["cpu_efficiency_criterion_passed"],
        "model_positive_proofs":report["positive_kernel_theorems"]},sort_keys=True),flush=True)
    print("FC09_REPORT",out/"FC09_report.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="train":
        if args.weights is not None:
            raise ValueError("training cannot read test weight/output artifacts")
        train(args.source,args.out)
    else:
        if args.weights is None:
            raise ValueError("evaluation requires frozen trained weights")
        evaluate(args.source,args.weights,args.out)


if __name__=="__main__":
    main()
