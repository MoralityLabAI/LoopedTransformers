"""FC-07: preregistered misleading-ancestry proof selection with LR-01.

Train/validate only on a pinned FC-07 TRAIN-ONLY source checkout which has no
test_data.py. Download frozen checkpoints to a second job, and only then mount
the separate FC-07 test generator. Independently verify both certificate
polarities and compile the model's original accepted Lean proof traces.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fc03_compare as fc03
import fc04_adaptive as fc04
import fc06_joint as fc06

PREREG_SHA="d3216371ae480b7e1b42cb913a33847ed5ad47c2"
FC07_TRAIN_SOURCE_SHA="f470a97502b853addf2b0f67c9f6225001074a90"
FC07_TRAIN_DATA_SHA="f725c31a18aa84b2b831b90d8d2f6ca62c2d20d40530b8f8284429d8c80e1359"
FC07_TEST_SOURCE_SHA="__FC07_TEST_SOURCE__"
FC07_TEST_DATA_SHA="__FC07_TEST_DATA__"
SEEDS=(17,29)
BUDGET=5
METHODS=("learned_joint","random_legal","goal_first","backward_slice")


def load_train_only(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc07 import train_data
    from research.refutation_bench.formal_curriculum import generator as fc02
    records=train_data.generate_train_dev()
    train_data.audit_train_dev(records)
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in records)).hexdigest()
    if digest!=FC07_TRAIN_DATA_SHA:
        raise AssertionError("train source hash changed "+digest)
    train=[x for x in records if x["split"]=="train"]
    dev=[x for x in records if x["split"]=="dev"]
    if (len(train),len(dev))!=(768,192):
        raise AssertionError("training source split differs from prereg")
    return train,dev,digest


def train(root:Path,out:Path):
    torch.set_num_threads(2)
    training,development,digest=load_train_only(root)
    fc02_train,fc02_dev,_=fc03.load_source_training(root)
    from research.refutation_bench.certificates import verify_countermodel
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        "experiment":"FC-07",
        "phase":"train_frozen_without_test_generator",
        "prereg_sha":PREREG_SHA,
        "fc07_train_source_sha":FC07_TRAIN_SOURCE_SHA,
        "fc07_train_data_sha256":digest,
        "fc02_data_sha256":fc03.FC02_DATA_SHA,
        "test_data_accessible_at_training":False,
        "train_size":len(training),
        "development_size":len(development),
        "seed_list":list(SEEDS),
        "actor_train_steps":fc06.ACTOR_STEPS,
        "proof_rule_budget":BUDGET,
        "weights":[],
    }
    for seed in SEEDS:
        gate,gate_meta=fc03._train_one(fc02_train,"tied4_width24",seed)
        theta=fc04.calibrate_threshold(gate,fc02_dev,verify_countermodel)
        actor,actor_meta=fc06._train_actor(gate,training,seed)
        name=f"fc07_s{seed}"
        gatefile=out/(name+"_gate.pt")
        actorfile=out/(name+"_actor.pt")
        torch.save(gate.state_dict(),gatefile)
        torch.save(actor.state_dict(),actorfile)
        details={
            "tag":name,"seed":seed,
            "theta":theta["selected_threshold"],
            "theta_grid_scores":theta["all_grid_scores"],
            "gate_sha256":hashlib.sha256(gatefile.read_bytes()).hexdigest(),
            "actor_sha256":hashlib.sha256(actorfile.read_bytes()).hexdigest(),
            "gate_parameters":sum(p.numel() for p in gate.parameters()),
            "gate_training_seconds":gate_meta["train_seconds"],
            **actor_meta,
        }
        manifest["weights"].append(details)
        print("FROZEN_FC07",json.dumps({k:details[k] for k in (
            "tag","theta","train_prefix_states","train_first128_accuracy",
            "actor_parameters","gate_sha256","actor_sha256")},sort_keys=True),flush=True)
    file=out/"train_manifest.json"
    file.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("FC07_TRAIN_RECEIPT",file)


def load_frozen_tests(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc07 import test_data
    from research.refutation_bench.fc05 import proofs
    from research.refutation_bench.fc06.posthoc_backward_slice import priority_trace
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    tasks=test_data.generate_tests()
    test_data.audit_tests(tasks)
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in tasks)).hexdigest()
    if digest!=FC07_TEST_DATA_SHA:
        raise AssertionError("frozen test SHA-256 changed: "+digest)
    return tasks,proofs,verify_countermodel,priority_trace,digest


def _ancestry_output(task,pre,proofs,priority_trace):
    """Exact FC-06 syntactic backward-slice baseline, charged its CPU time."""
    item={
      "case_id":task["case_id"],"method":"backward_slice",
      "outcome":pre["outcome"],"preproof_outcome":pre["outcome"],
      "preproof_reason":pre["reason"],"stop_passes":pre["passes"],
      "stop_model_ms":pre["model_forward_ms"],
      "negative_checker_ms":pre["checker_ms"],
      "negative_checker_calls":pre["checker_calls"],
      "negative_countermodel":pre["certificate"] if pre["outcome"]=="refuted_certified" else None,
      "positive_trace":None,"proposal_trace":None,"positive_checked":False,
      "positive_checker_ms":0.,"positive_checker_calls":0,
      "actor_calls":0,"actor_recurrent_cells":0,
      "actor_forward_ms":0.,"rule_actions":0,
      "proof_failure_reason":None,"backward_slice_ms":0.,
    }
    if pre["outcome"]!="established_provisional":
        return item
    start=time.perf_counter()
    trace=priority_trace(task,budget=BUDGET)
    scan_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter()
    checked=proofs.verify_trace(task,trace,budget=BUDGET)
    verify_ms=1000*(time.perf_counter()-start)
    item.update({
      "positive_trace":trace if checked["verified"] else None,
      "proposal_trace":trace,
      "positive_checked":bool(checked["verified"]),
      "positive_checker_ms":round(verify_ms,6),
      "positive_checker_calls":1,
      "rule_actions":len(trace),
      "proof_failure_reason":None if checked["verified"] else checked["reason"],
      "outcome":"established_certified" if checked["verified"] else "abstain",
      "backward_slice_ms":round(scan_ms,6),
    })
    return item


def _metrics(tasks,outputs):
    base=fc06._metrics(tasks,outputs)
    baseline_ms=sum(x.get("backward_slice_ms",0.) for x in outputs)
    base["symbolic_ancestry_ms_per_case"]=round(baseline_ms/len(outputs),6)
    base["total_measured_cpu_ms_per_case"]=round(
        base["model_and_checker_ms_per_case"]+baseline_ms/len(outputs),6)
    base["failed_trace_reasons"]=dict(Counter(
        x.get("proof_failure_reason") or "none" for x in outputs
        if x["preproof_outcome"]=="established_provisional" and
           x["outcome"]=="abstain"))
    base["proposed_rule_count"]=sum(x["rule_actions"] for x in outputs)
    base["proof_attempt_count"]=sum(x["positive_checker_calls"] for x in outputs)
    return base


def _pair(tasks,learned,base):
    win=lose=0
    for task,a,b in zip(tasks,learned,base):
        if not (task["case_id"]==a["case_id"]==b["case_id"]):
            raise AssertionError("paired case identities changed")
        if task["label"]!="established":
            continue
        success_a=a["outcome"]=="established_certified"
        success_b=b["outcome"]=="established_certified"
        win+=int(success_a and not success_b)
        lose+=int(success_b and not success_a)
    return {"learned_only":win,"baseline_only":lose,"net_learned":win-lose}


def evaluate(root:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifestfile=weights/"train_manifest.json"
    manifest=json.loads(manifestfile.read_text())
    if manifest["prereg_sha"]!=PREREG_SHA or manifest["fc07_train_data_sha256"]!=FC07_TRAIN_DATA_SHA:
        raise AssertionError("frozen manifest source mismatch")
    if manifest.get("test_data_accessible_at_training") is not False:
        raise AssertionError("train had test access")
    if {x["seed"] for x in manifest["weights"]}!=set(SEEDS):
        raise AssertionError("seed set changed")
    tasks,proofs,verify,priority_trace,test_digest=load_frozen_tests(root)
    out.mkdir(parents=True,exist_ok=True)
    report_items=[]
    pairs=[]
    accepted_model_proofs=[]
    for data in manifest["weights"]:
        seed=data["seed"];tag=data["tag"]
        gatefile=weights/(tag+"_gate.pt")
        actorfile=weights/(tag+"_actor.pt")
        if hashlib.sha256(gatefile.read_bytes()).hexdigest()!=data["gate_sha256"]:
            raise AssertionError("gate checkpoint digest mismatch")
        if hashlib.sha256(actorfile.read_bytes()).hexdigest()!=data["actor_sha256"]:
            raise AssertionError("actor checkpoint digest mismatch")
        gate=fc03.make_model("tied4_width24")
        gate.load_state_dict(torch.load(gatefile,map_location="cpu",weights_only=True))
        gate.eval()
        actor=fc06.JointActor()
        actor.load_state_dict(torch.load(actorfile,map_location="cpu",weights_only=True))
        actor.eval()
        theta=data["theta"]
        for split in ("test_same","test_deep"):
            cases=[x for x in tasks if x["split"]==split]
            pre=[fc04.run_one(gate,x,theta,"adaptive",verify) for x in cases]
            raw_by_method={}
            for method in METHODS:
                if method=="backward_slice":
                    raw=[_ancestry_output(x,g,proofs,priority_trace)
                         for x,g in zip(cases,pre)]
                else:
                    raw=[fc06.evaluate_condition(gate,actor,x,theta,method,proofs,verify,g)
                         for x,g in zip(cases,pre)]
                metric=_metrics(cases,raw)
                families={}
                for family in sorted({x["family_id"] for x in cases}):
                    idx=[i for i,x in enumerate(cases) if x["family_id"]==family]
                    families[family]=_metrics([cases[i] for i in idx],[raw[i] for i in idx])
                filename=out/(tag+"_"+split+"_"+method+".jsonl")
                filename.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in raw))
                metric.update({
                    "tag":tag,"seed":seed,"split":split,
                    "method":method,"theta":theta,
                    "by_family":families,
                    "predictions_sha256":hashlib.sha256(filename.read_bytes()).hexdigest(),
                })
                report_items.append(metric)
                raw_by_method[method]=raw
                if method=="learned_joint":
                    accepted_model_proofs.extend((x,y["positive_trace"])
                        for x,y in zip(cases,raw) if y["outcome"]=="established_certified")
                print("FC07_SCORE",json.dumps({key:metric[key] for key in (
                    "seed","split","method","eligible_positive",
                    "certified_positive","certified_negative",
                    "abstentions","total_recurrent_cells",
                    "total_measured_cpu_ms_per_case")},sort_keys=True),flush=True)
            for comparator in ("random_legal","goal_first","backward_slice"):
                pairs.append({"seed":seed,"split":split,"baseline":comparator,
                              **_pair(cases,raw_by_method["learned_joint"],
                                    raw_by_method[comparator])})
    leanfile=out/"FC07_model_proofs.lean"
    leanfile.write_text(proofs.lean_document(accepted_model_proofs))
    checks=[]
    for seed in SEEDS:
        for split in ("test_same","test_deep"):
            def find(method):
                return next(m for m in report_items if m["seed"]==seed and
                            m["split"]==split and m["method"]==method)
            learned,back=find("learned_joint"),find("backward_slice")
            checks.append({
                "seed":seed,"split":split,
                "learned_beats_backward_slice":learned["certified_positive"]>back["certified_positive"],
                "learned_lower_cpu_cost_per_positive": (
                    learned["total_measured_cpu_ms_per_case"]/
                    max(1,learned["certified_positive"]) <
                    back["total_measured_cpu_ms_per_case"]/
                    max(1,back["certified_positive"])),
                "zero_false_formal":learned["false_formal_certificates"]==0,
            })
    receipt={
        "experiment":"FC-07",
        "first_preregistered_test":True,
        "preregistration_commit":PREREG_SHA,
        "training_source_commit":FC07_TRAIN_SOURCE_SHA,
        "training_data_sha256":FC07_TRAIN_DATA_SHA,
        "test_source_commit":FC07_TEST_SOURCE_SHA,
        "test_data_sha256":test_digest,
        "training_manifest_sha256":hashlib.sha256(manifestfile.read_bytes()).hexdigest(),
        "model_proposed_kernel_theorems":len(accepted_model_proofs),
        "model_lean_source_sha256":hashlib.sha256(leanfile.read_bytes()).hexdigest(),
        "registered_learning_criterion":all(x["learned_beats_backward_slice"] and
                    x["zero_false_formal"] for x in checks),
        "registered_compute_criterion":all(x["learned_lower_cpu_cost_per_positive"] for x in checks),
        "primary":checks,"paired":pairs,"scores":report_items,
        "symbolic_oracle_accuracy":1.0,
        "scope":"public finite Horn entailment, not a raw-paper mathematical proof",
    }
    (out/"FC07_report.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("FC07_CRITERIA",json.dumps({
        "learning":receipt["registered_learning_criterion"],
        "compute":receipt["registered_compute_criterion"],
        "kernel_proofs":receipt["model_proposed_kernel_theorems"],
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="train":
        if a.weights is not None:
            raise ValueError("training cannot consume frozen test output")
        train(a.source,a.out)
    else:
        if a.weights is None:
            raise ValueError("test requires frozen training manifest")
        evaluate(a.source,a.weights,a.out)


if __name__=="__main__":
    main()
