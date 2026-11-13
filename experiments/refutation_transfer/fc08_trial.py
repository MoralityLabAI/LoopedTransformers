"""FC-08 preregistered conjunctive proof-choice against bounded symbolic search.

This is a controlled finite-Horn proof-search experiment, not a human
mathematical manuscript evaluator. The training job mounts the train-only
FC-08 source revision and FC-02 for the frozen gate. It saves SHA-identified
weights BEFORE the separate test job checks out FC-08 held-out examples.
Exact replay and Lean kernel certification gate all positive claims.
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
import fc08_symbolic as symbolic

PREREG_SHA="b6c8ffbef6606e89737bbfc32d73b8732b765e8d"
TRAIN_SOURCE_SHA="51d0082f27e4c6214ec0cebb4886df44a4962321"
TRAIN_DATA_SHA="0d51135f606d44085f2bf0643e44469c36cbebae2469c5ceb8dcbb52b75f763a"
TEST_SOURCE_SHA="__FC08_TEST_SOURCE__"
TEST_DATA_SHA="__FC08_TEST_HASH__"
SEEDS=(17,29)
BUDGET=5
METHODS=("learned_joint","random_legal","goal_first",
         "backward_slice","unary_bfs","bounded_best_first")
SYMBOLIC=("backward_slice","unary_bfs","bounded_best_first")


def load_source_train(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc08 import train_data as fc08
    from research.refutation_bench.formal_curriculum import generator as fc02
    data=fc08.generate_train_dev()
    fc08.audit_train_dev(data)
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in data)).hexdigest()
    if digest!=TRAIN_DATA_SHA:
        raise AssertionError("FC-08 train data changed from prereg source: "+digest)
    train=[x for x in data if x["split"]=="train"]
    dev=[x for x in data if x["split"]=="dev"]
    if (len(train),len(dev))!=(768,192):
        raise AssertionError("train/dev sample sizes changed")
    return train,dev,digest


def train_frozen(root:Path,out:Path):
    torch.set_num_threads(2)
    train,dev,digest=load_source_train(root)
    fc02train,fc02dev,_=fc03.load_source_training(root)
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.certificates import verify_countermodel
    out.mkdir(parents=True,exist_ok=True)
    report={
      "experiment":"FC08_conjunctive",
      "phase":"train_and_calibrate_before_test_source_checkout",
      "prereg_sha":PREREG_SHA,
      "train_source_sha":TRAIN_SOURCE_SHA,
      "train_data_sha256":digest,
      "fc02_data_sha256":fc03.FC02_DATA_SHA,
      "test_generator_absent_during_training":True,
      "source_train_samples":len(train),
      "source_dev_samples":len(dev),
      "model_seeds":list(SEEDS),
      "actor_training_recipe":"joint LR01 300 steps, 48 batch, lr=.002",
      "models":[],
    }
    for seed in SEEDS:
        gate,metadata=fc03._train_one(fc02train,"tied4_width24",seed)
        calibration=fc04.calibrate_threshold(gate,fc02dev,verify_countermodel)
        actor,actor_meta=fc06._train_actor(gate,train,seed)
        tag=f"fc08_s{seed}"
        gatefile=out/(tag+"_gate.pt")
        actorfile=out/(tag+"_actor.pt")
        torch.save(gate.state_dict(),gatefile)
        torch.save(actor.state_dict(),actorfile)
        item={
            "tag":tag,"seed":seed,
            "threshold":calibration["selected_threshold"],
            "threshold_grid":calibration["all_grid_scores"],
            "gate_parameter_count":sum(p.numel() for p in gate.parameters()),
            "gate_sha256":hashlib.sha256(gatefile.read_bytes()).hexdigest(),
            "actor_sha256":hashlib.sha256(actorfile.read_bytes()).hexdigest(),
            "gate_training_seconds":metadata["train_seconds"],
            **actor_meta
        }
        report["models"].append(item)
        print("FC08_FROZEN",json.dumps({k:item[k] for k in (
            "tag","threshold","train_prefix_states","train_first128_accuracy",
            "actor_parameters","gate_sha256","actor_sha256")},sort_keys=True),flush=True)
    file=out/"FC08_train_manifest.json"
    file.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC08_TRAIN_RECEIPT",file)


def load_test_source(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc08 import test_data as fc08
    from research.refutation_bench.fc05 import proofs
    from research.refutation_bench.fc08 import controls as public_controls
    from research.refutation_bench.certificates import verify_countermodel
    from research.refutation_bench.formal_curriculum import generator as fc02
    data=fc08.generate_tests()
    fc08.audit_tests(data)
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(x)+b"\n" for x in data)).hexdigest()
    if digest!=TEST_DATA_SHA:
        raise AssertionError("FC-08 test source does not match the frozen hash: "+digest)
    return data,proofs,verify_countermodel,public_controls,digest


def evaluate_symbolic(task:dict,pre:dict,method:str,proofs,
                      public_controls)->dict:
    """Source-only symbolic action trace with exact posthoc replay checker.

    The source-language graph, not the oracle result, determines every
    action. Failed attempted traces are retained in the raw record.
    """
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
      "actor_calls":0,"actor_recurrent_cells":0,
      "actor_forward_ms":0.,"rule_actions":0,
      "proof_failure_reason":None,
      "symbolic_search_ms":0.,"symbolic_states_popped":0,
    }
    if pre["outcome"]!="established_provisional":
        return result
    proposal=public_controls.baseline_trace(task,method)
    start=time.perf_counter()
    checked=proofs.verify_trace(task,proposal["trace"],budget=BUDGET)
    checker_ms=1000*(time.perf_counter()-start)
    result.update({
      "proposal_trace":proposal["trace"],
      "positive_trace":proposal["trace"] if checked["verified"] else None,
      "positive_checked":bool(checked["verified"]),
      "positive_checker_ms":round(checker_ms,6),
      "positive_checker_calls":1,
      "rule_actions":len(proposal["trace"]),
      "proof_failure_reason":None if checked["verified"] else checked["reason"],
      "outcome":"established_certified" if checked["verified"] else "abstain",
      "symbolic_search_ms":proposal["symbolic_search_ms"],
      "symbolic_states_popped":proposal["states_popped"],
    })
    return result


def _metrics(tasks:list[dict],results:list[dict])->dict:
    """Reuse certified dual-polarity metrics and add real symbolic compute."""
    a=fc06._metrics(tasks,results)
    a["symbolic_search_ms_per_case"]=round(
        sum(x.get("symbolic_search_ms",0.) for x in results)/len(results),6)
    a["total_measured_cpu_ms_per_case"]=round(
        a["model_and_checker_ms_per_case"]+
        a["symbolic_search_ms_per_case"],6)
    a["total_popped_symbolic_states"]=sum(
        x.get("symbolic_states_popped",0) for x in results)
    a["mean_symbolic_states_popped"]=round(
        a["total_popped_symbolic_states"]/len(results),6)
    a["proof_attempts"]=sum(x.get("positive_checker_calls",0) for x in results)
    a["first_failure_reasons"]=dict(Counter(x.get("proof_failure_reason") or "none"
              for x in results if x["preproof_outcome"]=="established_provisional"
              and x["outcome"]=="abstain"))
    a["raw_trace_actions"]=sum(x.get("rule_actions",0) for x in results)
    a["mean_recurrent_calls_per_case"]=round(a["total_recurrent_cells"]/len(results),6)
    return a


def _paired(tasks,model,competitor)->dict:
    learned_only=other_only=tied=0
    for task,a,b in zip(tasks,model,competitor):
        if not(task["case_id"]==a["case_id"]==b["case_id"]):
            raise AssertionError("paired result ID mismatch")
        if task["label"]!="established":
            continue
        aa=a["outcome"]=="established_certified"
        bb=b["outcome"]=="established_certified"
        learned_only+=int(aa and not bb)
        other_only+=int(bb and not aa)
        tied+=int(aa and bb)
    return {"learned_only":learned_only,"baseline_only":other_only,
            "both_certified":tied,"net_learned":learned_only-other_only}


def evaluate_frozen(root:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifest_file=weights/"FC08_train_manifest.json"
    manifest=json.loads(manifest_file.read_text())
    if manifest.get("prereg_sha")!=PREREG_SHA or manifest.get("train_data_sha256")!=TRAIN_DATA_SHA:
        raise AssertionError("frozen training hash or registration mismatch")
    if manifest.get("test_generator_absent_during_training") is not True:
        raise AssertionError("test data was available in training phase")
    if {x["seed"] for x in manifest["models"]}!=set(SEEDS):
        raise AssertionError("registered seeds changed")
    cases,proofs,verify,public_controls,test_sha=load_test_source(root)
    out.mkdir(parents=True,exist_ok=True)
    accepted=[]
    scores=[]
    comparisons=[]
    for model_meta in manifest["models"]:
        seed=model_meta["seed"]
        tag=model_meta["tag"]
        gatefile=weights/(tag+"_gate.pt")
        actorfile=weights/(tag+"_actor.pt")
        if hashlib.sha256(gatefile.read_bytes()).hexdigest()!=model_meta["gate_sha256"]:
            raise AssertionError("frozen gate checkpoint digest mismatch")
        if hashlib.sha256(actorfile.read_bytes()).hexdigest()!=model_meta["actor_sha256"]:
            raise AssertionError("frozen actor checkpoint digest mismatch")
        gate=fc03.make_model("tied4_width24")
        gate.load_state_dict(torch.load(gatefile,map_location="cpu",weights_only=True))
        gate.eval()
        actor=fc06.JointActor()
        actor.load_state_dict(torch.load(actorfile,map_location="cpu",weights_only=True))
        actor.eval()
        theta=model_meta["threshold"]
        for split in ("test_same","test_deep"):
            tasks=[x for x in cases if x["split"]==split]
            pre=[fc04.run_one(gate,x,theta,"adaptive",verify) for x in tasks]
            predictions={}
            for method in METHODS:
                if method in SYMBOLIC:
                    outputs=[evaluate_symbolic(t,p,method,proofs,public_controls)
                             for t,p in zip(tasks,pre)]
                else:
                    outputs=[fc06.evaluate_condition(gate,actor,t,theta,method,proofs,verify,p)
                             for t,p in zip(tasks,pre)]
                stats=_metrics(tasks,outputs)
                family_stats={}
                for family in sorted({x["family_id"] for x in tasks}):
                    indices=[i for i,x in enumerate(tasks) if x["family_id"]==family]
                    family_stats[family]=_metrics(
                        [tasks[i] for i in indices],[outputs[i] for i in indices])
                preds=out/(tag+"_"+split+"_"+method+".jsonl")
                preds.write_text("".join(json.dumps(p,sort_keys=True)+"\n" for p in outputs))
                stats.update({
                  "tag":tag,"seed":seed,"method":method,"split":split,
                  "threshold":theta,"by_family":family_stats,
                  "raw_predictions_sha256":hashlib.sha256(preds.read_bytes()).hexdigest()
                })
                scores.append(stats)
                predictions[method]=outputs
                if method=="learned_joint":
                    accepted.extend((task,entry["positive_trace"])
                        for task,entry in zip(tasks,outputs)
                        if entry["outcome"]=="established_certified")
                print("FC08_SCORE",json.dumps({k:stats[k] for k in (
                    "seed","split","method","eligible_positive",
                    "certified_positive","certified_negative",
                    "abstentions","total_recurrent_cells",
                    "mean_symbolic_states_popped",
                    "total_measured_cpu_ms_per_case")},sort_keys=True),flush=True)
            for control in METHODS:
                if control=="learned_joint":
                    continue
                comparisons.append({"seed":seed,"split":split,"control":control,
                                    **_paired(tasks,predictions["learned_joint"],
                                             predictions[control])})
    lean=out/"FC08_model_proofs.lean"
    lean.write_text(proofs.lean_document(accepted),encoding="utf8")
    criterion=[]
    for seed in SEEDS:
        for split in ("test_same","test_deep"):
            neural=next(s for s in scores if s["seed"]==seed and
                        s["split"]==split and s["method"]=="learned_joint")
            beam=next(s for s in scores if s["seed"]==seed and
                        s["split"]==split and s["method"]=="bounded_best_first")
            ncost=(neural["total_measured_cpu_ms_per_case"]*len(
                   [t for t in cases if t["split"]==split]))/max(1,neural["certified_positive"])
            bcost=(beam["total_measured_cpu_ms_per_case"]*len(
                   [t for t in cases if t["split"]==split]))/max(1,beam["certified_positive"])
            criterion.append({
              "seed":seed,"split":split,
              "learned_positive_count":neural["certified_positive"],
              "bounded_best_first_positive_count":beam["certified_positive"],
              "learned_beats_symbolic_search":neural["certified_positive"]>beam["certified_positive"],
              "learned_cost_ms_per_positive":round(ncost,5),
              "symbolic_cost_ms_per_positive":round(bcost,5),
              "learned_cheaper_per_certificate":ncost<bcost,
              "zero_false_formal_certificates":neural["false_formal_certificates"]==0,
            })
    report={
       "experiment":"FC08",
       "preregistration_commit":PREREG_SHA,
       "first_frozen_test":True,
       "training_source_commit":TRAIN_SOURCE_SHA,
       "training_corpus_sha256":TRAIN_DATA_SHA,
       "test_source_commit":TEST_SOURCE_SHA,
       "test_corpus_sha256":test_sha,
       "frozen_model_manifest_sha256":hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
       "positive_kernel_theorem_count":len(accepted),
       "positive_model_lean_sha256":hashlib.sha256(lean.read_bytes()).hexdigest(),
       "registered_learning_criterion":all(x["learned_beats_symbolic_search"] and
           x["zero_false_formal_certificates"] for x in criterion),
       "registered_compute_criterion":all(x["learned_cheaper_per_certificate"] for x in criterion),
       "primary":criterion,"scores":scores,"paired":comparisons,
       "exact_shortest_oracle_positive_recall":1.0,
       "model_test_input_labels_used":False,
       "scope":"12-atom finite conjunctive Horn graph, not human math manuscripts",
    }
    (out/"FC08_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC08_CRITERIA",json.dumps({
        "learning":report["registered_learning_criterion"],
        "compute":report["registered_compute_criterion"],
        "positive_kernel_proofs":report["positive_kernel_theorem_count"]
    },sort_keys=True),flush=True)
    print("FC08_REPORT",out/"FC08_report.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="train":
        if a.weights is not None:
            raise ValueError("training phase must not read test checkpoint files")
        train_frozen(a.source,a.out)
    else:
        if a.weights is None:
            raise ValueError("evaluation requires frozen checkpoint artifacts")
        evaluate_frozen(a.source,a.weights,a.out)


if __name__=="__main__":
    main()
