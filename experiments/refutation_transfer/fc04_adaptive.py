"""FC-04: validation-calibrated real early exit with exact countermodel gate.

Preregistered first, at Compactification-Theory commit:
ce7190129617d8387a7acf4ceed3e8d0570cfce4
The training job sees only FC-02. A separate dependent CI evaluation job sees
the new FC-04 source only after SHA-pinned model checkpoints and thresholds
have been uploaded. Full source->formal verification is out of scope.
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
import formal_student as fs
import fc03_compare as original

PREREG_SHA="ce7190129617d8387a7acf4ceed3e8d0570cfce4"
FC04_SOURCE_SHA="1b85412aa237d7522341c40d029654052e1d3267"
FC04_DATA_SHA="08e57752f02c886e0947e36687c0572d84cfc815a467e1ea0c2554a1a044c04c"
THRESHOLDS=(.55,.70,.85,.93,.98)
STAGES=(4,6,8,10,12)
MODES=("fixed4","fixed8","fixed12","adaptive","cert_only_first")
SEEDS=(17,29)


def _public_graph(task):
    """Never pass gold, family, split, proof trace or certificate to model inputs."""
    return {key:task[key] for key in ("case_id","atoms","facts","rules","goal")} | {
        "label":"refuted","reachable":[]}


def _decode_logits(task,logits):
    names=task["atoms"]
    probability=torch.sigmoid(logits[:len(names)])
    values={name:bool(float(probability[j])>=.5) for j,name in enumerate(names)}
    return values,float(probability[names.index(task["goal"])])


def _check_negative(task,values,verify):
    if values[task["goal"]]:
        return False
    p=task["formal"]
    return bool(verify(p["premises"],p["conclusion"],values)["verified"])


@torch.no_grad()
def validation_trajectories(model,dev,verify,batch_size=32):
    """Permitted validation-only depth sweep; all stages use the same weights."""
    trajectories=[]
    original_loops=model.loops
    model.loops=12
    try:
        for offset in range(0,len(dev),batch_size):
            tasks=dev[offset:offset+batch_size]
            inp=fs.encode([_public_graph(t) for t in tasks])
            outputs=model(inp)
            for j,task in enumerate(tasks):
                frames=[]
                for steps in STAGES:
                    assignment,p_goal=_decode_logits(task,outputs[steps-1][j])
                    checked=_check_negative(task,assignment,verify)
                    frames.append({"passes":steps,"goal_probability":p_goal,
                                   "countermodel_verified":checked})
                trajectories.append(frames)
    finally:
        model.loops=original_loops
    return trajectories


def decision_from_frames(frames,theta,mode):
    """Pure stopping rule: first valid witness always wins; positive needs stability."""
    if mode.startswith("fixed"):
        target=int(mode.removeprefix("fixed"))
        record=next(r for r in frames if r["passes"]==target)
        if record["countermodel_verified"]:
            return {"outcome":"refuted_certified","passes":target}
        if record["goal_probability"]>=theta:
            return {"outcome":"established_provisional","passes":target}
        return {"outcome":"abstain","passes":target}
    if mode not in ("adaptive","cert_only_first"):
        raise ValueError("unsupported stopping rule")
    previous=None
    for frame in frames:
        passes=frame["passes"]
        if frame["countermodel_verified"]:
            return {"outcome":"refuted_certified","passes":passes}
        if mode=="adaptive" and previous is not None and (
                frame["goal_probability"]>=theta and previous["goal_probability"]>=theta):
            return {"outcome":"established_provisional","passes":passes}
        previous=frame
    return {"outcome":"abstain","passes":12}


def _utility(tasks,decisions):
    total=0.
    for task,entry in zip(tasks,decisions):
        if entry["outcome"]=="refuted_certified":
            # A checked countermodel cannot refute a valid finite formula.
            if task["label"]!="refuted":
                raise AssertionError("verified false countermodel")
            total+=1.
        elif entry["outcome"]=="established_provisional":
            total+=1. if task["label"]=="established" else -5.
        total-=.01*entry["passes"]
    return round(total/len(tasks),9)


def calibrate_threshold(model,dev,verify):
    trajectories=validation_trajectories(model,dev,verify)
    ledger=[]
    for theta in THRESHOLDS:
        decisions=[decision_from_frames(f,theta,"adaptive") for f in trajectories]
        correct=sum(d["outcome"]=="established_provisional" and t["label"]=="established"
                    or d["outcome"]=="refuted_certified" and t["label"]=="refuted"
                    for t,d in zip(dev,decisions))
        wrong=sum(d["outcome"]=="established_provisional" and t["label"]=="refuted"
                  for t,d in zip(dev,decisions))
        item={"threshold":theta,"utility":_utility(dev,decisions),
              "correct_accepted":correct,"wrong_provisional":wrong,
              "abstained":sum(d["outcome"]=="abstain" for d in decisions),
              "mean_passes":round(sum(d["passes"] for d in decisions)/len(decisions),6)}
        ledger.append(item)
    best=max(ledger,key=lambda x:(x["utility"],x["threshold"]))
    return {"selected_threshold":best["threshold"],
            "selection": "validation-only maximum preregistered utility; ties prefer higher theta",
            "all_grid_scores":ledger}


def load_train_only(root):
    train,dev,hash_value=original.load_source_training(root)
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.certificates import verify_countermodel
    return train,dev,hash_value,verify_countermodel


def train_models(root,out):
    """Called without an FC-04 checkout: source access verified in CI."""
    torch.set_num_threads(2)
    train,dev,h,verify=load_train_only(root)
    out.mkdir(parents=True,exist_ok=True)
    manifest={"protocol_sha":PREREG_SHA,
              "phase":"train_and_calibrate_on_fc02_only",
              "fc02_data_sha256":h,
              "fc04_test_available_in_train":False,
              "threshold_grid":list(THRESHOLDS),
              "pass_checkpoints":list(STAGES),
              "models":[]}
    for seed in SEEDS:
        model,metadata=original._train_one(train,"tied4_width24",seed)
        tag=f"fc04_tied4_s{seed}"
        model_path=out/(tag+".pt")
        torch.save(model.state_dict(),model_path)
        checkpoint_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
        # Calibration reads only old FC-02 development labels and a frozen network.
        calibration=calibrate_threshold(model,dev,verify)
        metadata.update(tag=tag,checkpoint_sha256=checkpoint_sha,
                        calibration=calibration,
                        network_train_data_only=True,
                        stopping_rule_learned_from_fc02_dev_only=True)
        manifest["models"].append(metadata)
        print("CALIBRATED",json.dumps({"tag":tag,"param_count":metadata["param_count"],
              "checkpoint_sha256":checkpoint_sha,
              "selected_theta":calibration["selected_threshold"],
              "grid_scores":calibration["all_grid_scores"]},sort_keys=True),flush=True)
    path=out/"train_manifest.json"
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("FROZEN_TRAIN_MANIFEST",path)


def load_test_only(root):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc04 import generator as fc04
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    cases=fc04.generate_tests()
    fc04.audit_tests(cases)
    h=hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in cases)).hexdigest()
    if h!=FC04_DATA_SHA:
        raise AssertionError("FC-04 test corpus differs from frozen CI hash: "+h)
    return cases,h,verify_countermodel


@torch.no_grad()
def run_one(model,task,theta,mode,verify):
    """Real early stopping: execute only the cell calls the model actually uses."""
    b=fs.encode([_public_graph(task)])
    a=model.atom_in(b["a"])
    r=model.rule_in(b["r"])
    n=len(task["atoms"])
    prior_positive=False
    max_pass=int(mode.removeprefix("fixed")) if mode.startswith("fixed") else 12
    model_seconds=0.
    checker_seconds=0.
    checker_calls=0
    outcome="abstain"
    reason="budget"
    cert=None
    last_probability=0.
    steps=0
    for k in range(1,max_pass+1):
        start=time.perf_counter()
        a,r=model.cell(a,r,b["prem"],b["head"])
        if k in STAGES:
            logits=model.atom_out(a).squeeze(-1)[0]
        model_seconds+=time.perf_counter()-start
        steps=k
        if k not in STAGES:
            continue
        if mode.startswith("fixed") and k<max_pass:
            continue
        assignment,p_goal=_decode_logits(task,logits)
        last_probability=p_goal
        negative=False
        if not assignment[task["goal"]]:
            checker_calls+=1
            start=time.perf_counter()
            negative=_check_negative(task,assignment,verify)
            checker_seconds+=time.perf_counter()-start
        if negative:
            outcome="refuted_certified"
            reason="exact_countermodel"
            cert=assignment
            break
        if mode=="adaptive":
            current_positive=p_goal>=theta
            if current_positive and prior_positive:
                outcome="established_provisional"
                reason="two_stable_positive_checkpoints"
                break
            prior_positive=current_positive
        elif mode.startswith("fixed"):
            if p_goal>=theta:
                outcome="established_provisional"
                reason="fixed_goal_threshold"
            break
        elif mode=="cert_only_first":
            pass
        else:
            raise ValueError(mode)
    return {"case_id":task["case_id"],"outcome":outcome,"reason":reason,
            "passes":steps,
            "goal_probability":round(last_probability,8),
            "certificate":cert,
            "countermodel_verified":bool(cert is not None),
            "checker_calls":checker_calls,
            "model_forward_ms":round(model_seconds*1000,6),
            "checker_ms":round(checker_seconds*1000,6)}


def score(tasks,results):
    if len(tasks)!=len(results):
        raise AssertionError("missing cases")
    count=Counter()
    total_passes=0
    model_ms=0.
    checker_ms=0.
    checks=0
    for task,answer in zip(tasks,results):
        if task["case_id"]!=answer["case_id"]:
            raise AssertionError("case ordering differs")
        verdict=answer["outcome"]
        label=task["label"]
        if verdict=="refuted_certified" and label!="refuted":
            raise AssertionError("counterexample verifier accepted a positive theorem")
        count[(label,verdict)]+=1
        total_passes+=answer["passes"]
        model_ms+=answer["model_forward_ms"]
        checker_ms+=answer["checker_ms"]
        checks+=answer["checker_calls"]
    total=len(tasks)
    valid=sum(t["label"]=="established" for t in tasks)
    invalid=total-valid
    certified=count[("refuted","refuted_certified")]
    rightpositive=count[("established","established_provisional")]
    wrongpositive=count[("refuted","established_provisional")]
    accepted=certified+rightpositive+wrongpositive
    passes=sorted(x["passes"] for x in results)
    return {
        "count":total,
        "certified_refuted":certified,
        "certified_recall_invalid":round(certified/invalid,6),
        "verified_false_countermodels":0,
        "provisional_valid_correct":rightpositive,
        "provisional_valid_wrong":wrongpositive,
        "provisional_wrong_rate_on_invalid":round(wrongpositive/invalid,6),
        "selective_coverage":round(accepted/total,6),
        "selective_error_rate":round(wrongpositive/accepted,6) if accepted else None,
        "abstentions":total-accepted,
        "avg_passes":round(total_passes/total,6),
        "median_passes":statistics.median(passes),
        "p95_passes":passes[min(total-1,int(.95*total))],
        "total_cell_calls":total_passes,
        "verified_refutations_per_1000_cell_calls":round(1000*certified/total_passes,6) if total_passes else None,
        "model_forward_ms_per_case":round(model_ms/total,6),
        "checker_ms_per_case":round(checker_ms/total,6),
        "total_cpu_model_plus_checker_ms":round(model_ms+checker_ms,5),
        "exact_checker_calls":checks,
        "utility":_utility(tasks,results),
        "confusion":{f"{a}->{b}":count[(a,b)] for a in ("established","refuted")
                     for b in ("refuted_certified","established_provisional","abstain")}
    }


def evaluate(root,weights,out):
    torch.set_num_threads(2)
    manifest=json.loads((weights/"train_manifest.json").read_text())
    if manifest["protocol_sha"]!=PREREG_SHA or manifest["fc02_data_sha256"]!=original.FC02_DATA_SHA:
        raise AssertionError("train manifest incompatible with frozen preregistration")
    if manifest.get("fc04_test_available_in_train") is not False:
        raise AssertionError("FC-04 test data accessible at training")
    if {x["seed"] for x in manifest["models"]}!=set(SEEDS):
        raise AssertionError("seed or arm changed")
    rows,h,verify=load_test_only(root)
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    primary=[]
    for entry in manifest["models"]:
        seed=entry["seed"]
        name=entry["tag"]
        model_file=weights/(name+".pt")
        if hashlib.sha256(model_file.read_bytes()).hexdigest()!=entry["checkpoint_sha256"]:
            raise AssertionError("checkpoint hash differs")
        model=original.make_model("tied4_width24")
        model.load_state_dict(torch.load(model_file,weights_only=True,map_location="cpu"))
        model.eval()
        theta=entry["calibration"]["selected_threshold"]
        for split in ("test_same","test_deep"):
            tasks=[t for t in rows if t["split"]==split]
            split_scores={}
            for mode in MODES:
                preds=[run_one(model,t,theta,mode,verify) for t in tasks]
                metrics=score(tasks,preds)
                outfile=out/(f"{name}_{split}_{mode}.jsonl")
                outfile.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in preds))
                metrics.update(tag=name,seed=seed,split=split,mode=mode,
                               threshold=theta,
                               prediction_sha256=hashlib.sha256(outfile.read_bytes()).hexdigest())
                split_scores[mode]=metrics
                results.append(metrics)
                print("POLICY",json.dumps({key:metrics[key] for key in (
                    "tag","split","mode","threshold","certified_refuted",
                    "provisional_valid_correct","provisional_valid_wrong",
                    "avg_passes","total_cell_calls","utility",
                    "verified_refutations_per_1000_cell_calls")},sort_keys=True),flush=True)
            adaptive=split_scores["adaptive"]
            fixed=split_scores["fixed12"]
            baseline=split_scores["cert_only_first"]
            criteria={
                "cert_recall_ge_fixed12":
                   adaptive["certified_refuted"]>=fixed["certified_refuted"],
                "fewer_mean_calls":
                   adaptive["avg_passes"]<fixed["avg_passes"],
                "no_verified_false_counterexamples":
                   adaptive["verified_false_countermodels"]==0,
                "provisional_error_no_increase":
                   adaptive["provisional_wrong_rate_on_invalid"]<=fixed["provisional_wrong_rate_on_invalid"],
            }
            primary.append({"seed":seed,"split":split,
                            "meets_all":all(criteria.values()),
                            "registered_criteria":criteria,
                            "cert_only_coverage":baseline["certified_refuted"]})
    report={"experiment":"FC-04",
            "status":"first_sealed_test_pass",
            "preregistration_sha":PREREG_SHA,
            "train_manifest_sha256":hashlib.sha256((weights/"train_manifest.json").read_bytes()).hexdigest(),
            "test_source_sha":FC04_SOURCE_SHA,
            "test_data_sha256":h,
            "models_use_real_early_exit":True,
            "data_are_public_exact_Horn_cases":True,
            "test_criteria_met_all_seeds_and_splits":all(x["meets_all"] for x in primary),
            "registered_criteria":primary,
            "baseline_exact_oracle_accuracy":1.0,
            "scores":results}
    (out/"FC04_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("FINAL_CRITERION",json.dumps({"passed":report["test_criteria_met_all_seeds_and_splits"],
                                      "criteria":primary},sort_keys=True))
    print("FC04_REPORT",out/"FC04_report.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--fc02-root",type=Path)
    p.add_argument("--fc04-root",type=Path)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="train":
        if a.fc02_root is None or a.fc04_root is not None:
            raise ValueError("training must only see the FC-02 checkout")
        train_models(a.fc02_root,a.out)
    else:
        if a.fc04_root is None or a.weights is None:
            raise ValueError("evaluation needs the FC-04 generator and frozen weights")
        evaluate(a.fc04_root,a.weights,a.out)


if __name__=="__main__":
    main()
