"""LOOPSCOPE-04 post-run independent standard-library source receipt audit.

Source pinned to ORIGINAL successful full 3-seed run 44142413468;
replays per-sequence activation averages, 2000-draw original
sequence bootstrap bounds, optimizer ledger admissions, all 24576
one-token native causal arms, paired true margins and scientific gate.
Does not load Torch or reexecute the trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

SCHEMA="LGA-01/weight-tied-TRM-supervised-gradient-alignment-v1"
ORIGINAL_SHA="f4e5e92932f42ff362f1a69b0f72aa1f5408924453c36e1b98e4084f34b83c27"
RUN_URL="https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44142413468"
SEEDS=(103,211,307)
BOOT_SEED=41041
ARMS=("final_only","deep_020")
STEPS=(0,50,100,200)
GOALS=tuple(81*j//8 for j in range(8))
METRICS=("cosine","predicted_CE_gain","actual_CE_gain",
         "positive_predicted_fraction","actual_CE_decrease_fraction",
         "mean_delta_norm","mean_gradient_norm",
         "head_visible_squared_fraction")
SPLITS={
 "train":{"n_sequences":512,"source_seed":31041,
     "x_sha256":"12deb8f0b37bc4709f395786f455ee147b4fcf16606432fde8250d04818ad818",
     "y_sha256":"bdc287fe1da93e4e5cacaf0b3537f3819100bcf225c380b86e5e0c5773bbc7dd"},
 "val":{"n_sequences":128,"source_seed":31042,
     "x_sha256":"f1506d79fdc7daac45b2624297d9abc0634ad9f0b52c0f8c04d33e5853d71bc5",
     "y_sha256":"327708fcd22a23617b2fc23146a983a4f08bc4859f668eae1607b3009f99581d"},
 "test":{"n_sequences":256,"source_seed":31043,
     "x_sha256":"e0980800dce2b03d5fe18afdaa0a71e873634826e7641c9ae24bb385ad592fac",
     "y_sha256":"2926721bd0bb2d10c9632c3960ef8a4a1c7a506af28eeef1360648f72f3e8f16"},
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def seed(*parts):
    t="LGA01|"+"|".join(map(str,parts))
    return int.from_bytes(hashlib.sha256(t.encode()).digest()[:8],"big") & ((1<<63)-1)


def close(a,b,name="",tol=1e-9):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        raise ValueError("nonfinite saved statistical result: "+name)
    if abs(float(a)-float(b))>tol*max(1.,abs(float(a)),abs(float(b))):
        raise ValueError("saved statistical result differs: "+name)


def bootstrap(vals,local_seed):
    generator=random.Random(local_seed)
    n=len(vals);reps=2000
    b=sorted(sum(vals[generator.randrange(n)] for i in range(n))/n
             for j in range(reps))
    return [b[int(.025*reps)],b[int(.975*reps)-1]]


def check_profile(profile,seed_id,arm,step):
    if profile["n_independent_sequences"]!=256:
        raise ValueError("incorrect independent 256-string source cohort")
    rank=profile["measured_head_rank"]
    if not 1<=rank<=10:
        raise ValueError("invalid exact source output-head rank")
    if set(profile["per_loop"])!={str(k) for k in range(4)}:
        raise ValueError("one of four shared F iterations missing")
    if set(profile["readout"])!={str(k) for k in range(5)}:
        raise ValueError("five fixed source readouts missing")
    for k in range(5):
        row=profile["readout"][str(k)]
        for name,mean in (("per_sequence_CE","mean_CE"),
                          ("per_sequence_accuracy","mean_accuracy")):
            vals=row[name]
            if len(vals)!=256:
                raise ValueError("source per-sequence readout length changed")
            close(sum(vals)/len(vals),row[mean],mean)
        if not 0<=row["mean_accuracy"]<=1:
            raise ValueError("heldout accuracy outside admissible range")
    for k in range(4):
        row=profile["per_loop"][str(k)]
        cases=row["per_sequence"]
        if len(cases)!=256:
            raise ValueError("source token metrics were not grouped in 256 sequences")
        if set(row["mean"])!=set(METRICS):
            raise ValueError("source metric list does not match preregistration")
        for key in METRICS:
            values=[x[key] for x in cases]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("nonfinite source trajectory metric")
            close(sum(values)/len(values),row["mean"][key],key)
        for key in ("positive_predicted_fraction","actual_CE_decrease_fraction",
                    "head_visible_squared_fraction"):
            if any(not 0<=x[key]<=1+1e-9 for x in cases):
                raise ValueError("source fraction violated [0,1] range: "+key)
        if any(abs(x["cosine"])>1+1e-6 for x in cases):
            raise ValueError("source cosine not normalized")
        if set(row["head_only_bayes"])!={
            "bayes_utility","uniform_utility","always_abstain_utility",
            "action_rate","correct_nonzero"}:
            raise ValueError("source candidate decision diagnostic missing")
        b=row["head_only_bayes"]
        if not 0<=b["action_rate"]<=1 or not 0<=b["correct_nonzero"]<=1:
            raise ValueError("source action fractions not probabilities")
        for key in ("bayes_utility","uniform_utility","always_abstain_utility"):
            if not -1.1<=b[key]<=1:
                raise ValueError("source offline utility outside reward bounds")


def check_native(intervention,seed_id,arm):
    if set(intervention)!={"1","2"}:
        raise ValueError("native intervention has wrong loops")
    total=0
    source_means={}
    for k in (1,2):
        receipt=intervention[str(k)]
        if receipt["n_puzzles"]!=64 or receipt["n_goals"]!=512:
            raise ValueError("native task heldout intervention count changed")
        samples=receipt["cases"]
        if len(samples)!=64 or [r["sample"] for r in samples]!=list(range(64)):
            raise ValueError("native samples do not preserve first 64 test sequence IDs")
        per={"plus":[],"minus":[],"random":[]}
        for sample in samples:
            if len(sample["goals"])!=8:
                raise ValueError("eight fixed source goals per sequence required")
            staged={tag:{"CE_change":[],"margin_change":[]} for tag in per}
            for j,g in enumerate(sample["goals"]):
                total+=4
                if g["goal_token"]!=GOALS[j]:
                    raise ValueError("native intervention goal selection depends on outcomes")
                if not 1<=g["truth_digit"]<=9 or g["truth_digit"]==g["rival"] or not 0<=g["rival"]<=9:
                    raise ValueError("native digit competitor invalid")
                if abs(g["dose"]-g["sham_dose"])>1e-6 or g["dose"]<0:
                    raise ValueError("source random edit differs in physical norm")
                if set(g["arms"])!={"clean","plus","minus","random"}:
                    raise ValueError("source remaining-loop counterfactual arm missing")
                if set(g["contrasts"])!=set(per):
                    raise ValueError("source intervention contrast arm missing")
                for tag in per:
                    for field,key in (("CE_change","CE"),("margin_change","margin")):
                        raw=g["arms"][tag][key]-g["arms"]["clean"][key]
                        close(raw,g["contrasts"][tag][field],
                              f"native{k}.{tag}.{field}",tol=5e-7)
                        staged[tag][field].append(raw)
            for tag in per:
                per[tag].append({field:sum(values)/len(values)
                                 for field,values in staged[tag].items()})
        source_means[str(k)]={}
        for tag in per:
            source_means[str(k)][tag]={}
            for field in ("CE_change","margin_change"):
                mean=sum(x[field] for x in per[tag])/64
                close(mean,receipt["arms_mean"][tag][field],
                      f"native{k}.{tag}.{field}")
                source_means[str(k)][tag][field]=mean
    return total,source_means


def audit(full,raw):
    if sha(raw)!=ORIGINAL_SHA:
        raise ValueError("original source-registered full.json SHA mismatch")
    if full["schema"]!=SCHEMA or full["mode"]!="full":
        raise ValueError("source schema or pilot/full designation changed")
    if full["source_splits"]!=SPLITS:
        raise ValueError("source dataset hashes or independent sequence sizes changed")
    if full["seeds"]!=list(SEEDS) or full["training_arms"]!=list(ARMS):
        raise ValueError("source seed or supervision arm roster changed")
    if full["steps"]!=200 or full["checkpoints"]!=list(STEPS):
        raise ValueError("registered supervised optimizer budget changed")
    if set(full["results"])!={str(s) for s in SEEDS}:
        raise ValueError("independent training seed omitted")
    seed_results=[]
    n_native_arm_cases=0
    native_means={}
    for model_seed in SEEDS:
        pair=full["results"][str(model_seed)]
        if set(pair)!=set(ARMS):
            raise ValueError("one matched-seed supervision control missing")
        if (pair["final_only"]["checkpoints"]["0"]!=
            pair["deep_020"]["checkpoints"]["0"]):
            raise ValueError("paired optimization arms differ at identical initialization")
        native_means[str(model_seed)]={}
        for arm in ARMS:
            receipt=pair[arm]
            if set(receipt["checkpoints"])!={str(x) for x in STEPS}:
                raise ValueError("predeclared model checkpoint missing")
            if set(receipt["optimizer_ledger"])!={"50","100","200"}:
                raise ValueError("original BPTT weight optimizer ledger incomplete")
            for step,record in receipt["optimizer_ledger"].items():
                if not -1<=record["optimizer_update_cos_negative_BPTT_grad"]<=1:
                    raise ValueError("parameter update cosine outside [−1,1]")
                for key in ("optimizer_gradient_l2","optimizer_update_l2"):
                    if record[key]<=0:
                        raise ValueError("recorded AdamW update/gradient norm not positive")
            for step in STEPS:
                check_profile(receipt["checkpoints"][str(step)],model_seed,arm,step)
            final=receipt["checkpoints"]["200"]
            initial=receipt["checkpoints"]["0"]
            d=receipt["registered_diagnostics"]
            close(d["test_final_acc"],final["readout"]["4"]["mean_accuracy"])
            admitted=bool(final["readout"]["4"]["mean_accuracy"]>.2)
            if admitted!=d["task_admitted"]:
                raise ValueError("task-learning admission selection changed")
            if len(d["per_loop_gradient_alignment"])!=4:
                raise ValueError("missing original loopwise model classifications")
            qualifies=0
            matched=0
            for k,item in enumerate(d["per_loop_gradient_alignment"]):
                if item["k"]!=k:
                    raise ValueError("source loop classifier index changed")
                vals=[z["cosine"] for z in final["per_loop"][str(k)]["per_sequence"]]
                ci=bootstrap(vals,seed(BOOT_SEED,model_seed,arm,k))
                close(item["mean"],sum(vals)/len(vals))
                for a,b in zip(item["bootstrap95"],ci):
                    close(a,b)
                lower=ci[0]>0
                grew=(final["per_loop"][str(k)]["mean"]["cosine"]>
                      initial["per_loop"][str(k)]["mean"]["cosine"])
                if lower!=item["lower_above_zero"] or grew!=item["higher_than_initial"]:
                    raise ValueError("saved original depth-level gradient classification changed")
                qualifies+=int(lower)
                matched+=int(lower and grew)
            passed=bool(admitted and qualifies>=2)
            if passed!=d["seed_positive_descent"]:
                raise ValueError("registered seed-level gradient gate changed")
            n,means=check_native(receipt["native_remaining_loop_intervention"],
                                 model_seed,arm)
            n_native_arm_cases+=n
            native_means[str(model_seed)][arm]=means
            seed_results.append({"seed":model_seed,"arm":arm,
                                 "task_admitted":admitted,
                                 "positive_depths":qualifies,
                                 "improved_positive_depths":matched,
                                 "registered_seed_pass":passed,
                                 "final_acc":d["test_final_acc"],
                                 "final_cosine_means":[
                                    final["per_loop"][str(k)]["mean"]["cosine"]
                                    for k in range(4)],
                                 "final_head_visible_fraction":[
                                    final["per_loop"][str(k)]["mean"]["head_visible_squared_fraction"]
                                    for k in range(4)]})
    votes=[x for x in seed_results if x["arm"]=="final_only"]
    count=sum(x["registered_seed_pass"] and
              x["improved_positive_depths"]>=2 for x in votes)
    study=bool(count>=2)
    if study!=full["registered_study_positive"] or full["registered_study_status"]!=(
        "POSITIVE" if study else "FAIL_OR_NOT_ADMITTED"):
        raise ValueError("registered source study-level gradient result changed")
    return {"schema":"LOOPSCOPE-04/independent-scalar-source-audit-v1",
            "source_run":RUN_URL,"full_raw_json_sha256":ORIGINAL_SHA,
            "status":"PASS",
            "registered_scientific_status":full["registered_study_status"],
            "model_final_only_seed_classifications_passed":count,
            "n_distinct_heldout_source_sequences":256,
            "n_native_original_source_sequences":64,
            "n_native_per_goal_arms_checked":n_native_arm_cases,
            "supervision_and_seed_results":seed_results,
            "source_native_mean_effects":native_means,
            "limitations":"Replays exact source/optimizer/activation averages, bootstrap and stored native margin/CE arithmetic; does not independently reexecute tied teacher weights or derive gradients from hidden activations."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    raw=args.receipt.read_bytes()
    report=audit(json.loads(raw),raw)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("LGA01_INDEPENDENT_AUDIT_PASS",
          report["registered_scientific_status"],
          report["model_final_only_seed_classifications_passed"],
          report["n_native_per_goal_arms_checked"],flush=True)


if __name__=="__main__":
    main()
