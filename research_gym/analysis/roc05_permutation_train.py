"""ROC-05A six-seed paired CPU training: four source rules versus 512 new tables.

Frozen protocol: configs/roc05_fresh_permutation_replication_v1.json
First frozen source registration commit: a56c177bbc956274fbf65018b5724d291e25ef3c.
This trainer opens ONLY three named ROC05 development/training sources.
No code path reads original ROC02 sealed families 6/7 or ROC05 reserved cases.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from research_gym.analysis.roc02_tied_train import rows, tensors
from research_gym.analysis.roc03b_query_table_memory import (
    QueryConditionedTableMemory, targets, score
)
from research_gym.analysis.roc05_permutation_source import (
    swapped_row_counterfactual, nuisance_distractor_counterfactual,
)

SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
UPDATES=1920
BATCH=32
LR=.001
WEIGHT=.5
EVAL_STEPS=(240,480,960,1920)
FILES={"low_affine4":"train_low4.jsonl",
       "high_permutation512":"train_high512.jsonl",
       "development":"development.jsonl"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_sources(source_dir: Path):
    m=json.loads((source_dir/"source_manifest.json").read_text())
    if (m["study"]!="ROC-05A-independent-rule-pool-and-model-seed-replication-v1"
        or m["frozen_registration_commit"]!="a56c177bbc956274fbf65018b5724d291e25ef3c"):
        raise RuntimeError("wrong frozen source")
    if (m["reserved_test_examples_generated"] or m["roc05_reserved_test_examples_generated"] or not m["roc04_prior_1152_table_identities_excluded"] or m["old_roc02_sealed_families_read"]
        or not m["same_temporal_schedules_and_queries"]):
        raise RuntimeError("source provenance gate failed")
    groups={}
    for name,filename in FILES.items():
        p=source_dir/filename
        if sha(p)!=m["source_files"][filename]["sha256"]:
            raise RuntimeError("source SHA mismatch: "+filename)
        rr=rows(p)
        if len(rr)!=m["source_files"][filename]["records"]:
            raise RuntimeError("source length mismatch")
        groups[name]=(rr,tensors(rr))
    low=[r for r in groups["low_affine4"][0]]
    high=[r for r in groups["high_permutation512"][0]]
    dev=[r for r in groups["development"][0]]
    if len(low)!=1024 or len(high)!=1024 or len(dev)!=256:
        raise RuntimeError("unregistered source sizes")
    if len({r["rule_table_id"] for r in high})!=512 or len({r["rule_table_id"] for r in dev})!=256:
        raise RuntimeError("unregistered table source diversity")
    if {r["rule_table_id"] for r in high} & {r["rule_table_id"] for r in dev}:
        raise RuntimeError("rule-table train/dev overlap")
    if any(l["events"][1:]!=h["events"][1:] for l,h in zip(low,high)):
        raise RuntimeError("pair schedules differ")
    return groups,m


@torch.no_grad()
def dev_controls(model,dev_rows: list[dict],dev_data):
    original_x,original_mask,original_y=dev_data
    swapped_rows=[swapped_row_counterfactual(r) for r in dev_rows]
    swapped=tensors(swapped_rows)
    if not torch.equal(original_mask,swapped[1]):
        raise AssertionError("swapped source changed schedule")
    nuisance_rows=[nuisance_distractor_counterfactual(r) for r in dev_rows]
    eligible=[i for i,x in enumerate(nuisance_rows) if x is not None]
    baseline_logits,baseline_table=model(original_x,original_mask)
    swapped_logits,_=model(swapped[0],swapped[1])
    base_pred=baseline_logits.argmax(-1)
    swap_pred=swapped_logits.argmax(-1)
    nuisance_pred={}
    if eligible:
        eligible_data=tensors([nuisance_rows[i] for i in eligible])
        nuisance_logits,_=model(eligible_data[0],eligible_data[1])
        nuisance_pred={idx:int(p) for idx,p in zip(eligible,nuisance_logits.argmax(-1).tolist())}
    records=[]
    for i,record in enumerate(dev_rows):
        bpred=int(base_pred[i]);spred=int(swap_pred[i])
        original_label=int(original_y[i]);new_label=int(swapped[2][i])
        label_swap_valid=new_label!=original_label
        if not label_swap_valid:
            raise AssertionError("unavailable legal swap")
        has_nuisance=i in nuisance_pred
        npred=nuisance_pred.get(i)
        table_true=targets(original_x)[i]
        table_rows=baseline_table.argmax(-1)[i]
        rowwise=int((table_rows==table_true).sum().item())
        data={
            "dev_index":i,
            "source_id":record["id"],
            "rule_table_id":record["rule_table_id"],
            "query_value":record["events"][-1]["value"],
            "original_true":original_label,
            "original_pred":bpred,
            "original_correct":int(bpred==original_label),
            "rowwise_table_correct":rowwise,
            "whole_table_exact":int(rowwise==8),
            "swapped_true":new_label,
            "swapped_pred":spred,
            "swapped_correct":int(spred==new_label),
            "both_original_and_swap_correct":int(bpred==original_label and spred==new_label),
            "nuisance_eligible":int(has_nuisance),
            "nuisance_pred":npred,
            "nuisance_prediction_stable":int(npred==bpred) if has_nuisance else None,
            "nuisance_correct":int(npred==original_label) if has_nuisance else None,
        }
        records.append(data)
    eligible_n=len(eligible)
    return records,{
        "development_examples":len(records),
        "swapped_rule_accuracy":sum(x["swapped_correct"] for x in records)/len(records),
        "joint_base_and_swapped_accuracy":sum(x["both_original_and_swap_correct"] for x in records)/len(records),
        "nuisance_eligible_count":eligible_n,
        "nuisance_prediction_invariance":sum(x["nuisance_prediction_stable"] for x in records if x["nuisance_eligible"])/eligible_n if eligible_n else None,
        "nuisance_accuracy":sum(x["nuisance_correct"] for x in records if x["nuisance_eligible"])/eligible_n if eligible_n else None,
        "no_sealed_data_used":True
    }


def fit_seed(seed:int,source_dir:Path,outdir:Path):
    if seed not in SEEDS:
        raise ValueError("outside registered six-seed cohort")
    torch.set_num_threads(1)
    groups,manifest=checked_sources(source_dir)
    torch.manual_seed(seed)
    initial=QueryConditionedTableMemory().state_dict()
    models={}
    optimizers={}
    for arm in ARMS:
        m=QueryConditionedTableMemory()
        m.load_state_dict(copy.deepcopy(initial))
        models[arm]=m
        optimizers[arm]=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0.)
    ref=list(models[ARMS[0]].state_dict().values())
    if not all(torch.equal(a,b) for a,b in zip(ref,models[ARMS[1]].state_dict().values())):
        raise AssertionError("model starts differ")
    if sum(p.numel() for p in models[ARMS[0]].parameters())!=51520:
        raise AssertionError("unexpected parameter count")
    datasets={arm:groups[arm][1] for arm in ARMS}
    dev_data=groups["development"][1]
    sampler=torch.Generator().manual_seed(seed+20261116)
    receipts=[]
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,1024,(BATCH,),generator=sampler)
        for arm in ARMS:
            xb,mb,yb=(a[ids] for a in datasets[arm])
            m=models[arm]
            opt=optimizers[arm]
            opt.zero_grad(set_to_none=True)
            pred,table=m(xb,mb)
            loss=F.cross_entropy(pred,yb)+WEIGHT*F.cross_entropy(table.reshape(-1,8),targets(xb).flatten())
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}, arm={arm}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
            opt.step()
        if step in EVAL_STEPS:
            for arm in ARMS:
                row={"seed":seed,"step":step,"arm":arm,
                     "train":score(models[arm],datasets[arm]),
                     "development":score(models[arm],dev_data)}
                receipts.append(row)
                print(json.dumps(row,sort_keys=True),flush=True)
    outdir.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    controls={}
    for arm in ARMS:
        m=models[arm]
        per_case,cf_stats=dev_controls(m,groups["development"][0],dev_data)
        controls[arm]=cf_stats
        path=outdir/f"roc05_{arm}_S{seed}_step1920.pt"
        torch.save({
            "study":"ROC-05A-independent-rule-pool-and-model-seed-replication-v1",
            "seed":seed, "arm":arm, "optimizer_steps":UPDATES,
            "source_sha256":{name:manifest["source_files"][filename]["sha256"]
                              for name,filename in FILES.items()},
            "state_dict":m.state_dict(),
            "optimizer_state_dict":optimizers[arm].state_dict(),
        },path)
        checkpoints[arm]={"sha256":sha(path),"size":path.stat().st_size}
        per_case_path=outdir/f"roc05_{arm}_S{seed}_development_predictions.csv"
        with per_case_path.open("w",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(per_case[0]))
            writer.writeheader();writer.writerows(per_case)
        controls[arm]["per_case_sha256"]=sha(per_case_path)
    final={row["arm"]:row for row in receipts if row["step"]==UPDATES}
    difference=(final["high_permutation512"]["development"]["accuracy"]
                -final["low_affine4"]["development"]["accuracy"])
    result={
        "study":"ROC-05A-independent-rule-pool-and-model-seed-replication-v1",
        "registered_config":"configs/roc05_fresh_permutation_replication_v1.json",
        "registration_commit":"a56c177bbc956274fbf65018b5724d291e25ef3c",
        "seed":seed,"steps":UPDATES,"batch":BATCH,
        "identical_start":True,"identical_sample_indices":True,
        "source_manifest_sha256":sha(source_dir/"source_manifest.json"),
        "milestones":receipts,"final_arms":final,
        "primary_dev_accuracy_difference":difference,
        "causal_dev_controls":controls,
        "original_roc02_sealed_evaluated":False,
        "roc05_reserved_test_evaluated":False,
        "checkpoints":checkpoints,
    }
    path=outdir/f"roc05_S{seed}_receipt.json"
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"seed":seed,"primary_difference":difference,
                      "final":final,"controls":controls,"checkpoint_hashes":checkpoints},
                     sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--source-dir",required=True,type=Path)
    p.add_argument("--outdir",required=True,type=Path)
    a=p.parse_args()
    fit_seed(a.seed,a.source_dir,a.outdir)


if __name__=="__main__":
    main()
