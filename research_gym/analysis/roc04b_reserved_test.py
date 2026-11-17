"""ROC-04B one-shot independently reserved 384-table permutation evaluation.

Preregistered at f032c52db2ebb544b745b3972f5f86f4c47c8b38
before constructing reserved test source cases or scoring any checkpoint.
Loads the exact ROC-04A reproducibility ZIP; all twelve frozen model
weights are verified by SHA-256 before evaluation. There are no optimizer
updates and no reads of the original ROC-02 sealed family 6/7 inputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import zipfile
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from research_gym.analysis.roc03b_query_table_memory import (
    QueryConditionedTableMemory,targets
)
from research_gym.analysis.roc02_tied_train import tensors
from research_gym.analysis.roc02_delayed_rule_source import symbolic_execute
from research_gym.analysis.roc04_permutation_source import (
    source_pools,source_schedules,make_case,table_id,bytes_jsonl,
    swapped_row_counterfactual,nuisance_distractor_counterfactual,
    EXCLUDED_ORIGINAL
)

ARMS=("low_affine4","high_permutation512")
SEEDS=(2026111705,2026111707,2026111711,2026111713,2026111717,2026111719)
TEST_N=384
SCHEDULE_SEED=20261117101
CONFIG_PATH=Path("configs/roc04b_reserved_permutation_confirmation_v1.json")


def checksum(raw:bytes):
    return hashlib.sha256(raw).hexdigest()


def generate_reserved(cfg):
    source=cfg["data"]
    if source["test_skeleton_seed"]!=SCHEDULE_SEED or source["reserved_permutation_count"]!=TEST_N:
        raise AssertionError("reserved-test generation parameters changed")
    train,dev,reserved=source_pools()
    parent=cfg["parent_roc04a"]
    digest=lambda p:checksum(json.dumps(p).encode())
    assert digest(train)==parent["train_pool_digest"]
    assert digest(dev)==parent["development_pool_digest"]
    assert digest(reserved)==parent["reserved_table_identity_digest"]
    if len(reserved)!=TEST_N or len(set(reserved))!=TEST_N:
        raise AssertionError("reserve does not contain exactly 384 distinct tables")
    if set(reserved)&(set(train)|set(dev)|EXCLUDED_ORIGINAL):
        raise AssertionError("reserve overlaps exposed rules")
    schedules=source_schedules(SCHEDULE_SEED,TEST_N)
    cases=[make_case(schedule,table,"roc04b_reserved_test",i)
           for i,(schedule,table) in enumerate(zip(schedules,reserved))]
    if len({r["id"] for r in cases})!=TEST_N or len({r["rule_table_id"] for r in cases})!=TEST_N:
        raise AssertionError("reserved test source ID collision")
    for row in cases:
        label=symbolic_execute(row["events"])
        lookup=row["events"][0]["table"][row["events"][-1]["value"]]
        if label!={"status":"ok","label":row["label"]} or lookup!=row["label"]:
            raise AssertionError("independent oracle failure")
    return cases,bytes_jsonl(cases)


@torch.no_grad()
def evaluate(model,base_data,swap_data,nuisance_data,eligible,cases):
    x,mask,y=base_data
    bx,_,by=swap_data
    logits,table=model(x,mask)
    changed,_=model(bx,mask)
    if nuisance_data is not None:
        nlogits,_=model(nuisance_data[0],nuisance_data[1])
        nuisance_pred=dict(zip(eligible,nlogits.argmax(dim=1).tolist()))
    else:nuisance_pred={}
    pred=logits.argmax(dim=1)
    pred_swap=changed.argmax(dim=1)
    table_true=targets(x)
    correct_rows=(table.argmax(dim=2)==table_true).sum(dim=1)
    rows=[]
    for i in range(TEST_N):
        a=int(pred[i]);b=int(pred_swap[i])
        y0=int(y[i]);ys=int(by[i])
        if y0==ys:raise AssertionError("swapped legal source label unchanged")
        have=i in nuisance_pred
        z=nuisance_pred.get(i)
        rows.append({
            "test_index":i,
            "source_id":cases[i]["id"],
            "table_id":cases[i]["rule_table_id"],
            "query":cases[i]["events"][-1]["value"],
            "true":y0,"predicted":a,"correct":int(a==y0),
            "rowwise_correct":int(correct_rows[i]),
            "whole_table_correct":int(correct_rows[i]==8),
            "swapped_true":ys,"swapped_pred":b,"swapped_correct":int(b==ys),
            "both_original_swap_correct":int(a==y0 and b==ys),
            "nuisance_eligible":int(have),
            "nuisance_pred":z,
            "nuisance_stable":int(z==a) if have else None,
            "nuisance_correct":int(z==y0) if have else None,
        })
    if len(rows)!=TEST_N:raise AssertionError("test score missing cases")
    mask_correct=sum(r["correct"] for r in rows)
    eligible_count=len(eligible)
    score={
        "test_cases":TEST_N,
        "accuracy":mask_correct/TEST_N,
        "correct":mask_correct,
        "cross_entropy":float(F.cross_entropy(logits,y)),
        "row_reconstruction_accuracy":sum(r["rowwise_correct"] for r in rows)/(TEST_N*8),
        "whole_table_exact_accuracy":sum(r["whole_table_correct"] for r in rows)/TEST_N,
        "whole_table_exact_count":sum(r["whole_table_correct"] for r in rows),
        "legal_row_swap_accuracy":sum(r["swapped_correct"] for r in rows)/TEST_N,
        "both_original_swap_accuracy":sum(r["both_original_swap_correct"] for r in rows)/TEST_N,
        "nuisance_eligible":eligible_count,
        "nuisance_prediction_stability":sum(r["nuisance_stable"] for r in rows if r["nuisance_eligible"])/eligible_count if eligible_count else None,
        "nuisance_accuracy":sum(r["nuisance_correct"] for r in rows if r["nuisance_eligible"])/eligible_count if eligible_count else None,
    }
    return rows,score


def exact_signflip(d):
    observed=mean(d)
    outcomes=[mean(sign*a for sign,a in zip(signs,d))
              for signs in itertools.product((-1,1),repeat=6)]
    extreme=sum(v>=observed-1e-15 for v in outcomes)
    return extreme,len(outcomes),extreme/len(outcomes)


def run(cfg_path:Path,archive_path:Path,outdir:Path):
    torch.set_num_threads(1)
    cfg=json.loads(cfg_path.read_text())
    assert cfg["study"]=="ROC-04B-v1-independent-reserved-permutation-confirmation"
    if checksum(archive_path.read_bytes())!=cfg["parent_roc04a"]["model_archive_sha256"]:
        raise RuntimeError("ROC04A training archive SHA mismatch")
    archive=zipfile.ZipFile(archive_path)
    index=json.loads(archive.read("ROC04A_REPRODUCIBILITY_MANIFEST.json"))
    assert not index["includes_original_roc02_sealed_inputs"]
    assert not index["includes_roc04_reserved_test_examples"]
    expected_by_name={x["path"]:x for x in index["files"]}
    cases,raw=generate_reserved(cfg)
    if not raw or len(cases)!=TEST_N:raise AssertionError("sealed test generation failed")
    base=tensors(cases)
    swap_cases=[swapped_row_counterfactual(r) for r in cases]
    swap_data=tensors(swap_cases)
    if not torch.equal(base[1],swap_data[1]):
        raise AssertionError("counterfactual changed token timing")
    nuisances=[nuisance_distractor_counterfactual(r) for r in cases]
    eligible=[i for i,v in enumerate(nuisances) if v is not None]
    nuisance_data=tensors([nuisances[i] for i in eligible]) if eligible else None

    outputs=[];all_rows=[]
    for seed in SEEDS:
        cell={"seed":seed,"models":{}}
        for arm in ARMS:
            filename=f"roc04_{arm}_S{seed}_step1920.pt"
            data=archive.read(filename)
            archived=expected_by_name.get(filename)
            expected=cfg["frozen_model_weights_sha256"][str(seed)][arm]
            if (not archived or archived["sha256"]!=expected
                or checksum(data)!=expected or len(data)!=archived["bytes"]):
                raise RuntimeError("model checkpoint identity failure "+filename)
            loaded=torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
            if loaded["seed"]!=seed or loaded["arm"]!=arm or loaded["optimizer_steps"]!=1920:
                raise RuntimeError("model identity or training budget changed")
            model=QueryConditionedTableMemory()
            model.load_state_dict(loaded["state_dict"],strict=True)
            model.eval()
            rows,metrics=evaluate(model,base,swap_data,nuisance_data,eligible,cases)
            cell["models"][arm]=metrics
            for row in rows:
                all_rows.append({"seed":seed,"arm":arm,**row})
        cell["paired_accuracy_gain"]=cell["models"]["high_permutation512"]["accuracy"]-cell["models"]["low_affine4"]["accuracy"]
        outputs.append(cell)
    archive.close()
    ds=[x["paired_accuracy_gain"] for x in outputs]
    k,n,p=exact_signflip(ds)
    successes=sum(d>0 for d in ds)
    high_task=sum(x["models"]["high_permutation512"]["accuracy"]>=.70 for x in outputs)
    high_table=sum(x["models"]["high_permutation512"]["whole_table_exact_accuracy"]>=.50 for x in outputs)
    primary=(successes>=5 and mean(ds)>=.2 and p<=.05)
    competence=(high_task>=5 and high_table>=5)
    result={
        "study":cfg["study"],"registration_commit":"f032c52db2ebb544b745b3972f5f86f4c47c8b38",
        "original_roc02_sealed_evaluated":False,
        "roc04_reserved_test_evaluated":True,
        "model_training_updates_during_test":0,
        "frozen_checkpoint_count":12,
        "test_case_count":TEST_N,
        "unique_reserved_table_count":len({x["rule_table_id"] for x in cases}),
        "test_jsonl_sha256":checksum(raw),
        "source_pool_digest":cfg["parent_roc04a"]["reserved_table_identity_digest"],
        "per_seed":outputs,
        "primary_gain_high_minus_low":mean(ds),
        "positive_seeds":successes,
        "signflip_extreme":k,"signflip_total":n,"one_sided_exact_signflip_p":p,
        "primary_gate":"PASS" if primary else "FAIL",
        "high_70pct_task_seeds":high_task,
        "high_50pct_complete_table_seeds":high_table,
        "competence_gate":"PASS" if competence else "FAIL",
        "mean_test_accuracy":{arm:mean(x["models"][arm]["accuracy"] for x in outputs) for arm in ARMS},
        "mean_test_table_exact":{arm:mean(x["models"][arm]["whole_table_exact_accuracy"] for x in outputs) for arm in ARMS},
        "mean_legal_swap_accuracy":{arm:mean(x["models"][arm]["legal_row_swap_accuracy"] for x in outputs) for arm in ARMS},
        "mean_nuisance_stability":{arm:mean(x["models"][arm]["nuisance_prediction_stability"] for x in outputs) for arm in ARMS},
        "scope":"first and only ROC04B independent test on frozen original ROC04A model cohort"
    }
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"ROC04B_RESERVED_TEST_CASES.jsonl").write_bytes(raw)
    (outdir/"ROC04B_RESERVED_TEST_SUMMARY.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    csvpath=outdir/"ROC04B_PRIMARY_BY_SEED.csv"
    with csvpath.open("w",newline="") as f:
        fields=["seed","low_correct","high_correct","low_accuracy","high_accuracy","paired_gain","high_table_exact_count","high_legal_swap_accuracy","high_nuisance_stability"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in outputs:
            low=x["models"][ARMS[0]]
            high=x["models"][ARMS[1]]
            writer.writerow({
                "seed":x["seed"],"low_correct":low["correct"],"high_correct":high["correct"],
                "low_accuracy":low["accuracy"],"high_accuracy":high["accuracy"],
                "paired_gain":x["paired_accuracy_gain"],
                "high_table_exact_count":high["whole_table_exact_count"],
                "high_legal_swap_accuracy":high["legal_row_swap_accuracy"],
                "high_nuisance_stability":high["nuisance_prediction_stability"]
            })
    # Freeze exact per-case complete test predictions for independent re-scoring.
    packed=outdir/"ROC04B_PER_CASE_TEST.csv.gz"
    with packed.open("wb") as rawfp:
        with gzip.GzipFile(filename="",mode="wb",fileobj=rawfp,compresslevel=6,mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
    manifest={"study":cfg["study"],
              "origin_archive_sha256":cfg["parent_roc04a"]["model_archive_sha256"],
              "seal_original_roc02_families_6_7":True,
              "test_case_count":TEST_N,"model_count":12,"prediction_rows":len(all_rows),
              "files":{p.name:{"sha256":checksum(p.read_bytes()),"bytes":p.stat().st_size}
                       for p in outdir.iterdir() if p.is_file()}}
    (outdir/"ROC04B_RESULT_MANIFEST.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="per_seed"},sort_keys=True,indent=2))
    for x in outputs:
        print("SEED",x["seed"],"low",x["models"]["low_affine4"]["accuracy"],
              "high",x["models"]["high_permutation512"]["accuracy"],
              "delta",x["paired_accuracy_gain"],
              "table",x["models"]["high_permutation512"]["whole_table_exact_accuracy"])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=CONFIG_PATH)
    p.add_argument("--roc04a-archive",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.config,a.roc04a_archive,a.out_dir)


if __name__=="__main__":main()
