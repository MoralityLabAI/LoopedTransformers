"""ROC-05B independent 128-null geometry audit, stdlib only.

Use the original ROC-05A first-run model checkpoints and the completed
ROC-05B 256-case DEV-only outputs, especially the later full-null replay
run 44107680072. No model training or sealed-source access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

SEEDS=(20261117231,20261117233,20261117239,
       20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
METRICS=("a","b","c","s")
NULL_SEEDS=tuple(range(941001,941129))

def digest(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a,b,label,tol=1e-12):
    if a is None or b is None:
        if a is not b:
            raise AssertionError(label+" undefined disagreement")
        return
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tol:
        raise AssertionError(label+f" {a} != {b}")

def empirical(value,samples,lower):
    if value is None or any(x is None for x in samples):
        return None
    count=sum(x<=value if lower else x>=value for x in samples)
    return (1+count)/(len(samples)+1)

def holm_full_six(p):
    q=[None]*6
    running=0.
    for j,(value,i) in enumerate(sorted((v,i) for i,v in enumerate(p) if v is not None)):
        running=max(running,min(1.,(6-j)*value))
        q[i]=running
    return q

def audit(folder:Path, config_path:Path):
    cfg=json.loads(config_path.read_text())
    assert cfg["study"]=="ROC-05B-v1-matched-hidden-input-causal-rule-intervention"
    n=0
    cards=0
    tags=[]
    b_ratios=[]
    for seed in SEEDS:
        for arm in ARMS:
            f=folder/f"roc05b_S{seed}_{arm}_receipt.json"
            r=json.loads(f.read_text())
            if (r["checkpoint_sha256"]!=cfg["parent"]["frozen_1920step_checkpoint_sha256"][str(seed)][arm]
                or r["source_dev_sha256"]!=cfg["parent"]["exact_source_sha256"]["development"]
                or r["sealed_test_evaluated"] or r["model_training_updates_during_causal_analysis"]!=0):
                raise AssertionError("frozen model/source identity or sealed status changed")
            if len(r.get("null_rows",[]))!=256 or len(r["shadow_cards"])!=4:
                raise AssertionError("missing complete 2x128 null records")
            for shadow in (0,1):
                original=[x for x in r["null_rows"] if x["shadow_index"]==shadow]
                if ([int(x["null_seed"]) for x in original]!=list(NULL_SEEDS)
                    or any(x["rank"]!=7 for x in original)):
                    raise AssertionError("null seed, order, or rank changed")
                if any(any(x[metric] is None or not math.isfinite(x[metric])
                           for metric in METRICS) for x in original):
                    raise AssertionError("null geometric metric undefined")
                expected={metric:median(x[metric] for x in original)
                          for metric in METRICS}
                observed=[x for x in r["shadow_cards"] if x["shadow_index"]==shadow]
                if len(observed)!=2 or {x["basis"] for x in observed}!={"supervised","pca"}:
                    raise AssertionError("wrong observer coverage at fixed shadow")
                for item in observed:
                    if (item["rank"]!=7 or not item["calibration_only"]
                        or item["null_distinct"]!=128):
                        raise AssertionError("null-calibration status broken")
                    for metric in METRICS:
                        close(expected[metric],item["null_medians"][metric],
                              f"{seed}/{arm}/{shadow}/{metric}/median",1e-11)
                    obs=item["metric"]
                    tests=[
                        empirical(obs["a"],[x["a"] for x in original],True),
                        empirical(obs["b"],[x["b"] for x in original],True),
                        empirical(obs["c"],[x["c"] for x in original],True),
                        empirical(obs["b"],[x["b"] for x in original],False),
                        empirical(obs["s"],[x["s"] for x in original],False),
                        empirical(obs["s"],[x["s"] for x in original],True),
                    ]
                    corrected=holm_full_six(tests)
                    if len(item["one_sided_p"])!=6 or len(item["holm_q"])!=6:
                        raise AssertionError("six-test family missing")
                    for i in range(6):
                        close(tests[i],item["one_sided_p"][i],
                              f"{seed}/{arm}/{shadow}/p{i}")
                        close(corrected[i],item["holm_q"][i],
                              f"{seed}/{arm}/{shadow}/holm{i}")
                    rb=item["ratios"]["b"]
                    if rb is not None:
                        close(rb,obs["b"]/expected["b"],
                              f"{seed}/{arm}/{shadow}/rank-b-ratio")
                        if arm=="high_permutation512" and item["basis"]=="supervised":
                            b_ratios.append(rb)
                    tags.append(tuple(item["tags"]))
                    cards+=1
                n+=len(original)
    assert cards==48 and n==3072
    return {
        "study":"ROC-05B independent full 128-basis null distribution check",
        "source":"original frozen 12-model dev-only causal replay",
        "model_count":12,"shadow_states":24,
        "rank_conditional_null_records":n,
        "observer_cards":cards,
        "six_one_sided_and_holm_tests_checked":cards*6,
        "unclassified_cards":sum(t==("unclassified",) for t in tags),
        "mean_high_supervised_B_over_rank_matched_null":sum(b_ratios)/len(b_ratios),
        "mean_B_count":len(b_ratios),
        "original_roc02_sealed_families_6_7_evaluated":False,
        "roc05_reserved_future_evaluated":False,
        "validation":"PASS"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt-dir",required=True,type=Path)
    p.add_argument("--config",type=Path,
                   default=Path("configs/roc05b_matched_rule_causal_v1.json"))
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=audit(a.receipt_dir,a.config)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__":
    main()
