"""Pure-stdlib independent ROC-04B reserved-rule test receipt and p-value audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean

SEEDS=(2026111705,2026111707,2026111711,2026111713,2026111717,2026111719)
ARMS=("low_affine4","high_permutation512")


def sha(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(folder:Path):
    summary=json.loads((folder/"ROC04B_RESERVED_TEST_SUMMARY.json").read_text())
    manifest=json.loads((folder/"ROC04B_RESULT_MANIFEST.json").read_text())
    cfg=json.loads(Path("configs/roc04b_reserved_permutation_confirmation_v1.json").read_text())
    if manifest["prediction_rows"]!=4608 or manifest["test_case_count"]!=384:
        raise AssertionError("wrong case/model census")
    for name,metadata in manifest["files"].items():
        p=folder/name
        if not p.is_file() or p.stat().st_size!=metadata["bytes"] or sha(p)!=metadata["sha256"]:
            raise AssertionError("missing or corrupted file: "+name)
    cases=[json.loads(line) for line in (folder/"ROC04B_RESERVED_TEST_CASES.jsonl").read_text().splitlines() if line]
    if len(cases)!=384 or len({r["rule_table_id"] for r in cases})!=384:
        raise AssertionError("incorrect independent held-out source table identity")
    for i,x in enumerate(cases):
        assert x["source_index"]==i
        assert x["family"]=="roc04b_reserved_test"
        table=x["events"][0]["table"]
        q=x["events"][-1]["value"]
        assert x["label"]==table[q]
    with gzip.open(folder/"ROC04B_PER_CASE_TEST.csv.gz","rt",newline="") as f:
        allrows=list(csv.DictReader(f))
    if len(allrows)!=4608:
        raise AssertionError("unexpected per-case test row count")
    bykey=defaultdict(list)
    for x in allrows:
        seed=int(x["seed"]);arm=x["arm"];i=int(x["test_index"])
        if seed not in SEEDS or arm not in ARMS or i<0 or i>=384:
            raise AssertionError("unknown model or test case")
        case=cases[i]
        table=case["events"][0]["table"];q=case["events"][-1]["value"]
        expected_swap=table[(q+4)%8]
        if (x["source_id"]!=case["id"] or x["table_id"]!=case["rule_table_id"]
            or int(x["true"])!=case["label"] or int(x["swapped_true"])!=expected_swap
            or expected_swap==case["label"]):
            raise AssertionError("counterfactual label or source mismatch")
        c=int(int(x["predicted"])==int(x["true"]))
        sc=int(int(x["swapped_pred"])==int(x["swapped_true"]))
        if (c!=int(x["correct"]) or sc!=int(x["swapped_correct"])
            or int(x["both_original_swap_correct"])!=c*sc):
            raise AssertionError("incorrect original/edited success bit")
        rowcount=int(x["rowwise_correct"])
        if not 0<=rowcount<=8 or int(x["whole_table_correct"])!=int(rowcount==8):
            raise AssertionError("incorrect eight-row reconstruction count")
        eligible=int(x["nuisance_eligible"])
        if eligible:
            n=int(x["nuisance_pred"])
            if (int(x["nuisance_stable"])!=int(n==int(x["predicted"]))
                or int(x["nuisance_correct"])!=int(n==int(x["true"]))):
                raise AssertionError("nuisance control semantic mismatch")
        else:
            if x["nuisance_pred"] not in ("",None):
                raise AssertionError("present nuisance prediction for ineligible case")
        bykey[(seed,arm)].append(x)
    if len(bykey)!=12 or any(len(v)!=384 for v in bykey.values()):
        raise AssertionError("model × case coverage incomplete")
    reported={x["seed"]:x for x in summary["per_seed"]}
    diffs=[]
    for seed in SEEDS:
        scores={}
        for arm in ARMS:
            rows=bykey[(seed,arm)]
            if sorted(int(r["test_index"]) for r in rows)!=list(range(384)):
                raise AssertionError("test-row duplicates")
            scores[arm]=sum(int(r["correct"]) for r in rows)/384
            m=reported[seed]["models"][arm]
            if abs(m["accuracy"]-scores[arm])>1e-12:
                raise AssertionError("reported accuracy/row mismatch")
            if abs(m["whole_table_exact_accuracy"]-
                    sum(int(r["whole_table_correct"]) for r in rows)/384)>1e-12:
                raise AssertionError("reported table/row mismatch")
        diff=scores["high_permutation512"]-scores["low_affine4"]
        diffs.append(diff)
        if abs(diff-reported[seed]["paired_accuracy_gain"])>1e-12:
            raise AssertionError("paired seed difference mismatch")
    measured=mean(diffs)
    outcomes=[mean(a*d for a,d in zip(signs,diffs))
              for signs in itertools.product((-1,1),repeat=6)]
    extreme=sum(x>=measured-1e-15 for x in outcomes)
    if (extreme!=summary["signflip_extreme"] or
        abs(extreme/64-summary["one_sided_exact_signflip_p"])>1e-12 or
        abs(measured-summary["primary_gain_high_minus_low"])>1e-12):
        raise AssertionError("independent six-seed significance mismatch")
    assert summary["model_training_updates_during_test"]==0
    assert summary["original_roc02_sealed_evaluated"] is False
    assert cfg["parent_roc04a"]["model_archive_sha256"]==manifest["origin_archive_sha256"]
    passed=(sum(d>0 for d in diffs)>=5 and measured>=.20 and extreme/64<=.05)
    if summary["primary_gate"]!=("PASS" if passed else "FAIL"):
        raise AssertionError("primary criterion mismatch")
    return {"status":"PASS","source_examples":384,"per_case_rows":4608,
            "independent_models":12,"six_seed_differences":diffs,
            "mean_high_minus_low":measured,
            "positive_seeds":sum(x>0 for x in diffs),
            "extreme_assignments":extreme,"exact_signflip_p":extreme/64,
            "primary_gate":summary["primary_gate"],
            "competence_gate":summary["competence_gate"],
            "original_roc02_families_6_7_evaluated":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--result-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=audit(a.result_dir)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
