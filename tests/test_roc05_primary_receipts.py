"""ROC05A independent six-seed published first-run receipt test; stdlib only."""
from __future__ import annotations
import csv
import itertools
import unittest
from pathlib import Path

PATH=Path("experiments/roc05_fresh_replication/ROC05A_ORIGINAL_BY_SEED.csv")
SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)

def audit():
    with PATH.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if tuple(int(r["seed"]) for r in rows)!=SEEDS:
        raise AssertionError("wrong original model seed list")
    low=high=whole=0;diffs=[]
    for r in rows:
        if int(r["dev_cases"])!=256:
            raise AssertionError("wrong development case count")
        a,b,t=(int(r[f]) for f in ("low_correct","high_correct","high_whole_table_count"))
        if not all(0<=n<=256 for n in (a,b,t)):
            raise AssertionError("invalid case count")
        assert abs(float(r["low_accuracy"])-a/256)<1e-14
        assert abs(float(r["high_accuracy"])-b/256)<1e-14
        assert abs(float(r["paired_gain"])-(b-a)/256)<1e-14
        assert abs(float(r["high_whole_table_accuracy"])-t/256)<1e-14
        assert 0<=float(r["high_legal_swap_accuracy"])<=1
        assert 0<=float(r["high_nuisance_prediction_stability"])<=1
        diffs.append(b-a);low+=a;high+=b;whole+=t
    observed=sum(diffs)
    extrema=sum(sum(x*d for x,d in zip(signs,diffs))>=observed
               for signs in itertools.product((-1,1),repeat=6))
    p=extrema/64
    pos=sum(d>0 for d in diffs)
    task=sum(int(r["high_correct"])/256>=.70 for r in rows)
    tables=sum(int(r["high_whole_table_count"])/256>=.50 for r in rows)
    return {"low":low,"high":high,"whole":whole,
      "paired_gain":observed/(256*6),"positive_seeds":pos,"one_sided_p":p,
      "task_seeds":task,"table_seeds":tables,
      "primary_pass":pos>=5 and observed/(256*6)>=.20 and p<=.05,
      "competence_pass":task>=5 and tables>=5}

class ROC05AResultAudit(unittest.TestCase):
    def test_original_first_run_six_seed_gate_and_competence(self):
        x=audit()
        self.assertEqual((x["low"],x["high"],x["whole"]),(315,1147,206))
        self.assertEqual(x["paired_gain"],832/1536)
        self.assertEqual(x["one_sided_p"],1/64)
        self.assertEqual(x["positive_seeds"],6)
        self.assertEqual((x["task_seeds"],x["table_seeds"]),(6,0))
        self.assertTrue(x["primary_pass"])
        self.assertFalse(x["competence_pass"])

if __name__=="__main__":
    unittest.main()
