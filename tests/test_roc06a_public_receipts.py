"""ROC-06A public six-seed primary result: independent stdlib arithmetic."""
from __future__ import annotations
import csv,itertools,math,unittest
from pathlib import Path
from statistics import mean

PATH=Path("experiments/roc06_train_gradient_observer/ROC06A_PRIMARY_BY_SEED.csv")
SEEDS=(20261117231,20261117233,20261117239,
       20261117241,20261117251,20261117257)

def audit():
    with PATH.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if tuple(int(r["seed"]) for r in rows)!=SEEDS:
        raise AssertionError("missing or reordered six original model seeds")
    d=[];sup=[];inp=[];grad=[]
    for r in rows:
        if int(r["eligible"])!=256:
            raise AssertionError("not all triple-matched original dev cases eligible")
        g=float(r["gradient_gain"])
        s=float(r["supervised_gain"])
        i=float(r["soft_input_gain"])
        best=max(s,i)
        if any(not math.isfinite(x) for x in (g,s,i)):
            raise AssertionError("nonfinite original model gain")
        if abs(float(r["gradient_minus_supervised"])-(g-s))>1e-12:
            raise AssertionError("gradient-supervised difference changed")
        if abs(float(r["gradient_minus_soft_input"])-(g-i))>1e-12:
            raise AssertionError("gradient-soft difference changed")
        if abs(float(r["gradient_minus_best"])-(g-best))>1e-12:
            raise AssertionError("conservative best-comparator difference changed")
        if r["best_reference"]!=("supervised_observed" if s>=i else "input_soft"):
            raise AssertionError("reference comparator selected post-hoc")
        if not(0<=int(r["original_correct"])<=256 and 0<=int(r["legal_changed_correct"])<=256):
            raise AssertionError("illegal original/changed native task count")
        d.append(g-best);sup.append(s);inp.append(i);grad.append(g)
    observed=mean(d)
    flips=[mean(a*b for a,b in zip(signs,d))
           for signs in itertools.product((-1,1),repeat=6)]
    extreme=sum(x>=observed-1e-15 for x in flips)
    p=extreme/len(flips)
    improved=sum(g>s for g,s in zip(grad,sup))
    return dict(
      mean_gradient=mean(grad),mean_supervised=mean(sup),mean_soft_input=mean(inp),
      gradient_minus_supervised=mean(g-s for g,s in zip(grad,sup)),
      registered_gradient_minus_best=observed,
      gradient_supervised_positive_seeds=improved,
      registered_positive_seeds=sum(v>0 for v in d),
      registered_signflip_extreme=extreme,registered_signflip_p=p,
      legal_correct=sum(int(r["legal_changed_correct"]) for r in rows),
      original_correct=sum(int(r["original_correct"]) for r in rows)
    )

class ROC06APublicAudit(unittest.TestCase):
    def test_six_seed_primary_and_transfer_specificity(self):
        a=audit()
        self.assertAlmostEqual(a["registered_gradient_minus_best"],-0.0566440431951681,places=12)
        self.assertAlmostEqual(a["gradient_minus_supervised"],0.011409252462423561,places=12)
        self.assertAlmostEqual(a["mean_gradient"],0.12937700875287797,places=12)
        self.assertAlmostEqual(a["mean_supervised"],0.11796775629045442,places=12)
        self.assertAlmostEqual(a["mean_soft_input"],0.18602105194804608,places=12)
        self.assertEqual(a["registered_signflip_extreme"],64)
        self.assertEqual(a["registered_signflip_p"],1.)
        self.assertEqual(a["registered_positive_seeds"],0)
        self.assertEqual(a["gradient_supervised_positive_seeds"],6)
        self.assertEqual(a["original_correct"],1147)
        self.assertEqual(a["legal_correct"],1157)

if __name__=="__main__":
    unittest.main()
