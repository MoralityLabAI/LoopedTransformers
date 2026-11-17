"""ROC05B independent frozen primary paired six-seed CSV verification."""
import csv,itertools
import unittest
from pathlib import Path

P=Path("experiments/roc05_fresh_replication/ROC05B_PRIMARY_BY_SEED.csv")
SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)

class ROC05BPrimaryReceipts(unittest.TestCase):
    def test_full_original_primary_and_64_signs(self):
        with P.open(newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(tuple(int(x["seed"]) for x in rows),SEEDS)
        values=[]
        for x in rows:
            self.assertEqual(int(x["paired_eligible"]),256)
            a=float(x["observed_rule_margin_gain_high"])
            b=float(x["soft_input_margin_gain_high"])
            delta=float(x["observed_minus_soft_high"])
            self.assertLess(abs(a-b-delta),1e-12)
            self.assertEqual(int(x["native_correct_high"])<256,True)
            self.assertEqual(int(x["legal_full_changed_correct_high"])<256,True)
            self.assertGreaterEqual(float(x["observed_minus_soft_low"]),0)
            values.append(delta)
        mean=sum(values)/6
        count=sum(sum(s*v for s,v in zip(signs,values))/6 >=mean-1e-15
                  for signs in itertools.product((-1,1),repeat=6))
        self.assertAlmostEqual(mean,-0.06805337873205038,places=12)
        self.assertEqual(count,64)
        self.assertEqual(count/64,1.0)
        self.assertEqual(sum(v>0 for v in values),0)
        self.assertEqual(sum(int(x["native_correct_high"]) for x in rows),1147)
        self.assertEqual(sum(int(x["legal_full_changed_correct_high"]) for x in rows),1157)

if __name__=="__main__":
    unittest.main()
