"""NRF-09 exact rational and frozen retrospective margin audit."""
import csv
import unittest
from pathlib import Path
from research_gym.analysis.nrf09_median_drift_audit import objective,median8,audit_pair

class NRF09DriftTests(unittest.TestCase):
    @staticmethod
    def arms(e,l):
        return {"supervised":{"heldout_mse":e,"local_fiber_leakage":l},**{
            f"random_{i}":{"heldout_mse":2*e,"local_fiber_leakage":l} for i in range(8)}}

    def test_median_complete_arms(self):
        self.assertEqual(median8(self.arms(1,0)),2)
        a=self.arms(1,0)
        a['random_0']={"heldout_mse":0,"local_fiber_leakage":10}
        self.assertEqual(median8(a),2)

    def test_certified_winner_persists(self):
        old={"seed":131,"rounds":16,"optimizer_steps":16,"arms":self.arms(1,0)}
        new={"seed":131,"rounds":16,"optimizer_steps":32,"arms":self.arms(1.01,0)}
        r=audit_pair(old,new)
        self.assertTrue(r["sign_certified"])

    def test_mismatched_pair_rejected(self):
        old={"seed":131,"rounds":16,"optimizer_steps":16,"arms":self.arms(1,0)}
        new={"seed":137,"rounds":16,"optimizer_steps":32,"arms":self.arms(1,0)}
        with self.assertRaises(ValueError):
            audit_pair(old,new)

    def test_compact_drift_receipts(self):
        root=Path(__file__).resolve().parents[1]/"experiments/numerical_range_fixed_point_v1"
        with (root/"nrf09_margin_drift_certificates.csv").open() as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),12)
        for r in rows:
            before=float(r["margin_before_16_updates"])
            after=float(r["margin_after_32_updates"])
            bound=float(r["certified_armwise_drift_bound"])
            self.assertLessEqual(abs(after-before),bound+1e-10)
            if r["certificate_passed"]=="true":
                self.assertGreater(abs(before),bound)
                self.assertGreater(before*after,0)
        self.assertEqual(sum(r["certificate_passed"]=="true" for r in rows if r["rounds"]=="16"),6)
        self.assertEqual(sum(r["certificate_passed"]=="true" for r in rows if r["rounds"]=="32"),0)

if __name__=="__main__":
    unittest.main()
