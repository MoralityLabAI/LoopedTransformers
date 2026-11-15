"""NRF-11 frozen-observer compact receipts and optional geometry checks."""
import csv
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch=None

class NRF11CompactReceipts(unittest.TestCase):
    def test_study_integrity(self):
        src=Path(__file__).resolve().parents[1]/"experiments/numerical_range_fixed_point_v1/nrf11_compact_receipts.csv"
        with src.open(newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),12)
        self.assertEqual(len({(x["seed"],x["depth"]) for x in rows}),12)
        for r in rows:
            self.assertEqual(len(r["checkpoint_sha256"]),64)
            self.assertEqual(len(r["observer_adapter_sha256"]),64)
            self.assertIn(r["selected_gamma"],("0.05","0.2","1.0"))
            self.assertLess(float(r["new_joint_J"]),float(r["new_supervised_J"]))
            self.assertLess(float(r["new_joint_J"]),float(r["new_task_only_J"]))
            self.assertLess(float(r["new_joint_J"]),float(r["new_random_median_J"]))
            self.assertAlmostEqual(float(r["new_joint_J"]),float(r["new_joint_mse"])+float(r["new_joint_leakage"])**2,places=11)
        self.assertEqual(sum(r["primary_passed"]=="True" for r in rows if r["depth"]=="32"),5)
        self.assertEqual(sum(r["primary_passed"]=="True" for r in rows if r["depth"]=="16"),6)

@unittest.skipIf(torch is None,"PyTorch needed for native observer fixture")
class NRF11AdapterGeometry(unittest.TestCase):
    def test_nrf10_nrf11_registered_fold_contract(self):
        from research_gym.analysis import nrf10_leakage_aware_observer as n10
        from research_gym.analysis.nrf09_matched_optimizer_steps import (
            gaussian_inputs as n09_inputs)
        from research_gym.analysis.nrf04_semantic_fiber import gaussian_stream
        self.assertTrue(torch.equal(
            n10.gaussian_inputs(131, 1, 2)[0],
            n09_inputs(131, 1, 2)[0]))
        seen = []
        for fold in (9, 10, 11, 12, 13):
            x, y = n10.gaussian_inputs(131, fold, 2)
            self.assertEqual(tuple(x.shape), (2, 128))
            self.assertEqual(tuple(y.shape), (2, 8))
            self.assertTrue(torch.equal(x, n10.gaussian_inputs(131, fold, 2)[0]))
            stream = gaussian_stream(20261115 + 7919 * 131 + 104729 * fold)
            oracle = torch.tensor(
                [[next(stream) for _ in range(128)] for _ in range(2)],
                dtype=torch.float64)
            self.assertTrue(torch.equal(x, oracle))
            seen.append(x)
        self.assertTrue(all(not torch.equal(seen[0], z) for z in seen[1:]))
        for seed, fold, count in ((101, 9, 2), (131, 8, 2),
                                  (131, 14, 2), (131, 12, 0)):
            with self.assertRaises(ValueError):
                n10.gaussian_inputs(seed, fold, count)

    def test_projection_qr_and_ridge_shapes(self):
        from research_gym.analysis.nrf10_leakage_aware_observer import ridge_from_U
        torch.set_num_threads(1)
        raw=torch.randn(128,8,dtype=torch.float64,generator=torch.Generator().manual_seed(7))
        U=torch.linalg.qr(raw)[0]
        H=torch.randn(160,128,dtype=torch.float64)
        Y=torch.randn(160,8,dtype=torch.float64)
        head=ridge_from_U(H,Y,U,H.mean(0),Y.mean(0))
        self.assertEqual(head.shape,(8,8))
        self.assertLess(float(torch.linalg.norm(U.T@U-torch.eye(8,dtype=torch.float64))),1e-9)

if __name__=="__main__":
    unittest.main()
