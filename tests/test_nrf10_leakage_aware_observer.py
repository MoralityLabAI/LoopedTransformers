"""NRF-10 numerical and compact-receipt regression tests."""
import csv
import math
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "PyTorch needed for native numeric fixtures")
class NRF10GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_shear_and_basis_rotation(self):
        from research_gym.analysis.nrf10_leakage_aware_observer import leakage_values
        eye = torch.eye(128, dtype=torch.float64)
        U = eye[:, [0, 2, 3, 4, 5, 6, 7, 8]]
        self.assertAlmostEqual(float(leakage_values(U, eye.unsqueeze(0))[0]), 0., places=12)
        J = eye.clone()
        J[0, 1] = 2.
        expected = 2 / math.sqrt(8)
        self.assertAlmostEqual(float(leakage_values(U, J.unsqueeze(0))[0]), expected, places=12)
        Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64,
             generator=torch.Generator().manual_seed(11)))
        self.assertAlmostEqual(float(leakage_values(U @ Q, J.unsqueeze(0))[0]), expected, places=12)

    def test_dense_jacobian_matches_reverse_jvp(self):
        from research_gym.analysis.nrf10_leakage_aware_observer import dense_jacobians, leakage_values
        from research_gym.analysis.nrf04_semantic_fiber import leakage_at
        torch.manual_seed(3)
        w1 = torch.randn(128,128,dtype=torch.float64)*.03
        b1 = torch.randn(128,dtype=torch.float64)*.01
        w2 = torch.randn(128,128,dtype=torch.float64)*.03
        b2 = torch.randn(128,dtype=torch.float64)*.01
        params=(w1,b1,w2,b2)
        states=torch.randn(2,128,dtype=torch.float64)
        U=torch.linalg.qr(torch.randn(128,8,dtype=torch.float64))[0]
        vals=leakage_values(U,dense_jacobians(params,states,16))
        direct=[leakage_at(params,h,8,U.T) for h in states]
        for a,b in zip(vals.tolist(),direct):
            self.assertAlmostEqual(a,b,places=9)

class NRF10ReceiptTests(unittest.TestCase):
    def test_frozen_results(self):
        base=Path(__file__).resolve().parents[1]
        source=base/"experiments/numerical_range_fixed_point_v1/nrf10_compact_receipts.csv"
        with source.open(newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),12)
        self.assertEqual(len({(r["seed"],r["depth"]) for r in rows}),12)
        for r in rows:
            self.assertEqual(len(r["checkpoint_sha256"]),64)
            for name in ("supervised","task_only","joint"):
                e,l,j=(float(r[f"{name}_{term}"]) for term in ("mse","leakage","J"))
                self.assertAlmostEqual(e+l*l,j,places=11)
            self.assertIn(r["selected_gamma"],("0.05","0.2","1.0"))
            self.assertLess(float(r["orthogonality_error"]),1e-9)
            self.assertLess(float(r["joint_J"]),float(r["task_only_J"]))
        r16=[r for r in rows if r["depth"]=="16"]
        r32=[r for r in rows if r["depth"]=="32"]
        self.assertEqual(sum(r["primary_passed"]=="True" for r in r16),6)
        self.assertEqual(sum(r["primary_passed"]=="True" for r in r32),5)

if __name__=="__main__":
    unittest.main()
