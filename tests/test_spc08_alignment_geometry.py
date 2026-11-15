"""Small, checkpoint-independent invariance checks for SPC-08A overlap."""
import unittest
import torch
from experiments.spectral_state_cost_v0.spc08_alignment_geometry import overlap_summary

class SPC08AlignmentTests(unittest.TestCase):
    def test_identical_subspaces_have_unit_overlap(self):
        torch.manual_seed(74)
        A=torch.linalg.qr(torch.randn(48,48,dtype=torch.float64)).Q.T
        bases={key:A for key in ('phase','immediate','temporal','delayed_null')}
        Q=torch.eye(48,dtype=torch.float64)
        result=overlap_summary(bases,bases,Q,Q)
        for row in result.values():
            for value in row.values():
                self.assertAlmostEqual(value,1.0,places=9)

    def test_gauge_conjugacy_restores_subspace_overlap(self):
        torch.manual_seed(87)
        A=torch.linalg.qr(torch.randn(48,48,dtype=torch.float64)).Q.T
        Q=torch.linalg.qr(torch.randn(48,48,dtype=torch.float64)).Q
        src={key:A for key in ('phase','immediate','temporal','delayed_null')}
        tar={key:A@Q for key in src}
        result=overlap_summary(src,tar,Q,torch.eye(48,dtype=torch.float64))
        for row in result.values():
            self.assertAlmostEqual(row['paired'],1.0,places=9)
            self.assertTrue(0<=row['raw']<=1+1e-9)

if __name__=='__main__':unittest.main()
