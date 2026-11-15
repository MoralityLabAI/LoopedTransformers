"""Tests do not depend on pretrained checkpoint downloads."""
from __future__ import annotations
import unittest
import torch
try:
    from .causal_geometry import row_basis,rotate_vector,prepare_direction,geometry_certificate
    from .native_causal import summary,compare
except ImportError:
    from causal_geometry import row_basis,rotate_vector,prepare_direction,geometry_certificate
    from native_causal import summary,compare

class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(115)
        cls.head=torch.randn(6,32,dtype=torch.float64)
        cls.basis=row_basis(cls.head)
        cls.state=torch.randn(32,dtype=torch.float64)

    def test_basis_is_orthonormal(self):
        u=self.basis
        torch.testing.assert_close(u.T@u,torch.eye(u.shape[1],dtype=u.dtype),atol=1e-12,rtol=1e-12)

    def test_dark_rotation_exact_lens_and_norm(self):
        for sign in (-1,1):
            moved=rotate_vector(self.state,torch.arange(32,dtype=torch.float64),self.basis,sector='dark',angle=sign*0.05)
            torch.testing.assert_close(self.head@moved,self.head@self.state,atol=1e-10,rtol=1e-10)
            self.assertAlmostEqual(float(moved.norm()),float(self.state.norm()),places=12)
            self.assertGreater(float((moved-self.state).norm()),0)

    def test_visible_rotation_preserves_dark_component(self):
        moved=rotate_vector(self.state,torch.arange(32,dtype=torch.float64),self.basis,sector='visible',angle=.05)
        p=self.basis@self.basis.T
        torch.testing.assert_close((torch.eye(32,dtype=p.dtype)-p)@self.state,
                                   (torch.eye(32,dtype=p.dtype)-p)@moved,atol=1e-11,rtol=1e-11)
        self.assertAlmostEqual(float(moved.norm()),float(self.state.norm()),places=12)

    def test_full_and_scratch_preserve_norm(self):
        for s in ('full','scratch'):
            moved=rotate_vector(self.state,torch.arange(32,dtype=torch.float64),None,sector=s,angle=.05)
            self.assertAlmostEqual(float(self.state.norm()),float(moved.norm()),places=12)

    def test_visible_head_preimage(self):
        vec=torch.arange(6,dtype=torch.float64)
        converted=prepare_direction(vec,self.head,'visible')
        torch.testing.assert_close(self.head@converted,vec,atol=1e-10,rtol=1e-10)

    def test_reject_invalid_geometry(self):
        with self.assertRaises(ValueError):rotate_vector(self.state,torch.zeros(32,dtype=torch.float64),self.basis,sector='dark',angle=.05)
        with self.assertRaises(ValueError):rotate_vector(self.state,self.state,self.basis,sector='dark',angle=0)
        with self.assertRaises(ValueError):prepare_direction(torch.randn(5),self.head,'visible')

    def test_paired_causal_score(self):
        z=torch.tensor([[2.,1.,0.],[0.,2.,1.],[1.,0.,2.]])
        y=torch.tensor([0,1,2])
        clean=summary(z,y)
        moved=summary(torch.tensor([[0.,1.,2.],[0.,2.,1.],[1.,0.,2.]]),y)
        q=compare(clean,moved)
        self.assertEqual(q['clean_correct'],3)
        self.assertEqual(q['intervention_correct'],2)
        self.assertEqual(q['clean_to_wrong'],1)
        self.assertEqual(q['wrong_to_correct'],0)

if __name__=='__main__':unittest.main()
