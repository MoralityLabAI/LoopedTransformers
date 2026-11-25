"""ROC-06A analytic four-step VJP and train-only observer numerical gates."""
from __future__ import annotations
import unittest
import torch
from research_gym.analysis.roc03b_query_table_memory import QueryConditionedTableMemory
from research_gym.analysis.roc02_tied_train import tensors
from research_gym.analysis.roc05_permutation_source import (
    build_train_and_dev,swapped_row_counterfactual
)
from research_gym.analysis.roc06a_counterfactual_gradient import (
    midpoint, query_ids, margin_vjp_four_step, train_gradient_basis,
    observer_bases, suffix_logits, normalize, RADII
)

class GradientObserverGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(2026112506)
        cls.model=QueryConditionedTableMemory().eval()
        low,high,dev,reserved=build_train_and_dev()
        cls.train_rows=high
        cls.train=tensors(high)
        cls.dev_rows=dev

    def test_exact_legal_training_counterfactual(self):
        for r in self.train_rows[:32]:
            e=swapped_row_counterfactual(r)
            self.assertNotEqual(e["label"],r["label"])
            self.assertEqual(e["events"][1:],r["events"][1:])
            self.assertEqual(e["events"][-1]["value"],r["events"][-1]["value"])

    def test_analytic_reverse_vjp_matches_float64_autograd(self):
        rows=self.train_rows[:3]
        x,mask,y=tensors(rows)
        h=midpoint(self.model,x,mask)
        yp=torch.tensor([swapped_row_counterfactual(r)["label"] for r in rows])
        q=query_ids(x,mask)
        g=margin_vjp_four_step(self.model,h,q,y,yp)
        self.assertEqual(tuple(g.shape),(3,128))
        W1=self.model.block_in.weight.double()
        b1=self.model.block_in.bias.double()
        W2=self.model.block_out.weight.double()
        b2=self.model.block_out.bias.double()
        R=self.model.table_head.weight.double().reshape(8,8,128)
        for i in range(3):
            hi=h[i].detach().double().requires_grad_(True)
            z=hi
            for _ in range(4):
                z=z+.25*(W2@torch.tanh(W1@z+b1)+b2)
            contrast=(R[q[i],yp[i]]-R[q[i],y[i]])@z
            ref=torch.autograd.grad(contrast,hi)[0]
            self.assertLess(float(torch.max(torch.abs(ref-g[i]))),1e-10)
            v=torch.randn(128,dtype=torch.float64)
            eps=1e-5
            def value(point):
                z=point.clone()
                for _ in range(4):
                    z=z+.25*(W2@torch.tanh(W1@z+b1)+b2)
                return (R[q[i],yp[i]]-R[q[i],y[i]])@z
            fd=(value(hi.detach()+eps*v)-value(hi.detach()-eps*v))/(2*eps)
            self.assertLess(float(torch.abs(fd-g[i]@v)),1e-7)

    def test_all_1024_train_only_gradient_subspace(self):
        mid=midpoint(self.model,self.train[0],self.train[1])
        U,meta=train_gradient_basis(self.model,mid,self.train,self.train_rows)
        self.assertEqual(tuple(U.shape),(128,7))
        self.assertEqual(meta["train_gradient_count"],1024)
        self.assertGreater(meta["train_gradient_s7"],1e-9)
        self.assertLess(float(torch.max(torch.abs(U.T@U-torch.eye(7)))),1e-7)
        bases,_=observer_bases(mid,self.train[2])
        principal=torch.linalg.svdvals(U.T@bases["supervised"])
        self.assertEqual(tuple(principal.shape),(7,))
        self.assertLessEqual(float(principal.max()),1+1e-7)

    def test_normalized_observer_interventions_are_equal_cost(self):
        low=self.dev_rows[:3]
        x,mask,_=tensors(low)
        cf=tensors([swapped_row_counterfactual(r) for r in low])
        h0=midpoint(self.model,x,mask)
        h1=midpoint(self.model,cf[0],cf[1])
        basis,meta=train_gradient_basis(self.model,
            midpoint(self.model,self.train[0],self.train[1]),
            self.train,self.train_rows)
        delta=(h1-h0).double()
        proj=(delta@basis)@basis.T
        for r in RADII:
            shift,valid=normalize(proj,r)
            self.assertTrue(bool(valid.all()))
            self.assertLess(float((torch.linalg.vector_norm(shift,dim=1)-r).abs().max()),1e-10)
            q=query_ids(x,mask)
            z=suffix_logits(self.model,h0.double()+shift,q)
            self.assertTrue(torch.isfinite(z).all())
if __name__=="__main__":
    unittest.main()
