"""LBF-11 mathematical loss and frozen-gradient engineering tests."""
from __future__ import annotations
import math
import unittest
import torch
import torch.nn.functional as F

from ppt_gym.core import supervised_loss, Config, PlasticProofTransformer
from ppt_gym.lbf11_loss import objective, pass_terms, _validate_labels, _audit_model


class LossFactorialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.full=[1,0,1,0]
        self.single=[1,0,0,0]

    def outputs(self):
        return [
            {"logits":torch.tensor([.3,-.5,.9,.2],requires_grad=True),
             "outcome":torch.tensor([-.6,.7,.1,-.4],requires_grad=True)},
            {"logits":torch.tensor([-.2,1.1,.8,-.1],requires_grad=True),
             "outcome":torch.tensor([.1,.2,-.3,.5],requires_grad=True)}
        ]

    def test_legacy_endpoints_exact_loss_and_gradients(self):
        for arm,labels in (("b1_r1",self.single),("bs_rs",self.full)):
            out=self.outputs()
            variables=[v for p in out for v in (p["logits"],p["outcome"])]
            new=objective(out,self.full,self.single,arm)
            old=supervised_loss(out,tuple(labels))
            self.assertTrue(torch.equal(new,old),(arm,new,old))
            ga=torch.autograd.grad(new,variables,retain_graph=True)
            gb=torch.autograd.grad(old,variables)
            for x,y in zip(ga,gb): self.assertTrue(torch.equal(x,y))

    def test_factorial_mixed_targets(self):
        out=self.outputs()
        terms=[pass_terms(p,self.full,self.single) for p in out]
        expected=sum(x["bs"]+x["r1"] for x in terms)/2
        torch.testing.assert_close(objective(out,self.full,self.single,"bs_r1"),expected)
        expected=sum(x["b1"]+x["rs"] for x in terms)/2
        torch.testing.assert_close(objective(out,self.full,self.single,"b1_rs"),expected)

    def test_balanced_is_marginal_plus_conditional_kl(self):
        out=self.outputs()[0]
        t=pass_terms(out,self.full,self.single)
        pos=out["logits"][[0,2]]
        kl=-F.log_softmax(pos,0).mean()-math.log(2)
        torch.testing.assert_close(t["rb"],t["rs"]+kl)
        g=torch.autograd.grad(t["rb"],out["logits"])[0]
        want=out["logits"].detach().softmax(0)-torch.tensor([.5,0,.5,0])
        torch.testing.assert_close(g,want)

    def test_all_positive_marginal_gradient_vanishes(self):
        out=self.outputs()[0]
        terms=pass_terms(out,[1,1,1,1],[1,0,0,0])
        g=torch.autograd.grad(terms["rs"],out["logits"],retain_graph=True)[0]
        self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        balanced=torch.autograd.grad(terms["rb"],out["logits"])[0]
        self.assertGreater(float(balanced.norm()),0)
        self.assertEqual(float(terms["rs"].detach()),0.0)

    def test_single_positive_all_arms_equivalent(self):
        out=self.outputs()
        legacy=supervised_loss(out,tuple(self.single))
        for arm in ("b1_r1","bs_r1","b1_rs","bs_rs","bs_balanced"):
            torch.testing.assert_close(objective(out,self.single,self.single,arm),legacy)

    def test_reject_non_native_singleton_and_censored(self):
        for a,b in (([1,1],[0,1]),([1,0],[0,1]),([1,-1],[1,0]),([1,1],[1,1])):
            with self.assertRaises(ValueError): _validate_labels(a,b)
