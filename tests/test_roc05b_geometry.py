"""ROC05B numerical geometry/causal source checks on CPU synthetic weights."""
from __future__ import annotations
import unittest
import torch
from research_gym.analysis.roc03b_query_table_memory import QueryConditionedTableMemory
from research_gym.analysis.roc02_tied_train import tensors
from research_gym.analysis.roc05_permutation_source import (
    build_train_and_dev,swapped_row_counterfactual
)
from research_gym.analysis.roc05b_matched_rule import (
    midpoint,suffix_logits,query_ids,suffix_tangent,observer_bases,
    geom_metric,rand_basis,soft_input_states,RADII
)

class ROC05BGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(20261117)
        cls.model=QueryConditionedTableMemory().eval()
        cls.train_low,_,cls.development,_=build_train_and_dev()

    def test_midpoint_suffix_exact_native_route(self):
        x,mask,y=tensors(self.development[:6])
        with torch.no_grad():
            h=midpoint(self.model,x,mask)
            q=query_ids(x,mask)
            log=suffix_logits(self.model,h,q)
            native=self.model(x,mask)[0]
        self.assertLess(float((log-native.double()).abs().max()),2e-6)

    def test_analytic_suffix_tangent_against_autograd(self):
        x,m,_=tensors(self.development[:1])
        h=midpoint(self.model,x,m)[0]
        q=query_ids(x,m)
        T=suffix_tangent(self.model,h)
        R=self.model.table_head.weight.detach().double().reshape(8,8,128)[int(q[0])]
        expected=R@T
        def real_logits(t):
            return suffix_logits(self.model,t[None].float(),q)[0]
        torch.manual_seed(2)
        v=torch.randn(128,dtype=torch.float64)
        eps=1e-3
        dv=(real_logits(h.double()+eps*v)-real_logits(h.double()-eps*v))/(2*eps)
        self.assertLess(float(torch.linalg.vector_norm(dv-expected@v)),.002)

    def test_train_only_rank_seven_and_nulls(self):
        train=tensors(self.train_low[:96])
        h=midpoint(self.model,train[0],train[1])
        U,meta=observer_bases(h,train[2])
        I=torch.eye(7,dtype=torch.float64)
        for basis in U.values():
            self.assertEqual(tuple(basis.shape),(128,7))
            self.assertLess(float(torch.linalg.norm(basis.T@basis-I)),1e-9)
        self.assertGreater(meta["supervised_s7"],0)
        t=suffix_tangent(self.model,h[0])
        R=self.model.table_head.weight.detach().double().reshape(8,8,128)[0]
        for kind,frame in U.items():
            met=geom_metric(t,R,frame)
            self.assertIsNotNone(met)
            self.assertGreaterEqual(met["b"],0)
        first=rand_basis('ab'*32,0,941001)
        again=rand_basis('ab'*32,0,941001)
        self.assertTrue(torch.equal(first,again))

    def test_legal_source_swap_and_matched_input_radius(self):
        record=self.development[0]
        changed=swapped_row_counterfactual(record)
        self.assertNotEqual(record["label"],changed["label"])
        a=tensors([record]);b=tensors([changed])
        self.assertTrue(torch.equal(a[1],b[1]))
        with torch.no_grad():
            h0=midpoint(self.model,a[0],a[1])
            hf=midpoint(self.model,b[0],b[1])
            matches=soft_input_states(self.model,a[0],b[0],a[1],h0,hf)
        self.assertEqual(tuple(matches),RADII)
        for r,rec in matches.items():
            self.assertEqual(len(rec["eligible"]),1)
            if bool(rec["eligible"][0]):
                self.assertLess(abs(float(rec["actual_l2"][0])-r),2e-5)
            else:
                self.assertEqual(rec["ineligible_count"],1)

if __name__=="__main__":
    unittest.main()
