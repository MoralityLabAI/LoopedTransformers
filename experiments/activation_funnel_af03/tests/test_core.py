"""AF-03 invariants, oracle counterfactuals, patch gradients and end-to-end smoke."""
import unittest

import torch

from experiments.activation_funnel_af02 import gym
from experiments.activation_funnel_af02.model import NativeTeacher
from experiments.activation_funnel_af03.core import (TokenCodec, paired, patch_code,
                                                    donor_roster, synthetic_extra_waits)


class AF03CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cases = gym.canonical_cases('train')
        cls.roster = donor_roster(cls.cases)

    def test_roster_excludes_same_status_and_has_independent_composed_oracle(self):
        self.assertEqual(len(self.roster),108)
        self.assertEqual(sum(map(len,self.roster.values())),216)
        for i,pairs in self.roster.items():
            self.assertEqual(len(pairs),2)
            for p in pairs:
                self.assertNotEqual(gym.audit_targets(self.cases[i])[p.first_probed_branch],
                                    p.audit_target[p.first_probed_branch])
                self.assertEqual(gym.target_action(self.cases[p.hybrid_index]),p.expected_action)

    def test_patch_masks_affect_only_declared_tokens(self):
        for width in (1,2):
            rx=torch.zeros(2,5*width)
            dx=torch.ones_like(rx)
            first=torch.tensor([0,1])
            for condition,allowed in [('addressed',[(0,3),(1,4)]),
                                      ('other',[(1,4),(0,3)]),
                                      ('goal',[(2,),(2,)]),
                                      ('whole',[(0,1,2,3,4)]*2),
                                      ('self',[(),()])]:
                actual=patch_code(rx,dx,first,width,condition).reshape(2,5,width)
                for k in range(2):
                    active={i for i in range(5) if actual[k,i].any()}
                    self.assertEqual(active,set(allowed[k]))

    def test_full_code_patch_equals_direct_hybrid_after_cut(self):
        torch.manual_seed(7)
        model=NativeTeacher()
        for width in (1,2):
            codec=TokenCodec(width)
            cases=[self.cases[0],self.cases[1]]
            p=[self.roster[0][0],self.roster[1][1]]
            rx=torch.tensor([gym.feature_sequence(c) for c in cases])
            dx=torch.tensor([gym.feature_sequence(self.cases[q.donor_index]) for q in p])
            first=torch.tensor([q.first_probed_branch for q in p])
            out=paired(model,codec,rx,dx,first,cut_event=p[0].cut_event,
                       condition='whole',direct=True)
            self.assertTrue(torch.allclose(out.task_logits,out.direct_task_logits,atol=2e-5,rtol=0))
            self.assertTrue(torch.allclose(out.audit_logits,out.direct_audit_logits,atol=2e-5,rtol=0))

    def test_causal_patch_backward_reaches_native_cell_and_codec(self):
        torch.manual_seed(11)
        model=NativeTeacher();codec=TokenCodec(1)
        p=self.roster[0][0]
        rx=torch.tensor([gym.feature_sequence(self.cases[p.receiver_index])])
        dx=torch.tensor([gym.feature_sequence(self.cases[p.donor_index])])
        first=torch.tensor([p.first_probed_branch])
        out=paired(model,codec,rx,dx,first,cut_event=p.cut_event)
        loss=out.audit_logits.square().mean()+out.task_logits.square().mean()
        loss.backward()
        self.assertGreater(sum(float(t.grad.square().sum()) for t in model.cell.parameters() if t.grad is not None),0)
        self.assertGreater(sum(float(t.grad.square().sum()) for t in codec.parameters() if t.grad is not None),0)

    def test_synthetic_waits_are_correct_tensor_only_extrapolations(self):
        case=gym.canonical_cases('delay4')[0]
        original=torch.tensor(gym.feature_sequence(case))
        for extra,visits in ((1,16),(4,22)):
            future=synthetic_extra_waits(case,extra)
            self.assertEqual(2*len(future),visits)
            self.assertTrue(torch.equal(future[:len(original)-1],original[:-1]))
            self.assertTrue(torch.equal(future[-1],original[-1]))
            self.assertEqual(float(future[-extra-1:-1,:8].abs().sum()),0.)

if __name__=='__main__':unittest.main()
