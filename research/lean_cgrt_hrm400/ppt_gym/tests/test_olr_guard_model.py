"""OLR-01C trainable recurrence and matched-message controls."""
import unittest

import torch

from ppt_gym.olr_guard_model import (
    GuardScorer, ranking_loss, choose, exact_arithmetic_reference,
)
from ppt_gym.olr_guard_fit import fit


class OLRGuardLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        # Public operands: 2+3=5 vs 2+3=6, then opposite order.
        self.features = [
            [[2/40,3/40,5/80],[2/40,3/40,6/80]],
            [[2/40,3/40,6/80],[2/40,3/40,5/80]],
        ]
        self.rows = [
            {"features":self.features[0],"labels":[1,0]},
            {"features":self.features[1],"labels":[0,1]},
        ]

    def test_exact_arithmetic_oracle_and_pair_flip(self):
        self.assertEqual(exact_arithmetic_reference(self.features[0]),0)
        self.assertEqual(exact_arithmetic_reference(self.features[1]),1)
        with self.assertRaises(ValueError):
            exact_arithmetic_reference([[0,0,1],[0,0,2]])

    def test_recurrence_is_differentiable_at_all_message_depths(self):
        x = torch.tensor(self.features,dtype=torch.float32)
        labels=torch.tensor([[1.,0.],[0.,1.]])
        for mode in ("cross","isolated","direct"):
            model=GuardScorer(width=16,mode=mode)
            logits,states=model(x,depth=4,return_states=True)
            self.assertEqual(tuple(logits.shape),(2,2))
            self.assertEqual(len(states),0 if mode=="direct" else 4)
            loss=ranking_loss(logits,labels)
            loss.backward()
            self.assertIsNotNone(model.embed[0].weight.grad)
            self.assertTrue(torch.isfinite(model.embed[0].weight.grad).all())
        with self.assertRaises(ValueError):
            GuardScorer(mode="unrecognized")

    def test_label_custody_rejects_ambiguous_feedback(self):
        model=GuardScorer(width=16)
        logits=model(torch.tensor(self.features,dtype=torch.float32))
        with self.assertRaises(ValueError):
            ranking_loss(logits,torch.tensor([[1.,1.],[0.,1.]]))
        with self.assertRaises(ValueError):
            model(torch.ones(1,4,3))

    def test_source_training_is_deterministic_and_frozen_body_stays_frozen(self):
        model, receipt=fit(self.rows,seed=17,arm="cross",steps=8,depth=2,width=16)
        self.assertEqual(receipt["source_worlds"],2)
        self.assertEqual(receipt["seed"],17)
        self.assertTrue(0 <= receipt["source_correct"] <= 2)
        self.assertGreater(receipt["parameters"],0)
        self.assertEqual(model(torch.tensor(self.features,dtype=torch.float32)).shape,(2,2))
        model2,receipt2=fit(self.rows,seed=17,arm="cross",steps=8,depth=2,width=16)
        self.assertEqual(receipt["last_loss"],receipt2["last_loss"])
        self.assertEqual(choose(model,self.features[0],depth=2),
                         choose(model2,self.features[0],depth=2))
        frozen, frozen_receipt=fit(self.rows,seed=17,arm="frozen_cross",steps=8,depth=2,width=16)
        self.assertLess(frozen_receipt["trained_parameters"],frozen_receipt["parameters"])
        self.assertFalse(frozen.core[0].weight.requires_grad)


if __name__=="__main__":
    unittest.main()
