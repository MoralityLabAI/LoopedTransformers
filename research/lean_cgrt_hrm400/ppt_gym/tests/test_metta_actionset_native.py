"""CPU shape and source-only controls for OLR MeTTa native multi-witness gym."""
import unittest

import torch

from ppt_gym.metta_actionset_native import (
    ACTIONS, generate_roots, TiedSetScorer, fit,
)
from ppt_gym.metta_action_sets import normalized_action_set


class NativeTieContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_generator_has_ties_and_unique_source_roots(self):
        roots=generate_roots("train",12,20261128)
        self.assertEqual(len(roots),12)
        self.assertEqual(len({r.root_id for r in roots}),12)
        self.assertEqual(sum(r.mode=="tie" for r in roots),3)
        for root in roots:
            self.assertTrue(root.statement().endswith(", R"))
            good=[f"apply r{i}" for i,v in enumerate(root.targets)
                  if v==root.a+root.b]
            self.assertEqual(len(good),2 if root.mode=="tie" else 1)
            self.assertEqual(normalized_action_set(ACTIONS,good),
                             tuple(ACTIONS.index(x) for x in good))
        with self.assertRaises(ValueError):
            generate_roots("sealed",4,1)

    def test_recurrent_scorer_backpropagates_from_action_set_targets(self):
        model=TiedSetScorer()
        x=torch.tensor([[[.1,.2,.3],[.1,.2,.3],[.1,.2,.4]]],dtype=torch.float32)
        y=model(x,depth=3)
        self.assertEqual(tuple(y.shape),(1,3))
        from ppt_gym.metta_action_sets import objective
        loss=objective(y[0],ACTIONS,ACTIONS[:2],
            mode="set",root_id="test",state_sha256="public",model_seed=17)
        loss.backward()
        self.assertIsNotNone(model.embedding[0].weight.grad)
        self.assertTrue(torch.isfinite(model.embedding[0].weight.grad).all())
        with self.assertRaises(ValueError):
            model(x[:,:,:2])

    def test_matched_arms_start_from_identical_weights(self):
        roots=generate_roots("train",4,20261128)
        rows=[]
        for root in roots:
            optimal=[f"apply r{i}" for i,v in enumerate(root.targets)
                     if v==root.a+root.b]
            rows.append({"root_id":root.root_id,"state_sha256":root.root_id,
                         "legal":ACTIONS,"optimal":optimal,
                         "features":[[root.a/40,root.b/40,v/80] for v in root.targets]})
        a,ma=fit(rows,17,"canonical",steps=2)
        b,mb=fit(rows,17,"hashed",steps=2)
        c,mc=fit(rows,17,"set",steps=2)
        self.assertEqual(len({ma["initial_sha256"],mb["initial_sha256"],mc["initial_sha256"]}),1)
        self.assertEqual(ma["parameter_count"],mb["parameter_count"])
        self.assertEqual(ma["parameter_count"],mc["parameter_count"])
        self.assertEqual(ma["steps"],2)
        self.assertEqual(c(torch.tensor([rows[0]["features"]])).shape,(1,3))


if __name__=="__main__":
    unittest.main()
