"""CPU smoke tests for prospective ROC-02 tied model, without opening test."""
import unittest
import torch
try:
    from research_gym.analysis.roc02_tied_train import (SEEDS,DEPTHS,UPDATES,BATCH,WIDTH,TiedRuleModel,tensors,evaluate)
    from research_gym.analysis.roc02_delayed_rule_source import sample_case
except ImportError:
    from roc02_tied_train import (SEEDS,DEPTHS,UPDATES,BATCH,WIDTH,TiedRuleModel,tensors,evaluate)
    from roc02_delayed_rule_source import sample_case
import random

class ROC02TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fixed_model_shape_and_shared_initialization(self):
        torch.manual_seed(SEEDS[0]);a=TiedRuleModel(8)
        torch.manual_seed(SEEDS[0]);b=TiedRuleModel(16)
        self.assertEqual(len(SEEDS),6)
        self.assertEqual(DEPTHS,(8,16))
        self.assertEqual(UPDATES,240)
        self.assertEqual(BATCH,8)
        self.assertEqual(WIDTH,128)
        for key,v in a.state_dict().items():
            self.assertTrue(torch.equal(v,b.state_dict()[key]))
        with self.assertRaises(ValueError):TiedRuleModel(6)

    def test_padded_token_mask_and_backprop(self):
        rng=random.Random(7)
        rec=[sample_case(rng,(0,1)) for _ in range(4)]
        x,m,y=tensors(rec)
        model=TiedRuleModel(8)
        self.assertEqual(tuple(x.shape[:1]),(4,))
        self.assertEqual(tuple(x.shape[-1:]),(79,))
        self.assertEqual(tuple(model(x,m).shape),(4,8))
        loss=torch.nn.functional.cross_entropy(model(x,m),y)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in model.parameters()))
        stats=evaluate(model,(x,m,y))
        self.assertTrue(0<=stats['accuracy']<=1)
        bad=x[:,:,:-1]
        with self.assertRaises(ValueError):model(bad,m)

if __name__=='__main__':unittest.main()
