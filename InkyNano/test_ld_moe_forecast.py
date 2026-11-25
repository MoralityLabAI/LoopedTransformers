import unittest
import numpy as np
import torch

from ld_moe_forecast import causal_features, successor_indices, Mesh, MODES, WIDTH
from test_ld_moe_cache import fixture
from ld_moe_cache import normalize_events


class ForecastTests(unittest.TestCase):
    def test_prefix_causality(self):
        e,s=fixture(3)
        e=normalize_events(e,s,1,2)
        for mode in MODES:
            full=list(causal_features(e,s,mode))
            prefix=list(causal_features(e[:61],s,mode))
            for (a,l,p),(b,l2,p2) in zip(full,prefix):
                np.testing.assert_array_equal(a,b)
                self.assertEqual((l,p),(l2,p2))

    def test_next_visit(self):
        e,s=fixture(2)
        targets=successor_indices(e)
        for i,j in targets.items():
            self.assertGreater(j,i)
            self.assertEqual(e[i]["layer"],e[j]["layer"])
            self.assertFalse(any(v["layer"]==e[i]["layer"] for v in e[i+1:j]))
        self.assertEqual(len(targets),len(e)-24)

    def test_block_messages_not_future(self):
        e,s=fixture()
        v=list(causal_features(e,s,"pass_mesh"))
        self.assertEqual(v[0][0][-512:].sum(),0.)
        # Block 0's message becomes visible only after layer 5 executes.
        self.assertEqual(v[5][0][-512:].sum(),0.)
        self.assertEqual(v[6][0][-512:-384].sum(),8.)

    def test_model_grad(self):
        m=Mesh()
        x=torch.ones(4,WIDTH)
        l=torch.tensor([0,6,12,18]);p=torch.zeros(4,dtype=torch.long)
        z=m(x,l,p)
        self.assertEqual(tuple(z.shape),(4,128))
        z.sum().backward()
        for bank in m.encoder:self.assertIsNotNone(bank.weight.grad)
        self.assertLess(sum(x.numel() for x in m.parameters()),250000)


if __name__=="__main__":unittest.main()
