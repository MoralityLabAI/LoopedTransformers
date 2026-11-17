"""CPU structure tests for experimental Nanbeige 3050 NF4 backend.

Real 4-bit quantization, precision parity, VRAM and causal-effect measurements
require a local CUDA 4GiB checkpoint run. These mocks verify only the adapter
contracts without pretending the model was run.
"""
import types
import unittest

import torch
from mechinterp.dark_fiber_gyms.q4_3050 import (
    _slice_context, replace_linear4bit, Q4NanbeigeLoop,
)


class FakeParams4bit(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=False, **kwargs):
        return super().__new__(cls, data, requires_grad=requires_grad)


class FakeLinear4bit(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=True, **kwargs):
        super().__init__(input_features, output_features, bias=bias)


FAKE_BNB=types.SimpleNamespace(nn=types.SimpleNamespace(
    Linear4bit=FakeLinear4bit, Params4bit=FakeParams4bit))


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Linear(6,6,bias=False)
        self.nested=torch.nn.Sequential(torch.nn.Linear(6,6,bias=True))

    def forward(self, x):
        return self.nested(self.first(x))


class FakeLayer(torch.nn.Module):
    def forward(self, h, **kwargs):
        return (h + 0.05*h.roll(1, dims=1),)


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList([FakeLayer(),FakeLayer()])
        self.embed_tokens=torch.nn.Embedding(100,6)
        self.norm=torch.nn.Identity()
    def _update_causal_mask(self, *args):
        return None


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=FakeBase()
        self.lm_head=torch.nn.Linear(6,9,bias=False)


class TestQ4AdapterContracts(unittest.TestCase):
    def test_linear_replacement_preserves_placement_and_model_topology(self):
        torch.manual_seed(42)
        module=FakeDecoder()
        x=torch.randn(3,6)
        expected=module(x)
        n=replace_linear4bit(module,torch.device("cpu"),bnb=FAKE_BNB)
        self.assertEqual(n,2)
        self.assertIsInstance(module.first, FakeLinear4bit)
        self.assertIsInstance(module.nested[0], FakeLinear4bit)
        self.assertTrue(torch.allclose(module(x), expected))
        self.assertIsInstance(module.first.weight, FakeParams4bit)

    def test_context_slicing_preserves_position_and_mask(self):
        ctx={"pos":torch.arange(9).expand(16,-1),
             "mask":torch.ones(16,1,9,9),
             "cache_position":torch.arange(9)}
        out=_slice_context(ctx,3,5,16)
        self.assertEqual(out["pos"].shape,(2,9))
        self.assertEqual(out["mask"].shape,(2,1,9,9))
        self.assertEqual(out["cache_position"].shape,(9,))

    def test_recurrent_loop_microbatch_equivalence_and_readout(self):
        torch.manual_seed(3)
        model=FakeModel()
        a=Q4NanbeigeLoop(model,torch.device("cpu"),rows_per_batch=1)
        b=Q4NanbeigeLoop(model,torch.device("cpu"),rows_per_batch=16)
        tokens=torch.randint(0,100,(16,13))
        h,c=a.prepare(tokens)
        aa=a.run(h,0,2,c)
        bb=b.run(h,0,2,c)
        self.assertTrue(torch.allclose(aa,bb,atol=1e-6))
        self.assertTrue(torch.allclose(a.readout(aa,c),b.readout(bb,c)))
        with self.assertRaises(ValueError):
            a.run(h,0,1,c)


if __name__=="__main__":
    unittest.main()
