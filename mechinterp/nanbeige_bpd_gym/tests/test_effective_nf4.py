from types import SimpleNamespace
import unittest
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.effective_nf4 import EffectiveNF4Linear


class EffectiveNF4Tests(unittest.TestCase):
    def source(self):
        source = nn.Module()
        source.in_features = 4
        source.out_features = 3
        source.weight = nn.Parameter(torch.arange(6, dtype=torch.uint8).reshape(6, 1), requires_grad=False)
        source.weight.quant_state = SimpleNamespace(quant_type='nf4', nested=True,
            dtype=torch.bfloat16, shape=(3, 4))
        source.bias = nn.Parameter(torch.tensor([1., 2., 3.], dtype=torch.bfloat16), requires_grad=False)
        return source

    def test_preserves_packed_parameters_and_matches_common_effective_operator(self):
        source = self.source()
        matrix = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) / 8
        calls = []
        def dequantize(data, state):
            self.assertEqual(data.data_ptr(), source.weight.data_ptr())
            self.assertIs(state, source.weight.quant_state)
            calls.append(data.data_ptr())
            return matrix
        module = EffectiveNF4Linear(source, dequantize=dequantize)
        self.assertIs(module.weight, source.weight)
        self.assertIs(module.bias, source.bias)
        self.assertEqual(tuple(dict(module.named_parameters())), ('weight', 'bias'))
        x = torch.tensor([[[1., -2., 3., .5], [4., 2., -1., 0.]]], dtype=torch.float32)
        with torch.inference_mode():
            whole = module(x)
            tail = module(x[:, -1:])
            expected = torch.nn.functional.linear(x.to(torch.bfloat16), matrix, source.bias).float()
        torch.testing.assert_close(whole, expected, rtol=0, atol=0)
        torch.testing.assert_close(tail, whole[:, -1:], rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(whole.dtype, x.dtype)
        with self.assertRaises(RuntimeError):
            module(x)

    def test_rejects_changed_precision_or_quantization_contract(self):
        for key, value in (('nested', False), ('quant_type', 'fp4'), ('dtype', torch.float16), ('shape', (4, 3))):
            source = self.source()
            setattr(source.weight.quant_state, key, value)
            with self.assertRaises(ValueError):
                EffectiveNF4Linear(source, dequantize=lambda *args: None)


if __name__ == '__main__':
    unittest.main()
