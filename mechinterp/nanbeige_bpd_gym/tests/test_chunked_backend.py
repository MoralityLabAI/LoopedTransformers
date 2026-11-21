"""CPU construction contracts; no real quantization, model or CUDA inference."""
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from mechinterp.nanbeige_bpd_gym.chunked_backend import prepare_nf4_linear,build_nf4_loop,attach_cached_runtime


class FakeLinear4bit(nn.Linear):
    def __init__(self,a,b,*,bias,compute_dtype,compress_statistics,quant_type):
        super().__init__(a,b,bias=bias)
        self.compute_dtype=compute_dtype;self.compress_statistics=compress_statistics;self.quant_type=quant_type


def fake_params(data,*,requires_grad,compress_statistics,quant_type):
    value=nn.Parameter(data,requires_grad=requires_grad)
    value.compress_statistics=compress_statistics;value.quant_type=quant_type
    return value


FAKE=types.SimpleNamespace(nn=types.SimpleNamespace(Linear4bit=FakeLinear4bit,Params4bit=fake_params))


class ChunkedBackendTests(unittest.TestCase):
    def test_constructor_initialization_has_no_cpu_weight_storage(self):
        source=nn.Linear(32,48,bias=False,dtype=torch.bfloat16).requires_grad_(False)
        before=source.weight.clone();initializations=[];original=nn.Linear.reset_parameters
        def reset(module):
            initializations.append((module.weight.device.type,module.weight.shape));original(module)
        with patch.object(nn.Linear,'reset_parameters',reset):wrapped=prepare_nf4_linear(source,bnb=FAKE)
        self.assertEqual(initializations,[('meta',torch.Size([48,32]))])
        self.assertEqual(wrapped.weight.data_ptr(),source.weight.data_ptr())
        self.assertTrue(torch.equal(source.weight,before));self.assertTrue(torch.equal(wrapped.weight,before))
        self.assertEqual(wrapped.compute_dtype,torch.bfloat16)
        self.assertEqual(wrapped.quant_type,'nf4');self.assertTrue(wrapped.weight.compress_statistics)
        self.assertFalse(wrapped.training);self.assertFalse(wrapped.weight.requires_grad)
    def test_invalid_sources_and_cpu_quantization_are_refused(self):
        for source in (nn.Identity(),nn.Linear(4,5),nn.Linear(4,5,bias=False),
                       nn.Linear(4,5,bias=False,dtype=torch.bfloat16),
                       nn.Linear(4,5,bias=False,dtype=torch.bfloat16,device='meta').requires_grad_(False)):
            with self.assertRaises(ValueError):prepare_nf4_linear(source,bnb=FAKE)
        with self.assertRaisesRegex(ValueError,'CUDA'):build_nf4_loop(None,'cpu')
    def test_complete_cache_and_head_attachment_preserves_source_and_pruning(self):
        from .test_chunked_cache import Loop,Layer,Cache
        from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache
        loop=Loop();loop.layers=nn.ModuleList([Layer(i) for i in range(22)])
        loop.base.layers=loop.layers;loop.n=22;loop.n_gpu_layers=22;loop.stage_layer=None
        loop.model.lm_head=nn.Linear(4,64,bias=False,dtype=torch.float64).requires_grad_(False)
        before=loop.model.lm_head.weight.clone()
        # Only the tiny fixture's cache class replaces the Transformers cache.
        def make_cache(*args,**kwargs):return ChunkedTwoLoopCache(*args,**kwargs,cache_factory=Cache)
        with patch('mechinterp.nanbeige_bpd_gym.chunked_cache.ChunkedTwoLoopCache',make_cache):
            cache,head=attach_cached_runtime(loop,prefill_chunk=2,vocabulary_chunk=7)
        try:
            first=cache.prefill([[1,3,5,7],[2,4]])
            self.assertEqual(len(first['states']),80)
            cache.select_rows([1]);last=cache.step([11]);self.assertEqual(cache.physical_length,5)
            self.assertEqual(cache.semantic_lengths,[3]);self.assertEqual(len(last['states']),40)
            for capture in (first,last):
                for boundary in capture['boundaries']:
                    expected=torch.cat([nn.functional.linear(boundary,before[s:s+7]).float() for s in range(0,64,7)],-1)
                    self.assertTrue(torch.equal(head(boundary),expected))
            self.assertTrue(torch.equal(loop.model.lm_head.weight,before))
            self.assertEqual(head.staging_bytes,7*4*8)
        finally:cache.close();head.close()


if __name__=='__main__':unittest.main()
