"""Actual bnb pre-quantizer construction on CPU; no checkpoint or CUDA use."""
import hashlib
import importlib.util
import unittest
from unittest.mock import patch
import torch
from mechinterp.nanbeige_bpd_gym.chunked_backend import prepare_nf4_linear


@unittest.skipUnless(importlib.util.find_spec('bitsandbytes'),'Installed bnb construction lane required')
class ActualBnbMetaTests(unittest.TestCase):
    def test_real_bnb_source_recipe_and_no_throwaway_cpu_matrix(self):
        import bitsandbytes as bnb
        self.assertFalse(torch.cuda.is_initialized());torch.manual_seed(2232)
        source=torch.nn.Linear(32,48,bias=False,dtype=torch.bfloat16).requires_grad_(False)
        def digest(value):return hashlib.sha256(value.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
        before=digest(source.weight);initializers=[];reset=torch.nn.Linear.reset_parameters
        def record(module):initializers.append(module.weight.device.type);reset(module)
        reference=bnb.nn.Linear4bit(32,48,bias=False,compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
        reference.weight=bnb.nn.Params4bit(source.weight.detach().contiguous(),requires_grad=False,
            compress_statistics=True,quant_type='nf4')
        with patch.object(bnb.functional,'quantize_4bit',side_effect=AssertionError('Construction must not quantize')):
            with patch.object(torch.nn.Linear,'reset_parameters',record):
                actual=prepare_nf4_linear(source,bnb=bnb)
                # Seven pinned projection geometries; only meta is allocated.
                for output,input_ in ((6144,3072),(1024,3072),(1024,3072),(3072,6144),
                                      (10752,3072),(10752,3072),(3072,10752)):
                    with torch.device('meta'):
                        value=bnb.nn.Linear4bit(input_,output,bias=False,compute_dtype=torch.bfloat16,
                            compress_statistics=True,quant_type='nf4')
                    self.assertTrue(value.weight.is_meta);self.assertEqual(tuple(value.weight.shape),(output,input_))
            for field in ('blocksize','compress_statistics','quant_type','quant_storage','bnb_quantized'):
                self.assertEqual(getattr(actual.weight,field),getattr(reference.weight,field))
        self.assertEqual(initializers,['meta']*8)
        self.assertEqual(actual.compute_dtype,reference.compute_dtype)
        self.assertIsNone(actual.quant_state);self.assertIsNone(actual.weight.quant_state)
        self.assertEqual(actual.weight.data_ptr(),source.weight.data_ptr())
        self.assertEqual(digest(actual.weight),before);self.assertEqual(digest(source.weight),before)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
