"""CPU_ONLY wrapper/assembly checks; no quantization or pretrained forwards.

Real installed bitsandbytes construction is checked, including all pinned
projection shapes on meta. A small source tensor tests pre-quantizer byte and
recipe preservation. This cannot certify actual CUDA quantization identity.
"""
import argparse
from pathlib import Path
import unittest
from .bounded_cpu_engineering import CPUBudget


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();out=Path(a.out_dir);assert 'MOCK_ONLY' in out.name
    with CPUBudget(out) as budget:
        import hashlib
        import torch
        import bitsandbytes as bnb
        from unittest.mock import patch
        from experiments.nan_ram28.common import write,sha
        from .lazy_bf16_storage import CheckpointStore
        from .chunked_backend import prepare_nf4_linear
        from .tests import test_chunked_backend
        torch.set_num_threads(1);torch.manual_seed(2232001)
        assert not torch.cuda.is_initialized()
        source=torch.nn.Linear(32,48,bias=False,dtype=torch.bfloat16).requires_grad_(False)
        def digest(value):return hashlib.sha256(value.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
        before=digest(source.weight);initializations=[];reset=torch.nn.Linear.reset_parameters
        def record(module):
            initializations.append(dict(device=str(module.weight.device),shape=list(module.weight.shape),dtype=str(module.weight.dtype)))
            reset(module)
        with patch.object(torch.nn.Linear,'reset_parameters',record):wrapped=prepare_nf4_linear(source,bnb=bnb)
        assert initializations==[dict(device='meta',shape=[48,32],dtype='torch.float32')]
        # Reproduce the old pre-quantizer state on this tiny matrix only.
        reference=bnb.nn.Linear4bit(32,48,bias=False,compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
        reference.weight=bnb.nn.Params4bit(source.weight.detach().contiguous(),requires_grad=False,
            compress_statistics=True,quant_type='nf4')
        attributes=('blocksize','compress_statistics','quant_type','quant_storage','bnb_quantized')
        matched={name:str(getattr(wrapped.weight,name)) for name in attributes}
        assert all(getattr(wrapped.weight,name)==getattr(reference.weight,name) for name in attributes)
        assert wrapped.compute_dtype==reference.compute_dtype==torch.bfloat16
        assert wrapped.quant_state is None and wrapped.weight.quant_state is None and not wrapped.weight.bnb_quantized
        assert digest(wrapped.weight)==digest(source.weight)==before
        assert wrapped.weight.data_ptr()==source.weight.data_ptr()
        store=CheckpointStore(a.snapshot);shapes=[]
        with patch.object(torch.nn.Linear,'reset_parameters',record):
            for name,span in store.spans.items():
                if not name.startswith('model.layers.') or len(span.shape)!=2:continue
                budget.check()
                with torch.device('meta'):
                    value=bnb.nn.Linear4bit(span.shape[1],span.shape[0],bias=False,
                        compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
                assert value.weight.is_meta and value.weight.numel()*4==span.length*2
                shapes.append(dict(name=name,shape=list(span.shape),discarded_float32_constructor_bytes_avoided=value.weight.numel()*4))
                del value
        assert len(shapes)==154 and all(row['device']=='meta' for row in initializations)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_chunked_backend)
        result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
        assert not torch.cuda.is_initialized();budget.check()
        write(out/'receipt_MOCK_ONLY.json',dict(MOCK_ONLY=True,status='passed_prequantizer_and_runtime_attachment_CPU_only',
            source_sha256={name:sha(Path(__file__).parent/name) for name in ('chunked_backend.py','chunked_backend_engineering.py',
                'chunked_cache.py','streamed_vocab_head.py','tests/test_chunked_backend.py','tests/test_chunked_cache.py')},
            installed_bitsandbytes_version=bnb.__version__,torch_version=torch.__version__,constructor_initializations=initializations,
            prequantizer_parameter_bytes_unchanged=True,prequantizer_source_alias=True,prequantizer_recipe=matched,
            pinned_decoder_meta_shapes=shapes,largest_avoided_temporary_CPU_matrix_bytes=max(x['discarded_float32_constructor_bytes_avoided'] for x in shapes),
            cumulative_constructor_bytes_avoided=sum(x['discarded_float32_constructor_bytes_avoided'] for x in shapes),
            cumulative_bytes_are_not_simultaneous_RAM_or_measured_savings=True,unit_tests_run=result.testsRun,
            actual_CUDA_quantization_calls=0,pretrained_parameters_materialized=0,pretrained_forwards=0,
            CUDA_initialized=False,actual_NF4_target_digest_verified=False,actual_CUDA_capacity_verified=False,
            active_frozen_workers_modified=False,sealed_test_accessed=False))


if __name__=='__main__':main()
