"""No snapshot required: buffer admission, scoped hashes and observer lifetime."""
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from .lazy_bf16_storage import CheckpointStore
from .lazy_bf16_storage_tests import save_checkpoint
from .lazy_bf16_loop import make_lazy_loop,digest_dispatch,clear_unused_row_observations


def model_at(device='cpu'):
    import torch
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(8,8,bias=False,device=device,dtype=torch.bfloat16)
            self.register_buffer('inv_freq',torch.ones(2,device=device),persistent=False)
        def forward(self,h,**kwargs):return (self.linear(h),)
    m=torch.nn.Module();m.config=SimpleNamespace(hidden_size=8)
    m.model=torch.nn.Module();m.model.embed_tokens=torch.nn.Embedding(16,8,device=device,dtype=torch.bfloat16)
    m.model.layers=torch.nn.ModuleList([Layer(),Layer()]);m.model.norm=torch.nn.LayerNorm(8,device=device,dtype=torch.bfloat16)
    m.lm_head=torch.nn.Linear(8,16,bias=False,device=device,dtype=torch.bfloat16)
    return m.eval().requires_grad_(False),Layer


class LoopTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.path=Path('runs/lazy_bf16_loop_MOCK_ONLY_tests')/uuid.uuid4().hex
        self.dense,self.layer_cls=model_at();save_checkpoint(self.dense,self.path);self.store=CheckpointStore(self.path)
        self.meta,_=model_at('meta')
        self.store.bind_parameters(self.meta,[n for n,p in self.meta.named_parameters() if not n.startswith('model.layers.1.')])
        self.stage=self.layer_cls().eval().requires_grad_(False)
    def tearDown(self):
        del self.dense,self.meta,self.stage,self.store;gc.collect()
        for p in self.path.iterdir():p.unlink()
        self.path.rmdir()
    def initialize(self):
        import torch
        self.meta.model.layers[0].inv_freq=torch.ones(2)

    def test_uninitialized_resident_and_stage_buffers_are_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError,'resident prefix buffers'):make_lazy_loop(self.meta,self.store,'cpu',1,self.stage)
        self.initialize();self.stage.inv_freq=torch.ones(2,device='meta')
        with self.assertRaisesRegex(ValueError,'stage buffers'):make_lazy_loop(self.meta,self.store,'cpu',1,self.stage)

    def test_scoped_dispatch_restores_original_even_on_error(self):
        import experiments.nan_recomp09.common as common
        self.initialize();loop=make_lazy_loop(self.meta,self.store,'cpu',1,self.stage);original=common.target_digest
        expected=original(SimpleNamespace(model=self.dense))
        with self.assertRaisesRegex(RuntimeError,'fixture exit'):
            with digest_dispatch() as digest:
                self.assertEqual(digest(loop),expected)
                self.assertEqual(digest(SimpleNamespace(model=self.dense)),expected)
                raise RuntimeError('fixture exit')
        self.assertIs(common.target_digest,original)
        del loop

    def test_repeated_stage_copy_retains_no_CPU_decoder(self):
        import torch
        self.initialize();loop=make_lazy_loop(self.meta,self.store,'cpu',1,self.stage)
        for _ in range(5):
            active=loop._load_stage(loop.layers[1]);self.assertIs(active,self.stage)
            self.assertTrue(torch.equal(active.linear.weight,self.dense.model.layers[1].linear.weight))
        self.assertEqual(loop.stage_copies,5);self.assertEqual(loop.maximum_source_tensor_bytes,128)
        self.assertTrue(self.meta.model.layers[1].linear.weight.is_meta)
        with self.assertRaises(ValueError):loop._load_stage(self.layer_cls())
        del loop,active

    def test_clear_only_duplicate_legacy_observer(self):
        import torch
        from experiments.nan_full26.batch import RowInstrument
        row=RowInstrument.__new__(RowInstrument);row.base=SimpleNamespace(raw=[torch.ones(4) for _ in range(3)])
        row.raw_by_phase=[[torch.arange(4)],[torch.arange(4)+1]];before=row.raw_by_phase
        self.assertEqual(clear_unused_row_observations(row),3);self.assertEqual(row.base.raw,[])
        self.assertIs(row.raw_by_phase,before);self.assertTrue(torch.equal(row.raw_by_phase[1][0],torch.arange(4)+1))
        with self.assertRaises(TypeError):clear_unused_row_observations(SimpleNamespace(raw=[]))


if __name__=='__main__':unittest.main()
