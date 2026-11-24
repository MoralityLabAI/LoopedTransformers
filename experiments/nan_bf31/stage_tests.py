"""Synthetic actual-stage bytes and corruption detection; CPU only."""
import gc
from pathlib import Path
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop_tests import model_at
from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage_tests import save_checkpoint
from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore
from .stage_audit import audit


class Stage(unittest.TestCase):
    def setUp(self):
        self.path=Path('runs/BF31_MOCK_ONLY_contracts')/uuid.uuid4().hex
        self.model,Layer=model_at();save_checkpoint(self.model,self.path);self.store=CheckpointStore(self.path);self.stage=Layer().eval().requires_grad_(False)
        def load(layer):self.store.copy_parameters(self.stage,'model.layers.1');return self.stage
        self.loop=SimpleNamespace(layers=list(self.model.model.layers),n_gpu_layers=1,checkpoint_store=self.store,_load_stage=load)
    def tearDown(self):
        del self.model,self.stage,self.store,self.loop;gc.collect()
        for path in self.path.iterdir():path.unlink()
        self.path.rmdir()
    def test_actual_stage_bytes_are_checked(self):
        receipt=audit(self.loop,chunk_bytes=5);self.assertEqual(receipt['all_offloaded_layers'],1)
        self.assertEqual(receipt['pretrained_forwards'],0);self.assertEqual(receipt['checks'][0]['source_sha256'],receipt['checks'][0]['actual_stage_sha256'])
    def test_corrupt_copy_is_rejected_before_forward(self):
        import torch
        original=self.loop._load_stage
        def corrupt(layer):
            stage=original(layer)
            with torch.no_grad():stage.linear.weight[0,0]+=1
            return stage
        self.loop._load_stage=corrupt
        with self.assertRaisesRegex(AssertionError,'stage bytes differ'):audit(self.loop,chunk_bytes=5)


if __name__=='__main__':unittest.main()
