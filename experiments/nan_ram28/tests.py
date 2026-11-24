"""Synthetic CPU storage and RAM-policy checks; no checkpoint forwards."""
import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .loader import mapped_tensor,materialize
from .resources import soft_guard,HostMemoryBudgetReached

class StorageContracts(unittest.TestCase):
    def test_readonly_mapping_survives_file_scope_and_collection(self):
        with tempfile.TemporaryDirectory(prefix='.ram28-storage-',dir=Path.cwd()) as directory:
            assert Path(directory).resolve().parent==Path.cwd().resolve()
            path=Path(directory)/'data.bin';offset=65536+34
            values=torch.arange(8,dtype=torch.float32)
            path.write_bytes(b'x'*offset+values.numpy().tobytes())
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            result=mapped_tensor(path,offset,32,torch.float32,[2,4]);gc.collect()
            self.assertTrue(torch.equal(result,values.reshape(2,4)))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            del result;gc.collect()

    @unittest.skipUnless(os.name=='nt','Windows commit accounting')
    def test_readonly_view_avoids_full_COW_commit_charge(self):
        from .resources import memory
        with tempfile.TemporaryDirectory(prefix='.ram28-commit-',dir=Path.cwd()) as directory:
            assert Path(directory).resolve().parent==Path.cwd().resolve()
            path=Path(directory)/'sparse.bin';size=128*1024**2
            with path.open('wb') as stream:stream.truncate(size)
            before=memory()['private_bytes']
            with path.open('rb') as stream:copy=mmap.mmap(stream.fileno(),size,access=mmap.ACCESS_COPY)
            copy_charge=memory()['private_bytes']-before;copy.close();gc.collect()
            before=memory()['private_bytes'];view=mapped_tensor(path,0,size,torch.uint8,[size])
            read_charge=memory()['private_bytes']-before
            self.assertGreater(copy_charge,size*0.8)
            self.assertLess(read_charge,16*1024**2)
            del view;gc.collect()

    def test_complete_meta_model_gets_exact_BF16_parameters(self):
        with tempfile.TemporaryDirectory(prefix='.ram28-state-',dir=Path.cwd()) as directory:
            assert Path(directory).resolve().parent==Path.cwd().resolve()
            root=Path(directory)
            model=torch.nn.Linear(9,7,bias=True,device='meta',dtype=torch.bfloat16)
            tensors={name:torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape).to(torch.bfloat16)/16 for name,p in model.state_dict().items()}
            header={};parts=[];offset=0
            for name,t in tensors.items():
                raw=t.view(torch.uint8).numpy().tobytes();parts.append(raw)
                header[name]=dict(dtype='BF16',shape=list(t.shape),data_offsets=[offset,offset+len(raw)]);offset+=len(raw)
            encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
            (root/'data.safetensors').write_bytes(len(encoded).to_bytes(8,'little')+encoded+b''.join(parts))
            (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={k:'data.safetensors' for k in tensors})))
            info=materialize(model,root)
            self.assertEqual(info['mapped_parameter_count'],2)
            self.assertTrue(all(not p.is_meta and not p.requires_grad for p in model.parameters()))
            for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,tensors[name]))
            x=torch.ones((2,9),dtype=torch.bfloat16)
            self.assertTrue(torch.equal(model(x),torch.nn.functional.linear(x,tensors['weight'],tensors['bias'])))
            del value,model;gc.collect()

    def test_bad_storage_extent_is_refused(self):
        with self.assertRaises(AssertionError):mapped_tensor('never-opened',0,7,torch.float32,[2])

    def test_RAM_guard_preserves_exact_budget_boundary(self):
        cfg=dict(minimum_available_RAM_bytes=100,maximum_worker_private_bytes=200)
        with patch('experiments.nan_ram28.resources.memory',return_value=dict(available_physical_bytes=100,private_bytes=200)):
            self.assertEqual(soft_guard(cfg)['private_bytes'],200)

    @unittest.skipUnless(os.name=='nt','Windows process counters')
    def test_supervisor_can_measure_its_actual_PID(self):
        from .resources import memory
        sample=memory(os.getpid())
        self.assertGreater(sample['available_physical_bytes'],0)
        self.assertGreater(sample['private_bytes'],0)
        self.assertGreater(sample['working_set_bytes'],0)

    def test_RAM_guard_refuses_low_available_or_high_private(self):
        cfg=dict(minimum_available_RAM_bytes=100,maximum_worker_private_bytes=200)
        for available,private in ((99,200),(100,201)):
            with patch('experiments.nan_ram28.resources.memory',return_value=dict(available_physical_bytes=available,private_bytes=private)):
                with self.assertRaises(HostMemoryBudgetReached):soft_guard(cfg)

    def test_scope_and_numerical_budgets_are_inherited_with_BF16_deferred(self):
        cfg=json.loads(Path('experiments/nan_ram28/config.json').read_bytes())
        original=json.loads(Path('experiments/nan_mlp27/config.json').read_bytes())
        for key,value in original.items():
            if key not in ('experiment','backends'):self.assertEqual(cfg[key],value,key)
        self.assertEqual(cfg['backends'],['nf4'])
        self.assertEqual(cfg['baseline_contexts_per_backend'],1549)
        self.assertEqual(cfg['AB_worlds_per_route']*len(cfg['AB_routes']),216)

if __name__=='__main__':unittest.main()
