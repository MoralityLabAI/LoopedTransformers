"""Small CPU-only storage contracts; imported by the bounded engineer runner."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
from .lazy_bf16_storage import CheckpointStore,logical_target_digest


def toy(device='cpu',dtype=None):
    import torch
    dtype=dtype or torch.bfloat16
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(8,16,device=device,dtype=dtype)
            self.layers=torch.nn.ModuleList([torch.nn.Sequential(
                torch.nn.Linear(16,24,bias=True,device=device,dtype=dtype),torch.nn.SiLU(),
                torch.nn.Linear(24,16,bias=False,device=device,dtype=dtype)) for _ in range(4)])
            self.norm=torch.nn.LayerNorm(16,device=device,dtype=dtype)
    return Toy().eval().requires_grad_(False)


def save_checkpoint(model,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    import torch
    header={};pieces=[];offset=0;types={torch.bfloat16:'BF16',torch.float16:'F16',torch.float32:'F32'}
    for name,p in model.named_parameters():
        data=p.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name]=dict(dtype=types[p.dtype],shape=list(p.shape),data_offsets=[offset,offset+len(data)])
        pieces.append(data);offset+=len(data)
    encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
    (root/'weights.safetensors').write_bytes(len(encoded).to_bytes(8,'little')+encoded+b''.join(pieces))
    (root/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={name:'weights.safetensors' for name in header})),encoding='utf-8')


class StorageTests(unittest.TestCase):
    base=Path('runs/lazy_bf16_CPU_MOCK_ONLY_tests')
    def setUp(self):
        import torch
        torch.manual_seed(3131);self.dense=toy()
        self.path=self.base/uuid.uuid4().hex;save_checkpoint(self.dense,self.path)
        self.store=CheckpointStore(self.path);self.meta=toy('meta')
    def tearDown(self):
        del self.meta,self.dense,self.store;gc.collect()
        # Known files only, no recursive deletion or manual live-tensor unmap.
        for p in self.path.iterdir():p.unlink()
        self.path.rmdir()

    def test_all_meta_digest_equals_original_dense_digest(self):
        from experiments.nan_recomp09.common import target_digest
        expected=target_digest(SimpleNamespace(model=self.dense),chunk_bytes=17)
        with patch.object(self.store,'tensor',side_effect=AssertionError('Meta hash must stream bytes, not map tensors')):
            self.assertEqual(logical_target_digest(self.meta,self.store,19),expected)

    def test_partial_binding_leaves_other_layers_meta(self):
        import torch
        names=['embed.weight']+[n for n,p in self.meta.named_parameters() if n.startswith('layers.0.')]
        self.store.bind_parameters(self.meta,names)
        for name,p in self.meta.named_parameters():
            self.assertEqual(p.is_meta,name not in names)
            if name in names:
                self.assertTrue(torch.equal(p,dict(self.dense.named_parameters())[name]));self.assertFalse(p.requires_grad)
        self.assertTrue(all(not hasattr(v,'is_meta') for v in vars(self.store).values()))

    def test_mixed_digest_binds_actual_resident_bytes(self):
        from experiments.nan_recomp09.common import target_digest
        import torch
        expected=target_digest(SimpleNamespace(model=self.dense))
        self.store.bind_parameters(self.meta,['norm.weight'])
        self.assertEqual(logical_target_digest(self.meta,self.store),expected)
        # Use a mutable clone; never write a read-only checkpoint mapping.
        self.meta.norm.weight=torch.nn.Parameter(self.meta.norm.weight.detach().clone(),requires_grad=False)
        with torch.no_grad():self.meta.norm.weight.add_(1)
        self.assertNotEqual(logical_target_digest(self.meta,self.store),expected)

    def test_staged_two_loop_outputs_are_bit_exact(self):
        import torch
        for dtype in (torch.bfloat16,torch.float32):
            if dtype!=torch.bfloat16:
                path=self.path/'f32';dense=toy(dtype=dtype);save_checkpoint(dense,path);store=CheckpointStore(path)
            else:dense=self.dense;store=self.store;path=None
            stage=toy(dtype=dtype).layers[0]
            a=torch.arange(2*5*16,dtype=torch.float32).reshape(2,5,16).to(dtype)/100;b=a.clone()
            with torch.inference_mode():
                for phase in (1,2):
                    for layer in range(4):
                        stats=store.copy_parameters(stage,f'layers.{layer}')
                        a=dense.layers[layer](a);b=stage(b)
                        self.assertTrue(torch.equal(a,b));self.assertEqual(stats['parameters'],3)
            if path:
                del store,stage,dense;gc.collect()
                for p in path.iterdir():p.unlink()
                path.rmdir()

    def test_readonly_binding_survives_store_scope(self):
        import torch
        from experiments.nan_ram28.common import sha
        before=sha(self.path/'weights.safetensors')
        self.store.bind_parameters(self.meta,['embed.weight']);value=self.meta.embed.weight
        del self.store;gc.collect();self.store=CheckpointStore(self.path)
        self.assertTrue(torch.equal(value,self.dense.embed.weight));self.assertTrue(value._checkpoint_readonly)
        self.assertEqual(sha(self.path/'weights.safetensors'),before)
        del value

    def test_readonly_stage_meta_and_bad_layout_are_rejected(self):
        self.store.bind_parameters(self.meta,[n for n,p in self.meta.named_parameters() if n.startswith('layers.0.')])
        with self.assertRaises(ValueError):self.store.copy_parameters(self.meta.layers[0],'layers.0')
        with self.assertRaises(ValueError):self.store.copy_parameters(self.meta.layers[1],'layers.1')
        with self.assertRaises(ValueError):self.store.copy_parameters(toy().norm,'layers.0')

    def test_unknown_duplicate_and_nonmeta_binding_rejected(self):
        with self.assertRaises(ValueError):self.store.bind_parameters(self.meta,['not_a_parameter'])
        with self.assertRaises(ValueError):self.store.bind_parameters(self.meta,['embed.weight','embed.weight'])
        with self.assertRaises(ValueError):self.store.bind_parameters(self.dense,['embed.weight'])

    def test_invalid_extent_dtype_and_missing_tensor_rejected(self):
        import copy
        file=self.path/'weights.safetensors';raw=file.read_bytes();size=int.from_bytes(raw[:8],'little');header=json.loads(raw[8:8+size]);payload=raw[8+size:]
        first=next(iter(header))
        for change in ('extent','dtype','missing'):
            bad=copy.deepcopy(header)
            if change=='extent':bad[first]['data_offsets'][1]+=2
            elif change=='dtype':bad[first]['dtype']='UNSUPPORTED'
            else:del bad[first]
            encoded=json.dumps(bad).encode();file.write_bytes(len(encoded).to_bytes(8,'little')+encoded+payload)
            with self.assertRaises(ValueError):CheckpointStore(self.path)
        file.write_bytes(raw)
