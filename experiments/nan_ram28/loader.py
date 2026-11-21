"""File-backed checkpoint parameters; identical pinned NF4 conversion code.

Each tensor owns a read-only file mapping through torch.frombuffer's
documented owner reference. Replaced decoder storage is released normally;
there is no full BF16 host allocation or manual unmapping of live tensors.
Inference only: no code may write a mapped CPU parameter. Copy-on-write
mappings are deliberately avoided because Windows reserves their full commit.
"""
import json
import math
import mmap
from pathlib import Path

EXPECTED_TARGET='8e80e7d748d5daf858a0b37dd7843d38952554ce3aabea1104e52cf1ef7663b3'

def mapped_tensor(path,offset,length,dtype,shape):
    import torch
    assert math.prod(shape)*torch.empty((),dtype=dtype).element_size()==length and length>0
    aligned=offset-offset%mmap.ALLOCATIONGRANULARITY;delta=offset-aligned
    with Path(path).open('rb') as stream:
        owner=mmap.mmap(stream.fileno(),delta+length,access=mmap.ACCESS_READ,offset=aligned)
    return torch.frombuffer(owner,dtype=dtype,count=math.prod(shape),offset=delta).reshape(shape)

def materialize(model,snapshot):
    import torch
    snapshot=Path(snapshot);index=json.loads((snapshot/'model.safetensors.index.json').read_bytes())['weight_map']
    expected=model.state_dict();assert set(expected)==set(index),'Unsupported missing/extra checkpoint storage'
    dtypes={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}
    loaded=set();mapped_bytes=0
    for shard in sorted(set(index.values())):
        path=snapshot/shard
        with path.open('rb') as stream:
            header_size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(header_size))
        for name,spec in header.items():
            if name=='__metadata__':continue
            assert name in expected and index[name]==shard and name not in loaded
            reference=expected[name];assert reference.is_meta and list(reference.shape)==spec['shape']
            start,end=spec['data_offsets']
            value=mapped_tensor(path,8+header_size+start,end-start,dtypes[spec['dtype']],spec['shape'])
            if value.dtype!=reference.dtype:value=value.to(dtype=reference.dtype)
            parent_name,leaf=name.rsplit('.',1) if '.' in name else ('',name)
            parent=model.get_submodule(parent_name)
            if leaf in parent._parameters:parent.register_parameter(leaf,torch.nn.Parameter(value,requires_grad=False))
            else:
                assert leaf in parent._buffers;parent._buffers[leaf]=value
            loaded.add(name);mapped_bytes+=end-start
    assert loaded==set(expected)
    return dict(mapped_parameter_count=len(loaded),checkpoint_storage_bytes=mapped_bytes)

def empty_model(snapshot):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM
    config=AutoConfig.from_pretrained(str(snapshot),trust_remote_code=True,local_files_only=True)
    config._attn_implementation='sdpa'
    assert not config.tie_word_embeddings
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device('meta'):model=AutoModelForCausalLM.from_config(config,trust_remote_code=True,attn_implementation='sdpa')
    finally:torch.set_default_dtype(previous)
    return model

def load_backend(compact_root,snapshot,backend,*,device='cuda:0',chunk_rows=8192):
    assert backend=='nf4','RAM28 defers BF16 rather than silently changing its condition'
    import os,sys,torch
    from experiments.nan_bpd01.backend import NativeLoop
    from experiments.nan_bpd01.head import ChunkedHead
    from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
    from experiments.nan_recomp09.common import target_digest
    from .common import ROOT,RUNS,read,sha,write
    from .resources import memory
    NB.assert_oracle_isolated();sys.path.insert(0,str(Path(compact_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02 import runner,nanbeige
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(snapshot).resolve());entry=load_lock()['nanbeige']
    assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
    snapshot=Path(snapshot)
    for name,digest in entry['files_sha256'].items():assert sha(snapshot/name)==digest,name
    model=empty_model(snapshot);nanbeige.check_supported(model.config)
    info=materialize(model,snapshot)
    for layer in model.model.layers:NB._reinit_rotary_buffers(layer,torch.device('cpu'))
    assert all(not p.is_meta for p in model.parameters()) and all(not p.is_meta for p in model.buffers())
    model.eval();model.requires_grad_(False)
    assert model.model.embed_tokens.max_norm is None
    loop=NativeLoop(model,device,quantized=True)
    digest=target_digest(loop);assert digest==EXPECTED_TARGET,'Packed/native parameter bytes changed; stop before first forward'
    tok=nanbeige.load_tokenizer(snapshot,entry['bos_token_id'])
    head=ChunkedHead(model.lm_head.weight,runner.op_token_ids(tok),device=device,chunk_rows=chunk_rows)
    write(RUNS/'loader_attestation.json',dict(**info,backend='nf4',parameter_sha256=digest,
        reference_parameter_sha256=EXPECTED_TARGET,reference_condition='Original CTX21 NF4 before-forward manifest',
        parameter_bytes_match=True,quantization_state_identity_independently_verified=False,
        no_pretrained_forward_yet=True,copy_on_write_mappings=False,read_only_mappings=True,full_BF16_host_heap_allocated=False,
        memory=memory(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),sealed_test_accessed=False))
    NB.assert_oracle_isolated()
    return loop,head,tok,entry
