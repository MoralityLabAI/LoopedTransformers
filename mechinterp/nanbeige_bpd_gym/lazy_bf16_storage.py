"""Candidate BF16 storage primitives, not a registered model execution path.

Decoder parameters may remain meta. Map one source tensor for each synchronous
stage copy; do not retain a full CPU decoder or cache mapped tensors. Selected
CPU parameters can be bound read-only. Importing this module creates no CUDA
context and starts no inference. A new loader/protocol and real CUDA admission
are still required before use in a model condition.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


DTYPE_BYTES={'BF16':2,'F16':2,'F32':4}


@dataclass(frozen=True)
class Span:
    path:Path
    offset:int
    length:int
    dtype:str
    shape:tuple


class CheckpointStore:
    def __init__(self,snapshot):
        root=Path(snapshot);index=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
        self.spans={}
        for shard in sorted(set(index.values())):
            path=root/shard
            if path.resolve().parent!=root.resolve():raise ValueError('Shard outside snapshot')
            with path.open('rb') as stream:
                size=int.from_bytes(stream.read(8),'little')
                if not 0<size<=64*1024**2:raise ValueError('Invalid safetensors header size')
                header=json.loads(stream.read(size))
            file_bytes=path.stat().st_size
            for name,spec in header.items():
                if name=='__metadata__':continue
                if name not in index or index[name]!=shard or name in self.spans:raise ValueError('Unexpected/duplicate tensor')
                shape=tuple(spec['shape']);start,end=spec['data_offsets'];dtype=spec['dtype']
                if dtype not in DTYPE_BYTES or any(not isinstance(n,int) or n<1 for n in shape):raise ValueError('Unsupported tensor layout')
                if type(start) is not int or type(end) is not int:raise ValueError('Non-integer tensor extent')
                length=end-start;offset=8+size+start
                if start<0 or length!=math.prod(shape)*DTYPE_BYTES[dtype] or length<=0 or offset+length>file_bytes:
                    raise ValueError('Invalid tensor extent')
                self.spans[name]=Span(path,offset,length,dtype,shape)
        if set(self.spans)!=set(index):raise ValueError('Incomplete tensor index')

    def validate(self,model):
        import torch
        params=dict(model.named_parameters())
        if set(params)!=set(self.spans):raise ValueError('Parameter names differ from checkpoint')
        types={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}
        for name,p in params.items():
            spec=self.spans[name]
            if tuple(p.shape)!=spec.shape or p.dtype!=types[spec.dtype] or not p.is_contiguous():
                raise ValueError('Parameter layout mismatch: '+name)

    def tensor(self,name):
        import torch
        from experiments.nan_ram28.loader import mapped_tensor
        spec=self.spans[name];dtype={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}[spec.dtype]
        return mapped_tensor(spec.path,spec.offset,spec.length,dtype,spec.shape)

    def chunks(self,name,chunk_bytes=1024**2):
        if chunk_bytes<1:raise ValueError('Positive byte chunk required')
        spec=self.spans[name];remaining=spec.length
        with spec.path.open('rb') as stream:
            stream.seek(spec.offset)
            while remaining:
                data=stream.read(min(chunk_bytes,remaining))
                if not data:raise EOFError('Checkpoint changed during byte stream')
                remaining-=len(data);yield data

    def bind_parameters(self,model,names,device='cpu'):
        """Selected parameters only; CPU bindings remain read-only mappings."""
        import torch
        names=list(names)
        if len(set(names))!=len(names):raise ValueError('Duplicate binding')
        self.validate(model);params=dict(model.named_parameters());device=torch.device(device)
        for name in names:
            if name not in params or not params[name].is_meta:raise ValueError('Bind only known meta parameters')
        for name in names:
            value=self.tensor(name).to(device)
            parent,leaf=name.rsplit('.',1) if '.' in name else ('',name)
            p=torch.nn.Parameter(value,requires_grad=False)
            if device.type=='cpu':p._checkpoint_readonly=True
            model.get_submodule(parent).register_parameter(leaf,p)

    def copy_parameters(self,stage,prefix):
        """Same parameter copy arithmetic as NativeLoop; no source mutation."""
        import torch
        prefix=prefix.rstrip('.')+'.';params=dict(stage.named_parameters())
        expected={name[len(prefix):] for name in self.spans if name.startswith(prefix)}
        if set(params)!=expected:raise ValueError('Stage parameter names differ')
        types={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}
        for name,p in params.items():
            spec=self.spans[prefix+name]
            if p.is_meta or getattr(p,'_checkpoint_readonly',False):raise ValueError('Require writable allocated stage')
            if tuple(p.shape)!=spec.shape or p.dtype!=types[spec.dtype]:raise ValueError('Stage layout mismatch')
        total=largest=0
        with torch.no_grad():
            for name,p in params.items():
                spec=self.spans[prefix+name];source=self.tensor(prefix+name)
                if p.device.type=='cpu' and p.data_ptr()==source.data_ptr():raise ValueError('Stage aliases read-only source')
                p.copy_(source,non_blocking=False)
                total+=spec.length;largest=max(largest,spec.length)
                del source  # frombuffer owns the mapping until the tensor dies.
        return dict(parameters=len(params),copied_bytes=total,largest_source_tensor_bytes=largest)


def logical_target_digest(model,store,chunk_bytes=1024**2):
    """Original ordered name+parameter-byte digest, including meta file spans.

    This binds bytes, not executed kernels or CUDA stage correctness. A later
    model condition must also verify source files, actual stage copies and
    resident parameters. No quantized/meta substitution is supported here.
    """
    import torch
    store.validate(model);digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode())
        if p.is_meta:
            for block in store.chunks(name,chunk_bytes):digest.update(block)
        else:
            data=p.detach().reshape(-1).view(torch.uint8)
            for start in range(0,len(data),chunk_bytes):
                values=data[start:start+chunk_bytes].cpu().numpy();digest.update(memoryview(values))
    return digest.hexdigest()
