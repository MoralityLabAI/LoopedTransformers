"""Prospective memory-bounded BPD execution and exact embedding-lookup reuse.

No model, framework or numerical library is imported until explicitly called.
Chunked pointwise GEMMs are a new execution condition, not a bit-parity claim
for edited trajectories against the former full-prefix BPD implementation.
"""
import ast
from pathlib import Path
from types import MethodType


def last_snapshot_function(source=None):
    source=Path(source or 'mechinterp/nanbeige_bpd_gym/native_components.py')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward_kernel']
    assert len(funcs)==1
    function=funcs[0];before=ast.dump(function)
    class SnapshotOnly(ast.NodeTransformer):
        changed=0
        def visit_Dict(self,node):
            for index,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value in ('input','mask'):
                    value=node.values[index]
                    wanted='x.detach().clone()' if key.value=='input' else 'mask.detach().clone()'
                    assert ast.unparse(value)==wanted
                    expression=('x' if key.value=='input' else 'mask')+'[:, -1:, :].detach().clone()'
                    node.values[index]=ast.parse(expression,mode='eval').body;self.changed+=1
            return self.generic_visit(node)
    change=SnapshotOnly();function=change.visit(function);assert change.changed==2
    # Verify a reversible two-expression observation-only transformation.
    restored=ast.unparse(function).replace('x[:, -1:, :].detach().clone()','x.detach().clone()').replace('mask[:, -1:, :].detach().clone()','mask.detach().clone()')
    assert ast.dump(ast.parse(restored).body[0])==before
    import torch
    namespace={'torch':torch};module=ast.Module(body=[function],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source)+':last_snapshot','exec'),namespace)
    return namespace['forward_kernel']


def bound_components(core,rows):
    """Token-separable inference only; preserve native kernel and pass indexing."""
    assert rows>0
    from mechinterp.bpd.components import PassMaskSchedule
    original=MethodType(last_snapshot_function(),core)
    def forward(self,x,native):
        if self.mode=='target':return original(x,native)
        if self.record_inputs or self.update_running_means or self.intervention is not None:
            raise ValueError('STORY23 pointwise adapter is inference-only')
        assert x.ndim==3 and x.shape[0]==1 and self.schedule is not None
        schedule=self.schedule;calls=self.calls;prior=list(self.controller_observations);parts=[];last=[]
        try:
            for start in range(0,x.shape[1],rows):
                stop=min(x.shape[1],start+rows);self.calls=calls;self.controller_observations=[]
                self.schedule=PassMaskSchedule(schedule.comp_mask[:,:,start:stop],
                    schedule.delta_mask[:,:,start:stop] if schedule.delta_mask is not None else None,
                    schedule.routing_mask[:,:,start:stop] if schedule.routing_mask is not None else None)
                parts.append(original(x[:,start:stop],native[:,start:stop]))
                last=list(self.controller_observations)
        finally:
            self.calls=calls+1;self.schedule=schedule;self.controller_observations=prior+last
        import torch
        return torch.cat(parts,dim=1)
    core.forward_kernel=MethodType(forward,core)
    return original


class EmbeddingLookupCache:
    """Reuse only immutable token embeddings; no recurrent/attention states."""
    def __init__(self,embedding,device,maximum_extra):
        self.embedding=embedding;self.device=device;self.maximum_extra=maximum_extra
        self.original=embedding.forward;self.ids=None;self.buffer=None;self.rows_gathered=0
        owner=self
        def forward(module,ids):return owner.lookup(ids)
        embedding.forward=MethodType(forward,embedding)

    def reset(self):self.ids=None;self.buffer=None;self.rows_gathered=0

    def lookup(self,ids):
        import torch
        assert ids.device.type=='cpu' and ids.dtype==torch.long and ids.ndim==2 and len(ids)==1
        if self.ids is None:
            values=self.original(ids).to(self.device)
            self.buffer=torch.empty((1,ids.shape[1]+self.maximum_extra,values.shape[-1]),device=self.device,dtype=values.dtype)
            self.buffer[:,:ids.shape[1]].copy_(values);self.rows_gathered+=ids.shape[1]
        elif torch.equal(ids,self.ids):pass
        else:
            assert ids.shape[1]==self.ids.shape[1]+1 and torch.equal(ids[:,:-1],self.ids)
            assert ids.shape[1]<=self.buffer.shape[1]
            self.buffer[:,ids.shape[1]-1:ids.shape[1]].copy_(self.original(ids[:,-1:]).to(self.device))
            self.rows_gathered+=1
        self.ids=ids.clone()
        return self.buffer[:,:ids.shape[1]]

    def close(self):
        self.embedding.forward=self.original;self.reset()
