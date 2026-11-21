"""Candidate inference-only bounded gate/up workspace for a NEW condition.

No frozen worker imports this module. Chunking changes gate/up GEMM shapes;
native GPU bit parity or capacity must be measured before any scored use.
The physical down projection is invoked once on the complete intermediate
tensor, retaining its token coordinates and logical BPD pass counter.
"""
from types import MethodType


class BoundedMLPWorkspace:
    def __init__(self,modules,token_rows=256):
        if token_rows<1:raise ValueError('Positive token chunk required')
        self.modules=list(modules);self.token_rows=token_rows;self.originals=[];self.enabled=True
        if len({id(m) for m in self.modules})!=len(self.modules):raise ValueError('Duplicate physical MLP')
        for module in self.modules:
            if module.config.pretraining_tp!=1:raise ValueError('Only the pinned pretraining_tp=1 branch is supported')
            if not all(hasattr(module,name) for name in ('gate_proj','up_proj','down_proj','act_fn')):raise ValueError('Unsupported gated MLP')
        owner=self
        for module in self.modules:
            original=module.forward;self.originals.append((module,original))
            def forward(mlp,x,*,original=original):
                import torch
                if not owner.enabled:return original(x)
                if torch.is_grad_enabled():raise RuntimeError('Workspace adapter is inference-only')
                if x.ndim!=3 or x.shape[1]<1:raise ValueError('Require [batch,positive_positions,width]')
                middle=None
                for start in range(0,x.shape[1],owner.token_rows):
                    stop=min(x.shape[1],start+owner.token_rows)
                    gate=mlp.act_fn(mlp.gate_proj(x[:,start:stop]))
                    if middle is None:
                        middle=torch.empty((x.shape[0],x.shape[1],gate.shape[-1]),device=gate.device,dtype=gate.dtype)
                    up=mlp.up_proj(x[:,start:stop]);middle[:,start:stop].copy_(gate*up)
                    del gate,up
                return mlp.down_proj(middle)
            module.forward=MethodType(forward,module)

    def close(self):
        for module,original in self.originals:module.forward=original
        self.originals=[]


def install(loop,token_rows=256):
    """Patch resident and reusable staged MLPs, without changing any weight."""
    modules=[layer.mlp for layer in loop.layers]
    if loop.stage_layer is not None:modules.append(loop.stage_layer.mlp)
    return BoundedMLPWorkspace(modules,token_rows)
