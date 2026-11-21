"""New-condition native gradient execution with recomputation and resident head.

Frozen studies import none of this module. Selected physical component layers
are never checkpointed: their mutable per-pass hook counters run exactly once.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .native_components import attach_native_components,load_bpd
from .native_saved_tensors import NativeSavedTensors,SavedTensor,storage_key


class ResidentSavedTensors(NativeSavedTensors):
    """Keep version-guarded references to already resident immutable weights."""
    def __init__(self,loop,maximum_added_cpu_bytes=2*1024**3,constants=()):
        super().__init__(loop,maximum_added_cpu_bytes);self.resident_refs=0;self.checkpointed_blocks=0;self.recomputed_blocks=0
        self.failed_reservation_bytes=None;self.largest_activation_bytes=0
        for value in constants:
            if value is not None:
                if value.requires_grad or value.device.type!='cuda':raise ValueError('Only frozen CUDA constants may be referenced')
                self.resident[storage_key(value)]=value

    def charge(self,count):
        if self.live_bytes+count>self.maximum:self.failed_reservation_bytes=count
        return super().charge(count)

    def pack(self,tensor):
        key=storage_key(tensor)
        if tensor.device.type=='cuda' and key in self.resident:
            original=self.resident[key]
            if original.requires_grad:raise RuntimeError('Trainable parameter cannot use an immutable reference')
            if tensor.dtype!=original.dtype or tensor.device!=original.device:raise ValueError('Unsupported immutable weight dtype/device view')
            self.resident_refs+=1
            return SavedTensor(self,self.view(tensor,original,original),tensor.device,parameter=original)
        if tensor.device.type!='cpu' and key not in self.stage:self.largest_activation_bytes=max(self.largest_activation_bytes,tensor.numel()*tensor.element_size())
        return super().pack(tensor)

    def statistics(self):
        return super().statistics()|dict(resident_GPU_weight_references=self.resident_refs,
            checkpointed_frozen_blocks=self.checkpointed_blocks,recomputed_frozen_blocks=self.recomputed_blocks,
            failed_reservation_bytes=self.failed_reservation_bytes,largest_activation_snapshot_bytes=self.largest_activation_bytes,
            scope='immutable GPU weights referenced in place; staged CPU source weights aliased; independent activation snapshots charged to unchanged CPU cap')


class ResidentFrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,hidden,weight,chunk):
        if weight.device!=hidden.device or weight.requires_grad or hidden.shape[-1]!=weight.shape[1]:raise ValueError('Frozen compatible resident output head required')
        if int(chunk)<1:raise ValueError('Positive vocabulary chunk required')
        ctx.weight=weight;ctx.version=weight._version;ctx.chunk=int(chunk);ctx.dtype=hidden.dtype;ctx.shape=hidden.shape
        return torch.cat([torch.nn.functional.linear(hidden,weight[start:start+ctx.chunk].to(hidden.dtype)).float()
            for start in range(0,len(weight),ctx.chunk)],dim=-1)

    @staticmethod
    def backward(ctx,gradient):
        if ctx.weight._version!=ctx.version:raise RuntimeError('Frozen resident vocabulary weight changed before backward')
        total=torch.zeros(ctx.shape,device=gradient.device,dtype=torch.float32)
        for start in range(0,len(ctx.weight),ctx.chunk):
            block=ctx.weight[start:start+ctx.chunk].float()
            total.add_(gradient[...,start:start+len(block)].float()@block)
        return total.to(ctx.dtype),None,None


def attach_resident_components(loop,c_per_module,*,bpd_root,ablation,max_passes=2):
    """Actual pinned BPDModel contexts/hooks, with explicit resident placement."""
    original=attach_native_components(loop,c_per_module,bpd_root=bpd_root,ablation=ablation,max_passes=max_passes)
    Base=load_bpd(bpd_root)['components'].BPDModel
    class ResidentModel(Base):
        def __init__(self):
            nn.Module.__init__(self);object.__setattr__(self,'_native_instrument',original)
            self.target_model=original.target_model;self.module_order=original.module_order;self.c_per_module=original.c_per_module
            self.wrappers=original.wrappers
            self._registered_wrappers=nn.ModuleDict({k.replace('.','__'):v for k,v in self.wrappers.items()})
        def to(self,device):
            if torch.device(device)!=loop.device or loop.model.lm_head.weight.device!=loop.device:raise ValueError('Preserve the registered resident head and target placement')
            for wrapper in self.wrappers.values():wrapper.to(device)
            return self
        def close(self):self._native_instrument.close()
    return ResidentModel()


def native_pairs(loop,items,owner,exempt_layers,*,recompute=True):
    if len(items)!=1 or items[0][0].shape[0]!=1:raise ValueError('One source context per native gradient rollout')
    if torch.is_inference_mode_enabled():raise RuntimeError('Use ordinary/no_grad mode for native gradient execution')
    if any(p.requires_grad for p in loop.model.parameters()):raise RuntimeError('Freeze target parameters')
    if loop.stage_layer is not None and any(p.requires_grad for p in loop.stage_layer.parameters()):raise RuntimeError('Freeze the reusable target stage')
    states,context=items[0];phase=loop.phase
    with owner.context() if owner is not None else nullcontext():
        for index in range(loop.n):
            calls=[0]
            def execute(value,*,index=index,phase=phase,context=context,calls=calls):
                old_layer,old_phase=loop.active_layer,loop.phase
                loop.active_layer,loop.phase=index,phase
                try:
                    if calls[0] and owner is not None:owner.recomputed_blocks+=1
                    calls[0]+=1
                    layer=loop.layers[index] if index<loop.n_gpu_layers else loop._load_stage(loop.layers[index])
                    output=layer(value,attention_mask=context['mask'],position_ids=context['pos'],past_key_value=None,
                        output_attentions=False,use_cache=False,cache_position=context['cache_position'],loop_idx=0)[0]
                    return torch.cat([output])
                finally:loop.active_layer,loop.phase=old_layer,old_phase
            use_checkpoint=recompute and torch.is_grad_enabled() and states.requires_grad and index not in exempt_layers
            if use_checkpoint:
                if owner is not None:owner.checkpointed_blocks+=1
                states=checkpoint(execute,states,use_reentrant=False,preserve_rng_state=True)
            else:states=execute(states)
        normalized=torch.cat([loop.base.norm(states)])
    loop.active_layer=loop.n-1
    return [normalized]


def rollout_adapter(loop,owner,c_per_module,*,chunk_rows=8192,recompute=True):
    exempt={int(name.split('.')[2]) for name in c_per_module}
    def rollout(token_ids,n_passes,pass_offset,init_state):
        if init_state is None:
            ids=torch.as_tensor(token_ids,dtype=torch.long)
            if ids.ndim==1:ids=ids.unsqueeze(0)
            h,context=loop.prepare(ids)
        else:h,context=init_state
        readouts={};boundaries={}
        for tau in range(pass_offset,pass_offset+n_passes):
            loop.phase=tau;h=native_pairs(loop,[(h,context)],owner,exempt,recompute=recompute)[0]
            readouts[tau]=ResidentFrozenHead.apply(h[:,-1:,:],loop.model.lm_head.weight,chunk_rows);boundaries[tau]=h[:,-1:,:]
        return readouts,boundaries,None,None
    return rollout
