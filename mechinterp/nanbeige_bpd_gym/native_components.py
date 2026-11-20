"""Engineering bridge to pinned BPD factors with the native kernel retained.

No checkpoint loading, fitting or causal benchmark runs occur on import. The
native target stays authoritative; this adds component/residual masks around
its actual output instead of replacing an NF4 kernel with a dense surrogate.
"""
from __future__ import annotations
import importlib
import hashlib
from pathlib import Path
import torch
from torch import nn
from experiments.nan_gym05.common import read,sha


def load_bpd(root):
    root=Path(root).resolve();lock=read(Path(__file__).with_name('BPD_FRAMEWORK_LOCK.json'))
    for name,h in lock['git_LF_files_sha256'].items():
        # The source checkout uses Git's CRLF conversion on Windows. Bind to
        # verified commit bytes with ONLY that newline conversion normalized.
        data=(root/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Pinned BPD framework changed: '+name)
    import mechinterp
    parent=str(root/'mechinterp')
    if parent not in mechinterp.__path__:mechinterp.__path__.append(parent)
    modules={name:importlib.import_module('mechinterp.bpd.'+name) for name in ('components','config','losses','trainer')}
    for module in modules.values():
        if not Path(module.__file__).resolve().is_relative_to(root):raise ValueError('BPD import resolved outside the pinned source checkout')
    return modules


def component_class(bpd_root):
    """Subclass the actual pinned framework; keep its factors/loss interfaces."""
    base=load_bpd(bpd_root)['components'].TiedComponentLinear
    class NativeKernelComponents(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.live_controller=None
            self.controller_observations=[]

        def reset_pass_state(self,*,pass_offset=0):
            super().reset_pass_state(pass_offset=pass_offset)
            self.controller_observations=[]

        def forward(self,x):
            # CPU/reference use only. Native model hooks call forward_kernel
            # with the already computed output of the real physical module.
            output=torch.nn.functional.linear(x,self.W_target.to(x.dtype),self.bias.to(x.dtype) if self.bias is not None else None)
            return self.forward_kernel(x,output)

        def forward_kernel(self,x,native_output):
            if x.shape[-1]!=self.d_in or native_output.shape!=(*x.shape[:-1],self.d_out):raise ValueError('Wrong physical module coordinates')
            local=self.calls;tau=self.pass_offset+local;self.calls+=1
            if self.record_inputs:self.recorded_inputs.append(x.detach() if self.detach_recorded_inputs else x)
            if self.mode=='target' and not self.update_running_means:
                return self._apply_intervention(x,native_output,tau) if self.intervention is not None else native_output
            xf=x.float();acts=xf@self.V
            bias=self.bias.float() if self.bias is not None else 0.
            clean=native_output.float()-bias
            if self.update_running_means:
                self._ensure_pass_capacity(tau)
                with torch.no_grad():
                    values=(acts.detach().reshape(-1,self.C).mean(0),
                        (clean.detach()-acts.detach()@self.U.detach()).reshape(-1,self.d_out).mean(0),
                        clean.detach().square().mean(-1).add(self.ablation.eps).sqrt().mean())
                    for target,value in zip((self.mean_comp_acts[tau],self.mean_delta_out[tau],self.mean_write_rms[tau]),values):
                        if bool(self.calibrated_passes[tau]):target.lerp_(value,self.ablation.ema_momentum)
                        else:target.copy_(value)
                    self.calibrated_passes[tau]=True
            if self.mode=='target':
                return self._apply_intervention(x,native_output,tau) if self.intervention is not None else native_output
            if self.schedule is None:raise RuntimeError('Component mode needs a registered schedule')
            mask=self.schedule.comp_mask[:,local]
            if self.live_controller is not None:
                # The callback sees the input actually reached NOW, after any
                # preceding edits. It cannot substitute a cached clean state.
                mask=self.live_controller(x,tau)
                if mask.shape!=acts.shape:raise ValueError('Reached-state controller has wrong component coordinates')
                self.controller_observations.append({'pass':tau,'input':x.detach().clone(),'mask':mask.detach().clone()})
            calibrated=self.ablation.mode=='pass_mean' and tau<len(self.calibrated_passes) and bool(self.calibrated_passes[tau])
            mean=self.mean_comp_acts[tau] if calibrated else 0.
            masked=mask*acts+(1-mask)*mean
            dmask=self.schedule.delta_mask[:,local].unsqueeze(-1) if self.schedule.delta_mask is not None else 1.
            mean_delta=self.mean_delta_out[tau] if calibrated else 0.
            # Equivalent component/residual algebra, arranged so all-one masks
            # yield the native output even when low-rank factors have poor fit.
            adjustment=(dmask-1)*clean+(masked-dmask*acts)@self.U+(1-dmask)*mean_delta
            write=clean+adjustment
            if self.ablation.sphere_rescale and calibrated:
                eps=self.ablation.eps;frac=(1-mask.clamp(0,1)).mean(-1,keepdim=True)
                rms=write.square().mean(-1,keepdim=True).add(eps).sqrt()
                desired=(1-frac)*rms+frac*self.mean_write_rms[tau].clamp_min(eps)
                write=write*torch.where(write.square().mean(-1,keepdim=True)>eps,desired/rms,torch.ones_like(rms))
                adjustment=write-clean
            output=(native_output.float()+adjustment).to(native_output.dtype)
            if self.schedule.routing_mask is not None:
                output=torch.where(self.schedule.routing_mask[:,local].unsqueeze(-1),output,native_output)
            return output
    return NativeKernelComponents


def attach_native_components(loop,c_per_module,*,bpd_root,ablation,max_passes=2):
    """Create the actual BPDModel interface using physical-module hooks.

    Hooks leave staged BF16 parameter layouts intact and preserve NF4 kernels.
    Register observation hooks AFTER these hooks to observe the edited outputs.
    Keys are full model paths, e.g. model.layers.21.mlp.down_proj.
    """
    framework=load_bpd(bpd_root);Kernel=component_class(bpd_root)
    class NativeBPDModel(framework['components'].BPDModel):
        def __init__(self):
            nn.Module.__init__(self)
            if not c_per_module:raise ValueError('At least one physical module required')
            for p in loop.model.parameters():p.requires_grad_(False)
            # NativeLoop's reusable stage is a clone outside model.parameters().
            # Its original inference path did not need an autograd freeze.
            if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()
            loop.model.eval();self.target_model=loop.model
            self.module_order=sorted(c_per_module);self.c_per_module=dict(c_per_module)
            self.wrappers={};self.native_handles=[]
            for name in self.module_order:
                parts=name.split('.')
                if len(parts)!=5 or parts[:2]!=['model','layers'] or (parts[3],parts[4]) not in (('mlp','down_proj'),('self_attn','o_proj')):
                    raise ValueError('Only physical o_proj/down_proj sites are supported')
                layer=int(parts[2]);native=loop.model.get_submodule(name)
                if getattr(native.weight,'quant_state',None) is not None:
                    from bitsandbytes.functional import dequantize_4bit
                    effective=dequantize_4bit(native.weight.data,quant_state=native.weight.quant_state).detach()
                else:effective=native.weight.detach()
                fake=nn.Linear(effective.shape[1],effective.shape[0],bias=False,device='meta',dtype=effective.dtype)
                fake.weight=nn.Parameter(effective,requires_grad=False)
                if native.bias is not None:fake.bias=nn.Parameter(native.bias.detach(),requires_grad=False)
                core=Kernel(fake,int(c_per_module[name]),max_passes=max_passes,ablation=ablation,init_svd=False).to(loop.device)
                self.wrappers[name]=core
                def hook(module,args,output,*,site_layer=layer,component=core):
                    if loop.active_layer==site_layer:return component.forward_kernel(args[0],output)
                    return None
                self.native_handles.append(native.register_forward_hook(hook))
                if loop.stage_layer is not None and layer>=loop.n_gpu_layers:
                    staged=loop.stage_layer.get_submodule('.'.join(parts[3:]))
                    self.native_handles.append(staged.register_forward_hook(hook))
            self._registered_wrappers=nn.ModuleDict({k.replace('.','__'):v for k,v in self.wrappers.items()})

        def to(self,device):
            # BPDTrainer calls model.to(device). Moving the entire target would
            # pull its CPU embeddings/head onto the 4GB GPU, so move only the
            # added factors/effective-operator buffers; target placement is fixed.
            device=torch.device(device)
            if device!=loop.device:raise ValueError('Preserve the preplaced native decoder device')
            for wrapper in self.wrappers.values():wrapper.to(device)
            if loop.model.lm_head.weight.device.type!='cpu':raise ValueError('Native full output head must stay on CPU')
            return self

        def close(self):
            for handle in self.native_handles:handle.remove()
            self.native_handles=[]
    return NativeBPDModel()


class OffloadedHead(torch.autograd.Function):
    """Full-vocabulary native-dtype forward with a frozen CPU head.

    Backward streams weights again and accumulates the input derivative in FP32;
    it keeps no staged GPU head blocks alive between forward and backward.
    """
    @staticmethod
    def forward(ctx,hidden,weight,chunk):
        if weight.device.type!='cpu' or weight.requires_grad or hidden.shape[-1]!=weight.shape[1]:raise ValueError('Frozen compatible CPU output head required')
        if int(chunk)<1:raise ValueError('Positive vocabulary chunk required')
        ctx.weight=weight;ctx.chunk=int(chunk);ctx.device=hidden.device;ctx.dtype=hidden.dtype;ctx.shape=hidden.shape
        result=[]
        for start in range(0,len(weight),ctx.chunk):
            block=weight[start:start+ctx.chunk].to(device=hidden.device,dtype=hidden.dtype)
            result.append(torch.nn.functional.linear(hidden,block).float())
        return torch.cat(result,dim=-1)

    @staticmethod
    def backward(ctx,gradient):
        total=torch.zeros(ctx.shape,device=ctx.device,dtype=torch.float32)
        for start in range(0,len(ctx.weight),ctx.chunk):
            block=ctx.weight[start:start+ctx.chunk].to(device=ctx.device,dtype=torch.float32)
            total.add_(gradient[...,start:start+len(block)].float()@block)
        return total.to(ctx.dtype),None,None


def differentiable_pairs(loop,items,*,offload_saved_tensors=True):
    """Use the pinned native loop body with gradients and CPU saved tensors.

    B1 only: physical wrapper invocation counts index recurrent passes. A GPU
    fixture must verify streamed-weight backward snapshots before native fits.
    """
    if len(items)!=1:raise ValueError('One independent source context per differentiable rollout')
    if torch.is_inference_mode_enabled():raise RuntimeError('Native BPD gradients cannot run inside inference_mode')
    if loop.stage_layer is not None and any(p.requires_grad for p in loop.stage_layer.parameters()):raise RuntimeError('Freeze the reusable target stage before native BPD gradients')
    from experiments.nan_bpd01.backend import NativeLoop
    function=NativeLoop.run_pairs.__wrapped__
    if offload_saved_tensors:
        with torch.autograd.graph.save_on_cpu(pin_memory=False):return function(loop,items)
    if loop.stage_layer is not None:raise ValueError('Staged weights require saved-tensor CPU snapshots')
    return function(loop,items)


def native_rollout_adapter(loop,*,chunk_rows=8192,offload_saved_tensors=True):
    """BPDTrainer adapter: source token IDs only, full-vocabulary per-pass heads."""
    def rollout(token_ids,n_passes,pass_offset,init_state):
        if torch.is_inference_mode_enabled():raise RuntimeError('Differentiable native adapter requires ordinary/no_grad contexts')
        if init_state is None:
            ids=torch.as_tensor(token_ids,dtype=torch.long)
            if ids.ndim==1:ids=ids.unsqueeze(0)
            if ids.ndim!=2 or len(ids)!=1:raise ValueError('One source context required')
            h,context=loop.prepare(ids)
        else:h,context=init_state
        readouts={};boundaries={}
        for tau in range(pass_offset,pass_offset+n_passes):
            loop.phase=tau
            h=differentiable_pairs(loop,[(h,context)],offload_saved_tensors=offload_saved_tensors)[0]
            readouts[tau]=OffloadedHead.apply(h[:,-1:,:],loop.model.lm_head.weight,chunk_rows)
            boundaries[tau]=h[:,-1:,:]
        return readouts,boundaries,None,None
    return rollout
