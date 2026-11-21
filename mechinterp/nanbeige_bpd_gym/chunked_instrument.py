"""Unregistered phase-bound adapter for actual BPD hooks on query chunks.

Uses the unchanged qualified pointwise algebra. Every query chunk is bound to
native phase 1 or2, not counted as another recurrent pass. Captures only the
last valid token; no raw legacy-observer retention or controller-history bank.
No model or numerical package is imported until construction/call.
"""
from contextlib import contextmanager,nullcontext
from types import MethodType


class ChunkedRowInstrument:
    def __init__(self,loop,backend,bpd_root,cfg):
        from experiments.nan_story23.instrument import Instrument
        self.base=Instrument(loop,backend,bpd_root,cfg);self.core=self.base.core;self.original=self.core.forward_kernel
        self.loop=loop;self.frame=None;self.modes=();self.native={};self.inputs={};self.masks={};self.applied_tokens={}
        # The original snapshot consumer is not used by this new adapter.
        # Keep the real component edit hook; remove only base.raw observers.
        for handle in self.base.handles:handle.remove()
        self.base.handles=[];self.base.raw.clear();owner=self
        def forward(core,x,native):
            import torch
            f=owner.frame
            if f is None or f.phase not in (1,2):raise RuntimeError('Physical native phase must be explicit')
            for row,offset in enumerate(f.capture_offsets):
                if offset is not None:
                    key=f.phase,row
                    if key in owner.native:raise RuntimeError('Duplicate native last-token observation')
                    owner.native[key]=native[row,offset].detach().cpu().clone()
                    owner.inputs[key]=x[row,offset].detach().cpu().clone()
            if core.mode=='target':return native
            selected=[i for i,mode in enumerate(owner.modes) if mode!='native']
            if len(selected)!=1:raise RuntimeError('One genuine edited row required')
            row=selected[0];n=f.valid_lengths[row]
            if n==0:return native
            mode=owner.modes[row];masks=torch.ones((1,2,n,8),device=x.device)
            if mode=='static_CAL_mean':masks[:]=owner.base.static[None,:,None,:]
            delta=torch.ones((1,2,n),device=x.device) if mode=='all_one' else torch.zeros((1,2,n),device=x.device)
            core.schedule=owner.base.Schedule(masks,delta)
            if core.pass_offset!=0:raise RuntimeError('The fixed two-native-phase binding requires zero pass offset')
            core.calls=f.phase-1;core.controller_observations=[]
            edited=owner.original(x[row:row+1,:n],native[row:row+1,:n])
            if core.calls!=f.phase:raise RuntimeError('A query chunk changed the calibrated recurrent phase')
            owner.applied_tokens[f.phase]=owner.applied_tokens.get(f.phase,0)+n
            offset=f.capture_offsets[row]
            if offset is not None:
                if offset!=n-1:raise RuntimeError('Mask must correspond to the last actual token')
                mask=core.controller_observations[-1]['mask'][0,-1] if core.live_controller else masks[0,f.phase-1,-1]
                owner.masks[f.phase,row]=mask.detach().float().cpu().clone()
            core.controller_observations=[]
            result=native.clone();result[row:row+1,:n]=edited
            return result
        self.core.forward_kernel=MethodType(forward,self.core)
    def set_frame(self,frame):self.frame=frame
    @contextmanager
    def context(self,modes):
        self.modes=tuple(modes);selected=[m for m in modes if m!='native']
        if len(selected)>1:raise ValueError('One edited row per cache batch')
        self.native={};self.inputs={};self.masks={};self.applied_tokens={}
        context=self.base.context(selected[0],1) if selected else nullcontext()
        try:
            with context:yield
        finally:
            self.core.live_controller=None;self.core.controller_observations=[];self.base.raw.clear()
    def close(self):self.base.close()
