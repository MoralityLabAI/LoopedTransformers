"""Experimental cache-separated two-loop decoding, never a frozen-worker patch.

Full-prefix prefill and incremental decoding are distinct numerical conditions:
single-token GEMM/GEMV/SDPA kernels may differ from full-prefix arithmetic.
No pretrained use is authorized by importing this engineering implementation.
"""
import torch
from experiments.nan_gym05.observe import TowerObserver


class NativeTwoLoopCache:
    def __init__(self,loop,*,observe_layers=(5,11,17,21),cache_factory=None):
        if any(p.requires_grad for p in loop.model.parameters()):raise ValueError('Frozen target parameters required')
        if cache_factory is None:
            from transformers.cache_utils import DynamicCache
            cache_factory=DynamicCache
        self.loop=loop;self.factory=cache_factory;self.observer=TowerObserver(loop,observe_layers)
        self.caches=[];self.length=0;self.layers=tuple(observe_layers)

    def reset(self):
        self.caches=[];self.length=0

    def cache_bytes(self):
        return sum(x.numel()*x.element_size() for c in self.caches for x in (*c.key_cache,*c.value_cache))

    def _walk(self,h,ctx,cache):
        for index in range(self.loop.n):
            self.loop.active_layer=index
            layer=self.loop.layers[index] if index<self.loop.n_gpu_layers else self.loop._load_stage(self.loop.layers[index])
            if layer.self_attn.layer_idx!=index:raise RuntimeError('Physical staging cache index changed')
            h=layer(h,attention_mask=ctx['mask'],position_ids=ctx['pos'],past_key_value=cache,
                output_attentions=False,use_cache=True,cache_position=ctx['cache_position'],loop_idx=0)[0]
            h=torch.cat([h])
        return torch.cat([self.loop.base.norm(h)])

    def _forward(self,h,positions,*,capture):
        o=self.observer;o.enabled=capture;o.names=['context'];o.values={};o.counts={};boundaries=[]
        total=int(positions[-1])+1
        try:
            for phase in range(2):
                self.loop.phase=phase+1;o.phase=phase+1;cache=self.caches[phase]
                mask=self.loop.base._update_causal_mask(None,h,positions,cache,False)
                ctx={'mask':mask,'pos':positions.unsqueeze(0),'cache_position':positions}
                h=self._walk(h,ctx,cache);boundaries.append(h[:,-1:,:].detach().clone())
                if any(cache.get_seq_length(i)!=total for i in range(self.loop.n)):raise RuntimeError('Cache sequence lengths are not synchronized at every layer')
            if capture:
                expected={('context',t,l,s) for t in (1,2) for l in self.layers for s in
                    ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
                if set(o.values)!=expected:raise RuntimeError('Missing or duplicate physical cache observation')
            self.length=total
            return boundaries,dict(o.values)
        finally:o.enabled=False

    @torch.inference_mode()
    def prefill(self,token_ids,*,capture=False):
        ids=torch.as_tensor(token_ids,dtype=torch.long)
        if ids.ndim==1:ids=ids.unsqueeze(0)
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:raise ValueError('One nonempty source prefix required')
        self.reset();self.caches=[self.factory(),self.factory()]
        if self.caches[0] is self.caches[1]:raise RuntimeError('Each recurrent pass requires an independent cache')
        h=self.loop.base.embed_tokens(ids.to(self.loop.base.embed_tokens.weight.device)).to(self.loop.device)
        positions=torch.arange(ids.shape[1],device=self.loop.device)
        return self._forward(h,positions,capture=capture)

    @torch.inference_mode()
    def step(self,token_id,*,capture=False):
        if not self.caches or self.length<1:raise RuntimeError('Prefill before decoding')
        ids=torch.tensor([[int(token_id)]],dtype=torch.long,device=self.loop.base.embed_tokens.weight.device)
        h=self.loop.base.embed_tokens(ids).to(self.loop.device);positions=torch.tensor([self.length],device=self.loop.device)
        return self._forward(h,positions,capture=capture)

    def close(self):
        self.observer.close();self.reset()
