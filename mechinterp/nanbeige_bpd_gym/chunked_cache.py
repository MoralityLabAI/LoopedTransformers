"""Unregistered future cache prototype; no model loading or pretrained worker.

Two independent phase caches; layer-major chunked prefill loads a staged layer
once per phase. A padded physical cache canvas is preserved through decoding
and row pruning. Semantic RoPE positions remain separate from cache positions.
Chunked/incremental arithmetic is a new condition, not native bit parity.
"""
from contextlib import nullcontext
from types import SimpleNamespace


class ChunkObserver:
    def __init__(self,loop,layers):
        self.loop=loop;self.layers=tuple(layers);self.handles=[];self.values={};self.enabled=False;self.frame=None
        for layer in list(loop.layers)+([loop.stage_layer] if loop.stage_layer is not None else []):
            self.handles.append(layer.register_forward_hook(lambda m,a,o:self.store('residual_after_layer',o[0])))
            for name,module in [('down_proj',layer.mlp.down_proj),('o_proj',layer.self_attn.o_proj)]:
                def hook(m,a,o,site=name):self.store(site+'_input',a[0]);self.store(site+'_output',o)
                self.handles.append(module.register_forward_hook(hook))
    def store(self,site,value):
        if not self.enabled or self.loop.active_layer not in self.layers:return
        f=self.frame
        for row,offset in enumerate(f.capture_offsets):
            if offset is None:continue
            key=(row,f.phase,self.loop.active_layer,site)
            if key in self.values:raise RuntimeError('Duplicate last-valid-token physical observation')
            self.values[key]=value[row,offset].detach().clone()
    def close(self):
        for handle in self.handles:handle.remove()
        self.values={}


class ChunkedTwoLoopCache:
    def __init__(self,loop,*,prefill_chunk=128,observe_layers=(5,11,17,21),inplace=True,cache_factory=None,instrument_factory=None):
        import torch
        if prefill_chunk<1:raise ValueError('Positive prefill chunk required')
        if any(p.requires_grad for p in loop.model.parameters()):raise ValueError('Frozen target required')
        if loop.base.embed_tokens.max_norm is not None:raise ValueError('Immutable checkpoint embedding lookup required')
        if cache_factory is None:
            from transformers.cache_utils import DynamicCache
            cache_factory=DynamicCache
        self.loop=loop;self.chunk=prefill_chunk;self.inplace=inplace;self.factory=cache_factory
        self.instrument=instrument_factory(loop) if instrument_factory else None
        # Component hooks must precede physical output observations.
        self.observer=ChunkObserver(loop,observe_layers);self.reset()
    def reset(self):
        self.caches=[];self.physical_length=0;self.semantic_lengths=[];self.modes=();self.attention=None
        self.observer.values={};self.stage_materializations=0;self.layer_chunk_calls=0
    def cache_bytes(self):return sum(t.numel()*t.element_size() for c in self.caches for t in (*c.key_cache,*c.value_cache))
    def _context(self):return self.instrument.context(self.modes) if self.instrument else nullcontext()
    def _walk(self,h,phase,targets,prefill):
        import torch
        self.loop.phase=phase;cache=self.caches[phase-1];P=h.shape[1];chunk=self.chunk if prefill else 1
        for index in range(self.loop.n):
            self.loop.active_layer=index
            if index<self.loop.n_gpu_layers:layer=self.loop.layers[index]
            else:layer=self.loop._load_stage(self.loop.layers[index]);self.stage_materializations+=1
            if layer.self_attn.layer_idx!=index:raise RuntimeError('Physical cache layer index changed')
            destination=h if self.inplace else torch.empty_like(h)
            for start in range(0,P,chunk):
                stop=min(P,start+chunk)
                physical=torch.arange(start,stop,device=self.loop.device) if prefill else torch.tensor([self.physical_length],device=self.loop.device)
                positions=physical.unsqueeze(0).expand(len(self.semantic_lengths),-1) if prefill else torch.tensor(self.semantic_lengths,device=self.loop.device).unsqueeze(1)
                valid=[max(0,min(stop,n)-start) for n in self.semantic_lengths] if prefill else [1]*len(self.semantic_lengths)
                frame=SimpleNamespace(phase=phase,start=start,stop=stop,valid_lengths=valid,
                    capture_offsets=[p-start if start<=p<stop else None for p in targets],prefill=prefill)
                self.observer.frame=frame
                if self.instrument:self.instrument.set_frame(frame)
                mask=self.loop.base._update_causal_mask(self.attention,h[:,start:stop],physical,cache,False)
                value=layer(h[:,start:stop],attention_mask=mask,position_ids=positions,past_key_value=cache,
                    output_attentions=False,use_cache=True,cache_position=physical,loop_idx=0)[0]
                # In-place reuse is safe only when projected keys/values own
                # storage distinct from the full residual canvas.
                for data in (cache.key_cache[index],cache.value_cache[index]):
                    if data.untyped_storage().data_ptr()==h.untyped_storage().data_ptr():raise RuntimeError('K/V aliases overwritten residual input')
                destination[:,start:stop].copy_(value);self.layer_chunk_calls+=1
                del value,mask,frame
            h=destination
        return self.loop.base.norm(h)
    def _capture(self,h,targets,prefill,capture):
        self.observer.enabled=capture;self.observer.values={};boundaries=[]
        try:
            with self._context():
                for phase in (1,2):
                    h=self._walk(h,phase,targets,prefill)
                    import torch
                    boundaries.append(torch.stack([h[b,p] for b,p in enumerate(targets)]).detach().clone())
                length=h.shape[1] if prefill else self.physical_length+1
                if any(c.get_seq_length(i)!=length for c in self.caches for i in range(self.loop.n)):
                    raise RuntimeError('Every physical layer and phase must have the same cache canvas length')
            if capture and len(self.observer.values)!=len(self.semantic_lengths)*2*len(self.observer.layers)*5:
                raise RuntimeError('Missing last-valid-token physical vector')
            return dict(boundaries=boundaries,states=dict(self.observer.values))
        finally:self.observer.enabled=False
    def prefill(self,sequences,*,modes=None,capture=True):
        import torch
        if not sequences or any(not ids for ids in sequences):raise ValueError('Nonempty rows required')
        vocab=self.loop.base.embed_tokens.weight.shape[0]
        if any(type(t) is not int or not 0<=t<vocab for ids in sequences for t in ids):raise ValueError('Invalid token IDs')
        modes=tuple(['native']*len(sequences) if modes is None else modes)
        allowed={'native','all_one','component_only','static_CAL_mean','reached_state'}
        if any(mode not in allowed for mode in modes):raise ValueError('Unknown route label')
        if any(mode!='native' for mode in modes) and self.instrument is None:raise ValueError('Edited routes require an actual BPD instrument')
        if sum(mode!='native' for mode in modes)>1:raise ValueError('One edited row per genuine cache batch')
        if len(modes)!=len(sequences):raise ValueError('One fixed route per row required')
        self.reset();self.modes=modes;self.semantic_lengths=list(map(len,sequences));P=max(self.semantic_lengths)
        self.caches=[self.factory(),self.factory()]
        if self.caches[0] is self.caches[1]:raise ValueError('Independent phase caches required')
        embed=self.loop.base.embed_tokens
        with torch.inference_mode():
            h=torch.zeros((len(sequences),P,embed.weight.shape[1]),device=self.loop.device,dtype=embed.weight.dtype)
            self.attention=torch.zeros((len(sequences),P),device=self.loop.device,dtype=torch.long)
            for row,ids in enumerate(sequences):
                h[row,:len(ids)].copy_(embed(torch.tensor(ids,device=embed.weight.device)).to(self.loop.device));self.attention[row,:len(ids)]=1
            result=self._capture(h,[n-1 for n in self.semantic_lengths],True,capture);self.physical_length=P
        return result
    def step(self,tokens,*,capture=True):
        import torch
        if not self.caches:raise RuntimeError('Prefill before decoding')
        vocab=self.loop.base.embed_tokens.weight.shape[0]
        if len(tokens)!=len(self.semantic_lengths) or any(type(t) is not int or not 0<=t<vocab for t in tokens):raise ValueError('One valid token per active row required')
        with torch.inference_mode():
            embed=self.loop.base.embed_tokens
            h=embed(torch.tensor(tokens,device=embed.weight.device).unsqueeze(1)).to(self.loop.device)
            self.attention=torch.cat([self.attention,torch.ones((len(tokens),1),device=self.loop.device,dtype=torch.long)],dim=1)
            result=self._capture(h,[0]*len(tokens),False,capture)
            self.physical_length+=1;self.semantic_lengths=[n+1 for n in self.semantic_lengths]
        return result
    def select_rows(self,indices):
        import torch
        if not self.caches or not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(self.semantic_lengths) for i in indices):
            raise ValueError('Select nonempty unique active rows')
        with torch.inference_mode():
            index=torch.tensor(indices,device=self.loop.device)
            for cache in self.caches:
                for i in range(len(cache.key_cache)):
                    cache.key_cache[i]=cache.key_cache[i].index_select(0,index)
                    cache.value_cache[i]=cache.value_cache[i].index_select(0,index)
            self.attention=self.attention.index_select(0,index)
            self.semantic_lengths=[self.semantic_lengths[i] for i in indices];self.modes=tuple(self.modes[i] for i in indices)
    def close(self):
        self.observer.close()
        if self.instrument:self.instrument.close()
        self.reset()
