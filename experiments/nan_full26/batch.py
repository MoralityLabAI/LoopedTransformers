"""One true decoder GEMM batch; never the inherited serial B1 run_pairs loop."""
from types import MethodType

def install(loop):
    import torch
    original=loop.run_pairs
    @torch.inference_mode()
    def run_pairs(self,items):
        assert len(items)==1
        h,ctx=items[0]
        for index in range(self.n):
            self.active_layer=index
            layer=self.layers[index] if index<self.n_gpu_layers else self._load_stage(self.layers[index])
            h=layer(h,attention_mask=ctx['mask'],position_ids=ctx['pos'],past_key_value=None,
                output_attentions=False,use_cache=False,cache_position=ctx['cache_position'],loop_idx=0)[0]
        return [self.base.norm(h)]
    loop.run_pairs=MethodType(run_pairs,loop)
    return original

class BatchObserver:
    def __init__(self,loop,layers):
        self.loop=loop;self.layers=layers;self.handles=[];self.enabled=False
        for layer in list(loop.layers)+([loop.stage_layer] if loop.stage_layer is not None else []):
            self.handles.append(layer.register_forward_hook(lambda m,a,o:self.store('residual_after_layer',o[0])))
            for name,module in [('down_proj',layer.mlp.down_proj),('o_proj',layer.self_attn.o_proj)]:
                def hook(m,a,o,site=name):
                    self.store(site+'_input',a[0]);self.store(site+'_output',o)
                self.handles.append(module.register_forward_hook(hook))
    def store(self,site,value):
        if not self.enabled or self.loop.active_layer not in self.layers:return
        for b,length in enumerate(self.lengths):
            key=(b,self.phase,self.loop.active_layer,site);assert key not in self.values
            self.values[key]=value[b,length-1].detach().clone()
    def close(self):
        for h in self.handles:h.remove()

class Embeddings:
    """Exact token lookup reuse only; reset per trajectory pair, no KV cache."""
    def __init__(self,loop):self.loop=loop;self.cache={};self.gathered=0
    def reset(self):self.cache={};self.gathered=0
    def prepare(self,sequences):
        import torch
        device=self.loop.device;length=max(map(len,sequences));B=len(sequences)
        embed=self.loop.base.embed_tokens
        h=torch.zeros((B,length,embed.weight.shape[1]),device=device,dtype=embed.weight.dtype)
        attention=torch.zeros((B,length),device=device,dtype=torch.long)
        for b,ids in enumerate(sequences):
            old=self.cache.get(b)
            if old is None:values=embed(torch.tensor([ids],dtype=torch.long)).to(device);self.gathered+=len(ids)
            elif old[0]==ids:values=old[1]
            else:
                assert len(ids)==len(old[0])+1 and ids[:-1]==old[0]
                values=torch.cat([old[1],embed(torch.tensor([[ids[-1]]],dtype=torch.long)).to(device)],dim=1);self.gathered+=1
            self.cache[b]=(list(ids),values);h[b,:len(ids)].copy_(values[0]);attention[b,:len(ids)]=1
        cache=torch.arange(length,device=device);pos=cache.unsqueeze(0).expand(B,-1)
        mask=self.loop.base._update_causal_mask(attention,h,cache,None,False)
        return h,dict(mask=mask,pos=pos,cache_position=cache)

def logits(head,hidden):
    import torch
    import torch.nn.functional as F
    pieces=[]
    for start in range(0,len(head.weight),head.chunk):
        pieces.append(F.linear(hidden,head.weight[start:start+head.chunk]).float())
    return torch.cat(pieces,dim=-1)

class RowInstrument:
    """Dispatch only the B row into the unchanged pinned BPD pointwise algebra."""
    def __init__(self,loop,backend,bpd_root,cfg):
        from experiments.nan_story23.instrument import Instrument
        self.base=Instrument(loop,backend,bpd_root,cfg)
        self.core=self.base.core;self.original=self.core.forward_kernel
        owner=self
        def forward(core,x,native):
            import torch
            owner.native=[native[b,n-1].detach().cpu().clone() for b,n in enumerate(owner.lengths)]
            owner.raw_by_phase.append(owner.native)
            if core.mode=='target':return native
            selected=[i for i,m in enumerate(owner.modes) if m!='native'];assert len(selected)==1
            b=selected[0];n=owner.lengths[b]
            edited=owner.original(x[b:b+1,:n],native[b:b+1,:n])
            result=native.clone();result[b:b+1,:n]=edited
            # Keep actual final valid-token mask; never a right-padding mask.
            mask=core.controller_observations[-1]['mask'][0,-1].detach().cpu().clone() if core.live_controller else owner.base.last_masks[0,core.calls-1].copy()
            owner.mask_by_phase.append(mask)
            return result
        self.core.forward_kernel=MethodType(forward,self.core)
    def context(self,modes,lengths):
        from contextlib import nullcontext
        self.modes=modes;self.lengths=lengths;self.raw_by_phase=[];self.mask_by_phase=[]
        selected=[m for m in modes if m!='native']
        return self.base.context(selected[0],max(lengths)) if selected else nullcontext()
    def close(self):self.base.close()
