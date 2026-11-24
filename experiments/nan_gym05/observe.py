"""Read-only last-prompt-token observation at physical residual layer outputs."""
import torch

class TowerObserver:
    def __init__(self,loop,layers):
        self.loop=loop;self.layers=tuple(layers);self.enabled=False
        self.handles=[]
        modules=list(loop.layers)+([loop.stage_layer] if loop.stage_layer is not None else [])
        for layer in modules:
            self.handles.append(layer.register_forward_hook(self.hook))
            for name,module in [('down_proj',layer.mlp.down_proj),('o_proj',layer.self_attn.o_proj)]:
                self.handles.append(module.register_forward_hook(self.module_hook(name)))

    def store(self,site,h):
        if not self.enabled or self.loop.active_layer not in self.layers:return
        key=(self.phase,self.loop.active_layer,site)
        i=self.counts.get(key,0);self.counts[key]=i+1
        if i>=len(self.names):raise RuntimeError('Unexpected B1 site invocation')
        if h.shape[0]!=1:raise RuntimeError('Observer requires independent B1 contexts')
        record=(self.names[i],self.phase,self.loop.active_layer,site)
        if record in self.values:raise RuntimeError('Duplicate physical site')
        self.values[record]=h[0,-1].detach().float().cpu().clone()

    def module_hook(self,name):
        def hook(module,args,output):
            self.store(name+'_input',args[0]);self.store(name+'_output',output)
            return None
        return hook

    def hook(self,module,args,output):
        if not self.enabled or self.loop.active_layer not in self.layers:
            return None
        self.store('residual_after_layer',output[0])
        return None

    @torch.inference_mode()
    def forward(self,sequences,*,capture=False):
        self.enabled=capture;self.names=list(sequences);self.counts={};self.values={}
        prepared=[self.loop.prepare(torch.tensor([ids],dtype=torch.long)) for ids in sequences.values()]
        try:
            self.phase=1;one=self.loop.run_pairs(prepared)
            self.phase=2;two=self.loop.run_pairs([(h,ctx) for h,(_,ctx) in zip(one,prepared)])
            if capture:
                expected={(n,t,l,s) for n in sequences for t in (1,2) for l in self.layers
                          for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
                if set(self.values)!=expected:raise RuntimeError('Missing or extra tower observation')
            return two,dict(self.values),one
        finally:
            self.enabled=False

    def close(self):
        for h in self.handles:h.remove()
