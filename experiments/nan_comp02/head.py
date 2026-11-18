"""Single next-token full-vocabulary scoring, with a CPU-resident BF16 head."""
import math
import torch
import torch.nn.functional as F

class OperatorHead:
    def __init__(self,weight,aliases,device='cuda:0',chunk_rows=8192):
        if weight.device.type!='cpu' or weight.ndim!=2:raise ValueError('Full head must remain on CPU')
        self.weight=weight.detach();self.aliases=aliases;self.device=torch.device(device);self.chunk=int(chunk_rows)
        if set(aliases)!=set('INSR') or any(not values for values in aliases.values()):raise ValueError('Four nonempty operator classes required')
        self.ids=sorted(i for values in aliases.values() for i in values)
        if self.chunk<1 or len(set(self.ids))!=len(self.ids) or min(self.ids)<0 or max(self.ids)>=weight.shape[0]:raise ValueError('Invalid answer tokens/chunk')

    @torch.inference_mode()
    def score(self,h):
        if h.ndim!=3 or h.shape[0]!=1 or h.shape[-1]!=self.weight.shape[1]:raise ValueError('One source prompt required')
        tail=h[:,-1,:].to(self.device);vocab,width=self.weight.shape
        staging=torch.empty(min(self.chunk,vocab),width,device=self.device,dtype=self.weight.dtype)
        selected={};z=torch.tensor(-torch.inf,device=self.device,dtype=torch.float64);best=-math.inf;best_id=-1
        for start in range(0,vocab,self.chunk):
            stop=min(start+self.chunk,vocab);size=stop-start;staging[:size].copy_(self.weight[start:stop])
            logits=F.linear(tail,staging[:size]).float().flatten()
            z=torch.logaddexp(z,torch.logsumexp(logits.double(),0))
            value,index=logits.max(0);value=float(value)
            if value>best:best=value;best_id=start+int(index)
            for token in self.ids:
                if start<=token<stop:selected[token]=float(logits[token-start])
        normalizer=float(z)
        if not math.isfinite(normalizer) or len(selected)!=len(self.ids):raise ValueError('Nonfinite/incomplete full-vocabulary score')
        class_lp={}
        for op,ids in self.aliases.items():
            values=[selected[i] for i in ids];mx=max(values)
            class_lp[op]=mx+math.log(sum(math.exp(v-mx) for v in values))-normalizer
        free=next((op for op,ids in self.aliases.items() if best_id in ids),None)
        grammar_mass=sum(math.exp(v) for v in class_lp.values())
        if not 0<=grammar_mass<=1+1e-8:raise ValueError('Invalid full-vocabulary grammar mass')
        return {'vocabulary_size':vocab,'selected_logits':{str(k):v for k,v in selected.items()},'full_log_normalizer':normalizer,
                'free_top_token_id':best_id,'free_top_token_logit':best,'free_operator':free,'class_logprobs':class_lp,
                'constrained_operator':max('INSR',key=lambda op:class_lp[op]),'grammar_mass':grammar_mass}
