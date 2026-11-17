"""CPU-offloaded full-vocabulary head with the original ARG-02 normalization."""
from __future__ import annotations
import torch
import torch.nn.functional as F

OPS='INSR'

def program_scores(operator_logprobs):
    lp=torch.as_tensor(operator_logprobs,dtype=torch.float32,device='cpu')
    if lp.shape!=(16,3,4):
        raise ValueError('Expected all sixteen prefix rows and three operator sites')
    return [float(lp[0,0,i]+lp[4*i,1,j]+lp[4*i+j,2,k])
            for i in range(4) for j in range(4) for k in range(4)]

def scores_at_temperatures(record,temperatures=(1.,1.,1.)):
    logits=torch.tensor(record['operator_logits'],dtype=torch.float32)
    lp=logits.clone()
    for p,temp in enumerate(temperatures):
        z=torch.tensor(record['log_normalizers'][f'{temp:g}'],dtype=torch.float64)
        lp[:,p,:]=(logits[:,p,:].double()/temp-z[:,p,None]).float()
    return program_scores(lp)

class ChunkedHead:
    def __init__(self,weight,operator_ids,*,device='cuda:0',chunk_rows=8192,bias=None):
        if weight.ndim!=2 or weight.device.type!='cpu':
            raise ValueError('Full output-head weight must remain on CPU')
        if chunk_rows<1:
            raise ValueError('chunk_rows must be positive')
        self.weight=weight.detach()
        self.bias=None if bias is None else bias.detach().cpu()
        self.device=torch.device(device)
        self.chunk_rows=int(chunk_rows)
        self.ids=[int(operator_ids[c]) for c in OPS]
        if len(set(self.ids))!=4 or min(self.ids)<0 or max(self.ids)>=weight.shape[0]:
            raise ValueError('Invalid operator token identities')

    @torch.inference_mode()
    def score(self,states,temperatures=(.5,1.,2.)):
        if not states:
            return []
        b=states[0].shape[0]
        if b!=16 or any(h.ndim!=3 or h.shape[0]!=b or h.shape[-1]!=self.weight.shape[1] for h in states):
            raise ValueError('Original sixteen-prefix scoring contract required')
        temps=tuple(sorted(set(float(t) for t in temperatures)))
        if 1. not in temps or min(temps)<=0:
            raise ValueError('Positive temperatures including 1 required')
        tails=[h[:,-3:,:].to(self.device).reshape(-1,h.shape[-1]) for h in states]
        n=len(tails);vocab,width=self.weight.shape
        chunk=min(vocab,self.chunk_rows)
        staging=torch.empty((chunk,width),device=self.device,dtype=self.weight.dtype)
        op=torch.empty((n,b,3,4),device=self.device,dtype=torch.float32)
        z={t:torch.full((n,b,3),-torch.inf,device=self.device,dtype=torch.float64) for t in temps}
        ds=torch.zeros((n,b,3),device=self.device,dtype=torch.float64)
        d2=torch.zeros_like(ds)
        dmin=torch.full_like(ds,torch.inf);dmax=torch.full_like(ds,-torch.inf)
        for start in range(0,vocab,chunk):
            stop=min(vocab,start+chunk);length=stop-start
            staging[:length].copy_(self.weight[start:stop],non_blocking=False)
            bias=None if self.bias is None else self.bias[start:stop].to(self.device)
            ref=None
            for arm,tail in enumerate(tails):
                raw=F.linear(tail,staging[:length],bias).float().reshape(b,3,length)
                for i,token in enumerate(self.ids):
                    if start<=token<stop:
                        op[arm,:,:,i]=raw[:,:,token-start]
                for t in temps:
                    part=torch.logsumexp(raw.double()/t,dim=-1)
                    z[t][arm]=torch.logaddexp(z[t][arm],part)
                if arm==0:
                    ref=raw
                delta=(raw-ref).double()
                ds[arm]+=delta.sum(-1);d2[arm]+=delta.square().sum(-1)
                dmin[arm]=torch.minimum(dmin[arm],delta.amin(-1))
                dmax[arm]=torch.maximum(dmax[arm],delta.amax(-1))
        result=[]
        for arm in range(n):
            shift=z[1.][arm]-z[1.][0]
            # Full-vocabulary LP differences from sufficient statistics, no
            # vocabulary truncation or restricted operator-only denominator.
            sq=d2[arm]-2*shift*ds[arm]+vocab*shift.square()
            maxerr=torch.maximum((dmin[arm]-shift).abs(),(dmax[arm]-shift).abs()).amax()
            rec={'operator_logits':op[arm].cpu().tolist(),
                 'log_normalizers':{f'{t:g}':z[t][arm].cpu().tolist() for t in temps},
                 'full_vocab_logprob_l2_vs_reference':float(sq.clamp_min(0).sum().sqrt()),
                 'full_vocab_max_logprob_error_vs_reference':float(maxerr)}
            rec['scores']=scores_at_temperatures(rec)
            result.append(rec)
        assert self.weight.device.type=='cpu'
        return result
