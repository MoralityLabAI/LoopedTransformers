"""Development-only learned hard-bit recurrent indexing experiment."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
K,W,B=8,32,4
ROOTS=(101,211,307,401)
ARMS=('oneway','interactive','quadratic','guided','fixed')
class Agent(nn.Module):
    def __init__(self,quad):
        super().__init__(); self.enc=nn.Linear(K,W); self.rnn=nn.GRUCell(W+B+2,W); self.comp=nn.Linear(W,W); self.msg=nn.Linear(W,1); self.out=nn.Linear(W,1); self.quad=quad
    def head(self,h):
        z=self.comp(h)
        if self.quad:
            a,b=z.chunk(2,-1); z=torch.cat((a,a*b),-1)
        return self.msg(z).squeeze(-1)
    def step(self,e,h,bit,t):
        clock=e.new_zeros((len(e),B+1)); clock[:,t]=1
        return self.rnn(torch.cat((e,bit[:,None],clock),-1),h)
class Model(nn.Module):
    def __init__(self,arm):
        super().__init__(); self.a=Agent(arm=='quadratic'); self.b=Agent(arm=='quadratic'); self.arm=arm; self.schedule='AAAA' if arm=='oneway' else 'BBBA'
    def forward(self,x,j,replay=False):
        ea=self.a.enc(2*x-1).tanh(); eb=self.b.enc(F.one_hot(j,K).float()).tanh(); ha,hb=torch.zeros_like(ea),torch.zeros_like(eb); bit=x.new_zeros(len(x)); sent=[]; query=[]
        for t,s in enumerate(self.schedule):
            ha=self.a.step(ea,ha,bit,t); hb=self.b.step(eb,hb,bit,t)
            logit=self.a.head(ha) if s=='A' else self.b.head(hb)
            if s=='B': query.append(logit)
            if self.arm=='fixed' and s=='B': bit=((j>>(2-t))&1).float()
            elif replay: bit=(logit>=0).long().float()
            else:
                soft=logit.sigmoid(); bit=(logit>=0).float()+(soft-soft.detach())
            sent.append(bit)
        ha=self.a.step(ea,ha,bit,B); hb=self.b.step(eb,hb,bit,B)
        return self.b.out(hb).squeeze(-1),torch.stack(sent,-1),query
@torch.no_grad()
def evaluate(m,x,ids):
    rows=x[ids].repeat_interleave(K,0); j=torch.arange(K).repeat(len(ids)); y=rows[torch.arange(len(rows)),j].bool()
    z,b,_=m(rows,j); zz,bb,_=m(rows,j,True); assert torch.equal(z,zz) and torch.equal(b,bb); assert ((b==0)|(b==1)).all()
    q=b.reshape(-1,K,B)
    if m.schedule=='BBBA':
        q=q[:,:,:3]; assert torch.equal(q,q[:1].expand_as(q)); codes=len((q[0].long()@torch.tensor([4,2,1])).unique())
    else:
        assert torch.equal(q,q[:,:1].expand_as(q)); codes=None
    ok=(z>=0)==y
    return dict(correct=int(ok.sum()),total=len(y),exact_rows=int(ok.reshape(-1,K).all(-1).sum()),query_codes=codes,bits=B,correct_bits=''.join('1' if v else '0' for v in ok.tolist()))
def run(steps):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); results=[]
    x=((torch.arange(1<<K)[:,None]>>torch.arange(K))&1).float()
    for root in ROOTS:
        p=torch.randperm(len(x),generator=torch.Generator().manual_seed(root+10000)); train,test=p[:128],p[128:]
        for arm in ARMS:
            torch.manual_seed(root); m=Model(arm); opt=torch.optim.Adam(m.parameters(),lr=.003); g=torch.Generator().manual_seed(root+20000); start=time.monotonic()
            for _ in range(steps):
                ix=train[torch.randint(len(train),(128,),generator=g)]; j=torch.randint(K,(128,),generator=g); rows=x[ix]; target=rows[torch.arange(len(rows)),j]; z,_,q=m(rows,j); loss=F.binary_cross_entropy_with_logits(z,target)
                if arm=='guided':
                    truth=torch.stack([((j>>(2-t))&1).float() for t in range(3)],-1); loss=loss+F.binary_cross_entropy_with_logits(torch.stack(q,-1),truth)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            m.eval(); r=dict(root=root,arm=arm,parameters=sum(p.numel() for p in m.parameters()),train=evaluate(m,x,train),heldout=evaluate(m,x,test),full=evaluate(m,x,torch.arange(len(x))),seconds=round(time.monotonic()-start,3)); results.append(r)
            print(root,arm,r['heldout']['correct'],r['full']['correct'],r['full']['query_codes'],r['seconds'],flush=True)
    return dict(evidence='development only; synthetic GRU, BPD-inspired quadratic head',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,steps=steps,batch=128,k=K,width=W,bit_cap=B,roots=list(ROOTS),uniform_oneway_accuracy_ceiling=1-(2**K-2**B+max(0,2**K-2**B*(K+1)))/(2**K*K),results=results)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=350); p.add_argument('--out',default='receipt.json'); a=p.parse_args(); Path(a.out).write_text(json.dumps(run(a.steps),indent=2)+'\n')
