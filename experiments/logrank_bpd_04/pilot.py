"""LOGRANK-BPD-04: adaptive exact address assignment pilot."""
import argparse, hashlib, json, time, math
from pathlib import Path
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
K,W,B=8,32,4
ROOTS=(503,607,709,811)
ARMS=('assignment','assignment_quadratic')
LAMBDA=.5
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
        super().__init__(); self.a=Agent(arm in ('quadratic','joint_quadratic','assignment_quadratic')); self.b=Agent(arm in ('quadratic','joint_quadratic','assignment_quadratic')); self.arm=arm; self.schedule='AAAA' if arm=='oneway' else 'BBBA'
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
    def query_distribution(self):
        # Enumerate all eight forced autoregressive paths for every query.
        codes=((torch.arange(K)[:,None]>>torch.arange(2,-1,-1))&1).float()
        if self.arm=='fixed': return torch.eye(K),codes
        j=torch.arange(K).repeat_interleave(K); c=codes.repeat(K,1)
        e=self.b.enc(F.one_hot(j,K).float()).tanh(); h=torch.zeros_like(e); bit=e.new_zeros(len(e)); lp=bit.clone()
        for t in range(3):
            h=self.b.step(e,h,bit,t); logit=self.b.head(h); bit=c[:,t]; lp=lp+F.logsigmoid((2*bit-1)*logit)
        return lp.reshape(K,K).exp(),codes

def entropy(p): return -(p*p.clamp_min(1e-30).log()).sum(-1)
def information(p,codes):
    joint=entropy(p.mean(0))-entropy(p).mean()
    ones=(p@codes).clamp(0,1); binary=torch.stack((1-ones,ones),-1)
    marginal=(entropy(binary.mean(0))-entropy(binary).mean(0)).sum()
    return joint,marginal

def collision_ceiling(counts):
    return sum(s/2+sum(math.comb(s,i)*abs(s-2*i) for i in range(s+1))/2**(s+1) for s in counts)/K

@torch.no_grad()
def evaluate(m,x,ids,perm):
    rows=x[ids].repeat_interleave(K,0); j=torch.arange(K).repeat(len(ids)); y=rows[torch.arange(len(rows)),perm[j]].bool()
    z,b,_=m(rows,j); zz,bb,_=m(rows,j,True); assert torch.equal(z,zz) and torch.equal(b,bb); assert ((b==0)|(b==1)).all()
    q=b.reshape(-1,K,B)
    if m.schedule=='BBBA':
        q=q[:,:,:3]; assert torch.equal(q,q[:1].expand_as(q)); code=q[0].long()@torch.tensor([4,2,1]); hist=torch.bincount(code,minlength=K); codes=int((hist>0).sum())
    else:
        assert torch.equal(q,q[:,:1].expand_as(q)); codes=None
    p,c=m.query_distribution(); assert torch.allclose(p.sum(-1),torch.ones(K),atol=2e-6)
    joint,marginal=information(p,c)
    ok=(z>=0)==y
    return dict(correct=int(ok.sum()),total=len(y),exact_rows=int(ok.reshape(-1,K).all(-1).sum()),query_codes=codes,query_map=code.tolist(),histogram=hist.tolist(),hard_entropy_bits=float(entropy(hist.float()/K)/math.log(2)),soft_joint_bits=float(joint/math.log(2)),soft_marginal_bits=float(marginal/math.log(2)),collision_ceiling=collision_ceiling(hist.tolist()),transcripts=(b.long()@torch.tensor([8,4,2,1])).tolist(),bits=B,correct_bits=''.join('1' if v else '0' for v in ok.tolist()))
def run(steps,out):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); results=[]
    x=((torch.arange(1<<K)[:,None]>>torch.arange(K))&1).float()
    for root in ROOTS:
        g0=torch.Generator().manual_seed(root+10000); p=torch.randperm(len(x),generator=g0); perm=torch.randperm(K,generator=g0); train,test=p[:128],p[128:]
        for arm in ARMS:
            torch.manual_seed(root); m=Model(arm); opt=torch.optim.Adam(m.parameters(),lr=.003); g=torch.Generator().manual_seed(root+20000); start=time.monotonic()
            for _ in range(steps):
                ix=train[torch.randint(len(train),(128,),generator=g)]; j=torch.randint(K,(128,),generator=g); rows=x[ix]; target=rows[torch.arange(len(rows)),perm[j]]; z,_,q=m(rows,j); loss=F.binary_cross_entropy_with_logits(z,target)
                if arm in ('assignment','assignment_quadratic'):
                    prob,_=m.query_distribution(); logprob=prob.clamp_min(1e-30).log(); ii,jj=linear_sum_assignment(-logprob.detach().numpy()); loss=loss-LAMBDA*logprob[ii,jj].mean()
                elif arm=='guided':
                    truth=torch.stack([((j>>(2-t))&1).float() for t in range(3)],-1); loss=loss+F.binary_cross_entropy_with_logits(torch.stack(q,-1),truth)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            m.eval(); r=dict(root=root,arm=arm,permutation=perm.tolist(),train_ids=train.tolist(),heldout_ids=test.tolist(),parameters=sum(p.numel() for p in m.parameters()),train=evaluate(m,x,train,perm),heldout=evaluate(m,x,test,perm),full=evaluate(m,x,torch.arange(len(x)),perm),seconds=round(time.monotonic()-start,3)); assert r['full']['correct']/r['full']['total']<=r['full']['collision_ceiling']+1e-9; results.append(r)
            print(root,arm,r['heldout']['correct'],r['full']['correct'],r['full']['query_codes'],round(r['full']['soft_joint_bits'],3),round(r['full']['soft_marginal_bits'],3),r['seconds'],flush=True)
            Path(out).write_text(json.dumps(receipt(results,steps),indent=2)+'\n')
    return receipt(results,steps)
def receipt(results,steps):
    return dict(evidence='adaptive development follow-up; synthetic GRUs; selection-exposed roots',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,steps=steps,lambda_assignment_nats=LAMBDA,batch=128,k=K,width=W,bit_cap=B,roots=list(ROOTS),uniform_oneway_accuracy_ceiling=1-(2**K-2**B+max(0,2**K-2**B*(K+1)))/(2**K*K),results=results)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=500); p.add_argument('--out',default='receipt.json'); a=p.parse_args(); Path(a.out).write_text(json.dumps(run(a.steps,a.out),indent=2)+'\n')
