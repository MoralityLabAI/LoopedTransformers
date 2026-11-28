"""Hard sparse bank and recurrent router development pilot."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dictionary import fit, unpack, evaluate
ROOTS={16:[1901,1907,1913],32:[2003,2011,2017]}
ARMS=('teacher_joint','teacher_warmup')
W=32

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def onehot(logits):
    p=logits.softmax(-1); h=F.one_hot(p.argmax(-1),p.shape[-1]).to(p.dtype)
    return h+(p-p.detach())
def hard(logit):
    s=logit.tanh(); return torch.where(logit>=0,1.,-1.)+(s-s.detach())

class Router(nn.Module):
    def __init__(self,k,arm):
        super().__init__(); self.k=k; self.arm=arm
        self.emb=nn.Embedding(k,W); self.rnn=nn.GRUCell(W+3,W); self.idx=nn.Linear(W,k); self.sign=nn.Linear(W,1); self.op=nn.Linear(W,2); self.length=nn.Linear(W,3)
        rot=torch.linalg.qr(torch.randn(k,k))[0]
        self.V=nn.Parameter(torch.eye(k) if arm=='identity' else rot,requires_grad=arm not in ('identity','rotated_fixed'))
    def routes(self,j):
        e=self.emb(j); h=e; idx=[]; signs=[]
        for t in range(3):
            clock=F.one_hot(torch.full_like(j,t),3).float(); h=self.rnn(torch.cat((e,clock),-1),h); idx.append(self.idx(h)); signs.append(self.sign(h).squeeze(-1))
        return torch.stack(idx,1),torch.stack(signs,1),self.op(e),self.length(e)
    def forward(self,x,j):
        il,sl,ol,ll=self.routes(j); m=onehot(il); ss=hard(sl); op=onehot(ol); length=onehot(ll)
        active=torch.stack([length[:,t:].sum(-1) for t in range(3)],1)
        read=((x@self.V)[:,None,:]*m).sum(-1); literal=hard(read)*ss
        a=torch.ones_like(x[:,0]); b=a
        for t in range(3):
            v=literal[:,t]; g=active[:,t]; a=a*(1+g*(v-1)); b=b+g*(torch.minimum(b,v)-b)
        return op[:,0]*a+op[:,1]*b,(il,sl,ol,ll),read,active

def task(k,root):
    rng=np.random.default_rng(root); codes=rng.permutation(k).tolist(); classes=[('parity',1),('parity',2),('parity',3),('and',2),('and',3)]; truth=[]; used=set()
    for j in range(k):
        op,d=classes[j%len(classes)]
        while True:
            ix=tuple(sorted(rng.choice(k,d,replace=False).tolist())); signs=tuple(rng.choice([-1,1],d).tolist())
            if op=='parity': signs=(int(np.prod(signs)),)+(1,)*(d-1)
            p=(op,ix,signs)
            if p not in used: used.add(p); truth.append(p); break
    rng.shuffle(truth); ids=list(dict.fromkeys(rng.integers(0,1<<k,size=2048,dtype=np.uint64).tolist()))[:1280]; assert len(ids)==1280
    tr,te=ids[:256],ids[256:]; x,z=unpack(tr,k),unpack(te,k); y=np.stack([evaluate(x,p) for p in truth],1); target=np.stack([evaluate(z,p) for p in truth],1)
    chosen,best,ties,counts=fit(x,y,k); ps=chosen[2]
    assert all(p[0] in ('parity','and') and 1<=len(p[1])<=3 for p in ps)
    op=torch.tensor([int(p[0]=='and') for p in ps]); lens=torch.tensor([len(p[1])-1 for p in ps]); idx=torch.tensor([list(p[1])+[0]*(3-len(p[1])) for p in ps]); sign=torch.tensor([list(p[2])+[1]*(3-len(p[2])) for p in ps]).float()
    return dict(codes=codes,truth=truth,fitted=ps,train_ids=tr,heldout_ids=te,fit_ties=ties[2].tolist(),fit_correct=((best[2]+len(x))//2).tolist(),candidate_count=counts[2]),torch.tensor(x).float(),torch.tensor(y).float(),torch.tensor(z).float(),torch.tensor(target).float(),(op,lens,idx,sign)

@torch.no_grad()
def test(m,z,y,codes,meta):
    k=m.k; il,sl,ol,ll=m.routes(codes); ix=il.argmax(-1); signs=torch.where(sl>=0,1,-1); ops=ol.argmax(-1); lens=ll.argmax(-1)+1
    out=[]; reads=[]
    for x in z.split(64):
        rows=x.repeat_interleave(k,0); j=codes.repeat(len(x)); o,_,_,a=m(rows,j); assert ((o==1)|(o==-1)).all(); assert ((a==0)|(a==1)).all(); out.append(o.reshape(-1,k)); reads.append(a.sum(-1).reshape(-1,k))
    p=torch.cat(out); a=torch.cat(reads); ok=p==y; op,l,idx,s=meta; valid=torch.arange(3)[None,:]<=l[:,None]
    exact_route=(ops==op)&(lens==l+1)&((ix==idx)|~valid).all(-1)&((signs==s)|~valid).all(-1)
    return dict(correct=int(ok.sum()),total=ok.numel(),exact_rows=int(ok.all(1).sum()),mean_active_reads=float(a.mean()),exact_instruction_queries=int(exact_route.sum()),query_routes=[dict(indices=ix[j,:lens[j]].tolist(),signs=signs[j,:lens[j]].tolist(),op='and' if ops[j] else 'parity') for j in range(k)],V=m.V.detach().tolist(),predictions=''.join('1' if a else '0' for a in (p>0).flatten().tolist()))

def run(k,steps,out):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); results=[]; out=Path(out)
    receipt=dict(k=k,steps=steps,roots=ROOTS[k],arms=ARMS,bit_cap=k.bit_length(),source_sha256=sha(__file__),dictionary_sha256=sha(Path(__file__).with_name('dictionary.py')),evidence='adaptive development; teacher-forced read target; exposed roots',results=results)
    for root in ROOTS[k]:
        info,x,y,z,target,meta=task(k,root); op,lens,idx,sign=meta; codes=torch.tensor(info['codes']); armresults=[]
        for arm in ARMS:
            torch.manual_seed(root+10000); m=Router(k,arm); opt=torch.optim.Adam(m.parameters(),lr=.01); g=torch.Generator().manual_seed(root+20000); start=time.monotonic(); losses=[]
            for step in range(steps):
                i=torch.randint(len(x),(128,),generator=g); j=torch.randint(k,(128,),generator=g); xb=x[i]; label=y[i,j]; o,(il,sl,ol,ll),read,active=m(xb,codes[j]); mask=torch.arange(3)[None,:]<=lens[j,None]
                loss=((o-label)**2).mean()+.02*active.sum(-1).mean()
                if arm=='teacher_warmup' and step<150: loss=0.*loss
                if arm!='task_only':
                    loss=loss+F.cross_entropy(il[mask],idx[j][mask])+F.binary_cross_entropy_with_logits(sl[mask],(sign[j][mask]+1)/2)+F.cross_entropy(ol,op[j])+F.cross_entropy(ll,lens[j])
                    if arm!='no_trace': loss=loss+F.binary_cross_entropy_with_logits((xb@m.V).gather(1,idx[j])[mask],(xb.gather(1,idx[j])[mask]+1)/2)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.); opt.step()
                if step%100==0 or step==steps-1: losses.append([step,float(loss.detach())])
            r=dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),seconds=time.monotonic()-start,losses=losses,**test(m,z,target,codes,meta)); armresults.append(r); print(k,root,arm,r['correct'],r['total'],r['exact_instruction_queries'],round(r['mean_active_reads'],3),round(r['seconds'],2),flush=True)
        results.append(dict(root=root,**info,labels=''.join('1' if a else '0' for a in (target>0).flatten().tolist()),arms=armresults)); out.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--k',type=int,required=True); p.add_argument('--steps',type=int,default=500); p.add_argument('--out',required=True); a=p.parse_args(); run(a.k,a.steps,a.out)
