"""Fresh scaled indexing development pilot, not native BPD."""
import argparse, json, time, hashlib, math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
ARMS=('flat','hier','hier_factor','fixed')
ROOTS={16:[1103,1201],32:[1301,1409]}
W=32
class Agent(nn.Module):
    def __init__(self,k,q,factor):
        super().__init__(); self.q=q; self.enc=nn.Linear(k,W); self.rnn=nn.GRUCell(W+q+3,W); self.comp=nn.Sequential(nn.Linear(W,16,bias=False),nn.Linear(16,W)) if factor else nn.Linear(W,W); self.msg=nn.Linear(W,1); self.out=nn.Linear(W,1)
    def step(self,e,h,bit,t):
        clk=e.new_zeros(len(e),self.q+2); clk[:,t]=1
        return self.rnn(torch.cat((e,bit[:,None],clk),1),h)
    def head(self,h): return self.msg(self.comp(h)).squeeze(-1)
class Model(nn.Module):
    def __init__(self,k,arm):
        super().__init__(); self.k=k; self.q=k.bit_length()-1; self.arm=arm; self.a=Agent(k,self.q,arm=='hier_factor'); self.b=Agent(k,self.q,arm=='hier_factor')
    def forward(self,x,j,replay=False):
        ea=self.a.enc(2*x-1).tanh(); eb=self.b.enc(F.one_hot(j,self.k).float()).tanh(); ha=torch.zeros_like(ea); hb=torch.zeros_like(eb); bit=x.new_zeros(len(x)); bits=[]
        for t in range(self.q+1):
            ha=self.a.step(ea,ha,bit,t); hb=self.b.step(eb,hb,bit,t); z=self.b.head(hb) if t<self.q else self.a.head(ha)
            if self.arm=='fixed' and t<self.q: bit=((j>>(self.q-1-t))&1).float()
            elif replay: bit=(z>=0).float()
            else:
                s=z.sigmoid(); bit=(z>=0).float()+(s-s.detach())
            bits.append(bit)
        hb=self.b.step(eb,hb,bit,self.q+1)
        return self.b.out(hb).squeeze(-1),torch.stack(bits,1)
    def aux(self):
        k,q=self.k,self.q
        if self.arm=='flat':
            j=torch.arange(k).repeat_interleave(k); c=((torch.arange(k)[:,None]>>torch.arange(q-1,-1,-1))&1).float().repeat(k,1)
            e=self.b.enc(F.one_hot(j,k).float()).tanh(); h=torch.zeros_like(e); bit=e.new_zeros(len(e)); lp=bit.clone()
            for t in range(q):
                h=self.b.step(e,h,bit,t); z=self.b.head(h); bit=c[:,t]; lp+=F.logsigmoid((2*bit-1)*z)
            lp=lp.reshape(k,k); i,j=linear_sum_assignment(-lp.detach().numpy()); return -lp[i,j].mean()
        j=torch.arange(k); e=self.b.enc(F.one_hot(j,k).float()).tanh(); h=torch.zeros_like(e); bit=e.new_zeros(k); prefix=torch.zeros(k,dtype=torch.long); loss=0.
        for t in range(q):
            h=self.b.step(e,h,bit,t); z=self.b.head(h); target=torch.zeros_like(z)
            with torch.no_grad():
                for g in range(1<<t):
                    ids=torch.where(prefix==g)[0]; assert len(ids)==k>>t
                    order=ids[torch.argsort(z[ids],stable=True)]; target[order[len(ids)//2:]]=1.
                prefix=2*prefix+target.long()
            loss+=F.binary_cross_entropy_with_logits(z,target); bit=target
        assert len(prefix.unique())==k
        return loss
@torch.no_grad()
def evaluate(m,ids,perm):
    k=m.k; txs=[]; preds=[]; ys=[]
    for part in torch.tensor(ids).split(32):
        x=((part[:,None]>>torch.arange(k))&1).float().repeat_interleave(k,0); j=torch.arange(k).repeat(len(part)); y=x[torch.arange(len(x)),perm[j]].long(); z,b=m(x,j); zz,bb=m(x,j,True); assert torch.equal(z,zz) and torch.equal(b,bb); assert ((b==0)|(b==1)).all()
        txs.extend((b.long()@(2**torch.arange(m.q,-1,-1))).tolist()); preds.extend((z>=0).long().tolist()); ys.extend(y.tolist())
    mapping=[t//2 for t in txs[:k]]; hist=[mapping.count(i) for i in range(k)]; alice={}; bob={}
    for i,x in enumerate(ids):
        for j in range(k):
            ix=i*k+j; t=txs[ix]; assert 0<=t<2*k and t//2==mapping[j]; assert alice.setdefault((x,t//2),t%2)==t%2; assert bob.setdefault((j,t),preds[ix])==preds[ix]; assert ys[ix]==(x>>int(perm[j]))&1
    ok=[p==y for p,y in zip(preds,ys)]; ceiling=sum(s/2+sum(math.comb(s,i)*abs(s-2*i) for i in range(s+1))/2**(s+1) for s in hist)/k
    return dict(correct=sum(ok),total=len(ok),exact_rows=sum(all(ok[i:i+k]) for i in range(0,len(ok),k)),query_codes=sum(s>0 for s in hist),query_map=mapping,uniform_collision_ceiling=ceiling,transcripts=txs,predictions=''.join(map(str,preds)),labels=''.join(map(str,ys)))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--k',type=int,required=True); p.add_argument('--steps',type=int,default=500); p.add_argument('--out',required=True); args=p.parse_args(); k=args.k; q=k.bit_length()-1; torch.set_num_threads(1); torch.use_deterministic_algorithms(True); out=Path(args.out); results=[]
    receipt=dict(k=k,bit_cap=q+1,steps=args.steps,roots=ROOTS[k],arms=ARMS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),evidence='fresh development, indexing isomorphisms',results=results)
    for root in ROOTS[k]:
        g=torch.Generator().manual_seed(root+10000); ids=list(dict.fromkeys(torch.randint(0,1<<k,(4096,),generator=g).tolist()))[:2048]; assert len(ids)==2048; train,test=ids[:1024],ids[1024:]; perm=torch.randperm(k,generator=g); pool=((torch.tensor(train)[:,None]>>torch.arange(k))&1).float()
        for arm in ARMS:
            torch.manual_seed(root); m=Model(k,arm); opt=torch.optim.Adam(m.parameters(),lr=.003); g=torch.Generator().manual_seed(root+20000); start=time.monotonic(); auxsecs=0.
            for _ in range(args.steps):
                x=pool[torch.randint(1024,(128,),generator=g)]; j=torch.randint(k,(128,),generator=g); y=x[torch.arange(128),perm[j]]; z,_=m(x,j); loss=F.binary_cross_entropy_with_logits(z,y)
                if arm!='fixed':
                    t=time.monotonic(); aux=m.aux(); auxsecs+=time.monotonic()-t; loss=loss+.5*aux
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            secs=time.monotonic()-start; m.eval(); r=dict(root=root,arm=arm,parameters=sum(p.numel() for p in m.parameters()),permutation=perm.tolist(),train_ids=train,heldout_ids=test,training_seconds=secs,aux_forward_seconds=auxsecs,aux_path_positions_per_step=k*k*q if arm=='flat' else k*q if arm!='fixed' else 0,train=evaluate(m,train[:256],perm),heldout=evaluate(m,test[:256],perm)); results.append(r)
            if arm=='hier_factor': torch.save(m.state_dict(),out.with_name(f'factor_{k}_{root}.pt'))
            print(k,root,arm,r['heldout']['correct'],r['heldout']['total'],r['heldout']['query_codes'],r['heldout']['exact_rows'],round(secs,2),flush=True); out.write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__': main()
