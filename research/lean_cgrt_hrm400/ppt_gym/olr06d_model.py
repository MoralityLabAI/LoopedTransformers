"""OLR-06D: tied position/factor incidence reader for public rewrite distance.

All models see the complete public word pair, equation roster, applicability
features and exact PDB lower bound. Outputs are nonnegative distance residuals,
never proof facts or admissible bounds.
"""
from __future__ import annotations
import hashlib
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from .equality_suite import ALPHABET
from .olr06d_search import PatternDB

N=10
L=2*N-1
D=132
WIDTH=32
VISITS=3
ARMS=("bi","one_way","local","dense","full")
SEEDS=(17,29,43)
SCALES=(0.,.25,.5,1.,1.5,2.)


class PacketMaker:
    def __init__(self,case,pdb):
        self.case,self.pdb=case,pdb
        self.n=len(case.source)
        self.letters={c:i for i,c in enumerate(ALPHABET)}
        rules=sorted(tuple(sorted(e)) for e in case.equations)
        self.rules=tuple(rules)
        directed=[p for a,b in rules for p in (a,b)]
        self.left=torch.tensor([self.letters[w[0]] for w in directed])
        self.right=torch.tensor([self.letters[w[1]] for w in directed])
        goal=torch.tensor([self.letters[c] for c in case.source])
        self.goal=goal
        onehot=F.one_hot(goal,4).float()
        self.base=torch.zeros(L,D);self.inc=torch.zeros(L,L)
        eq=torch.tensor([self.letters[c] for pair in rules for word in pair for c in word])
        eq=F.one_hot(eq,4).float().flatten()
        n=self.n
        self.base[:n,0]=1.;self.base[:n,1]=1.
        self.base[N:N+n-1,0]=1.;self.base[N:N+n-1,2]=1.
        self.base[:n,3]=torch.arange(n)/9.
        self.base[N:N+n-1,3]=torch.arange(n-1)/9.
        self.base[:,4]=self.base[:,0]*n/10.
        self.base[:n,12:16]=onehot
        self.base[N:N+n-1,12:16]=onehot[:-1]
        self.base[N:N+n-1,20:24]=onehot[1:]
        self.base[:,36:]=self.base[:,0:1]*eq
        for j in range(n-1):
            self.inc[N+j,j]=1.;self.inc[N+j,j+1]=1.

    def batch(self,words,bounds=None):
        n=self.n;b=len(words)
        if not b or any(len(w)!=n for w in words):
            raise ValueError("invalid public word packet")
        if bounds is None:bounds=[self.pdb.bound(w) for w in words]
        a=torch.tensor([[self.letters[c] for c in w] for w in words])
        x=self.base.unsqueeze(0).expand(b,-1,-1).clone()
        one=F.one_hot(a,4).float();match=(a==self.goal)
        x[:,:,5]=x[:,:,0]*torch.tensor(bounds,dtype=torch.float32)[:,None]/15.
        x[:,:,6]=x[:,:,0]*(~match).sum(1)[:,None]/10.
        x[:,:n,7]=match.float()
        x[:,N:N+n-1,7]=(match[:,:-1]&match[:,1:]).float()
        x[:,:n,8:12]=one
        x[:,N:N+n-1,8:12]=one[:,:-1]
        x[:,N:N+n-1,16:20]=one[:,1:]
        legal=(a[:,:-1,None]==self.left)&(a[:,1:,None]==self.right)
        x[:,N:N+n-1,24:36]=legal.float()
        return x,self.inc.unsqueeze(0).expand(b,-1,-1)


class DistanceReader(nn.Module):
    def __init__(self,arm,width=WIDTH):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown distance reader")
        self.arm=arm
        if arm=="full":
            self.full=nn.Sequential(nn.Linear(L*D+L*L,2*width),nn.GELU(),
                                    nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,1))
            return
        self.enc=nn.Sequential(nn.Linear(D,width),nn.GELU(),nn.Linear(width,width))
        if arm in ("bi","one_way"):
            self.down=nn.Linear(width,width,bias=False)
            self.up=nn.Linear(width,width,bias=False)
            self.cell=nn.Sequential(nn.Linear(4*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.norm=nn.LayerNorm(width)
        if arm=="dense":
            self.inc=nn.Linear(2*L,width,bias=False)
            self.attn=nn.MultiheadAttention(width,4,batch_first=True)
            self.ff=nn.Sequential(nn.Linear(2*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,x,inc,depth=VISITS):
        if self.arm=="full":
            return self.full(torch.cat((x.flatten(1),inc.flatten(1)),1)).squeeze(-1).sigmoid()
        active=x[...,0:1];e=self.enc(x)*active
        if self.arm=="dense":e=e+self.inc(torch.cat((inc,inc.transpose(1,2)),-1))*active
        h=e
        if self.arm in ("bi","one_way"):
            for _ in range(depth):
                down=torch.bmm(inc,self.down(h))*float(self.arm=="bi")
                up=torch.bmm(inc.transpose(1,2),self.up(h))
                h=self.norm(h+self.cell(torch.cat((h,down,up,e),-1)))*active
        elif self.arm=="dense":
            for _ in range(depth):
                a,_=self.attn(h,h,h,key_padding_mask=active.squeeze(-1)<.5,need_weights=False)
                h=self.n1(h+a)
                h=self.n2(h+self.ff(torch.cat((h,e),-1)))*active
        pooled=(h*active).sum(1)/active.sum(1)
        maximum=h.masked_fill(active<.5,-torch.inf).max(1).values
        return self.head(torch.cat((pooled,maximum),1)).squeeze(-1).sigmoid()


def weight_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(json.dumps(t.detach().tolist(),separators=(",",":")).encode())
    return h.hexdigest()


def encode(cases,records):
    xs=[];incs=[];bounds=[];distances=[]
    for case,rows in zip(cases,records,strict=True):
        p=PatternDB(case.equations,case.source);maker=PacketMaker(case,p)
        words=[r["word"] for r in rows];h=[p.bound(w) for w in words]
        d=[r["distance"] for r in rows]
        if any(not 0<=a<=b<=15 for a,b in zip(h,d,strict=True)):
            raise ValueError("invalid exact source distance/abstraction bound")
        x,i=maker.batch(words,h);xs.append(x);incs.append(i)
        bounds.extend(h);distances.extend(d)
    return torch.cat(xs),torch.cat(incs),torch.tensor(bounds,dtype=torch.float32),torch.tensor(distances,dtype=torch.float32)


@torch.no_grad()
def predict(model,data,batch=128):
    x,i,h,d=data
    return torch.cat([model(x[j:j+batch],i[j:j+batch]) for j in range(0,len(x),batch)])*15.


def fit(train,calibration,seed,arm,steps=160):
    t=time.perf_counter()
    torch.manual_seed(seed);m=DistanceReader(arm);init=weight_hash(m)
    x,i,h,d=train;target=(d-h)/15.
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.005)
    rng=torch.Generator().manual_seed(seed+6620)
    roster=torch.randint(len(x),(steps,32),generator=rng)
    first=last=None
    for j,ids in enumerate(roster):
        z=m(x[ids],i[ids])
        loss=F.smooth_l1_loss(z,target[ids],beta=.1)
        if not torch.isfinite(loss):raise ValueError("nonfinite source residual loss")
        if j==0:first=float(loss.detach())
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        last=float(loss.detach())
    m.eval()
    pr=predict(m,calibration);_,_,ch,cd=calibration
    errors=[float((ch+a*pr-cd).abs().mean()) for a in SCALES]
    k=min(range(len(SCALES)),key=lambda j:(errors[j],j))
    tr=predict(m,train)
    return m,{"seed":seed,"arm":arm,"steps":steps,"parameters":sum(p.numel() for p in m.parameters()),
              "initial_sha256":init,"weights_sha256":weight_hash(m),
              "sampler_sha256":hashlib.sha256(json.dumps(roster.tolist()).encode()).hexdigest(),
              "first_loss":first,"last_loss":last,"scale":SCALES[k],
              "source_mae":float((h+SCALES[k]*tr-d).abs().mean()),
              "calibration_pdb_mae":errors[0],"calibration_mae":errors[k],
              "calibration_grid":dict(zip(map(str,SCALES),errors)),
              "training_seconds":time.perf_counter()-t}


def residual_reader(model,maker,scale):
    @torch.no_grad()
    def rank(words,bounds):
        # Count all requested model rows even at scale zero, preserving
        # the registered model inference schedule for calibrated arms.
        x,i=maker.batch(words,bounds)
        return (15.*scale*model(x,i)).tolist()
    return rank
