"""Source-supervised applicability priors with common public syntax encoding."""
from __future__ import annotations
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F

from .olr06c_native import N,Q,TOKENS,MAX_TOKENS

ARMS=("bi","one_way","dense","local","full")
SEEDS=(0,1,2)
WIDTH=48
L=N+Q
VISITS=4


def packet(root):
    """Entire prefix syntax and signed candidate graph, without oracle truth."""
    x=torch.zeros(N,MAX_TOKENS,dtype=torch.long)
    g=torch.zeros(L,6)
    inc=torch.zeros(L,L)
    endpoints=torch.zeros(Q,2,dtype=torch.long)
    mask=torch.zeros(Q,dtype=torch.bool)
    for i,e in enumerate(root.exprs):
        x[i,:len(e)]=torch.tensor([TOKENS.index(t) for t in e])
        g[i]=torch.tensor([1.,0.,1.,0.,0.,root.budget/6])
    for j,o in enumerate(root.offers):
        k=N+j
        g[k]=torch.tensor([0.,1.,1.,float(o.kind=="anchor"),float(o.sign),root.budget/6])
        inc[k,o.u]=1.;inc[k,o.v]=1.
        endpoints[j]=torch.tensor([o.u,o.v])
        mask[j]=True
    return x,g,inc,endpoints,mask


class SemanticPrior(nn.Module):
    def __init__(self,arm,width=WIDTH):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown applicability arm")
        self.arm=arm
        self.token=nn.Embedding(len(TOKENS),8,padding_idx=0)
        self.syntax=nn.Sequential(nn.Linear(MAX_TOKENS*8,width),nn.GELU(),nn.Linear(width,width))
        self.meta=nn.Linear(6,width)
        if arm in ("bi","one_way"):
            self.down=nn.Linear(width,width,bias=False)
            self.up=nn.Linear(width,width,bias=False)
            self.cell=nn.Sequential(nn.Linear(4*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.norm=nn.LayerNorm(width)
        if arm=="dense":
            self.inc=nn.Linear(2*L,width)
            self.attn=nn.MultiheadAttention(width,4,batch_first=True)
            self.ff=nn.Sequential(nn.Linear(2*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
        if arm=="full":
            self.full=nn.Sequential(nn.Linear(L*width+L*L,2*width),nn.GELU(),
                                    nn.Linear(2*width,N))
        else:
            self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,x,g,inc,depth=VISITS):
        e=self.meta(g)
        e=torch.cat((e[:,:N]+self.syntax(self.token(x).flatten(2)),e[:,N:]),1)
        active=g[...,2:3]
        if self.arm=="local":return self.head(e[:,:N]).squeeze(-1)
        if self.arm=="full":
            return self.full(torch.cat(((e*active).flatten(1),inc.flatten(1)),-1))
        if self.arm=="dense":
            e=e+self.inc(torch.cat((inc,inc.transpose(1,2)),-1))
            h=e;pad=g[...,2]<.5
            for _ in range(depth):
                a,_=self.attn(h,h,h,key_padding_mask=pad,need_weights=False)
                h=self.n1(h+a);h=self.n2(h+self.ff(torch.cat((h,e),-1)))
        else:
            h=e
            for _ in range(depth):
                # Both direction computations execute in both arms. Only
                # node->query is masked in the one-way control. Query->node
                # remains, so the node readout sees candidate metadata.
                d=torch.bmm(inc,self.down(h))*float(self.arm=="bi")
                u=torch.bmm(inc.transpose(1,2),self.up(h))
                h=self.norm(h+self.cell(torch.cat((h,d,u,e),-1)))*active
        return self.head(h[:,:N]).squeeze(-1)


def query_probs(logits,g,endpoints):
    p=logits.sigmoid()
    pu=torch.gather(p,1,endpoints[...,0]);pv=torch.gather(p,1,endpoints[...,1])
    eq=pu*pv+(1-pu)*(1-pv)
    anchor=g[:,N:,3]>.5
    positive=g[:,N:,4]>.0
    base=torch.where(anchor,pu,eq)
    return torch.where(positive,base,1-base).clamp(1e-5,1-1e-5)


def stack(items):return tuple(map(torch.stack,zip(*items,strict=True)))


def weight_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(json.dumps(t.detach().tolist(),separators=(",",":")).encode())
    return h.hexdigest()


def fit(items,labels,seed,arm,steps=200):
    torch.manual_seed(seed);m=SemanticPrior(arm);init=weight_hash(m)
    x,g,i,e,legal=stack(items);y=torch.stack(labels)
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.005)
    rng=torch.Generator().manual_seed(seed+6613)
    roster=torch.randint(len(x),(steps,16),generator=rng)
    first=last=None
    for j,ind in enumerate(roster):
        p=query_probs(m(x[ind],g[ind],i[ind]),g[ind],e[ind])
        loss=F.binary_cross_entropy(p[legal[ind]],y[ind][legal[ind]])
        if not torch.isfinite(loss):raise ValueError("nonfinite native label loss")
        if j==0:first=float(loss.detach())
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        last=float(loss.detach())
    m.eval()
    with torch.no_grad():
        p=query_probs(m(x,g,i),g,e)
        nll=float(F.binary_cross_entropy(p[legal],y[legal]))
        accuracy=float(((p[legal]>=.5)==(y[legal]>=.5)).float().mean())
    return m,{"parameters":sum(t.numel() for t in m.parameters()),"seed":seed,"arm":arm,
              "steps":steps,"initial_sha256":init,"weight_sha256":weight_hash(m),
              "sampler_sha256":hashlib.sha256(json.dumps(roster.tolist()).encode()).hexdigest(),
              "source_nll":nll,"source_query_accuracy":accuracy,"first_loss":first,"last_loss":last}


@torch.no_grad()
def prior(model,root):
    x,g,i,_,_=packet(root)
    z=model(x[None],g[None],i[None])[0,:root.n]
    return z.sigmoid().clamp(.001,.999).tolist()
