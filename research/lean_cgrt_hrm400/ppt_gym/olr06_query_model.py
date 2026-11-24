"""OLR-06B.1: exact-ledger readout and learned candidate query allocation."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .olr06_queries import N,Q,STOP

L=N+Q+1
D=12
ARMS=("bi","component_to_query","query_to_component","dense","direct")
SEEDS=(0,1,2)
WIDTH=48
VISITS=4


def packet(planner,a,s):
    """Public component quotient and query incidence; no outcome oracle.

    Each candidate row exposes exact immediate expected utility gain.
    Both endpoint directions are derived from the same incidence matrix.
    """
    r=planner.root
    groups,index,weights,resolved,coverage=planner.closure(s)
    x=torch.zeros(L,D)
    inc=torch.zeros(L,L)
    mask=torch.zeros(Q+1,dtype=torch.bool)
    remaining=r.budget-planner.spent(a)
    base=planner.stop_value(a,s)
    immediate=planner.values(a,s,1)
    for i,(w,known) in enumerate(zip(weights,resolved)):
        x[i]=torch.tensor([1.,0.,0.,1.,w/r.total,float(known),0.,0.,0.,
                           remaining/6.,coverage/r.total,0.])
    for j in planner.legal(a):
        if j==STOP:
            mask[STOP]=True
            x[-1]=torch.tensor([0.,0.,1.,1.,0.,0.,0.,0.,0.,
                                remaining/6.,coverage/r.total,0.])
            continue
        offer=r.offers[j]
        k=N+j;mask[j]=True
        x[k]=torch.tensor([0.,1.,0.,1.,0.,0.,float(offer.kind=="anchor"),
                           offer.cost/6.,offer.p,remaining/6.,coverage/r.total,
                           immediate[j]-base])
        inc[k,index[offer.u]]=1.
        inc[k,index[offer.v]]=1.
    return x,inc,mask


class IncidenceCell(nn.Module):
    """Same parameters/operations for all three direction masks.

    Component-to-query messages gather endpoint evidence. Reverse messages
    distribute query context to its components. No learned storage updates
    are committed to the exact ledger. Immutable features are reinjected.
    """
    def __init__(self,mode,width=WIDTH):
        super().__init__()
        self.mode=mode
        self.enc=nn.Sequential(nn.Linear(D,width),nn.GELU(),nn.Linear(width,width))
        self.down=nn.Linear(width,width,bias=False)
        self.up=nn.Linear(width,width,bias=False)
        self.cell=nn.Sequential(nn.Linear(4*width,2*width),nn.GELU(),
                                nn.Linear(2*width,width))
        self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,x,inc,*,depth=VISITS):
        e=self.enc(x);h=e
        active=x[...,3:4]
        for _ in range(depth):
            # Both projections and aggregation paths execute in every arm.
            down=torch.bmm(inc,self.down(h))*float(self.mode!="query_to_component")
            up=torch.bmm(inc.transpose(1,2),self.up(h))*float(self.mode!="component_to_query")
            h=self.norm(h+self.cell(torch.cat((h,down,up,e),-1)))*active
        return self.head(h[:,N:]).squeeze(-1)


class DenseCell(nn.Module):
    """Tied dense attention with the same public incidence information."""
    def __init__(self,width=WIDTH):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(D+2*L,width),nn.GELU(),nn.Linear(width,width))
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(2*width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))

    def forward(self,x,inc,*,depth=VISITS):
        e=self.enc(torch.cat((x,inc,inc.transpose(1,2)),-1));h=e
        pad=x[...,3]<.5
        for _ in range(depth):
            update,_=self.attn(h,h,h,key_padding_mask=pad,need_weights=False)
            h=self.norm1(h+update)
            h=self.norm2(h+self.ff(torch.cat((h,e),-1)))
        return self.head(h[:,N:]).squeeze(-1)


class Direct(nn.Module):
    """Full-packet feed-forward control, including every incidence entry."""
    def __init__(self,width=WIDTH):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(L*D+L*L,2*width),nn.GELU(),
                               nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,Q+1))

    def forward(self,x,inc,*,depth=VISITS):
        return self.net(torch.cat((x.flatten(1),inc.flatten(1)),dim=1))


def make(arm):
    if arm=="direct":return Direct()
    if arm=="dense":return DenseCell()
    if arm in ARMS:return IncidenceCell(arm)
    raise ValueError("unknown policy")


def hash_weights(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode())
        h.update(json.dumps(t.detach().cpu().tolist(),separators=(",",":"),
                            allow_nan=False).encode())
    return h.hexdigest()


def macs(arm):
    """Leading linear/attention MACs per padded decision; excludes activations.

    Batch matrix incidence products are implemented densely; count both
    projections/aggregations even when masked. These are analytical counts,
    not observed CPU FLOPs or wall time.
    """
    d=WIDTH
    if arm=="direct":return (L*D+L*L)*2*d+2*d*d+d*(Q+1)
    inp=D if arm!="dense" else D+2*L
    enc=L*(inp*d+d*d)
    if arm=="dense":
        per=L*(4*d*d+6*d*d)+2*L*L*d
    else:
        per=L*(2*d*d+10*d*d)+2*L*L*d
    head=(Q+1)*(d*d+d)
    return enc+VISITS*per+head


def stack_packets(items):
    return tuple(map(torch.stack,zip(*items,strict=True)))


def fit(tensors,*,seed,arm,steps=240,batch=32):
    if arm not in ARMS or not 1<=steps<=240:raise ValueError("bad registered fit")
    x,i,m,y=tensors
    torch.manual_seed(seed)
    model=make(arm);init=hash_weights(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    rng=torch.Generator().manual_seed(seed+6600)
    sample=torch.randint(len(x),(steps,min(batch,len(x))),generator=rng)
    first=last=None
    for step,ids in enumerate(sample):
        pred=model(x[ids],i[ids])
        loss=F.smooth_l1_loss(pred[m[ids]],y[ids][m[ids]],beta=.08)
        if not torch.isfinite(loss):raise ValueError("nonfinite query value fit")
        if step==0:first=float(loss.detach())
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();last=float(loss.detach())
    model.eval()
    with torch.no_grad():
        pred=model(x,i).masked_fill(~m,-torch.inf)
        selected=pred.argmax(1)
        best=y.masked_fill(~m,-torch.inf).max(1).values
        regret=float((best-y[torch.arange(len(y)),selected]).mean())
    return model,{"seed":seed,"arm":arm,"steps":steps,
                  "parameters":sum(p.numel() for p in model.parameters()),
                  "initial_sha256":init,"weights_sha256":hash_weights(model),
                  "sampler_sha256":hashlib.sha256(json.dumps(sample.tolist(),
                                    separators=(",",":")).encode()).hexdigest(),
                  "source_value_regret":regret,"first_loss":first,"last_loss":last,
                  "decision_macs":macs(arm),"attention_visits":0 if arm=="direct" else VISITS}


@torch.inference_mode()
def choose_many(model,packets,batch=128):
    x,inc,legal=packets
    result=[]
    for j in range(0,len(x),batch):
        z=model(x[j:j+batch],inc[j:j+batch])
        if not torch.isfinite(z).all():raise ValueError("nonfinite frozen policy")
        result.extend(z.masked_fill(~legal[j:j+batch],-torch.inf).argmax(1).tolist())
    return result
