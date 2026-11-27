"""Addressed vs dense weight-tied receipt memory, with explicit-history controls."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .olr_metta05_native import N

# Only these fields enter the network: event kind, address, tested branch,
# observed native verdict. No theorem bytes, root ID or private truth.
KINDS=("receipt","veto","delay","query")
D=len(KINDS)+N+2
ARMS=("addressed","dense","explicit_mlp")
SEEDS=(17,29,43)


def event(kind,addr=None,branch=None,ok=None):
    if kind not in KINDS:raise ValueError("unregistered event type")
    x=[float(kind==k) for k in KINDS]+[0.]*N+[0.,0.]
    if kind in ("receipt","query"):
        if type(addr) is not int or not 0<=addr<N:
            raise ValueError("missing public receipt/query address")
        x[len(KINDS)+addr]=1.
    elif addr is not None:
        raise ValueError("veto/delay events must be payload free")
    if kind=="receipt":
        if branch not in (0,1) or type(ok) is not bool:
            raise ValueError("invalid observed probe bit")
        x[-2]=float(2*branch-1);x[-1]=1. if ok else -1.
    elif branch is not None or ok is not None:
        raise ValueError("nonreceipt event may not leak a verdict")
    return x


def packets(receipts,recall):
    if set(r["addr"] for r in receipts)!=set(recall) or len(receipts)!=len(recall):
        raise ValueError("incomplete or duplicate receipt history")
    history=[]
    for r in receipts:
        history.extend((event("receipt",r["addr"],r["branch"],r["ok"]),event("veto")))
    history.extend((event("delay"),event("delay")))
    query=[event("query",a) for a in recall]
    # This is an explicit representation of the SAME actual admitted history,
    # with no product/XOR or winner computed for the learned MLP.
    byaddr={r["addr"]:r for r in receipts}
    table=[[float(2*byaddr[a]["branch"]-1),1. if byaddr[a]["ok"] else -1.,1.]
           for a in recall]
    return (torch.tensor(history,dtype=torch.float32),
            torch.tensor(query,dtype=torch.float32),
            torch.tensor(table,dtype=torch.float32))


def flip_receipt(receipts,addr):
    if sum(r["addr"]==addr for r in receipts)!=1:
        raise ValueError("targeted intervention needs exactly one receipt")
    return [{**r,"ok":not r["ok"]} if r["addr"]==addr else dict(r) for r in receipts]


class ReceiptLedger(nn.Module):
    """One tied self-attention/MLP cell, with matched dense/addressed writes.

    Both variants compute proposals for every slot on every event. The sole
    distinction is whether admitted receipts write to all slots or just the
    publicly addressed slot. Both have identical persistent state size and
    parameterization; both read the publicly queried slot after the read cell.
    """
    def __init__(self,width=24,mode="addressed"):
        super().__init__()
        if mode not in ("addressed","dense") or width%4:
            raise ValueError("unsupported write mask or attention width")
        self.width,self.mode=width,mode
        self.embed=nn.Linear(D,width)
        self.position=nn.Linear(N,width,bias=False)
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.norm1=nn.LayerNorm(width)
        self.norm2=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))
        self.register_buffer("addresses",torch.eye(N),persistent=False)

    def propose(self,memory,token):
        pos=self.position(self.addresses).unsqueeze(0)
        h=torch.cat((self.embed(token).unsqueeze(1),memory+pos),dim=1)
        a,_=self.attn(h,h,h,need_weights=False)
        h=self.norm1(h+.5*a)
        h=self.norm2(h+.5*self.ff(h))
        return h[:,1:,:]

    def forward(self,history,queries,*,erase=False):
        if (history.ndim!=3 or queries.ndim!=3 or history.shape[0]!=queries.shape[0]
                or history.shape[-1]!=D or queries.shape[-1]!=D):
            raise ValueError("history/query tensor shape changed")
        memory=history.new_zeros((len(history),N,self.width))
        for i in range(history.shape[1]):
            token=history[:,i]
            proposed=self.propose(memory,token)
            address=token[:,len(KINDS):len(KINDS)+N].unsqueeze(-1)
            write=token[:,0,None,None]>.5
            route=(address>.5) if self.mode=="addressed" else torch.ones_like(address,dtype=torch.bool)
            memory=torch.where(write & route,proposed,memory)
        if erase:
            memory=torch.zeros_like(memory)
        output=[]
        for i in range(queries.shape[1]):
            token=queries[:,i]
            proposed=self.propose(memory,token)
            address=token[:,len(KINDS):len(KINDS)+N].unsqueeze(-1)
            output.append(self.head((proposed*address).sum(dim=1)))
        return torch.stack(output,dim=1)


class ExplicitMap(nn.Module):
    """Same receipt information in a transparent addressed table, no latent carry."""
    def __init__(self,width=24):
        super().__init__()
        self.read=nn.Sequential(nn.Linear(3,width),nn.GELU(),
                                nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))
    def forward(self,table):return self.read(table)


def weight_hash(model):
    h=hashlib.sha256()
    for key,t in model.state_dict().items():
        h.update(key.encode());h.update(json.dumps(t.detach().cpu().tolist(),
                       separators=(",",":"),allow_nan=False).encode())
    return h.hexdigest()


def make(arm):
    return ExplicitMap() if arm=="explicit_mlp" else ReceiptLedger(mode=arm)


def fit(rows,*,seed,arm,steps=160,batch_size=16):
    if arm not in ARMS or not rows or not 1<=steps<=160:
        raise ValueError("invalid frozen source fit")
    torch.manual_seed(seed);model=make(arm)
    init=weight_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    encoded=[packets(r["receipts"],r["recall_order"]) for r in rows]
    histories,queries,tables=map(torch.stack,zip(*encoded,strict=True))
    y=torch.tensor([r["labels"] for r in rows],dtype=torch.long)
    rng=torch.Generator().manual_seed(seed+5800)
    ids=torch.randint(len(rows),(steps,min(batch_size,len(rows))),generator=rng)
    first=last=None
    model.train()
    for j,ind in enumerate(ids):
        scores=(model(tables[ind]) if arm=="explicit_mlp" else model(histories[ind],queries[ind]))
        loss=F.cross_entropy(scores.reshape(-1,2),y[ind].reshape(-1))
        if not torch.isfinite(loss):raise ValueError("nonfinite source memory loss")
        if j==0:first=float(loss.detach())
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();last=float(loss.detach())
    model.eval()
    with torch.no_grad():
        z=model(tables) if arm=="explicit_mlp" else model(histories,queries)
        accuracy=float((z.argmax(-1)==y).float().mean())
    return model,{"seed":seed,"arm":arm,"parameters":sum(p.numel() for p in model.parameters()),
        "initial_sha256":init,"weights_sha256":weight_hash(model),"steps":steps,
        "source_worlds":len(rows),"first_loss":first,"last_loss":last,
        "source_address_accuracy":accuracy,"batch_size":batch_size,
        "sampler_sha256":hashlib.sha256(json.dumps(ids.tolist(),separators=(",",":")).encode()).hexdigest()}


@torch.no_grad()
def choose(model,arm,receipts,recall,*,erase=False):
    h,q,m=packets(receipts,recall)
    scores=(model(m[None]) if arm=="explicit_mlp" else
            model(h[None],q[None],erase=erase))
    return scores[0].argmax(-1).tolist(),(0 if arm=="explicit_mlp" else len(h)+len(q))
