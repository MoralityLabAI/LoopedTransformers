"""OLR-METTA-03 learned verifier-budget policy and same-input controls.

The policy receives six MeTTa-curated claims (Bx6x6) and predicts expected
utility of skip-r0, skip-r1, and paid-probe. Real native Lean responses alone
define training returns. No theorem numerals or gold winners are in x.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .metta_budget_bridge import ACTIONS


class EvidenceLoop(nn.Module):
    """One tied physical attention/MLP cell with optional 1 or 3 visits."""

    def __init__(self,width:int=24):
        super().__init__()
        self.width=width
        self.embed=nn.Linear(6,width)
        self.role=nn.Parameter(torch.zeros(1,7,width))
        self.query=nn.Parameter(torch.zeros(1,1,width))
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),
                              nn.Linear(2*width,width))
        self.norm1=nn.LayerNorm(width)
        self.norm2=nn.LayerNorm(width)
        self.read=nn.Linear(width,3)

    def forward(self,x:torch.Tensor,*,depth:int=3)->torch.Tensor:
        if x.ndim!=3 or x.shape[1:]!=(6,6) or depth not in (1,3):
            raise ValueError("six public claims and depth 1/3 required")
        h=torch.cat((self.query.expand(x.shape[0],-1,-1),self.embed(x)),dim=1)
        h=h+self.role
        for _ in range(depth):
            attended,_=self.attn(h,h,h,need_weights=False)
            h=self.norm1(h+.25*attended)
            h=self.norm2(h+.25*self.ff(h))
        return self.read(h[:,0])


class EvidenceDirect(nn.Module):
    """Direct full-information MLP using exactly the same 36 public numbers."""

    def __init__(self,width:int=24):
        super().__init__()
        self.read=nn.Sequential(nn.Linear(36,4*width),nn.GELU(),
                                nn.Linear(4*width,width),nn.GELU(),
                                nn.Linear(width,3))

    def forward(self,x:torch.Tensor,*,depth:int=3)->torch.Tensor:
        if x.ndim!=3 or x.shape[1:]!=(6,6):
            raise ValueError("direct control expects the identical public packet")
        return self.read(x.reshape(len(x),-1))


def fit(source:list[dict], *, seed:int, arm:str, steps:int=80,
        batch_size:int=24, width:int=24):
    """Fit native source action values using a source-only identical sampler."""
    if arm not in ("loop3","loop1","direct") or not 1<=steps<=500:
        raise ValueError("unsupported bounded OLR evidence arm")
    if not source or len(source)>512:
        raise ValueError("source-only native reward rows required")
    torch.manual_seed(seed)
    model=EvidenceDirect(width) if arm=="direct" else EvidenceLoop(width)
    init={
        key:val.detach().clone()
        for key,val in model.state_dict().items()
    }
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    x=torch.tensor([r["model_features"] for r in source],dtype=torch.float32)
    target=torch.tensor([r["native_action_utilities"] for r in source],dtype=torch.float32)
    if x.shape!=(len(source),6,6) or target.shape!=(len(source),len(ACTIONS)):
        raise ValueError("source observation/label shape changed")
    sampler=torch.Generator().manual_seed(seed+9000)
    indexes=torch.randint(len(x),(steps,min(batch_size,len(x))),generator=sampler)
    d=1 if arm=="loop1" else 3
    first=last=None
    for i,which in enumerate(indexes):
        pred=model(x[which],depth=d)
        loss=F.smooth_l1_loss(pred,target[which],beta=.2)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite native value-regression loss")
        if i==0:first=float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        last=float(loss.detach())
    model.eval()
    with torch.no_grad():
        predicted=model(x,depth=d)
        source_choices=predicted.argmax(-1)
        source_truth=target.argmax(-1)
        accuracy=int((source_choices==source_truth).sum())
        average_source_utility=float(target[torch.arange(len(x)),source_choices].mean())
    params=sum(p.numel() for p in model.parameters())
    return model,{
        "seed":seed,"arm":arm,"steps":steps,"depth":d,
        "parameters":params,
        "first_loss":first,"last_loss":last,
        "source_best_action_match":accuracy,
        "source_worlds":len(x),
        "source_policy_utility":average_source_utility,
        "sampler_shape":list(indexes.shape),
    }


@torch.no_grad()
def select(model:nn.Module,features:list[list[float]],*,arm:str)->str:
    if arm not in ("loop3","loop1","direct"):
        raise ValueError("invalid policy arm")
    x=torch.tensor([features],dtype=torch.float32)
    d=1 if arm=="loop1" else 3
    scores=model(x,depth=d)
    return ACTIONS[int(scores.argmax().item())]


def majority_action(features:list[list[float]],threshold:float)->str:
    """Train-calibrated source-only baseline; requires no private truth label."""
    if len(features)!=6 or not 0<=threshold<=8:
        raise ValueError("malformed evidence majority controls")
    score=sum(row[0]*row[1]*3. for row in features)
    if abs(score)<threshold:
        return "probe_r0"
    return "skip_r0" if score>=0 else "skip_r1"


def source_threshold(source:list[dict])->dict:
    """Choose analytic majority confidence cutoff using only native TRAIN rows."""
    if not source:
        raise ValueError("empty threshold fit")
    grid=[0.,1.,2.,3.,4.,5.,6.,7.,8.]
    values={}
    for threshold in grid:
        score=sum(dict(zip(ACTIONS,row["native_action_utilities"],strict=True))[
            majority_action(row["model_features"],threshold)]
            for row in source)/len(source)
        values[str(threshold)]=round(score,8)
    best=max(grid,key=lambda th:(values[str(th)],th))
    return {"threshold":best,"source_utility":values[str(best)],
            "source_grid":values}
