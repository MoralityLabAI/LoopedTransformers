"""Small recurrent scorers with pending-goal and goal-count supervision."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
import torch
from torch import nn
import ram05_core as c

@dataclass(frozen=True)
class Config:
    name:str
    width:int
    loops:int

def token_ids(text,length,tail=False):
    tokens=re.findall(r"[\w']+|[^\w\s]",text,re.UNICODE)
    tokens=tokens[-length:] if tail else tokens[:length]
    values=[1+int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'big')%2047 for s in tokens]
    return values+[0]*(length-len(values))

def encode(rows):
    contexts=[];goals=[];pending=[];actions=[];families=[];counts=[]
    for row in rows:
        state=row['before'];current=state['goals']
        if not current or any(s.count('⊢')!=1 for s in current):raise ValueError('invalid visible goal list')
        context,goal=current[0].split('⊢',1)
        contexts.append(token_ids(context,96,True));goals.append(token_ids(goal,96))
        pending.append(token_ids(' || '.join(g.split('⊢',1)[1] for g in current[1:]),128))
        actions.append(token_ids(row['script'],96));families.append(c.FAMILIES.index(row['family']))
        counts.append([min(8,len(current))/8.,float(len(current)>1)])
    return dict(context=torch.tensor(contexts,dtype=torch.long),goal=torch.tensor(goals,dtype=torch.long),
      pending=torch.tensor(pending,dtype=torch.long),action=torch.tensor(actions,dtype=torch.long),
      family=torch.tensor(families,dtype=torch.long),counts=torch.tensor(counts,dtype=torch.float32))

class Policy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.width
        self.embedding=nn.Embedding(2048,d,padding_idx=0);self.family=nn.Embedding(len(c.FAMILIES),d)
        self.fuse=nn.Sequential(nn.Linear(5*d+2,2*d),nn.GELU(),nn.Linear(2*d,d),nn.GELU())
        if config.loops:
            self.cell=nn.Sequential(nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d));self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,4);self.delta=nn.Linear(d,1)
    def mean(self,ids):
        mask=(ids!=0).unsqueeze(-1)
        return (self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp_min(1)
    def forward(self,batch):
        fixed=self.fuse(torch.cat([self.mean(batch[k]) for k in ('context','goal','pending','action')]+
                                 [self.family(batch['family']),batch['counts']],-1))
        z=fixed
        for _ in range(self.config.loops):z=self.norm(z+self.cell(torch.cat([z,fixed],-1)))
        return self.head(z),self.delta(z).squeeze(-1)
    @torch.no_grad()
    def rank(self,state,candidates):
        if not candidates:return []
        rows=[dict(before=state,family=f,script=s) for f,s in candidates]
        logits=[];deltas=[];device=next(self.parameters()).device
        for i in range(0,len(rows),16):
            a,b=self({k:v.to(device) for k,v in encode(rows[i:i+16]).items()});logits.append(a);deltas.append(b)
        p=torch.cat(logits).softmax(-1);delta=torch.cat(deltas).clamp(-3,3)
        score=3*p[:,3]+1.5*p[:,1]+.3*p[:,2]-.6*p[:,0]+.4*delta
        return [candidates[i] for i in sorted(range(len(rows)),key=lambda i:(-float(score[i]),i))]
