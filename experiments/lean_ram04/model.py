"""Small recurrent scorers with pending-goal and goal-count supervision."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
import torch
from torch import nn
import ram04_core as c

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
    contexts=[];goals=[];pending=[];actions=[];operands=[];families=[];counts=[]
    for row in rows:
        state=row['before'];current=state['goals']
        if not current or any(s.count('⊢')!=1 for s in current):raise ValueError('invalid visible goal list')
        context,goal=current[0].split('⊢',1)
        contexts.append(token_ids(context,96,True));goals.append(token_ids(goal,96))
        pending.append(token_ids(' || '.join(current[1:]),128))
        actions.append(token_ids(row['script'],96));families.append(c.FAMILIES.index(row['family']))
        references=[v for v in c.visible_locals(state) if re.search(r'(?<!\w)'+re.escape(v['name'])+r'(?!\w)',row['script'])]
        operand=' || '.join(v['type'] for v in references);operands.append(token_ids(operand,128))
        a=set(re.findall(r'\w+',operand));b=set(re.findall(r'\w+',goal))
        overlap=len(a&b)/max(1,len(a|b))
        counts.append([min(8,len(current))/8.,float(len(current)>1),overlap,min(4,len(references))/4.,
                       sum(v['prop'] for v in references)/max(1,len(references))])
    return dict(context=torch.tensor(contexts,dtype=torch.long),goal=torch.tensor(goals,dtype=torch.long),
      pending=torch.tensor(pending,dtype=torch.long),action=torch.tensor(actions,dtype=torch.long),
      operand=torch.tensor(operands,dtype=torch.long),
      family=torch.tensor(families,dtype=torch.long),counts=torch.tensor(counts,dtype=torch.float32))

class Policy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.width
        self.embedding=nn.Embedding(2048,d,padding_idx=0);self.family=nn.Embedding(len(c.FAMILIES),d)
        positions=torch.arange(128,dtype=torch.float32).unsqueeze(1)+1
        frequencies=torch.exp(-torch.arange(d,dtype=torch.float32)*9.210340372/max(1,d-1))
        self.register_buffer('position_gate',1+.5*torch.sin(positions*frequencies),persistent=False)
        self.fuse=nn.Sequential(nn.Linear(6*d+5,2*d),nn.GELU(),nn.Linear(2*d,d),nn.GELU())
        if config.loops:
            self.cell=nn.Sequential(nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d));self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,4);self.delta=nn.Linear(d,1)
    def mean(self,ids):
        mask=(ids!=0).unsqueeze(-1)
        return (self.embedding(ids)*mask*self.position_gate[:ids.shape[1]]).sum(1)/mask.sum(1).clamp_min(1)
    def forward(self,batch):
        fixed=self.fuse(torch.cat([self.mean(batch[k]) for k in ('context','goal','pending','action','operand')]+
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
