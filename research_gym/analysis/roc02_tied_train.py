"""ROC-02 CPU tied-residual delayed-rule benchmark training (pre-registered).

Records: ROC-01 Part4 and Amendment C, plus
configs/roc02_delayed_rule_source_impl_v1.json.
For each of six new seeds and depths 8/16, exactly 240 AdamW updates are
performed with batch size 8. The sampler and initialization are paired
across depths. Only train labels and calibration labels are ever loaded;
sealed-test input records are not read, evaluated or used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from research_gym.analysis.roc02_delayed_rule_source import (
        FEATURE_DIM, SPLITS, encode_event, symbolic_execute)
except ModuleNotFoundError:
    from roc02_delayed_rule_source import (
        FEATURE_DIM, SPLITS, encode_event, symbolic_execute)

SEEDS=(2026111601,2026111603,2026111607,
       2026111609,2026111613,2026111619)
DEPTHS=(8,16)
UPDATES=240
BATCH=8
WIDTH=128


def rows(path: Path) -> list[dict]:
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        if "label" not in record or record["label"] not in range(8):
            raise ValueError("only registered, labeled training/development cases are supported")
        if symbolic_execute(record["events"])["label"] != record["label"]:
            raise AssertionError("stored label differs from independent symbolic rule interpreter")
    return records


def tensors(records: list[dict]) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if not records:
        raise ValueError("dataset is empty")
    n=max(len(r["events"]) for r in records)
    X=torch.zeros(len(records),n,FEATURE_DIM,dtype=torch.float32)
    mask=torch.zeros(len(records),n,dtype=torch.float32)
    y=torch.tensor([r["label"] for r in records],dtype=torch.long)
    for i,r in enumerate(records):
        for j,event in enumerate(r["events"]):
            X[i,j]=torch.tensor(encode_event(event),dtype=torch.float32)
            mask[i,j]=1.0
    return X,mask,y


class TiedRuleModel(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError("unregistered recurrent thinking depth")
        self.depth=depth
        self.event_embed=nn.Linear(FEATURE_DIM,WIDTH)
        self.block_in=nn.Linear(WIDTH,WIDTH)
        self.block_out=nn.Linear(WIDTH,WIDTH)
        self.readout=nn.Linear(WIDTH,8)

    def delta(self,h:torch.Tensor) -> torch.Tensor:
        return 0.25*self.block_out(torch.tanh(self.block_in(h)))

    def forward(self,x:torch.Tensor,mask:torch.Tensor) -> torch.Tensor:
        if x.ndim!=3 or x.shape[-1]!=FEATURE_DIM or mask.shape!=x.shape[:2]:
            raise ValueError("expected padded [batch,event,79] and [batch,event] mask")
        h=x.new_zeros(x.shape[0],WIDTH)
        for t in range(x.shape[1]):
            update=0.25*self.event_embed(x[:,t,:])+self.delta(h)
            h=h+mask[:,t,None]*update
        for _ in range(self.depth):
            h=h+self.delta(h)
        return self.readout(h)


def evaluate(model:TiedRuleModel,data:tuple[torch.Tensor,torch.Tensor,torch.Tensor]) -> dict:
    x,m,y=data
    with torch.no_grad():
        logits=model(x,m)
        loss=float(F.cross_entropy(logits,y))
        acc=float((logits.argmax(-1)==y).float().mean())
    return {"cross_entropy":loss,"accuracy":acc,"cases":len(y)}


def train(seed:int,depth:int,folder:Path,output:Path,updates:int=UPDATES):
    if seed not in SEEDS or depth not in DEPTHS:
        raise ValueError("outside frozen model cohort")
    if updates!=UPDATES:
        raise ValueError("only preregistered 240-update full fits accepted here")
    torch.set_num_threads(1)
    train_path=folder/'train_inputs.jsonl'
    dev_path=folder/'calibration_inputs.jsonl'
    train_rows,dev_rows=rows(train_path),rows(dev_path)
    if len(train_rows)!=SPLITS['train']['n'] or len(dev_rows)!=SPLITS['calibration']['n']:
        raise RuntimeError("source case count mismatch")
    train_data=tensors(train_rows)
    dev_data=tensors(dev_rows)
    torch.manual_seed(seed)
    model=TiedRuleModel(depth)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    sampler=torch.Generator().manual_seed(seed+20261116)
    step_losses=[]
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,len(train_rows),(BATCH,),generator=sampler)
        xb,mb,yb=(v[ids] for v in train_data)
        optimizer.zero_grad(set_to_none=True)
        logits=model(xb,mb)
        loss=F.cross_entropy(logits,yb)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite model training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step in (1,16,64,128,240):
            step_losses.append({"step":step,"loss":float(loss.detach())})
    checkpoint_dir=output/'checkpoints'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    ck=checkpoint_dir/f'roc02_R{depth}_S{seed}_step{UPDATES}.pt'
    torch.save({
        'study':'ROC-02-delayed-rule-source-implementation-v1',
        'seed':seed,'depth':depth,'steps':UPDATES,
        'state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'source_sha256':{
            'train':hashlib.sha256(train_path.read_bytes()).hexdigest(),
            'calibration':hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        },
    },ck)
    result={
        'study':'ROC-02-delayed-rule-source-implementation-v1',
        'source_status':'calibration-only; sealed test not opened',
        'seed':seed,'depth':depth,'steps':UPDATES,
        'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
        'train':evaluate(model,train_data),
        'calibration':evaluate(model,dev_data),
        'step_losses':step_losses,
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/f'roc02_R{depth}_S{seed}.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--depth',type=int,choices=DEPTHS,required=True)
    a=p.parse_args()
    print(json.dumps(train(a.seed,a.depth,a.source_dir,a.output_dir),sort_keys=True))


if __name__=='__main__':
    main()
