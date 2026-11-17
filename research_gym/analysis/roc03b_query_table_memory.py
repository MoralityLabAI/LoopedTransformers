"""ROC-03B query-conditioned A-table memory pilot; source-locked CPU.

Precommitted: configs/roc03b_query_conditioned_table_memory_v1.json.
Uses only training source families 0-3 and calibration families 4-5.
Final query cue is passed to readout explicitly; the carried A-table must be
reconstructed from the recurrent pre-query hidden state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from research_gym.analysis.roc02_tied_train import rows, tensors

SEED=2026111703
UPDATES=1920
BATCH=32
DEPTH=8
WIDTH=128
EVAL_STEPS=(240,480,960,1920)
WEIGHTS={"task_only":0.0,"table_supervised":0.5}
TRAIN_SHA="843de4a066b456129ff32df8906d076e051cd7352f7fd73a07dbb78beb0f4054"
CAL_SHA="a9a8190aee2b4c2111afd8ceaf59422b55b483f7e54182795d986cc8739b818c"

class QueryConditionedTableMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_embed=nn.Linear(79,WIDTH)
        self.block_in=nn.Linear(WIDTH,WIDTH)
        self.block_out=nn.Linear(WIDTH,WIDTH)
        self.table_head=nn.Linear(WIDTH,64)
    def delta(self,h):
        return 0.25*self.block_out(torch.tanh(self.block_in(h)))
    def forward(self,x,mask):
        if x.ndim!=3 or x.shape[-1]!=79 or mask.shape!=x.shape[:2]:
            raise ValueError("expected [batch,event,79] and mask")
        query_mask=x[:,:,3]*mask
        if not torch.all(query_mask.sum(dim=1)==1):
            raise AssertionError("exactly one query event per case")
        query=(x[:,:,71:79]*query_mask[:,:,None]).sum(dim=1)
        if not torch.all(query.sum(dim=1)==1):
            raise AssertionError("invalid one-hot query")
        memory_mask=mask*(1-query_mask)
        h=x.new_zeros(x.shape[0],WIDTH)
        for t in range(x.shape[1]):
            h=h+memory_mask[:,t,None]*(0.25*self.event_embed(x[:,t,:])+self.delta(h))
        for _ in range(DEPTH):
            h=h+self.delta(h)
        table=self.table_head(h).reshape(-1,8,8)
        logits=torch.einsum("bq,bqv->bv",query,table)
        return logits,table

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_data(source_dir):
    a=source_dir/"data"/"train_inputs.jsonl"
    b=source_dir/"data"/"calibration_inputs.jsonl"
    if sha(a)!=TRAIN_SHA or sha(b)!=CAL_SHA:
        raise RuntimeError("source-lock digest mismatch")
    atr=rows(a); bcal=rows(b)
    if len(atr)!=1024 or len(bcal)!=256:
        raise ValueError("registered split cardinality")
    if any(r["family"] not in (0,1,2,3) for r in atr):
        raise ValueError("train family leakage")
    if any(r["family"] not in (4,5) for r in bcal):
        raise ValueError("calibration family leakage")
    return tensors(atr),tensors(bcal)

def targets(x):
    if not torch.all(x[:,0,0]==1) or not torch.all(x[:,0,4]==1):
        raise AssertionError("expected first event to bind source A")
    return x[:,0,7:71].reshape(-1,8,8).argmax(-1)

@torch.no_grad()
def score(model,dataset):
    x,mask,y=dataset
    logits,table=model(x,mask)
    row_target=targets(x)
    row_correct=table.argmax(dim=2)==row_target
    return {
        "accuracy":float((logits.argmax(-1)==y).float().mean()),
        "cross_entropy":float(F.cross_entropy(logits,y)),
        "row_reconstruction_accuracy":float(row_correct.float().mean()),
        "whole_table_exact_accuracy":float(row_correct.all(dim=1).float().mean()),
        "cases":int(len(y))
    }

def fit(source_dir:Path,out_dir:Path):
    torch.set_num_threads(1)
    train,dev=load_data(source_dir)
    torch.manual_seed(SEED)
    base=QueryConditionedTableMemory()
    initial=copy.deepcopy(base.state_dict())
    models={};optimizers={};report=[]
    for name in WEIGHTS:
        m=QueryConditionedTableMemory();m.load_state_dict(initial)
        models[name]=m
        optimizers[name]=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0.)
    initial_equal=all(torch.equal(a,b) for a,b in zip(models["task_only"].state_dict().values(),models["table_supervised"].state_dict().values()))
    if not initial_equal:raise AssertionError("not an identical-start comparison")
    sampler=torch.Generator().manual_seed(SEED+20261116)
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,len(train[2]),(BATCH,),generator=sampler)
        xb,mb,yb=(a[ids] for a in train)
        row_target=targets(xb)
        for name,weight in WEIGHTS.items():
            m=models[name];opt=optimizers[name]
            opt.zero_grad(set_to_none=True)
            out,table=m(xb,mb)
            task_ce=F.cross_entropy(out,yb)
            table_ce=F.cross_entropy(table.reshape(-1,8),row_target.flatten())
            loss=task_ce+weight*table_ce
            if not torch.isfinite(loss):raise ValueError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
            opt.step()
        if step in EVAL_STEPS:
            for name,m in models.items():
                rec={"step":step,"arm":name,"aux_table_weight":WEIGHTS[name],
                     "train":score(m,train),"calibration":score(m,dev)}
                report.append(rec)
                print(json.dumps(rec,sort_keys=True),flush=True)
    out_dir.mkdir(parents=True,exist_ok=True)
    ck_info={}
    for name,m in models.items():
        path=out_dir/f"roc03b_{name}_R8_S{SEED}_step{UPDATES}.pt"
        torch.save({"study":"ROC03B-query-conditioned-A-table-memory-v1",
                    "seed":SEED,"depth":DEPTH,"steps":UPDATES,"arm":name,
                    "source_sha256":{"train":TRAIN_SHA,"calibration":CAL_SHA},
                    "state_dict":m.state_dict(),"optimizer_state_dict":optimizers[name].state_dict()},path)
        ck_info[name]={"checkpoint_sha256":sha(path),"checkpoint_bytes":path.stat().st_size}
    final=next(r for r in report if r["step"]==UPDATES and r["arm"]=="table_supervised")
    gate=final["train"]["accuracy"]>=.9 and final["calibration"]["accuracy"]>=.7
    receipt={"study":"ROC03B-query-conditioned-A-table-memory-v1",
       "registered_config":"configs/roc03b_query_conditioned_table_memory_v1.json",
       "seed":SEED,"depth":DEPTH,"steps":UPDATES,"batch":BATCH,
       "train_sha256":TRAIN_SHA,"calibration_sha256":CAL_SHA,
       "initialization_equal":initial_equal,
       "parameter_count":sum(p.numel() for p in base.parameters()),
       "checkpoints":report,"saved_models":ck_info,"gate":"PASS" if gate else "FAIL",
       "sealed_test_evaluated":False,"scope":"single-seed train/calibration feasibility"}
    (out_dir/"roc03b_R8_S2026111703_metrics.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"gate":receipt["gate"],"final":final,"checkpoints":ck_info},sort_keys=True))
    return receipt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();fit(a.source_dir,a.out_dir)

if __name__=="__main__":main()
