"""ROC-03A: source-locked stronger-training delayed-rule pilot (CPU).

Pre-registered at configs/roc03a_source_locked_training_v1.json
before any new ROC03A model training. Reads only training/calibration source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research_gym.analysis.roc02_tied_train import TiedRuleModel, rows, tensors, evaluate

SEED=2026111701
DEPTH=8
WIDTH=128
UPDATES=1920
BATCH=32
LR=0.001
EVAL_STEPS=(240,480,960,1920)
TRAIN_SHA="843de4a066b456129ff32df8906d076e051cd7352f7fd73a07dbb78beb0f4054"
DEV_SHA="a9a8190aee2b4c2111afd8ceaf59422b55b483f7e54182795d986cc8739b818c"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def data(source_dir: Path):
    train=source_dir/"data"/"train_inputs.jsonl"
    dev=source_dir/"data"/"calibration_inputs.jsonl"
    if sha(train)!=TRAIN_SHA or sha(dev)!=DEV_SHA:
        raise RuntimeError("ROC03A source or split hash mismatch")
    a,b=rows(train),rows(dev)
    if len(a)!=1024 or len(b)!=256:
        raise RuntimeError("wrong train/calibration source count")
    if any(r["family"] not in (0,1,2,3) for r in a):
        raise AssertionError("train family leakage")
    if any(r["family"] not in (4,5) for r in b):
        raise AssertionError("calibration family leakage")
    return tensors(a),tensors(b)

def train(source_dir:Path,outdir:Path):
    torch.set_num_threads(1)
    tr,dev=data(source_dir)
    torch.manual_seed(SEED)
    model=TiedRuleModel(DEPTH)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
    sampler=torch.Generator().manual_seed(SEED+20261116)
    checkpoints=[]
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,len(tr[2]),(BATCH,),generator=sampler)
        xb,mb,yb=(v[ids] for v in tr)
        opt.zero_grad(set_to_none=True)
        logits=model(xb,mb)
        loss=torch.nn.functional.cross_entropy(logits,yb)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if step in EVAL_STEPS:
            train_scores=evaluate(model,tr)
            cal_scores=evaluate(model,dev)
            row={"step":step,"batch":BATCH,"training_loss":float(loss.detach()),
                 "train":train_scores,"calibration":cal_scores}
            checkpoints.append(row)
            print(json.dumps(row,sort_keys=True),flush=True)
    final=checkpoints[-1]
    pass_gate=(final["train"]["accuracy"]>=.90 and
               final["calibration"]["accuracy"]>=.70)
    outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f"roc03a_R{DEPTH}_S{SEED}_step{UPDATES}.pt"
    torch.save({"study":"ROC03A-competence-scaling-v1","seed":SEED,
                "depth":DEPTH,"steps":UPDATES,"batch":BATCH,
                "state_dict":model.state_dict(),
                "optimizer_state_dict":opt.state_dict(),
                "sampler_state":sampler.get_state(),
                "source_sha256":{"train":TRAIN_SHA,"calibration":DEV_SHA}},path)
    receipt={"study":"ROC03A-competence-scaling-v1","seed":SEED,"depth":DEPTH,
             "model_updates":UPDATES,"batch_size":BATCH,
             "train_sha256":TRAIN_SHA,"calibration_sha256":DEV_SHA,
             "evaluation_steps":list(EVAL_STEPS),"checkpoints":checkpoints,
             "checkpoint_sha256":sha(path),"checkpoint_bytes":path.stat().st_size,
             "gate":("PASS" if pass_gate else "FAIL"),
             "sealed_source_families":[6,7],"sealed_test_evaluated":False,
             "inference_scope":"single-seed calibration feasibility, no prospective result"}
    (outdir/f"roc03a_R{DEPTH}_S{SEED}_metrics.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"gate":receipt["gate"],"final":final,"checkpoint_sha256":receipt["checkpoint_sha256"]},sort_keys=True),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    train(args.source_dir,args.out_dir)

if __name__=="__main__":
    main()
