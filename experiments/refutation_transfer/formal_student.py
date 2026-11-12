"""FC-02: exact-certificate transfer using the real LR-01 TypedLoop cell.

Train a shared recurrent proposition/rule graph policy to approximate least
Horn closure. Every training target is supplied by a separate, exact
least-fixed-point generator and independently checked by exhaustive truth
tables. Validation and OOD-test families are absent from gradient updates.

The student proposes a Boolean assignment for a false entailment; the
independent checker *verifies*, without repairing, that the assignment
satisfies all premises while falsifying the goal. The model does not call an
oracle during inference and does not use Lean to fill in missing conclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lean_reasoner"))
from train import TypedLoop

MAX_A = 12
MAX_R = 20
ATOM_F = 5
RULE_F = 4
WIDTH = 24
GENERATOR_SHA = "8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f"


def load_frozen_curriculum(bench_root: Path):
    root = bench_root.resolve()
    sys.path.insert(0, str(root))
    from research.refutation_bench.formal_curriculum import generator as gen
    from research.refutation_bench.certificates import verify_countermodel
    data = gen.generate()
    gen.audit(data)
    digest = hashlib.sha256(b"".join(gen.canonical_bytes(t) + b"\n" for t in data)).hexdigest()
    if digest != GENERATOR_SHA:
        raise RuntimeError("curriculum changed from reviewed manifest: " + digest)
    train = [t for t in data if t["split"] == "train"]
    dev = [t for t in data if t["split"] == "dev"]
    test = [t for t in data if t["split"] == "test"]
    assert (len(train),len(dev),len(test)) == (1152, 192, 192)
    assert {t["family_id"] for t in train}.isdisjoint({t["family_id"] for t in dev + test})
    assert {t["family_id"] for t in dev}.isdisjoint({t["family_id"] for t in test})
    return train, dev, test, digest, verify_countermodel


def encode(tasks: list[dict]) -> dict:
    """Read only graph syntax, atoms, goal and explicit premises.

    Split, family, label, certificate, reachable, fingerprint and proof traces
    are never used to construct the graph's input features.
    """
    count = len(tasks)
    a = torch.zeros(count, MAX_A, ATOM_F)
    r = torch.zeros(count, MAX_R, RULE_F)
    prem = torch.zeros(count, MAX_R, MAX_A)
    head = torch.zeros_like(prem)
    atom_mask = torch.zeros(count, MAX_A, dtype=torch.bool)
    atom_targets = torch.zeros(count, MAX_A)
    goal_idx = torch.zeros(count, dtype=torch.long)
    goal_truth = torch.zeros(count)
    actual_count = torch.zeros(count, dtype=torch.long)
    for i, task in enumerate(tasks):
        atoms, facts, rules, goal = task["atoms"], task["facts"], task["rules"], task["goal"]
        n = len(atoms)
        if n > MAX_A or len(rules) > MAX_R or n == 0:
            raise ValueError("task exceeds graph size cap: " + task["case_id"])
        idx = {name:j for j,name in enumerate(atoms)}
        actual_count[i] = n
        atom_mask[i,:n] = True
        goal_idx[i] = idx[goal]
        for atom in atoms:
            j = idx[atom]
            a[i,j,0] = 1.0 if atom in facts else 0.0
            a[i,j,1] = 1.0 if atom == goal else 0.0
            a[i,j,2] = 1.0
        for j,rule in enumerate(rules):
            ps = rule["premises"]
            if not ps:
                raise ValueError("empty antecedent in student curriculum")
            r[i,j,:] = torch.tensor([len(ps) / 4.0,
                                     float(rule["head"] == goal),
                                     float(set(ps).issubset(facts)), 1.0])
            head[i,j,idx[rule["head"]]] = 1.0
            for p in ps:
                prem[i,j,idx[p]] = 1.0
                a[i,idx[p],3] += 1.0
            a[i,idx[rule["head"]],4] += 1.0
        a[i,:,3:] /= MAX_R
        # Label-only targets. These tensors have no path into the forward pass.
        known = set(task["reachable"])
        for name,j in idx.items():
            atom_targets[i,j] = float(name in known)
        goal_truth[i] = float(task["label"] == "established")
    return {"a":a,"r":r,"prem":prem,"head":head,"mask":atom_mask,
            "goal_idx":goal_idx,"goal_truth":goal_truth,
            "atom_targets":atom_targets,"counts":actual_count}


def batch_select(full: dict, idx: torch.Tensor) -> dict:
    return {k: tensor[idx] for k,tensor in full.items()}


class LoopReach(nn.Module):
    """Exact LR-01 recurrent TypedLoop, trained on source-domain-proof graphs."""

    def __init__(self, loops: int, width: int = WIDTH):
        super().__init__()
        self.loops = loops
        self.atom_in = nn.Linear(ATOM_F, width)
        self.rule_in = nn.Linear(RULE_F, width)
        self.cell = TypedLoop(width)
        self.atom_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, b: dict):
        a = self.atom_in(b["a"])
        r = self.rule_in(b["r"])
        intermediate = []
        for _ in range(self.loops):
            a,r = self.cell(a,r,b["prem"],b["head"])
            intermediate.append(self.atom_out(a).squeeze(-1))
        return intermediate


def objective(logits_list: list[torch.Tensor], batch: dict) -> torch.Tensor:
    targets = batch["atom_targets"]
    mask = batch["mask"].float()
    goal = batch["goal_truth"]
    idx = batch["goal_idx"]
    final = logits_list[-1]
    atom_losses = F.binary_cross_entropy_with_logits(final, targets, reduction="none")
    atom_loss = (atom_losses * mask).sum() / mask.sum().clamp_min(1)
    goal_logits = final.gather(1,idx[:,None]).squeeze(1)
    goal_loss = F.binary_cross_entropy_with_logits(goal_logits,goal)
    # Auxiliary final-fixpoint supervision also constrains hidden passes.
    aux = torch.stack([
        (F.binary_cross_entropy_with_logits(l,targets,reduction="none") * mask).sum()
        / mask.sum().clamp_min(1)
        for l in logits_list[:-1]
    ]).mean() if len(logits_list) > 1 else atom_loss * 0
    return atom_loss + 2.0*goal_loss + 0.15*aux


def train_model(train: list[dict], *,loops: int,seed:int,steps:int,batch_size:int=48):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = LoopReach(loops)
    full = encode(train)
    optim = torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.01)
    g = torch.Generator().manual_seed(seed+6154)
    start = time.perf_counter()
    losses = []
    for step in range(steps):
        idx = torch.randint(len(train),(batch_size,),generator=g)
        b = batch_select(full,idx)
        model.train()
        optim.zero_grad(set_to_none=True)
        outputs = model(b)
        loss = objective(outputs,b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
        if step in (0, steps//4, steps//2, (3*steps)//4, steps-1):
            losses.append([step+1,round(float(loss.detach()),6)])
    model.eval()
    return model, {"loops":loops,"seed":seed,"steps":steps,
                   "model_parameters":sum(p.numel() for p in model.parameters()),
                   "train_samples":len(train),
                   "losses":losses,
                   "train_wall_seconds":round(time.perf_counter()-start,3)}


@torch.no_grad()
def infer(model:LoopReach,tasks:list[dict],check_countermodel,batch_size:int=48):
    model.eval()
    full = encode(tasks)
    output=[]
    for start in range(0,len(tasks),batch_size):
        rows=tasks[start:start+batch_size]
        b=batch_select(full,torch.arange(start,start+len(rows)))
        logits=model(b)[-1]
        prob=torch.sigmoid(logits)
        for k,task in enumerate(rows):
            p=prob[k,:len(task["atoms"])]
            values={name: bool(p[j]>=0.5) for j,name in enumerate(task["atoms"])}
            goal_pred=values[task["goal"]]
            verified=False
            if not goal_pred:
                f=task["formal"]
                verdict=check_countermodel(f["premises"],f["conclusion"],values)
                verified=verdict["verified"]
            output.append({
                "case_id":task["case_id"],
                "predicted": "established" if goal_pred else "refuted",
                "confidence": round(float(p[task["atoms"].index(task["goal"])]),6),
                "countermodel_verified":verified,
                "assignment":values if not goal_pred else None,
            })
    return output


def score(tasks:list[dict],predictions:list[dict]):
    if len(tasks) != len(predictions):
        raise ValueError("missing predictions")
    cm=Counter()
    right=0
    true_neg=0
    false_refute=0
    verifier_success=0
    certified_negative=0
    for t,p in zip(tasks,predictions):
        assert t["case_id"]==p["case_id"]
        cm[(t["label"],p["predicted"])]+=1
        right+=p["predicted"]==t["label"]
        if t["label"]=="established" and p["predicted"]=="refuted":
            false_refute+=1
        if t["label"]=="refuted":
            true_neg+=1
            if p["predicted"]=="refuted" and p["countermodel_verified"]:
                certified_negative+=1
        if p["predicted"]=="refuted" and p["countermodel_verified"]:
            verifier_success+=1
    positives=sum(t["label"]=="established" for t in tasks)
    refuted_predictions=sum(p["predicted"]=="refuted" for p in predictions)
    # F1 for the two exhaustive verdict classes.
    tp=cm[("refuted","refuted")]
    fp=cm[("established","refuted")]
    fn=cm[("refuted","established")]
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return {
        "n":len(tasks),"accuracy":round(right/len(tasks),6),
        "f1_refuted":round(f1,6),
        "false_refutation_rate":round(false_refute/positives,6),
        "refuted_prediction_count":refuted_predictions,
        "verified_proposed_countermodels":verifier_success,
        "certified_recall_on_invalid":round(certified_negative/true_neg,6),
        "unverified_negative_predictions":refuted_predictions-verifier_success,
        "confusion":{a+"->"+b:int(cm[(a,b)]) for a in ("established","refuted")
                     for b in ("established","refuted")},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--loops",type=int,nargs="+",default=[1,4])
    parser.add_argument("--seeds",type=int,nargs="+",default=[17,29])
    parser.add_argument("--steps",type=int,default=350)
    parser.add_argument("--batch-size",type=int,default=48)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    train,dev,test,curriculum_hash,verify=load_frozen_curriculum(args.benchmark_root)
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    runs=[]
    for loops in args.loops:
        for seed in args.seeds:
            model,receipt=train_model(train,loops=loops,seed=seed,
                                      steps=args.steps,batch_size=args.batch_size)
            tag=f"horn_lr01_l{loops}_s{seed}"
            # All arms and hyperparameters were fixed before this evaluation.
            dev_pred=infer(model,dev,verify)
            test_pred=infer(model,test,verify)
            train_pred=infer(model,train[:192],verify)
            weights=args.out/(tag+".pt")
            torch.save(model.state_dict(),weights)
            receipt.update({
                "tag":tag,"weights_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
                "train_192":score(train[:192],train_pred),
                "dev":score(dev,dev_pred),
                "test":score(test,test_pred)
            })
            for split,pred in (("dev",dev_pred),("test",test_pred)):
                p=args.out/(tag+"_"+split+"_predictions.jsonl")
                p.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in pred))
            print("MODEL",json.dumps({
                "tag":tag,"params":receipt["model_parameters"],
                "train":receipt["train_192"],"dev":receipt["dev"],"test":receipt["test"]
            },sort_keys=True),flush=True)
            runs.append(receipt)
    report={
        "title":"FC-02 LR-01 trained reachability and certificate proposal",
        "source":"LoopedTransformers LR-01 train.py:TypedLoop",
        "pretrained_checkpoint_loaded":False,
        "curriculum_hash":curriculum_hash,
        "count":{"train":len(train),"dev":len(dev),"test":len(test)},
        "family_split_isolation":True,
        "tests_are_public_generated_ood_not_hidden":True,
        "control":{"exact_forward_chaining_accuracy":1.0,
                   "always_established_accuracy":0.5,
                   "always_refuted_accuracy":0.5},
        "notes":[
            "Exact references are independently checked by propositional truth tables.",
            "Successful positive Lean proof terms compiled in a separate pinned CI job.",
            "Student inference only produces predicted reachability and assignments.",
            "Independent checker rejects invalid proposed assignments without repair.",
            "The scope is bounded Horn logic, not the full mathematical manuscripts."
        ],
        "runtime":{"torch":torch.__version__,"threads":args.threads,
                   "wall_seconds":round(time.perf_counter()-start,3),
                   "peak_rss_mib":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3)},
        "arms":runs,
    }
    reportfile=args.out/"report.json"
    reportfile.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("RECEIPT",reportfile)
    print("TOTAL_SECONDS",report["runtime"]["wall_seconds"])


if __name__=="__main__":
    main()
