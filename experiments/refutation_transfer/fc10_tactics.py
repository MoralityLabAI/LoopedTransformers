"""FC-10: actual Lean tactic-state LR-01 policy and kernel-checked evaluation.

Proof tasks are *real Lean 4.19.0 declarations*. Goal strings are harvested
from Lean's elaborator via trace_state. No theorem signature, case ID,
reference proof, family or gold tactic enters neural inference. A candidate
tactic counts as successful ONLY if a fresh Lean subprocess compiles the
corresponding original theorem with that selected candidate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"lean_reasoner"))
from train import TypedLoop

PREREG_SHA="32330e86366f3ea6378ceb1bf6a4b189610b4483"
TRAIN_SOURCE_SHA="631ccbb1dd393d2608b5d2c7e154aeb3ce5099b0"
TRAIN_GOAL_STATES_SHA="3d2c5b6055ffb0bf0118c69f9a6ea405a92a922f7f869b591e26ae5feb9cb03b"
TEST_SOURCE_SHA="171d23e84dfc5af3ed8723ac99f75badbaf6a619"
TEST_GOAL_STATES_SHA="8e1792c737c47124b8962e3377ec4be12ec91c083b9d87644e5293fc892b0eed"
REGISTRY=(
    "omega","rfl","assumption","constructor <;> assumption",
    "exact h.1","exact h.2","exact Or.inl h","exact Or.inr h",
    "exact And.intro h.2 h.1","simp [List.append_assoc]",
    "simp_all","exact False.elim h",
)
WIDTH=24
MAX_TOKENS=40
BUCKETS=2048
SEEDS=(17,29)
LOOPS=(1,4)
TRAIN_STEPS=250
BATCH=24
LEARN_RATE=.002


def goal_tokens(goal:str)->list[str]:
    if "⊢" not in goal:
        raise AssertionError("student input must contain real Lean turnstile")
    parts=re.findall(r"[\w']+|[^\w\s]",goal,flags=re.UNICODE)
    if len(parts)>MAX_TOKENS:
        parts=parts[:MAX_TOKENS//2]+parts[-MAX_TOKENS//2:]
    return parts


def token_bucket(word:str)->int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf8"),digest_size=8).digest(),"big")%BUCKETS


def features(goals:list[str])->dict[str,torch.Tensor]:
    B=len(goals);A=MAX_TOKENS;R=len(REGISTRY)
    tokens=torch.zeros((B,A),dtype=torch.long)
    present=torch.zeros((B,A),dtype=torch.float32)
    for b,goal in enumerate(goals):
        pieces=goal_tokens(goal)
        for i,piece in enumerate(pieces):
            tokens[b,i]=token_bucket(piece)
            present[b,i]=1.
    prem=present[:,None,:].expand(-1,R,-1).clone()
    # All tactic nodes point at the last goal-token node, while
    # their premise incidence includes the whole observed state.
    head=torch.zeros((B,R,A),dtype=torch.float32)
    for b,goal in enumerate(goals):
        pieces=goal_tokens(goal)
        # The goal conclusion follows the Lean turnstile; preserve its
        # distinguishing token as the shared candidate target node.
        indices=[i for i,piece in enumerate(pieces) if piece=="⊢"]
        if not indices:
            raise AssertionError("Lean goal marker was truncated")
        head[b,:,indices[0]]=1.
    return {"tokens":tokens,"present":present,"prem":prem,"head":head}


class TacticLoop(nn.Module):
    """Use the EXACT original LR-01 TypedLoop shared-weight cell."""

    def __init__(self,loops:int=4,width:int=WIDTH):
        super().__init__()
        self.loops=loops
        self.token_emb=nn.Embedding(BUCKETS,width)
        self.action_emb=nn.Embedding(len(REGISTRY),width)
        self.cell=TypedLoop(width)
        self.score=nn.Sequential(nn.Linear(3*width,width),nn.GELU(),
                                 nn.Linear(width,1))

    def forward(self,inp:dict[str,torch.Tensor]):
        token=inp["tokens"]
        a=self.token_emb(token)*inp["present"][...,None]
        indices=torch.arange(len(REGISTRY),device=token.device)[None,:].expand(token.shape[0],-1)
        r=self.action_emb(indices)
        pre=inp["prem"]
        head=inp["head"]
        for _ in range(self.loops):
            a,r=self.cell(a,r,pre,head)
        parents=pre@a/pre.sum(-1,keepdim=True).clamp_min(1.)
        target=head@a
        return self.score(torch.cat((r,parents,target),-1)).squeeze(-1)


def _digest(goal_rows):
    return hashlib.sha256("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n"
              for x in goal_rows).encode("utf8")).hexdigest()


def _read_rows(path:Path):
    rows=[json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    if any("⊢" not in row["goal_state"] for row in rows):
        raise AssertionError("missing compiler-origin Lean goal state")
    if len({row["id"] for row in rows})!=len(rows):
        raise AssertionError("duplicate Lean goal ID")
    if any(row["reference_tactic"] not in REGISTRY for row in rows):
        raise AssertionError("unregistered tactic in source labels")
    return rows


def _labels(rows):
    return torch.tensor([REGISTRY.index(r["reference_tactic"]) for r in rows],dtype=torch.long)


def _subset(inp,indices):
    return {key:value[indices] for key,value in inp.items()}


def train_one(train_rows,dev_rows,seed:int,loops:int):
    torch.manual_seed(seed+loops*1000)
    torch.use_deterministic_algorithms(True)
    model=TacticLoop(loops=loops)
    tensor=features([r["goal_state"] for r in train_rows])
    labels=_labels(train_rows)
    opt=torch.optim.AdamW(model.parameters(),lr=LEARN_RATE,weight_decay=.01)
    sampler=torch.Generator().manual_seed(seed+6154)
    started=time.perf_counter()
    losslog=[]
    for step in range(TRAIN_STEPS):
        ids=torch.randint(len(train_rows),(BATCH,),generator=sampler)
        model.train()
        scores=model(_subset(tensor,ids))
        loss=F.cross_entropy(scores,labels[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in (0,TRAIN_STEPS//2,TRAIN_STEPS-1):
            losslog.append([step+1,round(float(loss.detach()),6)])
    model.eval()
    with torch.no_grad():
        devpred=model(features([r["goal_state"] for r in dev_rows])).argmax(-1)
    return model,{
       "seed":seed,"loops":loops,
       "model_parameters":sum(p.numel() for p in model.parameters()),
       "steps":TRAIN_STEPS,"batch":BATCH,"lr":LEARN_RATE,
       "train_seconds":round(time.perf_counter()-started,4),
       "dev_reference_label_agreement":round((devpred==_labels(dev_rows)).float().mean().item(),6),
       "losses":losslog,
    }


def lexical_rank(train_rows,goal_state:str)->list[int]:
    """Train-only token-Jaccard nearest neighbor; deterministic tie order."""
    q=set(goal_tokens(goal_state))
    candidates=[]
    for row in train_rows:
        v=set(goal_tokens(row["goal_state"]))
        sim=len(q&v)/max(1,len(q|v))
        candidates.append((sim,row["reference_tactic"]))
    tally=Counter()
    for sim,tactic in candidates:
        tally[tactic]=max(tally[tactic],sim)
    priors=Counter(x["reference_tactic"] for x in train_rows)
    return sorted(range(len(REGISTRY)),
        key=lambda i:(-tally[REGISTRY[i]],-priors[REGISTRY[i]],i))


def frozen_train(source:Path,out:Path):
    """Train only against pinned public source without FC-10 test generator."""
    torch.set_num_threads(2)
    sys.path.insert(0,str(source.resolve()))
    from research.refutation_bench.fc10 import core
    if tuple(core.TACTICS)!=REGISTRY:
        raise AssertionError("registered tactic action order changed")
    root=out/"source"
    root.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,"-m","research.refutation_bench.fc10.train_data",
                    "--out-dir",str(root)],check=True,cwd=str(source))
    proc=subprocess.run(["lean",str(root/"TrainDev.lean")],
          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    (root/"lean_states.log").write_text(proc.stdout)
    if proc.returncode!=0 or "sorryAx" in proc.stdout:
        raise RuntimeError("Lean training corpus failed kernel compilation")
    subprocess.run([sys.executable,"-m","research.refutation_bench.fc10.train_data",
                    "--out-dir",str(root),"--extract-log",str(root/"lean_states.log")],
                   check=True,cwd=str(source))
    goal_file=root/"train_states.jsonl"
    rows=_read_rows(goal_file)
    if len(rows)!=156 or _digest(rows)!=TRAIN_GOAL_STATES_SHA:
        raise AssertionError("training goal states differ from frozen compiler CI")
    train=[x for x in rows if x["split"]=="train"]
    dev=[x for x in rows if x["split"]=="dev"]
    manifest={
      "experiment":"FC10_real_lean",
      "phase":"train_only_before_heldout_test_source_access",
      "prereg_sha":PREREG_SHA,
      "train_source_sha":TRAIN_SOURCE_SHA,
      "train_actual_goals_sha256":TRAIN_GOAL_STATES_SHA,
      "test_source_available_during_training":False,
      "tactic_registry":list(REGISTRY),
      "train_count":len(train),"dev_count":len(dev),
      "models":[],
    }
    for seed in SEEDS:
        for loops in LOOPS:
            model,meta=train_one(train,dev,seed,loops)
            tag=f"fc10_loop{loops}_s{seed}"
            file=out/(tag+".pt")
            torch.save(model.state_dict(),file)
            meta["tag"]=tag
            meta["checkpoint_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
            manifest["models"].append(meta)
            print("FC10_TRAINED",json.dumps({k:meta[k] for k in (
                 "tag","model_parameters","dev_reference_label_agreement",
                 "train_seconds","checkpoint_sha256")},sort_keys=True),flush=True)
    # The lexical baseline may see only these original training states.
    (out/"lexical_train_states.jsonl").write_bytes(goal_file.read_bytes())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


def _lean_candidate(core,row,tactic,working_dir:Path):
    """Actual Lean kernel-checked tactic application: no reference proof fallback."""
    source=core.lean_document([row],trace=False,tactic_map={row["id"]:tactic})
    working_dir.mkdir(parents=True,exist_ok=True)
    path=working_dir/"Candidate.lean"
    path.write_text(source,encoding="utf8")
    start=time.perf_counter()
    proc=subprocess.run(["lean",str(path)],stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=15)
    elapsed=1000*(time.perf_counter()-start)
    valid=proc.returncode==0 and "sorryAx" not in proc.stdout
    return {
      "tactic":tactic,
      "lean_accepted":valid,
      "lean_wall_ms":round(elapsed,5),
      "compiler_message":"" if valid else proc.stdout[-420:],
      "source_sha256":hashlib.sha256(source.encode("utf8")).hexdigest(),
    }


def heldout_evaluate(source:Path,train_dir:Path,out:Path):
    torch.set_num_threads(2)
    manifest=json.loads((train_dir/"manifest.json").read_text())
    if manifest["prereg_sha"]!=PREREG_SHA or (
        manifest["train_actual_goals_sha256"]!=TRAIN_GOAL_STATES_SHA or
        manifest["test_source_available_during_training"] is not False):
        raise AssertionError("incompatible frozen model manifest")
    root=out/"source"
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source.resolve()))
    from research.refutation_bench.fc10 import core
    if tuple(core.TACTICS)!=REGISTRY:
        raise AssertionError("tactic action registry changed in heldout source")
    subprocess.run([sys.executable,"-m","research.refutation_bench.fc10.test_data",
          "--out-dir",str(root)],check=True,cwd=str(source))
    proc=subprocess.run(["lean",str(root/"Test.lean")],
          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    (root/"lean_states.log").write_text(proc.stdout)
    if proc.returncode!=0 or "sorryAx" in proc.stdout:
        raise RuntimeError("Lean held-out reference corpus failed")
    subprocess.run([sys.executable,"-m","research.refutation_bench.fc10.test_data",
          "--out-dir",str(root),"--extract-log",str(root/"lean_states.log")],
                   check=True,cwd=str(source))
    test_rows=_read_rows(root/"test_states.jsonl")
    if len(test_rows)!=48 or _digest(test_rows)!=TEST_GOAL_STATES_SHA:
        raise AssertionError("unseen real Lean test states differ from source CI")
    source_rows=[json.loads(s) for s in (root/"test.jsonl").read_text().splitlines()]
    if [x["id"] for x in test_rows]!=[x["id"] for x in source_rows]:
        raise AssertionError("test record ID and goal states mismatch")
    lexical_rows=_read_rows(train_dir/"lexical_train_states.jsonl")
    if _digest(lexical_rows)!=TRAIN_GOAL_STATES_SHA:
        raise AssertionError("lexical training states were modified")
    out.mkdir(parents=True,exist_ok=True)
    models={}
    for meta in manifest["models"]:
        file=train_dir/(meta["tag"]+".pt")
        if hashlib.sha256(file.read_bytes()).hexdigest()!=meta["checkpoint_sha256"]:
            raise AssertionError("pretest checkpoint SHA256 changed")
        model=TacticLoop(loops=meta["loops"])
        model.load_state_dict(torch.load(file,weights_only=True,map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits=model(features([x["goal_state"] for x in test_rows]))
            ordered=logits.argsort(-1,descending=True)
        models[meta["tag"]]=ordered.tolist()
    rankings={"lexical":[lexical_rank(lexical_rows,s["goal_state"]) for s in test_rows]}
    rankings.update(models)

    # Cache exact per-theorem per-tactic Lean outcomes across model arms,
    # without consulting the reference label to choose the tactic.
    cache={}
    def checked(i:int,action_idx:int):
        key=(i,action_idx)
        if key not in cache:
            cache[key]=_lean_candidate(core,source_rows[i],REGISTRY[action_idx],
                          out/"lean_candidate")
        return cache[key]

    all_scores=[]
    all_predictions=[]
    # Lexical and neural top1 and top3 are checked by a real Lean subprocess.
    for name,orderings in rankings.items():
        for limit in (1,3):
            correct=0
            attempts=0
            ms=0.
            rows=[]
            for i,(goal,order) in enumerate(zip(test_rows,orderings)):
                outcomes=[]
                for candidate in order[:limit]:
                    result=checked(i,candidate)
                    outcomes.append(result)
                    attempts+=1
                    ms+=result["lean_wall_ms"]
                    if result["lean_accepted"]:
                        break
                accepted=bool(outcomes and outcomes[-1]["lean_accepted"])
                correct+=int(accepted)
                row={"id":goal["id"],"family":goal["family"],
                    "policy":name,"limit":limit,
                    "goal_state_sha256":goal["goal_state_sha256"],
                    "chosen_tactics":[x["tactic"] for x in outcomes],
                    "accepted":accepted,"attempts":len(outcomes),
                    "kernel_trials":outcomes}
                rows.append(row)
            file=out/(f"{name}_top{limit}.jsonl")
            file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
            all_predictions.extend(rows)
            metric={"policy":name,"max_tactic_attempts":limit,
                    "cases":len(rows),"compiled_proofs":correct,
                    "coverage":round(correct/len(rows),6),
                    "total_tactic_attempts":attempts,
                    "total_lean_wall_ms":round(ms,4),
                    "ms_per_compiled_proof":round(ms/correct,4) if correct else None,
                    "predictions_sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
            all_scores.append(metric)
            print("FC10_SCORES",json.dumps(metric,sort_keys=True),flush=True)

    # Gold labels are consulted only to compute coverage of a separate
    # explicitly labelled upper control, NOT as neural inference inputs.
    symbolic=[]
    for i,item in enumerate(test_rows):
        hits=[]
        for candidate in range(len(REGISTRY)):
            result=checked(i,candidate)
            hits.append(result)
            if result["lean_accepted"]:
                break
        symbolic.append({"id":item["id"],
                         "success":bool(hits[-1]["lean_accepted"]),
                         "trials":hits})
    success=sum(x["success"] for x in symbolic)
    symbolic_file=out/"symbolic_full_registry.jsonl"
    symbolic_file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in symbolic))
    symbolic_metric={"policy":"symbolic_full_registry",
                     "max_tactic_attempts":len(REGISTRY),
                     "compiled_proofs":success,
                     "cases":len(symbolic),
                     "total_tactic_attempts":sum(len(x["trials"]) for x in symbolic),
                     "total_lean_wall_ms":round(sum(t["lean_wall_ms"] for x in symbolic
                                          for t in x["trials"]),4)}
    all_scores.append(symbolic_metric)
    print("FC10_SYMBOLIC",json.dumps(symbolic_metric,sort_keys=True),flush=True)
    primary=next(s for s in all_scores if s["policy"]=="lexical" and s["max_tactic_attempts"]==1)
    results=[]
    for seed in SEEDS:
        for loops in LOOPS:
            m=next(s for s in all_scores if s["policy"]==f"fc10_loop{loops}_s{seed}"
                   and s["max_tactic_attempts"]==1)
            results.append({"seed":seed,"loops":loops,
                    "compiled":m["compiled_proofs"],
                    "lexical_top1":primary["compiled_proofs"],
                    "better_than_lexical_top1":m["compiled_proofs"]>primary["compiled_proofs"]})
    outcome={
       "experiment":"FC10_first_real_lean_goal_states",
       "preregistration_sha":PREREG_SHA,
       "train_source_commit":TRAIN_SOURCE_SHA,
       "train_real_goals_sha256":TRAIN_GOAL_STATES_SHA,
       "test_source_commit":TEST_SOURCE_SHA,
       "test_real_goals_sha256":TEST_GOAL_STATES_SHA,
       "test_count":len(test_rows),
       "runtime_lean_version":"4.19.0",
       "kernel_verification_mandatory":True,
       "no_gold_reference_tactic_used_for_model_inference":True,
       "candidate_cache_lean_invocations":len(cache),
       "all_candidate_trials_kernel_checked":True,
       "full_symbolic_search_recall":success,
       "model_success_criterion":any(x["loops"]==4 and x["better_than_lexical_top1"]
                         for x in results),
       "primary":results,"scores":all_scores,
       "note":"Actual printed Lean tactic states on synthetic theorem templates; not arbitrary mathlib theorems.",
    }
    (out/"FC10_report.json").write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n")
    print("FC10_CRITERIA",json.dumps({k:outcome[k] for k in (
        "model_success_criterion","full_symbolic_search_recall",
        "candidate_cache_lean_invocations")},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="train":
        if a.weights is not None:
            raise ValueError("training may not be given hidden-test artifacts")
        frozen_train(a.source,a.out)
    else:
        if a.weights is None:
            raise ValueError("test must consume frozen train/checkpoint artifact")
        heldout_evaluate(a.source,a.weights,a.out)


if __name__=="__main__":
    main()
