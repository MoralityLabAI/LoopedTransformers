"""FC-06 preregistered joint LR-01 proof policy under a real 5-action budget.

Source registration: Compactification-Theory commit 81f53bcf606e45d8b3fea9acba7907641c339cab.
Training has no access to test_data.py. Model-selected rule lists alone enter
independent trace replay and Lean kernel compilation. Negative claims remain
exactly verified countermodels. Source-side teacher BFS is train-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fc03_compare as fc03
import fc04_adaptive as fc04
import fc05_dual as fc05
import formal_student as fs

PREREG_SHA="81f53bcf606e45d8b3fea9acba7907641c339cab"
TRAIN_SOURCE_SHA="1bf514325788b633432eb9f3e49627bd923a216e"
TRAIN_DATA_SHA="c841727d6a3e82c5ce5769b2b324e9bde078d15930d03fad5462292657d73ccf"
TEST_SOURCE_SHA="__FC06_SOURCE_SHA__"
TEST_DATA_SHA="__FC06_TEST_SHA__"
SEEDS=(17,29)
BUDGET=5
ACTOR_STEPS=300
ACTOR_BATCH=48
METHODS=("learned_joint","random_legal","goal_first")


def _source_input(task,known):
    """Whitelist public graph and the current derived facts; no answer features."""
    item=fc04._public_graph(task)
    b=fs.encode([item])
    atoms=task["atoms"]
    idx={name:i for i,name in enumerate(atoms)}
    for atom,i in idx.items():
        b["a"][0,i,0]=float(atom in known)
    legal=torch.zeros(fs.MAX_R,dtype=torch.bool)
    for j,rule in enumerate(task["rules"]):
        applicable=set(rule["premises"])<=known
        b["r"][0,j,2]=float(applicable)
        legal[j]=applicable and rule["head"] not in known
    return {k:b[k] for k in ("a","r","prem","head","goal_idx")},legal


def _stack(examples):
    keys=("a","r","prem","head","goal_idx")
    data={key:torch.cat([x[0][key] for x in examples],dim=0) for key in keys}
    legality=torch.stack([x[1] for x in examples])
    targets=torch.tensor([x[2] for x in examples],dtype=torch.long)
    return data,legality,targets


class JointActor(nn.Module):
    """FC-02/FC-04 LR-01 tied cell plus trainable goal-conditioned rule head."""

    def __init__(self):
        super().__init__()
        self.trunk=fc03.make_model("tied4_width24")
        self.head=fc05.RuleHead()

    def forward(self,b):
        a=self.trunk.atom_in(b["a"])
        r=self.trunk.rule_in(b["r"])
        for _ in range(4):
            a,r=self.trunk.cell(a,r,b["prem"],b["head"])
        heads=b["head"]@a
        parents=b["prem"]@a / b["prem"].sum(-1,keepdim=True).clamp_min(1.)
        goal=a.gather(1,b["goal_idx"][:,None,None].expand(-1,1,fs.WIDTH))
        goal=goal.expand(-1,r.shape[1],-1)
        features=torch.cat([r,heads,parents,goal],dim=-1)
        return self.head(features)


def load_train_source(root:Path):
    root=root.resolve()
    sys.path.insert(0,str(root))
    from research.refutation_bench.fc06 import train_data
    from research.refutation_bench.formal_curriculum import generator as fc02
    rows=train_data.generate_train_dev()
    report=train_data.audit_train_dev(rows)
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in rows)).hexdigest()
    if digest!=TRAIN_DATA_SHA:
        raise AssertionError("FC-06 train data changed: "+digest)
    training=[t for t in rows if t["split"]=="train"]
    dev=[t for t in rows if t["split"]=="dev"]
    if (len(training),len(dev))!=(768,192):
        raise AssertionError("train/dev split mutated")
    return train_data,training,dev,digest,report


def _training_states(training):
    """Teacher traces are used ONLY here, exclusively on training examples."""
    samples=[]
    positive_count=0
    for t in training:
        if t["label"]!="established":
            continue
        positive_count+=1
        trace=t["shortest_trace"]
        if trace is None or len(trace)>BUDGET or len(trace)!=t["min_steps"]:
            raise AssertionError("missing or overbudget training teacher")
        known=set(t["facts"])
        for choice in trace:
            data,legal=_source_input(t,known)
            if not legal[choice]:
                raise AssertionError("teacher selects illegal next rule")
            samples.append((data,legal,choice))
            known.add(t["rules"][choice]["head"])
        if t["goal"] not in known:
            raise AssertionError("training teacher failed to prove goal")
    return (*_stack(samples),positive_count)


def _train_actor(gate,training,seed):
    torch.manual_seed(seed+80017)
    actor=JointActor()
    actor.trunk.load_state_dict(gate.state_dict())
    data,legal,y,npositive=_training_states(training)
    opt=torch.optim.AdamW(actor.parameters(),lr=.002,weight_decay=.01)
    generator=torch.Generator().manual_seed(6154+seed)
    start=time.perf_counter()
    losslog=[]
    for step in range(ACTOR_STEPS):
        ids=torch.randint(len(y),(ACTOR_BATCH,),generator=generator)
        batch={key:tensor[ids] for key,tensor in data.items()}
        masks=legal[ids]
        actor.train()
        opt.zero_grad(set_to_none=True)
        scores=actor(batch).masked_fill(~masks,-1e4)
        loss=F.cross_entropy(scores,y[ids])
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(),1.)
        opt.step()
        if step in (0,ACTOR_STEPS//4,ACTOR_STEPS//2,ACTOR_STEPS-1):
            losslog.append([step+1,round(float(loss.detach()),6)])
    actor.eval()
    with torch.no_grad():
        # Train-only in-sample diagnostic; no checkpoint selection.
        probe=torch.arange(min(128,len(y)))
        logits=actor({key:tensor[probe] for key,tensor in data.items()})
        logits=logits.masked_fill(~legal[probe],-1e4)
        train_accuracy=(logits.argmax(-1)==y[probe]).float().mean().item()
    return actor,{
        "train_positive_cases":npositive,
        "train_prefix_states":len(y),
        "actor_steps":ACTOR_STEPS,
        "actor_batch":ACTOR_BATCH,
        "actor_parameters":sum(p.numel() for p in actor.parameters()),
        "trunk_parameters":sum(p.numel() for p in actor.trunk.parameters()),
        "head_parameters":sum(p.numel() for p in actor.head.parameters()),
        "train_first128_accuracy":round(train_accuracy,6),
        "losses":losslog,
        "training_seconds":round(time.perf_counter()-start,4)
    }


def train(root:Path,out:Path):
    """Only the train-only revision of Compactification-Theory may be mounted."""
    torch.set_num_threads(2)
    train_data,positives,_,digest,source_report=load_train_source(root)
    source_train,source_dev,_=fc03.load_source_training(root)
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.certificates import verify_countermodel
    out.mkdir(parents=True,exist_ok=True)
    manifest={
        "experiment":"FC-06",
        "preregistration_sha":PREREG_SHA,
        "train_source_sha":TRAIN_SOURCE_SHA,
        "fc06_train_data_sha256":digest,
        "fc02_train_data_sha256":fc03.FC02_DATA_SHA,
        "test_source_available_during_training":False,
        "train_set_size":len(positives),
        "training_model_recipe":"FC04 base 350 steps + FC06 actor joint 300 steps",
        "seed_list":list(SEEDS),
        "budget":BUDGET,
        "models":[],
    }
    for seed in SEEDS:
        gate,gate_meta=fc03._train_one(source_train,"tied4_width24",seed)
        theta=fc04.calibrate_threshold(gate,source_dev,verify_countermodel)
        actor,actor_meta=_train_actor(gate,positives,seed)
        name=f"fc06_s{seed}"
        gatepath=out/(name+"_gate.pt")
        actorpath=out/(name+"_actor.pt")
        torch.save(gate.state_dict(),gatepath)
        torch.save(actor.state_dict(),actorpath)
        info={
            "tag":name,"seed":seed,
            "threshold":theta["selected_threshold"],
            "threshold_grid":theta["all_grid_scores"],
            "gate_params":sum(p.numel() for p in gate.parameters()),
            "gate_sha256":hashlib.sha256(gatepath.read_bytes()).hexdigest(),
            "actor_sha256":hashlib.sha256(actorpath.read_bytes()).hexdigest(),
            "base_training_seconds":gate_meta["train_seconds"],
            **actor_meta
        }
        manifest["models"].append(info)
        print("FROZEN_FC06",json.dumps({k:info[k] for k in (
             "tag","threshold","train_prefix_states","train_first128_accuracy",
             "actor_parameters","gate_sha256","actor_sha256")},sort_keys=True),flush=True)
    path=out/"train_manifest.json"
    path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print("FROZEN_MANIFEST",path)


def _load_test(root:Path):
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.fc06 import test_data
    from research.refutation_bench.fc05 import proofs
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    tasks=test_data.generate_tests()
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in tasks)).hexdigest()
    if digest!=TEST_DATA_SHA:
        raise AssertionError("heldout test changed: "+digest)
    return tasks,proofs,verify_countermodel,digest


@torch.no_grad()
def propose(actor:JointActor,task:dict,method:str):
    """All actions must be currently legal and novel; learned head ranks them."""
    if method not in METHODS:
        raise ValueError(method)
    rng=random.Random(fc05._stable_seed(task["case_id"]))
    known=set(task["facts"])
    trace=[]
    actor_calls=0
    actor_ms=0.
    for _ in range(BUDGET):
        if task["goal"] in known:
            break
        data,mask=_source_input(task,known)
        options=mask.nonzero(as_tuple=True)[0].tolist()
        if not options:
            break
        if method=="random_legal":
            choice=rng.choice(options)
        elif method=="goal_first":
            targets=[i for i in options if task["rules"][i]["head"]==task["goal"]]
            choice=min(targets) if targets else rng.choice(options)
        else:
            start=time.perf_counter()
            scores=actor(data)[0].masked_fill(~mask,-1e4)
            choice=int(scores.argmax())
            actor_ms+=1000*(time.perf_counter()-start)
            actor_calls+=1
        if choice not in options:
            raise AssertionError("model returned illegal action")
        trace.append(choice)
        known.add(task["rules"][choice]["head"])
    return {
        "trace":trace,
        "actor_network_calls":actor_calls,
        "actor_recurrent_cells":4*actor_calls,
        "actor_forward_ms":round(actor_ms,6),
        "syntactic_goal_reached":task["goal"] in known
    }


@torch.no_grad()
def evaluate_condition(gate,actor,task,theta,method,proofs,verify,pre):
    """Apply shared FC-04 gate; certify only original selected source rules."""
    result={
        "case_id":task["case_id"],
        "method":method,
        "outcome":pre["outcome"],
        "preproof_outcome":pre["outcome"],
        "preproof_reason":pre["reason"],
        "stop_passes":pre["passes"],
        "stop_model_ms":pre["model_forward_ms"],
        "negative_checker_ms":pre["checker_ms"],
        "negative_checker_calls":pre["checker_calls"],
        "negative_countermodel":pre["certificate"] if pre["outcome"]=="refuted_certified" else None,
        "positive_trace":None,
        "proposal_trace":None,
        "positive_checked":False,
        "positive_checker_ms":0.,
        "positive_checker_calls":0,
        "actor_calls":0,
        "actor_recurrent_cells":0,
        "actor_forward_ms":0.,
        "rule_actions":0,
        "proof_failure_reason":None,
    }
    if pre["outcome"]!="established_provisional":
        return result
    draft=propose(actor,task,method)
    start=time.perf_counter()
    cert=proofs.verify_trace(task,draft["trace"],budget=BUDGET)
    checker_ms=1000*(time.perf_counter()-start)
    result.update(
        proposal_trace=draft["trace"],
        positive_checked=bool(cert["verified"]),
        positive_checker_calls=1,
        positive_checker_ms=round(checker_ms,6),
        actor_calls=draft["actor_network_calls"],
        actor_recurrent_cells=draft["actor_recurrent_cells"],
        actor_forward_ms=draft["actor_forward_ms"],
        rule_actions=len(draft["trace"]),
        proof_failure_reason=None if cert["verified"] else cert["reason"],
        outcome="established_certified" if cert["verified"] else "abstain",
        positive_trace=draft["trace"] if cert["verified"] else None,
    )
    return result


def _metrics(tasks,items):
    count=Counter()
    source=Counter()
    metrics=Counter()
    trace_lengths=[]
    trace_bytes=[]
    calls=[]
    totalms=0.
    for task,item in zip(tasks,items):
        if task["case_id"]!=item["case_id"]:
            raise AssertionError("case order mismatch")
        if item["outcome"]=="established_certified" and task["label"]!="established":
            raise AssertionError("false positive certificate")
        if item["outcome"]=="refuted_certified" and task["label"]!="refuted":
            raise AssertionError("false negative certificate")
        count[(task["label"],item["outcome"])]+=1
        source[(task["label"],item["preproof_outcome"])]+=1
        metrics["stop_calls"]+=item["stop_passes"]
        metrics["actor_cells"]+=item["actor_recurrent_cells"]
        metrics["checker_calls"]+=item["positive_checker_calls"]+item["negative_checker_calls"]
        totalms+=item["stop_model_ms"]+item["actor_forward_ms"]+item["negative_checker_ms"]+item["positive_checker_ms"]
        cells=item["stop_passes"]+item["actor_recurrent_cells"]
        calls.append(cells)
        if item["positive_trace"] is not None:
            trace_lengths.append(len(item["positive_trace"]))
            trace_bytes.append(len(json.dumps(item["positive_trace"]).encode("utf-8")))
    good=sum(t["label"]=="established" for t in tasks)
    bad=len(tasks)-good
    pos=count["established","established_certified"]
    neg=count["refuted","refuted_certified"]
    eligible_good=source["established","established_provisional"]
    return {
        "n":len(tasks),"valid":good,"invalid":bad,
        "eligible_positive":eligible_good,
        "certified_positive":pos,"certified_negative":neg,
        "positive_recall":round(pos/good,6),
        "negative_recall":round(neg/bad,6),
        "success_on_eligible_positive":round(pos/eligible_good,6) if eligible_good else None,
        "combined_certified_coverage":round((pos+neg)/len(tasks),6),
        "false_formal_certificates":0,
        "abstentions":len(tasks)-pos-neg,
        "suppressed_wrong_provisional":source["refuted","established_provisional"],
        "total_stop_cells":metrics["stop_calls"],
        "total_actor_cells":metrics["actor_cells"],
        "total_recurrent_cells":metrics["stop_calls"]+metrics["actor_cells"],
        "certified_positive_per_1000_cells":round(1000*pos/(metrics["stop_calls"]+metrics["actor_cells"]),6),
        "mean_total_cells":round(statistics.mean(calls),6),
        "p95_total_cells":sorted(calls)[min(len(calls)-1,int(.95*len(calls)))],
        "mean_accepted_rule_actions":round(statistics.mean(trace_lengths),6) if trace_lengths else None,
        "mean_accepted_trace_bytes":round(statistics.mean(trace_bytes),6) if trace_bytes else None,
        "exact_checker_calls":metrics["checker_calls"],
        "model_and_checker_ms_per_case":round(totalms/len(tasks),6),
        "confusion":{label+"->"+verdict:count[label,verdict]
             for label in ("established","refuted")
             for verdict in ("established_certified","refuted_certified","abstain")},
    }


def _paired(tasks,a,b):
    win=lose=0
    for case,x,y in zip(tasks,a,b):
        if x["case_id"]!=y["case_id"]!=case["case_id"]:
            raise AssertionError("paired ID mismatch")
        good=case["label"]=="established"
        xwin=good and x["outcome"]=="established_certified"
        ywin=good and y["outcome"]=="established_certified"
        win+=int(xwin and not ywin)
        lose+=int(ywin and not xwin)
    return {"learned_only":win,"baseline_only":lose,"net_learned":win-lose}


def evaluate(root:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifest=json.loads((weights/"train_manifest.json").read_text())
    if manifest["preregistration_sha"]!=PREREG_SHA or manifest["fc06_train_data_sha256"]!=TRAIN_DATA_SHA:
        raise AssertionError("training manifest incompatible with registration")
    if manifest.get("test_source_available_during_training") is not False:
        raise AssertionError("training claimed test access")
    if {m["seed"] for m in manifest["models"]}!=set(SEEDS):
        raise AssertionError("seed set modified")
    cases,proofs,verify,testhash=_load_test(root)
    out.mkdir(parents=True,exist_ok=True)
    accepted=[]
    scores=[]
    comparisons=[]
    for row in manifest["models"]:
        seed=row["seed"];tag=row["tag"]
        gate_file=weights/(tag+"_gate.pt")
        actor_file=weights/(tag+"_actor.pt")
        if hashlib.sha256(gate_file.read_bytes()).hexdigest()!=row["gate_sha256"]:
            raise AssertionError("gate checkpoint changed")
        if hashlib.sha256(actor_file.read_bytes()).hexdigest()!=row["actor_sha256"]:
            raise AssertionError("actor checkpoint changed")
        gate=fc03.make_model("tied4_width24")
        gate.load_state_dict(torch.load(gate_file,weights_only=True,map_location="cpu"))
        gate.eval()
        actor=JointActor()
        actor.load_state_dict(torch.load(actor_file,weights_only=True,map_location="cpu"))
        actor.eval()
        theta=row["threshold"]
        for split in ("test_same","test_deep"):
            tasks=[t for t in cases if t["split"]==split]
            # Freeze gate decisions for fair paired positive-action evaluation.
            pre=[fc04.run_one(gate,t,theta,"adaptive",verify) for t in tasks]
            by_method={}
            for method in METHODS:
                preds=[evaluate_condition(gate,actor,t,theta,method,proofs,verify,p)
                       for t,p in zip(tasks,pre)]
                result=_metrics(tasks,preds)
                groups={}
                for family in sorted(set(t["family_id"] for t in tasks)):
                    subset=[(t,p) for t,p in zip(tasks,preds) if t["family_id"]==family]
                    groups[family]=_metrics([t for t,_ in subset],[p for _,p in subset])
                result.update(method=method,seed=seed,split=split,
                              theta=theta,by_family=groups)
                file=out/(tag+"_"+split+"_"+method+".jsonl")
                file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in preds))
                result["predictions_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
                scores.append(result)
                by_method[method]=preds
                if method=="learned_joint":
                    accepted.extend((t,p["positive_trace"]) for t,p in zip(tasks,preds)
                                    if p["outcome"]=="established_certified")
                print("FC06_SCORE",json.dumps({k:result[k] for k in (
                     "method","seed","split","eligible_positive","certified_positive",
                     "certified_negative","success_on_eligible_positive",
                     "abstentions","total_recurrent_cells",
                     "certified_positive_per_1000_cells")},sort_keys=True),flush=True)
            for method in ("random_legal","goal_first"):
                comparisons.append({"seed":seed,"split":split,"against":method,
                    **_paired(tasks,by_method["learned_joint"],by_method[method])})
    lean=out/"FC06_model_proofs.lean"
    lean.write_text(proofs.lean_document(accepted))
    primary=[]
    for seed in SEEDS:
        for split in ("test_same","test_deep"):
            learned=next(s for s in scores if s["seed"]==seed and s["split"]==split
                         and s["method"]=="learned_joint")
            random_base=next(s for s in scores if s["seed"]==seed and s["split"]==split
                         and s["method"]=="random_legal")
            goal_base=next(s for s in scores if s["seed"]==seed and s["split"]==split
                         and s["method"]=="goal_first")
            primary.append({
                "seed":seed,"split":split,
                "learned_better_than_both":learned["certified_positive"]>max(
                    random_base["certified_positive"],goal_base["certified_positive"]),
                "learned_more_certified_per_compute":learned["certified_positive_per_1000_cells"]>
                    max(random_base["certified_positive_per_1000_cells"],
                        goal_base["certified_positive_per_1000_cells"]),
                "zero_false_formal":learned["false_formal_certificates"]==0
            })
    report={
        "experiment":"FC-06",
        "preregistration_sha":PREREG_SHA,
        "train_source_commit":TRAIN_SOURCE_SHA,
        "train_data_sha256":TRAIN_DATA_SHA,
        "test_source_commit":TEST_SOURCE_SHA,
        "test_data_sha256":testhash,
        "train_manifest_sha256":hashlib.sha256((weights/"train_manifest.json").read_bytes()).hexdigest(),
        "first_frozen_test":True,
        "strict_budget":BUDGET,
        "positive_kernel_proofs":len(accepted),
        "lean_proofs_sha256":hashlib.sha256(lean.read_bytes()).hexdigest(),
        "registered_learning_criterion":all(x["learned_better_than_both"]
                                   and x["zero_false_formal"] for x in primary),
        "registered_compute_criterion":all(x["learned_more_certified_per_compute"]
                                   for x in primary),
        "primary":primary,"paired":comparisons,"scores":scores,
        "scope":"public bounded Horn entailment, proof trace and full Boolean countermodel"
    }
    (out/"FC06_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("FC06_CRITERIA",json.dumps({
        "learning":report["registered_learning_criterion"],
        "compute":report["registered_compute_criterion"],
        "positive_proofs":report["positive_kernel_proofs"]},sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="train":
        if a.weights is not None:
            raise ValueError("no evaluation checkpoint should be supplied to train")
        train(a.source,a.out)
    else:
        if a.weights is None:
            raise ValueError("missing pretrained frozen artifacts")
        evaluate(a.source,a.weights,a.out)


if __name__=="__main__":
    main()
