"""FC-05: a learned, bounded positive proof trace beside FC-04 negative certificates.

The model's selected rule indices are replayed by an independent exact
checker, and the same action trace is rendered as a Lean 4.19.0 proof term
without oracle search or repair. The proof head is trained only on FC-02
*training* positives, and the FC-04 stopping threshold on FC-02 dev.
The full FC-05 test set is checked out only in the dependent evaluation job.
"""
from __future__ import annotations
import argparse
from collections import Counter,deque
from copy import deepcopy
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
import formal_student as fs
import fc03_compare as fc03
import fc04_adaptive as fc04

PREREG_SHA="e6ab5716e3316441adad131b087a50d1ca0f2734"
FC05_SOURCE_SHA="__FC05_PIN__"
FC05_DATA_SHA="__FC05_DATA_HASH__"
STUDENT_SEEDS=(17,29)
RULE_BUDGET=10
TRAIN_HEAD_STEPS=250
HEAD_BATCH=48
BASE_MODEL="tied4_width24"
RULE_METHODS=("learned_trace","random_legal","immediate_goal_random")


def _stable_seed(case_id:str)->int:
    digest=hashlib.sha256(case_id.encode()).digest()
    return 2917+int.from_bytes(digest[:4],"big")


def _known_facts(task):
    return frozenset(task["facts"])


def legal_rules(task,known:set|frozenset)->list[int]:
    """Only source-specified implications, with all premises already derived."""
    return [i for i,r in enumerate(task["rules"])
            if r["head"] not in known and set(r["premises"])<=known]


def shortest_training_trace(task,budget:int=RULE_BUDGET)->list[int]:
    """Training-only exact BFS labeler; never called on FC-05 test inputs."""
    source=frozenset(task["facts"])
    queue=deque([(source,())])
    visited={source}
    while queue:
        known,trace=queue.popleft()
        if task["goal"] in known:
            return list(trace)
        if len(trace)>=budget:
            continue
        for index in legal_rules(task,known):
            next_known=known|frozenset([task["rules"][index]["head"]])
            if next_known in visited:
                continue
            visited.add(next_known)
            queue.append((next_known,trace+(index,)))
    raise ValueError("training proof has no <=10-action derivation")


def _features(model,task,known):
    """Build exact graph features from public graph plus current proved facts."""
    sample=fc04._public_graph(task)
    data=fs.encode([sample])
    ids={atom:i for i,atom in enumerate(task["atoms"])}
    for atom,i in ids.items():
        data["a"][0,i,0]=float(atom in known)
    for j,rule in enumerate(task["rules"]):
        data["r"][0,j,2]=float(set(rule["premises"])<=set(known))
    a=model.atom_in(data["a"])
    r=model.rule_in(data["r"])
    for _ in range(4):
        a,r=model.cell(a,r,data["prem"],data["head"])
    parent=data["prem"]@a / data["prem"].sum(-1,keepdim=True).clamp_min(1.)
    head_emb=data["head"]@a
    goal_index=ids[task["goal"]]
    goal=a[:,goal_index:goal_index+1,:].expand(-1,r.shape[1],-1)
    features=torch.cat((r,head_emb,parent,goal),-1)
    legal=torch.zeros(fs.MAX_R,dtype=torch.bool)
    legal[legal_rules(task,known)]=True
    return features[0],legal


class RuleHead(nn.Module):
    """Only this 4,705-parameter rule readout is trained on proof traces."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4*fs.WIDTH,48),nn.GELU(),
                               nn.Linear(48,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def _make_training_features(model,train):
    # Train-only BFS is the source of next-rule labels.
    pieces,valid,target=[],[],[]
    n_positive=0
    for task in train:
        if task["label"]!="established":
            continue
        trace=shortest_training_trace(task)
        n_positive+=1
        known=set(task["facts"])
        for rule_id in trace:
            x,legal=_features(model,task,known)
            if not legal[rule_id]:
                raise AssertionError("training oracle chose an inapplicable action")
            pieces.append(x.cpu())
            valid.append(legal.cpu())
            target.append(rule_id)
            known.add(task["rules"][rule_id]["head"])
        if task["goal"] not in known:
            raise AssertionError("reference trace failed to close goal")
    return (torch.stack(pieces),torch.stack(valid),
            torch.tensor(target,dtype=torch.long),n_positive)


def train_head(model,train,seed:int):
    model.eval()
    features,legality,target,count=_make_training_features(model,train)
    torch.manual_seed(seed+80017)
    head=RuleHead()
    opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+9173)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    losses=[]
    for step in range(TRAIN_HEAD_STEPS):
        indexes=torch.randint(len(target),(HEAD_BATCH,),generator=g)
        output=head(features[indexes]).masked_fill(~legality[indexes],-1e4)
        loss=F.cross_entropy(output,target[indexes])
        head.train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        opt.step()
        if step in (0,TRAIN_HEAD_STEPS//2,TRAIN_HEAD_STEPS-1):
            losses.append([step+1,round(float(loss.detach()),6)])
    head.eval()
    return head,{"training_only_positive_problems":count,
                 "training_only_positive_prefix_states":len(target),
                 "training_steps":TRAIN_HEAD_STEPS,
                 "head_parameter_count":sum(p.numel() for p in head.parameters()),
                 "rule_training_losses":losses,
                 "head_training_seconds":round(time.perf_counter()-started,4)}


@torch.no_grad()
def propose_trace(model,head,task,method:str,budget:int=RULE_BUDGET):
    """Bounded model action sequence; independent checker is called later."""
    if method not in RULE_METHODS:
        raise ValueError(method)
    rng=random.Random(_stable_seed(task["case_id"]))
    known=set(task["facts"])
    trace=[]
    forward_time=0.
    head_calls=0
    for _ in range(budget):
        if task["goal"] in known:
            break
        legal=legal_rules(task,known)
        if not legal:
            break
        if method=="random_legal":
            choice=rng.choice(legal)
        elif method=="immediate_goal_random":
            goal_rules=[i for i in legal if task["rules"][i]["head"]==task["goal"]]
            choice=min(goal_rules) if goal_rules else rng.choice(legal)
        else:
            start=time.perf_counter()
            features,flags=_features(model,task,known)
            scores=head(features).masked_fill(~flags,-1e4)
            choice=int(scores.argmax())
            forward_time+=time.perf_counter()-start
            head_calls+=1
        if choice not in legal:
            raise AssertionError("policy proposed an inapplicable rule")
        trace.append(choice)
        known.add(task["rules"][choice]["head"])
    return {"trace":trace,"head_calls":head_calls,
            "proof_model_forward_ms":round(1000*forward_time,6),
            "goal_reached_syntactically":task["goal"] in known}


@torch.no_grad()
def release_certificates(model,head,task,theta,method,verify_negative,verify_positive):
    """FC-04 stopping decision followed by one bounded, checked positive proof."""
    raw=fc04.run_one(model,task,theta,"adaptive",verify_negative)
    output={
        "case_id":task["case_id"],
        "stop_passes":raw["passes"],
        "goal_probability":raw["goal_probability"],
        "fc04_preproof_outcome":raw["outcome"],
        "outcome":raw["outcome"],
        "negative_countermodel":raw["certificate"] if raw["outcome"]=="refuted_certified" else None,
        "positive_trace":None,
        "positive_certified":False,
        "rule_applications":0,
        "rule_head_forward_calls":0,
        "stop_model_forward_ms":raw["model_forward_ms"],
        "proof_model_forward_ms":0.,
        "negative_checker_ms":raw["checker_ms"],
        "positive_checker_ms":0.,
        "negative_checker_calls":raw["checker_calls"],
        "positive_checker_calls":0,
        "proof_attempt_failed":False,
    }
    if raw["outcome"]=="established_provisional":
        proposal=propose_trace(model,head,task,method)
        start=time.perf_counter()
        certificate=verify_positive(task,proposal["trace"])
        check_ms=1000*(time.perf_counter()-start)
        output["positive_checker_calls"]=1
        output["positive_checker_ms"]=round(check_ms,6)
        output["rule_head_forward_calls"]=proposal["head_calls"]
        output["rule_applications"]=len(proposal["trace"])
        output["proof_model_forward_ms"]=proposal["proof_model_forward_ms"]
        if certificate["verified"]:
            output["outcome"]="established_certified"
            output["positive_certified"]=True
            output["positive_trace"]=proposal["trace"]
        else:
            output["outcome"]="abstain"
            output["proof_attempt_failed"]=True
            output["proof_failure_reason"]=certificate["reason"]
    return output


def _score(tasks,results):
    if len(tasks)!=len(results):
        raise AssertionError("missing evaluation case")
    tally=Counter()
    actions=passes=head_calls=0
    proof_ms=stop_ms=neg_ms=pos_ms=0.
    proof_bytes=[]
    for case,p in zip(tasks,results):
        if case["case_id"]!=p["case_id"]:
            raise AssertionError("predicted case IDs differ")
        if p["outcome"]=="established_certified" and case["label"]!="established":
            raise AssertionError("false positive certificate escaped")
        if p["outcome"]=="refuted_certified" and case["label"]!="refuted":
            raise AssertionError("false negative certificate escaped")
        tally[(case["label"],p["outcome"])]+=1
        if p["outcome"]=="established_certified":
            proof_bytes.append(len(json.dumps(p["positive_trace"]).encode()))
        actions+=p["rule_applications"]
        passes+=p["stop_passes"]
        head_calls+=p["rule_head_forward_calls"]
        proof_ms+=p["proof_model_forward_ms"]
        stop_ms+=p["stop_model_forward_ms"]
        pos_ms+=p["positive_checker_ms"]
        neg_ms+=p["negative_checker_ms"]
    positives=sum(t["label"]=="established" for t in tasks)
    negatives=len(tasks)-positives
    pos=tally["established","established_certified"]
    neg=tally["refuted","refuted_certified"]
    accepted=pos+neg
    return {
       "n":len(tasks),
       "positive_certified":pos,
       "negative_certified":neg,
       "positive_certified_recall":round(pos/positives,6),
       "negative_certified_recall":round(neg/negatives,6),
       "combined_certified_coverage":round(accepted/len(tasks),6),
       "false_certificates":0,
       "abstained":len(tasks)-accepted,
       "provisional_positives_suppressed":sum(
         p["fc04_preproof_outcome"]=="established_provisional" and
         p["outcome"]!="established_certified" for p in results),
       "mean_stop_passes":round(passes/len(tasks),6),
       "mean_rule_applications":round(actions/len(tasks),6),
       "mean_rule_head_calls":round(head_calls/len(tasks),6),
       "total_recurrent_cell_invocations":passes+4*head_calls,
       "negative_exact_checker_ms_per_case":round(neg_ms/len(tasks),6),
       "positive_exact_checker_ms_per_case":round(pos_ms/len(tasks),6),
       "model_forward_ms_per_case":round((stop_ms+proof_ms)/len(tasks),6),
       "mean_proof_trace_bytes":round(sum(proof_bytes)/len(proof_bytes),3) if proof_bytes else None,
       "certificate_per_1000_cell_calls":round(1000*accepted/(passes+4*head_calls),6)
           if passes+4*head_calls else None,
       "outcome_counts":dict(Counter(p["outcome"] for p in results)),
    }


def frozen_train(fc02_root:Path,out:Path):
    """Isolated job can see FC-02 train/dev but cannot import new FC-05 cases."""
    torch.set_num_threads(2)
    train,dev,h,verify=fc04.load_train_only(fc02_root)
    out.mkdir(parents=True,exist_ok=True)
    manifest={"phase":"train_and_calibrate_fc02_only",
              "protocol_sha":PREREG_SHA,
              "fc02_data_sha256":h,
              "fc05_test_accessible":False,
              "base":"FC04 original tied LoopReach4",
              "heads":[]}
    for seed in STUDENT_SEEDS:
        base,base_meta=fc03._train_one(train,BASE_MODEL,seed)
        calibrate=fc04.calibrate_threshold(base,dev,verify)
        head,head_meta=train_head(base,train,seed)
        tag=f"fc05_s{seed}"
        base_path=out/(tag+"_base.pt")
        head_path=out/(tag+"_head.pt")
        torch.save(base.state_dict(),base_path)
        torch.save(head.state_dict(),head_path)
        info={**base_meta,**head_meta,
              "tag":tag,"seed":seed,
              "base_sha256":hashlib.sha256(base_path.read_bytes()).hexdigest(),
              "head_sha256":hashlib.sha256(head_path.read_bytes()).hexdigest(),
              "threshold":calibrate["selected_threshold"],
              "threshold_grid_scores":calibrate["all_grid_scores"]}
        manifest["heads"].append(info)
        print("FROZEN_PROOF_HEAD",json.dumps({key:info[key] for key in (
            "tag","base_sha256","head_sha256","training_only_positive_prefix_states",
            "head_parameter_count","threshold")},sort_keys=True),flush=True)
    file=out/"train_manifest.json"
    file.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("TRAIN_MANIFEST",file)


def heldout_test(root:Path,weights:Path,out:Path):
    torch.set_num_threads(2)
    manifest=json.loads((weights/"train_manifest.json").read_text())
    if manifest["protocol_sha"]!=PREREG_SHA or manifest["fc02_data_sha256"]!=fc03.FC02_DATA_SHA:
        raise AssertionError("mismatched training source or preregistration")
    if manifest.get("fc05_test_accessible") is not False:
        raise AssertionError("training saw test material")
    sys.path.insert(0,str(root.resolve()))
    from research.refutation_bench.fc05 import generator as fc05
    from research.refutation_bench.fc05 import proofs
    from research.refutation_bench.formal_curriculum import generator as fc02
    from research.refutation_bench.certificates import verify_countermodel
    cases=fc05.generate_tests()
    digest=hashlib.sha256(b"".join(fc02.canonical_bytes(t)+b"\n" for t in cases)).hexdigest()
    if digest!=FC05_DATA_SHA:
        raise AssertionError("test dataset hash does not match frozen source: "+digest)
    out.mkdir(parents=True,exist_ok=True)
    scores=[]
    accepted_proofs=[]
    for meta in manifest["heads"]:
        seed=meta["seed"];tag=meta["tag"]
        base_path=weights/(tag+"_base.pt")
        head_path=weights/(tag+"_head.pt")
        if hashlib.sha256(base_path.read_bytes()).hexdigest()!=meta["base_sha256"]:
            raise AssertionError("base checkpoint changed")
        if hashlib.sha256(head_path.read_bytes()).hexdigest()!=meta["head_sha256"]:
            raise AssertionError("proof head checkpoint changed")
        model=fc03.make_model(BASE_MODEL)
        model.load_state_dict(torch.load(base_path,weights_only=True,map_location="cpu"))
        model.eval()
        head=RuleHead()
        head.load_state_dict(torch.load(head_path,weights_only=True,map_location="cpu"))
        head.eval()
        theta=meta["threshold"]
        for split in ("test_same","test_deep"):
            tasks=[t for t in cases if t["split"]==split]
            for method in RULE_METHODS:
                predictions=[release_certificates(model,head,t,theta,method,
                                  verify_countermodel,proofs.verify_trace) for t in tasks]
                score=_score(tasks,predictions)
                file=out/f"{tag}_{split}_{method}.jsonl"
                file.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in predictions))
                score.update(tag=tag,seed=seed,split=split,
                             rule_method=method,theta=theta,
                             predictions_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                scores.append(score)
                print("PROOFS",json.dumps({key:score[key] for key in (
                    "tag","split","rule_method","positive_certified",
                    "negative_certified","false_certificates",
                    "abstained","mean_rule_applications",
                    "total_recurrent_cell_invocations")},sort_keys=True),flush=True)
                if method=="learned_trace":
                    for task,answer in zip(tasks,predictions):
                        if answer["outcome"]=="established_certified":
                            accepted_proofs.append((task,answer["positive_trace"]))
    lean=out/"FC05_model_proposed.lean"
    lean.write_text(proofs.lean_document(accepted_proofs))
    n=len(accepted_proofs)
    primary=[]
    for seed in STUDENT_SEEDS:
        for split in ("test_same","test_deep"):
            chosen=next(s for s in scores if s["seed"]==seed and s["split"]==split
                        and s["rule_method"]=="learned_trace")
            random_score=next(s for s in scores if s["seed"]==seed and s["split"]==split
                        and s["rule_method"]=="random_legal")
            goal_score=next(s for s in scores if s["seed"]==seed and s["split"]==split
                        and s["rule_method"]=="immediate_goal_random")
            primary.append({
                "seed":seed,"split":split,
                "model_proof_count":chosen["positive_certified"],
                "model_beats_both_heuristics":chosen["positive_certified"]>
                     max(random_score["positive_certified"],goal_score["positive_certified"]),
                "no_false_certificates":chosen["false_certificates"]==0,
            })
    report={
        "experiment":"FC-05",
        "protocol_sha":PREREG_SHA,
        "source_commit":FC05_SOURCE_SHA,
        "test_data_sha256":digest,
        "first_frozen_trial":True,
        "source_is_public_exact_Horn":True,
        "model_uses_test_oracle_for_action_selection":False,
        "positive_lean_proof_count":n,
        "positive_proofs_sha256":hashlib.sha256(lean.read_bytes()).hexdigest(),
        "criterion_at_least_one_positive_per_split_seed":all(x["model_proof_count"]>0
                                      and x["no_false_certificates"] for x in primary),
        "learned_policy_beats_both_heuristics_every_split":all(
                                x["model_beats_both_heuristics"] for x in primary),
        "source_test_size":len(cases),
        "positive_oracle_accuracy":1.0,
        "per_condition":scores,
        "primary":primary,
        "remarks":[
            "Model chooses each rule. Python checker cannot repair missing actions.",
            "Accepted positive Lean proofs use the exact chosen traces, and kernel compilation happens in CI.",
            "Training-only BFS is never called on test inputs.",
            "A negative certificate is exact propositional countermodel, not a refutation of a named conjecture.",
        ]
    }
    file=out/"FC05_report.json"
    file.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("FINAL_CRITERION",json.dumps({key:report[key] for key in (
          "criterion_at_least_one_positive_per_split_seed",
          "learned_policy_beats_both_heuristics_every_split",
          "positive_lean_proof_count")},sort_keys=True))
    print("FC05_REPORT",file)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","eval"))
    p.add_argument("--fc02-root",type=Path)
    p.add_argument("--fc05-root",type=Path)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="train":
        if args.fc02_root is None or args.fc05_root is not None:
            raise ValueError("training cannot access new test corpus")
        frozen_train(args.fc02_root,args.out)
    else:
        if args.fc05_root is None or args.weights is None:
            raise ValueError("evaluate requires frozen weights and new source")
        heldout_test(args.fc05_root,args.weights,args.out)


if __name__=="__main__":
    main()
