"""LRF-02: prospective native-adjudicated generator and critic training."""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent.parent/"lean_reasoner"))
import reward as R
import proof_data as P

SEEDS=(171,173)
BUDGETS=(1,2,4,8)
POLICIES=("first","generator_first","random","critic","exact")
SPLITS=(("train",64,False),("cal",16,False),("test_standard",24,False),("test_deep",24,True))
STEPS=10
GEN_DIM=14
CRIT_DIM=26
HIST_TASKS="7e15b7d055b49c829b901a43adfddb868e51fee5fc4293a9a313b970c9083d09"
HIST_SOURCE="1c89d80a41c98d560598d7dc6c4e06b827474e66e0ea3c3d7919635ddfb06fab"

def freeze(out):
    old,manifest=P.build(out/"historical_disjointness")
    assert manifest["tasks_sha256"]==HIST_TASKS and manifest["lean_source_sha256"]==HIST_SOURCE
    used={t["structure_id"] for t in old}
    rng=random.Random(120201);rows=[];tries=0
    for split,n,deep in SPLITS:
        count=0
        while count<n:
            tries+=1
            assert tries<100000
            t=P.generate_one(rng,deep)
            if t is None or t["structure_id"] in used:continue
            used.add(t["structure_id"])
            t["split"]=split;t["id"]=f"{split}_{count:03d}"
            rows.append(t);count+=1
    assert len(rows)==128 and len(used)==1408
    (out/"prospective.jsonl").write_text("".join(json.dumps(t,sort_keys=True)+"\n" for t in rows))
    print("CORPUS",json.dumps({"families":len(rows),"attempts":tries,"disjoint":True}),flush=True)
    return rows

def distance(t):
    rev={}
    for ps,h in t["rules"]:
        rev.setdefault(h,[]).extend(ps)
    ds={t["goal"]:0};q=deque(ds)
    while q:
        h=q.popleft()
        for x in rev.get(h,[]):
            if x not in ds:ds[x]=ds[h]+1;q.append(x)
    return ds

def gen_features(t,known,actions,step):
    n=t["n"];m=max(1,len(t["rules"]));ds=distance(t)
    fan=Counter(x for ps,_ in t["rules"] for x in ps)
    result=[]
    for j in actions:
        ps,h=t["rules"][j]
        child=[(xs,y) for xs,y in t["rules"] if h in xs and y not in known]
        useful=[len(set(xs)-known-{h})==0 for xs,_ in child]
        result.append([float(h==t["goal"]),float(h in ds),ds.get(h,n)/n,len(ps)/n,
                       fan[h]/m,len(child)/m,sum(useful)/m,len(known)/n,len(actions)/m,
                       step/STEPS,(STEPS-step)/STEPS,h/n,
                       len(t["facts"])/n,len(t["rules"])/max(1,2*n)])
    assert all(len(x)==GEN_DIM for x in result)
    return result

class Ranker(nn.Module):
    def __init__(self,dim,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def action_groups(rows,seed):
    groups=[]
    for t in rows:
        if t["split"]!="train":continue
        rng=random.Random(seed+int(t["structure_id"][:12],16))
        for _ in range(2):
            known=set(t["facts"])
            for step in range(STEPS):
                if t["goal"] in known:break
                actions=P.legal(t,known);assert actions
                required=P.required(t,known)
                good=[j in required for j in actions];assert any(good)
                groups.append((gen_features(t,known,actions,step),good))
                j=rng.choice(actions) if rng.random()<.15 else rng.choice([j for j in actions if j in required])
                known.add(t["rules"][j][1])
    assert groups
    return groups

def fit_generator(model,groups,seed,epochs=10):
    rng=random.Random(seed);opt=torch.optim.AdamW(model.parameters(),lr=.003)
    history=[]
    for _ in range(epochs):
        order=list(range(len(groups)));rng.shuffle(order);losses=[]
        for off in range(0,len(order),64):
            batch=[groups[i] for i in order[off:off+64]]
            width=max(len(x[0]) for x in batch)
            x=torch.zeros(len(batch),width,GEN_DIM)
            valid=torch.zeros(len(batch),width,dtype=torch.bool)
            good=torch.zeros_like(valid)
            for i,(fs,ys) in enumerate(batch):
                x[i,:len(fs)]=torch.tensor(fs);valid[i,:len(fs)]=True;good[i,:len(fs)]=torch.tensor(ys)
            z=model(x)
            loss=(torch.logsumexp(z.masked_fill(~valid,-1e9),-1)-torch.logsumexp(z.masked_fill(~good,-1e9),-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    return history

@torch.no_grad()
def rollout(model,task,rng=None):
    t=R.canonical(task);known=set(t["facts"]);trace=[]
    for step in range(STEPS):
        if t["goal"] in known:break
        actions=P.legal(t,known)
        if not actions:break
        z=model(torch.tensor(gen_features(t,known,actions,step))).tolist()
        if rng is None:i=max(range(len(actions)),key=lambda i:z[i])
        else:
            m=max(z)
            i=rng.choices(list(range(len(actions))),weights=[math.exp(x-m) for x in z],k=1)[0]
        j=actions[i];trace.append(j);known.add(t["rules"][j][1])
    return trace

def pool(model,t,seed):
    c=R.canonical(t);first=c["facts"][0]
    assert c["goal"] not in c["facts"]
    key=int(t["structure_id"][:12],16)+seed
    greedy=rollout(model,c)
    easy={**c,"goal":first}
    extra={**c,"facts":sorted(c["facts"]+[c["goal"]])}
    shortcut={**c,"rules":c["rules"]+[[[first],c["goal"]]]}
    entries=[("greedy",{"trace":greedy})]
    for j in range(2):
        entries.append((f"sample{j}",{"trace":rollout(model,c,random.Random(key+j))}))
    entries.extend([
        ("easier",{"claim":easy,"trace":rollout(model,easy)}),
        ("extra_hyp",{"claim":extra,"trace":rollout(model,extra)}),
        ("extra_rule",{"claim":shortcut,"trace":rollout(model,shortcut)}),
        ("axiom",{"mode":"fixture_axiom"}),
        ("alias",{"mode":"fixture_alias"}),
        ("unused",{"trace":greedy,"unused_axiom":True}),
    ])
    seen=set();unique=[]
    for tag,candidate in entries:
        h=R.sha(json.dumps(candidate,sort_keys=True))
        if h not in seen:
            seen.add(h);unique.append((tag,candidate))
    return unique,len(entries)-len(unique)

def rule_set(t):return {(tuple(ps),h) for ps,h in t["rules"]}
def jaccard(x,y):return len(x&y)/max(1,len(x|y))

def critic_features(t,candidate):
    a=R.canonical(t);b=R.canonical(candidate.get("claim",t))
    n=a["n"];ar=rule_set(a);br=rule_set(b);af=set(a["facts"]);bf=set(b["facts"])
    trace=candidate.get("trace",[])
    known=set(bf);steps=0
    for j in trace:
        ps,h=b["rules"][j]
        if not set(ps)<=known or h in known:break
        known.add(h);steps+=1
    mode=candidate.get("mode","trace");ds=distance(b)
    # Features summarize the two public interfaces and executed trace,
    # without a native verdict, full equality flag or proof-completion flag.
    values=[n/128,a["goal"]/n,b["goal"]/n,len(af)/n,len(bf)/n,len(ar)/(2*n),len(br)/(2*n),
            jaccard(af,bf),jaccard(ar,br),len(af-bf)/n,len(bf-af)/n,
            len(ar-br)/(2*n),len(br-ar)/(2*n),len(trace)/STEPS,steps/STEPS,len(known)/n,
            len(known&af)/n,len(known&bf)/n,
            sum(x in ds for x in known)/n,
            float(a["goal"] in af),float(b["goal"] in bf),
            float(mode=="trace"),float(mode=="fixture_axiom"),float(mode=="fixture_alias"),
            float(candidate.get("unused_axiom",False)),
            sum(h==b["goal"] for _,h in b["rules"])/max(1,len(b["rules"]))]
    assert len(values)==CRIT_DIM
    return values

def verify_pool(model,t,seed,stage,out,lean):
    entries,duplicates=pool(model,t,seed)
    rows=[]
    for tag,candidate in entries:
        receipt=R.evaluate_candidate(t,candidate,out/"native"/str(seed)/stage/t["id"]/tag,lean)
        label=receipt["reward"]
        if label==1:
            assert R.canonical(candidate.get("claim",t))==R.canonical(t)
            assert candidate.get("mode","trace")=="trace"
            assert receipt["kernel"] and receipt["foundation"] and receipt["contract"]
        elif label==0:
            assert receipt["kernel"] and (receipt["foundation"] is False or receipt["contract"] is False)
        if label is None:
            assert receipt["classification"]=="incomplete_trace_censored", (t["id"],tag,receipt["classification"])
        row={"tag":tag,"candidate":candidate,"receipt":receipt}
        rows.append(row)
    return rows,duplicates

def feedback_groups(t,rows):
    """Only actual native-certified original-goal trajectories add labels."""
    out=[]
    for r in rows:
        if r["receipt"]["reward"]!=1:continue
        trace=r["candidate"]["trace"];known=set(t["facts"])
        for step,j in enumerate(trace):
            actions=P.legal(t,known);assert j in actions
            remaining=set(trace[step:])
            good=[k in remaining for k in actions];assert any(good)
            out.append((gen_features(t,known,actions,step),good))
            known.add(t["rules"][j][1])
        assert t["goal"] in known
    return out

def fit_critic(model,data,seed):
    assert data and any(y for _,y in data) and any(not y for _,y in data)
    rng=random.Random(seed);opt=torch.optim.AdamW(model.parameters(),lr=.003)
    xs=torch.tensor([x for x,_ in data]);ys=torch.tensor([float(y) for _,y in data])
    hist=[]
    for _ in range(10):
        ids=list(range(len(data)));rng.shuffle(ids);losses=[]
        for off in range(0,len(ids),64):
            ix=ids[off:off+64]
            z=model(xs[ix]);loss=F.binary_cross_entropy_with_logits(z,ys[ix])
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        hist.append(sum(losses)/len(losses))
    return hist

def exact_screen(t,c):
    if c.get("mode","trace")!="trace":return False
    if R.canonical(c.get("claim",t))!=R.canonical(t):return False
    try:return R.checked_trace(R.canonical(t),c["trace"]) is not None
    except (KeyError,ValueError,TypeError):return False

def verify_exact_predictions(t,rows):
    for row in rows:
        # All proposals are from the restricted renderer. Censored traces
        # are not positive; known target certificates should agree.
        if exact_screen(t,row["candidate"])!=(row["receipt"]["reward"]==1):
            raise AssertionError((t["id"],row["tag"],row["receipt"]["classification"]))

@torch.no_grad()
def rank(t,rows,critic,seed,policy):
    ix=list(range(len(rows)))
    if policy=="first":return ix[:1]
    if policy=="generator_first":return ix
    if policy=="random":
        random.Random(int(t["structure_id"][:12],16)+seed).shuffle(ix);return ix
    if policy=="exact":
        return sorted(ix,key=lambda i:not exact_screen(t,rows[i]["candidate"]))
    assert policy=="critic"
    scores=critic(torch.tensor([critic_features(t,r["candidate"]) for r in rows])).tolist()
    return sorted(ix,key=lambda i:-scores[i])

@torch.no_grad()
def evaluate(seed,initial,gen,critic,rows,out,lean):
    scores=[];all_receipts=[]
    for split in ("cal","test_standard","test_deep"):
        subset=[t for t in rows if t["split"]==split]
        stats=Counter();policy_counts={(p,b):Counter() for p in POLICIES for b in BUDGETS}
        for t in subset:
            rs,duplicates=verify_pool(gen,t,seed,"final_eval",out,lean)
            verify_exact_predictions(t,rs)
            initial_receipt=R.evaluate_candidate(t,{"trace":rollout(initial,t)},out/"native"/str(seed)/"initial_eval"/t["id"],lean)
            assert initial_receipt["reward"]==1 or initial_receipt["classification"]=="incomplete_trace_censored"
            stats["n"]+=1;stats["initial_greedy"]+=initial_receipt["reward"]==1
            stats["final_greedy"]+=rs[0]["receipt"]["reward"]==1
            stats["duplicates"]+=duplicates;stats["proposals"]+=len(rs)
            stats["native_calls"]+=sum(r["receipt"]["compiler_calls"] for r in rs)+initial_receipt["compiler_calls"]
            stats["attainable"]+=any(r["receipt"]["reward"]==1 for r in rs)
            logits=critic(torch.tensor([critic_features(t,r["candidate"]) for r in rs])).tolist()
            for z,r in zip(logits,rs):
                y=r["receipt"]["reward"]
                stats["completed_deceptive"]+=y==0
                stats["false_approval"]+=y==0 and z>=0.
                stats["censored"]+=y is None
                stats["native_positive"]+=y==1
                stats["false_accepted"]+=y==1 and not exact_screen(t,r["candidate"])
                r["critic_logit"]=z
            for p in POLICIES:
                order=rank(t,rs,critic,seed,p)
                for b in BUDGETS:
                    attempts=[]
                    for i in order[:b]:
                        attempts.append(i)
                        if rs[i]["receipt"]["reward"]==1:break
                    c=policy_counts[p,b]
                    c["n"]+=1;c["solved"]+=bool(attempts and rs[attempts[-1]]["receipt"]["reward"]==1)
                    c["proposals"]+=len(attempts)
                    c["hypothetical_native_calls"]+=sum(rs[i]["receipt"]["compiler_calls"] for i in attempts)
            all_receipts.append({"seed":seed,"split":split,"id":t["id"],"initial_greedy":initial_receipt,"pool":rs})
        result={"seed":seed,"split":split,**stats}
        print("POOL",json.dumps(result),flush=True)
        for (p,b),cs in policy_counts.items():
            row={"seed":seed,"split":split,"policy":p,"budget":b,**cs,
                 "attainable":stats["attainable"],"false_approval":stats["false_approval"],
                 "completed_deceptive":stats["completed_deceptive"]}
            scores.append(row);print("SCORE",json.dumps(row),flush=True)
    (out/f"eval-{seed}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in all_receipts))
    return scores

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--lean",default="lean")
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    version=subprocess.check_output([a.lean,"--version"],text=True).strip();assert "4.19.0" in version
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=freeze(out);all_scores=[];fits=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        gen=Ranker(GEN_DIM,16)
        base_groups=action_groups(rows,seed)
        initial_loss=fit_generator(gen,base_groups,seed)
        initial=copy.deepcopy(gen).eval()
        extra=[];labels=[];counts=Counter();train_receipts=[]
        for t in rows:
            if t["split"]!="train":continue
            rs,duplicates=verify_pool(initial,t,seed,"train",out,a.lean);verify_exact_predictions(t,rs)
            extra.extend(feedback_groups(t,rs))
            for row in rs:
                y=row["receipt"]["reward"]
                counts["positive"]+=y==1;counts["negative"]+=y==0;counts["censored"]+=y is None
                counts["native_calls"]+=row["receipt"]["compiler_calls"]
                if y is not None:labels.append((critic_features(t,row["candidate"]),y))
            counts["duplicates"]+=duplicates
            train_receipts.append({"id":t["id"],"pool":rs})
        assert extra and labels
        (out/f"train-{seed}.jsonl").write_text("".join(json.dumps(x)+"\n" for x in train_receipts))
        torch.manual_seed(seed+1000);critic=Ranker(CRIT_DIM,32)
        critic_loss=fit_critic(critic,labels,seed+1000)
        final_loss=fit_generator(gen,base_groups+extra,seed+2000)
        checkpoint=out/f"seed-{seed}.pt"
        torch.save({"generator_initial":initial.state_dict(),"generator_final":gen.state_dict(),"critic":critic.state_dict()},checkpoint)
        fit={"seed":seed,"generator_parameters":sum(p.numel() for p in gen.parameters()),
             "critic_parameters":sum(p.numel() for p in critic.parameters()),
             "base_groups":len(base_groups),"verified_feedback_groups":len(extra),"critic_labels":len(labels),
             "train_counts":dict(counts),"initial_loss":initial_loss,"final_loss":final_loss,
             "critic_loss":critic_loss,"sha256":R.sha(checkpoint.read_bytes())}
        fits.append(fit);print("FIT",json.dumps(fit),flush=True)
        all_scores+=evaluate(seed,initial,gen.eval(),critic.eval(),rows,out,a.lean)
    primary=[s for s in all_scores if s["split"].startswith("test") and s["policy"]=="critic" and s["budget"]==2]
    assert len(primary)==4
    passed=all(s["attainable"]>0 and 10*s["solved"]>=9*s["attainable"]
               and 20*s["false_approval"]<=s["completed_deceptive"] for s in primary)
    gate={"passed":passed,"primary":primary,"fits":fits,"native_toolchain":version}
    (out/"scores.json").write_text(json.dumps(all_scores,indent=2))
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print("GATE",json.dumps({"passed":passed,"primary":primary}),flush=True)

if __name__=="__main__":main()
