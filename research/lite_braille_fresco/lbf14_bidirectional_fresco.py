"""LBF-14: exact counterfactual bidirectional layer decisions."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,dataclass
from pathlib import Path
import random
import time
import math

import torch
import torch.nn.functional as F
import lbf13_budget_fresco as prior

base=prior.base
WORK=128
DEPTH=16
SEED=20261209
COUNTS={"train":48,"calibration":32,"ordinary":16,"ood":16}
SEEDS=(107,109)
STREAMS=(17,31,47,61,83)
STEPS=360
MODES=("normal","zero","roll")
ARMS=("standard","opened")


@dataclass
class State:
    fronts:list
    parent:list
    dist:list
    work:int=0
    expanded:int=0
    path:list|None=None

    @classmethod
    def start(cls,c):
        return cls([[c.source],[c.target]],[{c.source:None},{c.target:None}],
                   [{c.source:0},{c.target:0}])

    def live(self,budget):
        return bool(self.path is None and self.work<budget and all(self.fronts))

    def depth(self,side):
        return self.dist[side][self.fronts[side][0]]


def smaller(s):
    return 0 if len(s.fronts[0])<=len(s.fronts[1]) else 1


def meeting_path(rg,c,s,nxt):
    left=rg._path(s.parent[0],nxt)
    back=rg._path(s.parent[1],nxt)
    right=[];word=nxt
    for old in reversed(back):
        a,b=c.rules[old.rule]
        if old.direction:a,b=b,a
        p=old.position
        dest=word[:p]+a+word[p+len(b):]
        inv=rg.Move(old.rule,1-old.direction,p,dest)
        word=rg.apply_move(c,word,inv);right.append(inv)
    return left+right


def advance(rg,c,s,side,budget=WORK,depth=DEPTH):
    """Mutate one layer, preserving pinned BFS word/move ordering."""
    if side not in (0,1) or not s.live(budget):
        raise ValueError("invalid layer intervention")
    new=[]
    for word in s.fronts[side]:
        if s.work>=budget:break
        g=s.dist[side][word]
        if g>=depth:continue
        s.expanded+=1
        for move in rg.neighbors(c,word):
            if s.work>=budget:break
            s.work+=1
            nxt=move.successor
            if nxt in s.dist[side]:continue
            s.dist[side][nxt]=g+1
            s.parent[side][nxt]=(word,move);new.append(nxt)
            if nxt in s.dist[1-side] and g+1+s.dist[1-side][nxt]<=depth:
                s.path=meeting_path(rg,c,s,nxt)
                rg.replay(c,s.path)
                return
    s.fronts[side]=new


def continue_smaller(rg,c,s,budget=WORK,depth=DEPTH):
    while s.live(budget):
        advance(rg,c,s,smaller(s),budget,depth)
    return s


def forced(rg,c,s,side,budget=WORK,depth=DEPTH):
    """Fixed continuation utility on a cloned complete public search state."""
    r=budget-s.work
    z=deepcopy(s)
    advance(rg,c,z,side,budget,depth)
    continue_smaller(rg,c,z,budget,depth)
    used=z.work-s.work
    if not 0<=used<=r:raise ValueError("unaccounted continuation work")
    if z.path is not None:
        rg.replay(c,z.path)
        if len(z.path)>depth:raise ValueError("path cap exceeded")
    return (used if z.path is not None else r+1),used,z.path


def features(rg,c,s,budget=WORK):
    """Summaries of discovered words only; charge Hamming computations separately."""
    a,b=s.fronts
    if not a or not b:raise ValueError("empty frontier observation")
    length=len(c.source)
    cross=[[rg.hamming(x,y) for y in b] for x in a]
    nearest=([min(row) for row in cross],[min(row[j] for row in cross) for j in range(len(b))])
    raw=[]
    for i,front in enumerate(s.fronts):
        target=c.target if i==0 else c.source
        h=[rg.hamming(w,target) for w in front]
        raw.append([len(front),len(s.dist[i]),s.depth(i),min(h),
                    round(1000*sum(h)/len(h)),round(1000*sum(nearest[i])/len(nearest[i]))])
    def norm(v):
        return [v[0]/(WORK+1),v[1]/(WORK+1),v[2]/DEPTH,
                v[3]/length,v[4]/(1000*length),v[5]/(1000*length)]
    rem=budget-s.work
    xs=[norm(raw[i])+norm(raw[1-i])+[rem/WORK,1.] for i in (0,1)]
    p=rg.public_packet(c,c.source,DEPTH)+f"|W:{rem}"
    menu=[f"side{i}:"+",".join(map(str,v)) for i,v in enumerate(raw)]
    if len(p.encode())>384 or any(len(x.encode())>384 for x in menu):
        raise ValueError("public encoding truncated")
    return p,menu,xs,raw,len(a)*len(b)+len(a)+len(b)


def result(rg,c,s,policy,calls,feature_work,decisions,budget=WORK,depth=DEPTH):
    r=rg._result(c,policy,s.path,s.work,s.expanded,calls,budget,depth)
    return {**r,"hamming_evaluations":feature_work,"layer_decisions":decisions}


def run(rg,c,policy,ranker=None,seed=0,budget=WORK,depth=DEPTH):
    s=State.start(c);calls=feature_work=decisions=0;rng=random.Random(seed)
    while s.live(budget):
        decisions+=1
        if policy=="bidir":side=smaller(s)
        elif policy=="depth":
            side=min((0,1),key=lambda i:(s.depth(i),len(s.fronts[i]),i))
        elif policy=="uniform":side=rng.randrange(2)
        elif policy in ("meeting","ridge","learned"):
            p,menu,xs,raw,work=features(rg,c,s,budget);feature_work+=work
            if policy=="meeting":
                side=min((0,1),key=lambda i:(raw[i][5],len(s.fronts[i]),i))
            else:
                if ranker is None:raise ValueError("missing ranker")
                score=ranker(c,p,menu,xs,budget-s.work,decisions)
                if len(score)!=2 or not all(math.isfinite(x) for x in score):
                    raise ValueError("invalid scores")
                calls+=1
                side=min((0,1),key=lambda i:(-score[i],len(s.fronts[i]),i))
        else:raise ValueError("unknown policy")
        advance(rg,c,s,side,budget,depth)
    return result(rg,c,s,policy,calls,feature_work,decisions,budget,depth)


def teacher(rg,suite,out):
    cases,m,refs=rg.load_split(suite,"train",witnesses=True)
    if len(cases)!=COUNTS["train"] or m["seed"]!=SEED:
        raise ValueError("incorrect training roster")
    out.mkdir()
    cert=base.certify(rg,[(c,refs[c.root],"train_"+c.root) for c in cases],out/"references")
    rows=[];audits=[];oracle_work=feature_work=0
    mixed=ties=0
    for c in cases:
        s=State.start(c);ordinal=0
        while s.live(WORK):
            ordinal+=1
            r=WORK-s.work
            p,menu,xs,raw,f=features(rg,c,s);feature_work+=f
            cs=[];successes=[]
            for i in (0,1):
                cost,used,path=forced(rg,c,s,i)
                oracle_work+=used;cs.append(cost);successes.append(path is not None)
                if path is not None and sum(x[0].family==c.family for x in audits)<12:
                    audits.append((c,path,f"counterfactual_{len(audits)}"))
            mixed+=int(successes[0]!=successes[1]);ties+=int(cs[0]==cs[1])
            rows.append({"root":c.root,"lineage":c.lineage,"ordinal":ordinal,"packet":p,
                         "menu":menu,"x":xs,"remaining":r,"cost":cs,
                         "target":[1-cost/(r+1) for cost in cs]})
            advance(rg,c,s,smaller(s))
        pinned=rg.search(c,"bidir",budget=WORK,depth=DEPTH)
        actual=rg._result(c,"bidir",s.path,s.work,s.expanded,0,WORK,DEPTH)
        if pinned!=actual:raise ValueError("teacher policy differs from pinned bidir")
    if len(audits)!=24:raise ValueError("counterfactual certificate census incomplete")
    audit=base.certify(rg,audits,out/"counterfactuals")
    base.write_jsonl(out/"frames.jsonl",rows)
    receipt={"roots":len(cases),"frames":len(rows),"mixed_closure":mixed,"cost_ties":ties,
             "oracle_work":oracle_work,"hamming_evaluations":feature_work,
             "mean_abs_utility_gap":sum(abs(r["target"][0]-r["target"][1]) for r in rows)/len(rows),
             "reference_certificates":cert["compiled_axiom_free"],
             "counterfactual_certificates":audit["compiled_axiom_free"],
             "frames_sha256":base.file_sha(out/"frames.jsonl")}
    base.write(out/"manifest.json",receipt)
    return rows,receipt


def ridge_fit(rows):
    x=torch.tensor([v for r in rows for v in r["x"]],dtype=torch.float64)
    y=torch.tensor([v for r in rows for v in r["target"]],dtype=torch.float64)
    penalty=torch.eye(x.shape[1],dtype=x.dtype)*.01;penalty[-1,-1]=0
    w=torch.linalg.solve(x.T@x+penalty,x.T@y)
    if not torch.isfinite(w).all():raise ValueError("nonfinite ridge weights")
    return w


def new_model(seed,arm):
    if arm not in ARMS:raise ValueError("unknown gate arm")
    torch.manual_seed(seed)
    m=base.PlasticProofTransformer(base.Config(plan="b",width=32,heads=4,max_bytes=384)).cpu().float()
    if m.parameter_count()!=57475:raise ValueError("parameter count changed")
    baseline=base._model_fingerprint(m)
    for gate in (m.gate_a,m.gate_b):
        if not torch.all(gate.gate.bias==-6):raise ValueError("unexpected original gate bias")
        if arm=="opened":
            with torch.no_grad():gate.gate.bias.fill_(-2.)
    return m,baseline


def fit(rows,seed,arm,out):
    m,baseline=new_model(seed,arm)
    initial=base._model_fingerprint(m)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    rng=random.Random(seed);byroot=defaultdict(list)
    for r in rows:byroot[r["root"]].append(r)
    roots=sorted(byroot);draws=[];trace=[];t0=time.monotonic()
    for i in range(STEPS):
        if time.monotonic()-t0>360:raise TimeoutError("CPU fit cap")
        row=rng.choice(byroot[rng.choice(roots)])
        pred=m(row["packet"],row["menu"],depth=2,checks_remaining=8,return_all=True,middle_order="ab")
        target=torch.tensor(row["target"],dtype=torch.float32)
        loss=torch.stack([F.mse_loss(x["outcome"].sigmoid(),target) for x in pred]).mean()
        if not torch.isfinite(loss):raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(m.parameters(),1.)):
            raise ValueError("nonfinite gradient")
        opt.step()
        draw={"step":i+1,"root":row["root"],"ordinal":row["ordinal"]}
        draws.append(draw);trace.append({**draw,"loss":float(loss.detach())})
    m.eval()
    for p in m.parameters():p.requires_grad_(False)
    out.mkdir()
    torch.save({"weights":m.state_dict(),"config":asdict(m.cfg),"arm":arm,"seed":seed},out/"checkpoint.pt")
    base.write_jsonl(out/"draws.jsonl",draws);base.write_jsonl(out/"trace.jsonl",trace)
    receipt={"seed":seed,"arm":arm,"parameters":57475,"steps":STEPS,
             "shared_initial":baseline,"initial":initial,"final":base._model_fingerprint(m),
             "draws":base.file_sha(out/"draws.jsonl"),"trace":base.file_sha(out/"trace.jsonl"),
             "checkpoint":base.file_sha(out/"checkpoint.pt")}
    base.write(out/"manifest.json",receipt)
    return m,receipt


def probe_summary(raw):
    out={}
    for arm in ARMS:
        rr=[r for r in raw if r["seed"].endswith(":"+arm)]
        if not rr:raise ValueError("missing gate probe arm")
        def gate_mean(name):
            vals=[float(r["normal"][name+"_strength"][1]["out"].sigmoid().mean()) for r in rr]
            return sum(vals)/len(vals)
        m={"decisions":len(rr),"pass2_gate_a_mean":gate_mean("gate_a"),
           "pass2_gate_b_mean":gate_mean("gate_b")}
        for mode in ("zero","roll"):
            pairs=[(r["utility_logits"][1],r["alternatives"][mode]["utility_logits"][1]) for r in rr]
            m[mode+"_max_logit_delta"]=max(float((a-b).abs().max()) for a,b in pairs)
            m[mode+"_top_changes"]=sum(int(a.argmax())!=int(b.argmax()) for a,b in pairs)
        out[arm]=m
    return out


def campaign(rg,out):
    if out.exists():raise ValueError("fresh LBF-14 destination required")
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    suite=out/"suite";sm=rg.freeze(suite,SEED,COUNTS)
    if sm["counts"]!=COUNTS or sm["unique_rule_system_lineages"]!=sum(COUNTS.values()):
        raise ValueError("source roster changed")
    frames,tm=teacher(rg,suite,out/"teacher")
    w=ridge_fit(frames);base.write(out/"ridge.json",{"weights":w.tolist(),"lambda":.01})
    models={};receipts={}
    for seed in SEEDS:
        for arm in ARMS:
            k=(seed,arm);models[k],receipts[k]=fit(frames,seed,arm,out/f"{seed}_{arm}")
        for key in ("shared_initial","draws"):
            if receipts[(seed,"standard")][key]!=receipts[(seed,"opened")][key]:
                raise ValueError("unmatched gate initialization or training draws")
    cases,_=rg.load_split(suite,"calibration")
    if len(cases)!=COUNTS["calibration"] or {c.lineage for c in cases}&{r["lineage"] for r in frames}:
        raise ValueError("calibration roster changed")
    rows=[];raw=[];observations=[]
    def ridge_ranker(c,p,menu,xs,r,n):
        return (torch.tensor(xs,dtype=torch.float64)@w).tolist()
    controls=("bidir","depth","meeting","ridge","uniform")
    for c in cases:
        r=run(rg,c,"bidir")
        pinned=rg.search(c,"bidir",budget=WORK,depth=DEPTH)
        if any(r[k]!=v for k,v in pinned.items()):
            raise ValueError("calibration pinned bidir equivalence failed")
        rows.append({**r,"condition":"bidir","seed":None,"stream":None})
        for name in ("depth","meeting","ridge"):
            r=run(rg,c,name,ridge_ranker if name=="ridge" else None)
            rows.append({**r,"condition":name,"seed":None,"stream":None})
        for stream in STREAMS:
            r=run(rg,c,"uniform",seed=int(base.digest([stream,c.root])[:16],16))
            rows.append({**r,"condition":"uniform","seed":None,"stream":stream})
        for seed in SEEDS:
            for arm in ARMS:
                m=models[(seed,arm)];rec=receipts[(seed,arm)]
                for mode in MODES:
                    def ranker(c,p,menu,xs,r,n):
                        with torch.no_grad():
                            if mode=="normal" and n<=2:
                                return prior.probe(m,c,p,menu,DEPTH,r,f"{seed}:{arm}",n,
                                                   rec["checkpoint"],raw,observations)
                            return m(p,menu,depth=2,checks_remaining=8,middle_order="ab",
                                     message_mode=mode)[-1]["outcome"].tolist()
                    r=run(rg,c,"learned",ranker)
                    if r["model_calls"]!=r["layer_decisions"]:raise ValueError("unaccounted model calls")
                    rows.append({**r,"condition":arm+"_"+mode,"seed":seed,"stream":None})
    for k,m in models.items():
        if base._model_fingerprint(m)!=receipts[k]["final"]:raise ValueError("frozen weights changed")
    names=controls+tuple(a+"_"+m for a in ARMS for m in MODES)
    if len(rows)!=len(cases)*(4+len(STREAMS)+len(SEEDS)*len(ARMS)*len(MODES)):
        raise ValueError("incomplete frozen episode roster")
    lookup={c.root:c for c in cases};proofs=[]
    for i,row in enumerate(rows):
        if row["closed"]:
            rg.replay(lookup[row["root"]],row["path"])
            proofs.append((lookup[row["root"]],row["path"],f"policy_{i}"))
    n=len(proofs);_,_,refs=rg.load_split(suite,"calibration",witnesses=True)
    proofs.extend((c,refs[c.root],"reference_"+c.root) for c in cases)
    cert=base.certify(rg,proofs,out/"calibration_certificates")
    if cert["compiled_axiom_free"]!=n+len(cases):raise ValueError("certificate count mismatch")
    for i,row in enumerate(rows):
        row["native_certified"]=row["closed"] and f"policy_{i}" in cert["declarations"]
        row["penalized_work"]=row["work"] if row["native_certified"] else WORK+1
    if len(raw)!=len(observations) or len(raw)!=len({r["decision_id"] for r in raw}):
        raise ValueError("probe identity mismatch")
    base.write_jsonl(out/"episodes.jsonl",rows)
    base.write_jsonl(out/"observations.jsonl",observations)
    torch.save(raw,out/"raw_boundary_tensors.pt")
    encoded=[base.encode(o) for o in observations]
    base.write_jsonl(out/"encoded.jsonl",encoded)
    (out/"fresco.svg").write_text(base.fresco_svg(encoded[0]),encoding="utf-8")
    rates={};costs={};seed_rates={}
    for name in names:
        rr=[r for r in rows if r["condition"]==name]
        rates[name]=sum(r["native_certified"] for r in rr)/len(rr)
        costs[name]={"episodes":len(rr),"work":sum(r["work"] for r in rr),
                     "mean_penalized_work":sum(r["penalized_work"] for r in rr)/len(rr),
                     "expanded":sum(r["expanded"] for r in rr),"model_calls":sum(r["model_calls"] for r in rr),
                     "hamming_evaluations":sum(r["hamming_evaluations"] for r in rr)}
    for seed in SEEDS:
        seed_rates[str(seed)]={}
        for name in names[len(controls):]:
            rr=[r for r in rows if r["condition"]==name and r["seed"]==seed]
            seed_rates[str(seed)][name]=sum(r["native_certified"] for r in rr)/len(rr)
    def root_mean(c,name,key):
        rr=[r for r in rows if r["root"]==c.root and r["condition"]==name]
        expected=len(STREAMS) if name=="uniform" else 1 if name in controls else len(SEEDS)
        if len(rr)!=expected:raise ValueError("incomplete paired root")
        return sum(r[key] for r in rr)/len(rr)
    pairs={}
    for name in names:
        if name=="opened_normal":continue
        q=[root_mean(c,"opened_normal","native_certified")-root_mean(c,name,"native_certified") for c in cases]
        v=[root_mean(c,"opened_normal","penalized_work")-root_mean(c,name,"penalized_work") for c in cases]
        pairs["opened_minus_"+name]={"closure_mean":sum(q)/len(q),"closure_95":base.bootstrap(q),
                                     "cost_mean":sum(v)/len(v),"cost_95":base.bootstrap(v),
                                     "wins":sum(x>0 for x in q),"losses":sum(x<0 for x in q)}
    best=max(controls,key=lambda n:(rates[n],-costs[n]["mean_penalized_work"]))
    gain=rates["opened_normal"]-rates[best]
    saved=1-costs["opened_normal"]["mean_penalized_work"]/costs[best]["mean_penalized_work"]
    primary=(gain>=.05 and rates["opened_normal"]>=.4 or gain>=0 and saved>=.1) and all(
             x["opened_normal"]>=rates[best] for x in seed_rates.values())
    result={"schema":"lbf14.bidir_fresco.v1","source":base.UPSTREAM_COMMIT,"generator_seed":SEED,
            "suite_manifest_sha256":base.file_sha(suite/"manifest.json"),"training":tm,
            "fits":[receipts[k] for k in sorted(receipts)],"rates":rates,"seed_rates":seed_rates,
            "costs":costs,"paired":pairs,"strongest_control":best,"primary_screen":primary,
            "memory_screen":all(pairs["opened_minus_opened_"+m]["closure_mean"]>=.05 for m in ("zero","roll")),
            "probe_summary":probe_summary(raw),"policy_certificates":n,"reference_certificates":len(cases),
            "sealed_outcomes_opened":False,"scientific_eligible":False,"native_online_tactic_calls":0,
            "artifacts":{x:base.file_sha(out/x) for x in ("ridge.json","episodes.jsonl","observations.jsonl",
                         "raw_boundary_tensors.pt","encoded.jsonl","fresco.svg")}}
    base.write(out/"manifest.json",result)
    print("LBF14_RESULT",base.canonical(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();campaign(base.load_upstream(a.upstream),a.out)


if __name__=="__main__":main()
