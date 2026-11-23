"""LBF-13 fixed-policy budget utility; same-root matched frontier search."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import heapq
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

import lbf12_rewrite_fresco as base

BUCKETS=(1,4,8,16,32,64,128)
WORK=128
DEPTH=16
SEEDS=(107,109)
STREAMS=(17,31,47,61,83)
GENERATOR_SEED=20261207
COUNTS={"train":32,"calibration":24,"ordinary":16,"ood":16}
STEPS=360


def budget_bucket(b):
    if not 1<=b<=WORK:
        raise ValueError("budget outside registered range")
    return max(x for x in BUCKETS if x<=b)


def packet(rg,case,word,depth,budget):
    if budget not in BUCKETS or not 1<=depth<=DEPTH:
        raise ValueError("unregistered query")
    p=rg.public_packet(case,word,depth)+f"|W:{budget}"
    if len(p.encode())>384:
        raise ValueError("budget query would truncate public packet")
    return p


def force(rg,case,word,move,depth,budget=WORK):
    """Exact first-success work of the specified restart policy, or censored."""
    if not 1<=depth<=DEPTH or not 1<=budget<=WORK:
        raise ValueError("invalid forced query")
    child=rg.apply_move(case,word,move)
    if child==case.target:
        return 1,[asdict(move)],1
    if depth==1 or budget==1:
        return None,[],1
    start=replace(case,source=child).validate()
    res=rg.search(start,"astar",budget=budget-1,depth=depth-1)
    charged=1+res["work"]
    if not res["closed"]:
        return None,[],charged
    path=[asdict(move)]+res["path"]
    rg.replay(replace(case,source=word),path)
    if len(path)>depth or charged>budget:
        raise ValueError("forced continuation exceeded budget")
    return charged,path,charged


def labels(costs,b):
    return [int(c is not None and c<=b) for c in costs]


def teacher(rg,suite,out):
    cases,m,refs=rg.load_split(suite,"train",witnesses=True)
    if m["seed"]!=GENERATOR_SEED or len(cases)!=COUNTS["train"]:
        raise ValueError("wrong train roster")
    out.mkdir()
    receipt=base.certify(rg,[(c,refs[c.root],"train_"+c.root) for c in cases],out/"references")
    rows=[]; audits=[]; prefix_checks=0; oracle_work=0
    counts={b:{"mixed":0,"all_positive":0,"all_negative":0,"positive":0,"candidates":0} for b in BUCKETS}
    for c in cases:
        states=rg.replay(c,refs[c.root])
        for i,word in enumerate(states[:-1]):
            d=DEPTH-i
            menu=rg.neighbors(c,word)
            costs=[]
            for move in menu:
                cost,path,charged=force(rg,c,word,move,d)
                oracle_work+=charged
                costs.append(cost)
                if prefix_checks<32*len(BUCKETS):
                    for b in BUCKETS:
                        smaller,_,w=force(rg,c,word,move,d,b)
                        if smaller!=(cost if cost is not None and cost<=b else None):
                            raise ValueError("continuation prefix inconsistency")
                        oracle_work+=w
                        prefix_checks+=1
                if cost is not None and sum(x[0].family==c.family for x in audits)<12:
                    audits.append((replace(c,source=word),path,f"forced_{len(audits)}"))
            mask=[int(mv.successor==states[i+1]) for mv in menu]
            if not any(mask):
                raise ValueError("witness action absent")
            for b in BUCKETS:
                ys=labels(costs,b)
                s=counts[b]
                s["mixed"]+=int(0<sum(ys)<len(ys))
                s["all_positive"]+=int(all(ys))
                s["all_negative"]+=int(not any(ys))
                s["positive"]+=sum(ys);s["candidates"]+=len(ys)
            rows.append({"root":c.root,"lineage":c.lineage,"step":i,"word":word,
                         "depth":d,"packets":{str(b):packet(rg,c,word,d,b) for b in BUCKETS},
                         "menu":[rg.action_token(x) for x in menu],"costs":costs,"mask":mask})
    if prefix_checks!=32*len(BUCKETS) or not audits:
        raise ValueError("incomplete oracle audit")
    a=base.certify(rg,audits,out/"forced_audit")
    base.write_jsonl(out/"frames.jsonl",rows)
    result={"train_roots":len(cases),"frames":len(rows),"oracle_work":oracle_work,
            "candidates":sum(len(r["menu"]) for r in rows),
            "censored":sum(x is None for r in rows for x in r["costs"]),
            "prefix_checks":prefix_checks,"reference_certificates":receipt["compiled_axiom_free"],
            "forced_audit_certificates":a["compiled_axiom_free"],"budget_census":counts,
            "frames_sha256":base.file_sha(out/"frames.jsonl")}
    base.write(out/"manifest.json",result)
    return rows,result


def fit(rows,seed,arm,out):
    torch.manual_seed(seed)
    model=base.PlasticProofTransformer(base.Config(plan="b",width=32,heads=4,max_bytes=384)).cpu().float()
    if model.parameter_count()!=57475:
        raise ValueError("parameter budget changed")
    initial=base._model_fingerprint(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=random.Random(seed); byroot=defaultdict(list)
    for row in rows:
        byroot[row["root"]].append(row)
    roots=sorted(byroot); draws=[]; trace=[]; t0=time.monotonic()
    for step in range(STEPS):
        if time.monotonic()-t0>360:
            raise TimeoutError("registered fit cap reached")
        row=rng.choice(byroot[rng.choice(roots)]); b=rng.choice(BUCKETS)
        pred=model(row["packets"][str(b)],row["menu"],depth=2,
                   checks_remaining=min(8,row["depth"]),return_all=True,middle_order="ab")
        if arm=="utility":
            target=torch.tensor(labels(row["costs"],b),dtype=torch.float32)
            loss=torch.stack([F.binary_cross_entropy_with_logits(x["outcome"],target) for x in pred]).mean()
        elif arm=="witness":
            mask=torch.tensor(row["mask"],dtype=torch.bool)
            loss=torch.stack([torch.logsumexp(x["logits"],0)-torch.logsumexp(x["logits"][mask],0)
                              for x in pred]).mean()
        else:
            raise ValueError("unknown arm")
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):
            raise ValueError("nonfinite gradient")
        opt.step()
        draw={"step":step+1,"root":row["root"],"frame":row["step"],"budget":b}
        draws.append(draw);trace.append({**draw,"loss":float(loss.detach())})
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    out.mkdir()
    torch.save({"weights":model.state_dict(),"config":asdict(model.cfg),"arm":arm,"seed":seed},out/"checkpoint.pt")
    base.write_jsonl(out/"draws.jsonl",draws)
    base.write_jsonl(out/"trace.jsonl",trace)
    receipt={"seed":seed,"arm":arm,"parameters":57475,"steps":STEPS,"initial":initial,
             "final":base._model_fingerprint(model),"checkpoint":base.file_sha(out/"checkpoint.pt"),
             "draws":base.file_sha(out/"draws.jsonl"),"trace":base.file_sha(out/"trace.jsonl")}
    base.write(out/"manifest.json",receipt)
    return model,receipt


def search(rg,case,policy="astar",ranker=None,seed=0,budget=WORK,depth=DEPTH):
    """Keep primary bound and sibling order fixed in frontier-only arms."""
    if policy not in ("astar","uniform","frontier","sibling"):
        raise ValueError("unknown search treatment")
    if (ranker is not None)!=(policy in ("frontier","sibling")):
        raise ValueError("wrong ranker treatment")
    rng=random.Random(seed)
    queue=[(rg.lower_bound(case.source,case.target),0.,0,case.source,0)]
    parent={case.source:None}; best={case.source:0}
    serial=work=expanded=calls=0
    while queue and work<budget:
        _,_,_,word,g=heapq.heappop(queue)
        if g!=best[word] or g>=depth:
            continue
        expanded+=1
        menu=rg.neighbors(case,word)
        if not menu:continue
        h=[-rg.hamming(m.successor,case.target) for m in menu]
        if ranker is not None:
            learned=ranker(case,word,menu,depth-g,budget-work)
            calls+=1
            if len(learned)!=len(menu) or any(not math.isfinite(x) for x in learned):
                raise ValueError("nonfinite or incomplete learned score")
        else:
            learned=[rng.random() for _ in menu] if policy=="uniform" else h
        order=sorted(range(len(menu)),key=lambda j:(-(learned[j] if policy=="sibling" else h[j]),menu[j]))
        secondary=learned if policy in ("frontier","uniform") else h
        for j in order:
            if work>=budget:break
            move=menu[j];work+=1
            nxt=move.successor;newg=g+1
            if nxt==case.target:
                return rg._result(case,policy,rg._path(parent,word)+[move],work,expanded,calls,budget,depth)
            if newg>=best.get(nxt,10**9) or newg>=depth:continue
            best[nxt]=newg;parent[nxt]=(word,move);serial+=1
            heapq.heappush(queue,(newg+rg.lower_bound(nxt,case.target),-float(secondary[j]),serial,nxt,newg))
    return rg._result(case,policy,None,work,expanded,calls,budget,depth)


def probe(model,case,p,menu,d,b,seed,ordinal,checkpoint,raw,observations):
    """Archive full boundary tensors separately from a label-free decision fresco."""
    normal,bank=base.capture(model,p,menu,min(8,d),"normal")
    alternatives={mode:base.capture(model,p,menu,min(8,d),mode) for mode in ("zero","roll")}
    identity=base.digest([case.root,seed,ordinal,p,menu])
    record={"schema":base.SCHEMA_OBS,"root_id":"rewrite:"+identity,"lineage":case.lineage,
            "split":"calibration","source":{"kind":"ppt_replay","source_sha256":base.digest([p,menu,d,b]),
            "checkpoint_sha256":checkpoint},"candidates":list(menu),"visits":[]}
    for i,o in enumerate(normal):
        incoming=bank["gate_a"][i]["bank_in"]; coda=o["state"][0].detach().cpu()
        tiles=[]
        for j,role in enumerate(base.ROLES):
            val=base.rms(incoming) if role=="message" else base.rms(coda[j])
            tiles.append({"candidate_id":"shared","role":role,"frame":"memory" if role=="message" else "state",
                          "activation":val/(1+val),"polarity":0})
        for candidate,z in zip(menu,o["outcome"]):
            tiles.append({"candidate_id":candidate,"role":"value","frame":"action",
                          "activation":float(z.sigmoid()),"polarity":0})
        record["visits"].append({"attempt":1,"pass":i+1,"tiles":tiles})
        for mode,(alt,alt_bank) in alternatives.items():
            if i==0:
                for k in ("outcome","logits","state","features"):
                    torch.testing.assert_close(o[k],alt[i][k],rtol=1e-6,atol=1e-6)
            if i and mode=="roll":
                torch.testing.assert_close(incoming.norm(),alt_bank["gate_a"][i]["bank_in"].norm(),rtol=1e-6,atol=1e-6)
    base.validate_observation(record)
    observations.append(record)
    raw.append({"decision_id":record["root_id"],"root":case.root,"seed":seed,"packet":p,
                "menu":list(menu),"budget":b,"depth":d,"normal":dict(bank),
                "utility_logits":[o["outcome"].detach().cpu().clone() for o in normal],
                "alternatives":{m:{"bank":dict(ab),"utility_logits":[o["outcome"].detach().cpu().clone() for o in alt]}
                                for m,(alt,ab) in alternatives.items()}})
    return normal[-1]["outcome"].tolist()


def campaign(rg,out):
    if out.exists():raise ValueError("fresh LBF-13 destination required")
    out.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    suite=out/"suite";sm=rg.freeze(suite,GENERATOR_SEED,COUNTS)
    if sm["counts"]!=COUNTS or sm["unique_rule_system_lineages"]!=sum(COUNTS.values()):
        raise ValueError("source roster changed")
    frames,tm=teacher(rg,suite,out/"teacher")
    models={};receipts={}
    for seed in SEEDS:
        for arm in ("witness","utility"):
            key=(seed,arm)
            models[key],receipts[key]=fit(frames,seed,arm,out/f"{seed}_{arm}")
        for k in ("initial","draws"):
            if receipts[(seed,"witness")][k]!=receipts[(seed,"utility")][k]:
                raise ValueError("unmatched initialization or sampling")
    cases,_=rg.load_split(suite,"calibration")
    if len(cases)!=COUNTS["calibration"] or {c.lineage for c in cases}&{r["lineage"] for r in frames}:
        raise ValueError("calibration lineage overlap or incomplete roster")
    rows=[];raw=[];observations=[]
    conditions=(("witness","normal","frontier","witness"),
                ("utility","normal","frontier","utility"),
                ("utility","zero","frontier","zero"),
                ("utility","roll","frontier","roll"),
                ("utility","normal","sibling","sibling"))
    for c in cases:
        ref=rg.search(c,"astar",budget=WORK,depth=DEPTH)
        r=search(rg,c)
        if r!=ref:
            raise ValueError("fixed symbolic control differs from pinned implementation")
        rows.append({**r,"condition":"astar","seed":None,"stream":None})
        rows.append({**rg.search(c,"bidir",budget=WORK,depth=DEPTH),
                     "condition":"bidir","seed":None,"stream":None})
        for stream in STREAMS:
            r=search(rg,c,"uniform",seed=int(base.digest([stream,c.root])[:16],16))
            rows.append({**r,"condition":"uniform","seed":None,"stream":stream})
        for seed in SEEDS:
            for arm,mode,policy,condition in conditions:
                model=models[(seed,arm)];receipt=receipts[(seed,arm)]
                calls=0
                def ranker(case,word,moves,d,work_left):
                    nonlocal calls
                    calls+=1
                    b=budget_bucket(work_left)
                    p=packet(rg,case,word,d,b);menu=[rg.action_token(m) for m in moves]
                    with torch.no_grad():
                        if condition=="utility" and calls<=4:
                            return probe(model,case,p,menu,d,b,seed,calls,receipt["checkpoint"],raw,observations)
                        pred=model(p,menu,depth=2,checks_remaining=min(8,d),middle_order="ab",message_mode=mode)[-1]
                        return pred["outcome" if arm=="utility" else "logits"].tolist()
                r=search(rg,c,policy,ranker)
                if calls!=r["model_calls"]:
                    raise ValueError("unaccounted model calls")
                rows.append({**r,"condition":condition,"seed":seed,"stream":None})
    for key,model in models.items():
        if base._model_fingerprint(model)!=receipts[key]["final"]:
            raise ValueError("model changed after freezing")
    expected=len(cases)*(2+len(STREAMS)+len(SEEDS)*len(conditions))
    if len(rows)!=expected:
        raise ValueError("incomplete search roster")
    byroot={c.root:c for c in cases};proofs=[]
    for i,row in enumerate(rows):
        if row["closed"]:
            rg.replay(byroot[row["root"]],row["path"])
            proofs.append((byroot[row["root"]],row["path"],f"policy_{i}"))
    n=len(proofs)
    _,_,refs=rg.load_split(suite,"calibration",witnesses=True)
    proofs.extend((c,refs[c.root],"reference_"+c.root) for c in cases)
    cert=base.certify(rg,proofs,out/"calibration_certificates")
    if cert["compiled_axiom_free"]!=n+len(cases):
        raise ValueError("native certificate count mismatch")
    for i,row in enumerate(rows):
        row["native_certified"]=row["closed"] and f"policy_{i}" in cert["declarations"]
    if len(raw)!=len(observations) or len(raw)!=len({r["decision_id"] for r in raw}):
        raise ValueError("duplicate/missing decision probes")
    base.write_jsonl(out/"episodes.jsonl",rows)
    base.write_jsonl(out/"observations.jsonl",observations)
    torch.save(raw,out/"raw_boundary_tensors.pt")
    encoded=[base.encode(o) for o in observations]
    base.write_jsonl(out/"encoded.jsonl",encoded)
    if not encoded:raise ValueError("no label-free probe traces")
    (out/"fresco.svg").write_text(base.fresco_svg(encoded[0]),encoding="utf-8")
    names=("astar","bidir","uniform","witness","utility","zero","roll","sibling")
    rates={};costs={};seed_rates={}
    for name in names:
        rr=[r for r in rows if r["condition"]==name]
        rates[name]=sum(r["native_certified"] for r in rr)/len(rr)
        costs[name]={"attempts":len(rr),"work":sum(r["work"] for r in rr),
                     "expanded":sum(r["expanded"] for r in rr),"model_calls":sum(r["model_calls"] for r in rr),
                     "closed_path_steps":sum(len(r["path"]) for r in rr)}
    for seed in SEEDS:
        seed_rates[str(seed)]={}
        for _,_,_,name in conditions:
            rr=[r for r in rows if r["condition"]==name and r["seed"]==seed]
            seed_rates[str(seed)][name]=sum(r["native_certified"] for r in rr)/len(rr)
    def root_rate(root,name):
        rr=[r for r in rows if r["root"]==root and r["condition"]==name]
        expected=5 if name=="uniform" else 1 if name in ("astar","bidir") else len(SEEDS)
        if len(rr)!=expected:raise ValueError("unmatched root condition")
        return sum(r["native_certified"] for r in rr)/len(rr)
    paired={}
    for name in names:
        if name=="utility":continue
        vals=[root_rate(c.root,"utility")-root_rate(c.root,name) for c in cases]
        paired["utility_minus_"+name]={"mean":sum(vals)/len(vals),"descriptive_95":base.bootstrap(vals),
            "wins":sum(v>0 for v in vals),"losses":sum(v<0 for v in vals),"ties":sum(v==0 for v in vals)}
    result={"schema":"lbf13.budget_fresco.v1","source":base.UPSTREAM_COMMIT,
            "generator_seed":GENERATOR_SEED,"suite_manifest_sha256":base.file_sha(suite/"manifest.json"),
            "training":tm,"fits":[receipts[k] for k in sorted(receipts)],"rates":rates,"seed_rates":seed_rates,
            "costs":costs,"paired":paired,"policy_certificates":n,"reference_certificates":len(cases),
            "probed_decisions":len(raw),"work_cap":WORK,"depth_cap":DEPTH,"budgets":BUCKETS,
            "utility_screen":paired["utility_minus_witness"]["mean"]>=.05 and rates["utility"]>=.4
                and rates["utility"]-max(rates[k] for k in ("astar","bidir","uniform"))>=.1
                and all(x["utility"]>=x["witness"] for x in seed_rates.values()),
            "memory_screen":all(paired["utility_minus_"+m]["mean"]>=.05 for m in ("zero","roll")),
            "sealed_outcomes_opened":False,"scientific_eligible":False,"native_online_tactic_calls":0,
            "artifacts":{name:base.file_sha(out/name) for name in ("episodes.jsonl","observations.jsonl",
                         "raw_boundary_tensors.pt","encoded.jsonl","fresco.svg")}}
    base.write(out/"manifest.json",result)
    print("LBF13_RESULT",base.canonical(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();campaign(base.load_upstream(a.upstream),a.out)


if __name__=="__main__":main()
