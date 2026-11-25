"""LBF-20F CPU learning, complete support labels and matched recurrence controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from lbf20f_csp import D,FULL,bits,generate,Solver,audit,FAMILIES

ARMS=("direct1","recur4","untied4")
SEEDS=(107,109,113)
WIDTH=24
NFEAT=7
STEPS=480
BATCH=32
BUDGET=64


def packet(s,dom):
    n=s.c.n
    x=torch.zeros(n,D,NFEAT)
    rel=torch.zeros(n*D,n*D)
    for i in range(n):
        deg=len(s.nb[i])
        for a in bits(dom[i]):
            vals=[(s.rel[i,j][a]&dom[j]).bit_count()/D for j in s.nb[i]]
            x[i,a]=torch.tensor([1.,dom[i].bit_count()/D,deg/max(1,n-1),
                                sum(vals)/max(1,deg),min(vals,default=1.),
                                max(vals,default=1.),float(dom[i].bit_count()==1)])
            for j in s.nb[i]:
                for b in bits(s.rel[i,j][a]&dom[j]):
                    rel[i*D+a,j*D+b]=1.
    norm=rel.sum(-1,keepdim=True).clamp_min(1.)
    return x.reshape(n*D,NFEAT),rel/norm


class Net(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.entry=nn.Linear(NFEAT,WIDTH)
        self.blocks=nn.ModuleList(nn.Linear(2*WIDTH,WIDTH) for _ in range(4 if arm=="untied4" else 1))
        self.head=nn.Linear(WIDTH,1)

    def forward(self,x,rel):
        h=torch.tanh(self.entry(x))
        for k in range(1 if self.arm=="direct1" else 4):
            m=torch.bmm(rel,h)
            h=torch.tanh(h+self.blocks[k if self.arm=="untied4" else 0](torch.cat((h,m),dim=-1)))
        return self.head(h).squeeze(-1)


def frames(cases):
    rng=random.Random(2026112527)
    xx=[];rr=[];yy=[];mm=[];roots=[]
    for c in cases:
        s=Solver(c)
        root=s.ac(c.domains)
        if root is None:continue
        states={root}
        for _ in range(8):
            dom=list(root)
            for _ in range(rng.randrange(1,4)):
                choices=[i for i,m in enumerate(dom) if m.bit_count()>1]
                if not choices:break
                i=rng.choice(choices);dom[i]=1<<rng.choice(list(bits(dom[i])))
            state=s.ac(tuple(dom))
            if state is not None:states.add(state)
        for dom in sorted(states):
            label=s.supports(dom)
            x,r=packet(s,dom)
            xx.append(x);rr.append(r)
            yy.append(torch.tensor([float(bool(label[i]&(1<<a))) for i in range(c.n) for a in range(D)]))
            mm.append(torch.tensor([bool(dom[i]&(1<<a)) for i in range(c.n) for a in range(D)]))
            roots.append(c.root)
    return torch.stack(xx),torch.stack(rr),torch.stack(yy),torch.stack(mm),roots


@torch.no_grad()
def score(model,s,dom,i):
    x,r=packet(s,dom)
    return model(x[None],r[None])[0,i*D:(i+1)*D].tolist()


def root_info(c):
    s=Solver(c);ac=s.ac(c.domains)
    exact=s.supports(c.domains)
    truth=any(exact)
    pc=s.pc(c.domains)
    assert not truth or (ac is not None and pc is not None)
    if truth:
        assert all(a&b==a for a,b in zip(exact,ac)) and all(a&b==a for a,b in zip(exact,pc))
    return {"root":c.root,"family":c.family,"n":c.n,"satisfiable":truth,"exact":exact,
            "ac":ac,"pc":pc,"ac_extra_values":sum(m.bit_count() for m in ac)-sum(m.bit_count() for m in exact) if ac else 0,
            "pc_extra_values":sum(m.bit_count() for m in pc)-sum(m.bit_count() for m in exact) if pc else 0}


def evaluate(cases,info,policy,model=None):
    rows=[]
    for c,truth in zip(cases,info):
        s=Solver(c)
        scorer=(lambda s,d,i:score(model,s,d,i)) if model is not None else None
        r=s.solve(policy=policy if model is None else "lcv",budget=BUDGET,scorer=scorer)
        assert r["status"]!="solved" or truth["satisfiable"]
        assert r["status"]!="refuted" or not truth["satisfiable"]
        ac=truth["ac"]
        choice=None;spread=None
        if ac and any(m.bit_count()>1 for m in ac):
            i=s.variable(ac)
            if model is not None:
                scores=scorer(s,ac,i)
                if c.family=="parity_ring" and ac==c.domains:
                    spread=max(scores)-min(scores)
                    assert spread<1e-5,spread
            elif policy=="bp":scores=s.bp(ac,i)
            else:scores=s.lcv(ac,i)
            a=min(bits(ac[i]),key=lambda a:(-scores[a],a))
            choice=bool(truth["exact"][i]&(1<<a))
        rows.append({**r,"root":c.root,"family":c.family,"satisfiable":truth["satisfiable"],
                     "first_ac_mrv_value_viable":choice,"ring_score_spread":spread})
    return rows


def aggregate(rows):
    sat=[r for r in rows if r["satisfiable"]]
    unsat=[r for r in rows if not r["satisfiable"]]
    viable=[r["first_ac_mrv_value_viable"] for r in sat if r["first_ac_mrv_value_viable"] is not None]
    return {"roots":len(rows),"sat_roots":len(sat),"solved":sum(r["status"]=="solved" for r in sat),
            "unsat_roots":len(unsat),"refuted":sum(r["status"]=="refuted" for r in unsat),
            "mean_nodes":statistics.mean(r["nodes"] for r in rows),
            "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in rows),
            "mean_arc_revisions":statistics.mean(r["arc_revisions"] for r in rows),
            "mean_pair_revisions":statistics.mean(r["pair_revisions"] for r in rows),
            "mean_model_calls":statistics.mean(r["model_calls"] for r in rows),
            "sat_solved_by_nodes":{str(k):sum(r["status"]=="solved" and r["nodes"]<=k for r in sat) for k in (2,4,8,16,64)},
            "first_value_viable_rate":statistics.mean(viable) if viable else None,
            "status_counts":{k:sum(r["status"]==k for r in rows) for k in ("solved","refuted","budget")},
            "mean_nodes_by_family":{f:statistics.mean(r["nodes"] for r in rows if r["family"]==f) for f in FAMILIES}}


def main(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    au=audit()
    print("CSP_AUDIT",json.dumps(au),flush=True)
    train=generate("train",6,96,2026112515,allow_unsat=True)
    held={"dev8":generate("dev",8,48,2026112516,allow_unsat=True),
          "stress12":generate("stress",12,36,2026112517,allow_unsat=True)}
    native=generate("native",6,12,2026112518)
    all_cases={"train":train,**held,"native":native}
    assert len({(c.n,c.domains,c.edges) for cs in all_cases.values() for c in cs})==sum(map(len,all_cases.values()))
    taskbytes=(json.dumps({k:[c.__dict__ for c in cs] for k,cs in all_cases.items()},sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(taskbytes)
    x,r,y,m,root_ids=frames(train)
    # All labels and frames derive from the train-root partition.
    assert set(root_ids)<={c.root for c in train}
    info={k:[root_info(c) for c in cs] for k,cs in held.items()}
    baselines={}
    for k,cs in held.items():
        baselines[k]={}
        for p in ("lcv","bp","pc"):
            rows=evaluate(cs,info[k],p)
            baselines[k][p]={"aggregate":aggregate(rows),"rows":rows}
    fits=[]
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            net=Net(arm)
            opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.0001)
            gen=torch.Generator().manual_seed(seed+3001)
            start=time.perf_counter()
            for step in range(STEPS):
                idx=torch.randint(len(x),(BATCH,),generator=gen)
                pred=net(x[idx],r[idx])
                valid=m[idx]
                loss=nn.functional.binary_cross_entropy_with_logits(pred[valid],y[idx][valid])
                opt.zero_grad();loss.backward();opt.step()
            seconds=time.perf_counter()-start
            p=out/f"{arm}_{seed}.pt";torch.save(net.state_dict(),p)
            rec={"architecture":arm,"seed":seed,"parameters":sum(z.numel() for z in net.parameters()),
                 "passes":1 if arm=="direct1" else 4,"steps":STEPS,"batch_size":BATCH,
                 "checkpoint_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                 "fit_seconds":seconds,"last_batch_loss":float(loss.detach()),"cohorts":{}}
            net.eval()
            for k,cs in held.items():
                rows=evaluate(cs,info[k],"lcv",net)
                rec["cohorts"][k]={"aggregate":aggregate(rows),"rows":rows}
            fits.append(rec)
            print("FIT",arm,seed,rec["parameters"],{k:v["aggregate"]["mean_nodes"] for k,v in rec["cohorts"].items()},flush=True)
    agg={}
    for k,cs in held.items():
        agg[k]={"roots":len(cs),"sat":sum(i["satisfiable"] for i in info[k]),
                "ac_extra_values":sum(i["ac_extra_values"] for i in info[k]),
                "pc_extra_values":sum(i["pc_extra_values"] for i in info[k]),
                "classical":{p:v["aggregate"] for p,v in baselines[k].items()},
                "learned":{a:{key:statistics.mean(f["cohorts"][k]["aggregate"][key] for f in fits if f["architecture"]==a)
                             for key in ("solved","refuted","mean_nodes","mean_wall_seconds","first_value_viable_rate")}
                           for a in ARMS}}
    summary={"schema":"lbf20f.shared-witness.v1","scope":"development_only","audit":au,
             "task_sha256":hashlib.sha256(taskbytes).hexdigest(),"train_roots":len(train),
             "train_frame_roots":len(set(root_ids)),"train_frames":len(x),"train_value_labels":int(m.sum()),
             "train_positive_rate":float(y[m].mean()),"fits":fits,"cohorts":agg}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"baseline_rows.json").write_text(json.dumps({"info":info,"baselines":baselines},indent=2)+"\n")
    print("SYNTHETIC_RESULTS",json.dumps({k:v for k,v in summary.items() if k!="fits"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
