"""LBF-20G: pair-relation cells, exact Boolean controls and root-only ranking.

Hard support has its own arithmetic. Learned scores are never used to prune.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from lbf20f_csp import Case, Solver, D, FULL, bits, generate

ARMS = ("soft1", "soft4", "untied4", "hard3_readout")
SEEDS = (107, 109, 113)
WIDTH = 24
STEPS = 320
BATCH = 8
BUDGET = 64


def tensor(case, dom):
    n = case.n
    rel = case.relations()
    a = torch.zeros(n,n,D,D)
    for i in range(n):
        for j in range(n):
            for x in bits(dom[i]):
                row = (1<<x) if i == j else rel.get((i,j),(FULL,)*D)[x]
                for y in bits(row & dom[j]):
                    a[i,j,x,y] = 1.
    return a


def compose(s):
    # [batch,i,j,k,a,b]; count over shared intermediate value u.
    return torch.einsum("bikau,bkjuv->bijkav",s,s)


def hard_step(s):
    return s * (compose(s).amin(dim=3) > 0).to(s.dtype)


def hard_rounds(s, k):
    for _ in range(k):
        s = hard_step(s)
    return s


def diag(s):
    idx = torch.arange(s.shape[1],device=s.device)
    return s[:,idx,idx].diagonal(dim1=-2,dim2=-1)


def masks(scores):
    return tuple(sum(1<<a for a in range(D) if float(row[a])>0.5) for row in scores)


def features(base,s):
    c=compose(s)/D
    return torch.stack((base,s,c.amin(3),c.mean(3),c.amax(3)),dim=-1)


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,WIDTH),nn.Tanh(),nn.Linear(WIDTH,1))
    def forward(self,base,s):
        return self.net(features(base,s)).squeeze(-1)


class Net(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.cells=nn.ModuleList(Cell() for _ in range(4 if arm=="untied4" else 1))
    def forward(self,base):
        s=base
        if self.arm=="hard3_readout":
            s=hard_rounds(s,3)
            return self.cells[0](base,s)
        for t in range(1 if self.arm=="soft1" else 4):
            logits=self.cells[t if self.arm=="untied4" else 0](base,s)
            s=base*torch.sigmoid(logits)
        return logits


def functional_projection(case,dom):
    """Exact when all constrained relations are total bijections.

    One anchor per component, at most D candidate assignments per anchor.
    Return None if outside fragment; tuple of zero masks for UNSAT.
    """
    s=Solver(case)
    if any(any(row.bit_count()!=1 for row in r) or len(set(r))!=D for r in s.rel.values()):
        return None
    seen=set();support=[0]*case.n
    for i in range(case.n):
        if i in seen:continue
        comp={i};q=[i]
        for u in q:
            for v in s.nb[u]:
                if v not in comp:comp.add(v);q.append(v)
        seen.update(comp)
        good=0
        for a in bits(dom[i]):
            v={i:a};ok=True
            for u in q:
                if not dom[u]&(1<<v[u]):ok=False;break
                for j in s.nb[u]:
                    z=next(bits(s.rel[u,j][v[u]]))
                    if j in v and v[j]!=z:ok=False;break
                    v[j]=z
                if not ok:break
            if ok:
                good+=1
                assert len(v)==len(comp)
                for j,a0 in v.items():support[j]|=1<<a0
        if not good:return (0,)*case.n
    return tuple(support)


def singleton_projection(case,dom):
    s=Solver(case)
    out=[]
    for i,m in enumerate(dom):
        keep=0
        for a in bits(m):
            ds=list(dom);ds[i]=1<<a
            if s.ac(tuple(ds)) is not None:keep|=1<<a
        out.append(keep)
    return tuple(out)


def colors(split,n,count,seed):
    rng=random.Random(seed)
    out=[]
    neq=tuple(7^(1<<a) if a<3 else 0 for a in range(D))
    for i in range(count):
        # Half planted 3-colorable, half unconstrained random graphs.
        # Exact search adjudicates the latter's status.
        p=[rng.randrange(3) for _ in range(n)]
        pairs=[]
        for u,v in itertools.combinations(range(n),2):
            if i%2==0 and p[u]==p[v]:continue
            if rng.random()<(.70 if i%2==0 else .48):
                pairs.append((u,v,neq))
        out.append(Case(f"{split}_color_{i:03d}","color",n,(7,)*n,tuple(pairs)))
    return out


def worlds(n):
    return torch.tensor(list(itertools.product(range(D),repeat=n)),dtype=torch.long)


def exact_pairs(case,dom,w):
    ok=torch.ones(len(w),dtype=torch.bool)
    for i,d in enumerate(dom):
        ok&=torch.tensor([bool(d&(1<<a)) for a in range(D)])[w[:,i]]
    for i,j,r in case.edges:
        allowed=torch.tensor([[bool(r[a]&(1<<b)) for b in range(D)] for a in range(D)])
        ok&=allowed[w[:,i],w[:,j]]
    v=w[ok];out=torch.zeros(case.n,case.n,D,D)
    for i in range(case.n):
        for j in range(case.n):
            out[i,j,v[:,i],v[:,j]]=1.
    return out


def audit():
    start=time.perf_counter()
    count=collections.Counter()
    w=worlds(4)
    for c in generate("audit_g",4,96,2026112532,allow_unsat=True):
        x=tensor(c,c.domains)[None]
        y=exact_pairs(c,c.domains,w)[None]
        assert masks(diag(y)[0])==Solver(c).supports(c.domains)
        count["systems"]+=1
        count["worlds"]+=len(w)
        for k in range(1,5):
            x=hard_step(x)
            assert not bool((y>x).any())
            count["round_checks"]+=1
        f=functional_projection(c,c.domains)
        if f is not None:
            assert f==masks(diag(y)[0]);count["functional_exact"]+=1
    # Pair consistency is incomplete: a clique on five variables with four
    # colors has no assignment but every unequal pair has a third-color witness.
    neq=tuple(FULL^(1<<a) for a in range(D))
    clique=Case("k5","clique",5,(FULL,)*5,tuple((i,j,neq) for i,j in itertools.combinations(range(5),2)))
    s=Solver(clique)
    assert s.supports(clique.domains)==(0,)*5
    assert torch.equal(hard_rounds(tensor(clique,clique.domains)[None],4),tensor(clique,clique.domains)[None])
    rings=[]
    for n in (6,8,12,16,24,32,64):
        cs=[c for c in generate("scale",n,12,2026112540+n) if c.family=="parity_ring"]
        for c in cs:
            t=time.perf_counter();f=functional_projection(c,c.domains);ft=time.perf_counter()-t
            assert f is not None and all(m.bit_count()==2 for m in f)
            assert singleton_projection(c,c.domains)==f
            x=tensor(c,c.domains)[None]
            r=0;t=time.perf_counter()
            while masks(diag(x)[0])!=f:
                x=hard_step(x);r+=1
                assert r<=math.ceil(math.log2(n))
            pt=time.perf_counter()-t
            assert r==math.ceil(math.log2(n))
            rings.append({"n":n,"rounds":r,"triple_revisions":r*n**3,
                          "functional_seconds":ft,"composition_seconds":pt})
    return {**dict(count),"false_negatives":0,"k5_incomplete":True,"rings":rings,
            "seconds":time.perf_counter()-start}


def train_frames(cases):
    w=worlds(6);rng=random.Random(2026112539)
    xx=[];yy=[];roots=[]
    for c in cases:
        s=Solver(c);root=s.ac(c.domains)
        if root is None:continue
        states={root}
        for _ in range(4):
            i=rng.randrange(c.n);d=list(root);d[i]=1<<rng.choice(list(bits(d[i])))
            q=s.ac(tuple(d))
            if q is not None:states.add(q)
        for d in sorted(states):
            x=tensor(c,d);y=exact_pairs(c,d,w)
            assert not bool((y>x).any())
            xx.append(x);yy.append(y);roots.append(c.root)
    return torch.stack(xx),torch.stack(yy),roots


def ap(pairs):
    positives=sum(y for _,y in pairs)
    if not positives:return None
    groups=collections.defaultdict(list)
    for s,y in pairs:groups[s].append(y)
    tp=seen=0;out=0.
    for s in sorted(groups,reverse=True):
        ys=groups[s];t=sum(ys);tp+=t;seen+=len(ys)
        out+=t*tp/seen
    return out/positives


@torch.no_grad()
def root_scores(case,dom,arm,net):
    if arm=="lcv":return None,dom,0
    if arm=="functional":
        f=functional_projection(case,dom)
        return None,dom if f is None else f,0
    if arm=="singleton":return None,singleton_projection(case,dom),0
    if arm in ("hard1","hard3","hard4"):
        k=int(arm[-1]);r=hard_rounds(tensor(case,dom)[None],k)
        return diag(r)[0].tolist(),masks(diag(r)[0]),k
    x=tensor(case,dom)[None]
    p=diag(torch.sigmoid(net(x))*x)[0]
    return p.tolist(),dom,1 if arm=="soft1" else 4


def exact_info(cases):
    return {c.root:Solver(c).supports(c.domains) for c in cases}


@torch.no_grad()
def evaluate(cases,truth,arm,net=None):
    rows=[];ap_data=[]
    for c in cases:
        start=time.perf_counter()
        s=Solver(c);exact=truth[c.root];sat=any(exact)
        dom=s.ac(c.domains)
        root_scores_sec=0.;first=None
        fn=fp=0;brier=0.;nlabels=0;rounds=0
        if dom is None:
            status="refuted";nodes=0;arc=0;sol=None
        else:
            t=time.perf_counter()
            scores,filtered,rounds=root_scores(c,dom,arm,net)
            root_scores_sec=time.perf_counter()-t
            if scores is None:
                scores=[[float(bool(filtered[i]&(1<<a))) for a in range(D)] for i in range(c.n)] if arm!="lcv" else [[float(bool(dom[i]&(1<<a))) for a in range(D)] for i in range(c.n)]
            if arm not in ARMS:
                assert all(a&b==a for a,b in zip(exact,filtered))
            for i in range(c.n):
                for a in bits(dom[i]):
                    v=float(scores[i][a]);y=int(bool(exact[i]&(1<<a)))
                    fn+=int(y and v<.5);fp+=int(not y and v>=.5)
                    brier+=(v-y)**2;nlabels+=1;ap_data.append((v,y))
            if any(m.bit_count()>1 for m in dom) and sat:
                i=s.variable(dom);a=min(bits(dom[i]),key=lambda a:(-scores[i][a],a))
                first=bool(exact[i]&(1<<a))
            # Learned policies only rank; use identical fixed root scores at
            # every later AC state. Boolean/functional/SAC may soundly filter.
            scorer=(lambda _s,_d,i:scores[i]) if arm in ARMS else None
            r=s.solve(filtered,policy="lcv",budget=BUDGET,scorer=scorer)
            status=r["status"];nodes=r["nodes"];arc=r["arc_revisions"];sol=r["solution"]
        assert status!="refuted" or not sat
        assert status!="solved" or (sat and c.valid(sol))
        rows.append({"root":c.root,"family":c.family,"sat":sat,"status":status,"solution":sol,
                     "nodes":nodes,"arc_revisions":arc,"triple_revisions":rounds*c.n**3,
                     "model_or_filter_seconds":root_scores_sec,"wall_seconds":time.perf_counter()-start,
                     "threshold_false_negatives":fn,"threshold_false_positives":fp,"label_count":nlabels,
                     "brier_sum":brier,"first_value_viable":first})
    return {"roots":len(rows),"solved":sum(r["status"]=="solved" for r in rows),
            "refuted":sum(r["status"]=="refuted" for r in rows),
            "budget":sum(r["status"]=="budget" for r in rows),
            "mean_nodes":statistics.mean(r["nodes"] for r in rows),
            "mean_wall_seconds":statistics.mean(r["wall_seconds"] for r in rows),
            "mean_filter_seconds":statistics.mean(r["model_or_filter_seconds"] for r in rows),
            "mean_triple_revisions":statistics.mean(r["triple_revisions"] for r in rows),
            "false_negatives":sum(r["threshold_false_negatives"] for r in rows),
            "false_positives":sum(r["threshold_false_positives"] for r in rows),
            "ap":ap(ap_data),"brier":sum(r["brier_sum"] for r in rows)/max(1,len(ap_data)),
            "first_viable":statistics.mean([r["first_value_viable"] for r in rows if r["first_value_viable"] is not None]),
            "nodes_by_family":{k:statistics.mean(r["nodes"] for r in rows if r["family"]==k)
                               for k in sorted({r["family"] for r in rows})},
            "rows":rows}


def main(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    au=audit();print("AUDIT",json.dumps({k:v for k,v in au.items() if k!="rings"}),flush=True)
    train=generate("train_g",6,96,2026112533,allow_unsat=True)+colors("train_g",6,24,2026112534)
    held={"dev8":generate("dev_g",8,48,2026112535,allow_unsat=True),
          "stress12":generate("stress_g",12,36,2026112536,allow_unsat=True),
          "color12":colors("stress_g",12,24,2026112537)}
    taskbytes=(json.dumps({k:[c.__dict__ for c in v] for k,v in {"train":train,**held}.items()},
                           sort_keys=True,separators=(",",":"))+"\n").encode()
    (out/"tasks.json").write_bytes(taskbytes)
    x,y,roots=train_frames(train)
    truth={k:exact_info(v) for k,v in held.items()}
    baseline={k:{a:evaluate(cs,truth[k],a) for a in ("lcv","functional","singleton","hard1","hard3","hard4")}
              for k,cs in held.items()}
    fits=[]
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            net=Net(arm);opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.0001)
            gen=torch.Generator().manual_seed(seed+3001)
            start=time.perf_counter()
            for step in range(STEPS):
                idx=torch.randint(len(x),(BATCH,),generator=gen)
                base=x[idx];target=y[idx];valid=base.bool()
                logits=net(base)
                loss=nn.functional.binary_cross_entropy_with_logits(logits[valid],target[valid])
                opt.zero_grad();loss.backward();opt.step()
            sec=time.perf_counter()-start
            p=out/f"{arm}_{seed}.pt";torch.save(net.state_dict(),p)
            net.eval()
            rec={"architecture":arm,"seed":seed,"parameters":sum(z.numel() for z in net.parameters()),
                 "passes":1 if arm=="soft1" else 4,"steps":STEPS,"batch_size":BATCH,
                 "checkpoint_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"fit_seconds":sec,
                 "last_batch_loss":float(loss.detach()),
                 "cohorts":{k:evaluate(cs,truth[k],arm,net) for k,cs in held.items()}}
            fits.append(rec)
            print("FIT",arm,seed,rec["parameters"],
                  {k:round(v["mean_nodes"],4) for k,v in rec["cohorts"].items()},flush=True)
    fields=("solved","refuted","budget","mean_nodes","mean_wall_seconds","mean_filter_seconds",
            "mean_triple_revisions","false_negatives","false_positives","ap","brier","first_viable")
    agg={}
    for k,cs in held.items():
        agg[k]={"roots":len(cs),"sat":sum(any(v) for v in truth[k].values()),
                "classical":{a:{f:v[f] for f in fields} for a,v in baseline[k].items()},
                "learned":{a:{f:statistics.mean(r["cohorts"][k][f] for r in fits if r["architecture"]==a)
                               for f in fields} for a in ARMS}}
    summary={"schema":"lbf20g.relation-composition.v1","scope":"development_only",
             "audit":au,"task_sha256":hashlib.sha256(taskbytes).hexdigest(),
             "train_roots":len(train),"train_frame_roots":len(set(roots)),"train_frames":len(x),
             "train_pair_labels":int(x.sum()),"train_positive_rate":float(y[x.bool()].mean()),
             "fits":fits,"cohorts":agg}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (out/"baseline_rows.json").write_text(json.dumps(baseline,indent=2)+"\n",encoding="utf-8")
    small={**summary,"fits":[{k:v for k,v in f.items() if k!="cohorts"} for f in fits],
           "audit":{k:v for k,v in au.items() if k!="rings"}}
    print("RESULTS",json.dumps(small),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);main(p.parse_args().out)
