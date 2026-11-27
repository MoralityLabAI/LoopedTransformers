"""OLR-06G: calibrated one-shot selection among paid rewrite schedules.

All search outcomes used for fitting are from distinct source equation
systems. The evaluation menu is run only after selectors have frozen their
choices. Native certification is inherited from the exact 06E compiler.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from itertools import combinations,product
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .data import canonical,sha256_file
from .equality_suite import ALPHABET,RULE_COUNT,BANDS,choose_shell,system_lineage
from .olr06e_multigoal import Root,exact,certify,read,write
from .olr06f_anytime import ARMS as SINGLE,anytime,path_audit,load as load_old

SEED=20270103
OLD_SHA="6e81000eef3dab7d3d26ef1b284dcdbb8bd4dd25072c0a49132ce84e7626b89c"
COUNTS={"train":{"dev":32,"transfer":32},"calibration":{"dev":8,"transfer":8},
        "evaluation":{"dev":24,"transfer":24}}
CAP=512
RESTARTS={
    "cache128_demand":("cached_independent",128,"demand"),
    "demand128_cache":("demand",128,"cached_independent"),
    "demand128_eager2":("demand",128,"eager2"),
    "eager192_demand":("eager2",192,"demand"),
}
MENU=(*SINGLE,*RESTARTS)
ARMS=("bi","one_way","bi1","local","dense","full")
SEEDS=(17,29,43)
MIXES=(0.,.25,.5,1.)
REG="olr06g.portfolio_registry.v1"
FIT="olr06g.portfolio_fit.v1"
SCORE="olr06g.portfolio_score.v1"
K=5
L=K+RULE_COUNT
D=156
WIDTH=32
PAIRS=tuple("".join(x) for x in product(ALPHABET,repeat=2))
METERS=("expansions","edges","logical","logical_edges","hits","abstract_expansions",
        "abstract_edges","rekeys","pdb_seconds","total_work")


def setup():
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)


def freeze(old,out):
    if out.exists() or sha256_file(old/"manifest.json")!=OLD_SHA:
        raise ValueError("fresh registry and pinned 06F bank required")
    excluded=set(read(old/"excluded.json"))
    excluded.update(r.lineage for r in load_old(old))
    if len(excluded)!=300:raise ValueError("prior lineage count changed")
    out.mkdir(parents=True)
    write(out/"excluded.json",sorted(excluded))
    used=set(excluded)
    pairs=list(combinations(PAIRS,2));files={}
    for role,panels in COUNTS.items():
        rows=[]
        for split,count in panels.items():
            rng=random.Random(f"olr06g/{SEED}/{role}/{split}")
            k=3 if split=="dev" else 5
            got=[]
            for _ in range(count*1000):
                if len(got)==count:break
                rules=tuple((a,b) if rng.randrange(2) else (b,a)
                            for a,b in rng.sample(pairs,RULE_COUNT))
                lineage=system_lineage(rules)
                if lineage in used:continue
                n=7+len(got)%3 if split=="dev" else 10
                source="".join(rng.choice(ALPHABET) for _ in range(n))
                targets=[]
                for j in range(k):
                    t=choose_shell(source,rules,BANDS[j%3],rng)
                    if t is None or t in targets:break
                    targets.append(t)
                if len(targets)!=k:continue
                rng.shuffle(targets)
                root=Root(split,rules,source,tuple(targets)).validate()
                used.add(lineage);got.append(asdict(root))
            if len(got)!=count:raise ValueError("fresh cohort generator exhausted")
            rows.extend(got)
        f=out/(role+".json");write(f,rows);files[f.name]=sha256_file(f)
    files["excluded.json"]=sha256_file(out/"excluded.json")
    m={"schema":REG,"seed":SEED,"counts":COUNTS,"old_manifest_sha256":OLD_SHA,
       "excluded_lineages":300,"files":files,"menu":MENU,"cap":CAP}
    write(out/"manifest.json",m);return m


def load(bank):
    m=read(bank/"manifest.json")
    if (m["schema"]!=REG or m["seed"]!=SEED or m["counts"]!=COUNTS
            or m["old_manifest_sha256"]!=OLD_SHA or tuple(m["menu"])!=MENU or m["cap"]!=CAP):
        raise ValueError("unregistered portfolio bank")
    for n,h in m["files"].items():
        if sha256_file(bank/n)!=h:raise ValueError("registry file changed")
    seen=set(read(bank/"excluded.json"))
    if len(seen)!=300:raise ValueError("exclusion census changed")
    groups={}
    for role,panels in COUNTS.items():
        rows=read(bank/(role+".json"));counts=Counter();roots=[]
        for r in rows:
            if set(r)!=set(Root.__dataclass_fields__):raise ValueError("nonpublic root metadata")
            root=Root(r["split"],tuple(tuple(e) for e in r["equations"]),
                      r["source"],tuple(r["targets"])).validate()
            if root.lineage in seen:raise ValueError("equation-system leakage")
            seen.add(root.lineage);counts[root.split]+=1;roots.append(root)
        if dict(counts)!=panels:raise ValueError("cohort count changed")
        groups[role]=roots
    return m,groups


def schedule(root,name):
    if name in SINGLE:
        r=anytime(root,name,CAP)
        r["phases"]=[{"arm":name,"cap":CAP,"work":r["total_work"],
                      "targets":list(root.targets),"resolved":sorted(r["found"])}]
        return r
    if name not in RESTARTS:raise ValueError("unregistered schedule")
    first,budget,second=RESTARTS[name];start=time.perf_counter()
    a=anytime(root,first,budget)
    out={k:a[k] for k in METERS}
    out["found"]={t:dict(q) for t,q in a["found"].items()}
    out["phases"]=[{"arm":first,"cap":budget,"work":a["total_work"],
                    "targets":list(root.targets),"resolved":sorted(a["found"])}]
    remaining=tuple(t for t in root.targets if t not in a["found"])
    if remaining:
        # A smaller target roster is an internal subproblem, not a new root.
        sub=Root(root.split,root.equations,root.source,remaining)
        cap=CAP-a["total_work"]
        b=anytime(sub,second,cap)
        for t,q in b["found"].items():
            out["found"][t]={**q,"fresh_at":q["fresh_at"]+a["expansions"],
                    "abstract_at":q["abstract_at"]+a["abstract_expansions"],
                    "found_at":q["found_at"]+a["total_work"]}
        for k in METERS:out[k]+=b[k]
        out["phases"].append({"arm":second,"cap":cap,"work":b["total_work"],
                              "targets":list(remaining),"resolved":sorted(b["found"])})
    out["seconds"]=time.perf_counter()-start
    path_audit(root,out,CAP)
    if sum(p["work"] for p in out["phases"])!=out["total_work"]:
        raise ValueError("restart spent-work accounting differs")
    return out


def packet(root):
    """Complete original word/rule packet; no search results or distances."""
    x=torch.zeros(L,D);inc=torch.zeros(L,L)
    n=len(root.source);k=len(root.targets)
    def word(w):
        a=torch.zeros(10,4)
        for j,c in enumerate(w):a[j,ALPHABET.index(c)]=1.
        return a.flatten()
    def hist(w):
        c=Counter(w[i:i+2] for i in range(n-1))
        return torch.tensor([c[p] for p in PAIRS],dtype=torch.float32)/(n-1)
    sw=word(root.source);sh=hist(root.source)
    hists=[hist(t) for t in root.targets]
    li=[PAIRS.index(a) for a,b in root.equations]
    ri=[PAIRS.index(b) for a,b in root.equations]
    sb=float((sh[li]+sh[ri]).sum())/RULE_COUNT
    for j,(target,h) in enumerate(zip(root.targets,hists,strict=True)):
        x[j,:6]=torch.tensor([1.,1.,0.,j/4,n/10,k/5])
        x[j,6:46]=sw;x[j,46:86]=word(target)
        x[j,86:102]=sh;x[j,102:118]=h
        x[j,150:156]=torch.tensor([sum(a!=b for a,b in zip(root.source,target))/n,
            float((sh-h).abs().sum()/2),sb,float((h[li]+h[ri]).sum())/RULE_COUNT,
            float(torch.minimum(sh,h).sum()),float((h>0).sum())/16])
        for r,(u,v) in enumerate(zip(li,ri,strict=True)):
            inc[K+r,j]=(sh[u]+sh[v]+h[u]+h[v])/2.
    for r,(u,v) in enumerate(zip(li,ri,strict=True)):
        x[K+r,:6]=torch.tensor([1.,0.,1.,r/5,n/10,k/5])
        x[K+r,6:46]=sw;x[K+r,86:102]=sh
        x[K+r,118+u]=1.;x[K+r,134+v]=1.
        avg=torch.stack(hists).mean(0)
        x[K+r,150:156]=torch.tensor([float(sh[u]),float(sh[v]),float(avg[u]),
                                     float(avg[v]),float(inc[K+r,:k].mean()),
                                     float(inc[K+r,:k].max())])
    # Each input pair is counted exactly once in hist(). Remaining operations
    # are fixed tensor feature construction and no concrete neighbor visits.
    return x,inc,(k+1)*(n-1)


def pack(roots):
    t=time.perf_counter();xs=[];incs=[];scans=0
    for root in roots:
        x,i,s=packet(root);xs.append(x);incs.append(i);scans+=s
    return torch.stack(xs),torch.stack(incs),{"seconds":time.perf_counter()-t,"pair_scans":scans}


class Reader(nn.Module):
    def __init__(self,arm,width=WIDTH):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown selector")
        self.arm=arm
        if arm=="full":
            self.full=nn.Sequential(nn.Linear(L*D+L*L,2*width),nn.GELU(),
                                    nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,len(MENU)))
            return
        # All token readers also receive both raw incidence rows.
        self.enc=nn.Sequential(nn.Linear(D+2*L,width),nn.GELU(),nn.Linear(width,width))
        if arm in ("bi","one_way","bi1"):
            self.down=nn.Linear(width,width,bias=False);self.up=nn.Linear(width,width,bias=False)
            self.cell=nn.Sequential(nn.Linear(4*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.norm=nn.LayerNorm(width)
        elif arm=="dense":
            self.attn=nn.MultiheadAttention(width,4,batch_first=True)
            self.ff=nn.Sequential(nn.Linear(2*width,2*width),nn.GELU(),nn.Linear(2*width,width))
            self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,len(MENU)))

    def forward(self,x,inc):
        if self.arm=="full":
            return self.full(torch.cat((x.flatten(1),inc.flatten(1)),1))
        active=x[...,0:1]
        e=self.enc(torch.cat((x,inc,inc.transpose(1,2)),-1))*active;h=e
        if self.arm in ("bi","one_way","bi1"):
            for _ in range(1 if self.arm=="bi1" else 3):
                down=torch.bmm(inc,self.down(h))
                up=torch.bmm(inc.transpose(1,2),self.up(h))*float(self.arm!="one_way")
                h=self.norm(h+self.cell(torch.cat((h,down,up,e),-1)))*active
        elif self.arm=="dense":
            for _ in range(3):
                a,_=self.attn(h,h,h,key_padding_mask=active.squeeze(-1)<.5,need_weights=False)
                h=self.n1(h+a);h=self.n2(h+self.ff(torch.cat((h,e),-1)))*active
        avg=h.sum(1)/active.sum(1)
        mx=h.masked_fill(active<.5,-torch.inf).max(1).values
        return self.head(torch.cat((avg,mx),1))


def utility(row,k):
    if row["total_work"]>CAP or len(row["found"])>k:
        raise ValueError("teacher exceeded legal resource contract")
    complete=len(row["found"])==k
    if complete and not row.get("joint_lemma"):
        raise ValueError("full teacher utility without compiled conjunction")
    return float(complete)+.25*len(row["found"])/k-.01*row["total_work"]/CAP


def digest_weights(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(json.dumps(t.detach().tolist(),separators=(",",":")).encode())
    return h.hexdigest()


def load_index(roots):
    return torch.tensor([int(r.split=="transfer") for r in roots],dtype=torch.long)


def flattened(x,i):return torch.cat((x.flatten(1),i.flatten(1)),1)


def calibrate(pred,target,base):
    """Choose blend only on source calibration; exact ties prefer fixed prior."""
    values=[]
    for a in MIXES:
        choice=((1-a)*base+a*pred).argmax(1)
        values.append(float(target[torch.arange(len(choice)),choice].mean()))
    ix=max(range(len(MIXES)),key=lambda j:(values[j],-j))
    return MIXES[ix],dict(zip(map(str,MIXES),values))


def classical_fit(x,i,y):
    z=flattened(x,i);mu=z.mean(0);sd=z.std(0,unbiased=False).clamp_min(.05)
    z=(z-mu)/sd
    center=y.mean(0)
    dual=torch.linalg.solve(z@z.T+16.*torch.eye(len(z)),y-center)
    return {"mean":mu,"std":sd,"source":z,"labels":y,"center":center,"dual":dual}


def classical_predict(state,x,i,arm):
    z=(flattened(x,i)-state["mean"])/state["std"]
    if arm=="ridge":return (z@state["source"].T)@state["dual"]+state["center"]
    if arm=="knn":
        ds=(z[:,None,:]-state["source"][None,:,:]).square().sum(-1)
        ids=torch.argsort(ds,dim=1,stable=True)[:,:5]
        return state["labels"][ids].mean(1)
    raise ValueError("unknown cheap selector")


def fit_reader(arm,seed,train,cal,base):
    x,i,y=train;cx,ci,cy=cal
    torch.manual_seed(seed);m=Reader(arm);initial=digest_weights(m)
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.005)
    rng=torch.Generator().manual_seed(seed+6700)
    sample=torch.randint(len(x),(160,16),generator=rng);start=time.perf_counter()
    for ids in sample:
        z=m(x[ids],i[ids])
        loss=F.smooth_l1_loss(z,y[ids],beta=.15)
        if not torch.isfinite(loss):raise ValueError("nonfinite source fit")
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    m.eval()
    with torch.no_grad():
        p=m(cx,ci);a,grid=calibrate(p,cy,base)
        train_mae=float((m(x,i)-y).abs().mean())
        cal_mae=float((p-cy).abs().mean())
    return m,{"arm":arm,"seed":seed,"parameters":sum(p.numel() for p in m.parameters()),
              "initial_sha256":initial,"weights_sha256":digest_weights(m),
              "sampler_sha256":hashlib.sha256(json.dumps(sample.tolist()).encode()).hexdigest(),
              "steps":160,"mix":a,"calibration_grid":grid,"source_mae":train_mae,
              "calibration_mae":cal_mae,"seconds":time.perf_counter()-start}


def matrix(roots):
    return [{"root":r.root,"split":r.split,"arm":arm,**schedule(r,arm)}
            for r in roots for arm in MENU]


def labels(roots,rows):
    index={(r["root"],r["arm"]):r for r in rows}
    return torch.tensor([[utility(index[r.root,a],len(r.targets)) for a in MENU]
                         for r in roots],dtype=torch.float32)


def files(path,names):return {n:sha256_file(path/n) for n in names}


def source(bank,out):
    if out.exists():raise ValueError("create-only source fit")
    setup();_,groups=load(bank);out.mkdir(parents=True);start=time.perf_counter()
    roots=groups["train"]+groups["calibration"];rows=matrix(roots)
    write(out/"frozen_source_plans.json",rows)
    cert=certify(roots,rows,out)
    write(out/"source_scores.json",rows)
    t=groups["train"];c=groups["calibration"]
    x,i,train_feature=pack(t);cx,ci,cal_feature=pack(c)
    y=labels(t,rows);cy=labels(c,rows)
    panel=load_index(t);cp=load_index(c)
    base=torch.stack([y[panel==s].mean(0) for s in (0,1)])
    state=classical_fit(x,i,y)
    meta={}
    for arm in ("ridge","knn"):
        pred=classical_predict(state,cx,ci,arm)
        a,grid=calibrate(pred,cy,base[cp])
        meta[arm]={"mix":a,"calibration_grid":grid}
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            m,stats=fit_reader(arm,seed,(x,i,y),(cx,ci,cy),base[cp])
            key=f"{seed}:{arm}";models[key]=m.state_dict();meta[key]=stats
            print("fit",key,stats["mix"],flush=True)
        a,b,d=(meta[f"{seed}:{arm}"] for arm in ("bi","one_way","bi1"))
        for name in ("parameters","initial_sha256","sampler_sha256"):
            if len({a[name],b[name],d[name]})!=1:
                raise ValueError("matched recurrence control changed source initialization")
    torch.save({"models":models,"classical":state,"base":base},out/"models.pt")
    names=("frozen_source_plans.json","source_scores.json","native_episodes.json",
           "conjunctions.lean","conjunction_compiler.log","leaves/standalone.lean",
           "leaves/compiler.log","leaves/compiler_receipt.json","models.pt")
    m={"schema":FIT,"registry_sha256":sha256_file(bank/"manifest.json"),
       "train_roots":len(t),"calibration_roots":len(c),"source_policy_trials":len(rows),
       "certificate":cert,"source_feature":train_feature,"calibration_feature":cal_feature,
       "menu":MENU,"arms":ARMS,"seeds":SEEDS,"models":meta,
       "source_fixed":{s:MENU[int(base[j].argmax())] for j,s in enumerate(("dev","transfer"))},
       "elapsed_seconds":time.perf_counter()-start,"files":files(out,names)}
    write(out/"manifest.json",m);return m


def fitted(bank,fit):
    m=read(fit/"manifest.json")
    if (m["schema"]!=FIT or m["registry_sha256"]!=sha256_file(bank/"manifest.json")
            or tuple(m["menu"])!=MENU or tuple(m["arms"])!=ARMS or tuple(m["seeds"])!=SEEDS):
        raise ValueError("source fit identity differs")
    for n,h in m["files"].items():
        if sha256_file(fit/n)!=h:raise ValueError("source artifact hash differs")
    state=torch.load(fit/"models.pt",map_location="cpu",weights_only=True)
    models={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{seed}:{arm}";model=Reader(arm)
            model.load_state_dict(state["models"][key],strict=True);model.eval()
            if digest_weights(model)!=m["models"][key]["weights_sha256"]:
                raise ValueError("frozen selector weights differ")
            models[seed,arm]=model
    return m,state,models


@torch.no_grad()
def choices(roots,m,state,models):
    # Per-root timing includes all feature work but no search. Each selector
    # is called on one packet, not an offline evaluation-sized batch.
    rows=[];feature=[]
    for r in roots:
        t=time.perf_counter();x,i,scans=packet(r)
        fs=time.perf_counter()-t;feature.append({"root":r.root,"seconds":fs,"pair_scans":scans})
        x=x[None];i=i[None];base=state["base"][int(r.split=="transfer")]
        def add(arm,seed,pred,alpha,seconds,uses_features):
            vector=(1-alpha)*base+alpha*pred
            j=int(vector.argmax())
            rows.append({"root":r.root,"split":r.split,"arm":arm,"seed":seed,
                         "schedule":MENU[j],"mix":alpha,"scores":vector.tolist(),
                         "feature_seconds":fs if uses_features else 0.,
                         "inference_seconds":seconds,"pair_scans":scans if uses_features else 0})
        add("source_fixed",-1,base,0.,0.,False)
        for arm in ("ridge","knn"):
            t=time.perf_counter();p=classical_predict(state["classical"],x,i,arm)[0]
            add(arm,-1,p,m["models"][arm]["mix"],time.perf_counter()-t,True)
        for (seed,arm),model in models.items():
            t=time.perf_counter();p=model(x,i)[0]
            add(arm,seed,p,m["models"][f"{seed}:{arm}"]["mix"],time.perf_counter()-t,True)
    return rows,feature


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["split"],r["arm"]].append(r)
    out={}
    for (split,arm),v in sorted(groups.items()):
        out[f"{split}:{arm}"]={"trials":len(v),"full":sum(r["full"] for r in v),
            "coverage":sum(r["coverage"] for r in v)/len(v),
            "utility":sum(r["utility"] for r in v)/len(v),
            "mean_work":sum(r["total_work"] for r in v)/len(v),
            "mean_search_seconds":sum(r["seconds"] for r in v)/len(v),
            "mean_selector_seconds":sum(r.get("feature_seconds",0)+r.get("inference_seconds",0)
                                         for r in v)/len(v),
            "choices":dict(Counter(r["schedule"] for r in v)),
            "per_seed":{str(s):{"trials":sum(r["seed"]==s for r in v),
                              "full":sum(r["full"] for r in v if r["seed"]==s)}
                        for s in sorted({r["seed"] for r in v})}}
    return out


def contrast(rows,split,left,right,key):
    idx=defaultdict(list)
    for r in rows:
        if r["split"]==split:idx[r["root"],r["arm"]].append(float(r[key]))
    ids=sorted({p[0] for p in idx});mean=lambda v:sum(v)/len(v)
    diffs=[mean(idx[i,left])-mean(idx[i,right]) for i in ids]
    rng=random.Random(f"olr06g/{split}/{left}/{right}/{key}")
    draws=sorted(mean([rng.choice(diffs) for _ in ids]) for _ in range(5000))
    return {"difference":mean(diffs),"ci95":[draws[125],draws[4875]],"roots":len(ids)}


def score(bank,fit,out):
    if out.exists():raise ValueError("create-only evaluation")
    setup();_,groups=load(bank);m,state,models=fitted(bank,fit)
    roots=groups["evaluation"];out.mkdir(parents=True);start=time.perf_counter()
    frozen,feature=choices(roots,m,state,models)
    write(out/"frozen_choices.json",frozen);write(out/"features_meter.json",feature)
    # No candidate search has executed on these roots before choices freeze.
    matrix_rows=matrix(roots)
    write(out/"frozen_reference_plans.json",matrix_rows)
    for root in roots:
        row=exact(root);row.update(total_work=row["expansions"],rekeys=0)
        matrix_rows.append({"root":root.root,"split":root.split,"arm":"exact",**row})
    cert=certify(roots,matrix_rows,out)
    write(out/"native_reference_scores.json",matrix_rows)
    root_map={r.root:r for r in roots};index={}
    rows=[]
    for p in matrix_rows:
        k=len(root_map[p["root"]].targets)
        full=bool(p["joint_lemma"])
        if p["arm"]=="exact" and not full:raise ValueError("unrestricted native ceiling failed")
        r={"root":p["root"],"split":p["split"],"arm":p["arm"],"seed":-1,
           "schedule":p["arm"],"full":full,"coverage":len(p["found"])/k,
           "total_work":p["total_work"],"seconds":p["seconds"],
           "utility":float(full)+.25*len(p["found"])/k-.01*p["total_work"]/CAP,
           "joint_lemma":p["joint_lemma"],"native_tactics":sum(q["native_tactics"] for q in p["found"].values())}
        if p["arm"]!="exact" and p["total_work"]>CAP:raise ValueError("unpaid portfolio work")
        rows.append(r);index[p["root"],p["arm"]]=r
    for q in frozen:
        r=index[q["root"],q["schedule"]]
        # Copy only the preselected certified schedule, not any other row.
        rows.append({**r,**q})
    write(out/"policy_rows.json",rows)
    summary=summarize(rows)
    best={s:max(summary[f"{s}:{a}"]["full"] for a in MENU) for s in ("dev","transfer")}
    portfolio=len({r["root"] for r in rows if r["arm"] in MENU and r["full"]})
    contrasts={f"{s}:bi-{a}:{key}":contrast(rows,s,"bi",a,key)
               for s in ("dev","transfer") for a in
               ("one_way","bi1","local","dense","full","ridge","knn","source_fixed") for key in ("full","utility")}
    def overall(arm):
        v=[r for r in rows if r["arm"]==arm]
        return sum(r["full"] for r in v)/len(v)
    simple=max(overall(a) for a in ("one_way","bi1","local","dense","full"))
    classical=max(overall(a) for a in ("source_fixed","ridge","knn",*MENU))
    seed_rates=[sum(r["full"] for r in rows if r["arm"]=="bi" and r["seed"]==seed)/len(roots)
                for seed in SEEDS]
    names=("frozen_choices.json","features_meter.json","frozen_reference_plans.json",
           "native_reference_scores.json","policy_rows.json","native_episodes.json",
           "conjunctions.lean","conjunction_compiler.log","leaves/standalone.lean",
           "leaves/compiler.log","leaves/compiler_receipt.json")
    report={"schema":SCORE,"registry_sha256":sha256_file(bank/"manifest.json"),
            "fit_sha256":sha256_file(fit/"manifest.json"),"roots":len(roots),
            "menu_trials":len(roots)*len(MENU),"frozen_selector_trials":len(frozen),
            "policy_trials":len(rows),"certificate":cert,"scorecard":summary,
            "contrasts":contrasts,"hindsight_portfolio":portfolio,
            "posthoc_best_fixed_by_panel":best,"hindsight_gap":(portfolio-sum(best.values()))/len(roots),
            "bi_rate":overall("bi"),"best_simple_rate":simple,"best_classical_rate":classical,
            "bi_seed_rates":seed_rates,
            "development_gate":overall("bi")>classical and overall("bi")-simple>=.05
                    and all(r>classical for r in seed_rates),
            "architecture_promotion_eligible":False,
            "elapsed_seconds":time.perf_counter()-start,"files":files(out,names)}
    write(out/"manifest.json",report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest="op",required=True)
    for op in ("freeze","fit","score"):
        a=sp.add_parser(op);a.add_argument("--out",type=Path,required=True)
        if op=="freeze":a.add_argument("--old-bank",type=Path,required=True)
        else:a.add_argument("--bank",type=Path,required=True)
        if op=="score":a.add_argument("--fit",type=Path,required=True)
    args=p.parse_args()
    if args.op=="freeze":m=freeze(args.old_bank,args.out)
    elif args.op=="fit":m=source(args.bank,args.out)
    else:m=score(args.bank,args.fit,args.out)
    print(canonical(m),flush=True)


if __name__=="__main__":main()
