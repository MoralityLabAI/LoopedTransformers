"""07B native library collector, CPU-only receipt policies and immutable audit.
Run through run_olr07b.py so resource limits precede importing torch.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from . import olr07a_library as A

MENU=A.MENU
BUDGET=4
SEED=20270119
HUMAN_HB=20000
OLD_MANIFEST="846958ede542bedd510d865b3559e6f4711a97963583084c2d1028e70da2328e"
OLD_STATES="029b8d591851c7034736db1c1ec09c6e956c80e339fefb8c42e7a0429ce18151"
ROSTER={
 "train": {"Nat":("Bitwise","Compare","Div","Dvd","Gcd","Linear","Log2","Mod"),
           "List":("Count","Erase","Find","MapIdx","MinMax","Nat","OfFn","Pairwise","Perm","Range","Sublist","TakeDrop","Zip"),
           "Option":("Attach","List","Monadic"),"Char":("Lemmas",),"String":("Lemmas",)},
 "calibration":{"Int":("Bitwise","Compare","DivMod","Gcd","Linear","LemmasAux","Order","Pow")},
 "evaluation":{"Array":("Attach","Count","Erase","Extract","Find","InsertIdx","MapIdx","Mem","OfFn","Perm","TakeDrop","Zip"),
               "Vector":("Erase","Extract","Find","Lemmas"),"Fin":("Bitwise","Fold","Log2"),"UInt":("Bitwise","Lemmas")}
}
SEEDS=(17,29,43)
ARMS=("bi","one_way","bi1","binary","direct")
CATS=("untried","closed","uncertified","open:1","open:2","open:3","resource","parse","type","no_match","rejected","malformed","binary_fail")
D=1024
E=64
W=64
N=len(MENU)

def hash_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+".tmp")
    q.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(q,p)
def stamp(root,names):return {n:sha(root/n) for n in names}
def check(root,files):
    for n,h in files.items():
        if not (root/n).is_file() or sha(root/n)!=h:raise ValueError("checkpoint changed: "+str(root/n))
def choose_files(profile):
    out=[]
    for role,families in ROSTER.items():
        rows=[(role,fam,f"src/Init/Data/{fam}/{f}.lean") for fam,fs in families.items() for f in fs]
        out.extend(rows if profile=="full" else rows[:2])
    return out

def old_exclusions(path):
    with zipfile.ZipFile(path) as z:
        raw=z.read("manifest.json")
        states=z.read("bank/states.json")
        if hash_bytes(raw)!=OLD_MANIFEST or hash_bytes(states)!=OLD_STATES:
            raise ValueError("expected original verified 07A artifact")
        if json.loads(raw)["files"]["bank/states.json"]!=OLD_STATES:
            raise ValueError("old registry identity mismatch")
    rows=json.loads(states)
    return set(r["goal"] for r in rows),{(r["file"],r["decl"]) for r in rows},{r["file"] for r in rows}

def trace(path,out,timeout,**kw):
    t=time.perf_counter()
    with LeanRepl(imports=("Lean",),timeout=timeout,exe=config.REPL_EXE,cwd=config.LEAN_DIR,
                  env=config.lean_env()) as r:
        if str(r.version)!="4.19.0":raise ValueError("wrong Lean version")
        rep=r.request(op="trace",path=str(path.resolve()),out=str(out.resolve()),**kw)
    if not rep.get("ok"):raise ValueError("trace failed "+str(rep)[:500])
    return rep,time.perf_counter()-t

def checkpoint(dir,fn):
    if (dir/"done.json").exists():
        m=read(dir/"done.json");check(dir,m["files"]);return m
    if dir.exists():shutil.rmtree(dir) # only an unfinished stage inside the run's own tree
    dir.mkdir(parents=True)
    stats=fn(dir)
    names=[p.relative_to(dir).as_posix() for p in dir.rglob("*") if p.is_file()]
    m={"stats":stats,"files":stamp(dir,names)}
    write(dir/"done.json",m);return m

def trace_file(source,out,item,timeout):
    role,fam,rel=item
    path=source/rel
    def work(d):
        rep,secs=trace(path,d/"trace.jsonl",timeout,cand_every=0)
        write(d/"report.json",rep)
        if rep.get("errors"):raise ValueError("source file elaboration errors: "+rel)
        return {"file":rel,"seconds":secs,"rows":rep.get("steps",0)}
    return checkpoint(out/"trace"/(fam+"_"+path.stem),work)

def eligible(raw,rep,rel,old_decl):
    valid={d["decl"] for d in rep["decls"] if d["in_env"] and
           isinstance(d["axioms"],list) and set(d["axioms"])<=A.AXIOMS and
           d["decl"] not in ("","[anonymous]","_anonymous")}
    seen=set()
    for r in raw:
        pos=(r["line"],r["col"])
        text=r.get("tactic","")
        # Ignore source token atoms, retaining the first outer tactic row.
        kind=str(r.get("kind",""))
        if kind.startswith("«") and kind.endswith("»") and kind[1:-1]==text.strip():continue
        if pos in seen:continue
        seen.add(pos)
        if (r["decl"] not in valid or (rel,r["decl"]) in old_decl or
            len(r.get("goals_before",[]))!=1 or r.get("goals_after")!=[] or
            not text or "…" in text or "…" in r["goals_before"][0] or
            re.match(r"\s*(decreasing_by|termination_by|termination_by\?|where)\b",text)):
            continue
        human=A.human_text(r)
        if A.FORBIDDEN.search(human):continue
        goal=r["goals_before"][0]
        ident=A.digest([SEED,rel,r["decl"],*pos,goal])
        yield ident,r,human,goal

def registry(source,out,prior,profile,workers,timeout):
    old_goal,old_decl,old_files=old_exclusions(prior)
    roster=choose_files(profile)
    if any(rel in old_files for _,_,rel in roster):raise ValueError("07A file reused")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for m in pool.map(lambda item:trace_file(source,out,item,timeout),roster):
            print("trace",m["stats"],flush=True)
    rows=[];seen=set(old_goal);cap=48 if profile=="full" else 4
    for role,fam,rel in roster:
        p=source/rel;d=out/"trace"/(fam+"_"+p.stem)
        candidates=sorted(eligible(A.read_lines(d/"trace.jsonl"),read(d/"report.json"),rel,old_decl),key=lambda x:x[0])
        picked=[];decls=set()
        for ident,r,human,goal in candidates:
            if r["decl"] in decls or goal in seen:continue
            decls.add(r["decl"]);seen.add(goal)
            picked.append({"id":ident,"role":role,"family":fam,"file":rel,"file_sha256":sha(p),
                           "decl":r["decl"],"line":r["line"],"col":r["col"],"goal":goal,"human":human})
            if len(picked)==cap:break
        rows.extend(picked)
    p=out/"registry.json"
    if p.exists() and read(p)!=rows:raise ValueError("frozen registry changed")
    write(p,rows);return rows

def grade_file(source,out,items,timeout):
    rel=items[0]["file"];role=items[0]["role"];fam=items[0]["family"];p=source/rel
    if any(sha(p)!=r["file_sha256"] for r in items):raise ValueError("source changed")
    def work(d):
        runs=[];times={}
        for typ,hb in (("candidate",A.HB),("human",HUMAN_HB)):
            sel=[{"line":r["line"],"col":r["col"],"tactics":list(MENU) if typ=="candidate" else [r["human"]]} for r in items]
            for rep in range(2):
                dst=d/f"{typ}{rep}.jsonl"
                meta,secs=trace(p,dst,timeout,select=sel,cand_heartbeats=hb)
                if meta.get("errors"):raise ValueError("source select errors "+rel)
                write(d/f"{typ}{rep}.report.json",meta);times[f"{typ}{rep}"]=secs
                rr=A.read_lines(dst);idx={(r["line"],r["col"]):r for r in rr}
                if len(idx)!=len(rr):raise ValueError("duplicate source positions")
                runs.append(idx)
        rows=[]
        for r in items:
            pos=(r["line"],r["col"]);xs=[a.get(pos) for a in runs]
            good=[x is not None and x.get("decl")==r["decl"] and x.get("goals")==1 and
                  x.get("goals_before")==[r["goal"]] for x in xs]
            if any(good)!=all(good):raise ValueError("source state replay disagreement")
            if not all(good):
                rows.append({**r,"scored":False,"reason":"unreached_or_mismatch"});continue
            if any(x.get("scope_version")!="pre_command_v1" for x in xs):
                raise ValueError("unsafe legacy tracer: rebuild before collecting receipts")
            scopes=[x.get("env_scope")=="before" and x.get("decl_in_env_before") is False for x in xs]
            if any(scopes)!=all(scopes):raise ValueError("native scope replay drift")
            if not all(scopes):
                rows.append({**r,"scored":False,"reason":"not_portable_before_command"});continue
            cert=[x.get("main_prop") is True and x.get("main_closed") is True for x in xs]
            if any(cert)!=all(cert):raise ValueError("native certifiability changed")
            if not all(cert):
                rows.append({**r,"scored":False,"reason":"uncertifiable"});continue
            maps=[{v["tactic"]:v for v in x["results"]} for x in xs]
            if set(maps[0])!=set(MENU) or set(maps[1])!=set(MENU) or any(
               set(m)!= {r["human"]} for m in maps[2:]):raise ValueError("menu drift")
            for a,b in ((maps[0],maps[1]),(maps[2],maps[3])):
                if any((A.closed(a[k]),A.category(a[k]))!=(A.closed(b[k]),A.category(b[k])) for k in a):
                    raise ValueError("native receipt replay drift "+r["id"])
            rows.append({**r,"scored":True,"human_closed":A.closed(maps[2][r["human"]]),
                         "human_receipts":[m[r["human"]] for m in maps[2:]],
                         "receipts":[maps[0][k] for k in MENU],"replay_receipts":[maps[1][k] for k in MENU]})
        write(d/"rows.json",rows)
        return {"file":rel,"role":role,"seconds":times,"selected":len(items),
                "scored":sum(r["scored"] for r in rows)}
    d=out/"native"/role/(fam+"_"+p.stem)
    m=checkpoint(d,work);rows=read(d/"rows.json")
    if [r["id"] for r in rows]!=[r["id"] for r in items]:raise ValueError("receipt roster changed")
    print("native",m["stats"],flush=True)
    return rows

def collect(source,out,rows,roles,workers,timeout):
    groups=defaultdict(list)
    for r in rows:
        if r["role"] in roles:groups[r["file"]].append(r)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for batch in pool.map(lambda a:grade_file(source,out,a,timeout),groups.values()) for r in batch]

def census(rows):
    n=len(rows);valid=[r for r in rows if r["scored"]]
    return {"selected":n,"scored":len(valid),"human_closed":sum(r["human_closed"] for r in valid),
            "human_rate":sum(r["human_closed"] for r in valid)/max(1,len(valid)),
            "certified_actions":sum(A.closed(x) for r in valid for x in r["receipts"]),
            "scored_menu_receipts":len(valid)*N,
            "exclusions":dict(Counter(r.get("reason","unknown") for r in rows if not r["scored"]))}

def classical(train,rows):
    if not train:return [],{},None
    order=A.fixed_order(train)
    trees={"binary":A.Empirical(train,False),"diagnostic":A.Empirical(train,True)}
    episodes=[]
    for r in rows:
        for a in A.ARMS:
            episodes.append(A.episode(r,a,order,trees))
        # Keep memory bounded even on a much larger source corpus.
        for tree in trees.values():tree.cache.clear()
    s=stats(episodes)
    best=max(A.ARMS,key=lambda a:(s[a]["closed"],-s[a]["mean_calls"],-A.ARMS.index(a))) if rows else None
    return episodes,s,best

def stats(episodes):
    groups=defaultdict(list)
    for e in episodes:groups[e["arm"]].append(e)
    return {a:{"roots":len(rs),"closed":sum(r["closed"] for r in rs),
               "rate":sum(r["closed"] for r in rs)/len(rs),"mean_calls":sum(r["calls"] for r in rs)/len(rs),
               "by_file":{f:{"roots":sum(r["file"]==f for r in rs),
                            "closed":sum(r["file"]==f and r["closed"] for r in rs)} for f in sorted({r["file"] for r in rs})}}
            for a,rs in groups.items()}

def hvec(text,d):
    tokens=re.findall(r"\w+|[^\w\s]",text.lower(),flags=re.UNICODE)
    x=torch.zeros(d)
    for tok in itertools.chain(tokens,(a+"\x1f"+b for a,b in zip(tokens,tokens[1:]))):
        h=int.from_bytes(hashlib.blake2b(tok.encode(),digest_size=8).digest(),"little")
        x[h%d]+=1 if h>>63==0 else -1
    return torch.sign(x)*torch.log1p(x.abs())/(max(1,len(tokens))**.5)

def receipt_features(r):
    x=torch.zeros(len(CATS)+E)
    c=A.category(r);x[CATS.index(c)]=1
    x[len(CATS):]=hvec(str(r.get("error","")),E)
    return x

def row_features(row):
    return hvec(row["goal"],D),torch.stack([receipt_features(x) for x in row["receipts"]]),torch.tensor(
        [A.closed(x) for x in row["receipts"]],dtype=torch.float32)

def frame(g,rs,y,history):
    # Only observed receipts are copied to the packet.
    x=torch.zeros_like(rs);x[:,0]=1
    mask=torch.ones(N,dtype=torch.bool)
    for j in history:x[j]=rs[j];mask[j]=False
    return g,x,y,mask

def dataset(rows):
    frames=[]
    for row in rows:
        g,rs,y=row_features(row)
        fail=[j for j in range(N) if not y[j]]
        h=[a for k in (1,2,3) for a in itertools.combinations(fail,k)]
        rng=random.Random(row["id"]+str(SEED));rng.shuffle(h)
        frames.extend(frame(g,rs,y,a) for a in [(),*h[:31]])
    return tuple(torch.stack(xs) for xs in zip(*frames))

class Reader(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.goal=nn.Sequential(nn.Linear(D,W),nn.GELU(),nn.LayerNorm(W))
        self.act=nn.Embedding(N,W)
        self.receipt=nn.Linear(len(CATS)+E,W)
        if arm=="direct":
            self.direct=nn.Sequential(nn.Linear(D+N*(len(CATS)+E),2*W),nn.GELU(),nn.Linear(2*W,N));return
        self.up=nn.Linear(W,W);self.down=nn.Linear(W,W)
        self.gcell=nn.Sequential(nn.Linear(3*W,2*W),nn.GELU(),nn.Linear(2*W,W))
        self.acell=nn.Sequential(nn.Linear(3*W,2*W),nn.GELU(),nn.Linear(2*W,W))
        self.gnorm=nn.LayerNorm(W);self.anorm=nn.LayerNorm(W)
        self.head=nn.Linear(W,1)
    def forward(self,g,r):
        if self.arm=="direct":return self.direct(torch.cat((g,r.flatten(1)),1))
        if self.arm=="binary":
            red=torch.zeros_like(r);red[:,:,0]=r[:,:,0];red[:,:,CATS.index("binary_fail")]=1-r[:,:,0]
            r=red
        g0=self.goal(g);x0=self.receipt(r)+self.act.weight[None,:,:]
        h=g0;x=x0+g0[:,None,:]
        for _ in range(1 if self.arm=="bi1" else 3):
            msg=self.up(x).mean(1)*float(self.arm!="one_way")
            h=self.gnorm(h+self.gcell(torch.cat((h,g0,msg),-1)))
            d=self.down(h)[:,None,:].expand_as(x)
            x=self.anorm(x+self.acell(torch.cat((x,x0,d),-1)))
        return self.head(x).squeeze(-1)

def weight_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():
        h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def fit(data,seed,arm,steps=400):
    g,r,y,mask=data
    torch.manual_seed(seed);m=Reader(arm);initial=weight_hash(m)
    gen=torch.Generator().manual_seed(seed+7019);sample=torch.randint(len(g),(steps,64),generator=gen)
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.005)
    t=time.perf_counter()
    for ix in sample:
        pred=m(g[ix],r[ix])
        loss=F.binary_cross_entropy_with_logits(pred[mask[ix]],y[ix][mask[ix]])
        if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    m.eval()
    return m,{"seed":seed,"arm":arm,"steps":steps,"parameters":sum(x.numel() for x in m.parameters()),
              "initial_sha256":initial,"sampler_sha256":hash_bytes(sample.numpy().tobytes()),
              "weights_sha256":weight_hash(m),"last_loss":float(loss.detach()),"seconds":time.perf_counter()-t}

@torch.no_grad()
def neural_episode(m,row,seed,arm):
    # Feature cache contains only immutable goal syntax. Receipt features are
    # built lazily after the selected action, never from the unseen matrix.
    g=hvec(row["goal"],D);r=torch.zeros(N,len(CATS)+E);r[:,0]=1
    mask=torch.ones(N,dtype=torch.bool);events=[];start=time.perf_counter()
    for _ in range(BUDGET):
        scores=m(g[None],r[None])[0].masked_fill(~mask,-torch.inf)
        j=int(scores.argmax());mask[j]=False
        rec=row["receipts"][j];ok=A.closed(rec)
        events.append({"action":MENU[j],"closed":ok,"status":A.category(rec)})
        if ok:break
        r[j]=receipt_features(rec)
    return {"id":row["id"],"role":row["role"],"family":row["family"],"file":row["file"],
            "arm":arm,"seed":seed,"closed":events[-1]["closed"],"calls":len(events),
            "events":events,"policy_seconds":time.perf_counter()-start}

def file_interval(rows,left,right):
    # Average repeated seeds within each theorem state before resampling files.
    index=defaultdict(list);file={}
    for r in rows:
        index[r["id"],r["arm"]].append(float(r["closed"]));file[r["id"]]=r["file"]
    diffs=defaultdict(list)
    for i,f in file.items():
        if (i,left) in index and (i,right) in index:
            a,b=index[i,left],index[i,right];diffs[f].append(sum(a)/len(a)-sum(b)/len(b))
    if not diffs:return None
    fs=sorted(diffs);rng=random.Random(f"07B/{left}/{right}")
    draws=[]
    for _ in range(2000):
        v=[d for f in rng.choices(fs,k=len(fs)) for d in diffs[f]]
        draws.append(sum(v)/len(v))
    draws.sort();allv=[v for rs in diffs.values() for v in rs]
    return {"difference":sum(allv)/len(allv),"ci95":[draws[50],draws[1950]],"files":len(fs)}

def contract(source,prior,profile):
    rev=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if rev!=A.COMMIT:raise ValueError("pin Lean source to "+A.COMMIT)
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no"],text=True).strip():
        raise ValueError("Lean source has modified tracked files")
    return {"schema":"olr07b.cpu.v1","source_commit":rev,"prior_sha256":sha(prior),"profile":profile,
            "menu":MENU,"budget":BUDGET,"candidate_hb":A.HB,"human_hb":HUMAN_HB,"seed":SEED,
            "torch":torch.__version__,"files":{rel:sha(source/rel) for _,_,rel in choose_files(profile)},
            "lean_sha256":sha(config.LEAN_EXE),"repl_sha256":sha(config.REPL_EXE),
            "code_sha256":sha(Path(__file__)),"reference_sha256":sha(Path(A.__file__))}

def run(source,prior,out,profile,workers,threads,timeout,collect_only=False):
    torch.set_num_threads(threads);torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    spec=contract(source,prior,profile);spec["collect_only"]=collect_only;spec["threads"]=threads
    spec=json.loads(json.dumps(spec))
    p=out/"contract.json"
    if p.exists() and read(p)!=spec:raise ValueError("resume contract changed; use a new output directory")
    write(p,spec)
    started=time.perf_counter()
    rows=registry(source,out,prior,profile,workers,timeout)
    sc=collect(source,out,rows,("train","calibration"),workers,timeout)
    train=[r for r in sc if r["role"]=="train" and r["scored"]]
    cal=[r for r in sc if r["role"]=="calibration" and r["scored"]]
    c={s:census([r for r in sc if r["role"]==s]) for s in ("train","calibration")}
    ce,cs,base=classical(train,cal);write(out/"cal_classical.json",ce)
    ceiling=sum(any(A.closed(x) for x in r["receipts"]) for r in cal)
    best=max((x["closed"] for x in cs.values()),default=0)
    valid=len(train)>=100 and len(cal)>=30 and min(x["human_rate"] for x in c.values())>=.95
    headroom=bool(cal) and (ceiling-best)/len(cal)>=.1 and best<=.8*len(cal)
    fit_gate=profile=="full" and valid and headroom and not collect_only
    models={};routes={};cal_neural=[]
    if fit_gate:
        data=dataset(train)
        for seed in SEEDS:
            signatures=[]
            for arm in ARMS:
                key=f"{seed}:{arm}";d=out/"fit"/f"{seed}_{arm}"
                def work(folder):
                    m,meta=fit(data,seed,arm)
                    torch.save(m.state_dict(),folder/"weights.pt")
                    ep=[neural_episode(m,r,seed,arm) for r in cal];write(folder/"cal.json",ep)
                    meta["calibration"]=stats(ep)[arm];write(folder/"fit.json",meta)
                    return meta
                cp=checkpoint(d,work);m=Reader(arm)
                m.load_state_dict(torch.load(d/"weights.pt",map_location="cpu",weights_only=True));m.eval()
                meta=cp["stats"]
                if weight_hash(m)!=meta["weights_sha256"]:raise ValueError("weight identity differs")
                if arm!="direct":signatures.append((meta["initial_sha256"],meta["sampler_sha256"]))
                z=meta["calibration"];b=cs[base]
                use=(z["closed"],-z["mean_calls"])>(b["closed"],-b["mean_calls"])
                routes[key]={"route":arm if use else base,"calibration":z,"fit":meta}
                models[key]=m;cal_neural.extend(read(d/"cal.json"))
                print("fit",key,routes[key]["route"],flush=True)
            if len(set(signatures))!=1:raise ValueError("matched reader initialization/sampler changed")
    write(out/"cal_neural.json",cal_neural)
    frozen={"fit_gate":fit_gate,"source_valid":valid,"cal_headroom":headroom,"cal_ceiling":ceiling,
            "cal_best":best,"base":base,"routes":routes,"train_count":len(train),"cal_count":len(cal)}
    f=out/"frozen_policy.json"
    if f.exists() and read(f)!=frozen:raise ValueError("frozen policy changed")
    write(f,frozen)
    # Evaluation candidate receipts do not exist until the policies freeze.
    ev=collect(source,out,rows,("evaluation",),workers,timeout)
    c["evaluation"]=census(ev)
    fresh=[r for r in ev if r["scored"]]
    ee,_,_=classical(train,fresh)
    base_rows={r["id"]:r for r in ee if r["arm"]==base}
    for key,m in models.items():
        seed,arm=key.split(":")
        for row in fresh:
            raw=neural_episode(m,row,int(seed),arm);ee.append(raw)
            selected=raw if routes[key]["route"]==arm else base_rows[row["id"]]
            ee.append({**selected,"arm":"routed_"+arm,"seed":int(seed)})
    write(out/"evaluation_episodes.json",ee)
    write(out/"all_receipts.json",sc+ev)
    es=stats(ee);valid_fresh=profile=="full" and valid and len(fresh)>=100 and c["evaluation"]["human_rate"]>=.95
    report={"schema":"olr07b.result.v1","profile":profile,"census":c,"source_gate":frozen,
            "native_valid":valid_fresh,"cal_classical":cs,"evaluation":es,
            "evaluation_ceiling":sum(any(A.closed(x) for x in r["receipts"]) for r in fresh),
            "contrasts":{a:file_interval(ee,"bi",a) for a in ("binary","one_way","bi1","direct",*A.ARMS)} if fit_gate else {},
            "diagnostic_contrast":file_interval(ee,"diagnostic","binary"),
            "model_training_updates":len(models)*400,"architecture_promotion_eligible":False,
            "seconds_this_invocation":time.perf_counter()-started}
    # A root/seed score is exploratory; no automatic promotion even on a pass.
    write(out/"summary.json",report);return report
