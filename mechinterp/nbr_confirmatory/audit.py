"""Independent source and gate audit for registered NBR-01/01B.

No model imports; source audit runs before checkpoint access.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ROSTER_SHA="763ea63c8565245928a7dd59f7b56b7916fad6e2f8d3126075a801c56a36f224"
PILOT_SHA="bcf9b3c47f5703d8e33222bbb66f34e057277476c44a555b896dc5325dd50a5b"
S8_SHA="1e90b2bb541291f34fb760cd0e02a74f6fd522cdb74bc0436454dc57334a3cc9"
CKPT_SHA="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
UNITS=([tuple(range(r*9,r*9+9)) for r in range(9)]+
       [tuple(range(c,81,9)) for c in range(9)]+
       [tuple(9*(r+i)+c+j for i in range(3) for j in range(3))
        for r in range(0,9,3) for c in range(0,9,3)])
ROWCOLS=[];COLROWS=[0]*324
for i in range(81):
    r,c=divmod(i,9);b=(r//3)*3+c//3
    for d in range(9):
        k=9*i+d; cs=(i,81+9*r+d,162+9*c+d,243+9*b+d)
        ROWCOLS.append(cs)
        for j in cs:COLROWS[j]|=1<<k
ROWCONF=[0]*729
for k,cs in enumerate(ROWCOLS):
    for j in cs:ROWCONF[k]|=COLROWS[j]
ROWMASK=[sum(1<<j for j in cs) for cs in ROWCOLS]

def sha(p):
    return p and hashlib.sha256(p.read_bytes()).hexdigest()

def solve(q):
    avail=(1<<729)-1;cols=(1<<324)-1;path=[];found=[]
    for i,v in enumerate(q):
        if not v:continue
        k=9*i+v-1
        if not (avail>>k)&1:return []
        avail&=~ROWCONF[k];cols&=~ROWMASK[k];path.append(k)
    def walk(a,c):
        if len(found)>=2:return
        if not c:
            ans=[0]*81
            for k in path:ans[k//9]=k%9+1
            found.append(tuple(ans));return
        scan=c;options=None
        while scan:
            bit=scan&-scan;scan-=bit;j=bit.bit_length()-1
            cand=COLROWS[j]&a
            if not cand:return
            if options is None or cand.bit_count()<options.bit_count():
                options=cand
                if cand.bit_count()==1:break
        while options and len(found)<2:
            bit=options&-options;options-=bit;k=bit.bit_length()-1
            path.append(k);walk(a&~ROWCONF[k],c&~ROWMASK[k]);path.pop()
    walk(avail,cols);return found

def singles(q):
    g=list(q)
    while True:
        can={i:set(range(1,10))-set(g[j] for u in UNITS if i in u for j in u)
             for i in range(81) if not g[i]}
        moved=False
        for i,v in can.items():
            if len(v)==1:g[i]=next(iter(v));moved=True;break
        if moved:continue
        for u in UNITS:
            for d in set(range(1,10))-{g[i] for i in u}:
                cells=[i for i in u if i in can and d in can[i]]
                if len(cells)==1:g[cells[0]]=d;moved=True;break
            if moved:break
        if not moved:return g.count(0)

def fingerprint(q):
    def groups(lines):
        return tuple(sorted(tuple(sorted(sum(bool(x) for x in line)
                      for line in lines[k:k+3])) for k in range(0,9,3)))
    a=[q[i*9:(i+1)*9] for i in range(9)];b=list(zip(*a))
    return (tuple(sorted([groups(a),groups(b)])),
            tuple(sorted(sum(bool(q[i]) for i in u) for u in UNITS[-9:])),
            tuple(sorted(q.count(d) for d in range(1,10))))

def digits(s):
    return tuple(map(int,s.replace(".","0") if isinstance(s,str) else s))

def source(roster,receipt,pilot,s8):
    assert sha(roster)==ROSTER_SHA and sha(pilot)==PILOT_SHA and sha(s8)==S8_SHA
    rc=json.loads(receipt.read_text())
    assert rc["roster_sha256"]==ROSTER_SHA and rc["n_rows"]==144
    assert not rc["model_inference_executed"] and rc["source_recheck"]=="PASS"
    assert sum(rc["historical_counts"].values())==573
    old=[(digits(r["question"]),digits(r["answer"])) for r in json.loads(pilot.read_text())]
    old.extend((digits(r["x"]),digits(r["target"])) for rs in
               json.loads(s8.read_text())["families"].values() for r in rs)
    assert len(old)==26
    qs={q for q,a in old};answers={a for q,a in old};inv={fingerprint(q) for q,a in old}
    ids=set();counts=collections.Counter()
    rows=json.loads(roster.read_text())
    for r in rows:
        q,a=digits(r["question"]),digits(r["answer"]);f=fingerprint(q)
        assert len(q)==len(a)==81 and all(0<=v<=9 for v in q)
        assert r["id"] not in ids and q not in qs and a not in answers and f not in inv
        assert all(set(a[i] for i in u)==set(range(1,10)) for u in UNITS)
        assert all(not v or v==a[i] for i,v in enumerate(q))
        assert solve(q)==[a]
        n=singles(q)
        assert n==r["singles_remaining"] and q.count(0)==r["holes"]
        assert (r["stratum"]=="singles" and n==0 and r["holes"]==52) or \
               (r["stratum"]=="stalled" and n>=8 and r["holes"]==56)
        counts[r["split"],r["stratum"]]+=1
        ids.add(r["id"]);qs.add(q);answers.add(a);inv.add(f)
    assert counts=={(s,t):24 for s in ("cal","eval","rep") for t in ("singles","stalled")}
    return {"source":"PASS","roster_sha256":ROSTER_SHA,"rows":len(rows),
            "historical_named_records":577,"independent_exact_cover":True}

def boot(values):
    if not values:return {"n":0,"mean":None,"ci95":None}
    v=np.asarray(values,dtype=np.float64)
    assert np.isfinite(v).all()
    rng=np.random.default_rng(20261125)
    draws=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
    return {"n":len(v),"mean":float(v.mean()),
            "ci95":np.quantile(draws,[.025,.975]).tolist()}

def gate(roster,native,fixed):
    assert sha(roster)==ROSTER_SHA
    ro={r["id"]:r for r in json.loads(roster.read_text())}
    n=json.loads((native/"report.json").read_text())
    f=json.loads(fixed.read_text())
    assert n["roster_sha256"]==ROSTER_SHA and n["checkpoint_sha256"]==CKPT_SHA
    assert f["checkpoint_sha256"]==CKPT_SHA and f["parent_report_sha256"]==sha(native/"report.json")
    assert f["seed"]==202611251031 and f["draws"]==8 and f["mix_per_draw"]==256
    nb={b["id"]:b for b in n["boards"]};fb={b["id"]:b for b in f["boards"]}
    assert set(nb)==set(fb)==set(ro) and len(n["boards"])==len(f["boards"])==144
    assert len({b["receipt"] for b in nb.values()})==144
    for i,b in nb.items():
        assert b["split"]==ro[i]["split"]==fb[i]["split"]
        assert sha(native/b["receipt"])==b["sha256"]
        blanks=[k for k,x in enumerate(ro[i]["question"]) if x in ".0"]
        assert b["site"]==blanks[len(blanks)//2]
    out={}
    for name,mode,threshold in (("NBR-01","raw",.10),("NBR-01B","residual",.05)):
        families={}
        for s in ("cal","eval","rep"):
            ids=[i for i in ro if ro[i]["split"]==s]
            def summary(ids):
                vals=[]
                for i in ids:
                    if name=="NBR-01":
                        v=nb[i]["stability"]["3-6/L0"][mode]
                        if v is not None:
                            assert abs(v["advantage"]-(v["same"]-v["null"]))<1e-12
                            vals.append(v["advantage"])
                    else:
                        v=fb[i]["stats"]["3-6/L0"][mode]
                        if v["advantage"] is not None:
                            assert abs(v["fixed_mix"]-np.mean(v["mix_draws"]))<1e-12
                            assert abs(v["advantage"]-(v["native"]-v["fixed_mix"]))<1e-12
                            vals.append(v["advantage"])
                return boot(vals)
            g=summary(ids)
            verdict=("INCOMPLETE" if g["n"]<40 else
                     "PASS" if g["mean"]>=threshold and g["ci95"][0]>0 else "FAIL")
            families[s]={**g,"verdict":verdict if s!="cal" else "DIAGNOSTIC",
                         "strata":{t:summary([i for i in ids if ro[i]["stratum"]==t])
                                   for t in ("singles","stalled")}}
        v=[families[s]["verdict"] for s in ("eval","rep")]
        out[name]={"threshold":threshold,"families":families,"verdict":
            "INCOMPLETE" if "INCOMPLETE" in v else "PASS" if v==["PASS","PASS"] else "FAIL"}
    return {"schema":"NBR/registered-gates-v1","roster_sha256":ROSTER_SHA,
            "gates":out,"native_specificity_supported":all(g["verdict"]=="PASS" for g in out.values()),
            "audit_scope":"hashes and arithmetic; does not independently rerun model, Jacobians or random mixes"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("source","gate"),required=True)
    for k in ("roster","receipt","pilot","s8","native","fixed","out"):
        p.add_argument("--"+k,type=Path)
    a=p.parse_args()
    r=source(a.roster,a.receipt,a.pilot,a.s8) if a.stage=="source" else gate(a.roster,a.native,a.fixed)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps(r,indent=2,allow_nan=False))

if __name__=="__main__":main()
