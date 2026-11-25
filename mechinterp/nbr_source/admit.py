"""Source-only NBR prospective candidate admission. Never imports a model.

Run with pinned historical receipts as specified in SOURCE_PROTOCOL.md.
The output remains a candidate until historical coverage review, independent
audit and a separate scientific registration commit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from mechinterp.neuron_basis import roster as R

SEED=202611250901
COUNT=24
MAX_ATTEMPTS=1200
SPLITS=("cal","eval","rep")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def digits(s):
    if isinstance(s,str):
        s=s.replace(".","0")
    a=tuple(map(int,s))
    if len(a)!=81 or not all(0<=x<=9 for x in a):
        raise ValueError("invalid history grid")
    return a

def historical(ls11,ls12,ls08,loopscope):
    from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
    from mechinterp.sae_causal_transfer.native_jvp_v1 import families as S3
    from mechinterp.sae_causal_transfer.native_jvp_v2 import families as S4
    from mechinterp.sae_causal_transfer.native_singular_v1 import families as S5
    from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as S6
    from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as S7
    old={}
    def add(name,rows,q="x",a="target"):
        old[name]=[(digits(r[q]),digits(r[a])) for r in rows]
        if not old[name]:
            raise ValueError("empty historical cohort "+name)
    a,_=S2.build_rosters(pilot=False); b,_=S3.build_new(pilot=False)
    add("LS-SAE02",[r for g in a.values() for r in g])
    add("LS-SAE03",[r for g in b.values() for r in g])
    for key,m,names in [("LS-SAE04",S4,S4.BASES),("LS-SAE05",S5,S5.SEEDS),
                         ("LS-SAE06",S6,S6.SEEDS),("LS-SAE07",S7,S7.BASES)]:
        add(key,[r for n in names for r in m.roster(n)])
    j=json.loads(ls11.read_text());add("LS-SAE11",[r for g in j["rosters"].values() for r in g])
    j=json.loads(ls12.read_text());add("LS-SAE12",j["pool"])
    j=json.loads(ls08.read_text());add("LS-SAE08",[r for g in ("test_52","test_54") for r in j["families"][g]["source_roster"]])
    j=json.loads(loopscope.read_text())
    add("LoopScope02-items",j["items"],"question","answer")
    add("LoopScope02-shots",j["shots"],"question","answer")
    return old

def singles(g):
    """Synchronous naked/hidden singles using visible clues only."""
    h=list(g)
    while True:
        c=[set(range(1,10))-{h[j] for j in R.PEERS[i]} if not h[i] else set()
           for i in range(81)]
        a={i:next(iter(v)) for i,v in enumerate(c) if len(v)==1}
        for unit in R.UNITS:
            for d in set(range(1,10))-{h[i] for i in unit}:
                idx=[i for i in unit if d in c[i]]
                if len(idx)==1:
                    if idx[0] in a and a[idx[0]]!=d:
                        raise ValueError("inconsistent propagation")
                    a[idx[0]]=d
        if not a:
            return h.count(0)
        for i,d in a.items():
            h[i]=d

def candidate(rng,holes):
    g=R.new_grid(rng); q=g[:]
    order=list(range(81));rng.shuffle(order)
    for i in order:
        old=q[i];q[i]=0
        if len(R.solve_mrv(q))!=1:
            q[i]=old
        if q.count(0)==holes:
            return q,g
    return None,None

def make(old,count=COUNT):
    rng=random.Random(SEED)
    inputs={q for g in old.values() for q,a in g}
    answers={a for g in old.values() for q,a in g}
    inv={R.orbit_invariant(list(q)) for q in inputs}
    rows=[]; rejection={"dig":0,"difficulty":0,"historical_or_new_collision":0}
    for split in SPLITS:
        for stratum,holes in (("singles",52),("stalled",56)):
            for k in range(count):
                for attempt in range(1,MAX_ATTEMPTS+1):
                    q,g=candidate(rng,holes)
                    if q is None:
                        rejection["dig"]+=1;continue
                    rem=singles(q)
                    if (stratum=="singles" and rem!=0) or (stratum=="stalled" and rem<8):
                        rejection["difficulty"]+=1;continue
                    key=R.orbit_invariant(q)
                    if tuple(q) in inputs or tuple(g) in answers or key in inv:
                        rejection["historical_or_new_collision"]+=1;continue
                    if R.solve_mrv(q)!=[tuple(g)] or R.solve_exact(q)!=[tuple(g)]:
                        raise RuntimeError("two source solvers disagree")
                    inputs.add(tuple(q));answers.add(tuple(g));inv.add(key)
                    rows.append({"id":f"{split}-{stratum}-{k:02d}","split":split,
                        "stratum":stratum,"holes":holes,"singles_remaining":rem,
                        "attempt":attempt,"question":"".join(str(v) if v else "." for v in q),
                        "answer":"".join(map(str,g))})
                    break
                else:
                    raise RuntimeError(f"fixed source rejection cap exceeded: {split}/{stratum}/{k}")
            print("source",split,stratum,len(rows),flush=True)
    return rows,rejection

def audit(rows,old,count):
    inputs={q for g in old.values() for q,a in g}
    answers={a for g in old.values() for q,a in g}
    inv={R.orbit_invariant(list(q)) for q in inputs}
    assert len(rows)==len(SPLITS)*2*count
    seen=set();counts={}
    for r in rows:
        q,a=digits(r["question"]),digits(r["answer"])
        key=R.orbit_invariant(list(q))
        assert r["id"] not in seen and q not in inputs and a not in answers and key not in inv
        assert len(set(a))==9 and all(set(a[i] for i in u)==set(range(1,10)) for u in R.UNITS)
        assert all(v==0 or v==a[i] for i,v in enumerate(q))
        assert R.solve_exact(q)==[a] and R.solve_mrv(q)==[a]
        rem=singles(q)
        assert rem==r["singles_remaining"] and q.count(0)==r["holes"]
        assert (r["stratum"]=="singles" and q.count(0)==52 and rem==0) or (r["stratum"]=="stalled" and q.count(0)==56 and rem>=8)
        tag=(r["split"],r["stratum"]);counts[tag]=counts.get(tag,0)+1
        seen.add(r["id"]);inputs.add(q);answers.add(a);inv.add(key)
    assert counts=={(s,t):count for s in SPLITS for t in ("singles","stalled")}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("ls11","ls12","ls08","loopscope","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--count",type=int,default=COUNT)
    a=p.parse_args()
    if a.count<1 or a.out.exists():
        p.error("positive count and new output required")
    old=historical(a.ls11,a.ls12,a.ls08,a.loopscope)
    rows,rejections=make(old,a.count)
    a.out.mkdir(parents=True)
    path=a.out/"roster.json"
    path.write_text(json.dumps(rows,indent=2)+"\n")
    audit(json.loads(path.read_text()),old,a.count)
    receipt={"schema":"NBR/source-candidate-v1","model_inference_executed":False,
       "candidate_only":True,"seed":SEED,"count_per_split_stratum":a.count,
       "roster_sha256":sha(path),"historical_counts":{k:len(v) for k,v in old.items()},
       "historical_receipt_sha256":{k:sha(getattr(a,k)) for k in ("ls11","ls12","ls08","loopscope")},
       "n_rows":len(rows),"rejections":rejections,"source_recheck":"PASS",
       "symmetry_method":"distinct necessary invariants; collisions conservatively rejected",
       "limitations":"named historical cohorts only; other Sudoku campaigns require review; no training-set disjointness claim"}
    (a.out/"source-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))

if __name__=="__main__":main()
