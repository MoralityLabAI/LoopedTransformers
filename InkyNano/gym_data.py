"""Convert native traces to strictly causal segment-demand training arrays."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from routing import parse_rows


def build(roster, rows, horizon):
    if horizon < 0:
        raise ValueError("negative horizon")
    meta, families = {}, {}
    for r in roster:
        key = r["root"], r["turn"]
        if key in meta or r["split"] not in ("train","dev"):
            raise ValueError("duplicate or non-development roster")
        f = r["family"]
        if f in families and families[f] != r["split"]:
            raise ValueError("family leakage")
        families[f] = r["split"]
        meta[key] = r
    root_splits = defaultdict(set)
    for r in roster:
        root_splits[r["root"]].add(r["split"])
    if any(len(v)>1 for v in root_splits.values()):
        raise ValueError("root leakage")
    traces = defaultdict(dict)
    for (root,turn,t,l,p), weights in rows:
        key = root,turn
        if key not in meta:
            raise ValueError("trace missing roster")
        if t >= len(meta[key]["token_ids"]):
            raise ValueError("token out of bounds")
        if t not in traces[key]:
            traces[key][t] = (np.zeros((24,128),np.float32),np.zeros(24,np.int32))
        mass,count = traces[key][t]
        for e,w in weights.items():
            mass[l,e] += w
        count[l] += 1
    if set(traces) != set(meta):
        raise ValueError("roster/trace mismatch")
    examples = []
    def norm(a):
        return a / np.maximum(a.sum(axis=-1,keepdims=True),1e-12)
    for key,r in sorted(meta.items()):
        n, p = len(r["token_ids"]), r["prompt_len"]
        if not 0 < p < n or set(traces[key]) != set(range(n)):
            raise ValueError("invalid prompt boundary or incomplete trace")
        masses,counts = zip(*(traces[key][t] for t in range(n)))
        masses,counts = np.stack(masses),np.stack(counts)
        expected = np.ones(24,np.int32); expected[6:18]=3
        if not np.all(counts == expected):
            raise ValueError("incomplete native visits")
        prompt = masses[:p].sum(axis=0)
        history = prompt.copy()
        h = horizon or n-p
        for start in range(p,n,h):
            end = min(n,start+h)
            x = np.concatenate((norm(prompt),norm(history)),axis=-1)
            target = masses[start:end].sum(axis=0)
            examples.append((x,norm(target),counts[start:end].sum(axis=0),key[0],key[1],
                             r["family"],r["split"],start,end))
            history += target
    if not examples:
        raise ValueError("empty examples")
    x,y,c,root,turn,family,split,start,end = zip(*examples)
    return dict(x=np.stack(x),y=np.stack(y),visit_count=np.stack(c),root=np.array(root),
                turn=np.array(turn),family=np.array(family),split=np.array(split),
                start=np.array(start),end=np.array(end),horizon=np.array(horizon))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roster",type=Path,required=True)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    roster=[json.loads(line) for line in a.roster.read_text(encoding="utf8").splitlines() if line.strip()]
    with a.trace.open(encoding="utf8") as f:
        rows=parse_rows(f,strict=True)
    a.out.mkdir(parents=True,exist_ok=True)
    for h in (0,32,64,128):
        data=build(roster,rows,h)
        np.savez_compressed(a.out/f"demand_H{h}.npz",**data)
        print(h,len(data["x"]),len(set(data["root"])))


if __name__=="__main__":
    main()
