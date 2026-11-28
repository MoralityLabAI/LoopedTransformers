"""Family-macro paired replay; lightweight pre-request depth lookup."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

def load(paths):
    arms={}
    key=None
    for p in paths:
        rows=[json.loads(l) for l in p.read_text(encoding="utf8").splitlines() if l.strip()]
        if not rows or any(r["status"]!="ok" for r in rows):raise ValueError("incomplete or failed run")
        c=rows[0]["config"]
        if any(r["config"]!=c for r in rows):raise ValueError("mixed run")
        k=c["loops"]
        common={x:c[x] for x in ("cache","roster_sha256","max_tokens","seed","temperature","top_p")}
        if key is not None and common!=key:raise ValueError("unmatched configuration")
        key=common
        if k in arms:raise ValueError("duplicate depth")
        arms[k]={r["root"]["root"]:r for r in rows}
        if len(arms[k])!=len(rows):raise ValueError("duplicate root")
    if set(arms)!={1,2,4}:raise ValueError("need depths 1,2,4")
    ids=set(arms[1])
    if any(set(a)!=ids for a in arms.values()):raise ValueError("unmatched roots")
    if any(arms[k][i]["root"]!=arms[1][i]["root"] for k in arms for i in ids):raise ValueError("root mismatch")
    return arms

def work(row,k):
    p,c=row["prompt_tokens"],row["completion_tokens"]
    if not isinstance(p,int) or not isinstance(c,int) or p<0 or c<0:raise ValueError("missing token counts")
    return (32+40*k)*(p+c)

def family_mean(pairs):
    groups=defaultdict(list)
    for family,v in pairs:groups[family].append(v)
    if not groups:raise ValueError("empty split")
    return sum(sum(v)/len(v) for v in groups.values())/len(groups)

def features(root):
    return root["category"],len(root["prompt"].split())>20

def evaluate(arms,penalty=0.10):
    roots={i:r["root"] for i,r in arms[1].items()}
    train=[i for i,r in roots.items() if r["split"]=="train"]
    dev=[i for i,r in roots.items() if r["split"]=="dev"]
    if not train or not dev or {roots[i]["family"] for i in train}&{roots[i]["family"] for i in dev}:
        raise ValueError("invalid train/dev partition")
    def best(ids):
        # Fixed per-request layer cost penalty: depth 1 = 0, depth 4 = penalty.
        return max((1,2,4),key=lambda k:(family_mean([(roots[i]["family"],float(arms[k][i]["correct"])) for i in ids])-penalty*(k-1)/3,-k))
    default=best(train)
    table={f:best([i for i in train if features(roots[i])==f]) for f in {features(roots[i]) for i in train}}
    choices={i:table.get(features(roots[i]),default) for i in dev}
    def summarize(selected):
        rows=[(i,k,arms[k][i]) for i,k in selected.items()]
        return {"family_macro_accuracy":family_mean([(roots[i]["family"],float(r["correct"])) for i,k,r in rows]),
                "mean_work_proxy":sum(work(r,k) for i,k,r in rows)/len(rows),
                "mean_wall_seconds":sum(r["wall_seconds"] for i,k,r in rows)/len(rows),
                "mean_depth":sum(selected.values())/len(selected),
                "truncated":sum(r["finish_reason"]=="length" for i,k,r in rows),
                "per_root":{i:{"depth":k,"correct":r["correct"],"work_proxy":work(r,k)} for i,k,r in rows}}
    oracle={i:min((1,2,4),key=lambda k:(not arms[k][i]["correct"],k)) for i in dev}
    return {"status":"development_only","train_roots":len(train),"dev_roots":len(dev),
            "dev_families":len({roots[i]["family"] for i in dev}),"penalty":penalty,
            "fixed":{str(k):summarize(dict.fromkeys(dev,k)) for k in (1,2,4)},
            "train_only_lookup":summarize(choices),"hindsight_min_correct_depth":summarize(oracle),
            "lookup":{str(k):v for k,v in table.items()},"default_depth":default}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--penalty",type=float,default=.1)
    a=p.parse_args()
    if not 0<=a.penalty<=1:raise ValueError("invalid penalty")
    report=evaluate(load(a.runs),a.penalty)
    with a.out.open("x",encoding="utf8") as f:json.dump(report,f,indent=2,allow_nan=False)

if __name__=="__main__":main()
