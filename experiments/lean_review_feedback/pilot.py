"""LRF-01: development replay of frozen LR-01 traces and matched reward controls."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
LR=HERE.parent/"lean_reasoner"
sys.path.insert(0,str(LR))
import proof_data as P
import reward as R

MANIFEST_BLOB="830bc63cb820bd75262f306afe15644fa6e6f84b"
PREDICTION_BLOB="9ba8352c169667d39de5a96f73af1d306862fd20"

def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()

def controls(task):
    t=R.canonical(task);trace=list(task["trace"])
    first=t["facts"][0]
    assert t["goal"] not in t["facts"] and trace
    extra={**t,"facts":sorted(t["facts"]+[t["goal"]])}
    shortcut={**t,"rules":t["rules"]+[[[first],t["goal"]]]}
    return [
        ("correct",{"trace":trace},(True,True,True,1)),
        ("easier_target",{"claim":{**t,"goal":first},"trace":[]},(True,True,False,0)),
        ("extra_hypothesis",{"claim":extra,"trace":[]},(True,True,False,0)),
        ("extra_rule",{"claim":shortcut,"trace":[len(t["rules"])]},(True,True,False,0)),
        ("excluded_axiom",{"mode":"fixture_axiom"},(True,False,True,0)),
        ("hidden_alias",{"mode":"fixture_alias"},(True,False,True,0)),
        ("unused_axiom",{"trace":trace,"unused_axiom":True},(True,True,True,1)),
        ("incomplete",{"trace":[]},(None,None,None,None)),
    ]

def vector(receipt):
    return tuple(receipt[k] for k in ("kernel","foundation","contract","reward"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lean",default="lean")
    a=p.parse_args();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    version=subprocess.check_output([a.lean,"--version"],text=True).strip()
    assert "4.19.0" in version,version
    raw=(LR/"results"/"corpus_manifest.json").read_bytes()
    predictions=(LR/"results"/"prediction_receipts.json.gz").read_bytes()
    assert blob(raw)==MANIFEST_BLOB and blob(predictions)==PREDICTION_BLOB
    manifest=json.loads(raw)
    tasks,rebuilt=P.build(out/"corpus")
    for key in ("tasks_sha256","lean_source_sha256","counts","unique_structure_groups"):
        assert rebuilt[key]==manifest[key],key
    byid={t["id"]:t for t in tasks}
    rows=json.loads(gzip.decompress(predictions))["rows"]
    assert len(rows)==384
    frozen={}
    for ident,solved,trace in rows:
        assert ident not in frozen
        t=byid[ident];assert t["split"] in ("test","depth_ood")
        assert len(trace)<=t["min_steps"]
        known=set(t["facts"])
        for j in trace:
            assert j in P.legal(t,known),(ident,j)
            known.add(t["rules"][j][1])
        assert bool(solved)==(t["goal"] in known)
        frozen[ident]=(bool(solved),trace)
    assert set(frozen)=={t["id"] for t in tasks if t["split"] in ("test","depth_ood")}
    selected=[i for split in ("test","depth_ood")
              for i in sorted(i for i in frozen if byid[i]["split"]==split)[:32]]
    assert len(selected)==64
    validation=sorted((t for t in tasks if t["split"]=="val"),key=lambda t:t["id"])[:16]
    receipts=[];errors=[];counts=Counter();start=time.monotonic()

    def run(pop,case,name,candidate,expected):
        receipt=R.evaluate_candidate(case,candidate,out/"cases"/pop/case["id"]/name,a.lean)
        item={"population":pop,"id":case["id"],"arm":name,
              "expected":expected,"actual":vector(receipt),"receipt":receipt}
        receipts.append(item)
        counts[(pop,name,receipt["classification"])]+=1
        if vector(receipt)!=expected:
            errors.append({"id":case["id"],"population":pop,"arm":name,
                           "expected":expected,"actual":vector(receipt),"reason":receipt["classification"]})
        with (out/"receipts.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(item,sort_keys=True)+"\n")
        print("CASE",pop,case["id"],name,receipt["classification"],receipt["reward"],flush=True)

    for t in validation:
        for name,candidate,expected in controls(t):
            run("validation_controls",t,name,candidate,expected)
    for ident in selected:
        solved,trace=frozen[ident]
        expected=(True,True,True,1) if solved else (None,None,None,None)
        run("historical_model",byid[ident],"frozen_trace",{"trace":trace},expected)
    assert len(receipts)==192
    control=[x for x in receipts if x["population"]=="validation_controls"]
    model=[x for x in receipts if x["population"]=="historical_model"]
    assert len(control)==128 and len(model)==64
    report={"schema":"LRF-01/pilot-v1","lean":version,
            "original_corpus_sha256":rebuilt["tasks_sha256"],"prediction_blob":PREDICTION_BLOB,
            "all_historical_traces_rechecked":len(frozen),
            "historical_native_sample":len(model),"historical_sample_ids":selected,
            "historical_sample_original_solved":sum(frozen[i][0] for i in selected),
            "historical_sample_native_accepted":sum(x["receipt"]["reward"]==1 for x in model),
            "validation_controls":len(control),
            "control_native_complete":sum(x["receipt"]["kernel"] is True for x in control),
            "control_rewards_one":sum(x["receipt"]["reward"]==1 for x in control),
            "control_rewards_zero":sum(x["receipt"]["reward"]==0 for x in control),
            "control_censored":sum(x["receipt"]["reward"] is None for x in control),
            "false_positive_rewards":sum(x["receipt"]["reward"]==1 and x["expected"][-1]!=1 for x in receipts),
            "native_compiler_calls":sum(x["receipt"]["compiler_calls"] for x in receipts),
            "seconds":time.monotonic()-start,"errors":errors,
            "counts":[{"population":k[0],"arm":k[1],"classification":k[2],"n":v} for k,v in sorted(counts.items())],
            "passed":not errors}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("LRF_GATE",json.dumps({k:v for k,v in report.items() if k not in ("counts","historical_sample_ids")}),flush=True)
    if errors:raise SystemExit(1)

if __name__=="__main__":main()
