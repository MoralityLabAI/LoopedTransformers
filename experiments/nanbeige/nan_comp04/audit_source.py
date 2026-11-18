"""Independent NAN-COMP-04 source-only audit, with no imports of the generator/oracle.

Verifies the complete 220-row CAL/DEV materialized cohort using a separate
finite-series inverse via geometric multiplication, root-level pairing,
cross-tier source identity, split/label secrecy, and stored file fingerprints.
No model inference, no prior ARG-03 test access, no confirmation generation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OPS=("I","N","S","R")


def _jsonl(path:Path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _hash(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def _fingerprint(obj)->str:
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def _multiply(f,g,n):
    return [sum(f[i]*g[k-i] for i in range(k+1)) for k in range(n)]


def _inverse_independent(f):
    """Independent geometric expansion (not the source.py recurrence algorithm)."""
    n=len(f)
    if n<1 or f[0]!=1:
        raise ValueError("only unit-constant series")
    negative_tail=[0]+[-k for k in f[1:]]
    power=[1]+[0]*(n-1)
    total=[0]*n
    for _ in range(n):
        total=[a+b for a,b in zip(total,power)]
        power=_multiply(power,negative_tail,n)
    if _multiply(f,total,n)!=[1]+[0]*(n-1):
        raise AssertionError("geometric inverse residual nonzero")
    return total


def _transform(f,op):
    if op=="I":
        return f[:]
    if op=="N":
        return [v*(-1 if k%2 else 1) for k,v in enumerate(f)]
    if op=="S":
        return [f[k//2] if k%2==0 else 0 for k in range(len(f))]
    if op=="R":
        return _inverse_independent(f)
    raise ValueError("unknown operator")


def _source(f,m):
    if len(f)!=2*m+3 or f[0]!=1:
        raise AssertionError("source length/constant invalid")
    if m not in (3,5) or any(f[k] for k in range(1,len(f)) if k not in (m,m+2)):
        raise AssertionError("source sparsity law invalid")
    if f[m]==0 or f[m+2]==0 or max(abs(f[m]),abs(f[m+2]))>5:
        raise AssertionError("source coefficient support invalid")


def _guard(path):
    if any(v.lower() in {"test","heldout","test_full168"} for v in path.parts):
        raise ValueError("forbidden held-out/test path")


def audit(source_dir:Path)->dict:
    _guard(source_dir)
    cfg=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    receipt=json.loads((source_dir/"source_receipt.json").read_text(encoding="utf-8"))
    if (receipt.get("schema")!="nan_comp04.sourceonly.v1"
        or receipt.get("model_inference") is not False
        or receipt.get("original_ARG03_168_test_opened") is not False
        or receipt.get("reserved_confirmation_sources_generated") is not False
        or receipt.get("BPD_transfer_attempted") is not False):
        raise AssertionError("source-only evidence boundary missing")
    for name in ("public.jsonl","host_only.jsonl","roots.json"):
        if receipt["files_sha256"][name]!=_hash(source_dir/name):
            raise AssertionError("materialized source SHA mismatch: "+name)
    if receipt["config_sha256"]!=_hash(HERE/"config.json"):
        raise AssertionError("generator configuration changed after source creation")
    rows=_jsonl(source_dir/"public.jsonl")
    labels=_jsonl(source_dir/"host_only.jsonl")
    roots=json.loads((source_dir/"roots.json").read_text(encoding="utf-8"))
    if len(rows)!=220 or len(labels)!=220 or len(roots)!=20:
        raise AssertionError("expected 220 rows, 220 labels, 20 root descriptions")
    labels_by={r["case_id"]:r for r in labels}
    if len(labels_by)!=220 or len({r["case_id"] for r in rows})!=220:
        raise AssertionError("duplicated case identifiers")
    if {r["case_id"] for r in rows}!=set(labels_by):
        raise AssertionError("public/hidden case ID mismatch")
    registered=set(cfg["calibration_roots"]+cfg["development_roots"])
    if {r["root"] for r in roots}!=registered or len(set(r["root"] for r in roots))!=20:
        raise AssertionError("source roots differ from fixed CAL/DEV")
    reserved=set(cfg["reserved_confirmation_roots"])
    if registered & reserved:
        raise AssertionError("root populations overlap")
    rmap={r["root_key"]:r for r in roots}
    if len(rmap)!=20 or len({r["source_identity"] for r in roots})!=20:
        raise AssertionError("source root keys/collisions")
    grouped=defaultdict(list)
    for row in rows:
        id=row["case_id"]
        h=labels_by[id]
        if (set(row)!={"case_id","root_group","split","kind","m","tier",
                       "degree","demos","query","public_sha256"}
            or row["public_sha256"]!=_fingerprint({k:v for k,v in row.items()
                                                   if k not in ("case_id","public_sha256")})
            or h["public_sha256"]!=row["public_sha256"]
            or h["root_key"]!=row["root_group"]
            or h["kind"]!=row["kind"] or h["tier"]!=row["tier"]
            or h["split"]!=row["split"]):
            raise AssertionError("source/label mismatch or public metadata leak")
        root=rmap[row["root_group"]]
        if root["root"]!=h["root"] or root["split"]!=row["split"] or row["m"]!=root["m"]:
            raise AssertionError("case belongs to wrong source root")
        if row["split"]!="development" and row["split"]!="calibration":
            raise AssertionError("unknown split")
        if h["root"] in reserved:
            raise AssertionError("reserved confirmation source found")
        m=row["m"]
        if row["kind"]=="classification":
            if h["truth"] not in OPS or row["query"] or len(row["demos"])!=2:
                raise AssertionError("classification task contract")
            if row["tier"] not in ("early","first","cross","erased"):
                raise AssertionError("classification tier")
            d={"early":2*m-1,"first":2*m,"cross":2*m+2,
               "erased":2*m}[row["tier"]]
            if row["degree"]!=d:
                raise AssertionError("evidence degree drift")
            for k,ex in enumerate(row["demos"]):
                expected_full=[root["a"],root["a_other"]][k]
                expected_c=[root["c"],root["c_other"]][k]
                f=[1]+[0]*(2*m+2)
                f[m]=expected_full;f[m+2]=expected_c
                if ex["input"]!=f[:d+1]:
                    raise AssertionError("source demonstration drift")
                if row["tier"]=="erased":
                    if ex["output"] is not None:
                        raise AssertionError("erasure leaked a label")
                elif ex["output"]!=_transform(ex["input"],h["truth"]):
                    raise AssertionError("published output fails independent oracle")
            if h["oracle_answer"]!=h["truth"]:
                raise AssertionError("classification target drift")
        elif row["kind"]=="coefficient":
            if (row["tier"]!="query_cross" or row["degree"]!=2*m+2
                or row["demos"] or h["truth"]!="R"):
                raise AssertionError("numeric task contract")
            query=[1]+[0]*(2*m+2)
            query[m]=root["query_a"];query[m+2]=root["query_c"]
            if row["query"]!=query:
                raise AssertionError("query source drift")
            _source(query,m)
            value=_inverse_independent(query)[2*m+2]
            if h["oracle_answer"]!=value or value!=2*root["query_a"]*root["query_c"]:
                raise AssertionError("numeric label invalid")
        else:
            raise AssertionError("non-task kind")
        grouped[row["root_group"]].append((row,h))
    by_tier=Counter()
    for key,entries in grouped.items():
        if len(entries)!=11:
            raise AssertionError("incomplete root task roster")
        root=rmap[key];m=root["m"]
        tiers=defaultdict(list)
        for row,h in entries:
            tiers[(row["kind"],row["tier"])].append((row,h))
            by_tier[(row["kind"],row["tier"])]+=1
        if len(tiers[("classification","first")])!=4:
            raise AssertionError("missing four-way first evidence")
        if len(tiers[("coefficient","query_cross")])!=1:
            raise AssertionError("missing numeric task")
        for t in ("early","cross","erased"):
            pair=tiers[("classification",t)]
            if len(pair)!=2 or {h["truth"] for _,h in pair}!={"N","R"}:
                raise AssertionError("N/R root pairing changed")
            (a,al),(b,bl)=pair
            if [x["input"] for x in a["demos"]] != [x["input"] for x in b["demos"]]:
                raise AssertionError("counterfactual source input drift")
            if t in ("early","erased"):
                if a["demos"]!=b["demos"] or a["public_sha256"]!=b["public_sha256"]:
                    raise AssertionError("identical-input control is distinguishable")
            else:
                if a["demos"]==b["demos"]:
                    raise AssertionError("late evidence not present")
        first=tiers[("classification","first")]
        if {h["truth"] for _,h in first}!=set(OPS):
            raise AssertionError("four-way source labels missing")
        for index in (0,1):
            f=[1]+[0]*(2*m+2)
            f[m]=[root["a"],root["a_other"]][index]
            f[m+2]=[root["c"],root["c_other"]][index]
            _source(f,m)
            vals={o:_transform(f,o) for o in OPS}
            if (vals["N"][:2*m]!=vals["R"][:2*m]
                or vals["R"][2*m]-vals["N"][2*m]!=f[m]**2
                or vals["R"][2*m+2]-vals["N"][2*m+2]!=2*f[m]*f[m+2]):
                raise AssertionError("late-series identity not certified")
    return {
        "schema":"nan_comp04.independent_source_audit.v1",
        "source_only":True,"no_model_calls":True,
        "sealed_ARG03_test_opened":False,"reserved_sources_generated":False,
        "roots":len(roots),"public_rows":len(rows),"host_only_rows":len(labels),
        "development_roots":len(cfg["development_roots"]),
        "calibration_roots":len(cfg["calibration_roots"]),
        "tasks_by_tier":{":".join(t):count for t,count in sorted(by_tier.items())},
        "source_file_sha256":{name:_hash(source_dir/name) for name in
                               ("public.jsonl","host_only.jsonl","roots.json","source_receipt.json")},
        "audit_source_sha256":_hash(Path(__file__)),
        "result":"PASS",
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    _guard(args.out)
    report=audit(args.source_dir)
    if args.out.exists():
        raise FileExistsError("refuse to replace independent source audit")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in
                      ("result","roots","public_rows","development_roots")},sort_keys=True))


if __name__=="__main__":
    main()
