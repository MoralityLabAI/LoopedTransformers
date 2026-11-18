"""Host-only NAN-COMP-04 exact source and numeric competence grader.

Run in a separate CPU process after model-only inference; no Torch and no
pretrained weights. Never uses or regenerates sealed ARG-03 heldout examples.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .cohort import HERE,PUBLIC_KEYS,fingerprint
from .run import forbid_test_paths,sha
from .source import transform


def load(path:Path)->list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def score_case(row:dict,label:dict,model:dict)->dict:
    if row["case_id"]!=label["case_id"] or row["case_id"]!=model["case_id"]:
        raise ValueError("source/host/model case-id mismatch")
    if (row["public_sha256"]!=label["public_sha256"] or
        row["public_sha256"]!=model["public_sha256"]):
        raise ValueError("source/host/model hash mismatch")
    if row["kind"]!=label["kind"] or row["tier"]!=label["tier"]:
        raise ValueError("source task mismatch")
    rec=model["result"];parsed=rec["parser"]
    if rec["kind"]!=row["kind"]:
        raise ValueError("model task mismatch")
    response=parsed.get("operator") if row["kind"]=="classification" else parsed.get("value")
    expected=label["oracle_answer"]
    correct=parsed.get("status")=="accepted" and response==expected
    if row["kind"]=="classification" and row["tier"]!="erased":
        if any(transform(d["input"],expected)[:row["degree"]+1]!=d["output"]
               for d in row["demos"]):
            raise AssertionError("host source-label mathematical witness invalid")
    if row["kind"]=="coefficient":
        m=row["m"];a=row["query"][m];c=row["query"][m+2]
        if expected!=2*a*c:
            raise AssertionError("host numeric future not equal to independent oracle")
    return {
        "case_id":row["case_id"],"root_group":row["root_group"],"split":row["split"],
        "tier":row["tier"],"kind":row["kind"],
        "expected":expected,"predicted":response,"correct":bool(correct),
        "format_valid":parsed.get("status")=="accepted",
        "tokens_generated":rec["tokens_generated"],
    }


def summarize(scored:list[dict],*,expected_roots:int)->dict:
    by=defaultdict(list)
    for rec in scored:
        by[rec["root_group"]].append(rec)
    if len(by)!=expected_roots or any(len(rows)!=11 for rows in by.values()):
        raise ValueError("root cohort incomplete")
    aggregate=Counter()
    root_results=[]
    for root,rows in sorted(by.items()):
        target={(r["kind"],r["tier"],str(r["expected"])):r for r in rows}
        n=target["classification","first","N"]
        r=target["classification","first","R"]
        q=next(x for x in rows if x["kind"]=="coefficient")
        e_n=target["classification","early","N"]
        e_r=target["classification","early","R"]
        erased_n=target["classification","erased","N"]
        erased_r=target["classification","erased","R"]
        if e_n["correct"] and e_r["correct"]:
            raise AssertionError("early N/R identical output pair cannot both classify correctly")
        if erased_n["correct"] and erased_r["correct"]:
            raise AssertionError("erased N/R pair cannot both classify correctly")
        both=n["correct"] and r["correct"]
        triple=both and q["correct"]
        aggregate["nr_both"]+=int(both)
        aggregate["cross_query"]+=int(q["correct"])
        aggregate["R_correct"]+=int(r["correct"])
        aggregate["joint"]+=int(triple)
        aggregate["mid_nr_worlds"]+=int(n["correct"])+int(r["correct"])
        aggregate["erased_nr_worlds"]+=int(erased_n["correct"])+int(erased_r["correct"])
        aggregate["early_nr_worlds"]+=int(e_n["correct"])+int(e_r["correct"])
        for item in rows:
            aggregate["correct_"+item["kind"]+"_"+item["tier"]]+=int(item["correct"])
            aggregate["invalid_"+item["kind"]+"_"+item["tier"]]+=int(not item["format_valid"])
        root_results.append({"root_group":root,"nr_both":bool(both),
                             "cross_query_exact":bool(q["correct"]),
                             "joint":bool(triple),"R_correct":bool(r["correct"])})
    return {"n_roots":len(by),"counts":dict(aggregate),"root_results":root_results}


def grade(source_dir:Path,runs:Path,backend:str,split:str)->dict:
    from .run import read_public
    forbid_test_paths(source_dir,runs)
    if backend not in ("bf16","nf4") or split not in ("calibration","development"):
        raise ValueError("only CAL/DEV native BF16/NF4 source conditions")
    public,receipt=read_public(source_dir)
    host_rows=load(source_dir/"host_only.jsonl")
    expected_host_hash=receipt["files_sha256"]["host_only.jsonl"]
    if sha(source_dir/"host_only.jsonl")!=expected_host_hash:
        raise ValueError("host label content hash mismatch")
    host={r["case_id"]:r for r in host_rows}
    if len(host)!=220 or set(host)!={r["case_id"] for r in public}:
        raise ValueError("missing or duplicate independent host labels")
    output=runs/backend/split
    manifest=json.loads((output/"manifest.json").read_text())
    if (manifest.get("status")!="completed" or
        manifest.get("backend")!=backend or manifest.get("split")!=split or
        manifest.get("original_ARG03_168_test_accessed") is not False or
        manifest.get("source_public_sha256")!=receipt["files_sha256"]["public.jsonl"]):
        raise ValueError("unverified model-only run manifest")
    all_public={r["case_id"]:r for r in public if r["split"]==split}
    wanted=set(manifest["roots_selected"])
    expected={cid:row for cid,row in all_public.items() if row["root_group"] in wanted}
    if len(expected)!=11*len(wanted):
        raise ValueError("root selection not complete")
    tape={}
    for root in wanted:
        payload=json.loads((output/("root_"+root+".json")).read_text(encoding="utf-8"))
        if (payload["root_group"]!=root or payload["backend"]!=backend or
            payload["split"]!=split or payload["source_public_sha256"]!=
            receipt["files_sha256"]["public.jsonl"] or
            payload["original_ARG03_168_test_accessed"] is not False):
            raise ValueError("model root provenance mismatch")
        for rec in payload["cases"]:
            cid=rec["case_id"]
            if cid in tape:
                raise ValueError("duplicate model output")
            tape[cid]=rec
    if set(expected)!=set(tape):
        raise ValueError("missing/extra model outputs")
    scored=[score_case(r,host[cid],tape[cid]) for cid,r in sorted(expected.items())]
    # Idempotence: common N/R early and erased inputs must have identical
    # actual generated tokens and prompt identifiers, irrespective of target.
    by=defaultdict(list)
    for cid,row in expected.items():
        if row["tier"] in ("early","erased"):
            by[(row["root_group"],row["tier"])].append(tape[cid]["result"])
    for key,items in by.items():
        if len(items)!=2 or any(items[0][field]!=items[1][field]
                                for field in ("initial_prompt_sha256","generated_ids")):
            raise AssertionError("observationally identical model input diverged: "+str(key))
    summary=summarize(scored,expected_roots=len(wanted))
    aggregate=summary["counts"]
    n=summary["n_roots"]
    is_full=(n==(4 if split=="calibration" else 16))
    gates={
        "nr_both":aggregate.get("nr_both",0)>=12,
        "cross_coefficient":aggregate.get("cross_query",0)>=12,
        "reciprocal":aggregate.get("R_correct",0)>=12,
        "source_use":(aggregate.get("mid_nr_worlds",0)-
                      aggregate.get("erased_nr_worlds",0))>=int(2*n*0.25),
    }
    return {
        "schema":"nan_comp04.devonly.exact_grade.v1",
        "split":split,"backend":backend,"source_receipt_sha256":sha(source_dir/"source_receipt.json"),
        "model_manifest_sha256":sha(output/"manifest.json"),
        "source_root_count":n,"model_case_count":len(scored),
        "summary":summary,"gates":gates,
        "decision":"PASS_DEV_PREREQUISITES" if is_full and all(gates.values()) else (
            "FAIL_DEV_PREREQUISITES" if is_full else "ENGINEERING_PILOT_ONLY"),
        "source_use_difference":(aggregate.get("mid_nr_worlds",0)-
                                 aggregate.get("erased_nr_worlds",0))/(2*n),
        "per_case":scored,
        "BPD_transfer_attempted":False,"original_ARG03_168_test_opened":False,
        "independent_confirmation_claim":False,
        "comment":"Source-root clustered development measurement; FSI-01 needed for inference.",
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--backend",choices=("bf16","nf4"),required=True)
    p.add_argument("--split",choices=("calibration","development"),default="development")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    forbid_test_paths(a.out)
    result=grade(a.source_dir,a.runs,a.backend,a.split)
    if a.out.exists():
        raise FileExistsError("refuse overwrite of development grades")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"n_roots":result["source_root_count"],"decision":result["decision"],
                      "counts":result["summary"]["counts"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
