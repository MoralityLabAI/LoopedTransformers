"""ARG-03/LOOPSCOPE-04 development-only post-hoc audit.

No model inference, no oracle import, and no held-out test reads. This module
consumes compact, source-hashed development and 120-board *existing*
calibration receipts. The frozen test protocol remains unchanged.

A response/displacement ratio is descriptive: the later swap/random state
displacements are unequal, so the existing H2a raw effect is confounded.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from pathlib import Path
from statistics import median
import argparse
import json

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data/dev96_compact.json"
GRAM = ROOT / "data/gramian_compact.json"

def _require_dev_only(doc: dict, path: Path) -> None:
    parts = tuple(x.lower() for x in path.parts)
    if any(x in {"test", "test.jsonl", "heldout", "test_full168"} for x in parts):
        raise ValueError("Held-out test paths are forbidden in a development audit")
    if doc.get("split") != "development" or doc.get("n_rows") != 96:
        raise ValueError("Only the original 96-case development receipt is supported")
    rows = doc.get("rows")
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError("Expected 96 development rows")
    if any(not r.get("case_id", "").startswith("arg03:development:")
           for r in rows):
        raise ValueError("Only development case IDs allowed")
    if len({r["case_id"] for r in rows}) != 96:
        raise ValueError("Duplicate case ID")
    if doc.get("ranking_list_available") is not False or doc.get("target_final_operator_available") is not False:
        raise ValueError("These per-case receipts lack full rankings and target-final-operator labels")

def paired(a: list[bool], b: list[bool]) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError("Paired nonempty binary outcomes required")
    if any(type(x) is not bool for x in a+b):
        raise ValueError("Outcomes must be Boolean")
    n11=sum(x and y for x,y in zip(a,b))
    n10=sum(x and not y for x,y in zip(a,b))
    n01=sum(not x and y for x,y in zip(a,b))
    n00=len(a)-n11-n10-n01
    denom=sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return {"both":n11,"a_only":n10,"b_only":n01,"neither":n00,
            "flips":n10+n01,"phi":(n11*n00-n10*n01)/denom if denom else None}

def depth_analysis(dev_rows: list[dict]) -> dict:
    ds=("1","2","4","8")
    def good(row,d): return row["baseline"][d]["future_correct_at_12"]
    pairlist=((1,2),(2,4),(2,8),(4,8))
    return {
        "by_depth": {d:{
            "future_correct":sum(good(r,d) for r in dev_rows),
            "top1_future_correct":sum(r["baseline"][d]["top1_future_correct"] for r in dev_rows),
            # Program IDs at 2/4/8 come from ranking_changes_vs_clean, which
            # includes only the baseline top-1, not the whole top-12.
            "distinct_baseline_top1_programs": (
                len({r["top1_program_id"][d] for r in dev_rows}) if d!="1" else None),
            "by_family":{f:sum(good(r,d) for r in dev_rows if r["family"]==f)
                         for f in sorted({r["family"] for r in dev_rows})}
        } for d in ds},
        "pairs":{f"{a}_vs_{b}":paired(
            [good(r,str(a)) for r in dev_rows],
            [good(r,str(b)) for r in dev_rows]) for a,b in pairlist},
        "correct_all_2_4_8":sum(all(good(r,d) for d in ("2","4","8")) for r in dev_rows),
        "correct_any_2_4_8":sum(any(good(r,d) for d in ("2","4","8")) for r in dev_rows),
        "full_top12_distinct_programs":None,
        "target_final_operator_by_case":None,
        "missing_information":"The committed 96-case diagnostics expose top1_a, not 64-program ranked lists or target operator labels. Recover these from a fresh dev-only answer tape and isolated grader."
    }

def displacement_analysis(rows: list[dict]) -> dict:
    out={}
    for d in ("2","4","8"):
        ratios,ns,nr,win=[],[],[],0
        for row in rows:
            hs=row["hidden_state"][d]
            rd=row["postloop_readout"][d]
            sw,rand=hs["swap_l2"],hs["random_l2"]
            if min(sw,rand)<=1e-12:
                raise ValueError("An effectively zero displacement makes per-unit comparison undefined")
            ds=rd["swap_l2"]/sw
            dr=rd["random_l2"]/rand
            ratios.append(sw/rand);ns.append(ds);nr.append(dr)
            win+=int(ds>dr)
        out[d]={
            "mean_state_displacement_ratio_swap_over_random":sum(ratios)/len(ratios),
            "mean_readout_l2_per_state_l2_swap":sum(ns)/len(ns),
            "mean_readout_l2_per_state_l2_random":sum(nr)/len(nr),
            "swap_larger_per_unit_cases":win,
            "n":len(rows),
            "interpretation":"Descriptive normalization only; later-state amplitudes are unequal, so prior H2a sign p cannot be read as specificity"
        }
    return out

def gramian_analysis(document: dict) -> dict:
    if document.get("split") != "rep_public_eval_calibration":
        raise ValueError("Only previously evaluated LOOPSCOPE-04 calibration boards permitted")
    if document.get("sampled_tangent_dimension") != 8 or document.get("ambient_kernel_dimension") != 501:
        raise ValueError("Unexpected sampled/ambient tangent dimensions")
    grouped=defaultdict(list)
    for r in document["rows"]:
        if r["probes"]!=8 or r["kernel_dim"]!=501:
            raise ValueError("Unexpected per-board Gramian source")
        if len(r["sampled_eigenvalues"])!=8 or min(r["sampled_eigenvalues"])<0:
            raise ValueError("Bad sampled tangent spectrum")
        grouped[(r["checkpoint"],r["stratum"])].append(float(r["lambda1"]))
    if len(document["rows"])!=240 or any(len(v)!=40 for v in grouped.values()):
        raise ValueError("Expected both checkpoints by three 40-board strata")
    return {
      ck:{st:{
        "n":len(v),"mean_lambda1":sum(v)/len(v),
        "median_lambda1":median(v),
        "top3_fraction_of_lambda1_sum":sum(sorted(v,reverse=True)[:3])/sum(v),
      } for (tag,st),v in sorted(grouped.items()) if tag==ck}
      for ck in ("gaoxin492","yagizdevre")
    }

def audit(dev: dict, gr: dict, *, source_path: Path=DEV) -> dict:
    _require_dev_only(dev, source_path)
    if dev.get("source_repository")!="MoralityLabAI/VPD":
        raise ValueError("Pinned public source repository changed")
    return {
        "schema":"arg03.fable.preunseal.dev_audit.v1",
        "source_split":"development",
        "heldout_test_loaded":False,
        "source_dev_blob_sha":dev["source_blob_sha"][0],
        "source_gramian_blob_shas":gr["source_blob_shas"],
        "source_answer_tape_sha256":dev["source_answer_tape_sha256"],
        "arg03": {
            "depth":depth_analysis(dev["rows"]),
            "displacement":displacement_analysis(dev["rows"]),
        },
        "loopscope04":{"gramian":gramian_analysis(gr)},
        "claim_boundary":"All ARG03 metrics are from 96 existing dev cases, all 3C metrics from 240 existing calibration board records. No new Nanbeige/TRM inference or sealed test read."
    }

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development",type=Path,default=DEV)
    parser.add_argument("--gramian",type=Path,default=GRAM)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    _require_dev_only({"split":"development","n_rows":96,"rows":[{"case_id":f"arg03:development:guard:{i}"} for i in range(96)],"ranking_list_available":False,"target_final_operator_available":False},args.development)
    doc=json.loads(args.development.read_text(encoding="utf-8"))
    grams=json.loads(args.gramian.read_text(encoding="utf-8"))
    result=audit(doc,grams,source_path=args.development)
    if "test.jsonl" in str(args.output).lower():
        raise ValueError("Audit output cannot be named test.jsonl")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"heldout_test_loaded":False,"depth_pairs":result["arg03"]["depth"]["pairs"]},sort_keys=True))

if __name__=="__main__":
    main()
