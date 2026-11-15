"""New final-R DEVELOPMENT-ONLY calibration from source-separated model tapes.

Requires a fresh public fit/valid roster from prepare_final_r.py.
Model-forward process emits case_id, public_sha256, 64-program baseline scores
and [16,3,4] first-pass site logits. This isolated scorer then combines
those logits with host-only source program IDs.

Ranks are co-primary diagnostics for the *true source program*. They are not
equivalent to future-correct@12, which needs independent oracle.search/grade
on the same ranked list and must be reported separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .calibration import (DevRankRow, fit_position_operator_bias, program_scores,
                          rank_of_gold, rank_summary)

FIT="development_finalR_fit"
VALID="development_finalR_valid"

def _load(path: Path)->list[dict]:
    if any(x.lower() in {"test","test.jsonl","test_full168"} for x in path.parts):
        raise ValueError("Sealed test paths are excluded")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

def _assemble(split:str, labels:list[dict], public:list[dict],
              tape:list[dict])->tuple[list[DevRankRow],list[dict]]:
    if split not in (FIT,VALID):
        raise ValueError("Only fresh final-R development fit and validation splits")
    host={r["case_id"]:r for r in labels if r["split"]==split}
    pub={r["case_id"]:r for r in public}
    mt={r["case_id"]:r for r in tape}
    if (not host or len(host)!=len(public) or len(host)!=len(tape)
            or set(host)!=set(pub) or set(host)!=set(mt)):
        raise ValueError("Public, host labels and model tape are incomplete or mismatched")
    out=[]
    for cid,h in sorted(host.items()):
        if (h["source_sha256"]!=pub[cid]["public_sha256"] or
                h["source_sha256"]!=mt[cid]["public_sha256"]):
            raise ValueError("Source hash mismatch for "+cid)
        if ("loop1_operator_site_logits" not in mt[cid]
            or "baseline" not in mt[cid]):
            raise ValueError("Need full model-only score tapes for "+cid)
        raw=tuple(tuple(tuple(v) for v in ctx)
                  for ctx in mt[cid]["loop1_operator_site_logits"])
        item=DevRankRow(cid,split,raw,int(h["gold_program_index"]))
        if any(len(mt[cid]["baseline"].get(str(t),[]))!=64
               for t in (1,2,4,8)):
            raise ValueError("Need original full 64-program scores at all depths")
        out.append(item)
    return out,[mt[r.case_id] for r in out]

def _baseline_summary(records:list[DevRankRow],tape:list[dict])->dict:
    return {str(t):{
      "n":len(records),
      "gold_rank":ranks,
      "gold_rank_median": (lambda x:(x[(len(x)-1)//2]+x[len(x)//2])/2)(sorted(ranks)),
      "top1_source_program":sum(v==1 for v in ranks),
      "source_program_in_top12":sum(v<=12 for v in ranks)
    } for t in (1,2,4,8)
    for ranks in [[rank_of_gold(tape[i]["baseline"][str(t)],c.gold_program)
                   for i,c in enumerate(records)]]}

def run(fit_pub:list[dict],valid_pub:list[dict],host:list[dict],
        fit_tape:list[dict],valid_tape:list[dict]) -> dict:
    fit,ft=_assemble(FIT,host,fit_pub,fit_tape)
    valid,vt=_assemble(VALID,host,valid_pub,valid_tape)
    if set(r.case_id for r in fit) & set(r.case_id for r in valid):
        raise ValueError("Fit/valid overlap")
    fprograms={r.gold_program for r in fit}
    vprograms={r.gold_program for r in valid}
    if fprograms & vprograms:
        raise ValueError("Fit/valid must hold out final-R PROGRAMS, not just input cases")
    frozen=fit_position_operator_bias(fit)
    b=frozen["parameters_3x4"]
    return {
        "schema":"arg03.finalR.devonly.ranked_calibration.v1",
        "sealed_test_loaded":False,
        "fit":{ "n":len(fit),"baseline":_baseline_summary(fit,ft),
                "bias12":rank_summary(fit,b)},
        "valid":{"n":len(valid),"baseline":_baseline_summary(valid,vt),
                 "bias12":rank_summary(valid,b)},
        "frozen_bias":frozen,
        "outcome_limit":"Source-program rank is not future-correct@12; future@12 requires isolated oracle-grade and no test unseal."
    }

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("fit-public","valid-public","host-only","fit-tape","valid-tape","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    args={k.replace("-","_"):v for k,v in vars(a).items()}
    result=run(_load(args["fit_public"]),_load(args["valid_public"]),
               _load(args["host_only"]),_load(args["fit_tape"]),_load(args["valid_tape"]))
    if "test.jsonl" in str(args["output"]).lower():
        raise ValueError("Test output path not permitted")
    args["output"].parent.mkdir(parents=True,exist_ok=True)
    args["output"].write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"fit":result["fit"]["n"],"valid":result["valid"]["n"],
                       "test_loaded":False},sort_keys=True))
if __name__=="__main__":main()
