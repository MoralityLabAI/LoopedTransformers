"""Independent standard-library PEF-12 receipt and preregistered-decision audit.

Recomputes Gaussian and uniform-phase forecasts without torch or model code.
Original binary checkpoint SHA256s are checked against the historical ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

N=768
REPS=16
GAUSS=(.002,.006,.02)
QUANT=(.005,.02,.05)
TANGENTS=("random","pair01")
ANGLES=(.25,4.)
EXPECTED_PRECISION="f0bd80d8dc7b4af99dbc24be2723828bcf85be5b6f60f46452581e4a53661741"
EXPECTED_TRAIN="1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stdnormal(x):
    return math.erfc(-x/math.sqrt(2))/2


def forecast(margins, derivatives, scale, rel, kind):
    if kind=="gauss":
        return statistics.fmean(
            stdnormal(m/(scale*rel*abs(j))) if j else (1. if m>0 else 0. if m<0 else .5)
            for m,j in zip(margins,derivatives,strict=True))
    return statistics.fmean(
        max(0.,min(1.,.5+m/(scale*rel*abs(j)))) if j else
        (1. if m>0 else 0. if m<0 else .5)
        for m,j in zip(margins,derivatives,strict=True))


def score_sanity(score):
    if (score["total"]!=N or score["correct"] not in range(N+1) or
        len(score["errors_by_focal_mask"])!=16 or
        sum(score["errors_by_focal_mask"])!=N-score["correct"]):
        raise AssertionError("wrong-mask or correct-count receipt invalid")


def audit(folder,output):
    folder,output=Path(folder),Path(output)
    sources=folder/"registered"
    ledger=json.loads((sources/"sealed_training_ledger.json").read_text())
    if sha(sources/"sealed_training_ledger.json")!=EXPECTED_TRAIN:
        raise AssertionError("training ledger hash changed")
    chk=ledger.get("checkpoints",ledger.get("training",[]))
    for rec in chk:
        name=rec.get("file",rec.get("checkpoint_file"))
        if sha(sources/name)!=rec["checkpoint_sha256"]:
            raise AssertionError("original checkpoint binary changed")
    p11=folder/"pef11_results/frozen_precision_ledger.json"
    if sha(p11)!=EXPECTED_PRECISION:
        raise AssertionError("PEF-11 probe selection changed")
    sealed_path=output/"sealed_probe_forecasts.json"
    frozen=json.loads(sealed_path.read_text())
    report=json.loads((output/"summary.json").read_text())
    if (frozen["campaign"]!="PEF-12" or report["campaign"]!="PEF-12" or
        sha(sealed_path)!=report["prediction_ledger_sha256"] or
        not report["all_forecasts_sealed_before_heldout"] or
        not frozen["all_probes_and_noise_seeds_frozen_before_heldout"]):
        raise AssertionError("PEF-12 frozen forecast receipt invalid")
    registered={(s,t) for s in (103,131,167) for t in ("coalition","threshold")}
    if {(m["seed"],m["task"]) for m in frozen["models"]}!=registered:
        raise AssertionError("wrong six-model cohort")
    if frozen["gaussian_relative"]!=list(GAUSS) or frozen["quant_relative"]!=list(QUANT):
        raise AssertionError("noise/quant grid changed")
    per_model=[]
    all_seed_ids=set()
    scored_rows=0
    for p in frozen["models"]:
        outcome=next(r for r in report["results"] if
                     (r["seed"],r["task"])==(p["seed"],p["task"]))
        if (p["checkpoint_sha256"] != outcome["checkpoint_sha256"] or
            len(p["conditions"])!=4 or len(outcome["conditions"])!=4):
            raise AssertionError("checkpoint or condition mismatch")
        pred_error=[]
        clean_error=[]
        for c in p["conditions"]:
            if (c["tangent"] not in TANGENTS or c["angle_deg"] not in ANGLES or
                len(c["probe_signed_margins"])!=N or
                len(c["probe_derivative_transverse"])!=N or
                len(c["probe_derivative_offplane"])!=N):
                raise AssertionError("incomplete frozen probe arrays")
            r=next(v for v in outcome["conditions"] if v["tangent"]==c["tangent"] and
                   v["angle_deg"]==c["angle_deg"])
            score_sanity(r["heldout_clean"])
            expected={(site,rel,i) for site in ("transverse","offplane")
                      for rel in GAUSS for i in range(REPS)}
            seen=set()
            quant_seen=set()
            for row in r["scores"]:
                score_sanity(row["score"])
                scored_rows+=1
                if row["kind"]=="gaussian":
                    key=row["site"],row["relative_level"],row["replica"]
                    if key in seen:
                        raise AssertionError("duplicate noise row")
                    seen.add(key)
                    seed=row["seed"]
                    same_seed=(120120000+100000*p["seed"]+
                        (50000 if p["task"]=="threshold" else 0)+
                        1000*c["tangent_index"]+300*c["angle_index"]+
                        50*GAUSS.index(row["relative_level"])+row["replica"])
                    if (seed != same_seed or seed !=
                        c["noise_seeds"][GAUSS.index(row["relative_level"])][row["replica"]]):
                        raise AssertionError("unexpected random seed")
                    all_seed_ids.add(seed)
                elif row["kind"]=="retained_quant":
                    quant_seen.add(row["relative_level"])
                else:
                    raise AssertionError("unregistered outcome branch")
            if seen!=expected or quant_seen!=set(QUANT) or len(r["scores"])!=99:
                raise AssertionError("incomplete PEF-12 per-condition scoring")
            for site in ("transverse","offplane"):
                j=c["probe_derivative_"+site]
                for k,rel in enumerate(GAUSS):
                    predicted=forecast(c["probe_signed_margins"],j,p["probe_scale"],rel,"gauss")
                    if abs(predicted-c["probe_gaussian_forecast"][site][k])>1e-10:
                        raise AssertionError("stored Gaussian prediction differs from independent CDF")
                    rows=[v for v in r["scores"] if v["kind"]=="gaussian" and
                          v["site"]==site and v["relative_level"]==rel]
                    observed=statistics.fmean(v["score"]["correct"]/N for v in rows)
                    if site=="transverse" and c["tangent"]=="random":
                        pred_error.append(abs(predicted-observed))
                        clean_error.append(abs(c["probe_clean_fraction"]-observed))
            for k,rel in enumerate(QUANT):
                quant=forecast(c["probe_signed_margins"],
                               c["probe_derivative_transverse"],p["probe_scale"],rel,"quant")
                if abs(quant-c["probe_quant_uniform_phase_forecast"][k])>1e-10:
                    raise AssertionError("stored quant uniform-phase approximation altered")
        mae=statistics.fmean(pred_error)
        baseline=statistics.fmean(clean_error)
        met=mae<=.07 and mae<=.75*baseline
        per_model.append({"seed":p["seed"],"task":p["task"],
                          "mae":mae,"clean_baseline_mae":baseline,"pass":met})
    if scored_rows!=6*4*99 or len(all_seed_ids)!=6*4*3*REPS:
        raise AssertionError("total score count or seed partition mismatch")
    count=sum(m["pass"] for m in per_model if m["task"]=="coalition")
    decision="criterion_met" if count>=2 else "criterion_not_met"
    if (decision!=report["decision"]["decision"] or
        count!=report["decision"]["coalition_seed_count"]):
        raise AssertionError("reported primary decision differs from independent audit")
    for item in per_model:
        saved=next(m for m in report["decision"]["models"] if
                   (m["seed"],m["task"])==(item["seed"],item["task"]))
        if (item["pass"]!=saved["random_calibration_pass"] or
            abs(item["mae"]-saved["random_gaussian_mae"])>1e-10):
            raise AssertionError("saved calibration score disagrees with audited score")
    receipt={"campaign":"PEF-12","audit":"independent_python_stdlib",
             "prediction_ledger_sha256":sha(sealed_path),
             "source_checkpoints_checked":len(chk),"condition_count":24,
             "gaussian_replica_rows":2304,"quantization_rows":72,
             "total_score_rows":scored_rows,"unique_noise_seeds":len(all_seed_ids),
             "primary_criterion":decision,"coalition_seeds_met":count,"models":per_model}
    (output/"independent_audit.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.evidence_root,args.out)
