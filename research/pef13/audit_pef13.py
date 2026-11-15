"""Independent Python-stdlib audit of PEF-13 phase forecasts and scoring receipts.

Does not import PEF-13, torch, the model architecture or predictor implementation.
Reconstructs round-to-even quantizer and conditional first-order classification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

STEPS=(0.0075,0.0125,0.035,0.08)
MODELS={(s,k) for s in (103,131,167) for k in ("coalition","threshold")}
PROTOCOL="9483da079c3707711b071efa9dbd04fa1a286b40"
PREVIOUS="fa9a3f3727b9551be721ca08e307082b81d1d2da2e359a34d88d26c9e7185b49"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correct(a,label):
    return int((a>=0)==bool(label))


def uniform(signed,deriv,step):
    if deriv==0: return 1. if signed>0 else 0. if signed<0 else .5
    return max(0.,min(1.,.5+signed/(step*abs(deriv))))


def audit(prediction_file,summary_file,evidence_folder):
    pred=json.loads(Path(prediction_file).read_text())
    report=json.loads(Path(summary_file).read_text())
    if (pred.get("campaign")!="PEF-13" or pred.get("registration_commit")!=PROTOCOL or
        pred.get("pef12_jacobian_ledger_sha256")!=PREVIOUS or
        pred.get("new_relative_steps")!=list(STEPS) or
        report.get("probe_forecast_ledger_sha256")!=sha(prediction_file) or
        not report.get("all_predictions_frozen_before_heldout")):
        raise AssertionError("frozen PEF-13 provenance differs")
    if len(pred["models"])!=6 or len(report["heldout_results"])!=6:
        raise AssertionError("six independent task/seed models required")
    if {(m["seed"],m["task"]) for m in pred["models"]}!=MODELS:
        raise AssertionError("model grid incomplete")
    orig=json.loads((Path(evidence_folder)/"sealed_training_ledger.json").read_text())
    if sha(Path(evidence_folder)/"sealed_training_ledger.json")!="1233c3de26d64318a19b18e10abe4530cdc8e5f2dadc9027d8798a993c099714":
        raise AssertionError("PEF-07 original ledger differs")
    checkpoints={(x["seed"],x["task"]):x for x in orig["training"]}
    models=[]
    count=0
    for model in pred["models"]:
        seed,task=model["seed"],model["task"]
        rec=checkpoints[(seed,task)]
        if model["checkpoint_sha256"]!=rec["checkpoint_sha256"] or sha(Path(evidence_folder)/rec["checkpoint_file"])!=rec["checkpoint_sha256"]:
            raise AssertionError("checkpoint authenticity failed")
        held=next(x for x in report["heldout_results"] if (x["seed"],x["task"])==(seed,task))
        if len(model["conditions"])!=4 or len(held["conditions"])!=4:
            raise AssertionError("unexpected model condition count")
        errors=[];uniform_errors=[];shuffle_errors=[];clean_errors=[]
        for cond in model["conditions"]:
            q,f,j,y,margin=(cond[key] for key in ("probe_q","probe_margin","probe_derivative","probe_labels","probe_signed_margin"))
            if not all(len(x)==768 for x in (q,f,j,y,margin)):
                raise AssertionError("768-row probe closure lost")
            if any(not math.isclose(ms,ff*(2*yy-1),rel_tol=1e-8,abs_tol=1e-8)
                   for ms,ff,yy in zip(margin,f,y)):
                raise AssertionError("probe label/margin alignment changed")
            hc=next(r for r in held["conditions"] if r["tangent"]==cond["tangent"] and r["angle_deg"]==cond["angle_deg"])
            if len(cond["steps"])!=4 or len(hc["steps"])!=4:
                raise AssertionError("four prospective quantizations expected")
            for step,observed in zip(cond["steps"],hc["steps"]):
                absstep=step["absolute_step"]
                if step["relative_step"] not in STEPS or observed["relative_step"]!=step["relative_step"]:
                    raise AssertionError("registered step mismatch")
                expected_delta=[absstep*round(a/absstep)-a for a in q]
                if max(abs(a-b) for a,b in zip(expected_delta,step["probe_delta"]))>1e-9:
                    raise AssertionError("lattice residual differs from independent Python round")
                forecast=sum(correct(ff+jj*dd,yy) for ff,jj,dd,yy in zip(f,j,expected_delta,y))/768
                uforecast=sum(uniform(mm,jj,absstep) for mm,jj in zip(margin,j))/768
                if abs(forecast-step["phase_aware"])>1e-10 or abs(uforecast-step["uniform_phase"])>1e-10:
                    raise AssertionError("forecast mismatch with independent math")
                score=observed["score"]
                if score["correct"]<0 or score["correct"]>768:
                    raise AssertionError("invalid test accuracy")
                if sum(score["errors_by_focal_mask"])!=768-score["correct"]:
                    raise AssertionError("focal mask error accounting mismatch")
                outcome=score["correct"]/768
                errors.append(abs(forecast-outcome))
                uniform_errors.append(abs(uforecast-outcome))
                shuffle_errors.append(abs(step["phase_shuffled"]-outcome))
                clean_errors.append(abs(step["clean_constant"]-outcome))
                count+=1
        mae=sum(errors)/len(errors)
        other=sum(uniform_errors)/len(uniform_errors)
        passed=(mae<=.035 and mae<=.75*other)
        original=next(m for m in report["decisions"]["models"] if (m["seed"],m["task"])==(seed,task))
        if (not math.isclose(mae,original["phase_aware_mae"],abs_tol=1e-12) or
            not math.isclose(other,original["uniform_phase_mae"],abs_tol=1e-12) or
            passed!=original["criterion_met"]):
            raise AssertionError("per-model decision mismatch")
        models.append({"seed":seed,"task":task,"phase_aware_mae":mae,
                       "uniform_phase_mae":other,"passed":passed})
    if count!=96:
        raise AssertionError("not exactly 96 prospective scores")
    winners=sum(x["passed"] for x in models if x["task"]=="coalition")
    verdict="criterion_met" if winners>=2 else "criterion_not_met"
    if (verdict!=report["decisions"]["primary_decision"] or
        winners!=report["decisions"]["coalition_seeds_met"]):
        raise AssertionError("primary decision differs from independent recomputation")
    return {"campaign":"PEF-13","audit":"independent_stdlib",
            "prediction_ledger_sha256":sha(prediction_file),
            "summary_sha256":sha(summary_file),"frozen_source_checkpoints":6,
            "recomputed_new_step_conditions":count,"coalition_seeds_met":winners,
            "primary_decision":verdict,"models":models}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--forecast",required=True,type=Path)
    cli.add_argument("--summary",required=True,type=Path)
    cli.add_argument("--evidence",required=True,type=Path)
    cli.add_argument("--out",type=Path)
    a=cli.parse_args()
    result=audit(a.forecast,a.summary,a.evidence)
    if a.out:a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=="__main__":main()