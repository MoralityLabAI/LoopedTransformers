"""POSTHOC PEF-13 decomposition: probe nonlinear replay vs probe local forecast vs heldout.

Run only after the full PEF-13 primary decision is sealed. These diagnostics
cannot change preregistered scoring or its negative outcome.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import torch
from research.pef08 import pef08 as p8
from research.pef09 import pef09 as p9
from research.pef11 import pef11 as p11
from research.pef12 import pef12 as p12
from research.pef13 import pef13 as p13


def diagnose(evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger,forecast_path,summary_path,output):
    original_forecast=p13.sha(forecast_path)
    base=json.loads(Path(summary_path).read_text())
    if base["decisions"]["primary_decision"]!="criterion_not_met" or base["probe_forecast_ledger_sha256"]!=original_forecast:
        raise AssertionError("POSTHOC requires frozen PEF-13 negative result and identical forecast")
    cat,recs,prec,_=p13.validate(evidence,selection_ledger,angle_ledger,precision_ledger,jacobian_ledger)
    forecasts=json.loads(Path(forecast_path).read_text())
    records=[]
    for rec in recs:
        seed,task=rec["seed"],rec["task"]
        model=p8.checked_model(evidence,rec)
        probe=cat["probe"]
        truth=p8.labels(probe,task)
        with torch.no_grad():
            h8=p8.states_by_loop(model,probe)[8]
        selected=next(x for x in prec["model_selections"] if (x["seed"],x["task"])==(seed,task))
        held_bits=cat["heldout"]
        held_truth=p8.labels(held_bits,task)
        with torch.no_grad():
            held_h8=p8.states_by_loop(model,held_bits)[8]
        frozen=next(x for x in forecasts["models"] if (x["seed"],x["task"])==(seed,task))
        observed=next(x for x in base["heldout_results"] if (x["seed"],x["task"])==(seed,task))
        center=p11.v64(selected["center"])
        for condition in frozen["conditions"]:
            c,fit,w=p13.condition_frozen(selected,condition["tangent"],condition["angle_deg"])
            with torch.no_grad():
                ablated=p9.projector(h8,center,p11.v64(c["direction"]))
                corrected=torch.stack([p11.scalar_quant(ablated,center,w,step["absolute_step"])
                                       for step in condition["steps"]])
                replay_scores=p12.heldout_score(model,probe,corrected,fit,truth)
                held_base=p9.projector(held_h8,center,p11.v64(c["direction"]))
            # Posthoc only: heldout derivatives were NEVER used in pre-registered predictions.
            held_margins,held_jac=p12.probe_margin_jacobians(model,held_bits,held_base,fit,{"transverse":w})
            held_q=(held_base.double()-center)@w
            hrow=next(x for x in observed["conditions"] if x["tangent"]==condition["tangent"]
                     and x["angle_deg"]==condition["angle_deg"])
            for s,sc,h in zip(condition["steps"],replay_scores,hrow["steps"],strict=True):
                pred=s["phase_aware"]
                on_probe=sc["correct"]/768
                on_held=h["score"]["correct"]/768
                held_linear=p13.phase_forecast(held_margins,held_jac["transverse"],held_truth,held_q,s["absolute_step"])
                records.append({"seed":seed,"task":task,"tangent":condition["tangent"],
                     "angle_deg":condition["angle_deg"],"relative_step":s["relative_step"],
                     "probe_linear":pred,"probe_nonlinear":on_probe,"heldout_nonlinear":on_held,
                     "heldout_linear_posthoc":held_linear,
                     "linear_phase_context_shift":abs(pred-held_linear),
                     "within_heldout_linear_error":abs(held_linear-on_held),
                     "within_probe_linear_error":abs(pred-on_probe),
                     "nonlinear_context_shift":abs(on_probe-on_held),
                     "total_forecast_error":abs(pred-on_held),
                     "triangle_slack":abs(pred-on_probe)+abs(on_probe-on_held)-abs(pred-on_held)})
    if len(records)!=96 or p13.sha(forecast_path)!=original_forecast:
        raise AssertionError("diagnostic changed frozen forecast or lost records")
    rows=[]
    for task in ("coalition","threshold"):
        group=[x for x in records if x["task"]==task]
        rows.append({"task":task,"condition_count":len(group),
            **{k:sum(row[k] for row in group)/len(group) for k in (
                "within_probe_linear_error","nonlinear_context_shift","total_forecast_error","triangle_slack",
                "linear_phase_context_shift","within_heldout_linear_error")}})
    report={"campaign":"PEF-13","analysis":"POSTHOC_REPLAY_DIAGNOSTIC",
       "negative_primary_decision_preserved":True,
       "forecast_sha256_unchanged":original_forecast,
       "rows":records,"summary_by_task":rows}
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    (output/"posthoc_diagnostic.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    with (output/"posthoc_diagnostic.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(rows,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence","selection-ledger","angle-ledger","precision-ledger","jacobian-ledger","forecast","summary","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    a=ap.parse_args()
    diagnose(a.evidence,a.selection_ledger,a.angle_ledger,a.precision_ledger,a.jacobian_ledger,a.forecast,a.summary,a.out)

if __name__=="__main__":main()