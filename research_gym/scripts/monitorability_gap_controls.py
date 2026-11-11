"""Run constructed, exact/analytic controls for RMP-MG-01."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from research_gym.analysis.monitorability_gap import (
    strongest_null_direction, discrete_cmi, balanced_bayes_accuracy,
    taylor_bounds, log_loss,
)


def controls() -> dict:
    cases = []
    def add(name, passed, **data):
        cases.append({"name": name, "passed": bool(passed), **data})
    # Full state is (x1,x2); the computation exposes x1*x2 as a new coordinate.
    # A balanced affine logistic decoder of (x1,x2) has its CE optimum at 0.5;
    # symmetry + convexity proves this. A fixed +/-8 score on x1*x2 reads it.
    points = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=float)
    labels = ((points[:,0] * points[:,1] + 1) / 2).astype(int)
    before = np.full((4,2), .5)
    score = 8 * points[:,0] * points[:,1]
    p1 = 1 / (1 + np.exp(-score))
    after = np.stack([1-p1,p1],1)
    early, late = log_loss(labels,before), log_loss(labels,after)
    # h_next=product, Z=label, previous state=which of four corners.
    joint = np.zeros((2,2,4))
    for i,z in enumerate(labels): joint[z,z,i] += .25
    cmi = discrete_cmi(joint)
    add("deterministic_cmi_zero_bounded_readout_improves", abs(cmi)<1e-14 and late<early,
        cmi_nats=cmi, early_logloss_nats=early, late_logloss_nats=late,
        logloss_gain_nats=early-late,
        scope="constructed XOR; affine-logistic early optimum, fixed late decoder")
    # Observation sees x, future sees y.
    c=np.array([[1.,0.]])
    b=np.array([[0.,1.]])
    d=strongest_null_direction(c,b)
    add("hidden_causal_coordinate", np.isclose(d.sensitivity,1) and d.observation_derivative_norm<1e-12,
        **d.receipt())
    # Blind coordinate exists, but future does not use it.
    d=strongest_null_direction(c,3*c)
    add("gauge_null_direction_has_zero_future_effect", d.sensitivity==0, **d.receipt())
    d=strongest_null_direction(np.eye(2),b)
    add("fully_observed_state", d.sensitivity==0 and d.nullity==0, **d.receipt())
    # Smooth observation g(x,y)=x+y^2; future k(x,y)=y.
    # At origin C=[1,0], B=[0,1]. M_g=2, M_k=0 globally.
    for r in [.001,.01,.1]:
        bound=taylor_bounds(r,1,2,0)
        add(f"quadratic_observation_linear_future_r{r:g}",
            np.isclose(r*r,bound["observation_drift_upper"]) and np.isclose(r,bound["future_drift_lower"]),
            radius=r, measured_observation_drift=r*r, measured_future_drift=r, **bound)
    # Identical CURRENT snapshots can conceal histories with different labels.
    bayes=balanced_bayes_accuracy(np.array([1.,0.]),np.array([1.,0.]))
    add("identical_observation_laws_chance_monitor", bayes==.5, balanced_accuracy=bayes)
    bayes=balanced_bayes_accuracy(np.array([.75,.25]),np.array([.25,.75]))
    add("known_total_variation_bound", bayes==.75, balanced_accuracy=bayes, total_variation=.5)
    # Blindness depends on a chosen horizon: C A initially zero on y, C A^2 nonzero.
    a=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    c=np.array([[1.,0.,0.]])
    one=strongest_null_direction(c,c@a)
    two=strongest_null_direction(np.vstack([c,c@a]),c@a@a)
    all_seen=strongest_null_direction(np.vstack([c,c@a,c@a@a]),c@a@a@a)
    add("history_can_close_observation_gap", one.sensitivity>.99 and two.sensitivity>.99 and all_seen.sensitivity==0,
        sensitivity_current=one.sensitivity, sensitivity_two_observations=two.sensitivity,
        sensitivity_three_observations=all_seen.sensitivity)
    return {"experiment":"RMP-MG-01", "status":"CONTROL_ONLY", "cases":cases,
            "passed":sum(x["passed"] for x in cases), "total":len(cases),
            "trained_model_evidence":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=controls()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# RMP-MG-01 constructed controls","",f"Status: **{result['status']}**.",
           f"Passed: **{result['passed']}/{result['total']}**.","",
           "These are analytic and constructed examples. Trained-model evidence: **false**.","",
           "| Control | Pass |","|---|---|"]
    lines += [f"| {c['name']} | {c['passed']} |" for c in result['cases']]
    (args.out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
    if result["passed"]!=result["total"]: raise SystemExit(1)


if __name__=="__main__": main()
