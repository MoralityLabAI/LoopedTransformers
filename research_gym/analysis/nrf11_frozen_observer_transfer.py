"""NRF-11 frozen-observer replay on new input folds 12 and 13.

Model weights, selected gamma and all rank-8 bases come from NRF-10's frozen
fold1/fold9 procedure. Validate source checkpoint SHA and calibration receipts
before evaluating either fresh fold. Save usable bases and ridge heads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from research_gym.analysis import nrf10_leakage_aware_observer as n10

SEEDS=n10.SEEDS
DEPTHS=n10.DEPTHS


def reconstruct(seed:int,depth:int,checkpoint:Path,source_receipts:Path,prior_receipt:Path):
    if seed not in SEEDS or depth not in DEPTHS:
        raise ValueError("outside preregistered cohort")
    actual=n10.source_sha(checkpoint,seed,depth,source_receipts)
    prior=json.loads(prior_receipt.read_text())
    if (prior["seed"],prior["depth"],prior["checkpoint_sha256"])!=(seed,depth,actual):
        raise RuntimeError("NRF-10 receipt identity mismatch")
    torch.set_num_threads(1)
    W=n10.weight_tuple(checkpoint)
    def data(fold,n):
        x,y=n10.gaussian_inputs(seed,fold,n)
        return n10.half_states(W,x,depth),y
    with torch.no_grad():
        train_h,train_y=data(1,160)
        cal_h,cal_y=data(9,96)
        shadow,_=data(9,12)
        mean_h,mean_y=train_h.mean(0),train_y.mean(0)
        supervised=n10.supervised_basis(train_h,train_y,mean_h,mean_y).T
        U0=torch.linalg.qr(supervised,mode="reduced")[0]
        Jcal=n10.dense_jacobians(W,shadow,depth)
        reference=n10.evaluate(U0,train_h,train_y,mean_h,mean_y,cal_h,cal_y,Jcal)
    task_only=n10.optimize(U0,train_h,train_y,mean_h,mean_y,Jcal,0.)
    options=[]
    for gamma in n10.GAMMAS:
        basis=n10.optimize(U0,train_h,train_y,mean_h,mean_y,Jcal,gamma)
        val=n10.evaluate(basis,train_h,train_y,mean_h,mean_y,cal_h,cal_y,Jcal)
        val["gamma"]=gamma
        val["eligible"]=val["heldout_mse"]<=1.10*reference["heldout_mse"]
        options.append((val,basis))
    eligible=[c for c in options if c[0]["eligible"]]
    selected=(min(eligible,key=lambda c:(c[0]["objective"],c[0]["gamma"]))
              if eligible else min(options,key=lambda c:(c[0]["heldout_mse"],c[0]["gamma"])))
    if selected[0]["gamma"]!=prior["calibration"]["selected_gamma"]:
        raise AssertionError("selected gamma moved after preregistration")
    saved={c["gamma"]:c for c in prior["calibration"]["candidates"]}
    for score,_ in options:
        for key in ("heldout_mse","local_fiber_leakage","objective"):
            if abs(score[key]-saved[score["gamma"]][key])>1e-11:
                raise AssertionError("NRF-10 calibration numerical replay failed")
    return dict(W=W,data=data,train_h=train_h,train_y=train_y,
                mean_h=mean_h,mean_y=mean_y,U0=U0,Utask=task_only,
                Ujoint=selected[1],gamma=selected[0]["gamma"],
                checksum=actual,prior=prior)


def run(seed:int,depth:int,checkpoint:Path,source_receipts:Path,prior_receipt:Path,outdir:Path):
    d=reconstruct(seed,depth,checkpoint,source_receipts,prior_receipt)
    # Only now generate fold12 task and fold13 shadow inputs.
    with torch.no_grad():
        test_h,test_y=d["data"](12,96)
        shadow,_=d["data"](13,12)
        J=n10.dense_jacobians(d["W"],shadow,depth)
        basis={"supervised":d["U0"],"task_only":d["Utask"],"joint":d["Ujoint"]}
        for i in range(8):
            basis[f"random_{i}"]=n10.random_basis(400000+31*i+17*seed).T
        arms={name:n10.evaluate(U,d["train_h"],d["train_y"],d["mean_h"],d["mean_y"],
                               test_h,test_y,J) for name,U in basis.items()}
        vals=sorted(arms[f"random_{i}"]["objective"] for i in range(8))
        median=(vals[3]+vals[4])/2
        sup,joint=arms["supervised"],arms["joint"]
        gate={
            "task_limit_passed":joint["heldout_mse"]<=1.10*sup["heldout_mse"],
            "leakage_limit_passed":joint["local_fiber_leakage"]<=.90*sup["local_fiber_leakage"],
            "joint_dominates_supervised_and_random":joint["objective"]<min(sup["objective"],median)}
        gate["primary_passed"]=all(gate.values())
        heads={name:n10.ridge_from_U(d["train_h"],d["train_y"],U,d["mean_h"],d["mean_y"]).detach().clone()
               for name,U in basis.items() if not name.startswith("random_")}
    outdir.mkdir(parents=True,exist_ok=True)
    observer_dir=outdir/"observers"
    observer_dir.mkdir(exist_ok=True)
    path=observer_dir/f"nrf11_observer_R{depth}_S{seed}.pt"
    torch.save({
        "study_id":"NRF-11-frozen-observer-fresh-input-transfer-v1",
        "seed":seed,"depth":depth,"model_checkpoint_sha256":d["checksum"],
        "selected_gamma":d["gamma"],"supervised_basis":d["U0"],
        "task_only_basis":d["Utask"],"joint_basis":d["Ujoint"],
        "mean_h":d["mean_h"],"mean_y":d["mean_y"],
        "supervised_head":heads["supervised"],"task_only_head":heads["task_only"],
        "joint_head":heads["joint"]},path)
    prior=d["prior"]["heldout"]
    result={
        "study_id":"NRF-11-frozen-observer-fresh-input-transfer-v1",
        "seed":seed,"depth":depth,"checkpoint_sha256":d["checksum"],
        "observer_adapter_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "observer_adapter_file":path.name,
        "selected_gamma_frozen_from_NRF10":d["gamma"],
        "comparison_random_median_complete_J":median,
        "arms":arms,"primary_gate":gate,
        "prior_NRF10":{"supervised":prior["arms"]["supervised"],
                       "task_only":prior["arms"]["task_only"],
                       "joint":prior["arms"]["joint"],
                       "random_median_complete_J":prior["random_median_complete_objective"]},
        "scope":"Fresh folds12/13 of the same Gaussian synthetic teacher, no new optimization after gamma selection."}
    (outdir/f"nrf11_R{depth}_S{seed}.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--depth",type=int,required=True,choices=DEPTHS)
    p.add_argument("--checkpoints",type=Path,required=True)
    p.add_argument("--source-receipts",type=Path,required=True)
    p.add_argument("--prior-receipts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.seed,a.depth,
        a.checkpoints/f"nrf09_tied_R{a.depth}_S{a.seed}_steps32.pt",
        a.source_receipts,
        a.prior_receipts/f"nrf10_R{a.depth}_S{a.seed}.json",a.out)
    s,j=result["arms"]["supervised"],result["arms"]["joint"]
    print(f"R{a.depth} S{a.seed} MSEratio={j['heldout_mse']/s['heldout_mse']:.4f} "
          f"Lratio={j['local_fiber_leakage']/s['local_fiber_leakage']:.4f} "
          f"J={j['objective']:.5f} primary={result['primary_gate']['primary_passed']}")


if __name__=="__main__":
    main()
