"""NRF-08X: exact two-block decomposition of full shadow Jacobians.

J is the first eight-step tangent from a fixed midpoint, S the next
eight-step tangent. P S J Q = (P S P)(P J Q) + (P S Q)(Q J Q).
This is an exploratory independent dense-matrix check of reverse-JVP leakage.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def tangent_pair(parameters,h,*,split=8):
    import torch
    w1,b1,w2,b2 = parameters
    eye = torch.eye(len(h),dtype=h.dtype)
    J,S = eye.clone(),eye.clone()
    state = h.clone()
    for t in range(2*split):
        a = torch.tanh(w1@state+b1)
        derivative = 1-a*a
        step = eye+.25*(w2*derivative[None,:])@w1
        state = state+.25*(w2@a+b2)
        if t < split:
            J = step@J
        else:
            S = step@S
    return J,S

def transport_components(parameters,midpoint,basis,*,split=8):
    import torch
    eye = torch.eye(len(midpoint),dtype=midpoint.dtype)
    P = basis.T@basis
    Q = eye-P
    J,S = tangent_pair(parameters,midpoint,split=split)
    old = P@J@Q
    A = P@S@P@old
    B = P@S@Q@J@Q
    total = P@S@J@Q
    norm = lambda m: float(torch.linalg.vector_norm(m))/math.sqrt(len(basis))
    residual = float(torch.linalg.vector_norm(total-A-B))
    if residual > 1e-9:
        raise AssertionError(f"block identity residual {residual}")
    interference = 2*float((A*B).sum())/len(basis)
    if abs(norm(A)**2+norm(B)**2+interference-norm(total)**2)>1e-9:
        raise AssertionError("Frobenius energy identity residual")
    return {
        "L8":norm(old), "advected":norm(A),
        "fresh_dark_injection":norm(B),
        "interference_2inner_over_k":interference, "L16":norm(total),
    }

def audit_snapshot(snapshot,*,split=8):
    import torch
    from research_gym.analysis.nrf07_independent_seed_train import gaussian_inputs
    from research_gym.analysis.nrf04_semantic_fiber import (
        blocks,half_states,supervised_basis,leakage_at)
    seed,T=snapshot["seed"],snapshot["rounds"]
    state=snapshot["state_dict"]
    parameters=blocks(tuple(state[key] for key in (
        "blocks.0.in_proj.weight","blocks.0.in_proj.bias",
        "blocks.0.out_proj.weight","blocks.0.out_proj.bias")))
    with torch.no_grad():
        train_x,train_y=gaussian_inputs(seed,1,160)
        train_h=half_states(parameters,train_x,T)
        basis=supervised_basis(train_h,train_y,train_h.mean(0),train_y.mean(0))
        shadow_x,_=gaussian_inputs(seed,6,12)
        shadow_h=half_states(parameters,shadow_x,T)
        rows=[]
        for h in shadow_h:
            r=transport_components(parameters,h,basis,split=split)
            for s,key in ((split,"L8"),(2*split,"L16")):
                measured=leakage_at(parameters,h,s,basis)
                if abs(r[key]-measured)>1e-9:
                    raise AssertionError("explicit Jacobian vs reverse-JVP mismatch")
            rows.append(r)
    means={name:sum(r[name] for r in rows)/len(rows) for name in rows[0]}
    denom=sum(r["L16"]**2 for r in rows)
    return {
        "id":f"R{T}_S{seed}_E{snapshot['exposure']}",
        "means":means,
        "positive_interference_count":sum(
            r["interference_2inner_over_k"]>0 for r in rows),
        "interference_fraction_of_final_squared_leakage":
            sum(r["interference_2inner_over_k"] for r in rows)/denom,
        "shadow_count":len(rows),"per_shadow":rows,
    }

def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(109,113,127),required=True)
    p.add_argument("--trained-rounds",type=int,choices=(16,32),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    item=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    snapshot={"seed":args.seed,"rounds":args.trained_rounds,
              "exposure":item["exposure"],"state_dict":item["state_dict"]}
    result=audit_snapshot(snapshot)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(result["id"],result["means"])

if __name__=="__main__":
    main()
