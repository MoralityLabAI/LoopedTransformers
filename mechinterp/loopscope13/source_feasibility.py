"""LGA-10 excluded 61064 source-only native curvature-matching feasibility.

No trained weights, reserved 31064, source-goal outcome scoring, or science
model/checkpoint claims. Report all 8x8 fixed source-goal admissions under the
prospectively registered 32-attempt, angularly separated curvature matching.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import plan_pair,project_into_k
from mechinterp.loopscope12.full_jacobian import measure_full_response
from mechinterp.loopscope13.curvature_match import (
    GOALS,EXCLUDED_SEED,NEW_SEED,make_curvature_shams,native_margin_hvp
)


class QuadraticObserver(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(-1)


def run(out=None):
    torch.set_num_threads(1)
    torch.manual_seed(EXCLUDED_SEED)
    model=TinyRecursiveModel().double().eval().requires_grad_(False)
    e=QuadraticObserver()
    data,_=fixed_data(8,EXCLUDED_SEED)
    p=full_head_kernel(model.head.weight).projector
    mu=torch.zeros(48,dtype=torch.float64)
    std=torch.ones(48,dtype=torch.float64)
    cases=[]
    start=time.monotonic()
    for base in range(8):
        x=data[base:base+1]
        with torch.no_grad():
            hrec=model.reason_states(x)[2]
        for goal in GOALS:
            donor,metadata=legal_predecessor_donor(x[0],goal)
            with torch.no_grad():
                hdon=model.reason_states(donor[None])[2]
            pair=plan_pair(
                model,e,hrec,hdon,mu,std,goal=goal,
                old_class=metadata["original_goal_label"],
                donor_class=metadata["donor_goal_label"],
                teacher_seed=EXCLUDED_SEED,base_index=base,head_projector=p)
            if not pair["eligible"]:
                cases.append({"base":base,"goal":goal,"eligible":False,
                              "reason":"PAIR_SOURCE_DEGENERATE"})
                continue
            future=measure_full_response(model,hrec,goal,pair,p)
            hvp=native_margin_hvp(
                model,hrec,goal,metadata["donor_goal_label"],
                metadata["original_goal_label"])
            f=pair["fiber"]
            project=lambda v:project_into_k(v.reshape(2,48),f,p).reshape(-1)
            shams=make_curvature_shams(
                pair["directions"]["pair_energy_tangent"],
                future.row_basis,project,hvp,
                teacher_seed=EXCLUDED_SEED,base_index=base,goal=goal)
            cases.append({
                "base":base,"goal":goal,"eligible":shams.eligible,
                "reason":shams.reason,"n_curvature_shams":len(shams.shams),
                "max_curvature_error":shams.max_curvature_error,
                "max_full_j_error":shams.max_jacobian_error,
                "attempts_used":shams.attempts_used,
                "rank_future_j":future.rank,
                "n_full_j_dark":future.future_null_dimension,
            })
    total=len(cases)
    passed=sum(z["eligible"] for z in cases)
    reasons=dict(Counter(str(z["reason"]) for z in cases if not z["eligible"]))
    record={
        "schema":"LGA10/excluded-source-61064-curvature-feasibility-v1",
        "evidence_class":"SOURCE_ONLY_UNTRAINED_ENGINEERING",
        "excluded_source_seed":EXCLUDED_SEED,
        "prospective_source_seed_reserved_unopened":NEW_SEED,
        "n_new_model_roots":0,
        "n_excluded_bases":8,
        "n_goals_per_base":8,
        "n_cases":total,
        "n_admitted":passed,
        "admission_fraction":passed/total,
        "registered_first_science_admission_gate":.95,
        "pilot_feasibility_status":
            "PASS_SOURCE_FEASIBILITY" if passed/total>=.95
            else "FAIL_SOURCE_FEASIBILITY",
        "rejection_reasons":reasons,"cases":cases,
        "source_only_cpu_seconds":time.monotonic()-start,
    }
    if out is not None:
        target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print("LGA10_SOURCE_ONLY_FEASIBILITY",
          record["pilot_feasibility_status"],passed,total,
          "reasons",reasons,flush=True)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    run(p.parse_args().out)
