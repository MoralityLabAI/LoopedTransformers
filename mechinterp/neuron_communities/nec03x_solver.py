"""NEC-03X frozen continuous disjoint physical-subspace null solver.

Fixed 495 supports, four nonnegative physical decoder coefficients,
projected box-constrained least squares with 32 deterministic steps.
All optimization uses native 32D physical geometry and clean-chosen
full-affine gradient, never task labels or intervention correctness.
"""
from __future__ import annotations

from itertools import combinations

import torch

SOLVER_SCHEMA="NEC03X/continuous-projected-disjoint-4/v1"
STEPS=32
COEFF_MULTIPLIER=3.0
RESTARTS=1
ANGLE_GATE=.25
COHESION_GATE=.75
CROSS_GRAM_FLOOR=.12
OUTPUT_VECTOR_COS_FLOOR=.25
SHARED_DOSE_FLOOR=1e-6
DOSE_RELATIVE_TOL=.01
COHESION_WEIGHT=.25
DIRECTION_WEIGHT=.25
EPS=1e-10


def cosine(a,b):
    return (a*b).sum(dim=-1)/(
        a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(EPS)


def projected_solver(physical_columns,initial_coefficients,target):
    """All supports [495,4,32] at once; original positive coefficients [495,4].

    Fixed-step projected gradient descent for ||B@w - target||_2^2/2.
    Frobenius^2 upper-bounds the spectral Hessian eigenvalue and ensures
    each projected gradient step is bounded. Initial coefficients z,
    no random restarts, 32 exact FP32 tensor operations.
    """
    if (physical_columns.ndim!=3 or physical_columns.shape[1:]!=(4,32)
        or initial_coefficients.shape!=physical_columns.shape[:2]
        or target.shape!=(32,)):
        raise ValueError("continuous matched null requires original 4×32 columns")
    upper=initial_coefficients*COEFF_MULTIPLIER
    if not bool((upper>=0).all()):
        raise AssertionError("source decoder coefficients must be nonnegative")
    x=initial_coefficients.clone()
    gram=physical_columns@physical_columns.transpose(1,2)
    rhs=(physical_columns*target[None,None,:]).sum(dim=-1)
    step=physical_columns.square().sum(dim=(1,2)).clamp_min(EPS)
    for _ in range(STEPS):
        gradient=(gram@x.unsqueeze(-1)).squeeze(-1)-rhs
        x=(x-gradient/step[:,None]).clamp_min(0)
        x=torch.minimum(x,upper)
    output=(physical_columns*x[:,:,None]).sum(dim=1)
    if (not torch.isfinite(x).all() or not torch.isfinite(output).all()
        or (x<0).any() or (x>upper+1e-6).any()):
        raise AssertionError("continuous sham solver left registered nonnegative box")
    return x,output


def physical_group_geometry(source_ids,options,cosine_matrix):
    k=len(source_ids)
    if k!=4 or len(options)!=495:
        raise ValueError("registered four-code original source and 495 disjoint controls required")
    idx=torch.as_tensor(options,dtype=torch.long)
    mean_src=sum(cosine_matrix[source_ids[i],source_ids[j]]
                 for i in range(k) for j in range(i+1,k))/6
    cand=torch.zeros(len(options),dtype=cosine_matrix.dtype)
    for i in range(k):
        for j in range(i+1,k):
            cand+=cosine_matrix[idx[:,i],idx[:,j]]
    cand/=6
    overlap=cosine_matrix[torch.as_tensor(source_ids,dtype=torch.long)[:,None,None],
                          idx[None,:,:]].abs().mean(dim=(0,2))
    return mean_src,cand,overlap


def select_continuous_sham(active,source_ids,z,physical_cols,decoder_cosines,gradient):
    """Select a single sham from all 495 supports using only frozen geometry.

    Source 32D donor equals sum(z[f]*C[f]) for original coherent quartet.
    A continuous sham has four support-restricted 64D decoder weights
    that lie in [0,3*z[f]]. Quality gating is part of the pre-registered
    selection: minimize objective among fully qualified candidates;
    otherwise keep globally minimum objective and declare inadmissible.
    """
    if (len(active)!=16 or len(source_ids)!=4
        or len(set(active))!=16 or len(set(source_ids))!=4
        or not set(source_ids)<=set(active)):
        raise ValueError("original positive TopK16 source quartet needed")
    options=list(combinations(sorted(set(active)-set(source_ids)),4))
    if len(options)!=495:
        raise AssertionError("disjoint quartet family size changed")
    idx=torch.as_tensor(options,dtype=torch.long)
    basis=physical_cols[idx]
    original_coeff=z[idx]
    source32=(z[source_ids,None]*physical_cols[source_ids]).sum(dim=0)
    continuous,control32=projected_solver(basis,original_coeff,source32)
    src_grad_cos=cosine(source32,gradient)
    angular=(cosine(control32,gradient)-src_grad_cos).abs()
    src_cohesion,cand_cohesion,overlap=physical_group_geometry(
        source_ids,options,decoder_cosines)
    cohesion_gap=(cand_cohesion-src_cohesion).abs()
    output_cos=cosine(control32,source32)
    joint=angular+COHESION_WEIGHT*cohesion_gap+DIRECTION_WEIGHT*(1-output_cos)
    feasible=((angular<=ANGLE_GATE)&(cohesion_gap<=COHESION_GATE)&
              (overlap>=CROSS_GRAM_FLOOR)&
              (output_cos>=OUTPUT_VECTOR_COS_FLOOR)&
              (control32.norm(dim=1)>SHARED_DOSE_FLOOR)&
              (source32.norm()>SHARED_DOSE_FLOOR)&
              (gradient.norm()>1e-7))
    candidate_scores=torch.where(feasible,joint,torch.full_like(joint,float("inf")))
    if bool(feasible.any()):
        winner=int(torch.argmin(candidate_scores))
    else:
        winner=int(torch.argmin(joint))
    selected=list(options[winner])
    coefficients=continuous[winner]
    values={
        "source_ids":list(source_ids),
        "selected_sham_ids":selected,
        "optimized_coefficients":coefficients.tolist(),
        "original_coefficients":original_coeff[winner].tolist(),
        "coefficient_upper_bounds":(COEFF_MULTIPLIER*original_coeff[winner]).tolist(),
        "angle_gap":float(angular[winner]),
        "cohesion_gap":float(cohesion_gap[winner]),
        "cross_group_absolute_gram_overlap":float(overlap[winner]),
        "physical_source_control_output_cosine":float(output_cos[winner]),
        "source_physical_cohesion":float(src_cohesion),
        "sham_physical_cohesion":float(cand_cohesion[winner]),
        "joint_score":float(joint[winner]),
        "source_physical_output_norm":float(source32.norm()),
        "control_physical_output_norm":float(control32[winner].norm()),
        "qualified_candidate_count":int(feasible.sum()),
        "selected_candidate_index":winner,
        "candidate_count":495,
        "selected_candidate_quality_valid":bool(feasible[winner]),
        "solver_iterations":STEPS,
        "solver_restarts":RESTARTS,
        "solver_max_relative_coefficient_multiplier":COEFF_MULTIPLIER,
        "solver_optimal_objective":float(joint[winner]),
    }
    if not all(torch.isfinite(torch.tensor(v)) for v in (
        values["angle_gap"],values["cohesion_gap"],values["joint_score"],
        values["physical_source_control_output_cosine"])):
        raise AssertionError("nonfinite physical sham objective")
    return selected,coefficients,values


def equalize_removal(source,sham,down):
    if source.shape!=sham.shape or source.ndim!=2 or source.shape[1]!=64:
        raise ValueError("matched nonhub 64D native donor required")
    n0=(source@down.T).norm(dim=1)
    n1=(sham@down.T).norm(dim=1)
    shared=torch.minimum(n0,n1)
    g0=torch.where(n0>EPS,shared/n0.clamp_min(EPS),torch.zeros_like(n0)).clamp(0,1)
    g1=torch.where(n1>EPS,shared/n1.clamp_min(EPS),torch.zeros_like(n1)).clamp(0,1)
    scaled={"source":source*g0[:,None],"sham":sham*g1[:,None]}
    norm0=(scaled["source"]@down.T).norm(dim=1)
    norm1=(scaled["sham"]@down.T).norm(dim=1)
    match=((norm0-norm1).abs()/shared.clamp_min(1e-6))<=DOSE_RELATIVE_TOL
    if (g0>1+1e-6).any() or (g1>1+1e-6).any():
        raise AssertionError("positive code donor amplified by matched null")
    return scaled,{
        "raw_source_output_L2":n0.tolist(),
        "raw_sham_output_L2":n1.tolist(),
        "shared_output_L2":shared.tolist(),
        "source_gain":g0.tolist(),"sham_gain":g1.tolist(),
        "dose_match_within_1pct":match.tolist(),
        "positive_matched_dose":(shared>SHARED_DOSE_FLOOR).tolist(),
    }
