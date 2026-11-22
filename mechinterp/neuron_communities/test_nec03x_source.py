"""NEC-03X frozen original source-only continuous null unit and method tests."""
from __future__ import annotations

import math
from itertools import combinations

import pytest
import torch

from mechinterp.neuron_communities.nec03x_source import (
    ROOTS,PANEL,ELIGIBLE,ITT_ZERO,CASES,FOCAL,MODES,verify_lock,group_measure,run,
)
from mechinterp.neuron_communities.nec03x_solver import (
    STEPS,COEFF_MULTIPLIER,RESTARTS,
    ANGLE_GATE,COHESION_GATE,CROSS_GRAM_FLOOR,OUTPUT_VECTOR_COS_FLOOR,
    SOLVER_SCHEMA,projected_solver,select_continuous_sham,
    physical_group_geometry,equalize_removal,
)
from mechinterp.neuron_communities.nec03x_aggregate import root_signflip
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as four
from research.pef07.pef07 import Looper


def test_frozen_original_cohort_solver_and_power_gates():
    cfg,sha=verify_lock()
    assert len(sha)==64
    assert ROOTS==tuple(range(8)) and PANEL==tuple(range(40,48))
    assert CASES==256 and FOCAL==32 and len(MODES)==9
    assert ELIGIBLE==(0,1,2,4,6,7) and ITT_ZERO==(3,5)
    assert cfg["original_root_indices"]==list(ROOTS)
    assert cfg["original_source_group_indices"]==list(PANEL)
    assert cfg["projected_solver_steps"]==STEPS==32
    assert cfg["projected_solver_restarts"]==RESTARTS==1
    assert cfg["coefficient_upper_multiplier"]==COEFF_MULTIPLIER==3.
    assert cfg["gradient_angle_gap_max"]==ANGLE_GATE==.25
    assert cfg["physical_group_cohesion_gap_max"]==COHESION_GATE==.75
    assert cfg["cross_gram_absolute_overlap_min"]==CROSS_GRAM_FLOOR==.12
    assert cfg["source_control_output_cosine_min"]==OUTPUT_VECTOR_COS_FLOOR==.25
    assert cfg["sham_supports_enumerated"]==495
    assert cfg["primary_meaningful_floor"]==.02
    assert cfg["qualified_root_positive_gate"]==5
    assert cfg["matched_input_gate_per_eligible_arm"]==205
    assert cfg["root_uncertainty_exploratory_signflip_p_gate"]==.05
    assert cfg["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert cfg["four_input_confirmation_sha256"]==four()["confirmation_full_spec_sha256"]
    assert all(cfg[k]==0 for k in (
        "new_native_or_SAE_training_updates","new_development_eval_groups_scored",
        "reserved_groups_scored","new_recruited_worlds_scored",
        "five_input_embargoed_worlds_scored","old_four_input_confirmation_worlds_scored"))


def test_exact_complete_training_source_groups_only():
    for i in ROOTS:
        world=worlds()[i]
        catalog,manifest=context_catalog(world)
        assert world["world_id"]==f"dev5-{i:03d}"
        assert len(catalog["train"])==2560
        assert len(catalog["development"])==len(catalog["reserved"])==768
        assert len(set(catalog["train"])&set(catalog["development"]))==0
        assert len(set(catalog["train"])&set(catalog["reserved"]))==0
        assert manifest["source_fit_context_count"]==40
        assert manifest["source_rank_context_count"]==40
        assert all(len(catalog["train"][g*32:(g+1)*32])==32 for g in PANEL)
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_unregistered_root_world_rejected_before_native_checkpoint_read(tmp_path):
    for i in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_projected_solver_nonnegative_bounded_and_objective_descends():
    torch.manual_seed(472)
    physical=torch.randn(7,4,32)/4
    starting=torch.rand(7,4)+.1
    target=torch.randn(32)/3
    before=(physical*starting[:,:,None]).sum(1)
    error_before=(before-target).square().sum(1)
    weights,pred=projected_solver(physical,starting,target)
    assert weights.shape==(7,4) and pred.shape==(7,32)
    assert torch.isfinite(weights).all()
    assert (weights>=0).all()
    assert (weights<=3*starting+1e-6).all()
    after=(pred-target).square().sum(1)
    assert torch.all(after<=error_before+1e-5)
    with pytest.raises(ValueError):
        projected_solver(torch.randn(7,5,32),starting,target)


def test_exact_495_supports_disjoint_and_finite_control_geometry():
    torch.manual_seed(90)
    phys=torch.randn(128,32)
    cosine=phys@phys.T/(phys.norm(dim=1)[:,None]*
                           phys.norm(dim=1)[None,:]).clamp_min(1e-10)
    codes=torch.zeros(128)
    codes[:16]=torch.rand(16)+.1
    active=list(range(16))
    source=[0,1,2,3]
    supports=list(combinations(list(range(4,16)),4))
    assert len(supports)==495
    cohesion,other,cross=physical_group_geometry(source,supports,cosine)
    assert other.shape==(495,) and cross.shape==(495,)
    assert -1<=float(cohesion)<=1 and torch.isfinite(cross).all()
    ids,weights,receipt=select_continuous_sham(
        active,source,codes,phys,cosine,torch.randn(32))
    assert len(ids)==4 and set(ids)<=set(range(4,16))
    assert len(set(ids))==4 and not set(ids)&set(source)
    assert weights.shape==(4,)
    assert receipt["selected_sham_ids"]==ids
    assert receipt["candidate_count"]==495
    assert receipt["solver_iterations"]==STEPS
    assert receipt["solver_restarts"]==1
    assert 0<=receipt["qualified_candidate_count"]<=495
    assert all(0<=a<=u+1e-5 for a,u in zip(
        receipt["optimized_coefficients"],receipt["coefficient_upper_bounds"]))


def test_min_physical_dose_equalized_without_source_or_sham_amplification():
    torch.manual_seed(67)
    down=torch.randn(32,64)/8
    source=torch.randn(32,64)
    sham=torch.randn(32,64)*2.
    pair,report=equalize_removal(source,sham,down)
    target=torch.tensor(report["shared_output_L2"])
    assert (target>0).all()
    for k in ("source","sham"):
        assert all(0<=gain<=1 for gain in report[f"{k}_gain"])
        assert torch.allclose((pair[k]@down.T).norm(dim=1),
                              target,rtol=1e-5,atol=1e-5)
    assert all(report["positive_matched_dose"])
    assert all(report["dose_match_within_1pct"])


def test_missing_source_candidate_all_modes_identical_and_no_fictional_sham():
    torch.set_num_threads(1)
    torch.manual_seed(87)
    world=worlds()[3]
    data,_=context_catalog(world)
    histories=data["train"][40*32:41*32]
    prepared={"world":world,"arm":"additive","model":Looper().eval(),
              "source_sites":{},"source_hubs":{},
              "sae":None,"mean":None,"scale":None,"ranks":None}
    result=group_measure(prepared,40,histories)
    assert result["group"]==40
    assert result["site_geometry"]=={}
    assert result["continuous_available_all_visits"]==[False]*32
    assert result["discrete_available_all_visits"]==[False]*32
    assert all(result["logits"][name]==result["logits"]["clean"] for name in MODES)
    with pytest.raises(PermissionError):
        group_measure(prepared,39,histories)


def test_root_signflip_exact_and_exploratory_only():
    assert root_signflip([0]*8)==(1.,0)
    p,k=root_signflip([.1,.1,.1,.1,0,0,0,0])
    assert k==4 and math.isclose(p,.125)
