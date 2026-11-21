"""NEC-03T full-affine SAE necessity, source-only roster and dose tests."""
from __future__ import annotations

from itertools import combinations
import math

import pytest
import torch

from mechinterp.neuron_communities.nec03t_source_necessity import (
    ROOTS,PANEL,SOURCE_ELIGIBLE,ITT_ZERO,MODES,BUDGETS,CASES,
    ANGLE_MAX,EFFECT_FLOOR,HASH_SALT,
    budget_width,fixed_random_control,shared_removal,
    group_measure,run,verify_lock,
)
from mechinterp.neuron_communities.nec03t_aggregate import signflip
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as four
from research.pef07.pef07 import Looper


def test_frozen_original_root_and_11_mode_contract():
    cfg,digest=verify_lock()
    assert len(digest)==64
    assert ROOTS==tuple(range(8))
    assert PANEL==tuple(range(24,32))
    assert CASES==256 and len(MODES)==11
    assert SOURCE_ELIGIBLE==(0,1,2,4,6,7)
    assert ITT_ZERO==(3,5)
    assert BUDGETS==("orig","wide")
    assert cfg["all_intervention_modes"]==list(MODES)
    assert cfg["source_group_indices"]==list(PANEL)
    assert cfg["paired_ITT_zero"]==[3,5]
    assert cfg["source_active_codes"]==16
    assert cfg["angular_gap_limit"]==ANGLE_MAX==.25
    assert cfg["minimum_meaningful_effect"]==EFFECT_FLOOR==.02
    assert cfg["primary_comparisons"]==1
    assert cfg["random_control_hash_salt"]==HASH_SALT.decode()
    assert cfg["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert cfg["old_four_input_confirmation_sha256"]==four()["confirmation_full_spec_sha256"]
    for key in ("native_and_SAE_optimizer_updates","new_development_eval_groups_scored",
                "reserved_groups_scored","newly_recruited_worlds_scored",
                "embargoed_worlds_scored","old_four_input_confirmation_worlds_scored"):
        assert cfg[key]==0


def test_only_complete_old_source_training_groups_are_scored():
    for i in ROOTS:
        world=worlds()[i]
        catalog,manifest=context_catalog(world)
        assert world["cohort"]=="development"
        assert len(catalog["train"])==2560
        assert len(catalog["development"])==len(catalog["reserved"])==768
        assert len(set(catalog["train"])&set(catalog["development"]))==0
        assert len(set(catalog["train"])&set(catalog["reserved"]))==0
        assert manifest["source_fit_context_count"]==40
        assert manifest["source_rank_context_count"]==40
        assert all(len(catalog["train"][j*32:(j+1)*32])==32 for j in PANEL)
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_unregistered_worlds_rejected_before_checkpoint_access(tmp_path):
    for i in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_disjoint_control_and_budget_cannot_exceed_active_topk():
    for n in range(1,9):
        assert budget_width(n,"orig")==n
        assert budget_width(n,"wide")==min(2*n,8)
    for n in (0,9,16):
        with pytest.raises(ValueError):
            budget_width(n,"orig")
    for k in (1,2,4,8):
        selected=set(range(k))
        others=sorted(set(range(16))-selected)
        options=list(combinations(others,k))
        chosen=fixed_random_control(options,world_sha="world123",
               arm="composed",group=24,mask=3,visit=2)
        assert tuple(chosen) in options and not selected.intersection(chosen)
        assert chosen==fixed_random_control(options,world_sha="world123",
               arm="composed",group=24,mask=3,visit=2)
    with pytest.raises(AssertionError):
        fixed_random_control([],world_sha="world123",
               arm="composed",group=24,mask=3,visit=2)


def test_same_dose_physically_equal_and_never_amplified():
    torch.manual_seed(337)
    down=torch.randn(32,64)/8
    src=torch.randn(6,64)
    sham=torch.randn(6,64)*2
    vector,receipt=shared_removal(src,sham,down)
    d=torch.tensor(receipt["shared_dose"])
    assert (d>0).all()
    for k in ("source","sham"):
        gain=torch.tensor(receipt["gains"][k])
        actual=(vector[k]@down.T).norm(dim=1)
        assert ((gain>=0)&(gain<=1)).all()
        assert torch.allclose(actual,d,atol=1e-5,rtol=1e-5)
    assert all(receipt["physical_dose_equal_within_1pct"])
    assert all(receipt["nonzero_shared_dose"])


def test_no_source_site_restores_identical_clean_and_zero_contrast():
    torch.set_num_threads(1)
    torch.manual_seed(217)
    world=worlds()[3]
    original,_=context_catalog(world)
    rows=original["train"][24*32:25*32]
    prepared={
        "world":world,"arm":"additive",
        "model":Looper().eval(),"source_sites":{},"source_hubs":{},
        "sae":None,"mean":None,"scale":None,"ranks":None,
    }
    rec=group_measure(prepared,24,rows)
    assert rec["source_group"]==24
    assert rec["site_geometry"]=={}
    assert len(rec["logits"])==len(MODES)
    assert all(rec["logits"][m]==rec["logits"]["clean"] for m in MODES)
    assert all(not any(rec["valid_angular_all_visits"][b]) for b in BUDGETS)
    assert not any(rec["valid_hash_all_visits"])
    with pytest.raises(PermissionError):
        group_measure(prepared,23,rows)


def test_root_sign_flip_is_exploratory_and_exact():
    assert signflip([0.]*8)==(1.0,0)
    p,k=signflip([.1,.1,.1,.1,0,0,0,0])
    assert k==4 and math.isclose(p,.125)
