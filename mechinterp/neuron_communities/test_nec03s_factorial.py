"""NEC-03S source-only original-world cohort, four-way dose and ITT tests."""
from __future__ import annotations

from itertools import combinations
import math

import pytest
import torch

from mechinterp.neuron_communities.nec03s_factorial import (
    ROOTS,PANEL,CASES,FOCAL,BUDGETS,OFFSETS,MODES,HASH_SALT,
    EFFECT_FLOOR,ANGLE_LIMIT,budget_width,hash_control,verify_lock,
    attenuate_to_shared_four,attenuate_random_pair,group_measure,run,
)
from mechinterp.neuron_communities.nec03s_aggregate import root_sign_flip
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as four
from research.pef07.pef07 import Looper


def test_original_eight_root_source_panel_and_26_modes_frozen():
    lock,identity=verify_lock()
    assert len(identity)==64
    assert ROOTS==tuple(range(8))
    assert PANEL==tuple(range(16,24))
    assert CASES==8*32 and FOCAL==32
    assert BUDGETS==("orig","wide") and OFFSETS==(0,1)
    assert len(MODES)==26 and len(set(MODES))==26
    assert lock["sae_active_k"]==16
    assert lock["paired_source_eligible_roots"]==[0,1,2,4,6,7]
    assert lock["paired_source_ineligible_ITT_zero_roots"]==[3,5]
    assert lock["angle_cosine_gate_both_offsets"]==ANGLE_LIMIT==.25
    assert lock["primary_minimum_meaningful_correctness_fraction"]==EFFECT_FLOOR==.02
    assert lock["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert lock["four_input_confirmation_sha256"]==four()["confirmation_full_spec_sha256"]
    assert lock["random_hash_salt"]==HASH_SALT.decode()
    for key in (
        "development_eval_groups_scored","reserved_groups_scored",
        "new_source_recruited_worlds_scored","embargoed_worlds_scored",
        "original_four_input_confirmation_worlds_scored",
        "native_and_sae_training_updates",
    ):
        assert lock[key]==0


def test_no_heldout_context_or_new_model_world_is_eligible():
    for i in ROOTS:
        world=worlds()[i]
        catalog,split=context_catalog(world)
        assert world["cohort"]=="development"
        assert len(catalog["train"])==2560
        assert len(catalog["development"])==len(catalog["reserved"])==768
        assert len(set(catalog["train"])&set(catalog["development"]))==0
        assert len(set(catalog["train"])&set(catalog["reserved"]))==0
        assert split["source_fit_context_count"]==40
        assert split["source_rank_context_count"]==40
        for group in PANEL:
            assert len(catalog["train"][group*32:(group+1)*32])==32
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_reject_any_outside_original_eight_roots_before_file_access(tmp_path):
    for i in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_pre_registered_original_double_cap_and_disjoint_sha_controls():
    for n in range(1,9):
        assert budget_width(n,"orig")==n
        assert budget_width(n,"wide")==min(2*n,8)
    for bad in (0,9,16):
        with pytest.raises(ValueError):
            budget_width(bad,"orig")
    active=list(range(16))
    for k in (1,2,4,8):
        ids=tuple(range(k))
        other=sorted(set(active)-set(ids))
        options=list(combinations(other,k))
        pick=hash_control(options,world="source-world",arm="additive",
                          group=16,mask=3,visit=7,budget="orig")
        assert pick in options and not set(ids).intersection(pick)
        assert pick==hash_control(options,world="source-world",arm="additive",
                          group=16,mask=3,visit=7,budget="orig")
    with pytest.raises(AssertionError):
        hash_control([],world="world",arm="composed",group=16,
                     mask=0,visit=0,budget="orig")


def test_all_four_offsets_and_shams_identically_attenuated_without_amplification():
    torch.manual_seed(194)
    down=torch.randn(32,64)/8
    donor={k:torch.randn(5,64)*(i+1)
           for i,k in enumerate(("src0","sham0","src1","sham1"))}
    scaled,receipt=attenuate_to_shared_four(donor,down)
    target=torch.tensor(receipt["shared"])
    assert (target>0).all()
    for key in donor:
        physical=(scaled[key]@down.T).norm(dim=1)
        assert torch.allclose(physical,target,rtol=1e-5,atol=1e-5)
        gain=torch.tensor(receipt["gains"][key])
        assert ((gain>=0)&(gain<=1)).all()
    assert all(receipt["matched_within_1pct"])
    assert all(receipt["nonzero_shared"])
    a,b,r=attenuate_random_pair(donor["src0"],donor["sham0"],down)
    assert torch.allclose((a@down.T).norm(dim=1),
                          (b@down.T).norm(dim=1),atol=1e-5)


def test_missing_source_site_has_zero_primary_and_all_modes_scoreable():
    torch.set_num_threads(1)
    torch.manual_seed(801)
    world=worlds()[3]
    catalog,_=context_catalog(world)
    rows=catalog["train"][16*32:17*32]
    prepared={
        "world":world,"arm":"additive","model":Looper().eval(),
        "source_sites":{},"source_hubs":{},"sae":None,
        "mean":None,"scale":None,"ranks":None,
    }
    item=group_measure(prepared,16,rows)
    assert item["original_source_group_index"]==16
    assert item["source_site_geometry"]=={}
    assert len(item["logits"])==26
    for budget in BUDGETS:
        assert sum(item["valid_all_visits"][budget])==0
        assert sum(item["valid_random_all_visits"][budget])==0
        assert item["logits"][f"{budget}_src0"]==item["logits"][f"{budget}_sham0"]
        assert item["logits"][f"{budget}_src1"]==item["logits"][f"{budget}_sham1"]
    assert all(item["logits"][key]==item["logits"]["clean"] for key in MODES)
    with pytest.raises(PermissionError):
        group_measure(prepared,8,rows)


def test_exact_root_sign_flip_is_declared_exploratory_only():
    assert root_sign_flip([0.]*8)==(1.,0)
    p,n=root_sign_flip([.1,.1,.1,.1,0,0,0,0])
    assert n==4 and math.isclose(p,.125)
