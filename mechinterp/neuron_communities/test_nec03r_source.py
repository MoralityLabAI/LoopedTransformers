"""NEC-03R native-source cohort, feature/dose, and sealed-data preflight."""
from __future__ import annotations

from itertools import combinations

import pytest
import torch

from mechinterp.neuron_communities.nec03r_source_recoverability import (
    ROOTS,SOURCE_PANEL,MODELS,CASES,verify_lock,
    hashed_disjoint_subset,group_geometry,run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.topk_sae import TopKSAE
from research.pef07.pef07 import Looper


def test_frozen_population_and_dose_instruments():
    lock,_=verify_lock()
    assert ROOTS==tuple(range(8))
    assert SOURCE_PANEL==tuple(range(8,16))
    assert len(MODES)==10 and "native_nonhub" in MODES
    assert "sae_affine_nonhub" in MODES
    assert "ranked_angle" in MODES and "angle_sham" in MODES
    assert "ranked_random" in MODES and "random_active" in MODES
    assert CASES==256
    assert lock["primary_root_ITT_eligible_indices"]==[0,1,2,4,6,7]
    assert lock["primary_root_ITT_zero_indices"]==[3,5]
    assert lock["source_context_group_indices"]==list(range(8,16))
    assert lock["sae_topk"]==16 and lock["sae_features"]==128
    assert lock["primary_meaningful_effect"]==.02
    assert lock["primary_comparisons"]==1
    assert lock["random_active_salt"].startswith("NEC03R-frozen")
    assert lock["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert lock["four_input_confirmation_sha256"]==old_registry()["confirmation_full_spec_sha256"]
    assert all(lock[key]==0 for key in (
        "new_development_eval_contexts_scored","reserved_contexts_scored",
        "embargoed_worlds_scored","original_four_input_confirmation_worlds_scored",
        "native_or_sae_training_updates"))


def test_original_eight_model_worlds_retain_complete_source_groups():
    for index in ROOTS:
        world=worlds()[index]
        assert world["world_id"]==f"dev5-{index:03d}"
        rows,manifest=context_catalog(world)
        assert len(rows["train"])==2560 and len(rows["development"])==768
        assert len(rows["reserved"])==768
        assert len(set(rows["train"])&set(rows["development"]))==0
        assert len(set(rows["train"])&set(rows["reserved"]))==0
        assert manifest["source_fit_context_count"]==40
        assert manifest["source_rank_context_count"]==40
        for group in SOURCE_PANEL:
            assert len(rows["train"][group*32:(group+1)*32])==32


def test_source_root_scope_rejects_any_other_world_before_checkpoint_access(tmp_path):
    for i in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_deterministic_disjoint_active_control_contains_exactly_n_without_source():
    active=list(range(16))
    src={0,1}
    pool=sorted(set(active)-src)
    opts=list(combinations(pool,2))
    one=hashed_disjoint_subset(pool,2,"test-world","composed",8,0,3)
    assert tuple(one) in opts and not src.intersection(one)
    assert one==hashed_disjoint_subset(pool,2,"test-world","composed",8,0,3)
    assert len(one)==2
    with pytest.raises(AssertionError):
        hashed_disjoint_subset([],2,"test-world","composed",8,0,3)


def test_absent_source_candidate_retains_all_correctness_and_zero_availability():
    torch.set_num_threads(1)
    torch.manual_seed(47)
    world=worlds()[0]
    catalogue,_=context_catalog(world)
    subset=catalogue["train"][8*32:9*32]
    model=Looper().eval()
    prepared={
        "model":model,"world":world,
        "source_sites":{},"source_hubs":{},
        "sae":None,"mean":None,"scale":None,"ranks":None,
        "arm":"additive"
    }
    receipt=group_geometry(prepared,8,subset)
    assert receipt["source_group"]==8
    assert len(receipt["focal_masks"])==32
    assert len(receipt["logits"])==len(MODES)
    assert receipt["site_geometry"]=={}
    assert not any(receipt["valid_angle_pair_on_all_source_visits"])
    assert not any(receipt["valid_ranked_active_on_all_source_visits"])
    assert not any(receipt["valid_random_pair_on_all_source_visits"])
    assert all(receipt["logits"][mode]==receipt["logits"]["clean"]
               for mode in MODES)
    with pytest.raises(PermissionError):
        group_geometry(prepared,0,subset)


def test_source_full_sae_decoder_affine_and_active_support_are_distinct():
    torch.manual_seed(74)
    sae=TopKSAE().eval()
    x=torch.randn(32,64)
    mean=x.mean(dim=0)
    scale=x.std(dim=0,unbiased=False).clamp_min(1e-3)
    with torch.no_grad():
        recon,codes=sae((x-mean)/scale)
        contrib=codes@(sae.decoder.weight.T*scale[None,:])
        full=recon*scale+mean
        offset=mean+sae.decoder.bias*scale
    assert sae.decoder.weight.shape==(64,128)
    assert torch.allclose(full,contrib+offset,rtol=1e-5,atol=2e-5)
    assert not torch.allclose(full,contrib)
