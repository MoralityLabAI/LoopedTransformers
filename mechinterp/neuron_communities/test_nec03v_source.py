"""NEC-03V original source-only coherent-group physical sham tests."""
from __future__ import annotations

from itertools import combinations
import math

import pytest
import torch

from mechinterp.neuron_communities.nec03v_coherent_source import (
    ROOTS,PANEL,QUALIFIED,ITT_ZERO,CASES,FOCAL,GROUPS,MODES,
    ANGLE_LIMIT,COHESION_LIMIT,HASH_SALT,verify_lock,
    source_coherent_group,physical_cohesion,hashed_sham,equal_output_dose,
    group_measure,run,
)
from mechinterp.neuron_communities.nec03v_aggregate import signflip
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as four
from research.pef07.pef07 import Looper


def test_original_eight_world_cohort_and_treatment_lock():
    cfg,sha=verify_lock()
    assert len(sha)==64
    assert ROOTS==tuple(range(8)) and PANEL==tuple(range(32,40))
    assert QUALIFIED==(0,1,2,4,6,7) and ITT_ZERO==(3,5)
    assert CASES==256 and FOCAL==32
    assert GROUPS=={"four":4,"six":6}
    assert len(MODES)==11 and len(set(MODES))==11
    assert cfg["intervention_modes"]==list(MODES)
    assert cfg["maximum_angle_cosine_gap"]==ANGLE_LIMIT==.25
    assert cfg["maximum_physical_cohesion_gap"]==COHESION_LIMIT==.50
    assert cfg["hash_salt"]==HASH_SALT.decode()
    assert cfg["primary_comparisons"]==1
    assert cfg["primary_minimum_meaningful_effect"]==.02
    assert cfg["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert cfg["old_four_input_spec_sha256"]==four()["confirmation_full_spec_sha256"]
    assert all(cfg[key]==0 for key in (
        "native_or_SAE_training_updates","new_development_eval_groups_scored",
        "reserved_groups_scored","new_recruited_worlds_scored",
        "five_input_embargoed_worlds_scored",
        "old_four_input_confirmation_worlds_scored"))


def test_original_source_training_and_protected_disjoint_groups():
    for i in ROOTS:
        world=worlds()[i]
        assert world["world_id"]==f"dev5-{i:03d}"
        catalog,manifest=context_catalog(world)
        assert len(catalog["train"])==2560
        assert len(catalog["development"])==len(catalog["reserved"])==768
        assert len(set(catalog["train"]) & set(catalog["development"]))==0
        assert len(set(catalog["train"]) & set(catalog["reserved"]))==0
        assert manifest["source_fit_context_count"]==40
        assert manifest["source_rank_context_count"]==40
        assert all(len(catalog["train"][g*32:(g+1)*32])==32 for g in PANEL)
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_unregistered_world_or_group_refused_before_checkpoint_access(tmp_path):
    for index in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(index,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_coherent_seed_plus_source_ranked_projected_decoder_neighborhood():
    # Physical decoder columns: seed0 and 1,2,3 point along x;
    # unrelated active columns point orthogonal/opposite.
    phys=torch.eye(16)
    phys[0,:]=0;phys[0,0]=1.
    phys[1,:]=0;phys[1,0]=.99;phys[1,1]=.1
    phys[2,:]=0;phys[2,0]=.98;phys[2,2]=.2
    phys[3,:]=0;phys[3,0]=.97;phys[3,3]=.3
    dot=phys@phys.T/(
        phys.norm(dim=1)[:,None]*phys.norm(dim=1)[None,:]).clamp_min(1e-10)
    rank={i:i for i in range(16)}
    src=source_coherent_group(list(range(16)),rank,dot,4)
    assert src==[0,1,2,3]
    assert math.isclose(float(physical_cohesion(src,dot)),
                        float(sum(dot[src[i],src[j]] for i in range(4) for j in range(i+1,4))/6))
    with pytest.raises(ValueError):
        source_coherent_group(list(range(15)),rank,dot,4)


def test_disjoint_quartet_and_sextet_hash_and_combinatorial_candidates():
    active=list(range(16))
    for k in (4,6):
        source=list(range(k))
        options=list(combinations(sorted(set(active)-set(source)),k))
        assert len(options)==math.comb(16-k,k)
        pick=hashed_sham(options,world_sha="fixedworld",arm="composed",
            group=32,mask=3,visit=1)
        assert tuple(pick) in options
        assert not set(pick)&set(source)
        assert pick==hashed_sham(options,world_sha="fixedworld",arm="composed",
            group=32,mask=3,visit=1)
    with pytest.raises(ValueError):
        hashed_sham([],world_sha="world",arm="composed",group=32,mask=0,visit=0)


def test_same_output_dose_and_nonamplifying_control():
    torch.manual_seed(63)
    down=torch.randn(32,64)/8
    src=torch.randn(10,64)
    sham=torch.randn(10,64)*2.
    pair,receipt=equal_output_dose(src,sham,down)
    expected=torch.tensor(receipt["shared_output_32d_dose"])
    assert (expected>0).all()
    for key in ("source","sham"):
        observed=(pair[key]@down.T).norm(dim=1)
        assert torch.allclose(observed,expected,atol=1e-5,rtol=1e-5)
        assert all(0<=x<=1 for x in receipt[f"{key}_gain"])
    assert all(receipt["matched_1pct"])
    assert all(receipt["nonzero_dose"])


def test_absent_source_site_retains_native_correctness_zero_sham_contrast():
    torch.set_num_threads(1)
    torch.manual_seed(501)
    world=worlds()[3]
    original,_=context_catalog(world)
    x=original["train"][32*32:33*32]
    prepared={"world":world,"arm":"additive","model":Looper().eval(),
              "source_sites":{},"source_hubs":{},
              "sae":None,"mean":None,"scale":None,"ranks":None}
    item=group_measure(prepared,32,x)
    assert item["source_group"]==32
    assert len(item["logits"])==len(MODES)
    assert item["site_geometry"]=={}
    assert all(not any(item["valid_group_source_sham_on_all_visits"][key])
               for key in GROUPS)
    assert not any(item["valid_hash_on_all_visits"])
    assert all(item["logits"][name]==item["logits"]["clean"] for name in MODES)
    with pytest.raises(PermissionError):
        group_measure(prepared,31,x)


def test_exact_signflip_root_only_and_registered_power_conservatism():
    assert signflip([0]*8)==(1.,0)
    p,n=signflip([.1,.1,.1,.1,0,0,0,0])
    assert n==4 and math.isclose(p,.125)
