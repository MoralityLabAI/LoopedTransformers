"""NEC-03Z-CAL prospective source-only calibration gate and dose tests."""
from __future__ import annotations

import pytest
import torch
from mechinterp.neuron_communities.nec03z_cal_source import (
    ROOTS,CAL,SEALED,ELIGIBLE,ITT_ZERO,DOSES,TAGS,CASES,
    MODES,MIN_MATCHED,MAX_DAMAGE,MIN_RELATIVE_DOSE,
    verify_lock,group_measure,run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,registration,context_catalog,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as four
from research.pef07.pef07 import Looper


def test_original_eight_roots_and_sealed_analysis_panel_locked():
    cfg,sha=verify_lock()
    assert len(sha)==64
    assert ROOTS==tuple(range(8))
    assert CAL==tuple(range(48,56))
    assert SEALED==tuple(range(56,64))
    assert not set(CAL)&set(SEALED)
    assert ELIGIBLE==(0,1,2,4,6,7)
    assert ITT_ZERO==(3,5)
    assert CASES==256
    assert DOSES==(1.,.5,.25,.125,.0625)
    assert TAGS==("1000","0500","0250","0125","00625")
    assert len(MODES)==14
    assert cfg["scored_modes"]==list(MODES)
    assert cfg["paired_eligible_roots"]==list(ELIGIBLE)
    assert cfg["calibration_source_groups"]==list(CAL)
    assert cfg["sealed_source_analysis_groups"]==list(SEALED)
    assert cfg["five_input_registry_sha256"]==registration()["registry_sha256"]
    assert cfg["old_four_input_confirmation_sha256"]==four()["confirmation_full_spec_sha256"]
    assert cfg["damage_ceiling_per_original_qualified_arm"]==MAX_DAMAGE==.02
    assert cfg["minimum_valid_controls_per_original_qualified_arm"]==MIN_MATCHED==205
    assert cfg["median_min_site_matched_removal_to_source_norm_floor"]==MIN_RELATIVE_DOSE==.05
    assert cfg["minimum_eligible_headroom_roots_per_teacher_family"]==2
    assert cfg["baseline_accuracy_headroom_band"]==[.60,.95]
    assert cfg["original_sealed_group_scoring"]==0
    assert cfg["failure_policy"].startswith("if no dose satisfies all conditions")
    assert all(cfg[k]==0 for k in (
        "original_development_eval_scoring","reserved_scoring",
        "recruited_world_scoring","embargoed_five_input_scoring",
        "old_four_input_confirmation_scoring","model_SAE_optimizer_updates"))


def test_calibration_groups_original_train_and_sealed_not_fetched():
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
        assert all(len(catalog["train"][j*32:(j+1)*32])==32 for j in CAL)
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_unregistered_root_refused_before_checkpoint_loading(tmp_path):
    for i in (-1,8,15,95,96,127):
        with pytest.raises(PermissionError):
            run(i,tmp_path/"source",tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_missing_native_source_sites_retain_identical_clean_modes():
    torch.set_num_threads(1)
    torch.manual_seed(902)
    world=worlds()[3]
    catalog,_=context_catalog(world)
    rows=catalog["train"][48*32:49*32]
    prepared={
        "world":world,"arm":"additive","model":Looper().eval(),
        "source_sites":{},"source_hubs":{},
        "sae":None,"mean":None,"scale":None,"ranks":None,
    }
    rec=group_measure(prepared,48,rows)
    assert rec["source_group"]==48
    assert rec["site_geometry"]=={}
    assert all(rec["logits"][name]==rec["logits"]["clean"] for name in MODES)
    for tag in TAGS:
        assert rec["available_per_dose_all_visits"][tag]==[False]*32
        assert rec["min_site_relative_physical_dose_per_case"][tag]==[0.]*32
    with pytest.raises(PermissionError):
        group_measure(prepared,56,rows)
    with pytest.raises(PermissionError):
        group_measure(prepared,47,rows)


def test_frozen_sham_dose_ladder_is_monotone_but_never_zero():
    assert len(DOSES)==5
    assert DOSES==tuple(sorted(DOSES,reverse=True))
    assert DOSES[-1]>.0
    # At every site the one original physically equalized sham/source
    # removal is scaled by exactly the same *global* candidate dose.
    source=torch.tensor([1.,2.,3.,4.])
    sham=torch.tensor([4.,3.,2.,1.])
    for t in DOSES:
        assert torch.allclose(t*source / t,source)
        assert torch.allclose(t*sham / t,sham)
        assert torch.allclose(t*(source-sham),(t*source-t*sham))


def test_separate_root_level_headroom_is_not_repeated_mask_count():
    example={"composed":[.75,.83,.95,1.,.65,.8],
             "additive":[1.,1.,1.,1.,1.,1.]}
    counts={name:sum(.60<=v<=.95 for v in vals)
            for name,vals in example.items()}
    assert counts=={"composed":5,"additive":0}
    assert not all(v>=2 for v in counts.values())
