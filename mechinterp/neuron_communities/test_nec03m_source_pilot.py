"""NEC-03M 5-input source-only tensor and cohort contract tests."""
import pytest
import torch

from mechinterp.neuron_communities.nec03m_source_pilot import (
    SCHEMA,INDICES,NATIVE_STEPS,SAE_STEPS,
    SOURCE_SITE_GROUPS,SOURCE_RANK_GROUPS,SOURCE_ANGLE_GROUPS,
    native_attributions32,source_roster,run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds,context_catalog,labels,registration,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from research.pef07.pef07 import Looper,as_tensor,model_digest


def test_5_input_contract_preserves_original_confirmation_and_source_groups():
    assert INDICES==(0,1,2,3)
    assert NATIVE_STEPS==900 and SAE_STEPS==480
    assert SOURCE_SITE_GROUPS==SOURCE_RANK_GROUPS==40
    assert SOURCE_ANGLE_GROUPS==8
    assert registration()["five_input_development_worlds"]==96
    assert len(registration()["new_five_input_embargoed_world_commitments"])==32
    assert len(old_registry()["confirmation_commitments"])==64
    for idx in INDICES:
        w=worlds()[idx]
        assert w["cohort"]=="development"
        catalog,source=context_catalog(w)
        assert {k:len(v) for k,v in catalog.items()}=={
            "train":2560,"development":768,"reserved":768}
        assert len({r for v in catalog.values() for r in v})==4096
        assert source["source_fit_context_count"]==40
        assert source["source_rank_context_count"]==40


def test_full_32_state_attribution_uses_64d_memory_token_per_visit():
    torch.set_num_threads(1)
    w=worlds()[0]
    c,_=context_catalog(w)
    bits=as_tensor(c["train"][:32])
    torch.manual_seed(42)
    model=Looper().eval()
    original=model_digest(model)
    candidates=native_attributions32(model,bits)
    assert len(candidates)==12*2
    assert len(set(candidates))==len(candidates)
    assert all(0<=int(k.split(":")[0])<12 and
               0<=int(k.split(":")[1])<64 for k in candidates)
    assert model_digest(model)==original
    with pytest.raises(ValueError,match="all 32"):
        native_attributions32(model,bits[:16])


def test_source_native_candidate_roster_can_not_open_24_group_dev_labels():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model=Looper().eval()
    w=worlds()[1]
    catalog,_=context_catalog(w)
    chosen,graph,manifest=source_roster(model,catalog,w)
    assert graph["contexts"]==40
    assert manifest["source_context_groups"]==40
    assert manifest["focal_states_per_context"]==32
    assert not manifest["source_selection_depends_on_scored_development_data"]
    if chosen:
        assert all(0<=int(n.split(":")[1])<64 for n in chosen["nodes"])
        assert len(set(chosen["nodes"]))==len(chosen["nodes"])
    assert isinstance(manifest["fallback_used"],bool)


def test_embargoed_world_cannot_generate_training_contexts():
    with pytest.raises(PermissionError,match="embargoed"):
        context_catalog(worlds()[96])
    with pytest.raises(PermissionError,match="embargoed"):
        context_catalog(worlds()[127])


def test_full_source_teacher_class_prior_unchanged_for_both_arms():
    w=worlds()[0]
    catalog,_=context_catalog(w)
    bits=as_tensor(catalog["train"][:32])
    z=labels(bits,w,control=False)
    y=labels(bits,w,control=True)
    assert z.shape==y.shape==(32,)
    assert z.sum()==y.sum()
    assert not torch.equal(z,y)
