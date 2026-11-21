"""NEC-03H site-selection and opened-root training calibration tests.

The original NEC-03G eight-world negative evidence and 64 confirmation
commitments are kept untouched. All tests use development-only worlds.
"""
from __future__ import annotations

from collections import Counter

import pytest
import torch

from mechinterp.neuron_communities import nec03h_engine as h
from mechinterp.neuron_communities.nec03g_worlds import (
    context_catalog, registry, worlds,
)
from research.pef07.pef07 import Looper, model_digest


def test_world_ranges_refuse_confirmation_and_unopened_calibration(tmp_path):
    assert h.training_world(0,calib=True)["world_id"]=="dev-000"
    assert h.training_world(8,calib=False)["world_id"]=="dev-008"
    with pytest.raises(PermissionError):
        h.training_world(64,calib=False)
    with pytest.raises(PermissionError):
        h.training_world(127,calib=False)
    with pytest.raises(PermissionError):
        h.training_world(8,calib=True)
    with pytest.raises(PermissionError):
        h.training_world(0,calib=False)
    assert len(registry()["confirmation_commitments"])==64
    assert registry()["confirmation_outcomes_exposed"] is False


def test_two_step_900_recipe_has_same_seed_as_original_stream(tmp_path):
    torch.set_num_threads(1)
    world=h.training_world(0,calib=True)
    catalog,_=context_catalog(world)
    model1,receipt1=h.train_native_900(world,catalog,tmp_path/"one",
                                        arm="composed",steps=2)
    model2,receipt2=h.train_native_900(world,catalog,tmp_path/"two",
                                        arm="composed",steps=2)
    assert receipt1["final_model_sha256"]==receipt2["final_model_sha256"]
    assert receipt1["initial_weights_sha256"]==receipt2["initial_weights_sha256"]
    assert receipt1["training_seed"]==world["trunk_seed"]
    assert receipt1["steps"]==2
    assert receipt1["same_seed_and_optimizer_stream_as_NEC03G_first_450_steps"]
    assert h.sha_file(tmp_path/"one"/receipt1["checkpoint_name"])==receipt1["checkpoint_sha256"]
    assert model_digest(model1)==model_digest(model2)
    with pytest.raises(ValueError):
        h.train_native_900(world,catalog,tmp_path/"bad",arm="composed",steps=901)


def test_source_only_fallback_is_deterministic_and_visits_follow_focal_inputs(monkeypatch):
    torch.manual_seed(103)
    model=Looper().eval()
    world=h.training_world(0,calib=True)
    catalog,_=context_catalog(world)
    k={"i":0}
    def fake_attr(_model,bits,top_k):
        assert bits.shape==(16,12) and top_k==2
        i=k["i"]
        k["i"]+=1
        return [f"{visit}:{(2*(i%16))%64}" for visit in range(12)]+[
            f"{visit}:{(2*(i%16)+1)%64}" for visit in range(12)]
    monkeypatch.setattr(h,"context_attributions",fake_attr)
    monkeypatch.setattr(h,"discover",lambda *args,**kwargs:[])
    candidate,graph,info=h.source_site_selection(model,catalog,world)
    assert candidate is not None
    assert info["fallback_used"] and not info["graph_has_qualifying_community"]
    assert len(candidate["nodes"])==4
    assert candidate["visits"]==sorted(world["focal_positions"])[-2:]
    assert len(set(candidate["nodes"]))==4
    assert not set(candidate["nodes"])&set(graph["hubs"])
    assert all(graph["frequency"][n]>=h.FALLBACK_MIN_CONTEXT_FREQ
               for n in candidate["nodes"])
    assert all(len(x["top_selection"])==2
               for x in info["fallback"]["visit_frequencies"].values())
    assert k["i"]==80


def test_existing_source_graph_candidate_keeps_fallback_disabled(monkeypatch):
    model=Looper().eval()
    world=h.training_world(1,calib=True)
    catalog,_=context_catalog(world)
    monkeypatch.setattr(h,"context_attributions",lambda model,bits,top_k:[
        f"{visit}:{visit%32}" for visit in range(12)])
    candidate={"id":"registered-source","nodes":["3:7","7:8"],"visits":[3,7],
               "graph_mean_phi":.4,"graph_edges":1}
    monkeypatch.setattr(h,"discover",lambda graph,**kwargs:[candidate])
    result,graph,info=h.source_site_selection(model,catalog,world)
    assert result==candidate
    assert info["graph_has_qualifying_community"]
    assert info["fallback_used"] is False
    assert info["fallback"] is None


def test_calibration_smoke_retains_failed_short_recipe_label(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    # Keep this smoke focused on provenance; source_selection has its own tests.
    monkeypatch.setattr(h,"source_site_selection",lambda model,catalog,world:
        (None,{"hubs":[]},{"graph_has_qualifying_community":False,
                           "fallback_used":True,"candidate":None,
                           "source_contexts":80}))
    result=h.run_calibration(0,tmp_path/"smoke",steps=2)
    assert result["full_calibration_recipe"] is False
    assert result["training_steps"]==2
    assert result["new_development_worlds_evaluated"]==0
    assert result["reserved_contexts_evaluated"]==0
    assert result["confirmation_worlds_trained_or_scored"]==0
    assert all(x["source_selection_kind"]=="none" for x in result["arms"].values())
