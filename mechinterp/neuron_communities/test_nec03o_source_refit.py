"""NEC-03O source-only refit preregistration, embargo, and geometry tests."""
from __future__ import annotations

import pytest
import torch

from mechinterp.neuron_communities.nec03o_source_refit import (
    ROOTS, FIT_ROWS, RANK_ROWS, SOURCE_GEOMETRY_GROUPS, run, verify_lock,
)
from mechinterp.neuron_communities.nec03o_source_audit import audit_geometry
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, context_catalog, registration,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry


def test_registered_new_dictionary_uses_frozen_native_reset_source_only():
    lock = verify_lock()
    assert ROOTS == (0, 1, 2, 3)
    assert FIT_ROWS == RANK_ROWS == 1280
    assert SOURCE_GEOMETRY_GROUPS == 8
    assert lock["original_model_run_id"] == 44894980345
    assert lock["reset1800_model_run_id"] == 44898361742
    assert lock["native_update_count"] == 1800
    assert lock["new_native_training_updates"] == 0
    assert lock["sae_features"] == 128 and lock["sae_active_k"] == 16
    assert lock["sae_updates"] == 480
    assert lock["maximum_normalized_sae_mse"] == .03
    assert lock["max_cosine_gap"] == .25
    assert lock["minimum_angle_coverage"] == .8


def test_original_context_partitions_and_both_embargo_seals_unchanged():
    lock = verify_lock()
    assert lock["five_input_registry_sha256"] == registration()["registry_sha256"]
    assert lock["four_input_confirmation_sha256"] == old_registry()["confirmation_full_spec_sha256"]
    for i in ROOTS:
        w = worlds()[i]
        rows, manifest = context_catalog(w)
        assert w["cohort"] == "development"
        assert len(rows["train"]) == 2560
        assert len(rows["development"]) == 768
        assert len(rows["reserved"]) == 768
        assert len(set(rows["train"]) & set(rows["development"])) == 0
        assert len(set(rows["train"]) & set(rows["reserved"])) == 0
        assert len(set(rows["development"]) & set(rows["reserved"])) == 0
        assert manifest["source_fit_context_count"] == 40
        assert manifest["source_rank_context_count"] == 40
    assert all(lock[k] == 0 for k in (
        "development_evaluation_groups_scored",
        "reserved_context_groups_scored",
        "embargoed_worlds_scored",
        "four_input_confirmation_worlds_scored"))
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_preexisting_and_new_source_worlds_rejected_by_phase_a(tmp_path):
    for i in (-1, 4, 7, 96, 127):
        with pytest.raises(PermissionError):
            run(i, tmp_path, tmp_path, tmp_path / "out")


def test_direction_audit_fails_closed_on_missing_source_routes():
    with pytest.raises(AssertionError, match="source-only geometry cohort"):
        audit_geometry({
            "source_contexts": 8,
            "focal_histories_per_source_group": 32,
            "heldout_development_reserved_or_embargoed_inputs_seen": 0,
            "source_training_label_free_control_selection": True,
            "source_contexts_and_site_receipts": []},
            {"nodes": ["1:3"]}, [list(range(128))] * 12,
            {"train": [(0,) * 12] * 2560})


def test_original_native_and_new_sae_imports_are_torch_cpu_compatible():
    torch.set_num_threads(1)
    from mechinterp.neuron_communities.topk_sae import TopKSAE
    from research.pef07.pef07 import Looper
    model, sae = Looper().eval(), TopKSAE().eval()
    assert sae.features == 128 and sae.k == 16
    assert model.block.mlp_down.weight.shape == (32, 64)
    assert sae.decoder.weight.shape == (64, 128)
