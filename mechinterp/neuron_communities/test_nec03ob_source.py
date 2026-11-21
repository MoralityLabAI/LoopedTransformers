"""NEC-03O-B prospective source-training allocation and embargo checks."""
from __future__ import annotations

import pytest
import torch

from mechinterp.neuron_communities.nec03ob_fresh_source import (
    ROOTS, SCHEMA, MODEL_SCHEMA, NATIVE_STEPS, RESET_STEPS,
    SOURCE_FIT_ROWS, SOURCE_RANK_ROWS, verify_lock, run,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, context_catalog, registration,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03nb_checkpoint_reset import batch_digest


def test_prospective_source_lock_is_complete_and_pinned_to_audit_run():
    x = verify_lock()
    assert ROOTS == (4,5,6,7)
    assert x["prospective_development_world_indices"] == list(ROOTS)
    assert x["preceding_source_A_verified_run_id"] == 44899750508
    assert x["source_A_feasible_model_arms"] == 7
    assert x["source_A_complete_feasible_worlds"] == 3
    assert NATIVE_STEPS == RESET_STEPS == 900
    assert x["native_adamw_lr"] == .003
    assert x["native_adamw_weight_decay"] == 0
    assert x["sae_features"] == 128 and x["sae_active_k"] == 16
    assert SOURCE_FIT_ROWS == SOURCE_RANK_ROWS == 1280
    assert x["source_geometry_groups"] == 8
    assert x["max_direction_cosine_gap"] == .25
    assert x["min_directional_coverage"] == .8
    assert x["min_physical_positive_coverage"] == .8
    assert x["min_median_retained_dose"] == .25


def test_four_unopened_worlds_are_registered_and_source_contexts_disjoint():
    x = verify_lock()
    assert registration()["registry_sha256"] == x["five_input_registry_sha256"]
    assert old_registry()["confirmation_full_spec_sha256"] == x["four_input_confirmation_sha256"]
    assert len(registration()["new_five_input_embargoed_world_commitments"]) == 32
    for i in ROOTS:
        world = worlds()[i]
        assert world["world_id"] == f"dev5-{i:03d}"
        assert world["cohort"] == "development"
        contexts, manifest = context_catalog(world)
        assert {k:len(v) for k,v in contexts.items()} == {
            "train":2560,"development":768,"reserved":768}
        assert len(set(contexts["train"]) & set(contexts["development"])) == 0
        assert len(set(contexts["train"]) & set(contexts["reserved"])) == 0
        assert manifest["source_fit_context_count"] == 40
        assert manifest["source_rank_context_count"] == 40
    for key in ("new_development_evaluation_groups_scored",
                "reserved_context_groups_scored","embargoed_worlds_scored",
                "four_input_confirmation_worlds_scored"):
        assert x[key] == 0


def test_forbidden_training_roots_are_rejected_before_context_read(tmp_path):
    for index in (0,1,2,3,8,95,96,127,-1):
        with pytest.raises(PermissionError):
            run(index,tmp_path)
    with pytest.raises(PermissionError):
        context_catalog(worlds()[96])


def test_registered_training_counter_and_source_minibatch_hashes():
    assert SCHEMA.startswith("NEC-03O-B/")
    assert MODEL_SCHEMA.startswith("NEC-03O-B/")
    a = torch.Generator().manual_seed(127*7919+59)
    b = torch.Generator().manual_seed(127*7919+59)
    first = torch.randint(2560,(128,),generator=a)
    for _ in range(900):
        torch.randint(2560,(128,),generator=b)
    resumed = torch.randint(2560,(128,),generator=b)
    assert batch_digest(first) != batch_digest(resumed)
    assert len(batch_digest(resumed)) == 64
