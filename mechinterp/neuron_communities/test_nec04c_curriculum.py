"""Prospective NEC-04C native parity-curriculum registration and sampler tests."""
import pytest
import torch

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle
from mechinterp.neuron_communities.nec04c_pretraining import (
    ROOTS, ARMS, CHECKPOINTS, SOURCE, VALIDATION, verify_lock, seed_for_root,
)
from mechinterp.neuron_communities.nec04c_native_curriculum import (
    BATCH, STEPS, STAGE1, sample_indices,
)
from mechinterp.neuron_communities.nec04c_curriculum_aggregate import (
    PRIMARY_FLOOR, MIN_POSITIVE_ROOTS, ALT_RETENTION_FLOOR,
    macro_difference,
)


def test_frozen_new_calibration_roots_and_teacher_config():
    cfg, checksum = verify_lock()
    assert len(checksum) == 64
    assert ROOTS == (6, 7, 8, 9, 10, 11)
    assert ARMS == ("vanilla", "parity_curriculum")
    assert CHECKPOINTS == (240, 900)
    assert SOURCE == 80 * 32
    assert VALIDATION == 24 * 32
    assert (BATCH, STEPS, STAGE1) == (64, 900, 240)
    assert cfg["source_SAE_training"] is False
    assert cfg["original_development_world_root_specs_materialized"] == 0
    assert cfg["original_confirmation_world_root_specs_materialized"] == 0
    assert cfg["FSI_01_gate"] == "REDESIGN_OR_PILOT_ONLY"
    assert (PRIMARY_FLOOR, MIN_POSITIVE_ROOTS, ALT_RETENTION_FLOOR) == (0.10, 4, -0.05)


def test_original_seed_formula_is_fixed_and_old_pilot_roots_rejected():
    assert len({seed_for_root(i) for i in ROOTS}) == 6
    assert seed_for_root(6) == seed_for_root(6)
    with pytest.raises(PermissionError):
        seed_for_root(5)
    with pytest.raises(PermissionError):
        seed_for_root(12)


def test_native_sampler_stage1_is_parity_only_and_stage2_balanced():
    p = torch.arange(0, 1280, dtype=torch.long)
    a = torch.arange(1280, 2560, dtype=torch.long)
    rng = torch.Generator().manual_seed(9477)
    phase1 = sample_indices("parity_curriculum", 240, rng, p, a)
    assert phase1.numel() == BATCH
    assert torch.all(phase1 < 1280)
    phase2 = sample_indices("parity_curriculum", 241, rng, p, a)
    assert phase2.numel() == BATCH
    assert int((phase2 < 1280).sum()) == 32
    assert int((phase2 >= 1280).sum()) == 32
    with pytest.raises(ValueError):
        sample_indices("parity_curriculum", 901, rng, p, a)


def test_sampler_replays_identical_sequence_from_seed():
    p = torch.arange(0, 1280, dtype=torch.long)
    a = torch.arange(1280, 2560, dtype=torch.long)
    for arm, step in (("vanilla", 1), ("parity_curriculum", 200), ("parity_curriculum", 800)):
        first = torch.Generator().manual_seed(123456)
        second = torch.Generator().manual_seed(123456)
        assert torch.equal(
            sample_indices(arm, step, first, p, a),
            sample_indices(arm, step, second, p, a),
        )


def test_original_parity_branch_is_exactly_balanced_per_complete_source_context():
    world, _ = gym.calibration_world(6)
    source_contexts, _ = gym.contexts_for_world(6)
    pos = world["context_positions"]
    focal = world["focal_positions"]
    found = False
    for context in source_contexts:
        if context[0] != 1:
            continue
        labels = []
        for mask in range(32):
            history = [0] * 12
            for k, bit in enumerate(focal):
                history[bit] = (mask >> k) & 1
            for k, bit in enumerate(pos):
                history[bit] = context[k]
            labels.append(int(oracle(world["canonical_program"], history)))
        assert sum(labels) == 16
        found = True
        break
    assert found


def test_macro_difference_uses_worlds_as_units():
    roots = [{"d": 0.1}, {"d": 0.2}, {"d": -0.1}]
    assert abs(macro_difference(roots, "d") - (0.2 / 3)) < 1e-12
