"""Posthoc PEF-03 paired replay integrity and target-only negative controls."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from research.pef03.pef03 import run
from research.pef03.pef03_paired import paired_replay


def test_paired_counterfactual_uses_immutable_original_receipt(tmp_path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=3, jacobian=False)
    before = json.loads((tmp_path / "seed17.json").read_text())
    replay = paired_replay(17, tmp_path)
    assert replay["posthoc_selected_after_frozen_scored_result"]
    assert replay["heldout_pairs"] == 52
    assert replay["supervised_task_targets_unchanged"]
    assert replay["last_two_bit_target_only_probe_flip_count"] == 0
    assert replay["last_two_bit_task_head_class_unchanged"] <= 52
    assert replay["oldest_bit_probe_both_directions_correct"] <= 52
    assert replay["oldest_bit_probe_original_correct"] <= 52
    assert replay["oldest_bit_probe_flipped_correct"] <= 52
    after = json.loads((tmp_path / "seed17.json").read_text())
    assert before == after, "posthoc read-only run changed the registered receipt"


def test_paired_replay_refuses_modified_frozen_checkpoint(tmp_path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    ck = tmp_path / "seed17.frozen.pt"
    with ck.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(AssertionError, match="digest mismatch"):
        paired_replay(17, tmp_path)
