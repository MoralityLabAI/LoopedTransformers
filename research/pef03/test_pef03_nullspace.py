"""Posthoc PEF-03N frozen final-head nullspace causality regression tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research.pef03.pef03 import BitMemoryLooper, run
from research.pef03.pef03_nullspace import frozen_nullspace_audit


def test_exact_final_head_kernel_algebra_on_random_initialized_attention():
    torch.manual_seed(14903)
    model = BitMemoryLooper().eval()
    with torch.no_grad():
        w = model.head.weight.double()
        _, s, vh = torch.linalg.svd(w, full_matrices=True)
        rank = int((s > 1e-8).sum())
        assert rank == 4
        row = vh[:rank].T
        null = vh[rank:].T
        assert null.shape == (16, 12)
        assert float((w @ null).abs().max()) < 1e-11
        # Random hidden memories and arbitrary nullspace displacements
        # must preserve every final logit up to FP32 arithmetic.
        memory = torch.randn(35, 16)
        displacement = (torch.randn(35, 12, dtype=torch.float64) @ null.T).float()
        delta = (model.head(memory + displacement) -
                 model.head(memory)).abs().max()
        assert float(delta) < 1e-4
        assert float(((row @ row.T) + (null @ null.T) -
                      torch.eye(16, dtype=torch.float64)).abs().max()) < 1e-10


def test_posthoc_frozen_two_sided_pair_audit_does_not_edit_registered_data(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=3, jacobian=False)
    before = (tmp_path / "seed17.json").read_bytes()
    result = frozen_nullspace_audit(17, tmp_path)
    assert result["head_weight_rank"] == 4
    assert result["rowspace_dimension"] == 4
    assert result["nullspace_dimension"] == 12
    assert result["train_only_probe_size"] == 56
    assert result["original_heldout_size"] == 52
    assert result["posthoc_selected_after_pef03_paired_results"]
    assert result["max_abs_final_logit_change_after_nullspace_deletion"] < 1e-4
    assert result["max_abs_final_logit_change_after_nullspace_donor_patch"] < 1e-4
    assert set(result["ridge_probes"]) == {
        "full_16d", "head_rowspace_4d", "head_nullspace_12d"
    }
    assert all(row["denominator"] == 52 for row in result["ridge_probes"].values())
    assert (tmp_path / "seed17.json").read_bytes() == before
    assert json.loads((tmp_path / "seed17.nullspace.json").read_text()) == result


def test_nullspace_audit_rejects_checkpoint_tampering(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    ckpt = tmp_path / "seed17.frozen.pt"
    with ckpt.open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(AssertionError, match="digest"):
        frozen_nullspace_audit(17, tmp_path)
