"""PEF-03F: internal nullspace hybrid intervention and Jacobian controls."""
from pathlib import Path

import pytest
import torch

from research.pef03.pef03 import run
from research.pef03.pef03_future import future_reactivation_audit


def test_recurrent_future_nullspace_reactivation_and_terminal_invariance(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    result = future_reactivation_audit(17, tmp_path)
    assert result["posthoc_after_frozen_mechanism_selection"]
    assert result["frozen_checkpoint_unchanged"]
    assert result["final_task_head_rank"] == 4
    assert result["head_invisible_nullspace_dimension"] == 12
    rows = result["per_visit_nullspace_transplant"]
    assert len(rows) == 8
    assert rows[-1]["remaining_recurrent_visits"] == 0
    assert rows[-1]["changed_predicted_task_classes"] == 0
    assert rows[-1]["max_abs_final_logit_shift"] <= 1e-4
    assert all(x["paired_test_count"] == 52 for x in rows)
    assert all(x["hybrid_off_manifold_possible"] for x in rows)
    jac = result["reference_local_jacobian_reactivation"]
    assert len(jac) == 4
    assert jac[0]["after_visit"] == 0
    assert jac[-1]["after_visit"] == 7
    assert jac[-1][
        "local_frobenius_norm_future_logit_jacobian_on_head_nullspace"
    ] < 1e-8


def test_reactivation_rejects_mutated_original_checkpoint(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    checkpoint = tmp_path / "seed17.frozen.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(AssertionError, match="integrity"):
        future_reactivation_audit(17, tmp_path)
