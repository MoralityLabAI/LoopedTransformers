"""PEF-03G independent mechanism controls on frozen/original paths."""
import hashlib
from pathlib import Path
import torch
import pytest

from research.pef03.pef03 import run
from research.pef03.pef03_future import future_reactivation_audit
from research.pef03.pef03_mediation import audit_seed


def test_mediation_reconstructs_observed_causal_effect_and_pinned_patches(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    before = hashlib.sha256((tmp_path / "seed17.frozen.pt").read_bytes()).hexdigest()
    report = audit_seed(17, tmp_path)
    reference = future_reactivation_audit(17, tmp_path)
    assert report["posthoc"]
    assert report["n_heldout_original_histories"] == 52
    assert report["all_counterfactuals_oldest_bit_only"]
    assert report["supervised_targets_identical_in_pairs"]
    assert report["head_nullspace_dimension"] == 12
    assert report["frozen_checkpoint_unchanged"]
    assert len(report["mixed_subspace_mediation"]) == 8
    for r, parent in zip(report["mixed_subspace_mediation"],
                         reference["per_visit_nullspace_transplant"]):
        assert r["patched_after_visit"] == parent["patched_after_visit_zero_based"]
        assert r["natural_full_history_flip_task_class_changes"] == report[
            "natural_oldest_bit_flip_task_class_changes"
        ]
        assert r["nullspace_only_task_class_changes"] == parent[
            "changed_predicted_task_classes"
        ]
        assert abs(r["mean_nullspace_patch_final_logit_l2"] -
                   parent["mean_final_task_logit_l2"]) < 1e-4
        assert r["max_algebraic_reconstruction_error"] < 1e-5
    assert report["mixed_subspace_mediation"][-1]["mean_nullspace_patch_final_logit_l2"] < 1e-4
    assert report["mixed_subspace_mediation"][-1]["mean_interaction_final_logit_l2"] < 1e-4
    assert [p["visit"] for p in report["paired_direction_local_jvp"]] == [0,3,6,7]
    assert report["paired_direction_local_jvp"][-1][
        "local_future_logits_jvp_along_natural_nullspace_delta_l2"
    ] < 1e-5
    assert before == hashlib.sha256((tmp_path / "seed17.frozen.pt").read_bytes()).hexdigest()


def test_mediation_sha_tamper_gate(tmp_path: Path):
    torch.set_num_threads(2)
    run(17, tmp_path, steps=2, jacobian=False)
    checkpoint = tmp_path / "seed17.frozen.pt"
    with checkpoint.open("ab") as f:
        f.write(b"altered")
    with pytest.raises(AssertionError, match="SHA256"):
        audit_seed(17, tmp_path)
