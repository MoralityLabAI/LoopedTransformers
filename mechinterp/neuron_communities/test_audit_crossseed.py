"""CPU-only NEC-02B aggregation and fail-closed source checks."""
import math
import pytest

from mechinterp.neuron_communities.audit_crossseed import (
    exact_count, candidate_record, audit,
)


def test_discrete_errors_are_exact_counts():
    assert exact_count(9/768) == 9
    assert exact_count(0.) == 0
    assert exact_count(1.) == 768
    with pytest.raises(AssertionError):
        exact_count(0.10003)
    with pytest.raises(AssertionError):
        exact_count(float("nan"))


def test_keep_negative_recovery_and_all_five_controls():
    c = lambda e: {"correctness_drop":{"mean":e/768, "contexts":48}}
    row = {
        "id":"demo","nodes":["0:1","1:4"],"visits":[0,1],
        "condition_summaries":{
            "ablate":c(0),"sham":c(0),"retain":c(12),
            "hubs":c(5),"blank":c(18)},
        "necessity_excess_kl_over_frequency_matched_sham":
            {"contexts":48,"mean":1e-7},
        "sufficiency_kl_gain_over_hubs_only":
            {"contexts":48,"mean":-0.05},
    }
    result = candidate_record(row)
    assert result["errors_after_hubs_plus_community"] == 12
    assert result["conditional_rescued_predictions"] == -7
    assert result["errors_after_targeted_ablation"] == 0
    assert result["errors_after_blank_MLP"] == 18
    del row["condition_summaries"]["sham"]
    with pytest.raises(AssertionError,match="incomplete causal"):
        candidate_record(row)


def test_missing_evidence_is_not_silently_retrained(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)
