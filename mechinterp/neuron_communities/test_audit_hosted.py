"""NEC-02 audit tests keep old PEF-07 evidence separate from new hosted runs."""
import hashlib
import json

import pytest

from mechinterp.neuron_communities.audit_hosted import (
    ORIGINAL_LOCAL_BYTES, SOURCE_SHA, audit_exploration, audit_training,
    verify_registered_partition,
)


def test_historical_hash_inventory_has_six_unique_bytes():
    assert len(ORIGINAL_LOCAL_BYTES) == 6
    assert set(x[0] for x in ORIGINAL_LOCAL_BYTES) == {103, 131, 167}
    assert set(x[1] for x in ORIGINAL_LOCAL_BYTES) == {"coalition", "threshold"}
    assert len(set(ORIGINAL_LOCAL_BYTES.values())) == 6
    assert all(len(x) == 64 for x in ORIGINAL_LOCAL_BYTES.values())
    assert len(SOURCE_SHA) == 64


def test_audit_fails_closed_on_missing_frozen_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_training(tmp_path)


def test_audit_accepts_zero_candidates_and_full_context_accounting(tmp_path):
    folder = tmp_path/"analysis"/"seed103_coalition"
    folder.mkdir(parents=True)
    (tmp_path/"pef07").mkdir()
    (tmp_path/"pef07"/"source_split.json").write_text(json.dumps({
        "context_assignment": {}, "sha256": SOURCE_SHA
    }))
    empty_raw = folder/"raw_paired_context_traces.jsonl"
    empty_raw.write_bytes(b"")
    src = {"schema": "NEC-01/pef07-context-paired-native-neurons/v1",
           "identity": {"provenance": "pef07-frozen-trained",
                        "checkpoint_sha256": "a"*64},
           "source_sha256": SOURCE_SHA,
           "seed": 103, "task": "coalition",
           "source_split": "train_only", "context_groups_used": 160,
           "counterfactuals_per_context": 16,
           "top_k_per_visit": 2,
           "candidates": [], "graph": {"edges": [], "hubs": []}}
    raw = json.dumps(src, sort_keys=True).encode()
    (folder/"discovery.json").write_bytes(raw)
    (folder/"discovery.sha256").write_text(hashlib.sha256(raw).hexdigest()+"\n")
    report = {"schema": src["schema"], "identity": src["identity"],
              "source_sha256": SOURCE_SHA,
              "context_count": 48, "counterfactuals_per_context": 16,
              "random_smoke": False, "tests": [],
              "candidate_roster_sha256": hashlib.sha256(raw).hexdigest(),
              "raw_paired_trace_filename": "raw_paired_context_traces.jsonl",
              "raw_paired_trace_sha256": hashlib.sha256(b"").hexdigest(),
              "raw_paired_trace_records": 0}
    (folder/"heldout_causal.json").write_text(json.dumps(report))
    ledger = {"checkpoints": [{"seed": 103, "task": "coalition",
                               "checkpoint_sha256": "a"*64}]}
    result = audit_exploration(tmp_path, "coalition", ledger)
    assert result["candidate_count"] == 0
    assert result["candidates"] == []
    assert result["raw_context_trace_audit"]["raw_context_records"] == 0
    report["candidate_roster_sha256"] = "b"*64
    (folder/"heldout_causal.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        audit_exploration(tmp_path, "coalition", ledger)


def test_source_assignment_matches_registration_and_rejects_tampering():
    from research.pef07.pef07 import source_split
    _, original = source_split()
    assert original["sha256"] == SOURCE_SHA
    verify_registered_partition(original)
    altered = dict(original)
    altered["context_assignment"] = dict(original["context_assignment"])
    key = next(iter(altered["context_assignment"]))
    altered["context_assignment"][key] = "heldout"
    with pytest.raises(AssertionError, match="membership"):
        verify_registered_partition(altered)
