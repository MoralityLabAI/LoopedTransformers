"""CPU-only deterministic checks; random weights are never research evidence."""
from pathlib import Path
import hashlib
import json

import pytest
import torch

from mechinterp.neuron_communities import core
from mechinterp.neuron_communities import pef07_adapter as n
from research.pef07.pef07 import Looper, model_digest, source_split, as_tensor


def test_graph_groups_correlated_visit_neurons_and_matches_shams():
    rows = ([["0:1", "1:2"]] * 4) + ([["0:4", "2:3"]] * 4)
    graph = core.build_graph(rows, min_pair_count=2, min_phi=0.4)
    assert graph["contexts"] == 8
    assert len(graph["edges"]) == 2
    candidates = core.discover(graph, max_size=8)
    assert len(candidates) == 2
    for item in candidates:
        assert len(item["nodes"]) == 2
        sham = core.matched_sham(item, graph, mlp_width=8)
        assert len(sham) == 2
        assert {core.decode(x)[0] for x in sham} == set(item["visits"])
        assert not set(sham).intersection(item["nodes"])


def test_frequency_hubs_and_context_bootstrap_are_explicit():
    graph = core.build_graph([["0:0", "0:1", "1:5"],
                              ["0:0", "0:1", "1:5"],
                              ["0:0", "0:2", "1:7"],
                              ["0:0", "0:2", "1:7"]],
                             min_pair_count=2, min_phi=0.1)
    assert graph["hubs"] == ["0:0"]
    assert all(e["a"] != "0:0" and e["b"] != "0:0" for e in graph["edges"])
    result = core.interval([0.1, 0.2, 0.3, 0.4], iterations=200, seed=42)
    assert result["contexts"] == 4
    assert result["mean"] == pytest.approx(0.25)
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_recurrent_interventions_modify_outputs_without_mutating_parameters():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    model = Looper().eval()
    hashes = model_digest(model)
    data, _ = source_split()
    bits = as_tensor(data["train"][:16])
    candidates = ["0:1", "4:1", "8:3"]
    sham = ["0:2", "4:2", "8:4"]
    assert len({core.decode(k)[0] for k in candidates}) == 3
    clean = model.logits(bits).detach()
    all_logits = {}
    for mode in ("ablate", "sham", "retain", "hubs", "blank"):
        masks = n.visit_masks(mode, candidates, ["0:0"], sham)
        all_logits[mode] = n.intervened_logits(model, bits, masks)
        assert torch.isfinite(all_logits[mode]).all()
        assert all_logits[mode].shape == clean.shape
    assert not torch.equal(clean, all_logits["blank"])
    assert not torch.equal(clean, all_logits["ablate"])
    assert model_digest(model) == hashes
    assert n.paired_effect(clean, clean, torch.zeros(16, dtype=torch.long))["kl"] == pytest.approx(0.0)


def test_context_complete_gradient_attribution_exposes_distinct_visits():
    torch.set_num_threads(1)
    torch.manual_seed(131)
    model = Looper().eval()
    data, _ = source_split()
    bits = as_tensor(data["train"][:16])
    original = model_digest(model)
    selected = n.context_attributions(model, bits, 2)
    assert len(selected) == 2*12
    assert {core.decode(x)[0] for x in selected} == set(range(12))
    assert all(0 <= core.decode(k)[1] < 64 for k in selected)
    assert model_digest(model) == original


def test_discovery_freeze_then_smoke_evaluation(tmp_path):
    torch.set_num_threads(1)
    base = ["--smoke", "--out", str(tmp_path), "--seed", "103",
            "--task", "coalition"]
    result = n.main(["discover"] + base + ["--train-contexts", "4",
                 "--top-k", "2", "--min-joint", "2", "--max-candidates", "3"])
    assert result["status"] == "exploratory-discovery"
    assert result["identity"]["provenance"] == "random-initialized-smoke"
    raw = (tmp_path/"discovery.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (tmp_path/"discovery.sha256").read_text().strip()
    output = n.main(["evaluate"] + base + ["--heldout-contexts", "2"])
    assert output["random_smoke"]
    assert output["context_count"] == 2
    assert output["candidate_roster_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (tmp_path/"heldout_causal.json").exists()
    # Any post-discovery tuning invalidates the evaluation transcript.
    altered = json.loads(raw)
    altered["top_k_per_visit"] += 1
    (tmp_path/"discovery.json").write_bytes(n.canonical(altered))
    with pytest.raises(ValueError, match="changed after candidate freeze"):
        n.main(["evaluate"] + base + ["--heldout-contexts", "2"])


def test_registered_mode_refuses_missing_frozen_weights(tmp_path):
    with pytest.raises(ValueError, match="frozen checkpoint"):
        n.main(["discover", "--out", str(tmp_path), "--seed", "103"])
