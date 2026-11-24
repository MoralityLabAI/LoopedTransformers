"""Exercise gauge, rounding, exact packed heads and delayed score separation on CPU."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import DelayedCapture
from experiments.nan_passreuse96.capture import SharedPassCapture
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_readpath37.run import bound, digest, write_new

from .geometry import METHODS, bind_banks, materialize, plan
from .instrument import collect, score
from .run import BASE, MANIFEST, natural_reach, registered


def bank_fixture(x, width, np):
    rng = np.random.default_rng(99017)
    read = np.linalg.qr(rng.normal(size=(len(x), 8)))[0]
    write = np.linalg.qr(rng.normal(size=(width, 8)))[0].T
    initial = read, write
    t1 = read.copy(), write + .03 * rng.normal(size=write.shape)
    t2 = read.copy(), write + .04 * rng.normal(size=write.shape)
    banks = bind_banks((initial, t1), (initial, t2), np)
    gauged = []
    order = [1, 5, 4, 6, 7, 0, 2, 3]
    gains = np.asarray([2., -.5, 4., -.25, -.5, .5, 2., 8.])
    for v, u in (initial, t1, t2):
        gauged.append((v[:, order] * gains, u[order] / gains[:, None]))
    transformed = bind_banks((gauged[0], gauged[1]), (gauged[0], gauged[2]), np)
    a = materialize(x, banks, "fixture", "A", np)
    b = materialize(-x, banks, "fixture", "B", np)
    c = materialize(x, transformed, "fixture", "A", np)
    for method in METHODS:
        np.testing.assert_allclose(a[method], -b[method], rtol=0, atol=1e-13)
        np.testing.assert_allclose(a[method], c[method], rtol=0, atol=1e-13)
    for label, fit in (("t1", t1), ("t2", t2)):
        np.testing.assert_allclose(a[f"{label}_update"], x @ (fit[0] @ fit[1] - read @ write),
                                   rtol=1e-12, atol=1e-13)
        assert abs(float(a[f"{label}_update"] @ a["orthogonal_sham"])) < 1e-12
    rounded = natural_reach(np.ones(2), {"t1_update": np.array([2**-24, 0.]),
                                         "t2_update": np.array([.125, 0.])}, np)
    assert rounded["t1_update"]["positive_is_noop"] and rounded["t1_update"]["negative_is_noop"]
    assert rounded["t1_update"]["half_vector_cosine"] is None
    assert rounded["t2_update"]["half_vector_norm_over_ideal"] == rounded["t2_update"]["half_vector_cosine"] == 1.
    return banks


def score_fixture(record, labels):
    episode = deepcopy(record)
    row = episode["cases"]["t2_update"]["positive"]["physical_edit"]
    base = episode["baseline"][0]
    current, future = episode["edited_heads"][row]
    current.update(base, max_abs_logit_drift=.25)
    future["selected_logits"] = [0.] * len(labels)
    future["selected_logits"][0] = 2.
    future["selected_logits"][1] = -2.
    future.update(winner=labels[0], peak_logit=2., best_other_logit=1.)
    result = score(episode, labels, labels[0], labels[1])
    assert result["methods"]["t2_update"]["phase2_winners"][0] == labels[0]
    assert result["methods"]["t2_update"]["passes_current_head_gate"] is False
    methods = result["methods"]
    contrast = result["contrasts"]["t2_update_vs_t1_update"]
    assert contrast["passes_current_head_gate"] is False
    assert contrast["phase2_paired_label_half_effect_delta"] == (
        methods["t2_update"]["phase2_paired_label_half_effect"] - methods["t1_update"]["phase2_paired_label_half_effect"])
    assert methods["t2_update"]["phase2_paired_label_half_effect"] != methods["t2_update"]["phase2_unrestricted_half_effect"]


def exercise(snapshot, cfg, np, torch, check):
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(97031)
    model, loop = build_nanbeige_model_and_loop(snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=1, random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 2,
            "hidden_size": 128, "intermediate_size": 72, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    original, hooks = target_digest(loop), hook_count(loop)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    reference = DelayedCapture(loop, head, layer=1, chunk=3)
    shared = SharedPassCapture(loop, head, layer=1, chunk=3)
    left, right = [1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91]
    labels = list(range(16))
    try:
        mixed = reference.capture([left, right])
        reference.cache.reset()
        before = numpy_state(mixed["residual"][1, 0])
        delta = numpy_state(mixed["inputs"][1, 1]) - numpy_state(mixed["inputs"][1, 0])
        banks = bank_fixture(delta, len(before), np)
        directions = materialize(delta, banks, "fixture", "A", np)
        aliases, patches = plan(before, {name: directions["svd"] for name in METHODS}, cfg, np)
        assert aliases["qualified_panels"]["all_arms"] and len(patches) == 2
        inert, missing = plan(before, {name: np.zeros_like(before) for name in METHODS}, cfg, np)
        assert not inert["qualified_panels"]["all_arms"] and not missing
        count = 0
        for ids, peer, world in ((left, right, "A"), (right, left, "B")):
            packed = collect(reference, shared, ids, peer, banks, labels, cfg, "fixture", world, np, check, audit=True)
            assert packed["distinct_rounded_edits"] == 12
            assert packed["packed_phase2_branches"] == 6 and packed["independent_full_prefills"] == 14
            assert packed["qualified_panels"]["all_arms"]
            score_fixture(packed, labels)
            count += len(packed["edited_heads"])
    finally:
        reference.close()
        shared.close()
        head.close()
    assert target_digest(loop) == original and hook_count(loop) == hooks and not torch.cuda.is_initialized()
    return {"physical_factor_gauge_invariant": True, "dense_learned_action_exact_to_fp64_tolerance": True,
            "reciprocal_world_sham_orthogonal": True, "exact_bf16_alias_deduplication": True,
            "zero_direction_refused": True, "native_fixture_worlds": 2, "independently_checked_edited_rows": count,
            "first_pass_gate_and_two_endpoints_exercised": True, "unscaled_bf16_noop_and_exact_reach_verified": True,
            "gpu_initialized": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    args = cli.parse_args()
    seal = registered()
    if json.loads(MANIFEST.read_bytes()) != seal:
        raise ValueError("Current source differs from the registered commissioning")
    base = json.loads(BASE.read_bytes())
    if args.snapshot.name != base["revision"] or "torch" in sys.modules:
        raise ValueError("Require the pinned snapshot before loading Torch")
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work / "hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        if {name: version(name) for name in PACKAGES} != PACKAGES:
            raise ValueError("Pinned numerical packages changed")
        import numpy as np
        import torch

        torch.set_num_threads(1)
        result = exercise(args.snapshot, seal["dose"], np, torch, check)
    write_new(args.work / "verification.json", result | {
        "status": "passed", "source_manifest_sha256": digest(MANIFEST), "pretrained_forwards": 0,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })


if __name__ == "__main__":
    main()
