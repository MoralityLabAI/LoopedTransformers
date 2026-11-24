"""Verify shared and packed phase reuse against the actual staged BF16 CPU decoder."""

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from experiments.nan_delay39.instrument import DelayedCapture, clean_row_closes, prepare
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, hook_count
from experiments.nan_readpath37.run import bound, digest, write_new
from experiments.nan_nativerepair91.source import PIN

from .capture import SharedPassCapture


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "SOURCE_MANIFEST.json"


def seal():
    parents = ("experiments/nan_delay39/registration.json", "experiments/nan_delay39/SOURCE_MANIFEST.json",
               "experiments/nan_delay39/instrument.py", "experiments/nan_delay39/source.py",
               "experiments/nan_nativerepair91/SOURCE_MANIFEST.json", "experiments/nan_nativerepair91/SOURCE_CHECK.json",
               "experiments/nan_nativerepair91/source.py", "experiments/nan_tangent87/source.py",
               "experiments/nan_readpath37/registration.json", "experiments/nan_phase35/run.py",
               "experiments/nan_premlp64/verify.py", "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
               "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py",
               "mechinterp/dark_fiber_gyms/arg03_nanbeige.py")
    sources = [HERE / n for n in (".gitattributes", "__init__.py", "PROTOCOL.md", "capture.py", "verify.py", "source.py")]
    return {"experiment": "NAN-PASSREUSE-96", "files_sha256": {
        p.relative_to(REPO).as_posix(): digest(p) for p in (*sources, *(REPO / n for n in parents))}}


def patches(native, cfg, np):
    rng = np.random.default_rng(96028)
    output = []
    for _ in range(128):
        direction = rng.normal(size=len(native))
        pair = [prepare(native, s * direction, cfg["phase1_dose_fraction"], cfg["dose_relative_tolerance"],
                        cfg["minimum_direction_cosine"], np) for s in (1, -1)]
        if not all(p.eligible for p in pair):
            continue
        if abs(pair[0].realized_l2 - pair[1].realized_l2) > cfg["max_contrast_dose_gap_fraction"] * pair[0].requested_l2:
            continue
        output.extend(pair)
        if len(output) >= 18:
            break
    if len(output) != 18 or len({p.after.tobytes() for p in output[:17]}) != 17:
        raise RuntimeError("The staged CPU fixture did not find its 17 distinct physical edits")
    return output[:17]


def rows_equal(a, i, b, j, torch):
    if ((a["physical_length"], a["semantic_lengths"]) != (b["physical_length"], b["semantic_lengths"])
            or not torch.equal(a["logits"][:, i], b["logits"][:, j])
            or any(not torch.equal(a[site][phase, i], b[site][phase, j])
                   for site in ("inputs", "residual") for phase in (1, 2))):
        raise RuntimeError("Shared phase branch differs from the exact reference trajectory")


def native_fixture(snapshot, check, np, torch):
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(96131)
    model, loop = build_nanbeige_model_and_loop(snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=1, random_init=True, config_overrides={"vocab_size": 97, "num_hidden_layers": 2,
            "hidden_size": 128, "intermediate_size": 72, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    parameter, hooks = target_digest(loop), hook_count(loop)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    cfg = json.loads((REPO / "experiments/nan_delay39/registration.json").read_bytes())
    original = DelayedCapture(loop, head, layer=1, chunk=3)
    shared = SharedPassCapture(loop, head, layer=1, chunk=3)
    sizes = []
    clean_heads = clean_edit = packed_edit = 0
    try:
        for seed in ([1, 4, 32, 5, 29, 6, 42], [8, 39, 3, 4, 65, 77, 91, 2, 3, 7, 43]):
            check()
            ids = [seed, seed]
            baseline = original.capture(ids)
            edits = patches(baseline["residual"][1, 1].float().numpy().astype(np.float64), cfg, np)
            references = [original.capture(ids, edit) for edit in edits]
            independent_self = original.capture(ids)
            clean_row_closes(baseline, independent_self, expect_edit=False)
            before = tuple(shared.cache.caches)
            shared.prime(ids)
            assert len(shared.cache.caches) == 2 and shared.cache.cache_bytes() == 0
            assert all(a is not b for a in shared.cache.caches for b in before)
            snapshot_raw = shared.cache.first["raw"].clone()
            sizes.append(snapshot_raw.numel()*snapshot_raw.element_size())
            clean = shared.branch(ids)
            for i in (0, 1):
                rows_equal(clean, i, independent_self, i, torch)
                clean_heads += 2
            for edit, reference in zip(edits, references, strict=True):
                check()
                capture = shared.branch(ids, (None, edit))
                clean_row_closes(clean, capture, expect_edit=True)
                for i in (0, 1):
                    rows_equal(capture, i, reference, i, torch)
                clean_edit += 1
            for i in range(0, len(edits), 2):
                check()
                pair = (edits[i], edits[i+1] if i+1 < len(edits) else None)
                capture = shared.branch(ids, pair)
                assert capture["applied"] == sum(x is not None for x in pair)
                rows_equal(capture, 0, references[i], 1, torch)
                second = references[i+1] if pair[1] is not None else baseline
                rows_equal(capture, 1, second, 1, torch)
                packed_edit += sum(x is not None for x in pair)
            stale = replace(edits[0], before=edits[0].before + 1.)
            for tokens, edit, text in ((ids, stale, "stale"), ([seed, seed[:-1]], edits[0], "exact primed")):
                try:
                    shared.branch(tokens, (None, edit))
                except ValueError as exc:
                    assert text in str(exc)
                else:
                    raise AssertionError("The shared branch admitted stale tokens or bytes")
            for i in (0, 1):
                rows_equal(shared.branch(ids), i, baseline, i, torch)
            assert torch.equal(snapshot_raw, shared.cache.first["raw"]) and shared.cache.cache_bytes() == 0
            # One passing prime, one clean call, 17 companion calls, nine packed and two rechecks.
            assert shared.cache.stage_materializations == 30
    finally:
        shared.close()
        original.close()
        head.close()
    assert target_digest(loop) == parameter and hook_count(loop) == hooks
    assert not torch.cuda.is_initialized()
    return {"native_cpu_vocabulary": 97, "clean_head_rows_exact": clean_heads,
            "clean_companion_branches_exact": clean_edit, "packed_edited_rows_exact": packed_edit,
            "independent_self_retained": True, "fresh_branch_cache": True, "shared_canvas_reused_exactly": True,
            "max_fixture_raw_canvas_bytes": max(sizes), "stale_edit_and_rebound_tokens_refused": True,
            "random_weights": True, "pretrained_forwards": 0, "cuda_initialized": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    frozen = seal()
    if args.freeze:
        write_new(MANIFEST, frozen)
        return
    if (not __debug__ or not args.work or not args.snapshot or args.snapshot.name != PIN
            or json.loads(MANIFEST.read_bytes()) != frozen or "torch" in sys.modules):
        cli.error("Require frozen sources, pinned snapshot and a fresh work folder")
    base = json.loads((REPO / "experiments/nan_readpath37/registration.json").read_bytes())
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HOME=str((args.work / "hf_home").resolve()), HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        if {name: version(name) for name in PACKAGES} != PACKAGES:
            raise RuntimeError("Numerical fixture version changed")
        import numpy as np
        import torch

        torch.set_num_threads(1)
        proof = native_fixture(args.snapshot, check, np, torch)
    assert seal() == frozen
    receipt = proof | {"experiment": frozen["experiment"], "status": "passed", "source_manifest_sha256": digest(MANIFEST),
                       "sealed_test_accessed": False,
                       "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
                       "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples),
                       "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples)}
    write_new(args.work / "verification.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
