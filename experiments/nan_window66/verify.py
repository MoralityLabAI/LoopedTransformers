"""Commission the earlier carry window through the real signed four-arm CPU path."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
import sys
from pathlib import Path

from experiments.nan_dose51.native import RouteCapture
from experiments.nan_phase35.run import lease
from experiments.nan_premlp64.verify import BUDGET, CONFIGURATION, PACKAGES, check_reference, hook_count
from experiments.nan_readpath37.run import bound, digest, write_new
from experiments.nan_route58.capture import fingerprint
from experiments.nan_route58.custody import settings

from .capture import WindowCapture
from .telemetry import KEYS, run_window_continuation, score_profile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "experiments/nan_readpath37/registration.json"
SEED = 64017


def seal():
    local = [HERE / f for f in (".gitattributes", "__init__.py", "capture.py",
                               "telemetry.py", "verify.py", "PROTOCOL.md")]
    parents = [PARENT] + [REPO / f for f in (
        "experiments/nan_premlp64/capture.py", "experiments/nan_premlp64/verify.py",
        "experiments/nan_premlp64/account.py", "experiments/nan_recomp09/common.py",
        "experiments/nan_sham56/native.py", "experiments/nan_sham56/live_minimax.py",
        "experiments/nan_sham56/minimax.py", "experiments/nan_round54/geometry.py",
        "experiments/nan_dose51/native.py", "experiments/nan_dose51/route.py",
        "experiments/nan_gated46/gated.py", "experiments/nan_delay39/instrument.py",
        "experiments/nan_route58/RUNTIME_MANIFEST.json", "experiments/nan_route58/capture.py",
        "experiments/nan_route58/host.py", "experiments/nan_phase35/run.py",
        "mechinterp/dark_fiber_gyms/arg03_nanbeige.py", "mechinterp/nanbeige_bpd_gym/chunked_cache.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in local + parents}


def stale_rejected(operation, text):
    try:
        operation()
    except (ValueError, RuntimeError) as exc:
        assert text in str(exc)
    else:
        raise AssertionError(f"Expected refusal: {text}")


def exercise(snapshot, check, np, torch):
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import build_nanbeige_model_and_loop
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.manual_seed(SEED)
    layers = (1, 2, 3, 4)
    model, loop = build_nanbeige_model_and_loop(
        snapshot, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        n_gpu_layers=2, random_init=True,
        config_overrides={"vocab_size": 97, "num_hidden_layers": 5, "hidden_size": 64,
                          "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2,
                          "head_dim": 16, "kv_channels": 16})
    model.requires_grad_(False)
    loop.stage_layer.requires_grad_(False)
    head = StreamedVocabularyHead(model.lm_head.weight, loop.device, 13)
    prior_hooks = hook_count(loop)
    ids = [1, 4, 32, 5, 29, 6, 42]
    unequal = [ids, [8, 39, 77]]
    duplicate = [ids, ids]

    original = RouteCapture(loop, head, layer=4, chunk=3)
    try:
        # Give the tiny fixture a stable phase-one winner without changing the native decoder.
        pre = original.capture(duplicate)["residual"][1, 1]
        with torch.no_grad():
            model.lm_head.weight.mul_(.125)
            model.lm_head.weight[0].copy_(pre.float().div(pre.float().norm()).to(torch.bfloat16))
        fingerprint_before = target_digest(loop)
        pair, baseline = [original.capture(p) for p in (unequal, duplicate)]
    finally:
        original.close()
    assert hook_count(loop) == prior_hooks

    capture = WindowCapture(loop, head, layers=layers, chunk=3)
    try:
        check_reference(pair, capture.capture(unequal), torch)
        check_reference(baseline, capture.capture(duplicate), torch)
        assert capture.cache.stage_materializations == 6
        winner = int(baseline["logits"][1, 1].argmax())
        labels = ((winner + 1) % head.weight.shape[0], winner)
        rng = np.random.default_rng(SEED)
        y = baseline["native_writes"][1, 1].float().numpy()
        bank = {"V": np.eye(96, 8, dtype=np.float32),
                "U": np.linalg.qr(rng.normal(size=(64, 8)), mode="reduced")[0].T.astype(np.float32),
                "mean_comp_acts": np.zeros((2, 8), dtype=np.float32),
                "mean_delta_out": np.stack((y*.2, y*.2)), "mean_write_rms": np.full(2, .6, dtype=np.float32)}
        masks = lambda *_: np.array([[.1, .9, .1, .9, .2, .8, .3, .7]], dtype=np.float32)
        cfg, sham, round_cfg, dose = settings()
        toy = dose | {"phase1_dose_fraction": .12, "maximum_head_delta": .5,
                      "maximum_margin_delta": .25, "block_rows": 17}
        fixture_bank = (bank, None, 1e-6, 1 - masks()[0])
        expected = fingerprint(baseline, 1)
        result = run_window_continuation(capture, expected, ids, labels, "window66",
                                         fixture_bank, tuple(cfg["sham_policy"]),
                                         sham, round_cfg, toy, check, masks, np)
        if result["status"] != "closed":
            raise RuntimeError(f"Four-arm tiny experiment refused: {result['status']} {result.get('reason', '')}")
        scored = score_profile(result, winner, labels[0], round_cfg, toy, layers)
        assert scored is not None and tuple(result["delayed_window"]) == KEYS
        assert len(scored["delayed_window_effects"]) == len(layers)
        final_error = max(a[str(layers[-1])]["max_selected_logit_error"] for a in result["delayed_window"].values())
        for bad, message in (("layer_omission", "registered layers"), ("arithmetic", "do not close"),
                             ("native_binding", "does not bind")):
            broken = deepcopy(result)
            cells = broken["delayed_window"][KEYS[0]]
            if bad == "layer_omission":
                del cells[str(layers[0])]
            elif bad == "arithmetic":
                cells[str(layers[0])]["finite"]["mlp_write"] += 1.
            else:
                cells[str(layers[-1])]["actual_selected_head_swing"] += 1.
            stale_rejected(lambda: score_profile(broken, winner, labels[0], round_cfg, toy, layers), message)
        assert target_digest(loop) == fingerprint_before
        return scored, {
            "resident_and_reused_stage": True, "window_layers": list(layers),
            "signed_pass2_runs": result["edited_pass2_runs"], "checked_signed_site_records": len(KEYS)*len(layers),
            "matched_shapes": [list(map(len, p)) for p in (unequal, duplicate)],
            "final_layer_max_selected_logit_error": final_error, "malformed_receipts_refused": 3,
            "finite_coalitions_per_site_record": 32,
        }
    finally:
        capture.close()
        head.close()
        assert hook_count(loop) == prior_hooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Enable Python assertions")
    start_seal = seal()
    base = json.loads(PARENT.read_bytes())
    assert args.snapshot.name == base["revision"]
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    with lease(BUDGET) as (check, samples):
        args.work.mkdir(parents=True, exist_ok=False)
        os.environ.update(USE_TORCH="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str((args.work / "hf_home").resolve()),
                          HF_MODULES_CACHE=str((args.work / "hf_modules").resolve()))
        os.environ.pop("TRANSFORMERS_CACHE", None)
        for name in ("config.json", "modeling_nanbeige.py"):
            bound(args.snapshot / name, base["snapshot_pins"][name])
        bound(args.snapshot / "configuration_nanbeige.py", CONFIGURATION)
        assert {p: version(p) for p in PACKAGES} == PACKAGES
        import numpy as np
        import torch

        torch.set_num_threads(1)
        scores, metrics = exercise(args.snapshot, check, np, torch)
        assert not torch.cuda.is_initialized()
    assert seal() == start_seal
    write_new(args.work / "verification.json", {
        "experiment": "NAN-WINDOW-66", "status": "passed", "source_sha256": start_seal,
        "fixture": metrics, "fixture_signed_effects": scores, "random_seed": SEED,
        "pretrained_forwards": 0, "cuda_initialized": False, "formal_comparisons": 0,
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
    })
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
