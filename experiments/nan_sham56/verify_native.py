"""Verify signed sham admission and phase-two read capture on a tiny CPU model."""
import argparse
import json
import os
from pathlib import Path

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_dose51.native import RouteCapture
from experiments.nan_gated46.verify import fixture_loop
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import digest, write_new

from .native import continuation
from .run import DOSE, HERE, MANIFEST, REPO, ROUND


LOCK = HERE / "NATIVE_MANIFEST.json"


def seal():
    paths = [HERE / name for name in ("native.py", "verify_native.py", "NATIVE.md", "geometry.py",
                                      "SOURCE_MANIFEST.json", "MINIMAX_MANIFEST.json", "minimax.py", "live_minimax.py")]
    paths += [REPO / name for name in (
        "experiments/nan_dose51/native.py", "experiments/nan_gated46/gated.py",
        "experiments/nan_gated46/verify.py", "experiments/nan_delay39/instrument.py")]
    return {p.relative_to(REPO).as_posix(): digest(p) for p in paths}


def fixture(cfg, round_cfg, registered, check, policy):
    import numpy as np
    import torch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.set_num_threads(1)
    torch.manual_seed(56)
    rng = np.random.default_rng(56)
    loop = fixture_loop(torch, torch.nn)
    weights = (.012 * torch.randn(128, 64)).to(torch.bfloat16)
    weights[0].fill_(.125)
    head = StreamedVocabularyHead(weights, loop.device, 7)
    capture = RouteCapture(loop, head, layer=1, chunk=2)
    ids = [1, 3, 8, 14, 22]
    toy = registered | {"phase1_dose_fraction": .06, "maximum_head_delta": .5}
    mask = np.array([.1, .9, .1, .9, .2, .8, .3, .7], dtype=np.float32)
    try:
        initial_calls = loop.second_calls
        baseline = capture.capture([ids, ids])
        baseline_calls = loop.second_calls - initial_calls
        assert baseline_calls > 0
        y = numpy_state(baseline["native_writes"][1, 1]).astype(np.float32)
        bank = {"V": np.eye(64, 8, dtype=np.float32),
                "U": np.linalg.qr(rng.normal(size=(64, 8)), mode="reduced")[0].T.astype(np.float32),
                "mean_comp_acts": np.zeros((2, 8), np.float32),
                "mean_delta_out": np.stack((y*.2, y*.2)),
                "mean_write_rms": np.full(2, .6, np.float32)}
        model = (bank, None, 1e-6, 1-mask)

        def execute(dose):
            return continuation(capture, ids, (2, 1), "fixture", model, policy, cfg, round_cfg,
                                dose, check, lambda *_: mask[None], np)

        start = loop.second_calls
        okay = execute(toy)
        assert okay["status"] == "closed", okay
        if policy[0] == "ambient_peak_minimax":
            design = okay["cases"]["sham_design"]
            assert design["predicted_peak_after"] < design["predicted_peak_before"]
        assert okay["edited_pass2_runs"] == 4 and loop.second_calls - start == baseline_calls * 5
        assert all(len(out["phase2_bpd_read"]) == 8 for out in okay["outcomes"].values())
        for sign in ("positive", "negative"):
            a = okay["outcomes"][f"contrast/{sign}"]
            b = okay["outcomes"][f"sham/{sign}"]
            assert a["phase2_bpd_read"] != b["phase2_bpd_read"]
        start = loop.second_calls
        reject = execute(toy | {"maximum_head_delta": 0.})
        assert reject["status"] == "head_unqualified", reject
        assert reject["edited_pass2_runs"] == 0 and loop.second_calls - start == baseline_calls
        start = loop.second_calls
        under = execute(toy | {"phase1_dose_fraction": 1e-7})
        assert under["status"] == "physical_unqualified", under
        assert under["edited_pass2_runs"] == 0 and loop.second_calls - start == baseline_calls
        assert not torch.cuda.is_initialized()
    finally:
        capture.close()
        head.close()
    return {"four_edited_continuations": 4, "failed_head_edited_continuations": 0,
            "failed_physical_edited_continuations": 0, "different_signed_future_reads_observed": True,
            "cuda_initialized": False, "new_pretrained_forwards": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(LOCK, seal())
        return
    if args.output is None:
        cli.error("--output required")
    if not __debug__ or json.loads(LOCK.read_bytes()) != seal():
        raise ValueError("Native source freeze changed")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    round_cfg = json.loads((ROUND / "registration.json").read_bytes())
    dose = json.loads((DOSE / "registration.json").read_bytes())
    with lease(cfg | {"maximum_seconds": 120}) as (check, samples):
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        proof = {name: fixture(cfg, round_cfg, dose, check, (name, .25))
                 for name in ("screened_bpd", "ambient_peak_minimax")}
        check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = proof | {"status": "passed", "source_sha256": digest(LOCK), "archive_source_sha256": digest(MANIFEST),
                       "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                       "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples)}
    write_new(args.output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
