"""Exercise physical signed routes through the native CPU phase-2 admission boundary."""
import argparse
import json
import os
from pathlib import Path

from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import clean_row_closes
from experiments.nan_gated46.verify import fixture_loop
from experiments.nan_phase35.run import lease
from experiments.nan_readpath37.run import digest, write_new

from .native import RouteCapture, gated_continuations
from .route import ARMS, SIGNS, directions
from .run import HERE, MANIFEST, seal


def native_fixture(cfg, check):
    import numpy as np
    import torch
    from torch import nn
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

    torch.set_num_threads(1)
    torch.manual_seed(51)
    random = np.random.default_rng(51)
    loop = fixture_loop(torch, nn)
    weight = (.015 * torch.randn(32, 64)).to(torch.bfloat16)
    weight[0].fill_(.125)
    head = StreamedVocabularyHead(weight, loop.device, 7)
    engine = RouteCapture(loop, head, layer=1, chunk=2)
    ids = [1, 3, 8, 14, 22]
    toy = cfg | {"phase1_dose_fraction": .05, "maximum_head_delta": .5}
    try:
        check()
        baseline = engine.capture([ids, ids])
        x = numpy_state(baseline["inputs"][1, 1]).astype(np.float32)
        y = numpy_state(baseline["native_writes"][1, 1]).astype(np.float32)
        bank = {
            "V": np.eye(64, 8, dtype=np.float32),
            "U": np.linalg.qr(random.normal(size=(64, 8)), mode="reduced")[0].T.astype(np.float32),
            "mean_comp_acts": np.zeros((2, 8), np.float32),
            "mean_delta_out": np.stack((y * .2, y * .2)),
            "mean_write_rms": np.full(2, .6, np.float32),
        }
        mask = np.array([.1, .9, .1, .9, .2, .8, .3, .7], dtype=np.float32)
        effects, error = directions(x, y, mask, 1 - mask, bank, 1e-6, np)
        assert error < 1e-6 and not np.array_equal(*effects.values())
        result = gated_continuations(engine, ids, baseline, effects, (2, 1), toy, np)
        assert result["status"] == "closed" and result["edited_pass2_runs"] == 4
        for sign in SIGNS:
            a, b = (result["outcomes"][f"{arm}/{sign}"] for arm in ARMS)
            assert a is not b
            assert not torch.equal(a["inputs"][2, 1], b["inputs"][2, 1])
        same = dict.fromkeys(ARMS, effects["learned"])
        dedup = gated_continuations(engine, ids, baseline, same, (2, 1), toy, np)
        assert dedup["edited_pass2_runs"] == 2
        assert all(dedup["outcomes"][f"learned/{sign}"] is dedup["outcomes"][f"static/{sign}"] for sign in SIGNS)
        start = loop.second_calls
        refused = gated_continuations(engine, ids, baseline, effects, (2, 1), toy | {"maximum_head_delta": 0.}, np)
        assert refused["status"] == "head_unqualified" and refused["edited_pass2_runs"] == 0
        bad_dose = effects | {"static": effects["static"] * 0}
        under = gated_continuations(engine, ids, baseline, bad_dose, (2, 1), toy, np)
        assert under["status"] == "dose_unqualified" and under["edited_pass2_runs"] == 0
        assert loop.second_calls == start
        clean_row_closes(baseline, engine.capture([ids, ids]), expect_edit=False)
    finally:
        engine.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return {"distinct_bf16_edit_pass2_calls": 4, "identical_route_pass2_calls": 2,
            "failed_gate_pass2_calls": 0, "failed_dose_pass2_calls": 0, "native_recovery_exact": True,
            "cuda_initialized": False, "new_pretrained_forwards": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != seal():
        raise ValueError("DOSE51 source freeze changed")
    cfg = json.loads((HERE / "registration.json").read_bytes())
    with lease(cfg | {"maximum_seconds": 120}) as (check, samples):
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        result = native_fixture(cfg, check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.update(status="passed", source_manifest_sha256=digest(MANIFEST),
                  max_sampled_private_bytes=max(v["private_bytes"] for v in samples),
                  min_sampled_available_RAM_bytes=min(v["available_physical_bytes"] for v in samples))
    write_new(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
