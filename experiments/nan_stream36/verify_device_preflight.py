"""Exercise the actual prospective B2 gate on tiny CPU DynamicCache and head."""
import argparse
import json
import os
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha


def verify(check):
    import torch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from mechinterp.nanbeige_bpd_gym.tests.test_chunked_cache import Loop

    from .batch_engine import BatchedDepthEngine
    from .device_preflight import compare, preflight

    torch.set_num_threads(1)
    loop = Loop()
    head = StreamedVocabularyHead(torch.randn(64, 4, dtype=torch.float64), loop.device, 7)
    engine = BatchedDepthEngine(loop, head, chunk=3)
    guards = []

    def guard(_began, batch, position):
        assert 1 <= batch <= 2 and position <= 11
        guards.append((batch, position))
        check()

    try:
        result = preflight(engine, [[1, 3, 7, 9], [2, 5, 10, 4]], 11, guard)
    finally:
        engine.close()
        head.close()
    assert set(result) == {"1", "2"}
    assert all(v["tested_max_position"] == 11 and v["checks"]["padded_CAL"]["padded_tokens_at_prefill"] == 1
               for v in result.values())
    assert len(guards) == 38 and not torch.cuda.is_initialized()
    reference = torch.tensor([[1.0, 0.0]])
    failures = [
        (torch.tensor([[1.0, 0.5]]), "logits exceed"),
        (torch.tensor([[0.0, 0.02]]), "greedy choices differ"),
    ]
    for computed, reason in failures:
        expected = reference if reason == "logits exceed" else torch.tensor([[0.02, 0.0]])
        try:
            compare(computed, expected)
        except RuntimeError as error:
            assert reason in str(error)
        else:
            raise AssertionError("Preflight failed to reject a mismatched readout")
    return result, len(guards)


def main():
    if not __debug__:
        raise RuntimeError("Use an assertion-enabled interpreter")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    cfg = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
           "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    with lease(cfg) as (check, samples):
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        depths, guard_calls = verify(check)
    here = Path(__file__).resolve().parent
    proof = {
        "status": "passed", "scope": "actual DynamicCache and streamed-head preflight on synthetic CPU weights",
        "per_depth": depths, "guard_calls": guard_calls, "bad_readout_controls_rejected": 2,
        "pretrained_forwards": 0, "cuda_initialized": False,
        "files_sha256": {name: sha(here / name) for name in ("batch_engine.py", "device_preflight.py",
                                                            "verify_device_preflight.py")},
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
        "min_sampled_available_RAM_bytes": min(v["available_physical_bytes"] for v in samples),
    }
    write_new(args.output, proof)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
