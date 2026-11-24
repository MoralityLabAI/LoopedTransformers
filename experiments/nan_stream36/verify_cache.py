"""Real small CPU KV and streamed-head smoke for the separate two-row adapter."""
import argparse
import json
import os
from pathlib import Path

from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha


def check_cache(guard):
    import torch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from mechinterp.nanbeige_bpd_gym.tests.test_chunked_cache import Loop

    from .batch_engine import BatchedDepthEngine

    torch.set_num_threads(1)
    torch.manual_seed(36)
    loop = Loop()
    weight = torch.randn(64, 4, dtype=torch.float64)
    head = StreamedVocabularyHead(weight, loop.device, 7)
    engine = BatchedDepthEngine(loop, head, chunk=2)
    results = {}
    try:
        for depth in (1, 2):
            guard()
            inputs = [[1, 3, 7, 9], [2, 5]]
            engine.reset(depth)
            batch_prefill = engine.prefill_logits(inputs)
            assert engine.cache.attention.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
            after_prune = engine.advance_logits([1], [29])
            assert engine.cache.attention.tolist() == [[1, 1, 0, 0, 1]]
            assert engine.cache.physical_length == 5 and engine.cache.semantic_lengths == [3]
            solo = []
            for prompt in inputs:
                engine.reset(depth)
                solo.append(engine.prefill_logits([prompt]))
            engine.reset(depth)
            engine.prefill_logits([inputs[1]])
            solo_next = engine.advance_logits([0], [29])
            prefill_error = float((batch_prefill - torch.cat(solo)).abs().max())
            next_error = float((after_prune - solo_next).abs().max())
            assert prefill_error <= 1e-6 and next_error <= 1e-6
            assert batch_prefill.argmax(dim=-1).tolist() == torch.cat(solo).argmax(dim=-1).tolist()
            assert after_prune.argmax(dim=-1).tolist() == solo_next.argmax(dim=-1).tolist()
            results[str(depth)] = {
                "batch_after_pruning": 1, "padded_canvas_after_pruning": 5, "semantic_length_after_pruning": 3,
                "B2_B1_prefill_max_abs_logit_delta": prefill_error,
                "pruned_B2_B1_next_max_abs_logit_delta": next_error,
                "greedy_argmax_agreement": True,
            }
    finally:
        engine.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return results


def main():
    if not __debug__:
        raise RuntimeError("Fixture requires assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    limits = {
        "minimum_start_available_RAM_bytes": 5 * 2**30,
        "minimum_available_RAM_bytes": 4 * 2**30,
        "maximum_private_bytes": 768 * 2**20,
        "maximum_seconds": 120,
    }
    with lease(limits) as (check, samples):
        os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")
        depths = check_cache(check)
    here = Path(__file__).resolve().parent
    payload = {
        "status": "passed",
        "scope": "actual small CPU DynamicCache and streamed vocabulary code, synthetic weights",
        "per_depth": depths,
        "files_sha256": {name: sha(here / name) for name in ("batch_engine.py", "verify_cache.py", "stream.py")},
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
        "pretrained_forwards": 0,
        "cuda_initialized": False,
    }
    write_new(args.output, payload)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
