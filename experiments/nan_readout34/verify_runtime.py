"""Run the fixed-schedule software contracts and tiny real cache/head checks."""
import argparse
from collections import Counter
import ctypes
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from experiments.nan_ram28.resources import memory
from experiments.nan_ranksite13.resources import below_normal

from . import source, worlds
from .runtime_audit import audit_episode, collect, paired, summarize
from .runtime_plan import CAP, context_bindings, execute, schedule, write_new
from .tokenizer_check import id_hash, sha


HERE = Path(__file__).resolve().parent
START = 5 * 2**30
RESERVE = 4 * 2**30
PRIVATE = 768 * 2**20
MAX_SECONDS = 120


def synthetic_checks(receipt_path, scratch, check):
    receipt = context_bindings(receipt_path)
    rows = source.build()
    slots = schedule(rows)
    assert len(slots) == 672 and {slot.passes for slot in slots} == {1, 2}
    assert sum(slot.source["split"] == "DEV" for slot in slots) == 504
    assert Counter((slot.passes, slot.source["arm"]) for slot in slots) == {
        (1, "compute"): 96, (1, "copy"): 96, (1, "erased"): 48,
        (2, "compute"): 192, (2, "copy"): 192, (2, "erased"): 48,
    }
    assert len({slot.file for slot in slots}) == 672
    for i, slot in enumerate(slots):
        if slot.passes == 1:
            assert slots[i + 1].passes == 2 and slots[i + 1].source == slot.source

    class SyntheticEngine:
        vocabulary_size = 1024

        def reset(self, passes):
            self.passes = passes
            self.answer = iter(())

        def first(self, prompt):
            assert self.passes in (1, 2)
            self.answer = iter(script[prompt[0]])
            return next(self.answer)

        def next(self, token):
            return next(self.answer)

    # ASCII fixture tokens exercise actual stop/capture/audit paths, independent of the pinned tokenizer.
    ids = {row["id"]: [100 + i] for i, row in enumerate(rows)}
    bindings = {key: {"tokens": 1, "ids_sha256": id_hash(value)} for key, value in ids.items()}
    script = {ids[row["id"]][0]: list(worlds.canonical(row["label"], row["schema"]["readout"]).encode("ascii"))
              for row in rows}
    selected = slots[:8]
    script[ids[selected[2].source["id"]][0]] = [0]
    script[ids[selected[4].source["id"]][0]] = [900]
    script[ids[selected[6].source["id"]][0]] = [65] * CAP

    def decode(tokens):
        return "".join("x" * 513 if item == 900 else chr(item) for item in tokens)

    scratch.mkdir(parents=True, exist_ok=False)
    seen = {}

    def record(slot, answer):
        path = scratch / slot.file
        write_new(path, answer)
        with (scratch / "closed.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"file": slot.file, "sha256": sha(path)}) + "\n")
        seen[slot.ordinal] = answer

    execute(SyntheticEngine(), selected, bindings, tokens=lambda row: ids[row["id"]],
            decode=decode, eos_token=0, budget=lambda _: check(), record=record)
    assert len(seen) == len(selected)
    assert set(answer["stop_reason"] for answer in seen.values()) == {
        "complete_packet", "eos", "character_cap", "token_cap",
    }
    terminal = {
        "status": "partial", "backend": "nf4", "runtime_manifest_sha256": "fixture",
        "target_before": "fixture_target", "target_after": "fixture_target",
        "quant_before": {}, "quant_after": {}, "checkpoint_verified_after": True, "preflight_passed": True,
        "vocabulary_size": 1024, "closed_slots": len(seen), "closed_journal_sha256": sha(scratch / "closed.jsonl"),
    }
    write_new(scratch / "terminal.json", terminal)
    tok = SimpleNamespace(tok=SimpleNamespace(decode=lambda ids, **_: decode(ids), eos_token_id=0))
    kwargs = {"bindings": bindings, "tok": tok, "manifest_sha": "fixture",
              "expected_target": "fixture_target", "backend": "nf4"}
    verified = collect(scratch, slots, **kwargs)
    assert len(verified) == len(seen)
    report = summarize(slots, {"nf4": verified, "bf16": {}})
    assert sum(row["worlds"]["registered"] for row in report["outcomes"]
               if row["metric"] == "joint_exact") == 1344
    assert all(row["unknown"] == row["registered_pairs"] for row in report["nf4_to_bf16"])
    assert paired([(0, 1), (1, 0), (1, 1), (0, 0), (None, 1)]) == {
        "registered_pairs": 5, "repair": 1, "regression": 1,
        "both_correct": 1, "both_incorrect": 1, "unknown": 1,
    }

    changed = dict(seen[0], generated_ids=seen[0]["generated_ids"] + [0])
    try:
        audit_episode(changed, slots[0], bindings[slots[0].source["id"]], decode, 0, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid first-stop history was accepted")
    with (scratch / selected[0].file).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    try:
        collect(scratch, slots, **kwargs)
    except ValueError as error:
        assert "differs from its journal" in str(error)
    else:
        raise AssertionError("Changed immutable episode was accepted")
    return {"pinned_contexts": len(receipt["contexts"]), "slots_per_backend": len(slots),
            "single_pass_slots_per_backend": 240, "two_pass_slots_per_backend": 432,
            "synthetic_verified_episodes": len(verified), "synthetic_stop_reasons": 4,
            "rejected_changed_episodes": 1, "rejected_nonfirst_stop": 1}


def tiny_model_checks(check):
    import torch
    import torch.nn.functional as F
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from .depth_engine import DepthEngine

    torch.set_num_threads(1)
    torch.manual_seed(64034)

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.self_attn = SimpleNamespace(layer_idx=index, o_proj=torch.nn.Identity())
            self.mlp = SimpleNamespace(down_proj=torch.nn.Identity())

        def forward(self, x, *, past_key_value, cache_position, **_kwargs):
            k = x.unsqueeze(1).clone()
            _, v = past_key_value.update(k, k.clone(), self.self_attn.layer_idx)
            context = torch.stack([v[:, 0, :int(p) + 1].float().mean(1) for p in cache_position], dim=1)
            return ((x.float() * 0.75 + context * 0.25).to(x.dtype),)

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(23, 8, dtype=torch.bfloat16)
            self.layers = torch.nn.ModuleList([Layer(0), Layer(1)])
            self.norm = torch.nn.LayerNorm(8, dtype=torch.bfloat16)

        def _update_causal_mask(self, *unused):
            return None

    base = Base().eval().requires_grad_(False)
    head_weights = torch.randn(23, 8, dtype=torch.bfloat16)
    loop = SimpleNamespace(model=base, base=base, layers=base.layers, stage_layer=None,
                           n=2, n_gpu_layers=2, device=torch.device("cpu"))
    head = StreamedVocabularyHead(head_weights, loop.device, 5)
    engine = DepthEngine(loop, head, chunk=4)
    try:
        report = engine.preflight([1, 4, 0, 3, 5], 12, check)
        for depth in (1, 2):
            engine.reset(depth)
            result = engine.cache.prefill([[1, 4, 0, 3, 5]], capture=False)
            hidden = result["boundaries"][-1]
            assert torch.equal(head(hidden), F.linear(hidden, head_weights).float())
            assert len(engine.cache.caches) == depth
            assert sum(engine.cache.caches[i].get_seq_length(0) for i in range(depth)) == 5 * depth
    finally:
        engine.close()
        head.close()
    assert not torch.cuda.is_initialized()
    return {"torch": version("torch"), "transformers": version("transformers"),
            "device": "cpu", "synthetic_layers": 2, "head_rows": 23,
            "streamed_full_head_equal": True, "preflight": report,
            "pretrained_forwards": 0, "cuda_initialized": False}


def main():
    if not __debug__:
        raise RuntimeError("Run with assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--attempt", type=Path, required=True)
    args = cli.parse_args()
    below_normal()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                      USE_TORCH="1", USE_TF="0", USE_FLAX="0")
    if memory()["available_physical_bytes"] < START:
        raise RuntimeError("Tiny-model fixture requires 5 GiB of available RAM")
    stopped = threading.Event()
    start = time.monotonic()
    samples = []

    def check():
        sample = memory()
        samples.append(sample)
        if (sample["available_physical_bytes"] < RESERVE or sample["private_bytes"] > PRIVATE
                or time.monotonic() - start > MAX_SECONDS):
            raise RuntimeError(f"Software verification resource limit: {sample}")

    def watch():
        while not stopped.wait(1):
            try:
                check()
            except Exception as error:
                print(error, file=sys.stderr, flush=True)
                os._exit(3)

    args.attempt.parent.mkdir(parents=True, exist_ok=True)
    args.attempt.mkdir(exist_ok=False)
    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()
    try:
        check()
        software = synthetic_checks(args.tokenizer_receipt, args.attempt / "synthetic", check)
        numerical = tiny_model_checks(check)
        check()
    finally:
        stopped.set()
        watchdog.join()
    paths = ["depth_engine.py", "runtime.py", "runtime_plan.py", "runtime_audit.py", "verify_runtime.py"]
    verification = {
        "status": "passed",
        "code_sha256": {name: sha(HERE / name) for name in paths},
        "software": software,
        "tiny_cpu_cache_and_head": numerical,
        "resources": {"max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
                      "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
                      "seconds": round(time.monotonic() - start, 3), "sample_count": len(samples)},
    }
    write_new(args.attempt / "verification.json", verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
