"""Bind the verified READOUT34 runtime, diagnostic roster and sanitized receipts."""
import argparse
import json
from pathlib import Path

from . import source
from .runtime import GPU_PACKAGES, HERE, LIMITS, MANIFEST, REPO, TARGETS, verify_manifest
from .runtime_plan import context_bindings, schedule, write_new
from .tokenizer_check import sha


FILES = [
    "experiments/nan_readout34/.gitattributes",
    "experiments/nan_readout34/SOURCE_MANIFEST.json",
    "experiments/nan_readout34/TOKENIZER_VERIFICATION.json",
    "experiments/nan_readout34/RUNTIME_VERIFICATION.json",
    "experiments/nan_readout34/RUNTIME.md",
    "experiments/nan_readout34/depth_engine.py",
    "experiments/nan_readout34/freeze_runtime.py",
    "experiments/nan_readout34/runtime.py",
    "experiments/nan_readout34/runtime_audit.py",
    "experiments/nan_readout34/runtime_plan.py",
    "experiments/nan_readout34/score.py",
    "experiments/nan_readout34/tokenizer_check.py",
    "experiments/nan_readout34/verify_runtime.py",
    "experiments/nan_bf31/loader.py",
    "experiments/nan_bf31/stage_audit.py",
    "experiments/nan_cache32/inventory.py",
    "experiments/nan_cache32/loader.py",
    "experiments/nan_cache32/resources.py",
    "experiments/nan_cache32/worker.py",
    "experiments/nan_full26/tokenizer_receipt.json",
    "experiments/nan_gym05/common.py",
    "experiments/nan_ram28/loader.py",
    "experiments/nan_ram28/resources.py",
    "experiments/nan_ranksite13/resources.py",
    "experiments/nan_recomp09/common.py",
    "experiments/nan_storyaudit30/common.py",
    "mechinterp/dark_fiber_gyms/arg03_nanbeige.py",
    "mechinterp/nanbeige_bpd_gym/chunked_backend.py",
    "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
    "mechinterp/nanbeige_bpd_gym/lazy_bf16_backend.py",
    "mechinterp/nanbeige_bpd_gym/lazy_bf16_loop.py",
    "mechinterp/nanbeige_bpd_gym/lazy_bf16_storage.py",
    "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py",
]


def main():
    if not __debug__:
        raise RuntimeError("Freeze with assertions enabled")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tokenizer-receipt", required=True, type=Path)
    cli.add_argument("--fixture", required=True, type=Path)
    cli.add_argument("--write", action="store_true")
    args = cli.parse_args()
    receipt = context_bindings(args.tokenizer_receipt)
    fixture = json.loads(args.fixture.read_bytes())
    assert fixture["status"] == "passed"
    assert fixture["software"]["pinned_contexts"] == len(receipt["contexts"]) == 432
    assert fixture["software"]["slots_per_backend"] == 672
    assert fixture["tiny_cpu_cache_and_head"]["cuda_initialized"] is False
    assert fixture["tiny_cpu_cache_and_head"]["pretrained_forwards"] == 0
    assert fixture["tiny_cpu_cache_and_head"]["streamed_full_head_equal"] is True
    assert set(fixture["tiny_cpu_cache_and_head"]["preflight"]) == {"1", "2"}
    for name, expected in fixture["code_sha256"].items():
        assert sha(HERE / name) == expected, name
    proof = HERE / "RUNTIME_VERIFICATION.json"
    if args.write:
        write_new(proof, fixture)
    else:
        assert json.loads(proof.read_bytes()) == fixture
    slots = schedule(source.build())
    seal = {
        "experiment": "NAN-READOUT-34",
        "stage": "runtime_software_verified_gpu_pending",
        "slots_per_backend": len(slots),
        "one_pass_slots_per_backend": 240,
        "two_pass_slots_per_backend": 432,
        "max_context_plus_generation": receipt["max_prompt_tokens"] + receipt["generated_token_cap"],
        "schedule_sha256": source.digest([[slot.ordinal, slot.passes, slot.source["id"]] for slot in slots]),
        "limits": LIMITS,
        "expected_targets": TARGETS,
        "packages": GPU_PACKAGES,
        "files_sha256": {name: sha(REPO / name) for name in sorted(FILES)},
        "pretrained_forwards_at_freeze": 0,
        "formal_comparisons": 0,
        "original_ARG03_test_accessed": False,
    }
    if args.write:
        write_new(MANIFEST, seal)
    assert verify_manifest() == seal
    print(json.dumps({"runtime_manifest_sha256": sha(MANIFEST), "slots_per_backend": len(slots),
                      "schedule_sha256": seal["schedule_sha256"], "max_position": seal["max_context_plus_generation"],
                      "bound_files": len(FILES), "status": seal["stage"]}))


if __name__ == "__main__":
    main()
