"""Replay the physical CPU proof, frozen roster, and independent host scorer."""
from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

from .closed import checked_bytes, collect, lock


def main():
    from experiments.nan_boundrescue109 import host as serial_host
    from experiments.nan_phase35.run import lease
    from experiments.nan_readout34.tokenizer_check import sha
    from experiments.nan_readpath37.run import write_new
    from experiments.nan_rescuegpu111.lane import verified
    from experiments.nan_rescuegroup110 import verify as physical
    from experiments.nan_vocab83.custody import HOST_BUDGET, vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "source-dir", "staged-proof", "host-fixture-dir", "evalmath", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    registration, register_sha = lock()
    if (not __debug__ or args.snapshot.name != registration["source"]["checkpoint"]
            or "torch" in sys.modules):
        parser.error("use assertions and the pinned tokenizer on CPU")

    staged_sha = sha(args.staged_proof)
    staged = json.loads(checked_bytes(args.staged_proof, staged_sha))
    committed = json.loads((physical.HERE / "CPU_VERIFICATION.json").read_bytes())
    evidence = (
        "source_manifest_sha256", "full_root_exact", "native_physical_vectors_exact",
        "full_B2_edited_heads_exact", "independent_exact_k_rescues", "blocked_root_zero_edits",
        "stale_plan_before_staging", "unexpected_rescue_refused", "partial_group_guard_recovery",
        "cuda_initialized", "pretrained_forwards",
    )
    if (staged["status"] != "passed" or any(staged[key] != committed[key] for key in evidence)
            or staged["source_manifest_sha256"] != sha(physical.MANIFEST)):
        raise ValueError("staged physical re-execution did not reproduce the frozen parity receipt")

    runtime = verified()
    host_path = args.host_fixture_dir / "verification.json"
    host_sha = sha(host_path)
    host = json.loads(checked_bytes(host_path, host_sha))
    if (host["runtime_sha256"] != runtime or host["status"] != "passed"
            or host["fixture_closed_roots"] != 1 or host["pretrained_forwards"] != 0
            or host["fixture_signed_macro"]["k_rescue_removed"] != .125
            or any(host[key] is not True for key in (
                "synthetic_refused_as_pretrained", "tampered_grouped_count_refused", "tampered_capacity_refused",
            ))):
        raise ValueError("the grouped host proof did not reproduce its custody gates")
    with lease(HOST_BUDGET) as (check, samples):
        source, pins = serial_host.host_inputs(args.source_dir)
        tok, classes = vocabulary(args.snapshot, check, pins["tokenizer_pins"])
        summary, private = collect(args.host_fixture_dir / "host_fixture", source, classes, tok, runtime, fixture=True)
        observed = [r for r in private if r["status"] == "closed"]
        decision = summary["decision"]
        if (len(observed) != 1 or decision["decision"] != "CPU_FIXTURE_ONLY"
                or observed[0]["intact_advantage"] != -.125 or observed[0]["restored_advantage"] != 0.
                or observed[0]["removed_advantage"] != -.125 or observed[0]["sham_only_removal"] != 0.
                or decision["gate_mean_removed_logits"] != -.125 / 16):
            raise ValueError("the independent donor-logit contrast did not close on the host fixture")

        planner = args.evalmath / "skills/evalmath-mechinterp-power/scripts/preflight.py"
        planner_sha = "38cfcfd2757749cedf9838949c03aac7171c28e7a41d6ec5f11f57715d7778a9"
        checked_bytes(planner, planner_sha)
        module = spec_from_file_location("pinned_evalmath", planner)
        implementation = module_from_spec(module)
        module.loader.exec_module(implementation)
        q16, k16 = implementation.upper_quantile([0] * 16)
        q59, k59 = implementation.upper_quantile([0] * 59)
        if q16 is not None or k16 is not None or q59 != 0 or k59 != 59:
            raise ValueError("the independent sham tolerance boundary changed")

        if "torch" in sys.modules:
            raise ValueError("the host verifier loaded a decoder runtime")
    output = {
        "experiment": "NAN-KRESCUE114", "status": "cpu_verified",
        "registration_sha256": register_sha, "runtime_sha256": runtime,
        "reproduced_staged_proof_sha256": staged_sha,
        "staged_physical": {key: staged[key] for key in evidence},
        "staged_execution_resources": {key: staged[key] for key in (
            "max_sampled_private_bytes", "max_sampled_working_set_bytes", "min_sampled_available_RAM_bytes",
        )},
        "host_fixture_recomputed": summary,
        "host_original_custody_checks": host,
        "host_original_proof_sha256": host_sha,
        "frozen_source": {
            "manifest_sha256": pins["manifest_sha256"], "payload_sha256": pins["payload_sha256"],
            "source_roots": pins["roots"], "prompts": pins["prompts"], "max_position": pins["max_position"],
            "previously_exposed_prompts_excluded": pins["prior_exposed_prompts"],
            "previously_exposed_graphs_excluded": pins["prior_complete_labeled_graphs"],
            "registered_payloads_reproduced_by_digest": True,
        },
        "evalmath_sham_coverage_check": {"n16_rank": k16, "n59_rank": k59, "script_sha256": planner_sha},
        "max_sampled_private_bytes": max(s["private_bytes"] for s in samples),
        "max_sampled_working_set_bytes": max(s["working_set_bytes"] for s in samples),
        "min_sampled_available_RAM_bytes": min(s["available_physical_bytes"] for s in samples),
        "pretrained_forwards": 0, "gpu_initialized": False, "sealed_test_accessed": False,
    }
    args.work.mkdir(parents=True, exist_ok=False)
    write_new(args.work / "verification.json", output)


if __name__ == "__main__":
    main()
