"""CPU-only source and outcome audit of the guarded two-row model lane."""
import argparse
import json
from pathlib import Path
import sys
from statistics import median

from experiments.nan_gym05.common import prompt_ids
from experiments.nan_phase35.run import lease
from experiments.nan_readout34 import oracle, runtime, source
from experiments.nan_readout34.runtime_audit import summarize
from experiments.nan_readout34.runtime_plan import context_bindings, schedule, write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha, tokenizer

from .pipeline import collect, fixed_jobs
from .run import ARTIFACT, MANIFEST, MAXIMUM_POSITION, STREAM36, verify


def device_summary(work, state):
    if state == "no_integrity_terminal":
        return {}
    terminal = json.loads((work / "terminal.json").read_bytes())
    log = work / "resource_samples.jsonl"
    if sha(log) != terminal["resource_log_sha256"]:
        raise ValueError("Resource samples changed after model closure")
    lines = log.read_bytes().splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("Missing or incomplete GPU resource samples")
    samples = [json.loads(line) for line in lines]
    host = [item["host"] for item in samples]
    gpu = [item["process_device"] for item in samples]
    devices = [item["device"] for item in samples if "utilization.gpu" in item["device"]]
    answer = {
        "process_samples": len(samples), "device_samples": len(devices),
        "max_sampled_private_bytes": max(item["private_bytes"] for item in host),
        "max_sampled_working_set_bytes": max(item["working_set_bytes"] for item in host),
        "min_sampled_available_RAM_bytes": min(item["available_physical_bytes"] for item in host),
        "max_sampled_dedicated_GPU_bytes": max(item["dedicated_bytes"] for item in gpu),
        "max_sampled_shared_GPU_bytes": max(item["shared_bytes"] for item in gpu),
        "peak_torch_allocated_bytes": terminal["max_torch_allocated"],
        "peak_torch_reserved_bytes": terminal["max_torch_reserved"],
    }
    if devices:
        answer.update(observed_utilization_median_pct=median(item["utilization.gpu"] for item in devices),
                      observed_max_temperature_C=max(item["temperature.gpu"] for item in devices),
                      observed_SM_clock_range_MHz=[min(item["clocks.current.sm"] for item in devices),
                                                  max(item["clocks.current.sm"] for item in devices)])
    if terminal["preflight_passed"]:
        proof = json.loads((work / "preflight.json").read_bytes())
        answer["matched_CAL_timing"] = proof["matched_CAL_timing"]
    return answer


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    verify()
    limits = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
              "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 600}
    with runtime.exclusive_lane():
        runtime.predecessors_closed()
        with lease(limits) as (check, _):
            receipt = context_bindings(args.tokenizer_receipt)
            snapshot = args.snapshot.resolve()
            if snapshot.name != source.CONFIG["checkpoint_revision"]:
                raise ValueError("Scorer tokenizer revision mismatch")
            tok, pins = tokenizer(snapshot)
            if pins != receipt["tokenizer_files_sha256"]:
                raise ValueError("Scorer tokenizer files differ from the frozen receipt")
            bindings = receipt["contexts"]
            rows = source.build()
            for row in rows:
                check()
                ids = prompt_ids(tok, snapshot, row["prompt"])
                if bindings[row["id"]] != {"tokens": len(ids), "ids_sha256": id_hash(ids)}:
                    raise ValueError("Independent source token replay differs")
                if row["arm"] == "compute" and row["schema"]["readout"] == "final":
                    if oracle.replay_text(row["prompt"]) != row["label"]:
                        raise ValueError("Independent source oracle differs")
            if tok.tok.eos_token_id != receipt["eos_token_id"]:
                raise ValueError("Scorer prompt setup did not bind the registered EOS")
            slots = schedule(rows)
            jobs = fixed_jobs(slots, bindings, json.loads((STREAM36 / "SCHEDULING_VERIFICATION.json").read_bytes()))
            data, performance, states, device = {}, {}, {}, {}
            for backend, expected in runtime.TARGETS.items():
                data[backend], performance[backend], states[backend] = collect(
                    ARTIFACT / backend, jobs, bindings, tok, backend=backend, manifest_sha=sha(MANIFEST),
                    expected_target=expected, maximum_position=MAXIMUM_POSITION)
                device[backend] = device_summary(ARTIFACT / backend, states[backend])
                check()
            outcome = summarize(slots, data)
            for row in outcome["verified_compute"]:
                row["sum_response_latency_seconds"] = row.pop("wall_seconds")
            if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
                raise RuntimeError("Host scorer imported Torch")
            write_new(args.output, {
                "experiment": "NAN-STREAM-41", "runtime_manifest_sha256": sha(MANIFEST),
                "source_roots": {"CAL": 3, "DEV": 9}, "formal_comparisons": 0,
                "registered_slots_per_backend": len(slots), "registered_batches_per_backend": len(jobs),
                "verified_slots": {backend: len(records) for backend, records in data.items()},
                "backend_states": states, "physical_batch_performance_by_depth": performance,
                "device_telemetry": device,
                "sealed_test_accessed": False, **outcome,
            })


if __name__ == "__main__":
    main()
