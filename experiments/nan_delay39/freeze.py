"""Bind source, numerical and host verification to the DELAY39 commissioning manifest."""
import argparse
import json
from pathlib import Path

from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from .source import HERE, REPO


FILES = (
    "experiments/nan_comp14/parser.py",
    "experiments/nan_comp14/sources.py",
    "experiments/nan_dose38/SOURCE_MANIFEST.json",
    "experiments/nan_dose38/run.py",
    "experiments/nan_gym05/common.py",
    "experiments/nan_phase35/run.py",
    "experiments/nan_readout34/RUNTIME_MANIFEST.json",
    "experiments/nan_readout34/runtime.py",
    "experiments/nan_readout34/runtime_plan.py",
    "experiments/nan_readout34/tokenizer_check.py",
    "experiments/nan_readpath37/registration.json",
    "experiments/nan_readpath37/run.py",
    "experiments/nan_stream36/batch_engine.py",
    "experiments/nan_stream36/device_preflight.py",
    "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
    "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py",
)
LOCAL = (
    ".gitattributes", "__init__.py", "PROTOCOL.md", "SOURCE_CHECK.json", "CPU_VERIFICATION.json",
    "HOST_VERIFICATION.json", "COLD_ADMISSION_VERIFICATION.json", "COMMISSIONING_VERIFICATION.json", "registration.json", "contrasts.py",
    "ENGINEERING.md", "freeze.py", "graph_host.py", "host_gate.py", "instrument.py", "pilot.py", "sink_edges.py", "source.py",
    "verify_cpu.py", "verify_commissioning.py",
)


def main():
    if not __debug__:
        raise RuntimeError("Freeze requires assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    numerical = json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())
    host = json.loads((HERE / "HOST_VERIFICATION.json").read_bytes())
    cold = json.loads((HERE / "COLD_ADMISSION_VERIFICATION.json").read_bytes())
    source = json.loads((HERE / "SOURCE_CHECK.json").read_bytes())
    commissioning = json.loads((HERE / "COMMISSIONING_VERIFICATION.json").read_bytes())
    assert numerical["status"] == host["status"] == cold["status"] == commissioning["status"] == "passed"
    assert numerical["qualified_phase2_sink_edges"] == 8 and numerical["signed_edges_rows"] == 7
    assert numerical["max_full_tensor_reference_logit_difference"] == 0.0
    assert numerical["new_pretrained_forwards"] == 0 and not numerical["cuda_initialized"]
    assert host["synthetic_native_gate_rows"] == source["prompts"] == 48 and source["roots"] == 8
    assert not cold["torch_imported"] and not cold["device_work_created"]
    assert commissioning["closed_focused_fixture_rows"] == 16
    assert commissioning["failed_gate_prevented_all_synthetic_edits"]
    for certificate in (numerical, host, commissioning):
        for name, digest in certificate["source_sha256"].items():
            assert sha(HERE / name) == digest, name
    paths = [HERE / name for name in LOCAL] + [REPO / name for name in FILES]
    value = {"experiment": "NAN-DELAY-39", "stage": "device_pending_after_cpu_verification",
             "files_sha256": {path.relative_to(REPO).as_posix(): sha(path) for path in paths},
             "source_roots": 8, "registered_prompts_per_precision": 48,
             "registered_intervention_worlds_per_precision": 16,
             "maximum_edited_B2_batches_per_precision": 272,
             "pretrained_forwards_at_freeze": 0, "original_protected_test_read": False}
    path = HERE / "SOURCE_MANIFEST.json"
    if args.write:
        write_new(path, value)
    else:
        assert json.loads(path.read_bytes()) == value
    print(json.dumps({"manifest_sha256": sha(path), "bound_files": len(paths),
                      "edit_batches_per_precision": value["maximum_edited_B2_batches_per_precision"],
                      "registered_prompts_per_precision": value["registered_prompts_per_precision"],
                      "stage": value["stage"]}))


if __name__ == "__main__":
    main()
