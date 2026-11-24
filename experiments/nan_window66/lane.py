"""Run the registered ROUTE58 question once, with the WINDOW66 layer tape."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.nan_delay39 import pilot as delay
from experiments.nan_dose51.native import RouteCapture
from experiments.nan_gated46.worker import admission
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_route58 import worker as route
from experiments.nan_route58.capture import fingerprint
from experiments.nan_route58.custody import ARTIFACT, REPO, load_model_inputs, settings
from experiments.nan_route58.source import digest

from .capture import FOCUS, WindowCapture
from .telemetry import run_window_continuation

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "RUNTIME_MANIFEST.json"
STAGE = "native_cuda_pending_after_cpu_receipts"
EXPERIMENT = "NAN-WINDOW-66"


def file_hashes():
    parent = json.loads(route.RUNTIME.read_bytes())
    for source, expected in parent["files_sha256"].items():
        if sha(REPO / source) != expected:
            raise ValueError(f"Frozen ROUTE58 dependency changed: {source}")
    paths = [HERE / name for name in ("capture.py", "telemetry.py", "PROTOCOL.md", "CPU_VERIFICATION.json",
                                     "lane.py", "host.py", "verify_lane.py", "RUNTIME.md")]
    paths += [route.RUNTIME, REPO / "reports/nanbeige_bpd_gym/nan_inverse65_results.json"]
    for source, expected in json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())["source_sha256"].items():
        if sha(REPO / source) != expected:
            raise ValueError(f"CPU window receipt binds different code: {source}")
    return {**parent["files_sha256"], **{p.relative_to(REPO).as_posix(): sha(p) for p in paths}}


def registration():
    host_check = json.loads((HERE / "HOST_VERIFICATION.json").read_bytes())
    live_check = json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())
    if (host_check["status"] != "passed" or live_check["status"] != "passed"
            or live_check["fixture"]["signed_pass2_runs"] != 4
            or host_check["checked_world_slots"] != 24
            or host_check["files_sha256"] != file_hashes()):
        raise ValueError("The required CPU window and host supervision proofs are absent or stale")
    return {"experiment": EXPERIMENT, "stage": STAGE, "window_layers": list(FOCUS),
            "files_sha256": {**host_check["files_sha256"],
                             (HERE / "HOST_VERIFICATION.json").relative_to(REPO).as_posix():
                             sha(HERE / "HOST_VERIFICATION.json")}}


def verify():
    if not __debug__ or Path.cwd() != REPO or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("WINDOW66 frozen runtime changed")
    route.verify()
    return sha(MANIFEST)


def publication():
    commit = delay.verify_publication()
    path = MANIFEST.relative_to(REPO).as_posix()
    published = subprocess.check_output(["git", "show", f"{commit}:{path}"], creationflags=runtime.FLAGS, timeout=30)
    if published != MANIFEST.read_bytes():
        raise RuntimeError("WINDOW66 runtime has no matching research-branch publication")
    return commit


def write_worlds(work, data, custody, gate, trial):
    """Serialize exactly one content-bound journal row for every fixed source world."""
    expected = {(root, world) for root, arm, world in data if arm == "focused"}
    if (set(custody) != expected or not expected
            or any({world for key, world in expected if key == root} != {"A", "B"} for root, _ in expected)):
        raise ValueError("The complete paired focused roster is required")
    qualified = set(gate["qualified_focused_roots"])
    if not qualified <= {root for root, _ in expected}:
        raise ValueError("Native gate qualified a foreign root")
    journal = work / "closed.jsonl"
    with journal.open("x", encoding="utf-8", newline="\n") as stream:
        for i, ((root, world), (own, expected_bits)) in enumerate(sorted(custody.items())):
            source_id, prompt = data[root, "focused", world]
            if not gate["passed"] or root not in qualified:
                status = "native_gate_failed" if not gate["passed"] else "native_root_unqualified"
                row = {"status": status, "edited_pass2_runs": 0}
            else:
                other = custody[root, "B" if world == "A" else "A"][0]
                if own is None or other is None or own == other:
                    raise ValueError("Qualified native twins lack distinct parsed labels")
                row = trial(source_id, prompt, own, other, expected_bits)
            path = work / f"episode_{i:03d}.json"
            write_new(path, row | {"source_id": source_id})
            stream.write(json.dumps({"file": path.name, "sha256": sha(path)}) + "\n")
            stream.flush()
    return {"records": len(custody), "closed_journal_sha256": sha(journal)}


def qualify(source_dir, work, native):
    native_path = work / "native.json"
    gate_path = work / "qualification.json"
    write_new(native_path, native)
    subprocess.run([str(delay.HOST_PYTHON), "-X", "utf8", "-B", "-S", "-m", "experiments.nan_route58.host",
                    "--source-dir", str(source_dir), "--native-records", str(native_path), "--output", str(gate_path)],
                   timeout=60, check=True, creationflags=runtime.FLAGS)
    gate = json.loads(gate_path.read_bytes())
    if gate["native_records_sha256"] != digest(native):
        raise ValueError("Host qualification used different native records")
    return gate, {"native_sha256": sha(native_path), "qualification_sha256": sha(gate_path)}


def execute(args, work, commit, frozen):
    cfg, sham, round_cfg, dose = settings()
    if {pkg: version(pkg) for pkg in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("Pinned GPU environment changed")
    with delay.resources(work) as check:
        data, labels, tokenizer = load_model_inputs(args.source_dir, args.snapshot, check)
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_fit07.pointwise import masks
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_round54.run import READPATH, calibrations
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from experiments.nan_stream36.device_preflight import preflight
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

        torch.set_num_threads(1)
        props = torch.cuda.get_device_properties(0)
        if "3050" not in props.name or not 3.5 * 2**30 <= props.total_memory <= 4.5 * 2**30:
            raise RuntimeError("Expected the registered 4 GiB RTX 3050")
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        with digest_dispatch():
            loop, checkpoint, stage = runtime.load_model(args.backend, args.snapshot, args.adapter)
            before, quant_before = target_digest(loop), quant_state_digest(loop)
            if before != runtime.TARGETS[args.backend] or loop.n - 1 != FOCUS[-1]:
                raise ValueError("The loaded checkpoint or registered focus window changed")
            check(device=True, force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)

            def guard():
                check(device=True)

            try:
                engine = BatchedDepthEngine(loop, head)
                root = min(key[0] for key in data)
                try:
                    proof = preflight(engine, [data[root, "focused", side][1] for side in ("A", "B")],
                                      max(len(item[1]) for item in data.values()) + 2,
                                      lambda *_: guard())
                finally:
                    engine.close()
                plain = RouteCapture(loop, head, layer=FOCUS[-1])
                try:
                    native, custody, vocab = route.native_records(plain, data, tokenizer, cfg, guard)
                finally:
                    plain.close()

                capture = WindowCapture(loop, head)
                try:
                    guard()
                    checked = capture.capture([data[root, "focused", side][1] for side in ("A", "B")])
                    if any(fingerprint(checked, i) != custody[root, side][1]
                           for i, side in enumerate(("A", "B"))):
                        raise RuntimeError("Device window changes the unedited paired baseline")
                    additions = sum(2 * len(layer["residual"]) for layer in checked["window"].values())
                    del checked
                    write_new(work / "preflight.json", {"depths": proof, "stage": stage,
                        "window": {"layers": list(FOCUS), "same_shape_native_bytes_equal": True,
                                   "checked_residual_additions": additions}})
                    gate, hashes = qualify(args.source_dir, work, native)
                    if gate["passed"]:
                        base = json.loads((READPATH / "registration.json").read_bytes())
                        bank = calibrations(base, masks, np)[args.backend]

                        def trial(tag, prompt, own, other, expected_bits):
                            return run_window_continuation(capture, expected_bits, prompt,
                                                           (labels[other], labels[own]), tag, bank,
                                                           cfg["sham_policy"], sham, round_cfg, dose,
                                                           guard, masks, np)

                        out = write_worlds(work, data, custody, gate, trial)
                    else:
                        out = write_worlds(work, data, custody, gate, None)
                finally:
                    capture.close()
            finally:
                head.close()
                torch.cuda.empty_cache()
            guard()
            after, quant_after = target_digest(loop), quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            if before != after or quant_before != quant_after:
                raise RuntimeError("Native target or quantization state changed")
            write_new(work / "terminal.json", out | hashes | {
                "experiment": EXPERIMENT, "mode": "pretrained_cuda", "status": "completed",
                "window_layers": list(FOCUS), "backend": args.backend, "runtime_sha256": frozen,
                "published_commit": commit, "vocabulary_size": vocab, "target_before": before,
                "target_after": after, "quant_unchanged": True, "checkpoint_verified_after": True,
                "preflight_sha256": sha(work / "preflight.json"),
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--check-admission", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if args.freeze:
        write_new(MANIFEST, registration())
        print(sha(MANIFEST))
        return
    if not (args.snapshot and args.adapter and args.source_dir):
        parser.error("--snapshot, --adapter and --source-dir are required")
    args.backend = settings()[0]["backend"]
    frozen = verify()
    if "torch" in sys.modules:
        raise RuntimeError("Torch must load after lane admission")
    if args.check_admission:
        admission(args)
        print("Local hardware prerequisites passed")
        return

    with runtime.exclusive_lane():
        admission(args)
        commit = publication()
        # One shared artifact slot prevents repeating the same ROUTE58 question without the tape.
        work = ARTIFACT / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"experiment": EXPERIMENT, "backend": args.backend,
                                          "runtime_sha256": frozen, "pid": os.getpid(), "commit": commit})
        runtime.environment(work)
        try:
            execute(args, work, commit, frozen)
        except BaseException as error:
            write_new(work / "attempt_failure.json", {"type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
