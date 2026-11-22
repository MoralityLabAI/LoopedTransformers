"""Guarded single-model ROUTE58 runtime with a separate host gold process."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from experiments.nan_comp14.parser import label
from experiments.nan_delay39 import pilot as delay
from experiments.nan_dose51.native import RouteCapture
from experiments.nan_gated46.worker import admission
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_round54.run import READPATH, calibrations
from experiments.nan_sham56.native import continuation

from .capture import ObservedCapture, fingerprint
from .custody import ARTIFACT, HERE, REPO, RUNTIME, load_model_inputs, settings
from .source import SHAM, digest


def registration():
    own = [HERE / name for name in ("SOURCE_MANIFEST.json", "SOURCE_CHECK.json", "RUNTIME.md", "custody.py",
                                    "capture.py", "worker.py", "host.py", "verify.py", "CPU_VERIFICATION.json")]
    legacy = [REPO / name for name in (
        "experiments/nan_delay39/SOURCE_MANIFEST.json", "experiments/nan_delay39/pilot.py",
        "experiments/nan_delay39/host_gate.py", "experiments/nan_gated46/worker.py",
        "experiments/nan_gated46/host.py", "experiments/nan_queue43/LOCAL_MANIFEST.json",
        "experiments/nan_readout34/RUNTIME_MANIFEST.json", "experiments/nan_readout34/runtime.py",
        "experiments/nan_stream36/device_preflight.py", "experiments/nan_stream36/batch_engine.py",
        "experiments/nan_round54/registration.json", "experiments/nan_round54/run.py",
        "experiments/nan_dose51/registration.json", "experiments/nan_dose51/native.py",
        "experiments/nan_fit07/pointwise.py", "experiments/nan_route50/run.py",
        "reports/nanbeige_bpd_gym/nan_route50_results.json", "experiments/nan_frontier55/run.py")]
    linked = [SHAM / name for name in ("SOURCE_MANIFEST.json", "MINIMAX_MANIFEST.json", "NATIVE_MANIFEST.json")]
    checked = {}
    for path in linked:
        for dep, expected in json.loads(path.read_bytes()).items():
            if sha(REPO / dep) != expected:
                raise ValueError(f"The staged sham dependency changed: {dep}")
            checked[dep] = expected
    info = json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())
    if info["status"] != "passed" or info["synthetic_signed_edits"] != 4 or info["verified_host_worlds"] != 2:
        raise ValueError("Missing end-to-end CPU fixture")
    for name, expected in info["files_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("The verified CPU runtime code changed")
    paths = own + legacy + linked
    return {"experiment": "NAN-ROUTE-58", "stage": "gpu_pending_after_cpu_verification",
            "files_sha256": checked | {p.relative_to(REPO).as_posix(): sha(p) for p in paths}}


def verify():
    if not __debug__ or Path.cwd() != REPO or json.loads(RUNTIME.read_bytes()) != registration():
        raise ValueError("ROUTE58 frozen runtime changed")
    runtime.verify_manifest()


def publication():
    revision = delay.verify_publication()
    path = RUNTIME.relative_to(REPO).as_posix()
    remote = subprocess.check_output(["git", "show", f"{revision}:{path}"], creationflags=runtime.FLAGS, timeout=30)
    if remote != RUNTIME.read_bytes():
        raise RuntimeError("ROUTE58 runtime has no matching branch publication")
    return revision


def native_records(engine, data, tok, cfg, guard):
    records, custody = [], {}
    for root in sorted({key[0] for key in data}):
        for arm in cfg["arms"]:
            guard()
            twins = [data[root, arm, world] for world in ("A", "B")]
            actual = engine.capture([item[1] for item in twins])
            vocab = actual["logits"].shape[-1]
            for i, world in enumerate(("A", "B")):
                token = int(actual["logits"][1, i].argmax())
                text = delay.literal(tok, token)
                records.append({"id": twins[i][0], "token_id": token, "text": text})
                if arm == "focused":
                    custody[root, world] = (label(text), fingerprint(actual, i))
    return records, custody, vocab


def groups(engine, data, labels, tok, args, work, cfg, sham_cfg, round_cfg, dose, bank, guard, masks, np):
    native, custody, vocab = native_records(engine, data, tok, cfg, guard)
    native_path = work / "native.json"
    gate_path = work / "qualification.json"
    write_new(native_path, native)
    subprocess.run([str(delay.HOST_PYTHON), "-X", "utf8", "-B", "-S", "-m", "experiments.nan_route58.host",
                    "--source-dir", str(args.source_dir), "--native-records", str(native_path), "--output", str(gate_path)],
                   check=True, timeout=60, creationflags=runtime.FLAGS)
    gate = json.loads(gate_path.read_bytes())
    if gate["native_records_sha256"] != digest(native):
        raise ValueError("Host qualification binds a different native tranche")
    qualified = set(gate["qualified_focused_roots"])
    journal = work / "closed.jsonl"
    for i, ((root, world), (own, expected)) in enumerate(sorted(custody.items())):
        source_id, prompt = data[root, "focused", world]
        if not gate["passed"] or root not in qualified:
            status = "native_gate_failed" if not gate["passed"] else "native_root_unqualified"
            row = {"status": status, "edited_pass2_runs": 0}
        else:
            other = custody[root, "B" if world == "A" else "A"][0]
            if own not in labels or other not in labels or own == other:
                raise ValueError("Qualified native twins lack distinct labels")
            pair = labels[other], labels[own]
            observed = ObservedCapture(engine, expected, pair)
            value = continuation(observed, prompt, pair, source_id, bank, cfg["sham_policy"],
                                 sham_cfg, round_cfg, dose, guard, masks, np)
            row = observed.bind(value)
        path = work / f"episode_{i:03d}.json"
        write_new(path, row | {"source_id": source_id})
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"file": path.name, "sha256": sha(path)}) + "\n")
    return {"status": "completed", "records": len(custody), "vocabulary_size": vocab,
            "native_sha256": sha(native_path), "qualification_sha256": sha(gate_path), "closed_journal_sha256": sha(journal)}


def execute(args, work, commit):
    cfg, sham_cfg, round_cfg, dose = settings()
    if {name: version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("The pinned GPU environment changed")
    with delay.resources(work) as check:
        data, ids, tokenizer = load_model_inputs(args.source_dir, args.snapshot, check)
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_fit07.pointwise import masks
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_round54.run import calibrations
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
            if before != runtime.TARGETS[args.backend]:
                raise ValueError("Loaded target changed")
            check(device=True, force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            guard = lambda: check(device=True)
            try:
                probe = BatchedDepthEngine(loop, head)
                root = sorted(data)[0][0]
                probes = [data[root, "focused", world][1] for world in ("A", "B")]
                try:
                    proof = preflight(probe, probes, max(len(v[1]) for v in data.values()) + 2,
                                      lambda _start, _batch, _position: guard())
                finally:
                    probe.close()
                write_new(work / "preflight.json", {"depths": proof, "stage": stage})
                base = json.loads((READPATH / "registration.json").read_bytes())
                bundle = calibrations(base, masks, np)[args.backend]
                capture = RouteCapture(loop, head, layer=cfg["layer_zero_based"])
                try:
                    out = groups(capture, data, ids, tokenizer, args, work, cfg, sham_cfg, round_cfg, dose,
                                 bundle, guard, masks, np)
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
            write_new(work / "terminal.json", out | {
                "backend": args.backend, "runtime_sha256": sha(RUNTIME), "published_commit": commit,
                "target_before": before, "target_after": after, "quant_unchanged": True,
                "checkpoint_verified_after": True, "preflight_sha256": sha(work / "preflight.json"),
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--adapter", type=Path)
    cli.add_argument("--source-dir", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(RUNTIME, registration())
        print(sha(RUNTIME))
        return
    if not (args.snapshot and args.adapter and args.source_dir):
        cli.error("--snapshot, --adapter and --source-dir are required")
    cfg, _, _, _ = settings()
    args.backend = cfg["backend"]
    verify()
    with runtime.exclusive_lane():
        admission(args)
        commit = publication()
        work = ARTIFACT / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"experiment": cfg["experiment"], "backend": args.backend,
                                           "runtime_sha256": sha(RUNTIME), "pid": os.getpid(), "commit": commit})
        runtime.environment(work)
        try:
            execute(args, work, commit)
        except BaseException as error:
            write_new(work / "attempt_failure.json", {"type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
