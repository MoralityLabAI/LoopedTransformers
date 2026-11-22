"""Guarded device commissioning for the GATED46 cross-pass pilot."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess

from experiments.nan_comp14.parser import label
from experiments.nan_delay39 import pilot as prior
from experiments.nan_delay39.contrasts import numpy_state
from experiments.nan_delay39.instrument import clean_row_closes
from experiments.nan_ram28.resources import memory
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .gated import GateCapture, run_world
from .source import CONFIG, HERE, REPO, load_gpu_source, digest


MANIFEST = HERE / "SOURCE_MANIFEST.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeGated46-20261122")
DELAY_QUEUE = REPO / "runs/nan_queue43/attempt01"


def registration():
    deps = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json",
                                     "SOURCE_CHECK.json", "CPU_VERIFICATION.json", "gated.py", "source.py", "host.py",
                                     "worker.py", "verify.py")]
    deps += [REPO / name for name in (
        "experiments/nan_delay39/SOURCE_MANIFEST.json", "experiments/nan_delay39/host_gate.py",
        "experiments/nan_delay39/instrument.py", "experiments/nan_delay39/pilot.py",
        "experiments/nan_delay39/source.py", "experiments/nan_gauge42/transport.py",
        "experiments/nan_queue43/LOCAL_MANIFEST.json", "experiments/nan_readout34/RUNTIME_MANIFEST.json",
        "experiments/nan_readout34/runtime.py", "experiments/nan_stream36/device_preflight.py",
        "experiments/nan_stream36/batch_engine.py", "mechinterp/nanbeige_bpd_gym/chunked_cache.py",
        "mechinterp/nanbeige_bpd_gym/streamed_vocab_head.py",
        "mechinterp/nanbeige_bpd_gym/tests/test_chunked_cache.py")]
    fixture = json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())
    if fixture["status"] != "passed" or fixture["bad_plan_edited_pass2_calls"] != 0 or fixture["live_mismatch_edited_pass2_calls"] != 0:
        raise ValueError("CPU gate refusal fixture is missing")
    for name, expected in fixture["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Bound CPU fixture source changed")
    return {"experiment": "NAN-GATED-46", "pretrained_forwards_at_freeze": 0,
            "files_sha256": {path.relative_to(REPO).as_posix(): sha(path) for path in deps}}


def verify():
    if not __debug__ or Path.cwd() != REPO or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("The registered runtime differs from the GATED46 freeze")
    runtime.verify_manifest()


def admission(args):
    runtime.predecessors_closed()
    if not (DELAY_QUEUE / "terminal.json").is_file():
        raise RuntimeError("The local DELAY39 queue remains pending")
    terminal = json.loads((DELAY_QUEUE / "terminal.json").read_bytes())
    if (terminal["status"] != "completed" or terminal["local_manifest_sha256"] !=
            sha(REPO / "experiments/nan_queue43/LOCAL_MANIFEST.json")):
        raise RuntimeError("The registered DELAY39 predecessor has no valid closed receipt")
    if memory()["available_physical_bytes"] < runtime.LIMITS["minimum_start_available_RAM_bytes"]:
        raise RuntimeError("GATED46 requires 5 GiB available RAM at admission")
    if shutil.disk_usage(ARTIFACT.parent).free < runtime.LIMITS["minimum_free_disk_bytes"]:
        raise RuntimeError("Evidence volume is below its configured free-space floor")
    args.snapshot = args.snapshot.resolve()
    args.adapter = args.adapter.resolve()
    cfg = json.loads(CONFIG.read_bytes())
    if args.snapshot.name != cfg["checkpoint_revision"] or args.adapter != (REPO.parent / "Compactification-Theory-arg02"):
        raise ValueError("Wrong pinned checkpoint and adapter")
    commit = subprocess.check_output(["git", "-C", str(args.adapter), "rev-parse", "HEAD"], text=True,
                                     creationflags=runtime.FLAGS, timeout=15).strip()
    if commit != runtime.ADAPTER_COMMIT:
        raise ValueError("Pinned native adapter changed")
    if args.backend == "bf16" and not (ARTIFACT / "nf4" / "admission.json").is_file():
        raise RuntimeError("Admit NF4 before staged BF16")


def published():
    commit = prior.verify_publication()
    path = MANIFEST.relative_to(REPO).as_posix()
    frozen = subprocess.check_output(["git", "show", f"{commit}:{path}"], creationflags=runtime.FLAGS, timeout=30)
    if frozen != MANIFEST.read_bytes():
        raise ValueError("Published GATED46 frozen bytes differ")
    return commit


def native_records(engine, data, tok, cfg, guard):
    records, focused = [], {}
    for root in sorted({key[0] for key in data}):
        for arm in cfg["arms"]:
            guard()
            a, b = [data[root, arm, world] for world in ("A", "B")]
            native = engine.capture([a[1], b[1]])
            chosen = native["logits"][1].argmax(dim=-1).tolist()
            for idx, (source, token) in enumerate(zip((a, b), chosen)):
                text = prior.literal(tok, token)
                records.append({"id": source[0], "token_id": token, "text": text})
                if arm == "focused":
                    focused[root, "AB"[idx]] = (native["inputs"][1, idx], native["residual"][1, idx], token, label(text))
    return records, focused


def groups(engine, data, labels, tok, args, work, cfg, np, guard):
    import torch

    native, focused = native_records(engine, data, tok, cfg, guard)
    native_path = work / "native.json"
    gate_path = work / "qualification.json"
    write_new(native_path, native)
    subprocess.run([str(prior.HOST_PYTHON), "-X", "utf8", "-B", "-S", "-m", "experiments.nan_gated46.host",
                    "--source-dir", str(args.source_dir), "--native-records", str(native_path), "--output", str(gate_path)],
                   check=True, timeout=30, creationflags=runtime.FLAGS)
    gate = json.loads(gate_path.read_bytes())
    if gate["native_records_sha256"] != digest(native):
        raise ValueError("Independent qualification receipt points to different decisions")
    if not gate["passed"]:
        return {"status": "native_gate_failed", "records": 0, "qualification_sha256": sha(gate_path)}
    bank, writes = prior.get_bank(args.backend, np)
    count = 0
    journal = work / "closed.jsonl"
    for root, world in sorted(focused):
        name, prompt = data[root, "focused", world]
        if root not in gate["qualified_focused_roots"]:
            record = {"source_id": name, "status": "native_root_unqualified", "phase2_edit_count": 0}
        else:
            guard()
            a, h, winner, answer = focused[root, world]
            other, _, _, donor = focused[root, "B" if world == "A" else "A"]
            if answer not in labels or donor not in labels or answer == donor:
                raise ValueError("Independently qualified native worlds did not identify two bare labels")
            base = engine.capture([prompt, prompt])
            if (not torch.equal(base["logits"][:, 0], base["logits"][:, 1])
                    or any(not torch.equal(base[site][phase, 0], base[site][phase, 1])
                           for site in ("inputs", "residual") for phase in (1, 2))
                    or not torch.equal(base["inputs"][1, 1], a) or not torch.equal(base["residual"][1, 1], h)
                    or int(base["logits"][1, 1].argmax()) != winner):
                raise RuntimeError("Duplicate clean rows differ from the paired source tranche")
            clean_row_closes(base, engine.capture([prompt, prompt]), expect_edit=False)
            read_delta = (numpy_state(other) - numpy_state(a)) @ bank
            record = run_world(engine, base, prompt, read_delta, bank, writes, (labels[donor], labels[answer]),
                               name, cfg, tok, guard, np)
        episode = work / f"episode_{count:04d}.json"
        write_new(episode, record)
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"file": episode.name, "sha256": sha(episode)}) + "\n")
        count += 1
    return {"status": "closed", "records": count, "qualification_sha256": sha(gate_path),
            "closed_journal_sha256": sha(journal)}


def execute(args, work, commit):
    cfg = json.loads(CONFIG.read_bytes())
    if {name: version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("The pinned GPU environment changed")
    with prior.resources(work) as check:
        data, labels, tok = load_gpu_source(args.source_dir, args.snapshot, check)
        import numpy as np
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_recomp09.common import target_digest
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
                raise ValueError("Loaded checkpoint target does not match the registration")
            check(device=True, force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            guard = lambda: check(device=True)
            try:
                probe = BatchedDepthEngine(loop, head)
                root = sorted(data)[0][0]
                inputs = [data[root, "focused", world][1] for world in ("A", "B")]
                try:
                    admission_proof = preflight(probe, inputs, max(len(v[1]) for v in data.values()) + 2,
                                                lambda _start, _batch, _position: guard())
                finally:
                    probe.close()
                write_new(work / "preflight.json", {"depths": admission_proof, "stage": stage})
                engine = GateCapture(loop, head, layer=cfg["layer_zero_based"])
                try:
                    result = groups(engine, data, labels, tok, args, work, cfg, np, guard)
                finally:
                    engine.close()
            finally:
                head.close()
                torch.cuda.empty_cache()
            guard()
            after, quant_after = target_digest(loop), quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            if before != after or quant_before != quant_after:
                raise RuntimeError("Native target or quantization state changed")
            write_new(work / "terminal.json", result | {
                "backend": args.backend, "source_manifest_sha256": sha(MANIFEST), "published_commit": commit,
                "target_before": before, "target_after": after, "quant_unchanged": True,
                "checkpoint_verified_after": True, "preflight_sha256": sha(work / "preflight.json"),
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            })


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--backend", choices=tuple(runtime.TARGETS))
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--adapter", type=Path)
    cli.add_argument("--source-dir", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, registration())
        print(sha(MANIFEST))
        return
    if not all((args.backend, args.snapshot, args.adapter, args.source_dir)):
        cli.error("A backend, snapshot, pinned adapter and source directory are required")
    verify()
    with runtime.exclusive_lane():
        admission(args)
        commit = published()
        work = ARTIFACT / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"backend": args.backend, "pid": os.getpid(), "published_commit": commit,
                                           "source_manifest_sha256": sha(MANIFEST)})
        runtime.environment(work)
        try:
            execute(args, work, commit)
        except BaseException as error:
            write_new(work / "attempt_failure.json", {"type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
