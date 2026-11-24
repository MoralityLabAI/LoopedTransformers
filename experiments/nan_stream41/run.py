"""Guarded RTX 3050 execution of the READOUT34 two-row stream."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from time import monotonic, perf_counter

from experiments.nan_gym05.common import prompt_ids
from experiments.nan_ram28.resources import memory, soft_guard
from experiments.nan_readout34 import runtime, source
from experiments.nan_readout34.runtime_plan import context_bindings, schedule, write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha, tokenizer
from experiments.nan_stream36.device_preflight import compare, preflight

from .pipeline import append_new, commit, execute, filename, fixed_jobs


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARTIFACT = Path("E:/codex-artifacts/NanbeigeStream41-20261122")
MANIFEST = HERE / "RUNTIME_MANIFEST.json"
STREAM36 = REPO / "experiments/nan_stream36"
MAXIMUM_POSITION = 3463
SOURCES = [
    *sorted(str(path.relative_to(REPO).as_posix()) for path in (
        HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "pipeline.py", "run.py",
                                 "score.py", "verify.py"))),
    "experiments/nan_readout34/RUNTIME_MANIFEST.json",
    "experiments/nan_readout34/runtime_audit.py",
    "experiments/nan_readout34/score.py",
    "experiments/nan_stream36/stream.py",
    "experiments/nan_stream36/batch_engine.py",
    "experiments/nan_stream36/device_preflight.py",
    "experiments/nan_stream36/SCHEDULING_VERIFICATION.json",
    "experiments/nan_stream36/DEVICE_PREFLIGHT_CPU_VERIFICATION.json",
    "experiments/nan_gym05/common.py",
    "experiments/nan_phase35/run.py",
    "experiments/nan_ranksite13/resources.py",
    "experiments/nan_ram28/resources.py",
    "experiments/nan_cache32/resources.py",
    "experiments/nan_cache32/worker.py",
]


def registration():
    return {"experiment": "NAN-STREAM-41", "max_position": MAXIMUM_POSITION,
            "device_limits": runtime.LIMITS, "files_sha256": {p: sha(REPO / p) for p in SOURCES}}


def verify():
    runtime.verify_manifest()
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("STREAM41 runtime manifest differs from local code")


def publication():
    commit_id = runtime.publication()
    name = MANIFEST.relative_to(REPO).as_posix()
    published = subprocess.check_output(["git", "show", f"{commit_id}:{name}"],
                                        timeout=30, creationflags=runtime.FLAGS)
    if published != MANIFEST.read_bytes():
        raise ValueError("Published STREAM41 runtime manifest differs from local bytes")
    return commit_id


def admissions(args):
    runtime.predecessors_closed()
    if memory()["available_physical_bytes"] < runtime.LIMITS["minimum_start_available_RAM_bytes"]:
        raise RuntimeError("STREAM41 requires 5 GiB available RAM before admission")
    if shutil.disk_usage(ARTIFACT.parent).free < runtime.LIMITS["minimum_free_disk_bytes"]:
        raise RuntimeError("STREAM41 artifact volume is below the free-space floor")
    args.snapshot = args.snapshot.resolve()
    args.adapter = args.adapter.resolve()
    if (args.snapshot.name != source.CONFIG["checkpoint_revision"]
            or args.adapter != (REPO.parent / "Compactification-Theory-arg02").resolve()):
        raise ValueError("Incorrect checkpoint revision or adapter path")
    actual = subprocess.check_output(["git", "-C", str(args.adapter), "rev-parse", "HEAD"], text=True,
                                     creationflags=runtime.FLAGS, timeout=15).strip()
    if actual != runtime.ADAPTER_COMMIT:
        raise ValueError("Pinned adapter commit changed")
    if args.backend == "bf16" and not (ARTIFACT / "nf4" / "admission.json").is_file():
        raise RuntimeError("NF4 must be admitted before BF16")


def nvml_snapshot():
    exe = shutil.which("nvidia-smi")
    fields = ("utilization.gpu", "memory.used", "temperature.gpu", "clocks.current.sm")
    if exe is None:
        return {"status": "tool_unavailable"}
    try:
        raw = subprocess.check_output([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                      timeout=10, creationflags=runtime.FLAGS, text=True).strip().splitlines()
        if len(raw) != 1:
            return {"status": "ambiguous_device"}
        return dict(zip(fields, (int(value) for value in raw[0].split(",")), strict=True))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"status": type(error).__name__}


@contextmanager
def guard_lane(work):
    limits = runtime.LIMITS
    begin = monotonic()
    stop = threading.Event()

    def watch():
        while not stop.wait(2):
            state = memory()
            if (state["available_physical_bytes"] < limits["hard_minimum_available_RAM_bytes"]
                    or state["private_bytes"] > limits["hard_maximum_worker_private_bytes"]):
                print(f"STREAM41 own-process hard RAM stop: {state}", flush=True)
                os._exit(3)

    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()
    last_device = -600.0

    def check(job_start=None, batch=None, position=None, *, device=False, force=False):
        nonlocal last_device
        state = soft_guard(limits)
        now = monotonic()
        if (now - begin > limits["maximum_backend_seconds"]
                or (job_start is not None and now - job_start > limits["maximum_slot_seconds"])
                or shutil.disk_usage(work).free < limits["minimum_free_disk_bytes"]):
            raise RuntimeError("STREAM41 time or disk budget exhausted")
        if batch is not None and (batch not in (1, 2) or not 1 <= position <= MAXIMUM_POSITION):
            raise ValueError("Physical batch exceeds the admitted rows or position")
        if device and (force or now - begin - last_device >= 600):
            from experiments.nan_cache32.resources import gpu_memory

            gpu = gpu_memory(os.getpid())
            append_new(work / "resource_samples.jsonl", {
                "elapsed_seconds": now - begin, "host": state, "process_device": gpu, "device": nvml_snapshot()})
            last_device = now - begin
            if (gpu["dedicated_bytes"] > limits["maximum_sampled_dedicated_GPU_bytes"]
                    or gpu["shared_bytes"] > limits["maximum_sampled_shared_GPU_bytes"]):
                raise RuntimeError("STREAM41 sampled device allocation exceeds the registered limit")

    try:
        yield check
    finally:
        stop.set()
        watchdog.join()


def benchmark(engine, inputs, guard, sync):
    """Paired CAL prefill and four fixed-token increments, timed around each arm."""
    forced_ids = [values[-4:] for values in inputs]
    report = {}
    for depth in (1, 2):
        timings = {}

        def capture(indices):
            chosen = [inputs[i] for i in indices]
            canvas = max(map(len, chosen))
            engine.reset(depth)
            sync()
            started = monotonic()
            timer = perf_counter()
            guard(started, len(indices), canvas)
            scored = [engine.prefill_logits(chosen)]
            for step in range(4):
                guard(started, len(indices), canvas + step + 1)
                tokens = [forced_ids[index][step] for index in indices]
                scored.append(engine.advance_logits(list(range(len(indices))), tokens))
            sync()
            timings.setdefault(len(indices), []).append(perf_counter() - timer)
            engine.reset(depth)
            return scored

        try:
            if depth == 1:
                b1 = [capture([i]) for i in range(2)]
                b2 = capture([0, 1])
            else:
                b2 = capture([0, 1])
                b1 = [capture([i]) for i in range(2)]
            comparisons = [compare(batch[i:i + 1], b1[i][step])
                           for step, batch in enumerate(b2) for i in range(2)]
            report[str(depth)] = {
                "solo_wall_seconds": timings[1], "pair_wall_seconds": timings[2][0],
                "solo_total_over_pair_wall": sum(timings[1]) / timings[2][0],
                "B1_first": depth == 1, "comparisons": comparisons,
            }
        finally:
            engine.reset(depth)
    return report


def prepare(args, receipt, check):
    tok, pins = tokenizer(args.snapshot)
    if pins != receipt["tokenizer_files_sha256"]:
        raise ValueError("Pinned tokenizer files changed")
    rows = [source.public(row) for row in source.build()]
    bindings = receipt["contexts"]
    reference = json.loads((STREAM36 / "SCHEDULING_VERIFICATION.json").read_bytes())
    jobs = fixed_jobs(schedule(rows), bindings, reference)
    prompts = {}
    for row in rows:
        check()
        binding = bindings[row["id"]]
        ids = prompt_ids(tok, args.snapshot, row["prompt"])
        if binding != {"tokens": len(ids), "ids_sha256": id_hash(ids)}:
            raise ValueError("A runtime prompt differs from the frozen tokenizer receipt")
        prompts[row["id"]] = ids
    if tok.tok.eos_token_id != receipt["eos_token_id"]:
        raise ValueError("Prompt setup did not bind the registered EOS")
    cal = [job for job in jobs if job.slots[0].passes == 1 and all(
        slot.source["split"] == "CAL" and slot.source["arm"] == "compute"
        and slot.source["schema"]["readout"] == "final" for slot in job.slots)]
    pair = min(cal, key=lambda job: job.admitted_max_position)
    inputs = [prompts[slot.source["id"]] for slot in pair.slots]
    return tok, jobs, prompts, inputs


def gpu_run(args, work, receipt, published):
    from experiments.nan_readout34.runtime_audit import decode_with

    if {name: version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("GPU interpreter package pins changed")
    with guard_lane(work) as check:
        tok, jobs, prompts, inputs = prepare(args, receipt, check)
        import torch
        from experiments.nan_bf31.loader import verify_checkpoint
        from experiments.nan_cache32.worker import quant_state_digest
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_stream36.batch_engine import BatchedDepthEngine
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead

        torch.set_num_threads(1)
        props = torch.cuda.get_device_properties(0)
        if "3050" not in props.name or not 3.5 * 2**30 <= props.total_memory <= 4.5 * 2**30:
            raise RuntimeError("STREAM41 requires the registered 4 GiB RTX 3050")
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.reset_peak_memory_stats()
        with digest_dispatch():
            loop, checkpoint, stage = runtime.load_model(args.backend, args.snapshot, args.adapter)
            before, quant_before = target_digest(loop), quant_state_digest(loop)
            if before != runtime.TARGETS[args.backend]:
                raise ValueError("Physical target differs from the registered backend")
            check(device=True, force=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            engine = BatchedDepthEngine(loop, head, chunk=128)
            count, artifact_bytes, verified_preflight = 0, 0, False
            failure = None

            def guard(start, batch, position):
                if artifact_bytes >= runtime.LIMITS["maximum_artifact_bytes_per_backend"]:
                    raise RuntimeError("STREAM41 artifact budget exhausted")
                check(start, batch, position, device=True)

            try:
                depths = preflight(engine, inputs, MAXIMUM_POSITION, guard)
                performance = benchmark(engine, inputs, guard, torch.cuda.synchronize)
                write_new(work / "preflight.json", {"depths": depths, "stage_check": stage,
                                                    "matched_CAL_timing": performance})
                verified_preflight = True
                for result in execute(engine, jobs, prompts, receipt["contexts"], decode=decode_with(tok),
                                      eos=receipt["eos_token_id"], guard=guard, synchronize=torch.cuda.synchronize):
                    commit(work, result, count)
                    artifact_bytes += (work / filename(count)).stat().st_size
                    count += 1
                check(device=True, force=True)
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error)}
            finally:
                engine.close()
                head.close()
                torch.cuda.empty_cache()
            after, quant_after = target_digest(loop), quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            journal = work / "closed.jsonl"
            resource_log = work / "resource_samples.jsonl"
            write_new(work / "terminal.json", {
                "status": "completed" if count == len(jobs) and failure is None else "partial",
                "failure": failure, "backend": args.backend, "runtime_manifest_sha256": sha(MANIFEST),
                "published_commit": published, "preflight_passed": verified_preflight,
                "preflight_sha256": sha(work / "preflight.json") if verified_preflight else None,
                "target_before": before, "target_after": after, "quant_before": quant_before,
                "quant_after": quant_after, "checkpoint_verified_after": True,
                "closed_batches": count, "registered_batches": len(jobs), "closed_slots": 2 * count,
                "closed_journal_sha256": sha(journal) if journal.exists() else None,
                "resource_log_sha256": sha(resource_log),
                "vocabulary_size": engine.vocabulary_size, "max_torch_allocated": torch.cuda.max_memory_allocated(),
                "max_torch_reserved": torch.cuda.max_memory_reserved(),
            })
            if failure or before != after or quant_before != quant_after:
                raise RuntimeError("STREAM41 backend did not pass full execution and integrity")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--backend", choices=tuple(runtime.TARGETS))
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--adapter", type=Path)
    cli.add_argument("--tokenizer-receipt", type=Path)
    args = cli.parse_args()
    if args.freeze:
        runtime.verify_manifest()
        write_new(MANIFEST, registration())
        print(sha(MANIFEST))
        return
    if not all((args.backend, args.snapshot, args.adapter, args.tokenizer_receipt)):
        cli.error("--backend, --snapshot, --adapter and --tokenizer-receipt are required")
    verify()
    with runtime.exclusive_lane():
        admissions(args)
        published = publication()
        receipt = context_bindings(args.tokenizer_receipt)
        work = ARTIFACT / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"backend": args.backend, "pid": os.getpid(),
                                           "manifest_sha256": sha(MANIFEST), "published_commit": published})
        runtime.environment(work)
        try:
            gpu_run(args, work, receipt, published)
        except BaseException as error:
            write_new(work / "attempt_failure.json", {"error_type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
