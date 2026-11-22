"""READOUT34 GPU commissioning entrypoint. All admission precedes Torch."""
import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from experiments.nan_ram28.resources import memory, soft_guard
from experiments.nan_ranksite13.resources import below_normal

from . import source
from .runtime_plan import CAP, context_bindings, execute, schedule, write_new
from .tokenizer_check import id_hash, sha, tokenizer


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "RUNTIME_MANIFEST.json"
REF = "refs/heads/research/nanbeige-bpd-gym-suite-towers"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeReadout34-20261122")
PRIOR_ARTIFACT = Path("E:/codex-artifacts/NanbeigeCache32-Inventory2-20261122")
PREDECESSORS = (
    (28152, 134397440962966790), (62488, 134397440878169688),
    (81592, 134397685411004030), (48796, 134397844738228651),
)
GPU_PACKAGES = {"torch": "2.6.0+cu124", "transformers": "4.46.3", "tokenizers": "0.20.3",
                "bitsandbytes": "0.50.2", "numpy": "2.5.5"}
ADAPTER_COMMIT = "1ac36353b36581b6a634626bcd55542e53dae456"
TARGETS = {"nf4": "8e80e7d748d5daf858a0b37dd7843d38952554ce3aabea1104e52cf1ef7663b3",
           "bf16": "cf3db880c052137e7ff7a0dac1e498174f78e32fd3ae9a384aca341fb1d09186"}
LIMITS = {
    "minimum_start_available_RAM_bytes": 5 * 2**30,
    "minimum_available_RAM_bytes": 2 * 2**30,
    "maximum_worker_private_bytes": 6 * 2**30,
    "hard_minimum_available_RAM_bytes": 2**30,
    "hard_maximum_worker_private_bytes": 13 * 2**29,
    "maximum_sampled_dedicated_GPU_bytes": 4 * 2**30,
    "maximum_sampled_shared_GPU_bytes": 256 * 2**20,
    "maximum_backend_seconds": 7 * 24 * 3600,
    "maximum_slot_seconds": 3600,
    "minimum_free_disk_bytes": 20 * 2**30,
    "maximum_artifact_bytes_per_backend": 256 * 2**20,
}
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def verify_manifest():
    if not __debug__ or Path.cwd() != REPO:
        raise RuntimeError("Run READOUT34 from the canonical checkout with assertions enabled")
    seal = json.loads(MANIFEST.read_bytes())
    if (seal["stage"] != "runtime_software_verified_gpu_pending" or seal["limits"] != LIMITS
            or seal["expected_targets"] != TARGETS or seal["packages"] != GPU_PACKAGES):
        raise ValueError("Runtime manifest and worker disagree")
    for name, expected in seal["files_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Runtime dependency changed: {name}")
    return seal


def publication():
    remote = subprocess.check_output(["git", "ls-remote", "--heads", "origin", REF],
                                     creationflags=FLAGS, timeout=30).split()
    if len(remote) != 2 or remote[1].decode("ascii") != REF:
        raise RuntimeError("Current publication could not be verified")
    commit = remote[0].decode("ascii")
    path = MANIFEST.relative_to(REPO).as_posix()
    published = subprocess.check_output(["git", "show", f"{commit}:{path}"],
                                        creationflags=FLAGS, timeout=30)
    if hashlib.sha256(published).hexdigest() != sha(MANIFEST):
        raise RuntimeError("Published runtime freeze differs from local bytes")
    return commit


@contextmanager
def exclusive_lane():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    mutex = kernel.CreateMutexW(None, False, r"Local\MoralityLabAI-Nanbeige-Readout34-GPU")
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        acquired = kernel.WaitForSingleObject(mutex, 0) in (0, 0x80)
        if not acquired:
            raise RuntimeError("The READOUT34 GPU lane is already held")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(mutex)
        kernel.CloseHandle(mutex)


def predecessors_closed():
    from experiments.nan_cache32.inventory import ensure_clear
    from experiments.nan_storyaudit30.common import Handle

    for pid, creation in PREDECESSORS:
        try:
            process = Handle(pid)
        except OSError as error:
            if error.winerror != 87:
                raise
            continue
        try:
            if process.creation == creation and process.alive():
                raise RuntimeError(f"Existing Nanbeige process remains live: {pid}")
        finally:
            process.close()
    model = json.loads((PRIOR_ARTIFACT / "model_lane_terminal.json").read_bytes())
    host = json.loads((PRIOR_ARTIFACT / "host_lane_terminal.json").read_bytes())
    if model.get("all_actual_workers_ended") is not True or host.get("actual_host_launcher_ended") is not True:
        raise RuntimeError("Existing cached model and host lanes have not closed")
    ensure_clear()


def environment(work):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="0",
                      HF_HOME=str(work / "hf"), HF_MODULES_CACHE=str(work / "hf_modules"))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    below_normal()


def load_model(backend, snapshot, adapter):
    import torch
    from experiments.nan_bf31.loader import checkpoint_entry, verify_checkpoint
    from experiments.nan_ram28.loader import empty_model, materialize
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import _reinit_rotary_buffers, assert_oracle_isolated
    from mechinterp.nanbeige_bpd_gym.chunked_backend import build_nf4_loop
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_backend import build_loop
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_storage import CheckpointStore
    from experiments.nan_bf31.stage_audit import audit

    assert_oracle_isolated()
    entry = checkpoint_entry(adapter, snapshot)
    verify_checkpoint(snapshot, entry)
    model = empty_model(snapshot)
    from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige
    nanbeige.check_supported(model.config)

    def resource_guard():
        return soft_guard(LIMITS)

    if backend == "nf4":
        materialize(model, snapshot)
        for layer in model.model.layers:
            _reinit_rotary_buffers(layer, torch.device("cpu"))
        model.eval().requires_grad_(False)
        loop = build_nf4_loop(model, "cuda:0", guard=resource_guard)
        stage = None
    else:
        loop = build_loop(model, CheckpointStore(snapshot), "cuda:0", n_resident=4,
                          reference_modules=("model.layers.21.mlp.down_proj",), guard=resource_guard)
        stage = audit(loop, resource_guard)
    assert_oracle_isolated()
    return loop, entry, stage


def execute_gpu(args, work, receipt, commit):
    from experiments.nan_gym05.common import prompt_ids

    if {name: version(name) for name in GPU_PACKAGES} != GPU_PACKAGES:
        raise RuntimeError("Unexpected GPU package version")
    tok, pins = tokenizer(args.snapshot)
    if pins != receipt["tokenizer_files_sha256"]:
        raise RuntimeError("Pinned tokenizer changed")
    prepare = lambda row: prompt_ids(tok, args.snapshot, row["prompt"])
    rows = [source.public(row) for row in source.build()]
    slots = schedule(rows)
    bindings = receipt["contexts"]
    for row in rows:
        ids = prepare(row)
        binding = bindings[row["id"]]
        if len(ids) != binding["tokens"] or id_hash(ids) != binding["ids_sha256"]:
            raise RuntimeError("Runtime tokenizer does not reproduce all 432 prompt bindings")
    if tok.tok.eos_token_id != receipt["eos_token_id"]:
        raise RuntimeError("EOS binding changed")
    soft_guard(LIMITS)

    import torch
    from experiments.nan_bf31.loader import verify_checkpoint
    from experiments.nan_cache32.resources import gpu_memory
    from experiments.nan_cache32.worker import quant_state_digest
    from experiments.nan_recomp09.common import target_digest
    from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
    from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
    from .depth_engine import DepthEngine

    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    if "3050" not in props.name or not 3.5 * 2**30 <= props.total_memory <= 4.5 * 2**30:
        raise RuntimeError("Expected the registered 4 GiB RTX 3050 CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.cuda.reset_peak_memory_stats()
    loaded_started = time.monotonic()
    last_gpu = 0
    artifact_bytes = 0
    count = 0
    stopped = threading.Event()

    def hard_watch():
        while not stopped.wait(2):
            s = memory()
            if (s["available_physical_bytes"] < LIMITS["hard_minimum_available_RAM_bytes"]
                    or s["private_bytes"] > LIMITS["hard_maximum_worker_private_bytes"]):
                print(f"READOUT34 own-process hard RAM stop: {s}", file=sys.stderr, flush=True)
                os._exit(3)

    def budget(slot_start):
        nonlocal last_gpu
        soft_guard(LIMITS)
        now = time.monotonic()
        if (now - loaded_started > LIMITS["maximum_backend_seconds"]
                or now - slot_start > LIMITS["maximum_slot_seconds"]
                or artifact_bytes >= LIMITS["maximum_artifact_bytes_per_backend"]
                or shutil.disk_usage(work).free < LIMITS["minimum_free_disk_bytes"]):
            raise RuntimeError("READOUT34 runtime budget exhausted")
        if now - last_gpu >= 600:
            sample = gpu_memory(os.getpid())
            if (sample["dedicated_bytes"] > LIMITS["maximum_sampled_dedicated_GPU_bytes"]
                    or sample["shared_bytes"] > LIMITS["maximum_sampled_shared_GPU_bytes"]):
                raise RuntimeError("READOUT34 sampled device allocation exceeded its limits")
            last_gpu = now
            with (work / "gpu_samples.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(sample) + "\n")

    journal = work / "closed.jsonl"

    def record(slot, episode):
        nonlocal count, artifact_bytes
        path = work / slot.file
        write_new(path, episode)
        binding = {"file": slot.file, "sha256": sha(path)}
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(binding) + "\n")
        artifact_bytes += path.stat().st_size
        count += 1

    watchdog = threading.Thread(target=hard_watch, daemon=True)
    watchdog.start()
    try:
        with digest_dispatch():
            loop, checkpoint, stage = load_model(args.backend, args.snapshot, args.adapter)
            before = target_digest(loop)
            if before != TARGETS[args.backend]:
                raise RuntimeError("Wrong parameter bytes before first decoder pass")
            quant_before = quant_state_digest(loop)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            engine = DepthEngine(loop, head, chunk=128)
            preflight_ok = False
            failure = None
            try:
                budget(time.monotonic())
                prompt = prepare(next(row for row in rows if row["split"] == "CAL"))
                preflight = engine.preflight(prompt, receipt["max_prompt_tokens"] + CAP,
                                             lambda: budget(time.monotonic()))
                write_new(work / "preflight.json", {"depths": preflight, "stage_check": stage})
                preflight_ok = True
                execute(engine, slots, bindings, tokens=prepare, decode=lambda ids: tok.tok.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    eos_token=receipt["eos_token_id"], budget=budget, record=record)
            except Exception as error:
                failure = {"type": type(error).__name__, "message": str(error)}
            finally:
                engine.close()
                head.close()
                torch.cuda.empty_cache()
            after = target_digest(loop)
            quant_after = quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            write_new(work / "terminal.json", {
                "status": "completed" if count == len(slots) and not failure else "partial",
                "backend": args.backend, "runtime_manifest_sha256": sha(MANIFEST),
                "published_commit": commit, "preflight_passed": preflight_ok, "failure": failure,
                "preflight_sha256": sha(work / "preflight.json") if preflight_ok else None,
                "target_before": before, "target_after": after, "quant_before": quant_before,
                "quant_after": quant_after, "checkpoint_verified_after": True,
                "closed_slots": count, "registered_slots": len(slots),
                "closed_journal_sha256": sha(journal) if journal.is_file() else None,
                "vocabulary_size": engine.vocabulary_size,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
            if failure or before != after or quant_before != quant_after:
                raise RuntimeError("READOUT34 backend did not pass full execution and integrity")
    finally:
        stopped.set()
        watchdog.join()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=tuple(TARGETS), required=True)
    cli.add_argument("--tokenizer-receipt", type=Path, required=True)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--adapter", type=Path, required=True)
    args = cli.parse_args()
    verify_manifest()
    with exclusive_lane():
        predecessors_closed()
        if memory()["available_physical_bytes"] < LIMITS["minimum_start_available_RAM_bytes"]:
            raise RuntimeError("At least 5 GiB available RAM is required before starting the model")
        commit = publication()
        receipt = context_bindings(args.tokenizer_receipt)
        args.snapshot = args.snapshot.resolve()
        if args.snapshot.name != source.CONFIG["checkpoint_revision"]:
            raise RuntimeError("Incorrect checkpoint revision")
        args.adapter = args.adapter.resolve()
        adapter_head = subprocess.check_output(["git", "-C", str(args.adapter), "rev-parse", "HEAD"],
                                               creationflags=FLAGS, timeout=15, text=True).strip()
        if (args.adapter != (REPO.parent / "Compactification-Theory-arg02").resolve()
                or adapter_head != ADAPTER_COMMIT):
            raise RuntimeError("The local adapter differs from its pinned checkout")
        if args.backend == "bf16" and not (ARTIFACT / "nf4" / "admission.json").is_file():
            raise RuntimeError("Admit the NF4 attempt before BF16")
        work = ARTIFACT / args.backend
        work.mkdir(parents=True, exist_ok=False)
        write_new(work / "admission.json", {"backend": args.backend, "runtime_manifest_sha256": sha(MANIFEST),
                                           "published_commit": commit, "pid": os.getpid()})
        environment(work)
        execute_gpu(args, work, receipt, commit)


if __name__ == "__main__":
    main()
