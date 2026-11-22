"""Fail-closed single-resident-model commissioning for DELAY39."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from experiments.nan_ram28.resources import memory, soft_guard
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import id_hash, sha, tokenizer
from experiments.nan_gym05.common import prompt_ids
from .source import HERE, REPO, digest
from .contrasts import measure_pair
from .instrument import DelayedCapture


SEAL = HERE / "SOURCE_MANIFEST.json"
OLD_PINS = REPO / "experiments/nan_readpath37/registration.json"
ARTIFACT = Path("E:/codex-artifacts/NanbeigeDelay39-20261122")
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HOST_PYTHON = Path("C:/Python312/python.exe")
MAX_BACKEND_SECONDS = 8 * 3600


def verify_publication():
    if not __debug__ or Path.cwd() != REPO:
        raise RuntimeError("Run DELAY39 from its canonical repository with assertions enabled")
    manifest = json.loads(SEAL.read_bytes())
    if manifest["stage"] != "device_pending_after_cpu_verification":
        raise ValueError("DELAY39 runtime registration is incomplete")
    for name, expected in manifest["files_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"DELAY39 frozen dependency drifted: {name}")
    remote = subprocess.check_output(["git", "ls-remote", "--heads", "origin", runtime.REF],
                                     creationflags=FLAGS, timeout=30).split()
    if len(remote) != 2 or remote[1].decode("ascii") != runtime.REF:
        raise RuntimeError("The current publication could not be authenticated")
    commit = remote[0].decode("ascii")
    path = SEAL.relative_to(REPO).as_posix()
    published = subprocess.check_output(["git", "show", f"{commit}:{path}"],
                                        creationflags=FLAGS, timeout=30)
    if published != SEAL.read_bytes():
        raise RuntimeError("Local DELAY39 freeze differs from branch publication")
    return commit


def admissions(args):
    runtime.predecessors_closed()
    if memory()["available_physical_bytes"] < runtime.LIMITS["minimum_start_available_RAM_bytes"]:
        raise RuntimeError("The model needs at least 5 GiB available host RAM")
    if shutil.disk_usage(ARTIFACT.parent).free < runtime.LIMITS["minimum_free_disk_bytes"]:
        raise RuntimeError("The captured-evidence volume is below its free-space floor")
    if args.snapshot.name != json.loads((HERE / "registration.json").read_bytes())["checkpoint_revision"]:
        raise ValueError("Wrong checkpoint revision")
    args.adapter = args.adapter.resolve()
    if args.adapter != (REPO.parent / "Compactification-Theory-arg02").resolve():
        raise ValueError("Wrong oracle-isolated native adapter path")
    actual = subprocess.check_output(["git", "-C", str(args.adapter), "rev-parse", "HEAD"],
                                     creationflags=FLAGS, timeout=15, text=True).strip()
    if actual != runtime.ADAPTER_COMMIT:
        raise RuntimeError("The native adapter pin changed")


@contextmanager
def resources(work):
    stop = threading.Event()
    started = time.monotonic()

    def watch():
        while not stop.wait(2):
            sample = memory()
            if (sample["available_physical_bytes"] < runtime.LIMITS["hard_minimum_available_RAM_bytes"]
                    or sample["private_bytes"] > runtime.LIMITS["hard_maximum_worker_private_bytes"]):
                print(f"DELAY39 self-only hard memory stop: {sample}", flush=True)
                os._exit(3)

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    last_sample = [-600.0]

    def check(*, device=False, force_gpu_sample=False):
        sample = soft_guard(runtime.LIMITS)
        elapsed = time.monotonic() - started
        if elapsed > MAX_BACKEND_SECONDS or shutil.disk_usage(work).free < runtime.LIMITS["minimum_free_disk_bytes"]:
            raise RuntimeError("DELAY39 run resource budget exhausted")
        if device and (force_gpu_sample or elapsed - last_sample[0] >= 600):
            from experiments.nan_cache32.resources import gpu_memory

            gpu = gpu_memory(os.getpid())
            if (gpu["dedicated_bytes"] > runtime.LIMITS["maximum_sampled_dedicated_GPU_bytes"]
                    or gpu["shared_bytes"] > runtime.LIMITS["maximum_sampled_shared_GPU_bytes"]):
                raise RuntimeError("DELAY39 sampled GPU allocation exceeded its limit")
            with (work / "resource_samples.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"elapsed_seconds": elapsed, "host": sample, "device": gpu}) + "\n")
            last_sample[0] = elapsed
    try:
        yield check
    finally:
        stop.set()
        monitor.join()


def load_sources(args):
    pins = json.loads((HERE / "SOURCE_CHECK.json").read_bytes())
    pub = json.loads((args.source_dir / "public.json").read_bytes())
    ids = json.loads((args.source_dir / "tokens.json").read_bytes())
    if digest(pub) != pins["source_digest"] or digest(ids["row_bindings"]) != pins["token_digest"]:
        raise ValueError("Frozen source or prompt binding changed")
    tok, tok_pins = tokenizer(args.snapshot)
    if tok_pins != pins["tokenizer_sha256"]:
        raise ValueError("Pinned tokenizer changed")
    labels = [tok.tok.encode(char, add_special_tokens=False) for char in "ABCDEFGHIJKLMNOP"]
    if any(len(value) != 1 for value in labels) or dict(zip("ABCDEFGHIJKLMNOP", (v[0] for v in labels))) != ids["bare_label_token_ids"]:
        raise ValueError("A bare-label ID changed")
    data = {}
    for row in pub:
        old = ids["row_bindings"][row["id"]]
        new = prompt_ids(tok, args.snapshot, row["prompt"])
        if old != {"ids": new, "sha256": id_hash(new), "tokens": len(new)}:
            raise ValueError("A frozen pointer prompt changed")
        data[row["root"], row["arm"], row["world"]] = (row["id"], new)
    if len(data) != pins["prompts"]:
        raise ValueError("Missing source slot")
    return data, [v[0] for v in labels], tok.tok


def literal(tok, token):
    return tok.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def get_bank(backend, np):
    pins = json.loads(OLD_PINS.read_bytes())["backends"][backend]
    fit = REPO / "runs/nan_fit16_io18" / backend / "t2_layer21_down_proj_rank8"
    blob = (fit / "final.npz").read_bytes()
    from hashlib import sha256
    if sha256(blob).hexdigest() != pins["factor_bank"] or pins["target"] != runtime.TARGETS[backend]:
        raise ValueError("Fitted directions have the wrong target or custody")
    with np.load(BytesIO(blob), allow_pickle=False) as bank:
        V, U = (bank[k].astype(np.float64) for k in ("V", "U"))
    if (V.shape != (10752, 8) or U.shape != (8, 3072)
            or not np.isfinite(V).all() or not np.isfinite(U).all()):
        raise ValueError("Wrong physical shapes or nonfinite fitted directions")
    return V, U


def run_groups(engine, data, labels, tok, args, work, cfg, np, guard):
    results = []
    roots = sorted({key[0] for key in data})
    for root in roots:
        for arm in cfg["arms"]:
            ids = [data[root, arm, world] for world in ("A", "B")]
            guard()
            value = engine.capture([row[1] for row in ids])
            choices = value["logits"][1].argmax(dim=-1).tolist()
            for (name, _), token in zip(ids, choices):
                results.append({"id": name, "token_id": token, "text": literal(tok, token)})
    native_path = work / "native.json"
    gate_path = work / "qualification.json"
    write_new(native_path, results)
    subprocess.run([str(HOST_PYTHON), "-X", "utf8", "-B", "-S", "-m", "experiments.nan_delay39.host_gate",
                    "--source-dir", str(args.source_dir), "--native-records", str(native_path),
                    "--output", str(gate_path)], creationflags=FLAGS, check=True, timeout=30)
    gate = json.loads(gate_path.read_bytes())
    if gate["native_record_digest"] != digest(results):
        raise ValueError("Host qualification is not for this native tranche")
    if not gate["passed"]:
        return {"status": "native_gate_failed", "registered_focused_worlds": 16, "closed_focused_worlds": 0}
    V, U = get_bank(args.backend, np)
    journal = work / "closed.jsonl"
    count = 0
    for root in roots:
        for world, twin in (("A", "B"), ("B", "A")):
            tag, ids = data[root, "focused", world]
            donor = data[root, "focused", twin][1]
            outcome = measure_pair(engine, ids, donor, V, U, labels, cfg, tag, np, guard)
            for response in [outcome["baseline"], *(case["response"] for case in outcome["cases"].values()
                                                    if "response" in case)]:
                response["phase2_decoded"] = literal(tok, response["top_token"][1])
            path = work / f"episode_{count:04d}.json"
            write_new(path, {"source_id": tag, "prompt_sha256": id_hash(ids), **outcome})
            with journal.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"file": path.name, "sha256": sha(path)}) + "\n")
            count += 1
    return {"status": "closed", "registered_focused_worlds": 16, "closed_focused_worlds": count,
            "closed_journal_sha256": sha(journal)}


def execute(args, commit, work):
    cfg = json.loads((HERE / "registration.json").read_bytes())
    if {name: version(name) for name in runtime.GPU_PACKAGES} != runtime.GPU_PACKAGES:
        raise RuntimeError("The historical GPU environment pins changed")
    with resources(work) as check:
        check()
        data, labels, tok = load_sources(args)
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
            raise RuntimeError("DELAY39 requires the registered 4 GiB 3050")
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.reset_peak_memory_stats()
        with digest_dispatch():
            loop, checkpoint, stage = runtime.load_model(args.backend, args.snapshot, args.adapter)
            before, quant_before = target_digest(loop), quant_state_digest(loop)
            if before != runtime.TARGETS[args.backend]:
                raise RuntimeError("Wrong parameter bytes before first decoder forward")
            check(device=True, force_gpu_sample=True)
            head = StreamedVocabularyHead(loop.model.lm_head.weight, loop.device, 8192)
            guard = lambda: check(device=True)
            preflight_engine = BatchedDepthEngine(loop, head)
            first = sorted(data)[0][0]
            proofs = {key: data[first, "focused", key][1] for key in ("A", "B")}
            cap = max(len(v[1]) for v in data.values()) + 2
            try:
                preflight_result = preflight(preflight_engine, [proofs["A"], proofs["B"]], cap,
                                             lambda _started, _batch, _p: guard())
            finally:
                preflight_engine.close()
            write_new(work / "preflight.json", {"status": "passed", "depths": preflight_result, "stage": stage})
            engine = DelayedCapture(loop, head, layer=cfg["layer_zero_based"])
            try:
                result = run_groups(engine, data, labels, tok, args, work, cfg, np, guard)
            finally:
                engine.close()
                head.close()
                torch.cuda.empty_cache()
            guard()
            after, quant_after = target_digest(loop), quant_state_digest(loop)
            verify_checkpoint(args.snapshot, checkpoint)
            write_new(work / "terminal.json", {**result, "backend": args.backend, "target_before": before,
                                               "target_after": after, "quant_unchanged": quant_before == quant_after,
                                               "checkpoint_verified_after": True,
                                               "preflight_sha256": sha(work / "preflight.json"),
                                               "source_manifest_sha256": sha(SEAL), "published_commit": commit,
                                               "CUDA_peak": torch.cuda.max_memory_allocated(),
                                               "CUDA_reserved_peak": torch.cuda.max_memory_reserved()})
            if before != after or quant_before != quant_after:
                raise RuntimeError("Runtime target-byte integrity failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nf4", "bf16"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    with runtime.exclusive_lane():
        admissions(args)
        commit = verify_publication()
        work = ARTIFACT / args.backend
        work.mkdir(exist_ok=False, parents=True)
        write_new(work / "admission.json", {"pid": os.getpid(), "backend": args.backend,
                                            "source_manifest_sha256": sha(SEAL), "published_commit": commit})
        runtime.environment(work)
        try:
            execute(args, commit, work)
        except BaseException as error:
            write_new(work / "attempt_failure.json", {"error_type": type(error).__name__, "message": str(error)})
            raise


if __name__ == "__main__":
    main()
