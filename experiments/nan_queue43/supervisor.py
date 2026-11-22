"""Autonomous local handoff to the published DELAY39 model and host lanes."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from experiments.nan_delay39 import pilot
from experiments.nan_delay39.source import digest
from experiments.nan_ram28.resources import memory
from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34 import runtime
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_storyaudit30.common import Handle

from .job import child_tree


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "LOCAL_MANIFEST.json"
RUN = REPO / "runs/nan_queue43/attempt01"
PYTHON = Path("C:/Python312/python.exe")
GPU_ENV = Path("D:/codex-artifacts/NanbeigeQ4Pilot-20261117")
GPU_PYTHON = GPU_ENV / "venv/Scripts/python.exe"
SNAPSHOT = GPU_ENV / "snapshot/451ed48c3273ecef7ea8faaa43c31ce529763bb1"
ADAPTER = REPO.parent / "Compactification-Theory-arg02"
SOURCES = REPO / "runs/nan_delay39/source"
CONDITIONS = ("nf4", "bf16")
WAIT_SECONDS = 30 * 24 * 3600
WORKER_OBSERVATION_SECONDS = 9 * 3600
HOST_OBSERVATION_SECONDS = 600
POLL_SECONDS = 30
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "RAYON_NUM_THREADS")


def manifest():
    paths = [HERE / name for name in (".gitattributes", "__init__.py", "PROTOCOL.md", "job.py",
                                     "supervisor.py", "verify.py")]
    paths += [REPO / path for path in (
        "experiments/nan_delay39/SOURCE_MANIFEST.json",
        "experiments/nan_cache32/SOURCE_MANIFEST.json",
        "experiments/nan_cache32/inventory.py",
        "experiments/nan_readout34/runtime.py",
        "experiments/nan_storyaudit30/common.py",
    )]
    return {
        "name": "NAN-QUEUE-43", "files_sha256": {path.relative_to(REPO).as_posix(): sha(path) for path in paths},
        "predecessors": [[pid, str(creation)] for pid, creation in runtime.PREDECESSORS],
        "model_order": list(CONDITIONS), "wait_seconds": WAIT_SECONDS,
        "model_observation_seconds": WORKER_OBSERVATION_SECONDS, "host_observation_seconds": HOST_OBSERVATION_SECONDS,
        "startup_available_RAM_bytes": runtime.LIMITS["minimum_start_available_RAM_bytes"],
    }


def verify_local():
    if not __debug__ or Path.cwd() != REPO or Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use the pinned Windows base Python from the canonical checkout")
    if json.loads(MANIFEST.read_bytes()) != manifest():
        raise ValueError("Local supervision manifest changed")
    for path in (GPU_PYTHON, SNAPSHOT / "config.json", ADAPTER / ".git"):
        if not path.exists():
            raise FileNotFoundError(path)
    pins = json.loads((pilot.HERE / "SOURCE_CHECK.json").read_bytes())
    source = json.loads((SOURCES / "public.json").read_bytes())
    inputs = json.loads((SOURCES / "tokens.json").read_bytes())
    oracle = json.loads((SOURCES / "gold.json").read_bytes())
    if (digest(source), digest(inputs["row_bindings"]), digest(oracle)) != (
            pins["source_digest"], pins["token_digest"], pins["gold_digest"]):
        raise ValueError("Private DELAY39 source bindings changed")
    if "torch" in sys.modules:
        raise RuntimeError("The supervisor imported Torch")


@contextmanager
def predecessor_handles():
    opened = []
    observations = []
    try:
        for pid, creation in runtime.PREDECESSORS:
            state = "missing"
            try:
                handle = Handle(pid)
            except OSError as error:
                if error.winerror != 87:
                    raise
            else:
                if handle.creation == creation:
                    state = "live" if handle.alive() else "signaled"
                    opened.append((pid, handle))
                else:
                    handle.close()
                    state = "original_PID_reused"
            observations.append({"pid": pid, "creation": str(creation), "state": state})
        yield opened, observations
    finally:
        for _, handle in opened:
            handle.close()


def journal(event, **fields):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with (RUN / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def wait_predecessors(deadline):
    with predecessor_handles() as (handles, observed):
        journal("authenticated_predecessors", observations=observed, termination_rights=False)
        previous_live = None
        while remaining := [(pid, handle) for pid, handle in handles if handle.alive()]:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("The authenticated predecessor wait expired")
            pids = [pid for pid, _ in remaining]
            if pids != previous_live:
                journal("waiting_predecessors", pids=pids)
                previous_live = pids
            remaining[0][1].alive(max(1, int(min(POLL_SECONDS, deadline - now) * 1000)))
    runtime.predecessors_closed()
    cached = json.loads((runtime.PRIOR_ARTIFACT / "model_lane_terminal.json").read_bytes())
    if cached["source_manifest_sha256"] != sha(REPO / "experiments/nan_cache32/SOURCE_MANIFEST.json"):
        raise ValueError("The predecessor device terminal is for a different registration")
    journal("predecessor_lane_closed", CACHE32_model_closed=True, CACHE32_host_closed=True)


def wait_RAM(stage):
    deadline = time.monotonic() + WAIT_SECONDS
    waiting = False
    while True:
        sample = memory()
        if sample["available_physical_bytes"] >= runtime.LIMITS["minimum_start_available_RAM_bytes"]:
            journal("RAM_admitted", stage=stage, available_bytes=sample["available_physical_bytes"])
            return
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"RAM admission timeout at {stage}")
        if not waiting:
            journal("waiting_RAM", stage=stage, available_bytes=sample["available_physical_bytes"])
            waiting = True
        time.sleep(min(POLL_SECONDS, deadline - now))


def checked_admission(work, backend, expected_manifest):
    value = json.loads((work / "admission.json").read_bytes())
    if (value.get("backend") != backend or value.get("source_manifest_sha256") != expected_manifest
            or type(value.get("pid")) is not int or value["pid"] <= 0):
        raise ValueError("Worker admission has the wrong source, precision or PID")
    return value


def observe_worker(tree, work, backend, lock):
    try:
        admission = checked_admission(work, backend, lock)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    pid = admission["pid"]
    if pid not in tree.active_pids():
        return None
    try:
        handle = Handle(pid)
    except OSError as error:
        if error.winerror != 87:
            raise
        return None
    try:
        if not handle.alive() or pid not in tree.active_pids():
            return None
        return {"pid": pid, "creation": str(handle.creation)}
    finally:
        handle.close()


def run_tree(stage, command, timeout, observe=None):
    deadline = time.monotonic() + timeout
    actual = None
    with child_tree(command, REPO, RUN / f"{stage}.log") as tree:
        journal("owned_job_started", stage=stage, launcher_pid=tree.pid,
                launcher_creation=str(tree.launcher_creation()))
        while True:
            active, total = tree.accounting()
            if not active:
                code = tree.launcher_code()
                journal("owned_job_closed", stage=stage, total_tree_processes=total, exit_code=code, observed_worker=actual)
                if code:
                    raise RuntimeError(f"Owned stage {stage} ended with exit code {code}")
                return actual
            if observe is not None and actual is None:
                actual = observe(tree)
                if actual is not None:
                    journal("authenticated_actual_worker", stage=stage, **actual)
            if time.monotonic() >= deadline:
                journal("observation_expired_with_live_job", stage=stage, active=active)
                raise TimeoutError(f"Owned stage {stage} is still active; sequencing stopped")
            time.sleep(1 if observe is not None and actual is None else 5)


def verify_model(work, backend, source_manifest, actual):
    admission = checked_admission(work, backend, source_manifest)
    value = json.loads((work / "terminal.json").read_bytes())
    status = value.get("status")
    expected = runtime.TARGETS[backend]
    if (value.get("source_manifest_sha256") != source_manifest or value.get("backend") != backend
            or value.get("target_before") != expected or value.get("target_after") != expected
            or value.get("checkpoint_verified_after") is not True or value.get("quant_unchanged") is not True
            or sha(work / "preflight.json") != value.get("preflight_sha256")
            or value.get("published_commit") != admission.get("published_commit")
            or (actual is not None and actual["pid"] != admission["pid"])
            or value.get("registered_focused_worlds") != 16
            or status not in {"closed", "native_gate_failed"}):
        raise ValueError("GPU terminal integrity or stage origin failed")
    if status == "closed":
        if value.get("closed_focused_worlds") != 16 or sha(work / "closed.jsonl") != value.get("closed_journal_sha256"):
            raise ValueError("Closed DELAY39 backend has incomplete or changed graph rows")
    elif value.get("closed_focused_worlds") != 0 or (work / "closed.jsonl").exists():
        raise ValueError("Refused native qualification contains an edited row")
    return status


def launch():
    RUN.mkdir(parents=True, exist_ok=False)
    self_handle = Handle(os.getpid())
    try:
        write_new(RUN / "supervisor.json", {"pid": os.getpid(), "creation": str(self_handle.creation),
                                            "local_manifest_sha256": sha(MANIFEST)})
    finally:
        self_handle.close()
    try:
        journal("queue_started", published_worker=pilot.verify_publication(), local_manifest_sha256=sha(MANIFEST))
        if any((pilot.ARTIFACT / backend).exists() for backend in CONDITIONS):
            raise FileExistsError("An existing DELAY39 attempt needs independent review")
        wait_predecessors(time.monotonic() + WAIT_SECONDS)
        source_manifest = sha(pilot.SEAL)
        outcomes = {}
        for backend in CONDITIONS:
            wait_RAM(backend)
            verify_local()
            runtime.predecessors_closed()
            work = pilot.ARTIFACT / backend
            if work.exists():
                raise FileExistsError(f"An existing DELAY39 {backend} attempt prevents re-admission")
            command = [str(GPU_PYTHON), "-X", "utf8", "-B", "-u", "-m", "experiments.nan_delay39.pilot",
                       "--backend", backend, "--snapshot", str(SNAPSHOT), "--adapter", str(ADAPTER),
                       "--source-dir", str(SOURCES)]
            actual = run_tree(backend, command, WORKER_OBSERVATION_SECONDS,
                              lambda tree: observe_worker(tree, work, backend, source_manifest))
            outcomes[backend] = verify_model(work, backend, source_manifest, actual)
            journal("model_receipt_verified", backend=backend, status=outcomes[backend])
        for backend in CONDITIONS:
            stage = f"score_{backend}"
            wait_RAM(stage)
            runtime.predecessors_closed()
            result = RUN / f"{stage}.json"
            command = [str(PYTHON), "-X", "utf8", "-B", "-u", "-m", "experiments.nan_delay39.graph_host",
                       "--source-dir", str(SOURCES), "--artifact-dir", str(pilot.ARTIFACT / backend),
                       "--output", str(result)]
            run_tree(stage, command, HOST_OBSERVATION_SECONDS)
            value = json.loads(result.read_bytes())
            if value.get("terminal_status") != outcomes[backend] or value.get("backend") != backend:
                raise ValueError("Independent graph scorer did not close the expected backend")
            journal("score_verified", backend=backend, sha256=sha(result))
        write_new(RUN / "terminal.json", {"status": "completed", "model_states": outcomes,
                                          "local_manifest_sha256": sha(MANIFEST)})
        journal("queue_closed", model_states=outcomes)
    except Exception as error:
        journal("queue_failed", error_type=type(error).__name__, reason=str(error))
        write_new(RUN / "terminal.json", {"status": "stopped", "error_type": type(error).__name__,
                                          "local_manifest_sha256": sha(MANIFEST)})
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_mutually_exclusive_group()
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--observe", action="store_true")
    args = cli.parse_args()
    below_normal()
    for name in THREADS:
        os.environ[name] = "1"
    if args.freeze:
        write_new(MANIFEST, manifest())
        print(sha(MANIFEST))
        return
    verify_local()
    if args.observe:
        published = pilot.verify_publication()
        with predecessor_handles() as (_, observed):
            print(json.dumps({"published_worker": published, "predecessors": observed,
                              "artifact_backends_present": [b for b in CONDITIONS if (pilot.ARTIFACT / b).exists()],
                              "local_manifest_sha256": sha(MANIFEST), "available_RAM_bytes":
                              memory()["available_physical_bytes"], "torch_imported": "torch" in sys.modules}))
        return
    launch()


if __name__ == "__main__":
    main()
