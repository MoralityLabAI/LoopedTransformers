"""MOLT-11 Linux isolated-process supervisor; no torch import in parent.

Hard per-process virtual address-space and CPU-time limits plus a sampled
RSS guard and wall deadline. The RSS guard is not a kernel RAM reservation.
The worker contract forbids children; any observed child aborts the phase.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKER = "research_gym.interp.aux_gym_v2_isolated_worker"
ARMS = ("dense", "decode", "gram", "recompute")
MIB = 1024**2


def status(pid):
    result = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "VmSize"}:
                    result[key] = int(value.split()[0]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return {}
    return result


def children(pid):
    found = set()
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except FileNotFoundError:
        return found
    for task in tasks:
        try:
            found.update(int(x) for x in (task / "children").read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def phase(command, log, *, address_mib, rss_mib, wall_seconds, cpu_seconds):
    """Start a single subprocess with its own resource ledger."""
    limits = {"address_space_bytes": address_mib * MIB, "sampled_rss_guard_bytes": rss_mib * MIB,
              "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds, "poll_seconds": 0.025}

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (limits["address_space_bytes"],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    sampled_peak, sampled_hwm, virtual_peak, samples = 0, 0, 0, 0
    failure = None
    start = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, preexec_fn=limit)
        try:
            while proc.poll() is None:
                snap = status(proc.pid)
                if snap:
                    samples += 1
                    sampled_peak = max(sampled_peak, snap.get("VmRSS", 0))
                    sampled_hwm = max(sampled_hwm, snap.get("VmHWM", 0))
                    virtual_peak = max(virtual_peak, snap.get("VmSize", 0))
                if children(proc.pid):
                    failure = "unexpected_child_process"
                elif sampled_peak > limits["sampled_rss_guard_bytes"] or sampled_hwm > limits["sampled_rss_guard_bytes"]:
                    failure = "rss_guard_exceeded"
                elif time.monotonic() - start > wall_seconds:
                    failure = "wall_timeout"
                if failure:
                    kill_group(proc)
                    break
                time.sleep(limits["poll_seconds"])
            rc = proc.wait()
        except BaseException:
            kill_group(proc)
            proc.wait()
            raise
        # Disallow a surviving group even when the worker has exited.
        kill_group(proc)
    return {"status": failure or ("success" if rc == 0 else "worker_failed"),
            "returncode": rc, "limits": limits, "elapsed_seconds": time.monotonic() - start,
            "sample_count": samples, "sampled_peak_rss_bytes": sampled_peak,
            "sampled_kernel_hwm_bytes": sampled_hwm, "sampled_peak_virtual_bytes": virtual_peak,
            "log": log.name, "single_process_only": True,
            "hard_address_space_cap": True, "hard_process_tree_rss_cap": False}


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("isolated resource supervisor currently requires Linux")
    if args.out.exists():
        raise FileExistsError(args.out)
    if (not 512 <= args.rss_mib <= args.address_mib <= 32768
            or not 1 <= args.cpu_seconds <= 1800 or not 1 <= args.wall_seconds <= 1800):
        raise ValueError("invalid resource caps")
    args.out.mkdir(parents=True, exist_ok=False)
    logs = args.out / "logs"
    logs.mkdir()
    common = args.out / "common"
    arms_dir = args.out / "arms"
    arms_dir.mkdir()
    phases = {}
    report = {"schema": "molt11-isolation-v1", "status": "incomplete", "phases": phases,
              "hard_process_tree_rss_cap": False, "training_effect_claim_authorized": False,
              "promotion_authorized": False}
    report_path = args.out / "report.json"

    def persist():
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def launch(name, kind, path, extra=()):
        command = [sys.executable, "-m", WORKER, kind,
                   "--data", str(args.data), "--run", str(args.run), "--factors", str(args.factors),
                   "--common", str(common), "--arms-dir", str(arms_dir), "--out", str(path),
                   "--visit", str(args.visit), "--anchors", str(args.anchors), "--donors", str(args.donors),
                   "--batch", str(args.batch), "--updates", str(args.updates), "--seed", str(args.seed),
                   "--aux-weight", str(args.aux_weight), *extra]
        receipt = phase(command, logs / f"{name}.log", address_mib=args.address_mib,
                        rss_mib=args.rss_mib, wall_seconds=args.wall_seconds, cpu_seconds=args.cpu_seconds)
        phases[name] = receipt
        if receipt["status"] == "success":
            raw = json.loads((path / "receipt.json").read_text())
            peak = raw.get("lifetime_peak_at_receipt_bytes", raw.get("peak_rss_bytes"))
            receipt["worker_kernel_peak_rss_bytes"] = peak
            if peak is not None and peak > args.rss_mib * MIB:
                receipt["status"] = "worker_reported_rss_guard_exceeded"
            receipt["worker_receipt"] = str(path.relative_to(args.out) / "receipt.json")
        persist()
        if receipt["status"] != "success":
            report["status"] = "failed"
            persist()
            raise RuntimeError(f"{name}: {receipt['status']}; see {logs / (name + '.log')}")
        return raw

    try:
        common_receipt = launch("common", "common", common)
        report["teacher_validation"] = common_receipt["teacher_validation"]
        report["competence_gate_passed"] = common_receipt["competence_gate_passed"]
        persist()
        if args.require_competent and not report["competence_gate_passed"]:
            report["status"] = "competence_gate_failed"
            persist()
            return report
        target_dirs = {kind: args.out / f"target-{kind}" for kind in ("dense", "factor")}
        target_receipts = {}
        for kind, path in target_dirs.items():
            target_receipts[kind] = launch(f"prepare-{kind}", "targets", path, ("--arm", kind))
        # The preparation archives coexist on disk, never in a training worker.
        # The order is fixed and deliberately reported; timings are descriptive.
        receipts = []
        for arm in ARMS:
            extra = ["--arm", arm]
            if arm != "recompute":
                extra += ["--targets-dir", str(target_dirs["dense" if arm == "dense" else "factor"])]
            receipts.append(launch(f"train-{arm}", "train", arms_dir / arm, extra))
        audit = launch("audit", "audit", args.out / "audit")
        report["status"] = "engineering_pass"
        report["common"] = common_receipt
        report["target_archives"] = target_receipts
        report["arms"] = receipts
        report["audit"] = audit
        persist()
        print(json.dumps({
            "status": report["status"], "records": common_receipt["records"],
            "teacher_exact": common_receipt["teacher_validation"]["exact"],
            "constant_exact": common_receipt["teacher_validation"]["constant_exact"],
            "competence_gate_passed": report["competence_gate_passed"],
            "common_bytes": common_receipt["archive"]["tensor_payload_bytes"],
            "targets": {k: {"response_bytes": v["response_bytes"], "head_bytes": v["head_bytes"],
                            "file_bytes": v["archive"]["file_bytes"], "peak_rss_bytes": v["peak_rss_bytes"]}
                        for k, v in target_receipts.items()},
            "arms": [{k: r[k] for k in ("arm", "target_tensor_bytes", "extra_gram_bytes",
                                        "recompute_teacher_parameter_bytes", "recompute_centered_head_bytes",
                                        "rss_after_load_bytes", "lifetime_peak_after_training_bytes",
                                        "lifetime_peak_at_receipt_bytes", "training_wall_seconds",
                                        "training_cpu_seconds")} for r in receipts],
            "audit": audit["arms"], "hard_process_tree_rss_cap": False,
        }, sort_keys=True))
        return report
    except BaseException:
        report["status"] = "failed"
        persist()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "factors", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--visit", type=int, default=4)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--donors", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026112711)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--address-mib", type=int, default=8192)
    p.add_argument("--rss-mib", type=int, default=2048)
    p.add_argument("--cpu-seconds", type=int, default=180)
    p.add_argument("--wall-seconds", type=int, default=240)
    p.add_argument("--require-competent", action="store_true")
    args = p.parse_args()
    result = run(args)
    if result["status"] == "competence_gate_failed":
        raise SystemExit("teacher competence gate failed; no continuation run")


if __name__ == "__main__":
    main()
