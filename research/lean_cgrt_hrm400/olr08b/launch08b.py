"""OLR-08B CPU-first resource-limited launcher and authenticated audit archive packager.

Enforces:
  - CPU-only execution (CUDA_VISIBLE_DEVICES='')
  - 8 threads, 2 native workers (configurable)
  - Hard 12 GiB process-tree RAM cap (Windows Job Object / Linux cgroup memory.max)
  - OS-accounted Job CPU seconds + peak job commit + peak sampled process-tree RSS
  - Authenticated olr08b-<tag>_AUDIT.zip and olr08b-<tag>_FABLE.zip with SHA-256 audit_manifest.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import zipfile

HOME = Path(__file__).resolve().parent
OLR08_DIR = HOME.parent / "olr08"
for p in (str(HOME), str(OLR08_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def limit_memory_and_accounting(gib: float):
    """Apply hard process-tree RAM limit and return (peak_commit_fn, cpu_seconds_fn, desc)."""
    n = int(gib * (1024**3))
    if not n:
        return None, None, "explicit uncapped smoke"
    if sys.platform == "win32":
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [
                ("user", ctypes.c_longlong),
                ("job", ctypes.c_longlong),
                ("flags", w.DWORD),
                ("min", ctypes.c_size_t),
                ("max", ctypes.c_size_t),
                ("active", w.DWORD),
                ("aff", ctypes.c_size_t),
                ("pri", w.DWORD),
                ("sched", w.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [(x, ctypes.c_ulonglong) for x in ("read", "write", "other", "rb", "wb", "ob")]

        class Ext(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", IO),
                ("proc", ctypes.c_size_t),
                ("job", ctypes.c_size_t),
                ("peakproc", ctypes.c_size_t),
                ("peakjob", ctypes.c_size_t),
            ]

        class Acct(ctypes.Structure):
            _fields_ = [
                ("total_user", ctypes.c_longlong),
                ("total_kernel", ctypes.c_longlong),
                ("period_user", ctypes.c_longlong),
                ("period_kernel", ctypes.c_longlong),
                ("page_faults", w.DWORD),
                ("total_procs", w.DWORD),
                ("active_procs", w.DWORD),
                ("terminated_procs", w.DWORD),
            ]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = w.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        k.GetCurrentProcess.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]

        h = k.CreateJobObjectW(None, None)
        v = Ext()
        v.basic.flags = 0x200 | 0x2000  # JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE
        v.job = n
        if (
            not h
            or not k.SetInformationJobObject(h, 9, ctypes.byref(v), ctypes.sizeof(v))
            or not k.AssignProcessToJobObject(h, k.GetCurrentProcess())
        ):
            raise OSError(ctypes.get_last_error(), "Windows hard memory job assignment failed")

        def peak_commit() -> int:
            x = Ext()
            if not k.QueryInformationJobObject(h, 9, ctypes.byref(x), ctypes.sizeof(x), None):
                raise OSError("job memory query failed")
            return int(x.peakjob)

        def job_cpu_seconds() -> float:
            ac = Acct()
            if not k.QueryInformationJobObject(h, 1, ctypes.byref(ac), ctypes.sizeof(ac), None):
                return 0.0
            # Windows FILETIME ticks = 100 nanoseconds
            return float((ac.total_user + ac.total_kernel) * 1e-7)

        return peak_commit, job_cpu_seconds, "Windows Job Object hard process-tree commit limit"

    if sys.platform.startswith("linux"):
        rel = next(
            (s.split("::", 1)[1] for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::")),
            None,
        )
        if rel is not None:
            d = Path("/sys/fs/cgroup") / rel.lstrip("/")
            for q in (d, *d.parents):
                p = q / "memory.max"
                if p.exists():
                    v = p.read_text().strip()
                    if v != "max" and int(v) <= n:
                        return None, None, "verified cgroup memory.max=" + v
        raise SystemExit(f"Run inside systemd-run --user --scope -p MemoryMax={gib}G ...")
    raise SystemExit("Hard cap unsupported on this platform; uncapped smoke only")


def package_archives(out: Path, reissue_a: Path | None, reissue_b: Path | None) -> dict[str, Any]:
    """Package authenticated AUDIT and compact FABLE archives with SHA-256 audit_manifest.json."""
    import run08

    code = out / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(exist_ok=True)
    for src_root, sub in ((HOME, "olr08b"), (OLR08_DIR, "olr08")):
        for p in src_root.rglob("*"):
            if (
                p.is_file()
                and (
                    p.suffix in (".py", ".md", ".lean", ".toml", ".patch")
                    or p.name in ("lean-toolchain", "spent.json", "train_registry.json", "package_manifest.json")
                )
                and ".lake" not in p.parts
                and "__pycache__" not in p.parts
            ):
                dst = code / sub / p.relative_to(src_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)

    for k, p in (("07a", reissue_a), ("07b", reissue_b)):
        if p and p.is_file():
            shutil.copy2(p, out / f"reissue_{k}_manifest.json")

    files = {
        p.relative_to(out).as_posix(): run08.sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "audit_manifest.json" and not p.name.endswith(".zip")
    }
    run08.write(out / "audit_manifest.json", {"schema": "olr08b.audit_manifest.v1", "files": files})
    manifest_sha = run08.sha(out / "audit_manifest.json")

    audit_zip = out.with_name(out.name + "_AUDIT.zip")
    fable_zip = out.with_name(out.name + "_FABLE.zip")
    fable_keep = {
        "summary.json",
        "contract.json",
        "registry_census.json",
        "roster_audit.json",
        "resource.json",
        "frozen_policy.json",
        "prior.json",
        "registry.json",
        "registry.sha256",
    }
    fable_names = [n for n in files if n.startswith(("code/", "reissue_", "sentinel/")) or n in fable_keep]
    info: dict[str, Any] = {"audit_manifest_sha256": manifest_sha, "listed_files": len(files)}
    for label, zpath, names in (
        ("audit_zip", audit_zip, [*files, "audit_manifest.json"]),
        ("fable_zip", fable_zip, [*fable_names, "audit_manifest.json"]),
    ):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                z.write(out / n, n)
        zsha = run08.sha(zpath)
        info[label] = {"path": str(zpath), "bytes": zpath.stat().st_size, "sha256": zsha}
        print(f"PACKAGED {label} {zpath} {zpath.stat().st_size} {zsha}", flush=True)
    return info


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ram-gib", type=float, default=12.0)
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    from run08b import cli, main as run_main

    args = cli()
    if not 1 <= args.workers <= args.threads <= 64 or a.ram_gib < 0 or (args.profile == "full" and a.ram_gib == 0):
        raise SystemExit("invalid CPU/memory configuration")

    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        LEAN_NUM_THREADS=str(max(1, args.threads // args.workers)),
    )
    peak_fn, cpu_fn, kind = limit_memory_and_accounting(a.ram_gib)

    import psutil
    import run08

    proc = psutil.Process()
    if hasattr(proc, "cpu_affinity"):
        proc.cpu_affinity(proc.cpu_affinity()[: args.threads])

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {
        "schema": "olr08b.resource.v1",
        "os": platform.platform(),
        "python": sys.version,
        "memory_limit": kind,
        "requested_gib": a.ram_gib,
        "threads": args.threads,
        "workers": args.workers,
        "peak_sampled_tree_rss": 0,
        "peak_process_count": 1,
        "completed": False,
        "cpu_only": True,
    }
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            try:
                tree = [proc, *proc.children(recursive=True)]
                rss = sum(q.memory_info().rss for q in tree)
                stats["peak_sampled_tree_rss"] = max(int(stats["peak_sampled_tree_rss"]), int(rss))
                stats["peak_process_count"] = max(int(stats["peak_process_count"]), len(tree))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            done.wait(0.5)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    start = time.perf_counter()
    try:
        run_main(args)
        stats["completed"] = True
    except BaseException as e:
        stats["error"] = repr(e)
        raise
    finally:
        done.set()
        t.join()
        stats["seconds_this_invocation"] = round(time.perf_counter() - start, 4)
        if peak_fn:
            stats["peak_job_commit"] = peak_fn()
        if cpu_fn:
            stats["job_cpu_seconds"] = round(cpu_fn(), 4)
        run08.write(out / "resource.json", stats)

    if not args.stop_after_freeze:
        pkg = package_archives(out, args.reissue_a, args.reissue_b)
        run08.write(out / "archive_receipt.json", pkg)


if __name__ == "__main__":
    main()
