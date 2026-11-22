"""Process-tree limits and telemetry, installed before importing PyTorch."""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

GIB = 1024 ** 3


class TreeBudget:
    def __init__(self, gib: int, threads: int):
        self.limit = gib * GIB
        self.threads = threads
        self.job = None
        self.group = None
        if sys.platform == "win32":
            from lcgrt.ramcap import Job

            self.job = Job(self.limit, priority="normal")
            self.job.assign_self()
        elif sys.platform == "linux":
            relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                            if line.startswith("0::"))
            self.group = Path("/sys/fs/cgroup") / relative.lstrip("/")
            if (self.group == Path("/sys/fs/cgroup")
                    or (self.group / "memory.max").read_text().strip() != str(self.limit)
                    or (self.group / "memory.swap.max").read_text().strip() != "0"):
                raise RuntimeError(f"Run in a dedicated cgroup v2 scope with MemoryMax={gib}G and MemorySwapMax=0")
        else:
            raise RuntimeError("A Windows Job Object or Linux cgroup v2 is required")

        os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads),
                          MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                          LEAN_NUM_THREADS=str(threads))
        import psutil

        self.psutil = psutil
        self.peak_rss = self.peak_private_rss = self.samples = 0
        self.dropped_samples = 0
        self.started = time.monotonic()
        self.cpu_start = self.cpu_seconds()
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def cpu_seconds(self):
        if self.job is not None:
            from lcgrt.ramcap import _check, _k32

            class Accounting(ctypes.Structure):
                _fields_ = [(name, ctypes.c_longlong) for name in
                            ("user", "kernel", "period_user", "period_kernel")] + [
                    (name, ctypes.c_ulong) for name in
                    ("faults", "processes", "active", "terminated")]

            record = Accounting()
            _check(_k32.QueryInformationJobObject(self.job.h, 1, ctypes.byref(record),
                                                  ctypes.sizeof(record), None),
                   "QueryInformationJobObject(accounting)")
            return (record.user + record.kernel) / 1e7
        counters = dict(line.split() for line in (self.group / "cpu.stat").read_text().splitlines())
        return int(counters["usage_usec"]) / 1e6

    def _resident(self):
        pids = (self.job.pids() if self.job is not None else
                (int(pid) for path in self.group.rglob("cgroup.procs")
                 for pid in path.read_text().split()))
        rss, private = 0, 0
        for pid in set(pids):
            try:
                info = self.psutil.Process(pid).memory_full_info()
                rss += info.rss
                private += info.uss
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
            except MemoryError:
                # The full working-set query allocates memory under our own Job
                # cap. Keep OS enforcement and omit the incomplete sample.
                self.dropped_samples += 1
                return None, None
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_private_rss = max(self.peak_private_rss, private)
        self.samples += 1
        return rss, private

    def _monitor(self):
        while not self.stop.wait(0.05):
            self._resident()

    def stats(self):
        rss, private = self._resident()
        elapsed = max(1e-9, time.monotonic() - self.started)
        cpu = self.cpu_seconds() - self.cpu_start
        return {
            "cap_bytes": self.limit, "cpu_threads": self.threads,
            "enforcement": "windows_job_commit" if self.job is not None else "cgroup_v2_memory_max",
            "resident_bytes": rss, "private_resident_bytes": private,
            "sampled_peak_resident_bytes": self.peak_rss,
            "sampled_peak_private_resident_bytes": self.peak_private_rss,
            "resident_samples": self.samples, "sample_period_seconds": 0.05,
            "dropped_resident_samples": self.dropped_samples,
            "peak_committed_bytes": self.job.peak_bytes() if self.job is not None else None,
            "peak_cgroup_charged_bytes": (int((self.group / "memory.peak").read_text())
                                          if self.group is not None else None),
            "wall_seconds": elapsed, "cpu_seconds": cpu, "mean_cores_used": cpu / elapsed,
            "cpu_utilization_percent_of_one_core": 100 * cpu / elapsed,
        }

    def close(self):
        self.stop.set()
        self.monitor.join()
