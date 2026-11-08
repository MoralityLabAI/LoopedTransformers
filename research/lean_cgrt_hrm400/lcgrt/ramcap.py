"""A hard RAM cap and a CPU priority for a whole process tree, through a Windows Job Object.

    job = Job(parse_size("9G"), priority="idle"); job.assign_self()
    # every process started from here on (runners, trainers, evaluators, Lean) is in the job

The limit is on committed memory summed over every process in the job: an allocation that
would take the job past it fails inside the process that asked (torch raises, Lean reports
out of memory), and nothing outside the job is touched. Memory-mapped files (Lean's .olean
imports) are not commit and do not count. The job is closed, and every process still in it
killed, when the process holding the handle exits, so one kill pauses the whole tree.
Children inherit the job; a nested job (the venv launcher makes one) stays under this limit.
Under WDDM a CUDA process's device allocations are charged to its commit too (a trainer with
4.7 GB of VRAM in use committed 7.0 GiB while holding 1.3 GiB of RAM), so a job that uses the
GPU needs its VRAM allowance added to the cap; `resident_bytes` is the RAM actually held.

Windows only. On Linux the equivalent is `systemd-run --user --scope -p MemoryMax=9G ...`.
"""
from __future__ import annotations

import ctypes
import re
import sys

JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
PRIORITY = {"idle": 0x40, "below_normal": 0x4000, "normal": 0x20}


def parse_size(s: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?\s*", str(s), re.I)
    if not m:
        raise ValueError(f"size {s!r}: expected e.g. 9G, 512M")
    return int(float(m[1]) * 1024 ** "BKMGT".index(m[2].upper() or "B"))


if sys.platform == "win32":
    from ctypes import wintypes as wt

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes",
                                                     "OtherBytes")]

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class _Extended(ctypes.Structure):
        _fields_ = [("Basic", _Basic), ("IoInfo", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wt.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    _k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
    _k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    _k32.GetCurrentProcess.restype = wt.HANDLE


def _check(ok, what):
    if not ok:
        raise OSError(ctypes.get_last_error(), f"{what} failed")


class Job:
    def __init__(self, ram_bytes: int, priority: str = "idle"):
        if sys.platform != "win32":
            raise OSError("ramcap needs Windows job objects; on Linux use systemd-run --scope -p MemoryMax=...")
        self.limit = int(ram_bytes)
        self.h = _k32.CreateJobObjectW(None, None)
        _check(self.h, "CreateJobObjectW")
        info = _Extended()
        info.Basic.LimitFlags = (JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                 | JOB_OBJECT_LIMIT_PRIORITY_CLASS)
        info.Basic.PriorityClass = PRIORITY[priority]
        info.JobMemoryLimit = self.limit
        _check(_k32.SetInformationJobObject(self.h, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)), "SetInformationJobObject")

    def assign_self(self):
        _check(_k32.AssignProcessToJobObject(self.h, _k32.GetCurrentProcess()), "AssignProcessToJobObject")

    def peak_bytes(self) -> int:
        """The job's peak commit as Windows records it; a request the cap refused counts in it."""
        info = _Extended()
        _check(_k32.QueryInformationJobObject(self.h, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                              ctypes.sizeof(info), None), "QueryInformationJobObject")
        return int(info.PeakJobMemoryUsed)

    def pids(self) -> list[int]:
        n = 4096                                     # header: two DWORDs (assigned, listed), then ULONG_PTR ids
        buf = ctypes.create_string_buffer(8 + ctypes.sizeof(ctypes.c_size_t) * n)
        _check(_k32.QueryInformationJobObject(self.h, JobObjectBasicProcessIdList, buf, len(buf), None),
               "QueryInformationJobObject")
        listed = int.from_bytes(buf.raw[4:8], "little")
        ids = (ctypes.c_size_t * n).from_buffer_copy(buf.raw[8:])
        return [int(ids[i]) for i in range(listed)]

    def committed_bytes(self) -> int:
        """Current commit of the job's processes (sum of private bytes)."""
        import psutil
        tot = 0
        for pid in self.pids():
            try:
                tot += psutil.Process(pid).memory_info().private
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return tot

    def resident_bytes(self) -> int:
        """RAM the job's processes hold now: the sum of private working sets (Task Manager's Memory
        column). Unlike commit it leaves out what a CUDA process reserves for the card's memory."""
        import psutil
        tot = 0
        for pid in self.pids():
            try:
                tot += psutil.Process(pid).memory_full_info().uss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return tot
