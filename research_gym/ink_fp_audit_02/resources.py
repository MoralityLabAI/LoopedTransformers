"""Local resource guard: zero GPU, Windows Job Object RAM cap, DRAM hysteresis governor.

Import this module before any ML library. It sets CUDA_VISIBLE_DEVICES=-1 and caps
BLAS/OpenMP threads at import time.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass, field

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class Limits:
    job_memory_cap: int = 4 * GIB
    target_low: int = 512 * MIB
    target_high: int = 2 * GIB
    startup_min_available: int = 6 * GIB
    pause_below: int = 3 * GIB
    resume_above: int = 5 * GIB
    resume_hold_seconds: float = 60.0
    emergency_below: int = int(1.5 * GIB)
    max_threads: int = 4
    max_concurrency: int = 1


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_memory() -> int:
    """Available physical memory in bytes (GlobalMemoryStatusEx on Windows)."""
    if sys.platform == "win32":
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(st.ullAvailPhys)
    import psutil  # non-Windows fallback (CI)
    return int(psutil.virtual_memory().available)


def process_tree_rss(pid: int | None = None) -> int:
    import psutil
    root = psutil.Process(pid or os.getpid())
    total = 0
    for p in [root, *root.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def gpu_allocation_bytes() -> int:
    """Local GPU bytes allocated by this process: zero unless torch.cuda was initialised.

    Never imports torch; reads it only if something else already did.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


# ---- Windows Job Object -------------------------------------------------------------
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_ulong), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong),
                ("SchedulingClass", ctypes.c_ulong)]


class _EXT_LIMIT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JobObject:
    """Committed-memory cap for a whole process tree, plus below-normal priority.

    Children created after assignment inherit membership, so the cap covers the tree.
    """

    def __init__(self, memory_cap: int, below_normal: bool = True):
        if sys.platform != "win32":
            raise OSError("Job Objects require Windows")
        k = ctypes.windll.kernel32
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.OpenProcess.restype = ctypes.c_void_p
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_ulong, ctypes.c_void_p]
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        self._k = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError()
        info = _EXT_LIMIT()
        flags = JOB_OBJECT_LIMIT_JOB_MEMORY
        if below_normal:
            flags |= JOB_OBJECT_LIMIT_PRIORITY_CLASS
            info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
        info.BasicLimitInformation.LimitFlags = flags
        info.JobMemoryLimit = int(memory_cap)
        if not k.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                         ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError()
        self.memory_cap = int(memory_cap)

    def assign_pid(self, pid: int) -> None:
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        h = self._k.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            raise ctypes.WinError()
        try:
            if not self._k.AssignProcessToJobObject(self.handle, h):
                raise ctypes.WinError()
        finally:
            self._k.CloseHandle(h)

    def assign_current(self) -> None:
        if not self._k.AssignProcessToJobObject(self.handle, self._k.GetCurrentProcess()):
            raise ctypes.WinError()

    def peak_job_memory(self) -> int:
        info = _EXT_LIMIT()
        if not self._k.QueryInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                                 ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError()
        return int(info.PeakJobMemoryUsed)

    def close(self) -> None:
        if self.handle:
            self._k.CloseHandle(self.handle)
            self.handle = None


def set_below_normal_priority() -> None:
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(5)


# ---- DRAM governor -------------------------------------------------------------------
RUN, PAUSE, EMERGENCY = "run", "pause", "emergency"


@dataclass
class Governor:
    """Hysteresis state machine on available DRAM. Pure: readings and clock are injected."""

    limits: Limits = field(default_factory=Limits)
    state: str = RUN
    _above_since: float | None = None
    min_available: int | None = None
    transitions: list = field(default_factory=list)

    def observe(self, available: int, now: float) -> str:
        L = self.limits
        self.min_available = available if self.min_available is None else min(self.min_available, available)
        prev = self.state
        if available < L.emergency_below:
            self.state = EMERGENCY
        elif self.state == EMERGENCY:
            pass  # sticky: emergency means checkpoint and exit
        elif available < L.pause_below:
            self.state, self._above_since = PAUSE, None
        elif self.state == PAUSE:
            if available > L.resume_above:
                self._above_since = now if self._above_since is None else self._above_since
                if now - self._above_since >= L.resume_hold_seconds:
                    self.state, self._above_since = RUN, None
            else:
                self._above_since = None
        if self.state != prev:
            self.transitions.append({"t": now, "from": prev, "to": self.state, "available": available})
        return self.state


def startup_check(limits: Limits = Limits(), available: int | None = None) -> dict:
    avail = available_memory() if available is None else available
    return {"available_bytes": avail, "required_bytes": limits.startup_min_available,
            "ok": avail >= limits.startup_min_available,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


class EmergencyStop(RuntimeError):
    pass


@dataclass
class ResourceMonitor:
    """Wraps the governor with real readings and accounting for receipts."""

    limits: Limits = field(default_factory=Limits)
    reader: object = available_memory
    sleeper: object = time.sleep
    clock: object = time.monotonic
    poll_seconds: float = 5.0

    def __post_init__(self):
        self.governor = Governor(self.limits)
        self.t0_wall = time.time()
        self.t0 = self.clock()
        self.peak_tree_rss = 0
        self.api_requests = 0
        self.train_tokens = 0
        self.sample_tokens = 0

    def sample_rss(self) -> None:
        try:
            self.peak_tree_rss = max(self.peak_tree_rss, process_tree_rss())
        except Exception:
            pass

    def gate(self) -> None:
        """Block while paused; raise EmergencyStop below the emergency floor."""
        while True:
            self.sample_rss()
            state = self.governor.observe(self.reader(), self.clock())
            if state == RUN:
                return
            if state == EMERGENCY:
                raise EmergencyStop("available DRAM below emergency floor")
            self.sleeper(self.poll_seconds)

    def receipt(self) -> dict:
        try:
            import psutil
            cpu = psutil.Process().cpu_times()
            cpu_seconds = cpu.user + cpu.system
        except Exception:
            cpu_seconds = None
        return {"peak_process_tree_rss_bytes": self.peak_tree_rss,
                "min_available_bytes": self.governor.min_available,
                "cpu_seconds": cpu_seconds, "wall_seconds": self.clock() - self.t0,
                "gpu_allocation_bytes": gpu_allocation_bytes(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "api_requests": self.api_requests, "remote_sample_tokens": self.sample_tokens,
                "remote_train_tokens": self.train_tokens,
                "governor_transitions": self.governor.transitions}
