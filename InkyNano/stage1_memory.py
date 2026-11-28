"""Working-set, system headroom and CUDA allocator safeguards for the routing gym."""
import ctypes
import os
import sys

MIB = 1 << 20
GIB = 1 << 30


class MemoryPressure(RuntimeError):
    pass


def process_memory():
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (field, ctypes.c_size_t) for field in (
                    "peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")
            ]

        counters = Counters()
        counters.size = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(kernel.GetCurrentProcess()), ctypes.byref(counters), counters.size
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.working, counters.peak_working
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="ascii") as status:
            sizes = {row.split(":")[0]: int(row.split()[1]) * 1024
                     for row in status if row.startswith(("VmRSS:", "VmHWM:"))}
        return sizes["VmRSS"], sizes["VmHWM"]
    raise OSError("resource monitoring supports Windows and Linux")


def available_memory():
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (field, ctypes.c_ulonglong) for field in (
                    "total", "available", "total_page", "available_page", "total_virtual",
                    "available_virtual", "reserved")
            ]

        status = Status()
        status.size = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.available
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo", encoding="ascii") as info:
            return next(int(row.split()[1]) * 1024 for row in info if row.startswith("MemAvailable:"))
    raise OSError("resource monitoring supports Windows and Linux")


class MemoryGuard:
    def __init__(self):
        self.baseline, self.peak = process_memory()
        self.system_available_start = available_memory()
        self.system_available_min = self.system_available_start
        self.cuda = None
        self.device = None
        self.cuda_limit = 0
        self.cuda_free_start = None
        self.cuda_allocated_peak = self.cuda_reserved_peak = 0

    def configure_cuda(self, torch, device):
        cuda = torch.cuda
        if not cuda.is_available():
            raise RuntimeError("CUDA requires an existing CUDA-enabled PyTorch runtime and GPU")
        index = cuda.current_device() if device.index is None else device.index
        free, total = cuda.mem_get_info(index)
        # Leave space for the CUDA context and other users of the card.
        limit = min(1536 * MIB, free - 256 * MIB)
        if limit < 256 * MIB:
            raise MemoryPressure(f"only {free // MIB} MiB CUDA memory free")
        cuda.set_per_process_memory_fraction(limit / total, index)
        self.cuda, self.device, self.cuda_limit = cuda, index, limit
        self.cuda_free_start = free

    def check(self):
        _, peak = process_memory()
        self.peak = max(self.peak, peak)
        self.system_available_min = min(self.system_available_min, available_memory())
        if self.peak - self.baseline >= 2560 * MIB or self.system_available_min < GIB:
            raise MemoryPressure("host memory headroom crossed the routing gym stop threshold")
        if self.cuda and self.cuda.max_memory_reserved(self.device) > self.cuda_limit:
            raise MemoryPressure("CUDA reservation crossed the routing gym stop threshold")

    def receipt(self):
        _, peak = process_memory()
        self.peak = max(self.peak, peak)
        allocated = self.cuda.max_memory_allocated(self.device) if self.cuda else 0
        reserved = self.cuda.max_memory_reserved(self.device) if self.cuda else 0
        self.cuda_allocated_peak = max(self.cuda_allocated_peak, allocated)
        self.cuda_reserved_peak = max(self.cuda_reserved_peak, reserved)
        return {
            "pid": os.getpid(),
            "baseline_rss_bytes": self.baseline,
            "process_peak_rss_bytes": self.peak,
            "incremental_peak_rss_bytes": max(0, self.peak - self.baseline),
            "system_available_start_bytes": self.system_available_start,
            "system_available_min_sampled_bytes": self.system_available_min,
            "cuda_free_start_bytes": self.cuda_free_start,
            "cuda_allocator_limit_bytes": self.cuda_limit,
            "peak_cuda_allocated_bytes": allocated,
            "peak_cuda_reserved_bytes": reserved,
            "run_peak_cuda_allocated_bytes": self.cuda_allocated_peak,
            "run_peak_cuda_reserved_bytes": self.cuda_reserved_peak,
        }
