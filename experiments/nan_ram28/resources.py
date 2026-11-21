"""Small Windows RAM counters; no numerical import or memory trimming."""
import ctypes
from ctypes import wintypes

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in
        ('total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','extended')]
class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
        ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]

def memory(pid=None):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemory),wintypes.DWORD]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):raise ctypes.WinError(ctypes.get_last_error())
    handle=kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters=ProcessMemory();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(counters),counters.cb):raise ctypes.WinError(ctypes.get_last_error())
        return dict(available_physical_bytes=status.available_physical,private_bytes=counters.private,working_set_bytes=counters.working)
    finally:
        if pid is not None:kernel.CloseHandle(handle)

def soft_guard(cfg):
    sample=memory()
    if sample['available_physical_bytes']<cfg['minimum_available_RAM_bytes'] or sample['private_bytes']>cfg['maximum_worker_private_bytes']:
        raise HostMemoryBudgetReached(str(sample))
    return sample

class HostMemoryBudgetReached(RuntimeError):pass
