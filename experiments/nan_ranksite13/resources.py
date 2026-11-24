"""Explicit low CPU priority, including independently invoked workers/audits."""
import os
import json
import subprocess


def below_normal():
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p;k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())


def driver_sample():
    try:
        command=f"ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{ $_.Name -like 'pid_{os.getpid()}_*' }} | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)"
        rows=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW))
        nv=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW).strip().split(',')
        return dict(status='sampled',WDDM_worker_rows=rows,nvidia_device_used_MiB=int(nv[0]),nvidia_total_MiB=int(nv[1]),
            scope='Instantaneous driver counters; shared usage can include runtime/transfer allocations, not a lifetime physical peak/residency certificate')
    except Exception as error:return dict(status='unavailable',error_type=type(error).__name__)
