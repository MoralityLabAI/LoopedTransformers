"""Read-only dedicated/shared WDDM samples over one verified model lifetime."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
from experiments.nan_fit07.io import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    handle=k.OpenProcess(0x00100000|0x1000,False,a.pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    def ps(code):return subprocess.check_output(['powershell','-NoProfile','-Command',code],text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    checked=json.loads(ps(f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={a.pid}'; [bool]($p.CommandLine -like '*experiments.nan_prime_ctx10.server*' -and $p.CommandLine -like '*--backend bf16*') | ConvertTo-Json -Compress"));assert checked
    samples=[];started=time.monotonic()
    try:
        while k.WaitForSingleObject(handle,0)==258:
            r=dict(UTC=datetime.now(timezone.utc).isoformat())
            try:
                r['WDDM_process_rows']=json.loads(ps(f"ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{ $_.Name -like 'pid_{a.pid}_*' }} | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)"))
                gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip().split(',')
                r.update(nvidia_device_used_MiB=int(gpu[0]),nvidia_total_MiB=int(gpu[1]),nvidia_utilization_percent=int(gpu[2]))
            except Exception as error:r['sample_error_type']=type(error).__name__
            samples.append(r)
            with (out/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
            status=k.WaitForSingleObject(handle,30000)
            if status==0:break
            if status!=258:raise ctypes.WinError(ctypes.get_last_error())
    finally:k.CloseHandle(handle)
    def bounds(values):return {'minimum':min(values),'maximum':max(values),'samples':len(values)} if values else None
    valid=[r for r in samples if r.get('WDDM_process_rows')]
    result=dict(status='completed',source_sha256=sha(Path(__file__)),sample_period_seconds=30,samples=len(samples),elapsed_seconds=time.monotonic()-started,
        sample_error_count=sum('sample_error_type' in r for r in samples),
        sampled_worker_dedicated_bytes=bounds([sum(v['DedicatedUsage'] for v in r['WDDM_process_rows']) for r in valid]),
        sampled_worker_shared_bytes=bounds([sum(v['SharedUsage'] for v in r['WDDM_process_rows']) for r in valid]),
        sampled_whole_device_used_MiB=bounds([r['nvidia_device_used_MiB'] for r in samples if 'nvidia_device_used_MiB' in r]),
        scope='one verified BF16 model process; intermittent driver counters, not exact peaks or per-tensor residency. Shared usage may include runtime/transfer memory.',
        CUDA_peaks='separate authoritative frozen worker receipt',model_modified=False,sealed_test_accessed=False)
    write(out/'receipt.json',result);print({k:result[k] for k in ('status','samples','sample_error_count')})


if __name__=='__main__':main()
