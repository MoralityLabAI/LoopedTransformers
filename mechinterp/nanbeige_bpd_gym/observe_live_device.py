"""Finite read-only device observation bound to an authenticated model handle."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

def observe(a):
    from experiments.nan_ranksite13.resources import below_normal
    from experiments.nan_full26.common import write,sha
    below_normal();assert 0<a.seconds<=300 and a.interval>=5
    flags=0x08000000|0x4000
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.pid);assert handle
    query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={a.pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m experiments.nan_ctx21.worker *') {{ exit 3 }}"
    assert subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags).returncode==0
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    def count():
        counts={4096:0,8192:0}
        with Path('runs/nan_ctx21/nf4/events.jsonl').open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):continue
                r=json.loads(line)
                if r['event']=='case_closed':counts[r['context_budget']]+=1
        return counts
    start=time.monotonic();initial=count();samples=[];errors=[];terminal=False
    try:
        for index in range(int(a.seconds/a.interval)+1):
            delay=start+index*a.interval-time.monotonic()
            if delay>0:time.sleep(delay)
            state=kernel.WaitForSingleObject(handle,0);assert state in (0,258)
            if state==0:terminal=True;break
            row=dict(UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,worker_actual_handle_live=True)
            try:
                fields=['utilization.gpu','memory.used','memory.total','temperature.gpu','clocks.current.graphics']
                raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True,timeout=10,creationflags=flags)
                row.update(dict(zip(('GPU_percent','device_used_MiB','device_total_MiB','temperature_C','graphics_MHz'),[float(x.strip()) for x in raw.split(',')])))
                row['completed_contexts_by_budget']=count();samples.append(row)
                with (out/'samples.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            except Exception as e:errors.append(dict(elapsed_seconds=time.monotonic()-start,error_type=type(e).__name__,message=str(e)))
            if index%12==0:print(dict(event='live_device_sample',samples=len(samples),elapsed_seconds=time.monotonic()-start),flush=True)
    finally:kernel.CloseHandle(handle)
    assert samples,'No usable device sample'
    final=count();elapsed=time.monotonic()-start
    stat=lambda key:dict(minimum=min(r[key] for r in samples),median=statistics.median(r[key] for r in samples),mean=statistics.mean(r[key] for r in samples),maximum=max(r[key] for r in samples))
    result=dict(status='closed_observation',worker_pid=a.pid,observation_seconds=elapsed,samples=len(samples),errors=errors,
        worker_terminal_during_observation=terminal,initial_completed_contexts=initial,final_completed_contexts=final,
        new_closed_contexts=sum(final.values())-sum(initial.values()),
        closed_contexts_per_minute=(sum(final.values())-sum(initial.values()))*60/elapsed,
        GPU_utilization_percent=stat('GPU_percent'),device_memory_used_MiB=stat('device_used_MiB'),
        temperature_C=stat('temperature_C'),graphics_clock_MHz=stat('graphics_MHz'),
        scope='Device-wide NVML samples while the authenticated CTX21 NF4 process handle stayed live; not per-process kernel occupancy or full-run peak VRAM.',
        neural_calls_added=0,grades_added=0,inference_or_frozen_methods_changed=False,CPU_threads=1,priority='BelowNormal',sealed_test_accessed=False,
        sample_file_sha256=sha(out/'samples.jsonl'),observer_source_sha256=sha(__file__))
    write(out/'receipt.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);p.add_argument('--seconds',default=180,type=int)
    p.add_argument('--interval',default=5,type=int);p.add_argument('--out',required=True);observe(p.parse_args())
