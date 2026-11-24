"""Finite, lower-priority follow-on queue after actual fitting process exit."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.nan_fit07.io import read,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',required=True,type=int);p.add_argument('--gpu-python',required=True)
    p.add_argument('--cpu-python',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args();repo=Path.cwd().resolve();runs=repo/'runs/nan_ctx08'
    runs.mkdir(exist_ok=True);events=runs/'queue_events.jsonl'
    if events.exists():raise FileExistsError('Preserve and inspect the existing context queue')
    source=repo/'experiments/nan_ctx08/SOURCE_MANIFEST.json';read(source)
    def event(name,**v):
        r={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**v}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    write(runs/'queue_supervisor.json',{'pid':os.getpid(),'after_pid':a.after_pid,'source_manifest_sha256':sha(source)})
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    handle=k.OpenProcess(0x00100000|0x1000,False,a.after_pid)
    if handle:
        try:
            last=0
            while True:
                state=k.WaitForSingleObject(handle,5000)
                if state==0:break
                if state!=258:raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic()-last>=60:event('verified_wait_for_fitting_queue',pid=a.after_pid);last=time.monotonic()
        finally:k.CloseHandle(handle)
    elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_fit07.worker*' -or $_.CommandLine -like '*experiments.nan_prime06.server*' -or $_.CommandLine -like '*experiments.nan_ctx08.worker*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        result=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        if not result or not json.loads(result):break
        event('known_model_still_live',pids=json.loads(result));time.sleep(10)
    env=os.environ.copy();env|={'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};logs=runs/'queue_logs';logs.mkdir(exist_ok=True)
    reports=['reports/nanbeige_bpd_gym/NAN_CTX08_PROGRESS.md','reports/nanbeige_bpd_gym/nan_ctx08_progress.json']
    def refresh():
        with (logs/'CPU_report.log').open('a',encoding='utf-8') as log:
            r=subprocess.run([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_ctx08.report'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode:event('independent_report_failed',exit_code=r.returncode);return
        event('independent_report_refreshed')
        if not a.publish:return
        if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/nanbeige-bpd-gym-suite-towers':raise RuntimeError('Wrong report publication branch')
        for command in (['git','add','--sparse','--',*reports],['git','commit','--only','-m','Stream resident-head long-context development milestone','--',*reports],
                        ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
            with (logs/'git_publication.log').open('a',encoding='utf-8') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        event('aggregate_milestone_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    for backend in ('nf4','bf16'):
        with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as log:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_ctx08.worker','--backend',backend,
                '--snapshot',a.snapshot,'--compactification-root',a.compactification_root],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('resident_head_worker_started',backend=backend,launcher_pid=worker.pid);done=set();last=0
            while True:
                for level in (1024,2048,4096):
                    if level not in done and (runs/backend/f'level_{level}_receipt.json').exists():done.add(level);refresh()
                if worker.poll() is not None:break
                if time.monotonic()-last>=60:event('verified_model_worker_live',backend=backend,pid=worker.pid);last=time.monotonic()
                time.sleep(2)
            refresh();receipt=read(runs/backend/'receipt.json');event('resident_head_worker_terminal',backend=backend,exit_code=worker.returncode,status=receipt['status'])
            if worker.returncode and (receipt.get('failure') or {}).get('type')!='OutOfMemoryError':raise RuntimeError('Preserve the failed condition; do not restart blindly')
    event('bounded_long_context_queue_finished')


if __name__=='__main__':main()
