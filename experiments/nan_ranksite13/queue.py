"""Finite six-cell rank/site queue after the actual KV12 supervisor exits."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import read,sha,write,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);p.add_argument('--gpu-python',required=True);p.add_argument('--cpu-python',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    root=Path(__file__).parent;verify();base=Path('runs/nan_ranksite13');base.mkdir(parents=True,exist_ok=True);events=base/'queue_events.jsonl'
    if events.exists():raise FileExistsError('Preserve the existing registered rank/site run')
    logs=base/'queue_logs';logs.mkdir()
    def event(name,**fields):
        r=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(r,flush=True)
    write(base/'supervisor.json',dict(pid=os.getpid(),after_pid=a.after_pid,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json')))
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
                if time.monotonic()-last>=60:event('verified_wait_for_KV12_supervisor',pid=a.after_pid);last=time.monotonic()
        finally:k.CloseHandle(handle)
    elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_prime_ctx10.server*' -or ($_.CommandLine -like '*experiments.nan_ranksite13.worker*' -or ($_.CommandLine -like '*experiments.nan_kv11.worker*' -or $_.CommandLine -like '*experiments.nan_kv12.worker*')) -or $_.CommandLine -like '*experiments.nan_recomp09.worker*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        if not raw or not json.loads(raw):break
        event('known_model_still_live',pids=json.loads(raw));time.sleep(5)
    env=os.environ.copy();env.update({v:'1' for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE')})
    paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_RANKSITE13_PROGRESS.md','nan_ranksite13_progress.json')]
    def refresh(*,with_probes=False):
        with (logs/'independent_report.log').open('a',encoding='utf-8') as f:
            code=subprocess.run([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_ranksite13.report']+(['--with-probes'] if with_probes else []),env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
        if code:event('independent_report_held_back',exit_code=code);return
        if a.publish:
            assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
            try:
                with (logs/'publication.log').open('a',encoding='utf-8') as f:
                    for args in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Stream learned rank/site fitting and physical SAE probe evidence','--',*paths],['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                        subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                event('aggregate_milestone_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
            except subprocess.CalledProcessError as error:event('publication_held_back',exit_code=error.returncode,neural_retry=False)
    from .common import condition,key
    for index in range(6):
        cfg=condition(index);name=key(cfg);last_events=0
        with (logs/(name+'_worker.log')).open('x',encoding='utf-8') as f:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_ranksite13.worker',
                '--condition-index',str(index),'--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root],
                env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('ranksite_worker_started',condition=name,pid=worker.pid)
            while worker.poll() is None:
                file=base/'metadata'/name/'events.jsonl'
                if file.exists():
                    raw=file.read_bytes();closed=raw[:raw.rfind(b'\n')+1];count=len(closed.splitlines())
                    if count>last_events:last_events=count;refresh()
                time.sleep(3)
            event('ranksite_worker_terminal',condition=name,exit_code=worker.returncode);refresh(with_probes=True)
    event('bounded_ranksite_queue_finished',scope='six registered cells, all failures retained; no neural retries/donor transfer')


if __name__=='__main__':main()
