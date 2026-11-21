"""Wait on authenticated actual upstream handles, then finite native conditions."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, RUNS, verify, write

FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
BRANCH = 'research/nanbeige-bpd-gym-suite-towers'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--fit-supervisor-pid', required=True, type=int)
    p.add_argument('--xref-supervisor-pid', required=True, type=int); p.add_argument('--prime-supervisor-pid', required=True, type=int)
    p.add_argument('--python', required=True); p.add_argument('--cpu-python', required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--freeze-commit', required=True); p.add_argument('--publish', action='store_true'); a = p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify(); RUNS.mkdir(parents=True, exist_ok=False); logs = RUNS/'queue_logs'; logs.mkdir()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    for ref in ('HEAD',remote):
        subprocess.run(['git','merge-base','--is-ancestor',a.freeze_commit,ref],check=True)

    def event(name, **fields):
        value = dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with (RUNS/'queue_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value)+'\n')
        print(value,flush=True)

    upstream = {a.fit_supervisor_pid:'experiments.nan_fit16_io18.queue', a.xref_supervisor_pid:'experiments.nan_xref19.queue', a.prime_supervisor_pid:'experiments.nan_prime20.queue'}
    assert len(upstream)==3
    query = r"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    inventory = json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=FLAGS))
    bypid = {r['ProcessId']:r for r in inventory}
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]; kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]; kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles = {}
    try:
        for pid,module in upstream.items():
            if pid in bypid:
                assert '-m '+module+' ' in bypid[pid]['CommandLine'], 'Reject reused/unrelated upstream PID'
            handle = kernel.OpenProcess(0x00100000 | 0x1000,False,pid)
            if handle:
                assert pid in bypid
                handles[pid]=handle
            else:
                assert ctypes.get_last_error()==87
        write(RUNS/'supervisor.json',dict(pid=os.getpid(),freeze_commit=a.freeze_commit,upstream=upstream,authenticated_actual_handles=list(handles)))
        last = 0
        while handles:
            for pid,handle in list(handles.items()):
                state = kernel.WaitForSingleObject(handle,1000)
                if state==0:
                    kernel.CloseHandle(handle); del handles[pid]; event('upstream_handle_terminal',pid=pid)
                else:
                    assert state==258
            if handles and time.monotonic()-last>60:
                event('verified_wait_for_upstream_handles',pids=list(handles)); last=time.monotonic()
    finally:
        for handle in handles.values():
            kernel.CloseHandle(handle)
    query = r"$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=FLAGS).strip()
    assert not raw or not json.loads(raw), 'Another known model remains live'
    env = os.environ.copy(); env['HF_HUB_OFFLINE']='1'; env.pop('TRANSFORMERS_CACHE',None)
    for backend in ('nf4','bf16'):
        command = [a.python,'-X','utf8','-B','-u','-m','experiments.nan_ctx21.worker','--backend',backend,
            '--snapshot',a.snapshot,'--compactification-root',a.compactification_root]
        with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as stream:
            process = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
            event('worker_started',backend=backend,pid=process.pid); code=process.wait(); event('worker_terminal',backend=backend,exit_code=code)
        command = [a.python,'-X','utf8','-B','-m','experiments.nan_ctx21.audit','--backend',backend,
            '--snapshot',a.snapshot,'--compactification-root',a.compactification_root]
        with (logs/(backend+'_audit.log')).open('x',encoding='utf-8') as stream:
            code = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS).returncode
        event('audit_terminal',backend=backend,exit_code=code)
        with (logs/(backend+'_report.log')).open('x',encoding='utf-8') as stream:
            report_code = subprocess.run([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_ctx21.report'],env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS).returncode
        event('report_terminal',backend=backend,exit_code=report_code)
        assert report_code==0, 'Preserve report failure; do not claim missing/unverified results'
        if a.publish:
            paths = ['reports/nanbeige_bpd_gym/'+v for v in ('NAN_CTX21_PROGRESS.md','nan_ctx21_progress.json')]
            with (logs/'publication.log').open('a',encoding='utf-8') as stream:
                for command in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Stream bounded 4K and 8K native development evidence','--',*paths],['git','push','origin','HEAD:refs/heads/'+BRANCH]):
                    subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
            event('aggregate_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('registered_ctx21_queue_finished',scope='Bounded context milestone only;continuous goal remains active;no donor transfer.')


if __name__=='__main__':
    main()
