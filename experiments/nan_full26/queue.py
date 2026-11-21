"""Finite full-gym model lane and one concurrent streaming host lane."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,append,read,sha,verify,write

BRANCH='research/nanbeige-bpd-gym-suite-towers'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)

def main():
    p=argparse.ArgumentParser()
    for name in ('python','cpu-python','snapshot','compactification-root','bpd-root','artifact-root','freeze-commit'):p.add_argument('--'+name,required=True)
    for name in ('ctx21-pid','xref22-pid','story23-pid','xref25-pid'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args()
    for key in THREADS:os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();RUNS.mkdir(parents=True,exist_ok=False);logs=RUNS/'operations';logs.mkdir()
    artifact=Path(a.artifact_root);artifact.mkdir(parents=True,exist_ok=False)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    for ref in ('HEAD',remote):subprocess.run(['git','merge-base','--is-ancestor',a.freeze_commit,ref],check=True)
    def event(name,**fields):
        v=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields);append(RUNS/'queue_events.jsonl',v);print(v,flush=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles={}
    def acquire(pid,module):
        handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
        if not handle:assert ctypes.get_last_error()==87;return
        query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m {module} *') {{ exit 3 }}"
        code=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=FLAGS).returncode
        assert code==0 or kernel.WaitForSingleObject(handle,0)==0
        handles[pid]=handle;event('authenticated_upstream_handle_captured',pid=pid,module=module)
    def wait_all():
        last=0
        while handles:
            for pid,h in list(handles.items()):
                state=kernel.WaitForSingleObject(h,0)
                if state==0:kernel.CloseHandle(h);del handles[pid];event('upstream_handle_terminal',pid=pid)
                else:assert state==258
            if time.monotonic()-last>60:event('waiting_upstream_actual_handles',pids=list(handles));last=time.monotonic()
            if handles:time.sleep(2)
    upstream={a.ctx21_pid:'experiments.nan_ctx21.queue',a.xref22_pid:'experiments.nan_xref22.queue',
        a.story23_pid:'experiments.nan_story23.queue',a.xref25_pid:'experiments.nan_xref25.queue'}
    assert len(upstream)==4
    try:
        for pid,module in upstream.items():acquire(pid,module)
        write(RUNS/'supervisor.json',dict(pid=os.getpid(),freeze_commit=a.freeze_commit,artifact_root=str(artifact),
            authenticated_actual_upstream_handles=list(handles),upstream=upstream,predecessors_not_stopped=True))
        wait_all()
        # STORY23 creates this write-once manifest after its model lane starts.
        host_manifest=Path('runs/nan_story23/host_supervisor.json')
        if host_manifest.exists():acquire(read(host_manifest)['pid'],'experiments.nan_story23.host_queue');wait_all()
        query=r"@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' -and $_.CommandLine -notmatch 'experiments\.nan_xref(19|22)\.worker\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=FLAGS).strip()
        assert not raw or not json.loads(raw),'An actual model is still live'
        with (logs/'host.log').open('x',encoding='utf-8') as stream:
            host=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_full26.host','--artifact-root',str(artifact)],
                stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
        event('streaming_host_started',pid=host.pid)
        published=0
        def report(label):
            with (logs/'report.log').open('a',encoding='utf-8') as stream:
                code=subprocess.run(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-m','experiments.nan_full26.report','--artifact-root',str(artifact)],
                    stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS).returncode
                event('report_terminal',label=label,exit_code=code)
                if code==0 and a.publish:
                    paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_FULL26_PROGRESS.md','nan_full26_progress.json')]
                    subprocess.run(['git','add','--sparse','--',*paths],check=True,stdout=stream,stderr=subprocess.STDOUT)
                    changed=subprocess.run(['git','diff','--quiet','HEAD','--',*paths]).returncode
                    if changed:
                        subprocess.run(['git','commit','--only','-m','Publish full-gym trace and batched A/B milestone','--',*paths],check=True,stdout=stream,stderr=subprocess.STDOUT)
                        subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],check=True,stdout=stream,stderr=subprocess.STDOUT)
                        event('report_pushed',label=label,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        for backend in ('nf4','bf16'):
            with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as stream:
                command=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_full26.worker','--backend',backend,
                    '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--artifact-root',str(artifact)]
                worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
                event('model_worker_started',backend=backend,pid=worker.pid);last=0
                while worker.poll() is None:
                    if time.monotonic()-last>60:
                        event('actual_model_process_live',backend=backend,pid=worker.pid,streaming_host_live=host.poll() is None);last=time.monotonic()
                        journal=artifact/'host/events.jsonl'
                        count=sum(1 for _ in journal.open('rb')) if journal.exists() else 0
                        if count>=published+32:report('closed_episode_'+str(count));published=count
                    time.sleep(2)
                event('model_worker_terminal',backend=backend,pid=worker.pid,exit_code=worker.returncode)
            report(backend+'_model_terminal')
        write(artifact/'model_lane_terminal.json',dict(status='terminal',no_large_model_live=True))
        while host.poll() is None:event('waiting_streamed_host_terminal',pid=host.pid);time.sleep(30)
        event('host_terminal',pid=host.pid,exit_code=host.returncode)
        write(artifact/'host_lane_terminal.json',dict(pid=host.pid,exit_code=host.returncode))
        report('all_actual_processes_terminal')
        event('bounded_full26_milestone_finished',continuous_goal_remains_active=True,sealed_test_accessed=False)
    finally:
        for h in handles.values():kernel.CloseHandle(h)

if __name__=='__main__':main()
