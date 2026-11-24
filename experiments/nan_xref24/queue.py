"""Serialized finite independent host repairs alongside the real model lane."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,read,sha,verify,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-xref19-pid',type=int,required=True)
    p.add_argument('--xref22-pid',type=int,required=True);p.add_argument('--story23-pid',type=int,required=True)
    p.add_argument('--python',required=True);p.add_argument('--freeze-commit',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    branch='research/nanbeige-bpd-gym-suite-towers';flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
    for commit in ('HEAD',remote):subprocess.run(['git','merge-base','--is-ancestor',a.freeze_commit,commit],check=True)
    RUNS.mkdir(parents=True,exist_ok=True);logs=RUNS/'operations';logs.mkdir(exist_ok=False)
    def event(name,**values):
        value=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**values)
        with (RUNS/'queue_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
        print(value,flush=True)
    write(RUNS/'supervisor.json',dict(pid=os.getpid(),xref19_pid=a.after_xref19_pid,xref22_pid=a.xref22_pid,
        story23_model_pid=a.story23_pid,freeze_commit=a.freeze_commit,new_model_calls=0))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles={};start=time.monotonic()
    def acquire(key,pid,module):
        handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
        if not handle:assert ctypes.get_last_error()==87;return
        handles[key]=handle
        query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m {module} *') {{ exit 3 }}"
        result=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags)
        assert result.returncode==0 or kernel.WaitForSingleObject(handle,0)==0
        event('upstream_process_handle_authenticated',key=key,pid=pid)
    def wait(key):
        handle=handles.get(key)
        while handle:
            state=kernel.WaitForSingleObject(handle,30000)
            if state==0:break
            assert state==258;event('verified_wait_for_host_lane',key=key)
            if time.monotonic()-start>259200:raise TimeoutError('Registered72-hour between-stage wait budget; upstream untouched')
    def command(args,label):
        with (logs/(label+'.log')).open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
            event('host_helper_started',label=label,pid=process.pid);began=time.monotonic();last=0
            while process.poll() is None:
                if time.monotonic()-last>60:
                    event('verified_host_helper_live',label=label,pid=process.pid,observation_limit_exceeded=time.monotonic()-began>14400);last=time.monotonic()
                time.sleep(2)
            event('host_helper_terminal',label=label,exit_code=process.returncode);return process.returncode
    try:
        acquire('xref19',a.after_xref19_pid,'experiments.nan_xref19.queue')
        acquire('xref22',a.xref22_pid,'experiments.nan_xref22.queue')
        acquire('story23_model',a.story23_pid,'experiments.nan_story23.queue')
        for source in ('xref19','xref22'):
            wait(source)
            if source=='xref22':
                wait('story23_model')
                file=Path('runs/nan_story23/host_supervisor.json')
                if file.exists():
                    acquire('story23_host',read(file)['pid'],'experiments.nan_story23.host_queue');wait('story23_host')
            for backend in ('nf4','bf16'):
                receipt=Path('runs/nan_'+source)/backend/'receipt.json'
                if not receipt.exists() or read(receipt)['status']!='completed':
                    write(RUNS/'audits'/(source+'_'+backend+'.json'),dict(status='unrun_no_completed_upstream_worker',
                        source=source,backend=backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),new_model_calls=0,sealed_test_accessed=False))
                else:
                    command([a.python,'-X','utf8','-B','-u','-m','experiments.nan_xref24.audit','--source',source,'--backend',backend],source+'_'+backend+'_audit')
            code=command(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-m','experiments.nan_xref24.report'],source+'_report')
            if a.publish and code==0:
                assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
                paths=['reports/nanbeige_bpd_gym/'+name for name in ('NAN_XREF24_'+source.upper()+'_PROGRESS.md','nan_xref24_'+source+'_progress.json')]
                for cmd in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish separate decoder-orientation audit '+source,'--',*paths],
                    ['git','push','origin','HEAD:refs/heads/'+branch]):subprocess.run(cmd,check=True,creationflags=flags)
                event('repair_report_pushed',source=source,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        event('bounded_host_repairs_finished',continuous_goal_remains_active=True,sealed_test_accessed=False)
    finally:
        for handle in handles.values():kernel.CloseHandle(handle)


if __name__=='__main__':main()
