"""One serialized host lane after XREF22; overlap the next GPU backend only."""
import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import time
from .common import RUNS,append,read,rows,write,verify


def wait_handle(pid,event):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
    event('actual_process_handle_acquired',pid=pid,acquired=bool(handle))
    if not handle:
        assert ctypes.get_last_error()==87;return
    try:
        while True:
            state=kernel.WaitForSingleObject(handle,30000)
            if state==0:break
            assert state==258;event('waiting_actual_process',pid=pid)
    finally:kernel.CloseHandle(handle)


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',required=True,type=int);p.add_argument('--python',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0);logs=RUNS/'host_operations';logs.mkdir(exist_ok=False)
    def event(name,**fields):
        append(RUNS/'host_queue_events.jsonl',dict(event=name,**fields))
        if name=='actual_process_handle_acquired':
            # A zero-byte signal becomes visible only AFTER OpenProcess. The
            # model supervisor retains its original handle until this point.
            (RUNS/'host_guard_acquired.signal').touch(exist_ok=False)
    write(RUNS/'host_supervisor.json',dict(pid=os.getpid(),XREF22_pid=a.after_pid))
    wait_handle(a.after_pid,event)
    for backend in ('nf4','bf16'):
        while not any(e['event']=='model_worker_terminal' and e.get('backend')==backend for e in rows(RUNS/'queue_events.jsonl')):
            event('waiting_closed_story_backend',backend=backend);time.sleep(30)
        codes={}
        for name,args in [('audit',['--snapshot',a.snapshot]),('host',[])]:
            with (logs/(backend+'_'+name+'.log')).open('x',encoding='utf-8') as stream:
                code=subprocess.run([a.python,'-X','utf8','-B','-u','-m','experiments.nan_story23.'+name,'--backend',backend,*args],
                    stdout=stream,stderr=subprocess.STDOUT,creationflags=flags).returncode
            codes[name]=code;event('host_stage_terminal',backend=backend,stage=name,exit_code=code)
            if code:break
        with (logs/(backend+'_report.log')).open('x',encoding='utf-8') as stream:
            code=subprocess.run([a.python,'-X','utf8','-B','-u','-m','experiments.nan_story23.report'],stdout=stream,stderr=subprocess.STDOUT,creationflags=flags).returncode
        event('report_terminal',backend=backend,exit_code=code)
        if a.publish and code==0:
            paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_STORY23_PROGRESS.md','nan_story23_progress.json')]
            for cmd in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish terminal storyworld traces and fixed-probe evidence','--',*paths],
                ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                subprocess.run(cmd,check=True,creationflags=flags)
            event('report_pushed',backend=backend,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if all((RUNS/'audits'/(b+'.json')).exists() for b in ('nf4','bf16')):
        with (logs/'paired_geometry.log').open('x',encoding='utf-8') as stream:
            code=subprocess.run([a.python,'-X','utf8','-B','-u','-m','experiments.nan_story23.paired'],stdout=stream,stderr=subprocess.STDOUT,creationflags=flags).returncode
        event('paired_geometry_terminal',exit_code=code)
        subprocess.run([a.python,'-X','utf8','-B','-m','experiments.nan_story23.report'],check=True,creationflags=flags)
        if a.publish:
            paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_STORY23_PROGRESS.md','nan_story23_progress.json')]
            for cmd in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish exact-history storyworld geometry comparison','--',*paths],
                ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                subprocess.run(cmd,check=True,creationflags=flags)
    event('bounded_story_host_queue_finished',continuous_goal_remains_active=True)


if __name__=='__main__':main()
