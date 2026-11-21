"""Host-only finite follower of the authenticated actual CTX21 supervisor."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
import subprocess
from .common import RUNS,verify,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--cpu-python',required=True);p.add_argument('--freeze-commit',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();RUNS.mkdir(parents=True,exist_ok=False);logs=RUNS/'queue_logs';logs.mkdir()
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0);branch='research/nanbeige-bpd-gym-suite-towers'
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],text=True).split()[0]
    for ref in ('HEAD',remote):
        subprocess.run(['git','merge-base','--is-ancestor',a.freeze_commit,ref],check=True)
    query=f"Get-CimInstance Win32_Process -Filter 'ProcessId = {a.after_pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=flags).strip()
    if raw:
        upstream=json.loads(raw);assert '-m experiments.nan_ctx21.queue ' in upstream['CommandLine']
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]

    def event(name,**fields):
        v=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with (RUNS/'queue_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(v)+'\n')
        print(v,flush=True)

    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.after_pid)
    write(RUNS/'supervisor.json',dict(pid=os.getpid(),upstream_pid=a.after_pid,freeze_commit=a.freeze_commit,authenticated_upstream=bool(raw)))
    if handle:
        assert raw,'Reject an unrelated/reused upstream PID'
        try:
            while True:
                state=kernel.WaitForSingleObject(handle,30000)
                if state==0:
                    break
                assert state==258;event('verified_wait_for_CTX21_supervisor',pid=a.after_pid)
        finally:
            kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error()==87
    from .cache import admit
    env=os.environ.copy()
    for backend in ('nf4','bf16'):
        admission=admit(backend);event('terminal_cache_admission_closed',backend=backend,status=admission['status'])
        for stage in ('worker','audit','report'):
            command=[a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_xref22.'+stage]
            if stage!='report':
                command+=['--backend',backend]
            with (logs/(backend+'_'+stage+'.log')).open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
                event(stage+'_started',backend=backend,pid=process.pid);code=process.wait();event(stage+'_terminal',backend=backend,exit_code=code)
        if a.publish and not code:
            paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_XREF22_PROGRESS.md','nan_xref22_progress.json')]
            with (logs/'publication.log').open('a',encoding='utf-8') as stream:
                for command in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish prospective long-context SAE and learned-fiber probes','--',*paths],['git','push','origin','HEAD:refs/heads/'+branch]):
                    subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=flags)
            event('aggregate_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('registered_xref22_queue_finished',scope='Host-only bounded milestone;continuous goal stays active;no model calls or donor transfer.')


if __name__=='__main__':
    main()
