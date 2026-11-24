"""One complete frozen cached condition, both model backends then serialized host."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,read,write,append,sha,verify


def run_worker(a,cfg,backend,logs,event):
    from experiments.nan_ram28.resources import memory
    from experiments.nan_storyaudit30.common import Handle
    command=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_cache32.worker','--backend',backend,
        '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--artifact-root',a.artifact_root]
    write(RUNS/(backend+'_worker_permit.json'),dict(backend=backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),all_preceding_actual_workers_ended=True))
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0);hard_stop=None;actual=None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as stream:
        worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags);event('worker_launcher_started',backend=backend,pid=worker.pid)
        signal=RUNS/backend/'runtime_ready.signal'
        while not signal.exists() and worker.poll() is None:time.sleep(.2)
        if signal.exists():
            runtime=read(RUNS/backend/'runtime.json');pid=runtime['pid'];creation=int(runtime['creation'])
            assert runtime['backend']==backend and runtime['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
            query=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object Name,CommandLine | ConvertTo-Json -Compress"
            raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=flags).strip()
            if not raw:
                code=worker.wait();actual=dict(pid=pid,creation=str(creation),status='authoritatively_not_present_after_ready')
                event('worker_ended_before_handle_acquisition',backend=backend,**actual,exit_code=code)
                return dict(backend=backend,actual_worker_ended=True,actual_process=actual,exit_code=code,hard_RAM_stop=None)
            process=json.loads(raw)
            try:authenticated=Handle(pid)
            except OSError:
                raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=flags).strip()
                if raw:raise
                code=worker.wait();actual=dict(pid=pid,creation=str(creation),status='authoritatively_not_present_during_handle_acquisition')
                return dict(backend=backend,actual_worker_ended=True,actual_process=actual,exit_code=code,hard_RAM_stop=None)
            observed=authenticated.creation;authenticated.close()
            if observed!=creation:
                code=worker.wait();actual=dict(pid=pid,creation=str(creation),status='original_PID_reused_after_ready')
                event('original_worker_ended_PID_reused',backend=backend,**actual,exit_code=code)
                return dict(backend=backend,actual_worker_ended=True,actual_process=actual,exit_code=code,hard_RAM_stop=None)
            assert process['Name']=='python.exe' and f'-m experiments.nan_cache32.worker --backend {backend}' in process['CommandLine']
            handle=kernel.OpenProcess(0x00100000|0x1000|0x10|0x1,False,pid);assert handle
            values=[wintypes.FILETIME() for _ in range(4)]
            assert kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in values])
            assert ((values[0].dwHighDateTime<<32)|values[0].dwLowDateTime)==creation
            actual=dict(pid=pid,creation=str(creation));event('authenticated_actual_model_handle',backend=backend,**actual);last=0
            try:
                while True:
                    state=kernel.WaitForSingleObject(handle,2000);assert state in (0,258)
                    if state==0:break
                    try:sample=memory(pid)
                    except OSError:
                        if kernel.WaitForSingleObject(handle,0)==0:break
                        raise
                    if time.monotonic()-last>=10:
                        append(RUNS/'resource_samples.jsonl',dict(UTC=datetime.now(timezone.utc).isoformat(),backend=backend,**actual,**sample));last=time.monotonic()
                    if sample['available_physical_bytes']<cfg['hard_minimum_available_RAM_bytes'] or sample['private_bytes']>cfg['hard_maximum_worker_private_bytes']:
                        hard_stop=sample;event('self_worker_hard_RAM_stop',backend=backend,**actual,memory=sample)
                        assert kernel.TerminateProcess(handle,3);break
                while True:
                    state=kernel.WaitForSingleObject(handle,30000);assert state in (0,258)
                    if state==0:break
                    event('waiting_actual_model_termination',backend=backend,**actual)
            except BaseException:
                if kernel.WaitForSingleObject(handle,0)==258:
                    kernel.TerminateProcess(handle,4)
                    while True:
                        state=kernel.WaitForSingleObject(handle,30000);assert state in (0,258)
                        if state==0:break
                        event('waiting_own_worker_after_supervisor_failure',backend=backend)
                raise
            finally:kernel.CloseHandle(handle)
        code=worker.wait();event('worker_terminal',backend=backend,exit_code=code)
    return dict(backend=backend,actual_worker_ended=True,actual_process=actual,exit_code=code,hard_RAM_stop=hard_stop)


def main():
    p=argparse.ArgumentParser()
    for name in ('python','cpu-python','snapshot','compactification-root','bpd-root','artifact-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args()
    for name in THREADS:os.environ[name]='1'
    from experiments.nan_ranksite13.resources import below_normal
    from experiments.nan_ram28.resources import memory
    below_normal();verify('model');cfg=read(ROOT/'config.json');RUNS.mkdir(parents=True,exist_ok=False)
    artifact=Path(a.artifact_root);artifact.mkdir(parents=True,exist_ok=False);logs=RUNS/'launch';logs.mkdir()
    def event(name,**fields):
        value=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields);append(RUNS/'queue_events.jsonl',value);print(json.dumps(value),flush=True)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/nanbeige-bpd-gym-suite-towers'],text=True).split()[0]
    published=subprocess.check_output(['git','show',remote+':experiments/nan_cache32/SOURCE_MANIFEST.json'])
    assert hashlib.sha256(published).hexdigest()==sha(ROOT/'SOURCE_MANIFEST.json'),'Complete source freeze must be published before inference'
    write(RUNS/'source_publication.json',dict(remote_commit=remote,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    from .predecessors import wait,ensure_inventory_clear
    wait(cfg,event);verify('model')
    write(RUNS/'predecessor_lane_closed.json',dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),all_authenticated_predecessors_ended=True,known_model_host_audit_inventory_clear=True))
    write(RUNS/'supervisor.json',dict(pid=os.getpid(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    results=[]
    for backend in cfg['backends']:
        ensure_inventory_clear()
        while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
            event('waiting_model_RAM_admission',backend=backend,memory=memory());time.sleep(30)
        results.append(run_worker(a,cfg,backend,logs,event))
        write(RUNS/(backend+'_actual_model_ended.json'),results[-1])
    ensure_inventory_clear()
    write(artifact/'model_lane_terminal.json',dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),all_actual_workers_ended=True,backends=results))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        event('waiting_serial_host_RAM',memory=memory());time.sleep(30)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with (logs/'host.log').open('x',encoding='utf-8') as stream:
        host=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_cache32.host','--artifact-root',a.artifact_root,'--snapshot',a.snapshot],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=flags);event('serialized_host_started',launcher_pid=host.pid)
        code=host.wait();event('serialized_host_terminal',exit_code=code)
    ensure_inventory_clear()
    write(artifact/'host_lane_terminal.json',dict(actual_host_launcher_ended=True,exit_code=code))
    from .report import run,OUTPUTS
    run(artifact)
    if a.publish:
        for command in (['git','add','--sparse','--',*OUTPUTS],['git','commit','--only','-m','Publish full cached NF4 and BF16 gym evidence','--',*OUTPUTS],
            ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):subprocess.run(command,check=True,creationflags=flags)
        event('report_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('bounded_cache32_queue_finished',continuous_goal_remains_active=True)


if __name__=='__main__':main()
