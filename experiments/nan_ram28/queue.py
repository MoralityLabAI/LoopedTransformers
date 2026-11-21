"""One NF4 model under host-memory budgets, then one serialized host audit."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,append,read,write,sha,verify
from .resources import memory

def main():
    parser=argparse.ArgumentParser()
    for name in ('python','cpu-python','snapshot','compactification-root','bpd-root','artifact-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--publish',action='store_true');a=parser.parse_args()
    for name in THREADS:os.environ[name]='1'
    os.environ['HF_HUB_OFFLINE']='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();cfg=read(ROOT/'config.json');flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    artifact=Path(a.artifact_root);artifact.mkdir(parents=True,exist_ok=False)
    logs=RUNS/'launch';logs.mkdir(exist_ok=False)
    def event(name,**fields):
        row=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        append(RUNS/'queue_events.jsonl',row);print(json.dumps(row),flush=True)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
    query=r"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match '-m experiments\.nan_[^\s]+\.(worker|server)\s'} | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=flags).strip()
    assert not raw or not json.loads(raw),'A known model/host worker remains; inspect, do not overlap'
    write(RUNS/'supervisor.json',dict(pid=os.getpid(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        artifact_root=str(artifact),backend='nf4',BF16_deferred=True,one_numerical_thread=True,priority='BelowNormal'))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        event('waiting_host_RAM_admission',memory=memory());time.sleep(30)
    write(RUNS/'qualification.json',dict(status='explicit_user_host_RAM_resource_condition',run_both_backends=False))
    command=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_ram28.worker','--backend','nf4',
        '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--artifact-root',a.artifact_root]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    hard_stop=None
    with (logs/'nf4_worker.log').open('x',encoding='utf-8') as stream:
        worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        event('worker_launcher_started',pid=worker.pid)
        while not (RUNS/'nf4_runtime_ready.signal').exists() and worker.poll() is None:time.sleep(.2)
        if (RUNS/'nf4_runtime_ready.signal').exists():
            pid=read(RUNS/'nf4_runtime.json')['pid']
            query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';if ($p.CommandLine -notlike '*-m experiments.nan_ram28.worker --backend nf4*') {{exit 3}}"
            assert subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags).returncode==0
            handle=kernel.OpenProcess(0x00100000|0x1000|0x10|0x1,False,pid);assert handle
            event('authenticated_actual_model_handle',pid=pid);last=0
            try:
                while True:
                    state=kernel.WaitForSingleObject(handle,2000);assert state in (0,258)
                    if state==0:break
                    try:sample=memory(pid)
                    except OSError:
                        if kernel.WaitForSingleObject(handle,0)==0:break
                        raise
                    if time.monotonic()-last>=10:
                        append(RUNS/'resource_samples.jsonl',dict(UTC=datetime.now(timezone.utc).isoformat(),pid=pid,**sample));last=time.monotonic()
                    if sample['available_physical_bytes']<cfg['hard_minimum_available_RAM_bytes'] or sample['private_bytes']>cfg['hard_maximum_worker_private_bytes']:
                        hard_stop=sample;event('hard_host_RAM_stop',pid=pid,memory=sample)
                        assert kernel.TerminateProcess(handle,3);break
                while True:
                    state=kernel.WaitForSingleObject(handle,30000);assert state in (0,258)
                    if state==0:break
                    event('waiting_actual_model_termination',pid=pid)
            except BaseException:
                if kernel.WaitForSingleObject(handle,0)==258:
                    kernel.TerminateProcess(handle,4);kernel.WaitForSingleObject(handle,30000)
                raise
            finally:kernel.CloseHandle(handle)
        code=worker.wait();event('worker_terminal',backend='nf4',exit_code=code,hard_stop=hard_stop)
    write(artifact/'model_lane_terminal.json',dict(backend='nf4',actual_worker_ended=True,exit_code=code,BF16_deferred=True))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        event('waiting_serial_host_RAM',memory=memory());time.sleep(30)
    with (logs/'host.log').open('x',encoding='utf-8') as stream:
        host=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_ram28.host',
            '--artifact-root',a.artifact_root],stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        event('serialized_host_started',pid=host.pid);host_code=host.wait();event('serialized_host_terminal',exit_code=host_code)
    from .report import run,OUTPUTS
    run(artifact)
    if a.publish:
        for command in (['git','add','--sparse','--',*OUTPUTS],['git','commit','--only','-m','Publish RAM-bounded NF4 full-gym evidence','--',*OUTPUTS],
            ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):subprocess.run(command,check=True,creationflags=flags)
        event('report_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('bounded_RAM28_queue_finished',continuous_goal_remains_active=True)

if __name__=='__main__':main()
