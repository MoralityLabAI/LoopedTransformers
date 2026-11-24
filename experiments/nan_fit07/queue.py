"""Finite Prime->pretrained fitting queue; no optional questions or cloud jobs."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .io import read,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--prime-queue-pid',type=int,required=True);p.add_argument('--gpu-python',required=True)
    p.add_argument('--cpu-python',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True)
    p.add_argument('--compactification-root',required=True);a=p.parse_args()
    repo=Path.cwd().resolve();root=repo/'experiments/nan_fit07';seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen fitting source changed: '+name)
    runs=repo/'runs/nan_fit07';runs.mkdir(parents=True,exist_ok=True);events=runs/'queue_events.jsonl'
    if events.exists():raise FileExistsError('Inspect an existing fitting queue instead of duplicating it')
    logs=runs/'queue_logs';logs.mkdir(exist_ok=True);probes=[]
    def event(name,**value):
        record={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**value}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    event('waiting_for_Prime_queue',pid=a.prime_queue_pid,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.prime_queue_pid)
    if not handle:
        error=ctypes.get_last_error()
        if error!=87:raise ctypes.WinError(error)
        event('Prime_queue_handle_missing')
    else:
        last=time.monotonic()
        try:
            while True:
                state=kernel.WaitForSingleObject(handle,5000)
                if state==0:break
                if state!=258:raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic()-last>=60:event('verified_wait_for_Prime',pid=a.prime_queue_pid);last=time.monotonic()
        finally:kernel.CloseHandle(handle)
    # A terminal supervisor can leave an orphan after an error. Check actual
    # known model processes, not just lock/state files, before starting a model.
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_prime06.server*' -or $_.CommandLine -like '*experiments.nan_gym05.run --backend*' -or $_.CommandLine -like '*experiments.nan_fit07.worker*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        result=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        active=json.loads(result) if result else []
        if not active:break
        event('known_model_processes_still_live',pids=active);time.sleep(10)
    prime_events=repo/'runs/nan_prime06/queue_events.jsonl'
    prime_complete=prime_events.exists() and 'both_registered_Prime_backends_completed' in prime_events.read_text()
    event('Prime_workers_exited',both_benchmarks_complete=prime_complete,
        scope='incomplete Prime outcomes remain preserved; independent fitting pilot proceeds only after actual model exit')
    env=os.environ.copy();env|={'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for backend in ('nf4','bf16'):
        if (runs/backend).exists():raise FileExistsError('Preserve prior fitting attempts')
        with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as stream:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_fit07.worker','--backend',backend,
                '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--runs',str(runs)],
                cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('fitting_worker_started',backend=backend,launcher_pid=worker.pid)
            gate_done=False;probed=set();last=time.monotonic()
            while True:
                if not gate_done and (runs/backend/'baseline.json').exists():
                    gate=subprocess.run([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_fit07.gate','--backend',backend,'--runs',str(runs)],
                        cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    (logs/(backend+'_host_gate.log')).write_text(gate.stdout,encoding='utf-8')
                    if gate.returncode:raise RuntimeError('Host gate failed; preserve the live worker and inspect its logs')
                    gate_done=True;event('host_competence_gate_written',backend=backend,allow_fits=read(runs/backend/'host_gate.json')['allow_fits'])
                for passes in (1,2):
                    cell=runs/backend/f't{passes}_layer21_down_proj_rank8';receipt=cell/'receipt.json'
                    if passes not in probed and receipt.exists():
                        probed.add(passes);r=read(receipt);event('condition_terminal',backend=backend,passes=passes,status=r['status'])
                        if r['status']=='completed':
                            log=(logs/f'{backend}_t{passes}_probe.log').open('x',encoding='utf-8')
                            process=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_fit07.probe','--backend',backend,
                                '--passes',str(passes),'--runs',str(runs)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                            probes.append((backend,passes,process,log));event('CPU_learned_factor_probe_started',backend=backend,passes=passes,pid=process.pid)
                if worker.poll() is not None:break
                if time.monotonic()-last>=60:event('verified_fitting_worker_live',backend=backend,launcher_pid=worker.pid);last=time.monotonic()
                time.sleep(1)
            event('fitting_worker_exited',backend=backend,exit_code=worker.returncode,
                status=read(runs/backend/'receipt.json')['status'] if (runs/backend/'receipt.json').exists() else 'receipt_missing')
            if worker.returncode:raise RuntimeError('Fitting worker failed; do not restart blindly')
    for backend,passes,process,log in probes:
        code=process.wait();log.close();event('CPU_probe_exited',backend=backend,passes=passes,exit_code=code)
    event('initial_fitting_quartet_queue_finished',next='audit all actual fits, compare learned VPD/BPD and SAE geometry, then prospectively extend sites/ranks and causal controls')


if __name__=='__main__':main()
