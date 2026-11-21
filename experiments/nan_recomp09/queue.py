"""One bounded BF16 amendment after the actual context-eval supervisor exits."""
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
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);p.add_argument('--gpu-python',required=True)
    p.add_argument('--cpu-python',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--bpd-root',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    repo=Path.cwd().resolve();root=repo/'experiments/nan_recomp09';seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():assert sha(root/name)==h,name
    runs=repo/'runs/nan_recomp09';runs.mkdir(parents=True,exist_ok=True);events=runs/'queue_events.jsonl'
    if events.exists():raise FileExistsError('Preserve the existing amended fitting attempt')
    logs=runs/'queue_logs';logs.mkdir()
    def event(name,**values):
        r={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**values}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    write(runs/'queue_supervisor.json',{'pid':os.getpid(),'after_pid':a.after_pid,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json')})
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
                if time.monotonic()-last>=60:event('verified_wait_for_context_queue',pid=a.after_pid);last=time.monotonic()
        finally:k.CloseHandle(handle)
    elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_ctx08.worker*' -or $_.CommandLine -like '*experiments.nan_fit07.worker*' -or $_.CommandLine -like '*experiments.nan_prime06.server*' -or $_.CommandLine -like '*experiments.nan_recomp09.worker*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        if not raw or not json.loads(raw):break
        event('known_model_still_live',pids=json.loads(raw));time.sleep(10)
    env=os.environ.copy();env|={'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    reports=['reports/nanbeige_bpd_gym/NAN_RECOMP09_PROGRESS.md','reports/nanbeige_bpd_gym/nan_recomp09_progress.json']
    def command(args,name):
        with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
            return subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    def refresh():
        code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_recomp09.report'],'independent_report')
        if code:event('independent_report_held_back',exit_code=code);return False
        if a.publish:
            if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!='research/nanbeige-bpd-gym-suite-towers':raise RuntimeError('Wrong publication branch')
            for args in (['git','add','--sparse','--',*reports],['git','commit','--only','-m','Stream memory-bounded BF16 BPD fitting milestone','--',*reports],
                         ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                if command(args,'git_publication'):raise RuntimeError('Preserve the unpushed report and inspect Git failure')
            event('aggregate_milestone_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
        return True
    with (logs/'bf16_worker.log').open('x',encoding='utf-8') as log:
        worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_recomp09.worker','--snapshot',a.snapshot,
            '--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        event('amended_BF16_worker_started',launcher_pid=worker.pid);baseline_seen=False;last=0
        while True:
            if not baseline_seen and (runs/'bf16/baseline.json').exists():baseline_seen=True;refresh()
            if worker.poll() is not None:break
            if time.monotonic()-last>=60:event('verified_model_worker_live',pid=worker.pid);last=time.monotonic()
            time.sleep(2)
        event('amended_worker_terminal',exit_code=worker.returncode)
    checked=refresh();receipt=runs/'bf16/t2_layer21_down_proj_rank8/receipt.json'
    if checked and receipt.exists() and read(receipt)['status']=='completed':
        code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_recomp09.probe','--backend','bf16','--passes','2'],'learned_BPD_SAE_probe')
        event('learned_BPD_SAE_probe_terminal',exit_code=code);refresh()
    event('bounded_amended_fitting_queue_finished',scope='no automatic retry; all failures and original FIT07 attempt preserved')


if __name__=='__main__':main()
