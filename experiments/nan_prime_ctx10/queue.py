"""Finite official SDK context queue; no cloud calls or neural retries."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from .common import read,write,sha,verify_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);p.add_argument('--gpu-python',required=True);p.add_argument('--cpu-python',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    repo=Path.cwd().resolve();root=Path(__file__).parent;verify_source(root);base=repo/'runs/nan_prime_ctx10';base.mkdir(exist_ok=True)
    eventfile=base/'queue_events.jsonl'
    if eventfile.exists():raise FileExistsError('Preserve the existing registered queue')
    logs=base/'queue_logs';logs.mkdir()
    def event(name,**fields):
        r=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with eventfile.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(r,flush=True)
    write(base/'supervisor.json',dict(pid=os.getpid(),after_pid=a.after_pid,source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json')))
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    handle=k.OpenProcess(0x00100000|0x1000,False,a.after_pid)
    if handle:
        try:
            last=0
            while True:
                state=k.WaitForSingleObject(handle,5000)
                if state==0:break
                if state!=258:raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic()-last>=60:event('verified_wait_for_BPD_queue',pid=a.after_pid);last=time.monotonic()
        finally:k.CloseHandle(handle)
    elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_recomp09.worker*' -or $_.CommandLine -like '*experiments.nan_ctx08.worker*' -or $_.CommandLine -like '*experiments.nan_prime06.server*' -or $_.CommandLine -like '*experiments.nan_prime_ctx10.server*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        if not raw or not json.loads(raw):break
        event('known_model_still_live',pids=json.loads(raw));time.sleep(5)
    wslrepo='/mnt/'+repo.drive[0].lower()+repo.as_posix()[2:]
    def wsl(*args):return ['wsl','-d','Ubuntu','-u','duganist','--cd',wslrepo,'--exec',*args]
    route=subprocess.check_output(wsl('ip','route','show','default'),text=True).split();gateway=route[route.index('via')+1]
    guest=subprocess.check_output(wsl('hostname','-I'),text=True).split()[0]
    token=(base/'local_auth.txt').read_text().strip();assert len(token)>=32
    env=os.environ.copy();env.update({v:'1' for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE')});env['NAN_PRIME_CTX10_TOKEN']=token
    paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_PRIME_CTX10_PROGRESS.md','nan_prime_ctx10_progress.json')]
    def command(args,name):
        with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
            return subprocess.run(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    def refresh():
        code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_prime_ctx10.report','--snapshot',a.snapshot],'independent_report')
        if code:event('audit_publication_held_back',exit_code=code);return False
        if a.publish:
            assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
            for args in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Stream official context SDK and native-prefix evidence','--',*paths],['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                if command(args,'git_publication'):raise RuntimeError('Preserve unpushed aggregate report; inspect Git failure')
            event('aggregate_milestone_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        return True
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path,method='GET'):
        request=urllib.request.Request(f'http://{gateway}:18761'+path,headers={'Authorization':'Bearer '+token},method=method)
        with opener.open(request,timeout=5) as response:return json.load(response)
    for backend in ('nf4','bf16'):
        with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as log:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_prime_ctx10.server','--backend',backend,
                '--bind',gateway,'--client-ip',guest,'--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--out',str(base/backend)],
                cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('model_worker_started',backend=backend,pid=worker.pid);start=time.monotonic();ready=False
            while worker.poll() is None and time.monotonic()-start<1200:
                try:
                    r=endpoint('/health');assert r['backend']==backend and not r['mock'];ready=True;break
                except OSError:time.sleep(2)
            if not ready:
                if worker.poll() is None:raise RuntimeError('Preserve still-live loading worker; observation deadline is not termination')
                event('official_runner_skipped',backend=backend,exit_code=None);event('model_worker_terminal',backend=backend,exit_code=worker.returncode);refresh();continue
            with (logs/(backend+'_official_runner.log')).open('x',encoding='utf-8') as runnerlog:
                runner=subprocess.Popen(wsl('sh','experiments/nan_prime_ctx10/run_cli.sh',backend,gateway),cwd=repo,stdout=runnerlog,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                event('official_runner_started',backend=backend,pid=runner.pid);last=0;mark=time.monotonic()
                while runner.poll() is None:
                    file=base/backend/'progress.json'
                    try:n=read(file)['requests'] if file.exists() else 0
                    except (json.JSONDecodeError,FileNotFoundError):n=last
                    if n>=last+64:last=n;refresh()
                    if time.monotonic()-mark>9000:raise RuntimeError('Preserve actual runner/model processes at observation limit; inspect before any further job')
                    time.sleep(2)
                event('official_runner_terminal',backend=backend,exit_code=runner.returncode)
            if worker.poll() is None:
                try:endpoint('/shutdown','POST')
                except OSError:pass
            code=worker.wait(timeout=650);event('model_worker_terminal',backend=backend,exit_code=code);refresh()
    if refresh():
        current=read('reports/nanbeige_bpd_gym/nan_prime_ctx10_progress.json')['backends']['bf16']
        if current['neural_requests']==288 and current['physical_prefix_vectors_byte_exact']==11520:
            code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_prime_ctx10.learned_probe'],'BPD_VPD_context_probe');event('learned_BPD_context_probe_terminal',exit_code=code)
            if code==0 and a.publish:
                files=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_PRIME_CTX10_BPD_PROBES.md','nan_prime_ctx10_BPD_probes.json')]
                for args in (['git','add','--sparse','--',*files],['git','commit','--only','-m','Cross-reference learned BPD and VPD on matched context prefixes','--',*files],['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                    if command(args,'git_publication'):raise RuntimeError('Preserve unpushed learned-factor context report')
    event('bounded_official_context_queue_finished',scope='all failures retained; no automatic neural retry or old PRIME06 tail')


if __name__=='__main__':main()
