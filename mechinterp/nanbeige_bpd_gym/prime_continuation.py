"""One bounded, outcome-blind operational tail after the frozen Prime attempts.

Never alters the frozen model server/runner or overwrites primary evidence.
Only missing/operationally failed episodes qualify, including zero-reward successes
as already finished. No response replay and no retry-until-correct policy.
"""
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
from experiments.nan_gym05.common import read,rows,write,sha


def unfinished(public,episodes):
    expected={r['id']:r for r in public if r['split']=='development'};successful=set()
    for episode in episodes:
        data=episode['task']['data'];key=data['id']
        if key not in expected or data['prompt']!=expected[key]['prompt']:raise ValueError('Unexpected primary source')
        if episode['ok'] and len(episode['traces'])==1:
            trace=episode['traces'][0]
            if trace['ok'] and trace['is_completed'] and not trace['is_timeout'] and len(trace['calls'])==1:successful.add(key)
    # Deliberately never inspect correctness, format, reward or reference gold.
    return [r['id'] for r in public if r['split']=='development' and r['id'] not in successful]


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary-queue-pid',type=int,required=True);p.add_argument('--gpu-python',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);a=p.parse_args()
    repo=Path.cwd().resolve();base=repo/'runs/nan_prime06/continuation';base.mkdir(parents=True,exist_ok=False)
    registration=read(repo/'mechinterp/nanbeige_bpd_gym/PRIME_CONTINUATION_REGISTRATION.json')
    for name,h in registration['source_files_sha256'].items():
        if sha(repo/name)!=h:raise ValueError('Continuation registration source changed')
    events=base/'events.jsonl'
    def event(name,**data):
        row={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**data}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write(base/'supervisor.json',{'pid':os.getpid(),'primary_queue_pid':a.primary_queue_pid,'registration_sha256':sha(repo/'mechinterp/nanbeige_bpd_gym/PRIME_CONTINUATION_REGISTRATION.json')})
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x00100000|0x1000,False,a.primary_queue_pid)
    if not handle:
        error=ctypes.get_last_error()
        if error!=87:raise ctypes.WinError(error)
    else:
        try:
            last=0
            while True:
                result=k.WaitForSingleObject(handle,5000)
                if result==0:break
                if result!=258:raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic()-last>=60:event('verified_wait_for_primary_queue',pid=a.primary_queue_pid);last=time.monotonic()
        finally:k.CloseHandle(handle)
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*experiments.nan_prime06.server*' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    while True:
        active=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip()
        if not active or not json.loads(active):break
        event('primary_model_still_live',pids=json.loads(active));time.sleep(10)
    primary=read(repo/'experiments/nan_prime06/SOURCE_MANIFEST.json')
    cohort=repo/'runs/nan_gym05/cohort/public.jsonl'
    if sha(cohort)!=primary['prompt_cohort_sha256']:raise ValueError('Registered task cohort changed')
    public=rows(cohort);wslrepo='/mnt/'+repo.drive[0].lower()+repo.as_posix()[2:]
    def wsl(*args):return ['wsl','-d','Ubuntu','--cd',wslrepo,'--exec',*args]
    route=subprocess.check_output(wsl('ip','route','show','default'),text=True).split();gateway=route[route.index('via')+1]
    guest=subprocess.check_output(wsl('hostname','-I'),text=True).split()[0]
    token=(repo/'runs/nan_prime06/local_auth.txt').read_text().strip();assert len(token)>=24
    env=os.environ.copy();env|={'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NAN_PRIME_LOCAL_TOKEN':token}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path,method='GET'):
        request=urllib.request.Request(f'http://{gateway}:18761'+path,headers={'Authorization':'Bearer '+token},method=method)
        with opener.open(request,timeout=5) as response:return json.load(response)
    for backend in ('nf4','bf16'):
        file=repo/'runs/nan_prime06/prime'/('nan-prime06-'+backend)/'traces.jsonl'
        episodes=rows(file) if file.exists() else [];pending=unfinished(public,episodes)
        write(base/(backend+'_selection_receipt.json'),{'remaining_tasks':len(pending),'primary_episodes':len(episodes),
            'selection':'no successful completed official episode; no reward/format filtering','primary_trace_sha256':sha(file) if file.exists() else None})
        if not pending:event('no_tail_required',backend=backend);continue
        ids=base/(backend+'_pending_ids.txt');ids.write_text(''.join(x+'\n' for x in pending),encoding='utf-8',newline='\n')
        event('one_registered_tail_started',backend=backend,tasks=len(pending),selection_file_sha256=sha(ids))
        with (base/(backend+'_worker_console.log')).open('x',encoding='utf-8') as log:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_prime06.server','--backend',backend,
                '--bind',gateway,'--client-ip',guest,'--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--out',str(base/backend)],
                cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            ready=False;start=time.monotonic()
            while worker.poll() is None and time.monotonic()-start<1200:
                try:
                    status=endpoint('/health')
                    if status['backend']!=backend or status['mock']:raise RuntimeError('Wrong local model endpoint')
                    ready=True;break
                except OSError:time.sleep(2)
            if not ready:raise RuntimeError('Tail model not ready; preserve actual live worker, never duplicate it')
            relative_ids=ids.relative_to(repo).as_posix()
            cli=['/tmp/nanbeige-prime-ed70053/bin/vf-run','nanbeige_bpd_suite','--env.taskset.scope','local','--env.taskset.include-copy-controls','True',
                '--env.agent.harness.id','null','--env.agent.runtime.type','subprocess','--env.agent.max-turns','1','--env.agent.timeout.setup','120','--env.agent.timeout.rollout','600',
                '--client.base-url',f'http://{gateway}:18761/v1','--client.api-key-var','NAN_PRIME_LOCAL_TOKEN','--sampling.temperature','0','--sampling.top-p','1','--sampling.max-tokens','192',
                '--model','nanbeige-451ed48c-'+backend,'--max-concurrent','1','--select.limit',str(len(pending)),'--select.shuffle','False',
                '--select.include.files.ids',relative_ids,'--num-rollouts','1','--run.name','nan-prime06-'+backend+'-tail1','--output-dir','runs/nan_prime06/prime_tail1']
            shell='export PYTHONPATH="$PWD"; export NAN_PRIME_LOCAL_TOKEN="$(cat runs/nan_prime06/local_auth.txt)"; exec "$@"'
            with (base/(backend+'_runner_console.log')).open('x',encoding='utf-8') as output:
                runner=subprocess.Popen(wsl('sh','-c',shell,'nanbeige-local-tail',*cli),cwd=repo,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                code=runner.wait(timeout=9000)
            event('tail_runner_terminal',backend=backend,exit_code=code)
            if worker.poll() is None:
                try:endpoint('/shutdown','POST')
                except OSError:pass
            worker_code=worker.wait(timeout=1200)
            event('tail_worker_terminal',backend=backend,exit_code=worker_code,requests=read(base/backend/'receipt.json').get('requests',0))
    event('bounded_Prime_continuation_finished',scope='primary and tail receipts remain separate; audit their union without inventing missing scores')


if __name__=='__main__':main()
