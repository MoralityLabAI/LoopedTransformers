"""Finite native->Prime queue: actual local rollouts, no hosted client or asks."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from experiments.nan_gym05.common import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-python',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--compactification-root',required=True);a=p.parse_args()
    repo=Path.cwd().resolve();base=repo/'runs/nan_prime06';base.mkdir(exist_ok=True,parents=True)
    events=base/'queue_events.jsonl'
    if events.exists():raise FileExistsError('Inspect an existing Prime queue; do not duplicate it')
    def event(name,**value):
        data={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**value}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
    root=repo/'experiments/nan_prime06';seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Registered Prime source changed: '+name)
    token=(base/'local_auth.txt').read_text().strip();assert len(token)>=24
    env=os.environ.copy();env|={'NAN_PRIME_LOCAL_TOKEN':token,'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    event('waiting_for_registered_native_queue',source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'))
    last=0
    while True:
        native=repo/'runs/nan_gym05/queue_events.jsonl'
        rows=[json.loads(x) for x in native.read_text().splitlines() if x] if native.exists() else []
        finished=next((r for r in rows if r['event']=='registered_native_queue_finished'),None)
        if finished:
            if not all(finished[k] for k in ('BF16_audit','NF4_audit','BF16_fit','NF4_fit')):raise RuntimeError('Native stages require inspection; no duplicate job')
            break
        if time.monotonic()-last>=60:event('native_queue_still_running');last=time.monotonic()
        time.sleep(3)
    # Event above is emitted only after both large workers and both SAE fitters
    # actually exited. Read-only CPU audits may overlap the next model load.
    wslrepo='/mnt/'+repo.drive[0].lower()+repo.as_posix()[2:]
    def wsl(*args):return ['wsl','-d','Ubuntu','--cd',wslrepo,'--exec',*args]
    route=subprocess.check_output(wsl('ip','route','show','default'),text=True).split()
    gateway=route[route.index('via')+1];guest=subprocess.check_output(wsl('hostname','-I'),text=True).split()[0]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path,method='GET'):
        req=urllib.request.Request(f'http://{gateway}:18761'+path,headers={'Authorization':'Bearer '+token},method=method)
        with opener.open(req,timeout=5) as response:return json.load(response)
    for backend in ('nf4','bf16'):
        out=base/backend;prime=base/'prime'/('nan-prime06-'+backend)
        if out.exists() or prime.exists():raise FileExistsError('Preserve existing Prime condition attempts')
        with (base/(backend+'_server_console.log')).open('x',encoding='utf-8') as console:
            worker=subprocess.Popen([a.gpu_python,'-X','utf8','-u','-m','experiments.nan_prime06.server','--backend',backend,
                '--bind',gateway,'--client-ip',guest,'--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--out',str(out)],
                cwd=repo,env=env,stdout=console,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('model_worker_started',backend=backend,launcher_pid=worker.pid)
            ready=False;mark=time.monotonic()
            while worker.poll() is None and time.monotonic()-mark<1200:
                try:
                    status=endpoint('/health')
                    if status['backend']!=backend or status['mock']:raise RuntimeError('Wrong local endpoint identity')
                    ready=True;break
                except (OSError,urllib.error.URLError):time.sleep(2)
            if not ready:raise RuntimeError('Worker did not become ready; preserve live process/logs, never restart blindly')
            event('model_ready',backend=backend)
            with (base/(backend+'_runner_console.log')).open('x',encoding='utf-8') as log:
                runner=subprocess.Popen(wsl('sh','experiments/nan_prime06/run_cli.sh',backend,gateway),cwd=repo,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                event('official_runner_started',backend=backend,pid=runner.pid)
                code=runner.wait(timeout=9000)
            event('official_runner_exited',backend=backend,exit_code=code)
            if worker.poll() is None:
                # A failed/short runner must not leave an idle model resident.
                try:endpoint('/shutdown','POST')
                except (OSError,urllib.error.URLError):pass
            worker_code=worker.wait(timeout=650);receipt=read(out/'receipt.json')
            event('model_worker_exited',backend=backend,exit_code=worker_code,status=receipt['status'],requests=receipt['requests'])
            if code or worker_code or receipt['status']!='completed':raise RuntimeError('Incomplete Prime attempt preserved; inspect before further inference')
    event('both_registered_Prime_backends_completed',next='audit official traces and compare full replies separately from native prefix-stop outcomes')


if __name__=='__main__':main()
