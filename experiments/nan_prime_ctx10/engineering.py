"""One real-SDK mock transport test, with no pretrained model process."""
import argparse
from pathlib import Path
import os
import secrets
import subprocess
import time
import urllib.request
from .common import read,rows,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    repo=Path.cwd().resolve();base=repo/'runs/nan_prime_ctx10';base.mkdir(exist_ok=True)
    auth=base/'local_auth.txt'
    if not auth.exists():auth.write_text(secrets.token_hex(32),encoding='utf-8')
    token=auth.read_text().strip();assert len(token)>=32
    wslrepo='/mnt/'+repo.drive[0].lower()+repo.as_posix()[2:]
    def wsl(*args):return ['wsl','-d','Ubuntu','-u','duganist','--cd',wslrepo,'--exec',*args]
    route=subprocess.check_output(wsl('ip','route','show','default'),text=True).split();gateway=route[route.index('via')+1]
    guest=subprocess.check_output(wsl('hostname','-I'),text=True).split()[0]
    command='export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1; exec nice -n 19 /home/duganist/.venvs/nanbeige-prime-ed70053/bin/python -m experiments.nan_prime_ctx10.validate_linux --out "$1"'
    with (out/'SDK_discovery.log').open('x',encoding='utf-8') as log:
        subprocess.run(wsl('sh','-c',command,'ctx10-engineering',str(out/'discovery.json').replace('\\','/')),stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env['NAN_PRIME_CTX10_TOKEN']=token
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path,method='GET'):
        import json
        req=urllib.request.Request(f'http://{gateway}:18761'+path,headers={'Authorization':'Bearer '+token},method=method)
        with opener.open(req,timeout=5) as response:return json.load(response)
    with (out/'mock_server.log').open('x',encoding='utf-8') as log:
        worker=subprocess.Popen(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-u','-m','experiments.nan_prime_ctx10.server',
            '--backend','nf4','--bind',gateway,'--client-ip',guest,'--mock','--out',str(out/'mock')],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        ready=False;started=time.monotonic()
        while worker.poll() is None and time.monotonic()-started<60:
            try:ready=endpoint('/health')['mock'];break
            except OSError:time.sleep(.5)
        if not ready:raise RuntimeError('Mock endpoint did not become ready; inspect actual worker before retrying')
        with (out/'official_runner.log').open('x',encoding='utf-8') as runnerlog:
            code=subprocess.run(wsl('sh','experiments/nan_prime_ctx10/run_cli.sh','nf4',gateway,'2','mock'),stdout=runnerlog,stderr=subprocess.STDOUT,timeout=240).returncode
        if worker.poll() is None:
            try:endpoint('/shutdown','POST')
            except OSError:pass
        worker_code=worker.wait(timeout=60)
    sdk=Path('runs/nan_prime_ctx10/prime/nan-prime-ctx10-nf4-mock/traces.jsonl');episodes=rows(sdk)
    assert code==worker_code==0 and len(episodes)==2
    for e in episodes:
        assert e['ok'] and len(e['traces'])==1
        trace=e['traces'][0];assert trace['ok'] and trace['is_completed'] and len(trace['calls'])==1
        assert trace['nodes'][-1]['message']['content']==trace['last_reply']=='MOCK_ONLY' and trace['reward']==0
        assert trace['agent']['runtime']['type']=='subprocess' and trace['agent']['config']['harness']['id']=='null'
    receipt=read(out/'mock/receipt.json');assert receipt['mock_only'] and receipt['requests']==2
    write(out/'receipt.json',dict(status='passed',evidence_class='official SDK/mock local transport only; no pretrained model',
        discovery=read(out/'discovery.json'),official_mock_episodes=2,worker_exit_code=worker_code,runner_exit_code=code,
        source_sha256={name:sha(Path(__file__).parent/name) for name in ('server.py','taskset.py','common.py','run_cli.sh','validate_linux.py','engineering.py')},
        entry_point_sha256=sha('nanbeige_context_suite.py'),mock_trace_sha256=sha(sdk),model_rollouts=0,paid_API_calls=0,sealed_test_accessed=False))
    print({'status':'passed','official_mock_episodes':2,'model_rollouts':0})


if __name__=='__main__':main()
