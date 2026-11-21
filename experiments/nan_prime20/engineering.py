"""Six real official-SDK mock episodes, including legitimate duplicate prompts."""
import argparse
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.request
from .common import ROOT, RUNS, SDK_COMMIT, read, rows, sha, write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); RUNS.mkdir(exist_ok=True)
    discovery = ROOT / 'receipts/linux_discovery_v1.json'; proven = read(discovery)
    assert proven['status'] == 'passed' and proven['sdk_commit'] == SDK_COMMIT and proven['contexts'] == 96
    auth = RUNS / 'local_auth.txt'
    if not auth.exists():
        with auth.open('x', encoding='utf-8') as f:
            f.write(secrets.token_hex(32))
    token = auth.read_text().strip(); assert len(token) >= 32
    repo = Path.cwd().resolve(); wslrepo = '/mnt/' + repo.drive[0].lower() + repo.as_posix()[2:]
    def wsl(*args):
        return ['wsl','-d','Ubuntu','-u','duganist','--cd',wslrepo,'--exec',*args]
    route = subprocess.check_output(wsl('ip','route','show','default'), text=True).split(); gateway = route[route.index('via')+1]
    guest = subprocess.check_output(wsl('hostname','-I'), text=True).split()[0]
    env = os.environ.copy(); env['NAN_PRIME20_TOKEN'] = token
    env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    def endpoint(path, method='GET'):
        import json
        req = urllib.request.Request(f'http://{gateway}:18763'+path, headers={'Authorization':'Bearer '+token}, method=method)
        with opener.open(req, timeout=5) as response:
            return json.load(response)
    with (out / 'mock_server.log').open('x', encoding='utf-8') as log:
        server = subprocess.Popen(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-u','-m','experiments.nan_prime20.server',
            '--backend','nf4','--bind',gateway,'--client-ip',guest,'--mock','--mock-limit','6','--out',str(out / 'mock')],
            env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        ready = False; started = time.monotonic()
        while server.poll() is None and time.monotonic()-started < 90:
            try:
                ready = endpoint('/health')['mock']; break
            except OSError:
                time.sleep(.5)
        if not ready:
            raise RuntimeError('Inspect actual mock-server handle/log; never duplicate a live process')
        with (out / 'official_runner.log').open('x', encoding='utf-8') as log:
            code = subprocess.run(wsl('sh','experiments/nan_prime20/run_cli.sh','nf4',gateway,'6','mock'), stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags, timeout=600).returncode
        if server.poll() is None:
            try:
                endpoint('/shutdown','POST')
            except OSError:
                pass
        server_code = server.wait(timeout=60)
    file = RUNS / 'prime/nan-prime20-nf4-mock/traces.jsonl'; episodes = rows(file)
    assert code == server_code == 0 and len(episodes) == 6
    for episode in episodes:
        assert episode['ok'] and len(episode['traces']) == 1
        trace = episode['traces'][0]
        assert trace['ok'] and trace['is_completed'] and len(trace['calls']) == 1
        assert trace['nodes'][-1]['message']['content'] == trace['last_reply'] == 'MOCK_ONLY' and trace['reward'] == 0
        assert trace['agent']['runtime']['type'] == 'subprocess' and trace['agent']['config']['harness']['id'] == 'null'
    model = read(out / 'mock/receipt.json'); assert model['mock_only'] and model['requests'] == 6
    records = [read(out / f'mock/request_{i:04d}.json') for i in range(6)]
    assert records[2]['prompt_sha256'] == records[5]['prompt_sha256'] and records[2]['source_id'] != records[5]['source_id']
    result = dict(status='passed', evidence_class='Six official SDK/mock transport episodes, including two distinct erased-source slots with identical prompts; no pretrained model.',
        official_mock_episodes=6, source_slots=6, distinct_erased_sources_with_same_prompt=True,
        discovery_receipt_sha256=sha(discovery), mock_trace_sha256=sha(file), worker_exit_code=server_code, SDK_exit_code=code,
        source_sha256={name:sha(ROOT/name) for name in ('server.py','wire.py','taskset.py','common.py','run_cli.sh','engineering.py')},
        entry_point_sha256=sha('nanbeige_competence_suite.py'), model_rollouts=0, paid_API_calls=0, sealed_test_accessed=False)
    write(out / 'receipt.json', result); print(dict(status='passed', official_mock_episodes=6, model_rollouts=0), flush=True)


if __name__ == '__main__':
    main()
