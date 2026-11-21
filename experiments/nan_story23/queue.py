"""Health-only finite official SDK queue, after actual fitting/host termination."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.request
from .common import ROOT, RUNS, eligible_rows, read, sha, verify, write

FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--after-pid', type=int, required=True); p.add_argument('--host-after-pid', type=int, required=True)
    p.add_argument('--python', required=True); p.add_argument('--cpu-python', required=True); p.add_argument('--snapshot', required=True)
    p.add_argument('--compactification-root', required=True); p.add_argument('--bpd-root', required=True); p.add_argument('--freeze-commit', required=True); p.add_argument('--publish', action='store_true'); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json'); RUNS.mkdir(exist_ok=True); logs = RUNS / 'operations'; logs.mkdir(exist_ok=False)
    assert not any((RUNS / b).exists() for b in ('nf4', 'bf16'))
    branch = 'research/nanbeige-bpd-gym-suite-towers'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == branch
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], text=True).split()[0]
    for ref in ('HEAD', remote):
        subprocess.run(['git', 'merge-base', '--is-ancestor', a.freeze_commit, ref], check=True)

    def event(name, **fields):
        value = dict(UTC=datetime.now(timezone.utc).isoformat(), event=name, **fields)
        with (RUNS / 'queue_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(value) + '\n')
        print(value, flush=True)

    write(RUNS / 'supervisor.json', dict(pid=os.getpid(), after_pid=a.after_pid, host_after_pid=a.host_after_pid, freeze_commit=a.freeze_commit))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    host_handle = kernel.OpenProcess(0x00100000 | 0x1000, False, a.host_after_pid)
    if not host_handle: assert ctypes.get_last_error() == 87
    for pid in (a.after_pid,):
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if handle:
            try:
                while True:
                    state = kernel.WaitForSingleObject(handle, 30000)
                    if state == 0:
                        break
                    assert state == 258; event('verified_wait_for_upstream_supervisor', pid=pid)
            finally:
                kernel.CloseHandle(handle)
        else:
            assert ctypes.get_last_error() == 87
    query = r"$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' -and $_.CommandLine -notmatch 'experiments\.nan_xref(19|22)\.worker\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', query], text=True, creationflags=FLAGS).strip()
    assert not raw or not json.loads(raw), 'A known model remains live; no second large model starts'
    repo = Path.cwd().resolve(); wslrepo = '/mnt/' + repo.drive[0].lower() + repo.as_posix()[2:]
    def wsl(*args):
        return ['wsl', '-d', 'Ubuntu', '-u', 'duganist', '--cd', wslrepo, '--exec', *args]
    route = subprocess.check_output(wsl('ip', 'route', 'show', 'default'), text=True).split(); gateway = route[route.index('via')+1]
    guest = subprocess.check_output(wsl('hostname', '-I'), text=True).split()[0]
    auth = RUNS / 'local_auth.txt'
    if not auth.exists():
        with auth.open('x', encoding='utf-8') as f:
            f.write(secrets.token_hex(32))
    token = auth.read_text().strip(); assert len(token) >= 32
    env = os.environ.copy(); env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'HF_HUB_OFFLINE')})
    env['NAN_STORY23_TOKEN'] = token; env.pop('TRANSFORMERS_CACHE', None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path, method='GET'):
        request = urllib.request.Request(f'http://{gateway}:18764'+path, headers={'Authorization': 'Bearer '+token}, method=method)
        with opener.open(request, timeout=5) as response:
            return json.load(response)

    def stop_exact_sdk(backend):
        # No generic kill, WSL shutdown or math-process operation is permitted.
        code = "import os,signal; os.nice(19); found=[]\nfor n in os.listdir('/proc'):\n if not n.isdigit(): continue\n try: argv=open('/proc/'+n+'/cmdline','rb').read().decode().split('\\0')\n except (OSError,UnicodeError): continue\n if any(x.endswith('/vf-run') for x in argv) and 'nanbeige_storyworld_suite' in argv and 'nan-story23-"+backend+"-model' in argv: found.append(int(n))\nif len(found)!=1: raise SystemExit(4 if not found else 5)\nos.kill(found[0],signal.SIGTERM)"
        return subprocess.run(wsl('/home/duganist/.venvs/nanbeige-prime-ed70053/bin/python', '-c', code), creationflags=FLAGS).returncode

    def command(args, label):
        with (logs / (label + '.log')).open('a', encoding='utf-8') as f:
            return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=FLAGS).returncode

    with (logs / 'host_supervisor.log').open('x', encoding='utf-8') as hf:
        host = subprocess.Popen(['C:\\Python312\\python.exe', '-X', 'utf8', '-B', '-u', '-S', '-m', 'experiments.nan_story23.host_queue',
            '--after-pid', str(a.host_after_pid), '--python', a.python, '--snapshot', a.snapshot, *(['--publish'] if a.publish else [])],
            env=env, stdout=hf, stderr=subprocess.STDOUT, creationflags=FLAGS)
    event('host_supervisor_started', pid=host.pid, waits_for_XREF22=a.host_after_pid)
    while not (RUNS/'host_guard_acquired.signal').exists():
        assert host.poll() is None,'Host guard failed; preserve failure before model starts'
        time.sleep(.2)
    if host_handle: kernel.CloseHandle(host_handle)
    for backend in ('nf4', 'bf16'):
        assert subprocess.check_output(['git', '-C', a.compactification_root, 'rev-parse', 'HEAD'], text=True).strip() == seal['adapter_commit']
        pin_check = "import importlib.metadata as m,json; d=json.loads(m.distribution('verifiers').read_text('direct_url.json')); assert d['vcs_info']['commit_id']=='ed70053b6a7a4079d6976e03bdcd7d420356d594'; assert m.version('verifiers')=='0.4.2.dev92'"
        assert command(wsl('/home/duganist/.venvs/nanbeige-prime-ed70053/bin/python', '-c', pin_check), backend + '_SDK_pin') == 0
        assert not (RUNS / f'prime/nan-story23-{backend}-model').exists()
        limit=len(eligible_rows(backend))
        with (logs / (backend + '_server.log')).open('x', encoding='utf-8') as f:
            worker = subprocess.Popen([a.python, '-X', 'utf8', '-B', '-u', '-m', 'experiments.nan_story23.server', '--backend', backend,
                '--bind', gateway, '--client-ip', guest, '--snapshot', a.snapshot, '--compactification-root', a.compactification_root,
                '--bpd-root', a.bpd_root, '--out', str((RUNS / backend).resolve())], env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS)
            event('model_worker_started', backend=backend, pid=worker.pid); ready = False; started = time.monotonic(); last = 0
            while worker.poll() is None:
                try:
                    health = endpoint('/health'); assert health['backend'] == backend and not health['mock']; ready = True; break
                except OSError:
                    pass
                if time.monotonic()-last > 60:
                    event('verified_model_loading', backend=backend, pid=worker.pid, observation_limit_exceeded=time.monotonic()-started > 1200); last = time.monotonic()
                time.sleep(2)
            if ready:
                with (logs / (backend + '_official_runner.log')).open('x', encoding='utf-8') as rf:
                    runner = subprocess.Popen(wsl('sh', 'experiments/nan_story23/run_cli.sh', backend, gateway, str(limit)), stdout=rf, stderr=subprocess.STDOUT, creationflags=FLAGS)
                    event('official_runner_started', backend=backend, pid=runner.pid); last = 0; started = time.monotonic(); stop_sent = False
                    while runner.poll() is None:
                        count = None
                        try:
                            count = endpoint('/health')['completed_requests']
                        except OSError:
                            pass
                        if worker.poll() is not None:
                            terminal = read(RUNS / backend / 'receipt.json'); count = terminal['requests']
                            if terminal['status'] != 'completed' and not stop_sent:
                                code = stop_exact_sdk(backend); stop_sent = code == 0
                                event('exact_SDK_stop_after_verified_model_failure', backend=backend, result_code=code)
                        if time.monotonic()-last > 60:
                            event('verified_official_runner_live', backend=backend, pid=runner.pid, closed_neural_requests=count,
                                observation_limit_exceeded=time.monotonic()-started > 9000); last = time.monotonic()
                        time.sleep(2)
                    event('official_runner_terminal_or_skipped', backend=backend, exit_code=runner.returncode)
            else:
                event('official_runner_terminal_or_skipped', backend=backend, reason='verified_model_loading_failure')
            if worker.poll() is None:
                try:
                    endpoint('/shutdown', 'POST')
                except OSError:
                    pass
            while worker.poll() is None:
                event('verified_wait_for_model_termination', backend=backend, pid=worker.pid); time.sleep(10)
            event('model_worker_terminal', backend=backend, exit_code=worker.returncode)
    event('registered_story23_model_lane_finished', scope='Fresh storyworld model lane complete; host lane remains separately supervised, continuous goal active, test sealed.')


if __name__ == '__main__':
    main()
