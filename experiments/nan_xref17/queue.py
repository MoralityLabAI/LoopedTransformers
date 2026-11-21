"""Finite host-only follower; wait for the actual FIT16 supervisor to end."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
import subprocess
from .common import RUNS, verify, write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--after-pid', type=int, required=True)
    p.add_argument('--cpu-python', required=True); p.add_argument('--freeze-commit', required=True); p.add_argument('--publish', action='store_true'); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify(); RUNS.mkdir(parents=True, exist_ok=False); logs = RUNS / 'queue_logs'; logs.mkdir()
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0); branch = 'research/nanbeige-bpd-gym-suite-towers'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == branch
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], text=True).split()[0]
    for ref in ('HEAD', remote):
        subprocess.run(['git', 'merge-base', '--is-ancestor', a.freeze_commit, ref], check=True)

    def event(name, **fields):
        value = dict(UTC=datetime.now(timezone.utc).isoformat(), event=name, **fields)
        with (RUNS / 'queue_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(value) + '\n')
        print(value, flush=True)

    write(RUNS / 'supervisor.json', dict(pid=os.getpid(), upstream_pid=a.after_pid, freeze_commit=a.freeze_commit))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, a.after_pid)
    if handle:
        try:
            while True:
                state = kernel.WaitForSingleObject(handle, 30000)
                if state == 0:
                    break
                assert state == 258
                event('verified_wait_for_FIT16_queue', pid=a.after_pid)
        finally:
            kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error() == 87
    env = os.environ.copy(); env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    for backend in ('nf4', 'bf16'):
        for stage in ('worker', 'audit', 'report'):
            command = [a.cpu_python, '-X', 'utf8', '-B', '-u', '-m', 'experiments.nan_xref17.' + stage]
            if stage != 'report':
                command += ['--backend', backend]
            with (logs / (backend + '_' + stage + '.log')).open('x', encoding='utf-8') as f:
                process = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=flags)
                event(stage + '_started', backend=backend, pid=process.pid)
                code = process.wait(); event(stage + '_terminal', backend=backend, exit_code=code)
        if a.publish and not code:
            paths = ['reports/nanbeige_bpd_gym/NAN_XREF17_PROGRESS.md', 'reports/nanbeige_bpd_gym/nan_xref17_progress.json']
            with (logs / 'publication.log').open('a', encoding='utf-8') as f:
                for command in (['git', 'add', '--sparse', '--', *paths], ['git', 'commit', '--only', '-m', 'Publish audited learned factor and SAE cross-references', '--', *paths], ['git', 'push', 'origin', 'HEAD:refs/heads/' + branch]):
                    subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, creationflags=flags)
            event('aggregate_pushed', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    event('registered_xref17_queue_finished', scope='Host-only development milestone; continuous goal stays active; no model control or donor transfer.')


if __name__ == '__main__':
    main()
