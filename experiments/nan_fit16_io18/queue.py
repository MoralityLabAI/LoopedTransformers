"""Wait for actual GRAD15 termination; conditional bounded stock factor fits."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, RUNS, eligible, read, sha, verify, write

FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--after-pid', type=int, required=True)
    p.add_argument('--python', required=True); p.add_argument('--cpu-python', required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--bpd-root', required=True); p.add_argument('--freeze-commit', required=True); p.add_argument('--publish', action='store_true'); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify(); RUNS.mkdir(parents=True, exist_ok=False); logs = RUNS / 'queue_logs'; logs.mkdir()
    branch = 'research/nanbeige-bpd-gym-suite-towers'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == branch
    subprocess.run(['git', 'merge-base', '--is-ancestor', a.freeze_commit, 'HEAD'], check=True)
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], text=True).split()[0]
    subprocess.run(['git', 'merge-base', '--is-ancestor', a.freeze_commit, remote], check=True)
    def event(name, **fields):
        v = dict(UTC=datetime.now(timezone.utc).isoformat(), event=name, **fields)
        with (RUNS / 'queue_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(v) + '\n')
        print(v, flush=True)
    write(RUNS / 'supervisor.json', dict(pid=os.getpid(), upstream_pid=a.after_pid, freeze_commit=a.freeze_commit))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, a.after_pid)
    if handle:
        try:
            last = 0
            while True:
                state = kernel.WaitForSingleObject(handle, 5000)
                if state == 0:
                    break
                assert state == 258
                if time.monotonic() - last > 60:
                    event('verified_wait_for_GRAD15_queue', pid=a.after_pid); last = time.monotonic()
        finally:
            kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error() == 87
    for label, command in [
        ('original_NF16_audit', [a.python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16.audit', '--backend', 'nf4', '--snapshot', a.snapshot, '--compactification-root', a.compactification_root]),
        ('original_partial_audit', [a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16_io18.audit_original']),
        ('original_NF16_report', [a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16.report'])]:
        with (logs / (label + '.log')).open('x', encoding='utf-8') as f:
            code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
        event(label + '_terminal', exit_code=code)
        assert code == 0, 'Preserve original attempt; no model starts until terminal host checks pass'
    paths = {stage: {b: 'runs/' + stage + '/audits/' + b + '.json' for b in ('nf4', 'bf16')}
             for stage in ('nan_comp14', 'nan_grad15')}
    missing = [path for bybackend in paths.values() for path in bybackend.values() if not Path(path).exists()]
    if missing:
        decision = dict(passed=False, missing_qualification_audits=missing)
    else:
        audits = {stage: {b: read(path) for b, path in values.items()} for stage, values in paths.items()}
        decision = eligible(audits['nan_comp14'], audits['nan_grad15'])
    decision['audit_sha256'] = {stage: {b: sha(path) for b, path in values.items() if Path(path).exists()}
                               for stage, values in paths.items()}
    write(RUNS / 'eligibility.json', decision)
    if not decision['passed']:
        event('upstream_prerequisite_failed_or_missing', scope='No FIT16 model starts; no relaxed gate or substituted template')
        subprocess.run([a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16_io18.report'], check=True, creationflags=FLAGS)
        if a.publish:
            paths = ['reports/nanbeige_bpd_gym/' + x for x in ('NAN_FIT16_IO18_PROGRESS.md', 'nan_fit16_io18_progress.json', 'NAN_FIT16_PROGRESS.md', 'nan_fit16_progress.json')]
            for command in (['git', 'add', '--sparse', '--', *paths], ['git', 'commit', '--only', '-m', 'Preserve upstream-gated native fitting conditions', '--', *paths], ['git', 'push', 'origin', 'HEAD:refs/heads/' + branch]):
                subprocess.run(command, check=True, creationflags=FLAGS)
        return
    query = r"$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', query], text=True, creationflags=FLAGS).strip()
    assert not raw or not json.loads(raw), 'Another known model worker remains live'
    env = os.environ.copy(); env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'HF_HUB_OFFLINE')}); env.pop('TRANSFORMERS_CACHE', None)
    for backend in ('nf4', 'bf16'):
        with (logs / (backend + '_worker.log')).open('x', encoding='utf-8') as f:
            process = subprocess.Popen([a.python, '-X', 'utf8', '-B', '-u', '-m', 'experiments.nan_fit16_io18.worker', '--backend', backend,
                '--snapshot', a.snapshot, '--compactification-root', a.compactification_root, '--bpd-root', a.bpd_root],
                stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=FLAGS)
            event('worker_started', backend=backend, pid=process.pid); code = process.wait()
            event('worker_terminal', backend=backend, exit_code=code)
        for label, command in [('audit', [a.python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16_io18.audit', '--backend', backend, '--snapshot', a.snapshot, '--compactification-root', a.compactification_root]),
                               ('report', [a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_fit16_io18.report'])]:
            with (logs / (backend + '_' + label + '.log')).open('x', encoding='utf-8') as f:
                code = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
            event(label + '_terminal', backend=backend, exit_code=code)
            if code:
                break
        if a.publish and not code:
            paths = ['reports/nanbeige_bpd_gym/' + x for x in ('NAN_FIT16_IO18_PROGRESS.md', 'nan_fit16_io18_progress.json', 'NAN_FIT16_PROGRESS.md', 'nan_fit16_progress.json')]
            with (logs / 'publication.log').open('a', encoding='utf-8') as f:
                for command in (['git', 'add', '--sparse', '--', *paths], ['git', 'commit', '--only', '-m', 'Stream actual matched NF4 and BF16 factor fitting', '--', *paths], ['git', 'push', 'origin', 'HEAD:refs/heads/' + branch]):
                    subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS)
            event('aggregate_pushed', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    event('registered_fit16_queue_finished', scope='Actual registered fitting milestone only; broader goal stays active; no donor transfer.')


if __name__ == '__main__':
    main()
