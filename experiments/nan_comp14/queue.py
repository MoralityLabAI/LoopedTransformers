"""NF4 then matched BF16; finite registered jobs, no neural retries."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT, RUNS, events, read, sha, verify, write_new

FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--python', required=True); p.add_argument('--snapshot', required=True)
    p.add_argument('--compactification-root', required=True); p.add_argument('--freeze-commit', required=True)
    p.add_argument('--runs', default=str(RUNS)); p.add_argument('--publish', action='store_true'); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); verify(); cfg = read(ROOT / 'config.json'); runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=False); logs = runs / 'queue_logs'; logs.mkdir()
    branch = 'research/nanbeige-bpd-gym-suite-towers'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == branch
    subprocess.run(['git', 'merge-base', '--is-ancestor', a.freeze_commit, 'HEAD'], check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], text=True).split()[0]
    assert head == remote, 'Freeze must be pushed before inference'
    query = r"$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', query], text=True, creationflags=FLAGS).strip()
    assert not raw or not __import__('json').loads(raw), 'Another known model worker is live'
    env = os.environ.copy(); env.update({v: '1' for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'HF_HUB_OFFLINE')})
    env.pop('TRANSFORMERS_CACHE', None)
    def event(name, **fields):
        import json
        row = dict(UTC=datetime.now(timezone.utc).isoformat(), event=name, **fields)
        with (runs / 'queue_events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(row, flush=True)
    write_new(runs / 'supervisor.json', dict(pid=os.getpid(), freeze_commit=a.freeze_commit,
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), source_pushed_before_inference=True))
    paths = ['reports/nanbeige_bpd_gym/NAN_COMP14_PROGRESS.md', 'reports/nanbeige_bpd_gym/nan_comp14_progress.json']
    def report():
        with (logs / 'report.log').open('a', encoding='utf-8') as f:
            code = subprocess.run([a.python, '-X', 'utf8', '-B', '-m', 'experiments.nan_comp14.report', '--runs', str(runs)],
                                  env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
        event('report_terminal', exit_code=code)
        if code or not a.publish:
            return
        if not subprocess.check_output(['git', 'diff', '--name-only', '--', *paths], text=True).strip():
            return
        with (logs / 'publication.log').open('a', encoding='utf-8') as f:
            try:
                for command in (['git', 'add', '--sparse', '--', *paths],
                                ['git', 'commit', '--only', '-m', 'Stream fresh NF4/BF16 competence pilot evidence', '--', *paths],
                                ['git', 'push', 'origin', 'HEAD:refs/heads/' + branch]):
                    subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS)
                event('aggregate_pushed', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
            except subprocess.CalledProcessError as error:
                event('publication_held_back', exit_code=error.returncode, neural_retry=False)
    for backend in cfg['backends']:
        with (logs / (backend + '_worker.log')).open('x', encoding='utf-8') as f:
            process = subprocess.Popen([a.python, '-X', 'utf8', '-B', '-u', '-m', 'experiments.nan_comp14.worker',
                '--backend', backend, '--snapshot', a.snapshot, '--compactification-root', a.compactification_root,
                '--runs', str(runs)], env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS)
            event('worker_started', backend=backend, pid=process.pid); milestone = 0
            while process.poll() is None:
                closed = sum(v['event'] == 'case_closed' for v in events(runs / backend / 'events.jsonl'))
                if closed >= milestone + 40:
                    milestone = closed; report()
                time.sleep(3)
            event('worker_terminal', backend=backend, exit_code=process.returncode)
        with (logs / (backend + '_audit.log')).open('x', encoding='utf-8') as f:
            code = subprocess.run([a.python, '-X', 'utf8', '-B', '-m', 'experiments.nan_comp14.audit', '--backend', backend,
                '--snapshot', a.snapshot, '--compactification-root', a.compactification_root, '--runs', str(runs)],
                env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
        event('independent_audit_terminal', backend=backend, exit_code=code); report()
    event('registered_comp14_queue_finished', scope='Both fixed conditions attempted; no neural retry, gate relaxation or original test access. Broader continuous goal remains active.')


if __name__ == '__main__':
    main()
