"""Finite host follower: repair closed caches; never launch or control a model."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .repair import verify_registration

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def events(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    return [json.loads(v) for v in raw[:raw.rfind(b'\n') + 1].splitlines()]


def training_window(base, records):
    started = [v['condition'] for v in records if v['event'] == 'ranksite_worker_started']
    if not started:
        return False
    stages = {v['event'] for v in events(base / 'metadata' / started[-1] / 'events.jsonl')}
    return 'baseline_closed' in stages and 'fit_closed' not in stages


def reporters_live():
    query = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m experiments.nan_ranksite13.report*' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell', '-NoProfile', '-Command', query], text=True, creationflags=FLAGS).strip()
    return bool(raw and json.loads(raw))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--queue-pid', type=int, required=True)
    p.add_argument('--index-pid', type=int, required=True); p.add_argument('--cpu-python', required=True)
    p.add_argument('--publish', action='store_true'); a = p.parse_args()
    verify_registration()
    from experiments.nan_ranksite13.common import condition, key, read, write
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    base = Path('runs/nan_ranksite13'); out = Path('runs/nan_ranksite13_host01'); out.mkdir(parents=True, exist_ok=True)
    log = out / 'events.jsonl'
    if log.exists():
        raise FileExistsError('Preserve the existing host follower attempt')
    def event(name, **fields):
        row = dict(UTC=datetime.now(timezone.utc).isoformat(), event=name, **fields)
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(row, flush=True)
    write(out / 'supervisor.json', dict(pid=os.getpid(), queue_pid=a.queue_pid, index_pid=a.index_pid))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]; kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    def handle(pid):
        h = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not h and ctypes.get_last_error() != 87:
            raise ctypes.WinError(ctypes.get_last_error())
        return h
    def alive(h):
        if not h:
            return False
        state = kernel.WaitForSingleObject(h, 0)
        if state not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return state == 258
    queue_handle = handle(a.queue_pid); index_handle = handle(a.index_pid)
    env = os.environ.copy(); env.update({name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    processed = set(); started = time.monotonic(); last = 0
    try:
        while time.monotonic() - started < 18 * 3600:
            records = events(base / 'queue_events.jsonl'); live = alive(queue_handle)
            terminal = {v['condition'] for v in records if v['event'] == 'ranksite_worker_terminal'}
            window = not live or training_window(base, records)
            for index in range(6):
                name = key(condition(index))
                if index in processed or name not in terminal or not window:
                    continue
                if reporters_live():
                    break
                receipt_path = base / 'metadata' / name / 'receipt.json'
                if not receipt_path.exists():
                    processed.add(index); event('terminal_worker_without_receipt', condition=name); continue
                worker = read(receipt_path)
                if worker['status'] != 'completed':
                    processed.add(index); event('failed_worker_retained_without_cached_probe', condition=name, status=worker['status']); continue
                with (out / (name + '.log')).open('x', encoding='utf-8') as stream:
                    code = subprocess.run([a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_ranksite13_host01.repair', '--condition-index', str(index)],
                                          env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
                processed.add(index)
                event('amended_cached_probe_terminal', condition=name, exit_code=code, neural_retry=False)
            if not live:
                break
            if time.monotonic() - last > 60:
                event('verified_wait_for_registered_model_queue', queue_pid=a.queue_pid, amended_conditions_processed=len(processed)); last = time.monotonic()
            time.sleep(5)
        if alive(queue_handle):
            event('host_supervision_budget_reached', scope='No model action; original queue remains autonomous'); return
        # The original numerical auditors are now terminal. Wait for the old
        # index publisher too, then make the final report/index writes serial.
        while alive(index_handle) and time.monotonic() - started < 18 * 3600:
            time.sleep(5)
        if alive(index_handle):
            event('host_budget_reached_waiting_for_old_index', scope='No concurrent Git publication attempted'); return
        with (out / 'final_report.log').open('x', encoding='utf-8') as stream:
            code = subprocess.run([a.cpu_python, '-X', 'utf8', '-B', '-m', 'experiments.nan_ranksite13.report'], env=env,
                                  stdout=stream, stderr=subprocess.STDOUT, creationflags=FLAGS).returncode
        event('final_original_report_terminal', exit_code=code)
        if code:
            return
        def publish(paths, message):
            if not a.publish:
                return
            assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'research/nanbeige-bpd-gym-suite-towers'
            with (out / 'publication.log').open('a', encoding='utf-8') as stream:
                for args in (['git', 'add', '--sparse', '--', *paths], ['git', 'commit', '--only', '-m', message, '--', *paths],
                             ['git', 'push', 'origin', 'HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                    subprocess.run(args, check=True, stdout=stream, stderr=subprocess.STDOUT, creationflags=FLAGS)
            event('aggregate_pushed', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        publish(['reports/nanbeige_bpd_gym/' + x for x in ('NAN_RANKSITE13_PROGRESS.md', 'nan_ranksite13_progress.json')], 'Publish final rank/site evidence with registered host audit repair')
        from mechinterp.nanbeige_bpd_gym.evidence_index import refresh, OUTPUTS
        updated = refresh()
        if updated['changed']:
            publish(list(OUTPUTS), 'Update consolidated index after completed host audits')
        event('finite_host_follower_finished', processed_conditions=len(processed), scope='All failed attempts retained; zero neural calls/retries')
    finally:
        for h in (queue_handle, index_handle):
            if h:
                kernel.CloseHandle(h)


if __name__ == '__main__':
    main()
