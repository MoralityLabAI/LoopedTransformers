"""Wait for actual FULL26 termination, condition on capacity, preserve originals."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,append,read,sha,verify,write,qualify

BRANCH='research/nanbeige-bpd-gym-suite-towers'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)

def main():
    p=argparse.ArgumentParser()
    for name in ('python','cpu-python','snapshot','compactification-root','bpd-root','artifact-root','source26-artifact-root','freeze-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--after-pid',type=int,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    for key in THREADS:os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();RUNS.mkdir(parents=True,exist_ok=False);logs=RUNS/'operations';logs.mkdir()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    for ref in ('HEAD',remote):subprocess.run(['git','merge-base','--is-ancestor',a.freeze_commit,ref],check=True)
    artifact=Path(a.artifact_root);artifact.mkdir(parents=True,exist_ok=False);source26=Path(a.source26_artifact_root)
    def event(name,**fields):
        v=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields);append(RUNS/'queue_events.jsonl',v);print(v,flush=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles={};started=time.monotonic()
    def acquire(pid,module):
        h=kernel.OpenProcess(0x00100000|0x1000,False,pid)
        if not h:assert ctypes.get_last_error()==87;return
        query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m {module} *') {{ exit 3 }}"
        code=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=FLAGS).returncode
        assert code==0 or kernel.WaitForSingleObject(h,0)==0
        handles[pid]=h;event('authenticated_actual_handle_captured',pid=pid,module=module)
    def wait_all():
        while handles:
            for pid,h in list(handles.items()):
                state=kernel.WaitForSingleObject(h,0)
                if state==0:kernel.CloseHandle(h);del handles[pid];event('actual_handle_terminal',pid=pid)
                else:assert state==258
            if handles:
                event('waiting_actual_upstream_handles',pids=list(handles))
                if time.monotonic()-started>2592000:raise TimeoutError('Registered30-day metadata wait; upstream untouched')
                time.sleep(30)
    def inventory(module):
        query=f"@(Get-CimInstance Win32_Process | Where-Object {{$_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m {module} *'}} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=FLAGS).strip()
        result=json.loads(raw) if raw else [];return result if isinstance(result,list) else [result]
    def command(args,label):
        with (logs/(label+'.log')).open('a',encoding='utf-8') as stream:
            return subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS).returncode
    def report(label):
        code=command(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-m','experiments.nan_mlp27.report','--artifact-root',str(artifact)],'report')
        event('report_terminal',label=label,exit_code=code)
        if code or not a.publish:return
        paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_MLP27_PROGRESS.md','nan_mlp27_progress.json')]
        for cmd in (['git','add','--sparse','--',*paths],):
            if command(cmd,'publication'):event('publication_failed',label=label);return
        if subprocess.run(['git','diff','--quiet','HEAD','--',*paths]).returncode==0:return
        for cmd in (['git','commit','--only','-m','Publish conditional full-scope MLP capacity evidence','--',*paths],
            ['git','push','origin','HEAD:refs/heads/'+BRANCH]):
            if command(cmd,'publication'):event('publication_failed',label=label);return
        event('aggregate_pushed',label=label,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    try:
        acquire(a.after_pid,'experiments.nan_full26.queue')
        write(RUNS/'supervisor.json',dict(pid=os.getpid(),source26_supervisor_pid=a.after_pid,freeze_commit=a.freeze_commit,
            authenticated_actual_upstream_handles=list(handles),artifact_root=str(artifact),upstream_not_stopped=True))
        wait_all()
        # If the old metadata supervisor ended early, retain its actual model
        # and host handles. Never interpret parent exit as child termination.
        for pid in inventory('experiments.nan_full26.worker'):acquire(pid,'experiments.nan_full26.worker')
        wait_all()
        receipts={}
        for backend in ('nf4','bf16'):
            p=source26/backend/'receipt.json';journal=source26/backend/'events.jsonl'
            if p.exists() and journal.exists():
                from .common import rows
                assert any(v['event']=='worker_terminal' for v in rows(journal))
                receipts[backend]=read(p)
                assert receipts[backend]['source_manifest_sha256']==sha('experiments/nan_full26/SOURCE_MANIFEST.json')
                assert receipts[backend]['baseline_registered']==1549 and receipts[backend]['AB_pairs_registered']==216
                manifest=read(source26/backend/'manifest.json')
                assert manifest['backend']==backend and manifest['seal_sha256']==sha('experiments/nan_full26/SOURCE_MANIFEST.json')
        if source26.exists() and not (source26/'model_lane_terminal.json').exists():
            # A registered metadata-only handoff after verified writer exit;
            # no neural response or grade is reconstructed by this marker.
            write(source26/'model_lane_terminal.json',dict(status='terminal',no_large_model_live=True,
                prospective_handoff_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
            event('missing_source26_terminal_marker_closed_after_actual_handles')
        for pid in inventory('experiments.nan_full26.host'):acquire(pid,'experiments.nan_full26.host')
        wait_all()
        q=qualify(receipts);q.update(source26_receipt_sha256={b:sha(source26/b/'receipt.json') for b in receipts},
            source27_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'))
        write(RUNS/'qualification.json',q);event('qualification_closed',status=q['status']);report('qualification')
        if not q['run_both_backends']:
            event('conditional_stage_unrun',continuous_goal_remains_active=True);return
        query=r"@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'experiments\.nan_[^\s]+\.(worker|server|run)\s' -and $_.CommandLine -notmatch 'experiments\.nan_xref(19|22)\.worker\s' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
        raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,creationflags=FLAGS).strip()
        assert not raw or not json.loads(raw),'An actual model remains live'
        with (logs/'host.log').open('x',encoding='utf-8') as stream:
            host=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_mlp27.host','--artifact-root',str(artifact)],stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
        event('streaming_host_started',pid=host.pid);published=0
        for backend in ('nf4','bf16'):
            with (logs/(backend+'_worker.log')).open('x',encoding='utf-8') as stream:
                cmd=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_mlp27.worker','--backend',backend,
                    '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--artifact-root',str(artifact)]
                worker=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
                event('worker_started',backend=backend,pid=worker.pid);last=0
                while worker.poll() is None:
                    if time.monotonic()-last>60:
                        event('actual_model_process_live',backend=backend,pid=worker.pid,host_live=host.poll() is None);last=time.monotonic()
                        journal=artifact/'host/events.jsonl'
                        if journal.exists():
                            with journal.open('rb') as f:count=sum(1 for line in f if line.endswith(b'\n'))
                            if count>=published+32:report('closed_episode_'+str(count));published=count
                    time.sleep(2)
                event('worker_terminal',backend=backend,pid=worker.pid,exit_code=worker.returncode)
            report(backend+'_terminal')
        write(artifact/'model_lane_terminal.json',dict(status='terminal',no_large_model_live=True))
        while host.poll() is None:event('waiting_actual_host_terminal',pid=host.pid);time.sleep(30)
        write(artifact/'host_lane_terminal.json',dict(pid=host.pid,exit_code=host.returncode));report('all_terminal')
        event('bounded_conditional_stage_finished',continuous_goal_remains_active=True,sealed_test_accessed=False)
    finally:
        for h in handles.values():kernel.CloseHandle(h)

if __name__=='__main__':main()
