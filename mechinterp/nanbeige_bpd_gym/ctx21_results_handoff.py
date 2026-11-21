"""Metadata-only follower of the original CTX21 supervisor, never a model retry."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .ctx21_results import OUTPUTS, PRODUCER, SOURCE, SEAL, layout, validate

BRANCH = 'research/nanbeige-bpd-gym-suite-towers'
SELF = 'mechinterp/nanbeige_bpd_gym/ctx21_results_handoff.py'


def terminal_source(journal):
    """No NF4-only publication or unaudited BF16 report can admit this reader."""
    final_reports=[]
    for backend in ('nf4','bf16'):
        previous=-1
        for name in ('worker_terminal','audit_terminal','report_terminal'):
            entries=[(i,r) for i,r in enumerate(journal) if r['event']==name and r.get('backend')==backend]
            if len(entries)!=1:return None
            index,row=entries[0]
            if index<=previous or (name!='worker_terminal' and row['exit_code']!=0):return None
            previous=index
        final_reports.append(previous)
    pushes=[r for i,r in enumerate(journal) if i>max(final_reports) and r['event']=='aggregate_pushed']
    if not pushes:return None
    commit=pushes[-1]['commit']
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    return commit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--after-pid',type=int,required=True)
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    assert args.after_pid>0
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    from .evidence_index import below_normal
    below_normal();flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    root=Path('runs/nan_ctx21_results_handoff');root.mkdir(exist_ok=False)
    def event(name,**fields):
        row=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with (root/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    bindings={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (PRODUCER,SELF)}
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,args.after_pid)
    if handle:
        try:
            query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={args.after_pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m experiments.nan_ctx21.queue *') {{ exit 3 }}"
            verified=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags).returncode==0
            assert verified or kernel.WaitForSingleObject(handle,0)==0,'Unrelated/reused supervisor PID'
            with (root/'supervisor.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(pid=os.getpid(),upstream_pid=args.after_pid,authenticated_actual_handle=True,
                    source_sha256=bindings,new_model_calls=0,sealed_test_accessed=False),stream,indent=2)
            deadline=time.monotonic()+96*3600
            while True:
                state=kernel.WaitForSingleObject(handle,45000)
                if state==0:break
                assert state==258
                event('verified_wait_for_ctx21_handle',pid=args.after_pid)
                if time.monotonic()>=deadline:raise TimeoutError('96-hour metadata wait; original workers untouched')
        finally:kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error()==87
        event('upstream_handle_already_missing',pid=args.after_pid)
    for path,digest in bindings.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    journal=[json.loads(line) for line in Path('runs/nan_ctx21/queue_events.jsonl').read_bytes().splitlines() if line]
    commit=terminal_source(journal)
    if commit is None:
        event('unrun_missing_successful_terminal_audits_or_committed_final_aggregate');return
    blob=subprocess.check_output(['git','show',commit+':'+SOURCE]);data=json.loads(blob)
    validate(data,layout());assert data['source_manifest_sha256']==SEAL
    for backend,value in data['backends'].items():
        for path,field in ((Path('runs/nan_ctx21/audits')/(backend+'.json'),'audit_sha256'),
                           (Path('runs/nan_ctx21')/backend/'receipt.json','receipt_sha256')):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==value[field]
    event('closed_aggregate_bound',source_commit=commit,source_JSON_sha256=hashlib.sha256(blob).hexdigest())
    subprocess.run([sys.executable,'-X','utf8','-B','-S','-m',
        'mechinterp.nanbeige_bpd_gym.ctx21_results','--source-commit',commit],check=True,creationflags=flags)
    if args.publish:
        for command in (['git','add','--sparse','--',*OUTPUTS],
            ['git','commit','--only','-m','Present all matched CTX21 accuracy and physical geometry strata','--',*OUTPUTS],
            ['git','push','origin','HEAD:refs/heads/'+BRANCH]):
            subprocess.run(command,check=True,creationflags=flags)
        event('presentation_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('metadata_handoff_finished',new_model_calls=0,new_grades=0,sealed_test_accessed=False)


if __name__=='__main__':main()
