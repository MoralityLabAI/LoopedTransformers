"""Finite host-only publication after actual fit/SAE supervisor termination.

Uses existing committed presentation producers and immutable Git aggregates.
Never reads live worker/SDK progress, launches a model, or changes a grade.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BRANCH='research/nanbeige-bpd-gym-suite-towers'
ROOT=Path('runs/nan_continuation_presentation')
PREFIX='reports/nanbeige_bpd_gym/'
STAGES=(
    ('fit','experiments.nan_fit16_io18.queue','mechinterp.nanbeige_bpd_gym.io18_results',
        ('NAN_FIT16_IO18_RESULTS.md','nan_fit16_io18_results_receipt.json')),
    ('sae','experiments.nan_xref19.queue','mechinterp.nanbeige_bpd_gym.campaign_index',
        ('CONTINUATION_INDEX.md','continuation_index.json')),
)


def journal(event,**values):
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),event=event,**values)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
    print(value,flush=True)


def closed(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def pin_producers():
    seal=json.loads(Path('experiments/nan_story23/SOURCE_MANIFEST.json').read_bytes())
    result={}
    for _,_,module,_ in STAGES:
        path=module.replace('.','/')+'.py';blob=Path(path).read_bytes();expected=seal['dependencies'][path]
        assert hashlib.sha256(blob).hexdigest()==expected['runtime_sha256'],path
        result[path]=expected
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-pid',type=int,required=True);p.add_argument('--sae-pid',type=int,required=True)
    p.add_argument('--maximum-seconds',type=int,default=43200);p.add_argument('--publish',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    assert a.maximum_seconds>0
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    from .evidence_index import below_normal
    below_normal();pins=pin_producers();flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handles={};results=[];started=time.monotonic()
    try:
        for (stage,module,_,_),pid in zip(STAGES,(a.fit_pid,a.sae_pid)):
            handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
            if handle:
                handles[stage]=handle
                query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m {module} *') {{ exit 3 }}"
                result=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags)
                assert result.returncode==0 or kernel.WaitForSingleObject(handle,0)==0,'Live process identity differs'
            else:assert ctypes.get_last_error()==87
        if a.check_only:
            print(dict(status='passed',authenticated_live_handles=sorted(handles),pinned_producers=len(pins),model_calls=0));return
        ROOT.mkdir(parents=True,exist_ok=False)
        closed(ROOT/'manifest.json',dict(pid=os.getpid(),fit_pid=a.fit_pid,sae_pid=a.sae_pid,maximum_seconds=a.maximum_seconds,
            pinned_producers=pins,output_allowlist=[PREFIX+n for *_,names in STAGES for n in names],model_calls=0,sealed_test_accessed=False))
        for stage,_,producer,names in STAGES:
            handle=handles.get(stage)
            while handle:
                state=kernel.WaitForSingleObject(handle,30000)
                if state==0:break
                assert state==258
                journal('verified_wait_for_actual_supervisor',stage=stage)
                if time.monotonic()-started>=a.maximum_seconds:
                    closed(ROOT/'receipt.json',dict(status='presentation_wait_budget_expired',stages=results,model_calls=0,sealed_test_accessed=False));return
            assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
            source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();pin_producers()
            before={PREFIX+n:subprocess.check_output(['git','show',source+':'+PREFIX+n]) for n in names}
            if stage=='fit':
                aggregate=subprocess.check_output(['git','show',source+':'+PREFIX+'nan_fit16_io18_progress.json'])
                previous=json.loads(before[PREFIX+'nan_fit16_io18_results_receipt.json'])
                if hashlib.sha256(aggregate).hexdigest()==previous['source_sha256']:
                    results.append(dict(stage=stage,source_commit=source,producer_exit_code=0,published=False,reason='same_audited_aggregate_already_presented'))
                    journal('no_new_fit_audit',source_commit=source);continue
            with (ROOT/(stage+'.log')).open('x',encoding='utf-8') as stream:
                code=subprocess.run(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-m',producer,'--source-commit',source],
                    stdout=stream,stderr=subprocess.STDOUT,creationflags=flags).returncode
            value=dict(stage=stage,source_commit=source,producer_exit_code=code,published=False);results.append(value)
            journal('presentation_producer_terminal',**value)
            if code:continue
            paths=[PREFIX+n for n in names]
            if not any(Path(n).read_bytes().replace(b'\r\n',b'\n')!=before[n].replace(b'\r\n',b'\n') for n in paths):
                journal('no_new_presentation',stage=stage);continue
            if a.publish:
                assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
                for command in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Refresh audited Nanbeige '+stage+' sharing evidence','--',*paths],
                    ['git','push','origin','HEAD:refs/heads/'+BRANCH]):
                    with (ROOT/'publication.log').open('a',encoding='utf-8') as stream:
                        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
                value.update(published=True,publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
                journal('presentation_pushed',**value)
        closed(ROOT/'receipt.json',dict(status='completed_host_presentation' if all(v['producer_exit_code']==0 for v in results) else 'preserved_host_presentation_failure',
            stages=results,model_calls=0,new_grades=0,frozen_methods_changed=False,sealed_test_accessed=False,continuous_goal_remains_active=True))
    finally:
        for handle in handles.values():kernel.CloseHandle(handle)


if __name__=='__main__':main()
