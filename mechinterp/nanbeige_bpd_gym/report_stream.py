"""Finite, local-only CPU audit and aggregate publication behind the GPU queue.

Never changes a model, prompt, grade, checkpoint, source seal or GPU process.
Git mutations explicitly name four aggregate report files, excluding raw runs.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from experiments.nan_fit07.io import read,sha,write


BRANCH='research/nanbeige-bpd-gym-suite-towers'
REPORTS=['reports/nanbeige_bpd_gym/'+name for name in
    ('NAN_PRIME06_PROGRESS.md','nan_prime06_progress.json','NAN_FIT07_PROGRESS.md','nan_fit07_progress.json')]


def observables(repo):
    paths=[]
    for backend in ('nf4','bf16'):
        paths.append(f'runs/nan_prime06/{backend}/receipt.json')
        for name in ('baseline.json','receipt.json'):paths.append(f'runs/nan_fit07/{backend}/{name}')
        for passes in (1,2):
            paths.append(f'runs/nan_fit07/{backend}/t{passes}_layer21_down_proj_rank8/receipt.json')
            paths.append(f'runs/nan_fit07/probes/{backend}_t{passes}_aggregate.json')
    return {name:sha(repo/name) if (repo/name).exists() else None for name in paths}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit-queue-pid',required=True,type=int)
    p.add_argument('--gpu-python',required=True);p.add_argument('--cpu-python',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--state-dir',default='runs/nanbeige_report_stream');a=p.parse_args()
    repo=Path.cwd().resolve();folder=Path(a.state_dir);folder.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env|={'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def event(name,**values):
        row={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**values}
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def command(args,label,timeout=1800):
        with (folder/(label+'.log')).open('a',encoding='utf-8') as stream:
            subprocess.run(args,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    def refresh():
        for backend in ('nf4','bf16'):
            receipt=repo/f'runs/nan_prime06/{backend}/receipt.json';audit=repo/f'runs/nan_prime06/audit/{backend}_independent.json'
            if receipt.exists() and read(receipt)['status']=='completed':
                if not audit.exists() or read(audit).get('worker_receipt_sha256')!=sha(receipt):
                    try:
                        command([a.gpu_python,'-X','utf8','-B','-m','mechinterp.nanbeige_bpd_gym.audit_prime',
                            '--backend',backend,'--snapshot',a.snapshot,'--out',str(audit)],backend+'_Prime_audit')
                    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                        event('Prime_final_audit_not_passed',backend=backend,error_type=type(error).__name__,scope='keep missing/error coverage explicit; independent fitting/reporting continues')
                    else:event('Prime_independent_audit_passed',backend=backend,audit_sha256=sha(audit))
        refreshed=[]
        for module in ('prime_progress','fit_progress'):
            try:command([a.cpu_python,'-X','utf8','-B','-m','mechinterp.nanbeige_bpd_gym.'+module],module)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                event('aggregate_stage_held_back',module=module,error_type=type(error).__name__,scope='do not publish this failed stage; GPU and other reporting stages continue')
            else:refreshed+=REPORTS[:2] if module=='prime_progress' else REPORTS[2:]
        if not a.publish:return
        if not refreshed:return
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        if branch!=BRANCH:raise RuntimeError('Refuse publication from a different branch')
        difference=subprocess.run(['git','diff','--quiet','--',*refreshed],cwd=repo)
        if difference.returncode==0:return
        if difference.returncode!=1:raise RuntimeError('Git report diff failed')
        # --only leaves unrelated work and staged changes outside this commit.
        command(['git','add','--sparse','--',*refreshed],'git_publication',120)
        command(['git','commit','--only','-m','Stream audited local Prime and fresh-source fitting milestone','--',*refreshed],'git_publication',120)
        command(['git','push','origin','HEAD:refs/heads/'+BRANCH],'git_publication',180)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        event('aggregate_milestone_pushed',commit=commit,report_files=refreshed)
    write(folder/'supervisor.json',{'pid':os.getpid(),'fit_queue_pid':a.fit_queue_pid,'publish':a.publish,'source_sha256':sha(__file__),
        'scope':'finite CPU postprocessing; tokenizer-only audit, never decoder loading; raw artifacts excluded'})
    if a.once:
        refresh();event('single_postprocessing_check_completed');return
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x00100000|0x1000,False,a.fit_queue_pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    prior=observables(repo);last=0;started=time.monotonic()
    try:
        while time.monotonic()-started<32400:
            state=k.WaitForSingleObject(handle,5000)
            if state not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
            current=observables(repo)
            if current!=prior or state==0:
                refresh();prior=current;event('audited_source_milestone_refreshed')
            if state==0:event('verified_fitting_queue_terminal');return
            if time.monotonic()-last>=60:
                event('verified_wait_for_fitting_queue',pid=a.fit_queue_pid);last=time.monotonic()
        event('finite_monitor_budget_reached',seconds=32400,scope='GPU workers left untouched')
    finally:k.CloseHandle(handle)


if __name__=='__main__':main()
