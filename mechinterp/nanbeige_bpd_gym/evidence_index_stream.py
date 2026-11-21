"""Finite low-CPU publication index follower; it never launches/stops models."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
from .evidence_index import refresh,below_normal,BRANCH,OUTPUTS,FLAGS


def remote_head():
    data=subprocess.check_output(['git','ls-remote','--heads','origin','refs/heads/'+BRANCH],text=True,creationflags=FLAGS,timeout=30).strip().split()
    return data[0] if len(data)==2 and data[1]=='refs/heads/'+BRANCH else None


def publish_generated(source_commit,log):
    """Never commit over a new local report or an as-yet-unpushed worker commit."""
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip()
    if head!=source_commit or remote_head()!=head:return False
    for args in (['git','add','--sparse','--',*OUTPUTS],
        ['git','commit','--only','-m','Stream consolidated committed Nanbeige evidence index','--',*OUTPUTS],
        ['git','push','origin','HEAD:refs/heads/'+BRANCH]):
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=FLAGS)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--until-pid',type=int,required=True);p.add_argument('--max-seconds',type=int,default=64800)
    p.add_argument('--out',default='runs/nan_evidence_index_stream');p.add_argument('--publish',action='store_true');a=p.parse_args()
    below_normal();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);events=out/'events.jsonl'
    def event(name,**fields):
        value=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
        print(value,flush=True)
    if subprocess.check_output(['git','branch','--show-current'],text=True,creationflags=FLAGS).strip()!=BRANCH:raise ValueError('Unexpected publication branch')
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x00100000|0x1000,False,a.until_pid)
    if not handle and ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    event('verified_process_handle' if handle else 'target_already_terminal',pid=a.until_pid,scope='publication only; no model control')
    event('publication_order_guard',remote_match_required=True,observed_branch_quiet_seconds=30)
    start=time.monotonic();last_head=None;observed_head=None;quiet_since=start
    try:
        while True:
            state=k.WaitForSingleObject(handle,0) if handle else 0
            if state not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
            head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip()
            if head!=observed_head:observed_head=head;quiet_since=time.monotonic()
            if (head!=last_head or state==0) and (state==0 or time.monotonic()-quiet_since>=30):
                last_head=head
                try:
                    if a.publish and remote_head()!=head:
                        event('waiting_for_worker_report_push',local_commit=head);last_head=None
                        result=dict(changed=False)
                    else:result=refresh()
                    if result['changed']:
                        event('committed_evidence_index_generated',**result)
                        if a.publish:
                            with (out/'publication.log').open('a',encoding='utf-8') as f:
                                pushed=publish_generated(result['source_commit'],f)
                            if pushed:event('index_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip())
                            else:event('publication_deferred_for_new_worker_commit',model_action=False);last_head=None
                except Exception as error:
                    event('publication_held_back',error_type=type(error).__name__,message=str(error),model_action=False);last_head=None
            if state==0:event('target_queue_terminal_final_index_checked',pid=a.until_pid);break
            if time.monotonic()-start>=a.max_seconds:event('finite_index_observation_budget_reached',model_action=False);break
            # A verified handle wait; it does not pause or stop the research goal.
            state=k.WaitForSingleObject(handle,15000)
            if state not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if handle:k.CloseHandle(handle)


if __name__=='__main__':main()
