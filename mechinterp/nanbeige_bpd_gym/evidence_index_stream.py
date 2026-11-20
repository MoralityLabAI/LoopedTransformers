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
    start=time.monotonic();last_head=None
    try:
        while True:
            state=k.WaitForSingleObject(handle,0) if handle else 0
            if state not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
            head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip()
            if head!=last_head or state==0:
                last_head=head
                try:
                    result=refresh()
                    if result['changed']:
                        event('committed_evidence_index_generated',**result)
                        if a.publish:
                            with (out/'publication.log').open('a',encoding='utf-8') as f:
                                for args in (['git','add','--sparse','--',*OUTPUTS],
                                    ['git','commit','--only','-m','Stream consolidated committed Nanbeige evidence index','--',*OUTPUTS],
                                    ['git','push','origin','HEAD:refs/heads/'+BRANCH]):
                                    subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=FLAGS)
                            event('index_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,creationflags=FLAGS).strip())
                except Exception as error:
                    event('publication_held_back',error_type=type(error).__name__,message=str(error),model_action=False)
            if state==0:event('target_queue_terminal_final_index_checked',pid=a.until_pid);break
            if time.monotonic()-start>=a.max_seconds:event('finite_index_observation_budget_reached',model_action=False);break
            # A verified handle wait; it does not pause or stop the research goal.
            state=k.WaitForSingleObject(handle,15000)
            if state not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if handle:k.CloseHandle(handle)


if __name__=='__main__':main()
