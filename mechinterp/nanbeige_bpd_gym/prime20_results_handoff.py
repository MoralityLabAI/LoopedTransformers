"""Finite host-only matched-results/index publication after real SDK queue exit."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from .prime20_results import OUTPUTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    from .evidence_index import below_normal
    below_normal();branch='research/nanbeige-bpd-gym-suite-towers';flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
    producers=('mechinterp/nanbeige_bpd_gym/prime20_results.py','mechinterp/nanbeige_bpd_gym/campaign_index.py')
    pins={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in producers}
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.after_pid)
    root=Path('runs/nan_prime20_results_handoff');started=time.monotonic()
    try:
        if handle:
            query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={a.after_pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m experiments.nan_prime20.queue *') {{ exit 3 }}"
            result=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags)
            assert result.returncode==0 or kernel.WaitForSingleObject(handle,0)==0
        else:assert ctypes.get_last_error()==87
        if a.check_only:
            print(dict(status='passed',live_supervisor_handle=bool(handle),pinned_producers=len(pins),model_calls=0));return
        root.mkdir(exist_ok=False)
        (root/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),upstream_pid=a.after_pid,
            producer_sha256=pins,maximum_wait_seconds=43200,new_model_calls=0,sealed_test_accessed=False),indent=2)+'\n',encoding='utf-8')
        while handle:
            state=kernel.WaitForSingleObject(handle,30000)
            if state==0:break
            assert state==258
            with (root/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event='verified_wait_for_actual_Prime20_supervisor',pid=a.after_pid))+'\n')
            if time.monotonic()-started>=43200:raise TimeoutError('Finite12-hour presentation wait; all upstream jobs untouched')
        assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for name,h in pins.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'Presentation producer changed while waiting'
        for stage,module in (('results','prime20_results'),('index','campaign_index')):
            with (root/(stage+'.log')).open('x',encoding='utf-8') as stream:
                subprocess.run(['C:\\Python312\\python.exe','-X','utf8','-B','-S','-m','mechinterp.nanbeige_bpd_gym.'+module,
                    '--source-commit',source],stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=flags)
        paths=[*OUTPUTS,'reports/nanbeige_bpd_gym/CONTINUATION_INDEX.md','reports/nanbeige_bpd_gym/continuation_index.json']
        for command in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish matched official Prime execution and current evidence index','--',*paths],
            ['git','push','origin','HEAD:refs/heads/'+branch]):
            with (root/'publication.log').open('a',encoding='utf-8') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,creationflags=flags)
        (root/'receipt.json').write_text(json.dumps(dict(status='published',source_commit=source,
            publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_model_calls=0,new_grades=0,
            sealed_test_accessed=False,continuous_goal_remains_active=True),indent=2)+'\n',encoding='utf-8')
    finally:
        if handle:kernel.CloseHandle(handle)


if __name__=='__main__':main()
