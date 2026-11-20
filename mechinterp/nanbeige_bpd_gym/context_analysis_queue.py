"""Finite CPU-only context analysis after the actual inference supervisor exits."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
from experiments.nan_fit07.io import read,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);p.add_argument('--cpu-python',required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    root=Path.cwd();out=root/'runs/nan_ctx08/context_analysis_queue';out.mkdir(exist_ok=False)
    write(out/'supervisor.json',{'pid':os.getpid(),'after_pid':a.after_pid,'source_sha256':sha(Path(__file__))})
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    handle=k.OpenProcess(0x00100000|0x1000,False,a.after_pid)
    if handle:
        try:
            while True:
                status=k.WaitForSingleObject(handle,5000)
                if status==0:break
                if status!=258:raise ctypes.WinError(ctypes.get_last_error())
        finally:k.CloseHandle(handle)
    elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
    for b in ('bf16','nf4'):assert read(root/f'runs/nan_ctx08/{b}/receipt.json')['status']=='completed'
    env=os.environ.copy();env.update({v:'1' for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE')})
    commands=[('fixed_BF16_tower',['-m','mechinterp.nanbeige_bpd_gym.context_probes','--backend','bf16']),
        ('paired_tower_report',['-m','mechinterp.nanbeige_bpd_gym.context_probe_report']),
        ('matched_accuracy_geometry',['-m','mechinterp.nanbeige_bpd_gym.context_comparison'])]
    codes={}
    for name,args in commands:
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            code=subprocess.run([a.cpu_python,'-X','utf8','-B',*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
        codes[name]=code
        write(out/'receipt.json',dict(updated_UTC=datetime.now(timezone.utc).isoformat(),status='in_progress' if code==0 else 'failed',exit_codes=codes,sealed_test_accessed=False))
        print({'analysis':name,'exit_code':code},flush=True)
        if code:raise RuntimeError('Preserve failed CPU analysis; no silent alternative or publication')
    paths=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_CTX08_TOWER_PROBES.md','nan_ctx08_tower_probes.json','NAN_CTX08_COMPARISON.md','nan_ctx08_comparison.json')]
    if a.publish:
        assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
        with (out/'git_publication.log').open('x',encoding='utf-8') as log:
            for args in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Publish complete paired context accuracy and fixed-tower geometry','--',*paths],
                ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
                subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    write(out/'receipt.json',dict(updated_UTC=datetime.now(timezone.utc).isoformat(),status='completed',exit_codes=codes,
        reports_pushed=a.publish,report_sha256={n:sha(root/n) for n in paths},sealed_test_accessed=False))
    print({'status':'completed','reports_pushed':a.publish},flush=True)


if __name__=='__main__':main()
