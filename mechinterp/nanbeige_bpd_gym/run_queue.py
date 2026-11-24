"""One-shot local queue: verify the live BF16 process, then run registered work.

This is a finite research pipeline, not a recurring automation. It never starts
a replacement model merely because monitoring timed out. One large model at a
time; a small CAL-only BF16 SAE fit may overlap NF4 checkpoint loading.
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
from experiments.nan_gym05.common import read,rows,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--runs',default='runs/nan_gym05')
    p.add_argument('--bf16-pid',type=int,required=True);p.add_argument('--gpu-python',required=True);p.add_argument('--cpu-python',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--compactification-root',required=True);a=p.parse_args()
    if os.name!='nt':raise RuntimeError('Queue supervises the Windows 3050 worker')
    repo=Path(a.repo).resolve();runs=(repo/a.runs).resolve();logs=runs/'queue_logs';logs.mkdir(parents=True,exist_ok=True)
    events=runs/'queue_events.jsonl';seal=read(repo/'experiments/nan_gym05/SOURCE_MANIFEST.json')
    assert sha(runs/'cohort/public.jsonl')==seal['combined_public_sha256']
    expected={r['root'] for r in rows(runs/'cohort/public.jsonl')}
    assert read(runs/'bf16/manifest.json')['pid']==a.bf16_pid
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
    # HF_HOME is inherited from the explicitly configured launcher. No secrets
    # or complete environment maps enter logs or publication artifacts.
    children=[];streams=[]
    def event(name,**data):
        r={'UTC':datetime.now(timezone.utc).isoformat(),'event':name,**data}
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    def completed(backend):
        r=read(runs/backend/'receipt.json')
        if r['status']!='completed' or set(r['completed_roots'])!=expected:
            raise RuntimeError(backend+' is not complete; preserve the attempt and inspect it before continuing')
        return r
    def launch(name,python,args,*,fit=False):
        target_env=env.copy()
        if fit:target_env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        stream=(logs/(name+'.log')).open('x',encoding='utf-8');streams.append(stream)
        process=subprocess.Popen([python,'-X','utf8','-u',*args],cwd=repo,env=target_env,
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(process);event('started',stage=name,launcher_pid=process.pid)
        return process
    def checked(name,process):
        code=process.wait();event('exited',stage=name,exit_code=code)
        return code==0
    def cpu(name,module,*args):
        return checked(name,launch(name,a.cpu_python,['-m',module,*map(str,args)]))
    def post(backend):
        if not cpu(backend+'_grade','experiments.nan_gym05.grade','--runs',runs,'--out',runs/'graded','--allow-partial'):return False
        return cpu(backend+'_audit','mechinterp.nanbeige_bpd_gym.audit_native','--runs',runs,'--backend',backend,'--out',runs/'audit'/(backend+'_independent.json'))
    event('queue_started',bf16_pid=a.bf16_pid,scope='registered NF4 then CAL-only towers and CPU audits; no BPD transfer',
        small_BF16_SAE_may_overlap_NF4_loading=True)
    # Hold a real process handle so a PID cannot be mistaken for a restarted
    # worker after an observation timeout or reused process identifier.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.bf16_pid)
    if not handle:
        error=ctypes.get_last_error()
        if error!=87:raise ctypes.WinError(error)
        event('BF16_process_handle_missing')
    else:
        try:
            last=time.monotonic()
            while True:
                state=kernel.WaitForSingleObject(handle,5000)
                if state==0:break
                if state!=258:raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic()-last>=60:event('verified_wait',stage='bf16',pid=a.bf16_pid);last=time.monotonic()
        finally:kernel.CloseHandle(handle)
    completed('bf16');event('BF16_completed_and_process_exited')
    # A new large model starts only after the verified BF16 worker terminated.
    if (runs/'nf4').exists():raise FileExistsError('NF4 directory already exists; inspect the existing attempt, never duplicate it')
    nf4=launch('nf4_native',a.gpu_python,['-m','experiments.nan_gym05.run','--backend','nf4','--compactification-root',a.compactification_root,
        '--snapshot',a.snapshot,'--runs',str(runs)])
    bf_fit=launch('bf16_sae',a.gpu_python,['-m','experiments.nan_gym05.sae','--backend','bf16','--runs',str(runs)],fit=True)
    bf_audit_ok=post('bf16')
    bf_fit_ok=checked('bf16_sae',bf_fit)
    if bf_audit_ok and bf_fit_ok:
        cpu('bf16_probes','mechinterp.nanbeige_bpd_gym.probe_towers','--runs',runs,'--backend','bf16','--out',runs/'probes')
    if not checked('nf4_native',nf4):raise RuntimeError('NF4 terminated unsuccessfully; no blind restart')
    completed('nf4')
    nf_fit=launch('nf4_sae',a.gpu_python,['-m','experiments.nan_gym05.sae','--backend','nf4','--runs',str(runs)],fit=True)
    nf_audit_ok=post('nf4');nf_fit_ok=checked('nf4_sae',nf_fit)
    if nf_audit_ok and nf_fit_ok:
        cpu('nf4_probes','mechinterp.nanbeige_bpd_gym.probe_towers','--runs',runs,'--backend','nf4','--out',runs/'probes')
    event('registered_native_queue_finished',BF16_audit=bf_audit_ok,NF4_audit=nf_audit_ok,BF16_fit=bf_fit_ok,NF4_fit=nf_fit_ok,
        next_stage='actual local Prime model evaluation requires its registered serving/rollout condition')
    for stream in streams:stream.close()

if __name__=='__main__':main()
