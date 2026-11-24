"""Publish the correct XREF25 Markdown names after its frozen host queue ends.

The frozen queue publishes the correct machine JSON but retains XREF24's
Markdown name in its publication allowlist. This separate metadata-only
handoff preserves that source and binds the human reports to committed JSON.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);a=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    from .evidence_index import below_normal
    below_normal();flags=getattr(subprocess,'CREATE_NO_WINDOW',0);root=Path('runs/nan_xref25_report_handoff');root.mkdir(exist_ok=False)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000|0x1000,False,a.after_pid);started=time.monotonic()
    if handle:
        try:
            query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={a.after_pid}'; if ($p.Name -ne 'python.exe' -or $p.CommandLine -notlike '*-m experiments.nan_xref25.queue *') {{ exit 3 }}"
            check=subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags)
            assert check.returncode==0 or kernel.WaitForSingleObject(handle,0)==0
            while True:
                state=kernel.WaitForSingleObject(handle,30000)
                if state==0:break
                assert state==258
                with (root/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event='verified_wait_for_xref25',pid=a.after_pid))+'\n')
                if time.monotonic()-started>=345600:raise TimeoutError('Finite96-hour host-report handoff; upstream untouched')
        finally:kernel.CloseHandle(handle)
    else:assert ctypes.get_last_error()==87
    branch='research/nanbeige-bpd-gym-suite-towers';assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==branch
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();prefix='reports/nanbeige_bpd_gym/';bindings={}
    for tag in ('xref19','xref22'):
        path=prefix+'nan_xref25_'+tag+'_progress.json';blob=subprocess.check_output(['git','show',source+':'+path])
        bindings[path]=hashlib.sha256(blob).hexdigest()
    from experiments.nan_xref25.report import main as render
    render()
    for path,h in bindings.items():
        normalized=Path(path).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(normalized).hexdigest()==h,'Machine publication differs; preserve failure instead of orphaning human claims'
    receipt=prefix+'nan_xref25_markdown_handoff_receipt.json'
    Path(receipt).write_text(json.dumps(dict(source_commit=source,source_JSON_sha256=bindings,new_model_calls=0,new_grades=0,
        reason='Correct Markdown publication names outside frozen numerical sources',sealed_test_accessed=False),indent=2)+'\n',encoding='utf-8',newline='\n')
    paths=[prefix+'NAN_XREF25_'+tag.upper()+'_PROGRESS.md' for tag in ('xref19','xref22')]+[receipt]
    for command in (['git','add','--sparse','--',*paths],['git','commit','--only','-m','Synchronize XREF25 human reports with committed machine audits','--',*paths],
        ['git','push','origin','HEAD:refs/heads/'+branch]):subprocess.run(command,check=True,creationflags=flags)


if __name__=='__main__':main()
