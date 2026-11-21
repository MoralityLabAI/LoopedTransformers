"""After authenticated NF4/model/host/follower exit, one lazy BF16 control and serialized host audit."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,RUNS,THREADS,append,read,write,sha,verify
from experiments.nan_ram28.resources import memory

def main():
    parser=argparse.ArgumentParser()
    for name in ('python','cpu-python','snapshot','compactification-root','bpd-root','artifact-root','nf4-artifact-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--publish',action='store_true');a=parser.parse_args()
    for name in THREADS:os.environ[name]='1'
    os.environ['HF_HUB_OFFLINE']='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();cfg=read(ROOT/'config.json');flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    artifact=Path(a.artifact_root);artifact.mkdir(parents=True,exist_ok=False)
    logs=RUNS/'launch';logs.mkdir(parents=True,exist_ok=False)
    def event(name,**fields):
        row=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        append(RUNS/'queue_events.jsonl',row);print(json.dumps(row),flush=True)
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
    branch='refs/heads/research/nanbeige-bpd-gym-suite-towers'
    remote=subprocess.check_output(['git','ls-remote','origin',branch],text=True).split()[0]
    published=subprocess.check_output(['git','show',remote+':experiments/nan_bf31/SOURCE_MANIFEST.json'])
    assert __import__('hashlib').sha256(published).hexdigest()==sha(ROOT/'SOURCE_MANIFEST.json'),'Complete source freeze must be pushed before inference'
    write(RUNS/'source_publication.json',dict(remote_commit=remote,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    from .predecessors import wait
    wait(cfg,event)
    verify()
    write(RUNS/'predecessor_lane_closed.json',dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        all_authenticated_predecessors_ended=True,known_model_host_audit_inventory_clear=True))
    write(RUNS/'supervisor.json',dict(pid=os.getpid(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        artifact_root=str(artifact),backend='bf16',one_numerical_thread=True,priority='BelowNormal'))
    nf4=Path(a.nf4_artifact_root);runtime=nf4/'nf4/manifest.json';old_manifest=read(runtime)
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    assert old_manifest['backend']=='nf4' and old_manifest['source_manifest_sha256']==seal['RAM28_source_manifest_sha256']
    assert old_manifest['baseline_registered']==1549 and old_manifest['AB_pairs_registered']==216
    write(artifact/'matched_inputs.json',dict(NF4_condition='NAN-RAM-28',NF4_artifact_root=str(nf4.resolve()),
        NF4_source_manifest_sha256=seal['RAM28_source_manifest_sha256'],NF4_runtime_manifest_sha256=sha(runtime),
        BF16_condition='NAN-BF-31',BF16_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        NF4_normal_receipt_sha256=sha(nf4/'nf4/receipt.json') if (nf4/'nf4/receipt.json').exists() else None,
        NF4_canonical_host_receipt_sha256=sha(nf4/'host/receipt.json') if (nf4/'host/receipt.json').exists() else None,
        source_roster_tokenizer_parser_and_budgets_inherited=True,old_condition_relabelled=False,neural_replies_copied=False))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        event('waiting_host_RAM_admission',memory=memory());time.sleep(30)
    write(RUNS/'qualification.json',dict(status='prospective_RAM_bounded_BF16_control_matched_to_frozen_RAM28_NF4',run_both_backends=False))
    command=[a.python,'-X','utf8','-B','-u','-m','experiments.nan_bf31.worker','--backend','bf16',
        '--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--bpd-root',a.bpd_root,'--artifact-root',a.artifact_root]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    hard_stop=None
    with (logs/'bf16_worker.log').open('x',encoding='utf-8') as stream:
        worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        event('worker_launcher_started',pid=worker.pid)
        while not (RUNS/'bf16_runtime_ready.signal').exists() and worker.poll() is None:time.sleep(.2)
        if (RUNS/'bf16_runtime_ready.signal').exists():
            pid=read(RUNS/'bf16_runtime.json')['pid']
            query=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';if ($p.CommandLine -notlike '*-m experiments.nan_bf31.worker --backend bf16*') {{exit 3}}"
            assert subprocess.run(['powershell','-NoProfile','-Command',query],creationflags=flags).returncode==0
            handle=kernel.OpenProcess(0x00100000|0x1000|0x10|0x1,False,pid);assert handle
            event('authenticated_actual_model_handle',pid=pid);last=0
            try:
                while True:
                    state=kernel.WaitForSingleObject(handle,2000);assert state in (0,258)
                    if state==0:break
                    try:sample=memory(pid)
                    except OSError:
                        if kernel.WaitForSingleObject(handle,0)==0:break
                        raise
                    if time.monotonic()-last>=10:
                        append(RUNS/'resource_samples.jsonl',dict(UTC=datetime.now(timezone.utc).isoformat(),pid=pid,**sample));last=time.monotonic()
                    if sample['available_physical_bytes']<cfg['hard_minimum_available_RAM_bytes'] or sample['private_bytes']>cfg['hard_maximum_worker_private_bytes']:
                        hard_stop=sample;event('hard_host_RAM_stop',pid=pid,memory=sample)
                        assert kernel.TerminateProcess(handle,3);break
                while True:
                    state=kernel.WaitForSingleObject(handle,30000);assert state in (0,258)
                    if state==0:break
                    event('waiting_actual_model_termination',pid=pid)
            except BaseException:
                if kernel.WaitForSingleObject(handle,0)==258:
                    kernel.TerminateProcess(handle,4);kernel.WaitForSingleObject(handle,30000)
                raise
            finally:kernel.CloseHandle(handle)
        code=worker.wait();event('worker_terminal',backend='bf16',exit_code=code,hard_stop=hard_stop)
    write(artifact/'model_lane_terminal.json',dict(backend='bf16',actual_worker_ended=True,exit_code=code,BF16_control=True))
    while memory()['available_physical_bytes']<cfg['minimum_start_available_RAM_bytes']:
        event('waiting_serial_host_RAM',memory=memory());time.sleep(30)
    with (logs/'host.log').open('x',encoding='utf-8') as stream:
        host=subprocess.Popen([a.cpu_python,'-X','utf8','-B','-u','-m','experiments.nan_bf31.host',
            '--artifact-root',a.artifact_root],stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        event('serialized_host_started',pid=host.pid);host_code=host.wait();event('serialized_host_terminal',exit_code=host_code)
    write(artifact/'host_lane_terminal.json',dict(actual_host_ended=True,exit_code=host_code))
    from .report import run
    run(artifact)
    OUTPUTS=('reports/nanbeige_bpd_gym/NAN_BF31_PROGRESS.md','reports/nanbeige_bpd_gym/nan_bf31_progress.json')
    if a.publish:
        for command in (['git','add','--sparse','--',*OUTPUTS],['git','commit','--only','-m','Publish lazy BF16 full-gym control and matched NF4 evidence','--',*OUTPUTS],
            ['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):subprocess.run(command,check=True,creationflags=flags)
        event('report_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    event('bounded_BF31_queue_finished',continuous_goal_remains_active=True)

if __name__=='__main__':main()
