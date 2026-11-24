"""BF16-only operational supervision amendment; frozen inference is unchanged.

Poll authenticated health rather than worker progress files. Audit immutable
snapshots, and stop the exact SDK run if its neural server exits unsuccessfully.
"""
import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from experiments.nan_fit07.io import read,rows,sha,write

BASE=Path('runs/nan_prime_ctx10')
REG=Path('mechinterp/nanbeige_bpd_gym/PRIME_CTX10_SUPERVISION_REGISTRATION.json')


@contextmanager
def shared_open(path):
    """Request sharing for append-only SDK snapshots; never read live progress."""
    import msvcrt
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    k.CreateFileW.restype=wintypes.HANDLE
    handle=k.CreateFileW(str(Path(path).resolve()),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    descriptor=msvcrt.open_osfhandle(int(handle),os.O_RDONLY|os.O_BINARY)
    with os.fdopen(descriptor,'rb') as f:yield f


def shared_bytes(path):
    with shared_open(path) as f:return f.read()


def validate_io(out):
    import threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    folder=Path(out);folder.mkdir(parents=True,exist_ok=False);file=folder/'toy_progress.json';write(file,{'count':0})
    ordinary_denied=False
    with file.open('rb'):
        try:write(file,{'count':1})
        except PermissionError:ordinary_denied=True
    write(file,{'count':2})
    delete_shared_denied=False
    with shared_open(file):
        try:write(file,{'count':3})
        except PermissionError:delete_shared_denied=True
    state={'count':0};errors=[]
    class H(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_GET(self):
            body=json.dumps(state).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),H);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def writer():
        try:
            for i in range(100):state['count']=i;write(file,{'count':i});time.sleep(.005)
        except Exception as error:errors.append(type(error).__name__)
    producer=threading.Thread(target=writer);producer.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        with opener.open(f'http://127.0.0.1:{server.server_port}/health') as response:assert 0<=json.load(response)['count']<=99
        time.sleep(.005)
    producer.join();server.shutdown();server.server_close();thread.join();assert not errors and read(file)['count']==99
    result=dict(status='passed',ordinary_reader_reproduced_PermissionError=ordinary_denied,delete_shared_reader_also_denied_replace_on_this_host=delete_shared_denied,
        atomic_worker_style_writes=100,concurrent_health_RPC_reads=100,worker_write_errors=errors,live_progress_file_reads_during_positive_fixture=0,
        source_sha256=sha(Path(__file__)),model_rollouts=0,sealed_test_accessed=False)
    write(folder/'receipt.json',result);print(result)


def snapshot(counts):
    """Hardlink closed immutable model records; copy complete SDK JSONL lines."""
    dest=BASE/'supervision_snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');dest.mkdir(parents=True)
    for backend,count in counts.items():
        source=BASE/backend;folder=dest/backend;folder.mkdir()
        for name in ('manifest.json','resident_head_parity.json','receipt.json'):
            if (source/name).exists():os.link(source/name,folder/name)
        if (folder/'receipt.json').exists():count=read(folder/'receipt.json')['requests']
        write(folder/'progress.json',{'requests':count,'snapshot_count_authority':'terminal receipt or authenticated in-memory health; original progress file never opened'})
        for i in range(count):
            record=source/f'request_{i:04d}.json';r=read(record);state=r['state_file'];assert state==Path(state).name
            os.link(record,folder/record.name);os.link(source/state,folder/state)
        trace=BASE/f'prime/nan-prime-ctx10-{backend}-model/traces.jsonl'
        if trace.exists():
            raw=shared_bytes(trace);closed=raw[:raw.rfind(b'\n')+1]
            target=dest/f'prime/nan-prime-ctx10-{backend}-model/traces.jsonl';target.parent.mkdir(parents=True);target.write_bytes(closed)
    (dest/'queue_events.jsonl').write_bytes(shared_bytes(BASE/'queue_events.jsonl'))
    write(dest/'snapshot_receipt.json',dict(counts=counts,scope='closed record/state hardlinks and complete SDK lines; synthetic offline progress counters; no neural response replay'))
    return dest


def run(a):
    from experiments.nan_prime_ctx10.common import verify_source
    root=Path('experiments/nan_prime_ctx10');verify_source(root);registry=read(REG)
    assert registry['source_sha256']==sha(Path(__file__)) and registry['model_source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
    assert not (BASE/'bf16').exists()
    events=BASE/'queue_events.jsonl';ops=BASE/'operations';logs=ops/'BF16_supervision';logs.mkdir()
    repo=Path.cwd().resolve();wslrepo='/mnt/'+repo.drive[0].lower()+repo.as_posix()[2:]
    def event(name,**fields):
        r=dict(UTC=datetime.now(timezone.utc).isoformat(),event=name,**fields)
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(r,flush=True)
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    query="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -like '*experiments.nan_recomp09.worker*' -or $_.CommandLine -like '*experiments.nan_prime_ctx10.server*') } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True).strip();assert not raw or not json.loads(raw)
    def wsl(*args):return ['wsl','-d','Ubuntu','-u','duganist','--cd',wslrepo,'--exec',*args]
    route=subprocess.check_output(wsl('ip','route','show','default'),text=True).split();gateway=route[route.index('via')+1]
    guest=subprocess.check_output(wsl('hostname','-I'),text=True).split()[0]
    token=(BASE/'local_auth.txt').read_text().strip();assert len(token)>=32
    env=os.environ.copy();env.update({n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','HF_HUB_OFFLINE')});env['NAN_PRIME_CTX10_TOKEN']=token
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def endpoint(path,method='GET'):
        req=urllib.request.Request(f'http://{gateway}:18761'+path,headers={'Authorization':'Bearer '+token},method=method)
        with opener.open(req,timeout=5) as response:return json.load(response)
    def command(args,name):
        with (logs/(name+'.log')).open('a',encoding='utf-8') as log:return subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    reports=['reports/nanbeige_bpd_gym/'+n for n in ('NAN_PRIME_CTX10_PROGRESS.md','nan_prime_ctx10_progress.json')]
    def publish(paths,message):
        if not a.publish:return
        assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='research/nanbeige-bpd-gym-suite-towers'
        for args in (['git','add','--sparse','--',*paths],['git','commit','--only','-m',message,'--',*paths],['git','push','origin','HEAD:refs/heads/research/nanbeige-bpd-gym-suite-towers']):
            if command(args,'publication'):raise RuntimeError('Preserve the unpushed report')
        event('amended_supervision_report_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def refresh(count):
        snap=snapshot({'nf4':51,'bf16':count})
        code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_prime_ctx10.report','--snapshot',a.snapshot,'--runs',str(snap)],'independent_snapshot_audit')
        if code:event('snapshot_audit_held_back',exit_code=code);return False
        publish(reports,'Stream audited context evidence using immutable supervision snapshots');return True
    event('BF16_supervision_amendment_activated',registry_sha256=sha(REG),original_NF4_replayed=False)
    with (logs/'bf16_worker.log').open('x',encoding='utf-8') as log:
        worker=subprocess.Popen([a.gpu_python,'-X','utf8','-B','-u','-m','experiments.nan_prime_ctx10.server','--backend','bf16',
            '--bind',gateway,'--client-ip',guest,'--snapshot',a.snapshot,'--compactification-root',a.compactification_root,'--out',str((BASE/'bf16').resolve())],
            env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        event('model_worker_started',backend='bf16',pid=worker.pid,supervision_amendment_sha256=sha(REG));ready=False;started=time.monotonic()
        while worker.poll() is None and time.monotonic()-started<1200:
            try:r=endpoint('/health');assert r['backend']=='bf16' and not r['mock'];ready=True;break
            except OSError:time.sleep(2)
        if not ready:raise RuntimeError('Inspect actual loading process/receipt; never duplicate it')
        with (logs/'official_runner.log').open('x',encoding='utf-8') as log:
            runner=subprocess.Popen(wsl('sh','experiments/nan_prime_ctx10/run_cli.sh','bf16',gateway),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            event('official_runner_started',backend='bf16',pid=runner.pid);last=0;count=0;started=time.monotonic()
            while runner.poll() is None:
                try:count=endpoint('/health')['completed_requests']
                except OSError:pass
                if count>=last+64:last=count;refresh(count)
                if worker.poll() is not None:
                    receipt=read(BASE/'bf16/receipt.json');count=receipt['requests']
                    if receipt['status']!='completed':
                        event('SDK_abort_after_verified_model_failure',backend='bf16',worker_status=receipt['status'])
                        code="import os,signal; ps=[]\nfor n in os.listdir('/proc'):\n if not n.isdigit(): continue\n try: a=open('/proc/'+n+'/cmdline','rb').read().decode().split('\\0')\n except (OSError,UnicodeError): continue\n if any(x.endswith('/vf-run') for x in a) and 'nanbeige_context_suite' in a and 'nan-prime-ctx10-bf16-model' in a: ps.append(int(n))\nassert len(ps)==1,ps\nfor p in ps: os.kill(p,signal.SIGTERM)"
                        subprocess.run(wsl('/home/duganist/.venvs/nanbeige-prime-ed70053/bin/python','-c',code),check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                        runner.wait(timeout=60);break
                if time.monotonic()-started>9000:raise RuntimeError('Preserve and inspect actual runner; observation limit is not terminal')
                time.sleep(2)
            event('official_runner_terminal',backend='bf16',exit_code=runner.returncode)
        if worker.poll() is None:
            try:endpoint('/shutdown','POST')
            except OSError:pass
        code=worker.wait(timeout=650);event('model_worker_terminal',backend='bf16',exit_code=code)
    receipt=read(BASE/'bf16/receipt.json');count=receipt['requests']
    if refresh(count) and count==288:
        code=command([a.cpu_python,'-X','utf8','-B','-m','experiments.nan_prime_ctx10.learned_probe'],'learned_factor_probe');event('learned_BPD_context_probe_terminal',exit_code=code)
        if code==0:publish(['reports/nanbeige_bpd_gym/'+n for n in ('NAN_PRIME_CTX10_BPD_PROBES.md','nan_prime_ctx10_BPD_probes.json')],'Cross-reference learned BPD and VPD on measured context prefixes')
    event('amended_BF16_supervision_terminal',scope='no NF4 replay; model/SDK/scoring/frozen source unchanged; original partial NF4 coverage retained')


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-io');p.add_argument('--gpu-python');p.add_argument('--cpu-python');p.add_argument('--snapshot');p.add_argument('--compactification-root');p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.validate_io:validate_io(a.validate_io)
    else:run(a)


if __name__=='__main__':main()
