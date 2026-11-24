"""Small, independent resource lane; all numerical imports occur after admission."""
from datetime import datetime,timezone
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import threading
from experiments.nan_ram28.common import sha,read,write,append,THREADS

ROOT=Path(__file__).parent
RUNS=Path('runs/nan_storyaudit30')


def verify():
    seal=read(ROOT/'SOURCE_MANIFEST.json')
    for name,digest in seal['critical_files_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha('experiments/nan_ram28/SOURCE_MANIFEST.json')==seal['RAM28_source_manifest_sha256']
    from experiments.nan_ram28.common import verify as inherited
    inherited()
    return seal


def no_torch():
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)


def closed_entries(base):
    path=Path(base)/'closed.jsonl'
    if not path.exists():return []
    with path.open('rb') as stream:lines=stream.read().splitlines(keepends=True)
    entries=[json.loads(line) for line in lines if line.endswith(b'\n')]
    assert len({r['file'] for r in entries})==len(entries),'duplicate closed journal entry'
    return entries


def select(entries,roster):
    """Only the first 72 registered story slots; no output-dependent filtering."""
    import re
    selected=[];seen=set()
    for entry in entries:
        m=re.fullmatch(r'baseline_(\d{4})_r([01])_closed.json',entry['file'])
        if not m:continue
        ordinal,row=int(m[1]),int(m[2]);slot=ordinal+row
        if slot>=len(roster):continue
        assert ordinal%2==0 and slot not in seen,'story batch/slot collision'
        seen.add(slot);selected.append(dict(slot=slot,**entry))
    return sorted(selected,key=lambda r:r['slot'])


class Handle:
    def __init__(self,pid,expected_creation=None):
        k=ctypes.WinDLL('kernel32',use_last_error=True);self.k=k
        k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
        k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
        k.CloseHandle.argtypes=[wintypes.HANDLE]
        self.value=k.OpenProcess(0x00100000|0x1000,False,pid)
        if not self.value:raise ctypes.WinError(ctypes.get_last_error())
        values=[wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(self.value,*[ctypes.byref(v) for v in values]):
            self.close();raise ctypes.WinError(ctypes.get_last_error())
        self.creation=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
        if expected_creation is not None and self.creation!=expected_creation:
            self.close();raise RuntimeError('PID creation binding changed')
    def alive(self,wait_ms=0):
        status=self.k.WaitForSingleObject(self.value,wait_ms);assert status in (0,258)
        return status==258
    def close(self):
        if self.value:self.k.CloseHandle(self.value);self.value=None


class Guard:
    def __init__(self,model=None):self.model=model
    def __enter__(self):
        for name in THREADS:os.environ[name]='1'
        os.environ.update(TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',USE_TORCH='0',USE_TF='0',USE_FLAX='0',
                          CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
        os.environ.pop('TRANSFORMERS_CACHE',None)
        from experiments.nan_ranksite13.resources import below_normal
        from experiments.nan_ram28.resources import memory
        below_normal();no_torch();self.memory=memory;self.cfg=read(ROOT/'registration.json')
        self.stop=threading.Event();self.samples=[]
        assert self.check()['available_physical_bytes']>=self.cfg['minimum_start_available_RAM_bytes']
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start();return self
    def check(self):
        no_torch();sample=dict(UTC=datetime.now(timezone.utc).isoformat(),**self.memory())
        assert sample['private_bytes']<=self.cfg['maximum_private_bytes'],sample
        assert sample['available_physical_bytes']>=self.cfg['minimum_available_RAM_bytes'],sample
        if self.model:assert self.model.alive(),'model ended; yield to canonical serialized host'
        self.samples.append(sample);return sample
    def watch(self):
        while not self.stop.wait(2):
            try:self.check()
            except BaseException as error:
                try:write(RUNS/'resource_failure.json',dict(status='self_only_audit_stop',reason=str(error),samples=self.samples))
                finally:os._exit(3)
    def __exit__(self,kind,error,traceback):
        self.stop.set();self.thread.join()
        try:
            self.check()
            write(RUNS/'resources.json',dict(status='passed' if kind is None else 'failed',samples=self.samples,
                CPU_threads=1,priority='BelowNormal',Torch_imported=False,pretrained_forwards=0,sealed_test_accessed=False))
        finally:
            if self.model:self.model.close()
