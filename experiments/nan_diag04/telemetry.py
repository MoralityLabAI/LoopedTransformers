"""Append-only monitoring, isolated from inference failures; never replace a live log."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

class Telemetry:
    def __init__(self,out):self.out=Path(out);self.stop=threading.Event();self.samples=[];self.errors=[]
    def sample(self):
        r={'elapsed_seconds':round(time.perf_counter()-self.start,3),'pid':os.getpid()}
        try:
            x=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,creationflags=0x08000000)
            r['driver_device_global_mib']=[int(v.strip()) for v in x.stdout.strip().split(',')]
        except Exception as e:r['driver_error']=repr(e)
        try:
            cmd=f"@((Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {{$_.Name -like 'pid_{os.getpid()}_*'}}) | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted) | ConvertTo-Json -Compress"
            x=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,timeout=20,creationflags=0x08000000)
            r['windows_process_gpu_memory']=json.loads(x.stdout) if x.stdout.strip() else []
            if x.returncode:r['sampler_error']=x.stderr
        except Exception as e:r['sampler_error']=repr(e)
        self.samples.append(r)
        try:
            with self.out.with_suffix('.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(r)+'\n')
        except Exception as e:self.errors.append({'elapsed_seconds':r['elapsed_seconds'],'error':repr(e)})
    def worker(self):
        while not self.stop.wait(15):self.sample()
    def __enter__(self):
        self.start=time.perf_counter();self.sample();self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=35);self.sample()
        try:self.out.write_text(json.dumps(self.samples,indent=2)+'\n',encoding='utf-8',newline='\n')
        except Exception as e:self.errors.append({'phase':'final_write','error':repr(e)})
        try:self.out.with_name('telemetry_errors.json').write_text(json.dumps(self.errors,indent=2)+'\n',encoding='utf-8',newline='\n')
        except Exception as e:print('Telemetry final serialization failure: '+repr(e),flush=True)
