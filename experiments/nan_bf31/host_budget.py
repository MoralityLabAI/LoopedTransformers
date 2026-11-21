"""One post-model CPU host, bounded private commitment and system reserve."""
import os
import threading
from datetime import datetime,timezone
from .common import ROOT,RUNS,read,write,append,THREADS


class Budget:
    def __init__(self):
        from experiments.nan_ram28.resources import memory
        self.memory=memory;self.cfg=read(ROOT/'config.json');self.stop=threading.Event();self.samples=[]
        self.check();assert self.memory()['available_physical_bytes']>=self.cfg['minimum_start_available_RAM_bytes']
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def check(self):
        sample=dict(UTC=datetime.now(timezone.utc).isoformat(),**self.memory());self.samples.append(sample)
        assert sample['private_bytes']<=self.cfg['maximum_host_private_bytes'] and sample['available_physical_bytes']>=self.cfg['minimum_available_RAM_bytes'],sample
        return sample
    def watch(self):
        while not self.stop.wait(2):
            try:self.check()
            except BaseException as error:
                try:write(RUNS/'host_resource_failure.json',dict(status='self_only_host_stop',reason=str(error),samples=self.samples))
                finally:os._exit(3)
    def finish(self):
        self.stop.set();self.thread.join();self.check();write(RUNS/'host_resources.json',dict(status='passed',samples=self.samples))


def start():
    for key in THREADS:os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();return Budget()
