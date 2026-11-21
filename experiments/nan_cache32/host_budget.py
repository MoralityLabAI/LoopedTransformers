"""Serialized no-Torch host; own RAM watchdog never controls another process."""
from datetime import datetime,timezone
import os
import threading
import time
from .common import ROOT,RUNS,THREADS,read,write,assert_no_torch


class HostResourceBudgetReached(RuntimeError):pass


class Budget:
    def __init__(self):
        from experiments.nan_ram28.resources import memory
        from experiments.nan_ranksite13.resources import below_normal
        for name in THREADS:os.environ[name]='1'
        os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        below_normal();self.memory=memory;self.cfg=read(ROOT/'config.json');self.samples=[];self.stop=threading.Event();self.started=time.monotonic()
        assert self.check()['available_physical_bytes']>=self.cfg['minimum_start_available_RAM_bytes']
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def check(self):
        assert_no_torch();m=dict(UTC=datetime.now(timezone.utc).isoformat(),**self.memory());self.samples.append(m)
        if m['private_bytes']>self.cfg['maximum_host_private_bytes'] or m['available_physical_bytes']<self.cfg['minimum_available_RAM_bytes']:
            raise HostResourceBudgetReached(str(m))
        if time.monotonic()-self.started>172800:raise HostResourceBudgetReached('host_time_budget')
        return m
    def watch(self):
        while not self.stop.wait(2):
            try:self.check()
            except BaseException as error:
                try:write(RUNS/'host_resource_failure.json',dict(status='self_only_host_stop',reason=str(error),samples=self.samples))
                finally:os._exit(3)
    def close(self):
        self.stop.set();self.thread.join();self.check();write(RUNS/'host_resources.json',dict(status='closed',samples=self.samples,Torch_imported=False))
