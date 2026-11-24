"""Self-only budget for CPU tensor engineering; never controls another process."""
from datetime import datetime,timezone
import os
from pathlib import Path
import threading


class CPUBudget:
    maximum_private=3*2**28
    minimum_available=4*2**30
    minimum_start=5*2**30
    def __init__(self,out):self.out=Path(out)
    def __enter__(self):
        self.out.resolve().relative_to(Path('runs').resolve())
        self.out.mkdir(parents=True,exist_ok=False)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='1',HF_HOME=str(self.out/'HF_cache'),HF_MODULES_CACHE=str(self.out/'HF_modules'))
        os.environ.pop('TRANSFORMERS_CACHE',None)
        from experiments.nan_ranksite13.resources import below_normal
        from experiments.nan_ram28.resources import memory
        from experiments.nan_ram28.common import write
        below_normal();self.memory=memory;self.write=write;self.samples=[];self.stop=threading.Event()
        assert self.check()['available_physical_bytes']>=self.minimum_start
        write(self.out/'admission.json',dict(CPU_only=True,maximum_private_bytes=self.maximum_private,
            minimum_available_bytes=self.minimum_available,minimum_start_bytes=self.minimum_start,CPU_threads=1,priority='BelowNormal'))
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start();return self
    def check(self):
        m=dict(UTC=datetime.now(timezone.utc).isoformat(),**self.memory());self.samples.append(m)
        assert m['private_bytes']<=self.maximum_private and m['available_physical_bytes']>=self.minimum_available,m
        return m
    def watch(self):
        while not self.stop.wait(2):
            try:self.check()
            except BaseException as error:
                try:self.write(self.out/'resource_failure.json',dict(status='self_only_engineering_stop',error=str(error),samples=self.samples))
                finally:os._exit(3)
    def __exit__(self,kind,error,traceback):
        self.stop.set();self.thread.join();self.check()
        self.write(self.out/'resources.json',dict(status='passed' if kind is None else 'failed',
            failure=None if kind is None else dict(type=kind.__name__,message=str(error)),samples=self.samples))
