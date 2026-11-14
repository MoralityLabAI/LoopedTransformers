"""Bounded RSS/commit/VRAM monitoring for training and child Lean processes."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil

GIB = 1024**3

class ResourceLimit(RuntimeError):
    pass

def committed_available():
    if os.name != 'nt':
        return None
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys',
            'total_page', 'avail_page', 'total_virt', 'avail_virt', 'avail_ext')]
    m = Memory()
    m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx failed')
    return m.avail_page

def dedicated_vram():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=2, creationflags=0x08000000 if os.name == 'nt' else 0)
        if p.returncode == 0:
            total, used, free = (float(x.strip())*1024**2 for x in p.stdout.splitlines()[0].split(','))
            return dict(total_bytes=int(total), used_bytes=int(used), free_bytes=int(free))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None

def stop_tree(process):
    try:
        root = psutil.Process(process.pid)
        for child in root.children(recursive=True)[::-1]:
            try: child.kill()
            except psutil.NoSuchProcess: pass
        root.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait()

class ResourceGuard:
    def __init__(self, output: Path, torch_module=None, *, max_rss_gib=6.,
                 max_allocated_gib=2.5, min_available_gib=2., min_gpu_free_mib=512):
        self.output, self.torch = Path(output), torch_module
        self.max_rss = max_rss_gib*GIB
        self.max_allocated = max_allocated_gib*GIB
        self.min_available = min_available_gib*GIB
        self.min_gpu_free = min_gpu_free_mib*1024**2
        self.error = None
        self.peak = dict(tree_rss_bytes=0, cuda_allocated_bytes=0,
                         cuda_reserved_bytes=0, dedicated_used_bytes=0)
        self.low_available = psutil.virtual_memory().available
        self.done = threading.Event()
        self.last_gpu = None
        self.last_gpu_time = -float('inf')
        self.samples = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def sample(self, gpu=False):
        root = psutil.Process()
        rss = root.memory_info().rss
        for p in root.children(recursive=True):
            try: rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        available = psutil.virtual_memory().available
        commit = committed_available()
        allocated = reserved = 0
        if self.torch is not None and self.torch.cuda.is_initialized():
            allocated = self.torch.cuda.memory_allocated()
            reserved = self.torch.cuda.memory_reserved()
        if gpu or time.monotonic()-self.last_gpu_time >= 2:
            self.last_gpu = dedicated_vram()
            self.last_gpu_time = time.monotonic()
        row = dict(monotonic_seconds=time.monotonic(), tree_rss_bytes=rss,
                   available_ram_bytes=available, available_commit_bytes=commit,
                   cuda_allocated_bytes=allocated, cuda_reserved_bytes=reserved,
                   dedicated_vram=self.last_gpu)
        self.peak['tree_rss_bytes'] = max(self.peak['tree_rss_bytes'], rss)
        self.peak['cuda_allocated_bytes'] = max(self.peak['cuda_allocated_bytes'], allocated)
        self.peak['cuda_reserved_bytes'] = max(self.peak['cuda_reserved_bytes'], reserved)
        if self.last_gpu:
            self.peak['dedicated_used_bytes'] = max(self.peak['dedicated_used_bytes'], self.last_gpu['used_bytes'])
        self.low_available = min(self.low_available, available)
        self.samples += 1
        violations = []
        if rss >= self.max_rss: violations.append('process-tree RSS limit')
        if allocated >= self.max_allocated: violations.append('CUDA allocation limit')
        if available < self.min_available: violations.append('available RAM floor')
        if commit is not None and commit < self.min_available: violations.append('available commit floor')
        if self.last_gpu and self.last_gpu['free_bytes'] < self.min_gpu_free:
            violations.append('dedicated VRAM headroom floor')
        if violations:
            self.error = ResourceLimit(', '.join(violations))
            row['abort'] = str(self.error)
        with self.output.open('a', encoding='utf8', newline='\n') as f:
            f.write(json.dumps(row)+'\n')
        return row

    def check(self):
        if self.error is not None: raise self.error

    def _watch(self):
        while not self.done.wait(.1):
            try: self.sample()
            except Exception as exc:
                self.error = ResourceLimit('monitor failed: '+repr(exc))
            if self.error is not None: break

    def __enter__(self):
        self.sample(gpu=True)
        self.check()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=3)
        self.sample()

    def receipt(self):
        return dict(peaks=self.peak.copy(), minimum_available_ram_bytes=self.low_available,
                    samples=self.samples, monitor_period_seconds=.1,
                    nvidia_smi_period_seconds=2., abort=str(self.error) if self.error else None)

def guarded_process(command, log: Path, guard: ResourceGuard, timeout: float, cwd=None, env=None):
    """Redirect output to disk and kill the complete subprocess tree on expiry."""
    guard.check()
    log.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    timed_out = False
    with log.open('wb') as handle:
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                cwd=cwd, env=env,
                                creationflags=0x08000000 if os.name == 'nt' else 0)
        try:
            while proc.poll() is None:
                guard.check()
                if time.perf_counter()-began >= timeout:
                    timed_out = True
                    stop_tree(proc)
                    break
                time.sleep(.025)
        except BaseException:
            if proc.poll() is None: stop_tree(proc)
            raise
    guard.check()
    return dict(returncode=proc.returncode, timed_out=timed_out,
                wall_seconds=time.perf_counter()-began,
                log=log.read_text(encoding='utf8', errors='replace'))
