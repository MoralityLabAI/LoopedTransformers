"""Windows RAM and process-specific GPU allocation samples, not residency proofs."""
from datetime import datetime,timezone
import json
import os
import subprocess
from .common import RUNS,append
from experiments.nan_ram28.resources import memory,soft_guard


class ResourceAdmissionError(RuntimeError):pass


def gpu_memory(pid):
    if type(pid) is not int or pid<1:raise ValueError('An actual positive PID is required')
    query=(f"@(Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
        f"-Filter \"Name LIKE 'pid_{pid}_%'\" | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted) | ConvertTo-Json -Compress")
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',query],text=True,timeout=30,creationflags=flags)
    values=json.loads(raw) if raw.strip() else []
    if isinstance(values,dict):values=[values]
    return summarize_gpu(pid,values)


def summarize_gpu(pid,values):
    if not values:raise ResourceAdmissionError('Process-specific Windows GPU counters unavailable')
    for row in values:
        if not row['Name'].startswith(f'pid_{pid}_'):raise ResourceAdmissionError('GPU counter PID mismatch')
        for field in ('DedicatedUsage','SharedUsage','TotalCommitted'):
            if type(row[field]) is not int or row[field]<0:raise ResourceAdmissionError('Invalid GPU allocation counter')
    return dict(UTC=datetime.now(timezone.utc).isoformat(),pid=pid,instances=values,
        dedicated_bytes=sum(x['DedicatedUsage'] for x in values),shared_bytes=sum(x['SharedUsage'] for x in values),
        total_committed_bytes=sum(x['TotalCommitted'] for x in values),
        scope='All Windows GPU-process instances for this actual PID; allocation samples, not continuous peaks or cache-byte residency certificates')


def check_gpu(cfg,backend):
    sample=gpu_memory(os.getpid());append(RUNS/backend/'gpu_samples.jsonl',sample)
    if sample['dedicated_bytes']>cfg['maximum_sampled_dedicated_GPU_bytes'] or sample['shared_bytes']>cfg['maximum_sampled_shared_GPU_bytes']:
        raise ResourceAdmissionError('Sampled process GPU allocation budget exceeded')
    return sample
