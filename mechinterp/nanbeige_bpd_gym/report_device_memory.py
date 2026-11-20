"""Aggregate closed BF16 driver samples without claiming exact residency."""
import argparse
from pathlib import Path
from experiments.nan_fit07.io import read,rows,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_prime_ctx10');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    base=Path(a.runs);folder=base/'BF16_driver_memory_v1';sample=read(folder/'receipt.json');worker=read(base/'bf16/receipt.json');raw=rows(folder/'samples.jsonl')
    assert sample['status']==worker['status']=='completed' and worker['requests']==288 and sample['samples']==len(raw)==33
    assert sample['source_sha256']==sha(Path(__file__).with_name('sample_device_memory.py'))
    result={k:v for k,v in sample.items() if k!='source_sha256'}
    result.update(experiment='NAN-PRIME-CTX-10',backend='bf16',worker_receipt_sha256=sha(base/'bf16/receipt.json'),
        sampler_receipt_sha256=sha(folder/'receipt.json'),sampler_source_sha256=sample['source_sha256'],raw_samples_sha256=sha(folder/'samples.jsonl'),
        first_sample_UTC=raw[0]['UTC'],last_sample_UTC=raw[-1]['UTC'],
        CUDA_peak_allocated_bytes=worker['peak_cuda_allocated_bytes'],CUDA_peak_reserved_bytes=worker['peak_cuda_reserved_bytes'],
        sample_window_limitation='33 intermittent samples from part of the worker lifetime; not a full-lifetime dedicated/shared peak measurement',
        physical_interpretation='WDDM dedicated/shared counters are distinct from CUDA allocated/reserved bytes. Shared counters can include runtime/transfer memory and do not establish which tensors reside there.',
        NF4_driver_counter_comparison='not measured in this condition; prior partial NF4 CUDA peaks remain in its original receipt',
        privacy='aggregates/timestamps/hashes only; process IDs and raw adapter rows remain local')
    out=Path(a.out);write(out/'nan_prime_ctx10_BF16_memory.json',result)
    d=sample['sampled_worker_dedicated_bytes'];s=sample['sampled_worker_shared_bytes']
    text=('# CTX10 BF16: CUDA and driver memory measurements\n\n'+
        'The completed 288-episode BF16 worker used an unchanged BF16 head, four resident decoder layers and one reusable staging layer. '+
        'The math process was untouched; this read-only sampler ran Below Normal priority.\n\n'+
        '| Measurement | Value |\n|---|---:|\n'+
        f"| CUDA allocated peak | {worker['peak_cuda_allocated_bytes']/2**30:.3f} GiB |\n"+
        f"| CUDA reserved peak | {worker['peak_cuda_reserved_bytes']/2**30:.3f} GiB |\n"+
        f"| Sampled worker dedicated range | {d['minimum']/2**30:.3f}–{d['maximum']/2**30:.3f} GiB |\n"+
        f"| Sampled worker shared range | {s['minimum']/2**20:.2f}–{s['maximum']/2**20:.2f} MiB |\n"+
        f"| Driver samples / errors | {sample['samples']} / {sample['sample_error_count']} |\n\n"+
        'Driver counters were sampled every 30 seconds over part of the worker lifetime. They are not exact full-lifetime peaks '+
        'or per-tensor residency measurements. Shared usage may include runtime/transfer memory; it does not by itself show '+
        'model weights spilling out of dedicated VRAM. No matching NF4 driver-counter window was captured for this condition.\n\n'+
        'The original 168-case test stays sealed. [Aggregate counters and provenance](nan_prime_ctx10_BF16_memory.json).\n')
    (out/'NAN_PRIME_CTX10_BF16_MEMORY.md').write_text(text,encoding='utf-8',newline='\n');print({'status':'passed','samples':len(raw)})


if __name__=='__main__':main()
