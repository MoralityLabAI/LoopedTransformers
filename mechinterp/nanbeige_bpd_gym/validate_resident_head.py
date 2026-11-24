"""Small random-weight GPU placement check; no pretrained model loading."""
import argparse
from pathlib import Path
import time
import torch
from experiments.nan_fit07.io import sha,write
from experiments.nan_comp02.head import OperatorHead
from .resident_head import ResidentHead,check_parity


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();started=time.monotonic()
    path=Path(a.out)
    if path.exists():raise FileExistsError('Preserve the previous engineering receipt')
    torch.set_num_threads(1);torch.manual_seed(2026112008);torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    weight=(torch.randn(166144,64)*.02).to(torch.bfloat16)
    cpu=OperatorHead(weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);gpu=ResidentHead(cpu);checks=[]
    for length in (1,7):
        h=torch.randn(1,length,64,device='cuda',dtype=torch.bfloat16)
        checks.append(check_parity(cpu,gpu,h))
    write(path,{'status':'passed','evidence_class':'random width-64 full-vocabulary placement engineering only',
        'pretrained_weights_loaded':False,'checks':checks,'resident_source_sha256':sha(Path(__file__).with_name('resident_head.py')),
        'validation_source_sha256':sha(__file__),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'elapsed_seconds':time.monotonic()-started,'sealed_test_accessed':False})
    print({'status':'passed','full_vocabulary_logits_checked':sum(r['full_vocabulary_logits_checked'] for r in checks),
        'pretrained_weights_loaded':False,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
