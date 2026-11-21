"""Random-only production post-fit driver replay for all six closed fits."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_recomp09.common import target_digest
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers
from .common import read,rows,sha,write,key,condition
from .physical_probe import physical_probe
from .resources import below_normal


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--prior',default='runs/nan_ranksite13_random_v1');p.add_argument('--out',required=True);a=p.parse_args()
    below_normal();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    torch.set_num_threads(1);torch.manual_seed(2026112013);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    prior=Path(a.prior);proof=read(prior/'engineering_receipt.json');assert proof['status']=='passed' and not proof['pretrained_weights_loaded']
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':128,'num_attention_heads':8,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    loop=NativeLoop(model,'cuda:0',quantized=False);retain_decoder_layers(loop,4);loop.stage_layer.requires_grad_(False).eval()
    cpu=OperatorHead(model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);mount(loop,cpu);del cpu
    digest=target_digest(loop);public=rows('experiments/nan_fit07/fixtures/public.jsonl')
    selected=[next(r for r in public if r['split']==s and not r['copy_control']) for s in ('calibration','development')]
    rng=torch.Generator().manual_seed(2026112013);seq={r['id']:torch.randint(0,100,(19,),generator=rng).tolist() for r in selected}
    results=[]
    for index in range(6):
        cfg=condition(index);cfg.update(input_width=128,output_width=64,physical_probe_main_contexts=2)
        name=key(cfg);source=prior/name;receipt=read(source/'receipt.json');assert receipt['status']=='completed' and receipt['target_parameter_sha256_unchanged']==digest
        dest=out/'conditions'/name;dest.mkdir(parents=True,exist_ok=False)
        for field in ('initial_checkpoint','final_checkpoint'):
            filename=receipt[field]['file'];assert sha(source/filename)==receipt[field]['sha256'];os.link(source/filename,dest/filename)
        write(dest/'receipt.json',receipt);cell_started=time.monotonic();result=dict(condition_index=index,condition=name,status='failed')
        try:
            def budget():
                if time.monotonic()-cell_started>600:raise RuntimeError('Random post-fit cell budget exceeded')
            r=physical_probe(loop,selected,seq,cfg,dest,receipt,budget)
            assert target_digest(loop)==digest
            result.update(status='passed',signed_writes=r['local_signed_finite_writes'],physical_receipt_sha256=sha(dest/'physical_bridge/receipt.json'))
        except Exception as error:result.update(failure_type=type(error).__name__,failure_message=str(error));traceback.print_exc()
        results.append(result);print(result,flush=True);gc.collect();torch.cuda.empty_cache()
    root=Path(__file__).parent
    result=dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',results=results,
        source_sha256={n:sha(root/n) for n in ('physical_probe.py','basis.py','run_config.json','validate_postfit.py','POSTFIT_ENGINEERING_PLAN.md')},
        native_bridge_sha256=sha('mechinterp/nanbeige_bpd_gym/o_proj_bridge.py'),prior_engineering_receipt_sha256=sha(prior/'engineering_receipt.json'),
        target_parameter_sha256=digest,pretrained_weights_loaded=False,sealed_test_accessed=False,BPD_donor_transfer_attempted=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,CPU_threads=torch.get_num_threads(),formal_inferential_comparisons=0)
    write(out/'receipt.json',result);print({'status':result['status'],'cells':len(results)},flush=True)
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
