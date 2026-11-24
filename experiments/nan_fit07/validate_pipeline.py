"""Full fitting-pipeline engineering dry run on a tiny RANDOM 22-layer model."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
from experiments.nan_bpd01.backend import NativeLoop
from .io import read,rows,sha,write
from .worker import fit_one,target_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--backend',choices=('bf16','nf4'),default='bf16');p.add_argument('--passes',type=int,choices=(1,2),default=2);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve prior pipeline engineering attempts')
    out.mkdir(parents=True);started=time.monotonic();torch.manual_seed(2026112007)
    torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    root=Path(__file__).parent;cfg=read(root/'config.json');cfg.update(input_width=96,output_width=64)
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    loop=NativeLoop(copy.deepcopy(model),'cuda:0',quantized=a.backend=='nf4')
    if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()
    weight=loop.model.get_submodule(cfg['site']).weight
    if getattr(weight,'quant_state',None) is not None:
        from bitsandbytes.functional import dequantize_4bit
        W=dequantize_4bit(weight.data,quant_state=weight.quant_state).float().cpu()
    else:W=weight.detach().float().cpu()
    u,s,v=torch.linalg.svd(W,full_matrices=False)
    initial={'V8':(v[:8].T*s[:8].sqrt()).numpy(),'U8':(s[:8].sqrt()[:,None]*u[:,:8].T).numpy()}
    public=rows(root/'fixtures/public.jsonl');seq={r['id']:torch.randint(0,100,(19,)).tolist() for r in public}
    tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(a.snapshot)/'tokenizer.json')))
    source={'BPD_root':a.bpd_root,'manifest_sha256':'engineering_random_model_only_not_a_pretrained_result',
        'initial_svd_file_sha256':'generated_from_tiny_random_operator','effective_operator_sha256':'tiny_random_operator'}
    framework=load_bpd(a.bpd_root)
    def budget():
        if time.monotonic()-started>900:raise RuntimeError('Random engineering fixture exceeded its finite budget')
    result=fit_one(loop,tok,public,seq,cfg,a.backend,a.passes,out,initial,source,framework,budget,target_digest(loop))
    receipt={'status':'passed' if result['status']=='completed' else 'failed','evidence_class':'tiny random 22-layer worker-path engineering, not pretrained fitting',
        'pretrained_weights_loaded':False,'token_ids':'random 0..99; not source prompt inference','model_hidden_width':64,'model_layers':22,
        'backend':a.backend,'fit_passes':a.passes,
        'worker_condition_status':result['status'],'worker_failure':result.get('failure_message'),'completed_optimizer_steps':result['optimizer_steps_completed'],
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,
        'worker_source_sha256':sha(root/'worker.py'),'sealed_test_accessed':False}
    write(out/'engineering_receipt.json',receipt);print(receipt)
    if receipt['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
