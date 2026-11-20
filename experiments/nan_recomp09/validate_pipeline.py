"""Run the complete derived eight-step recipe on a small random 22-layer model."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from experiments.nan_bpd01.backend import NativeLoop
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers
from experiments.nan_comp02.head import OperatorHead
from .common import read,rows,sha,write,target_digest
from .fit import fit_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from transformers import PreTrainedTokenizerFast
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(2026112009);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats()
    root=Path(__file__).parent;cfg=read(root/'config.json');cfg.update(input_width=96,output_width=64,maximum_added_saved_tensor_CPU_bytes=64*1024**2)
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    loop=NativeLoop(model,'cuda:0',quantized=False);retain_decoder_layers(loop,4)
    cpu=OperatorHead(model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);head=mount(loop,cpu);del cpu
    weight=model.get_submodule(cfg['site']).weight.detach().float().cpu();u,s,v=torch.linalg.svd(weight,full_matrices=False)
    initial={'V8':(v[:8].T*s[:8].sqrt()).numpy(),'U8':(s[:8].sqrt()[:,None]*u[:,:8].T).numpy()}
    public=rows('experiments/nan_fit07/fixtures/public.jsonl');seq={r['id']:torch.randint(0,100,(19,)).tolist() for r in public}
    tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(a.snapshot)/'tokenizer.json')))
    source={'BPD_root':a.bpd_root,'manifest_sha256':'engineering_only_no_pretrained_weights','initial_svd_file_sha256':'random_small_operator','effective_operator_sha256':'random_small_operator'}
    def budget():
        if time.monotonic()-started>1200:raise RuntimeError('Finite random pipeline validation budget exceeded')
    result=fit_one(loop,tok,public,seq,cfg,'bf16',2,out,initial,source,load_bpd(a.bpd_root),budget,target_digest(loop))
    receipt=dict(status='passed' if result['status']=='completed' else 'failed',pretrained_weights_loaded=False,
        evidence_class='random width-64/22-layer derived full BPD pipeline; not a pretrained fit',fit_status=result['status'],
        optimizer_steps_completed=result['optimizer_steps_completed'],failure_type=result.get('failure_type'),failure_message=result.get('failure_message'),
        source_sha256={name:sha(root/name) for name in ('fit.py','common.py','validate_pipeline.py')},
        recomputation_source_sha256=sha('mechinterp/nanbeige_bpd_gym/recomputed_native.py'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,sealed_test_accessed=False)
    write(out/'engineering_receipt.json',receipt);print(receipt)
    if receipt['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
