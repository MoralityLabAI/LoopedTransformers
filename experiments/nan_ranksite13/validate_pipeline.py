"""Random-only full eight-step pipelines for every missing rank/site cell."""
import argparse
import gc
import os
from pathlib import Path
import time
from types import SimpleNamespace
import traceback
import torch
from experiments.nan_bpd01.backend import NativeLoop
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_recomp09.common import read,rows,sha,write,target_digest
from .fit import fit_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p;k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    from transformers import PreTrainedTokenizerFast
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(2026112013);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats()
    root=Path(__file__).parent;cfg=read(root/'config.json')
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':128,'num_attention_heads':8,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    loop=NativeLoop(model,'cuda:0',quantized=False);retain_decoder_layers(loop,4)
    cpu=OperatorHead(model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);mount(loop,cpu);del cpu
    public=rows('experiments/nan_fit07/fixtures/public.jsonl');rng=torch.Generator().manual_seed(2026112013)
    seq={r['id']:torch.randint(0,100,(19,),generator=rng).tolist() for r in public}
    tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(a.snapshot)/'tokenizer.json')))
    source={'BPD_root':a.bpd_root,'manifest_sha256':'engineering_only_no_pretrained_weights','initial_svd_file_sha256':'random_small_operator','effective_operator_sha256':'random_small_operator'}
    original=target_digest(loop);results=[];framework=load_bpd(a.bpd_root)
    for condition in cfg['conditions']:
        local={**cfg,**condition,'input_width':128,'output_width':64,'maximum_added_saved_tensor_CPU_bytes':64*1024**2}
        cell_started=time.monotonic();result=dict(condition=condition,status='failed')
        try:
            weight=model.get_submodule(local['site']).weight.detach().float().cpu();assert weight.shape==(64,128)
            u,s,v=torch.linalg.svd(weight,full_matrices=False);rank=local['rank']
            initial={'V':(v[:rank].T*s[:rank].sqrt()).numpy(),'U':(s[:rank].sqrt()[:,None]*u[:,:rank].T).numpy()}
            def budget():
                if time.monotonic()-cell_started>1200:raise RuntimeError('Finite random-cell validation budget exceeded')
            cell=fit_one(loop,tok,public,seq,local,'bf16',local['passes'],out,initial,source,framework,budget,original)
            result.update(status='passed' if cell['status']=='completed' and cell['gauge_diagnostic']['stable_at_1e_minus5'] else 'failed',
                fit_status=cell['status'],optimizer_steps_completed=cell['optimizer_steps_completed'],failure_type=cell.get('failure_type'),failure_message=cell.get('failure_message'),
                gauge=cell.get('gauge_diagnostic'),closure=cell.get('initial_effective_operator_closure'),factor_update_l2=cell.get('factor_update_l2'),
                controller_update_l2=cell.get('controller_update_l2'),target_parameters_unchanged=target_digest(loop)==original)
            assert result['target_parameters_unchanged']
        except Exception as error:
            result.update(failure_type=type(error).__name__,failure_message=str(error));traceback.print_exc()
        results.append(result);write(out/'closed_condition_results.json',results)
        print({'module':local['module'],'rank':local['rank'],'passes':local['passes'],'status':result['status']},flush=True)
        gc.collect();torch.cuda.empty_cache()
    receipt=dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',results=results,
        pretrained_weights_loaded=False,evidence_class='random width64/22-layer six-cell full genuine-trainer engineering',
        source_sha256={name:sha(root/name) for name in ('fit.py','gauge.py','config.json','validate_pipeline.py','ENGINEERING_PLAN.md')},
        recomputation_source_sha256=sha('mechinterp/nanbeige_bpd_gym/recomputed_native.py'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        CPU_threads=torch.get_num_threads(),elapsed_seconds=time.monotonic()-started,sealed_test_accessed=False,BPD_donor_transfer_attempted=False,
        formal_inferential_comparisons=0)
    write(out/'engineering_receipt.json',receipt);print({'status':receipt['status'],'completed_cells':sum(r['status']=='passed' for r in results)},flush=True)
    if receipt['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
