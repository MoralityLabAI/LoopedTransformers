"""Native 22-layer random-model cache test; no pretrained tensors are loaded."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_gym05.observe import TowerObserver
from experiments.nan_fit07.io import sha,write
from .native_cached import NativeTwoLoopCache
from .resident_head import mount,retain_decoder_layers,full_logits
from .validate_native_trainer import parameter_digest


def relative(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir();os.environ['HF_HOME']=str(cache.resolve())
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats();started=time.monotonic();results=[]
    for backend in ('fp32_math','bf16','nf4'):
        torch.manual_seed(2026112011);loop=None;cached=None;observer=None
        result=dict(backend=backend,status='failed',rows=[]);results.append(result)
        try:
            dtype=torch.float32 if backend=='fp32_math' else torch.bfloat16;vocab=4096 if backend=='fp32_math' else 166144
            model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=dtype,n_gpu_layers=22,random_init=True,
                config_overrides={'vocab_size':vocab,'num_hidden_layers':22,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
            loop=NativeLoop(model,'cuda:0',quantized=backend=='nf4')
            if backend!='nf4':retain_decoder_layers(loop,4)
            if backend=='fp32_math':loop.model.lm_head.to(loop.device)
            else:
                cpu=OperatorHead(loop.model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);mount(loop,cpu);del cpu
            before=parameter_digest(loop.model);cached=NativeTwoLoopCache(loop);observer=TowerObserver(loop,(5,11,17,21))
            rng=torch.Generator().manual_seed(2026112011);ids=torch.randint(1,100,(19,),generator=rng).tolist();extra=torch.randint(1,100,(8,),generator=rng).tolist()
            for step in range(9):
                if step:ids.append(extra[step-1])
                with torch.inference_mode():
                    two,observed,one=observer.forward({'context':ids},capture=True);reference=[one[0][:,-1:],two[0][:,-1:]]
                    states,tape=cached.prefill(ids,capture=True) if step==0 else cached.step(extra[step-1],capture=True)
                    assert observed.keys()==tape.keys() and len(tape)==40
                    site_error=max(relative(tape[k],observed[k]) for k in tape)
                    for phase,(candidate,target) in enumerate(zip(states,reference),1):
                        expected=full_logits(loop.model.lm_head.weight,target,loop.device,8192);actual=full_logits(loop.model.lm_head.weight,candidate,loop.device,8192)
                        row=dict(step=step,loop=phase,boundary_relative_l2=relative(candidate,target),full_vocabulary_logit_max_abs=float((actual-expected).abs().max()),
                            free_argmax_equal=int(actual.argmax())==int(expected.argmax()),logits_checked=len(actual),
                            physical_tape_site_relative_l2_max=site_error,cache_bytes=cached.cache_bytes(),cache_sequence_length=cached.length,
                            prefill_boundary_bit_exact=torch.equal(candidate,target),prefill_logits_bit_exact=torch.equal(actual,expected))
                        result['rows'].append(row)
                        if step==0:
                            assert row['prefill_boundary_bit_exact'] and row['prefill_logits_bit_exact'] and all(torch.equal(tape[k],observed[k]) for k in tape)
                        else:
                            assert row['boundary_relative_l2']<=(1e-5 if backend=='fp32_math' else .02)
                            assert row['full_vocabulary_logit_max_abs']<=(1e-5 if backend=='fp32_math' else .1)
                    del states,tape,two,observed,one,reference,candidate,target,expected,actual
            assert parameter_digest(loop.model)==before
            result.update(status='passed',target_parameters_unchanged=True,physical_vectors_checked=9*40,vocabulary=vocab,
                independent_phase_cache_instances=True,incremental_free_argmax_agreements=sum(r['free_argmax_equal'] for r in result['rows'] if r['step']>0),incremental_readouts=16)
        except Exception as error:
            result.update(failure_type=type(error).__name__,failure_message=str(error));traceback.print_exc()
        finally:
            if cached is not None:cached.close()
            if observer is not None:observer.close()
            del cached,observer,loop
            if 'model' in locals():del model
            gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',evidence_class='random native 22-layer cache engineering; not pretrained or causal evidence',
        source_sha256=sha(Path(__file__)),cache_implementation_sha256=sha(Path(__file__).with_name('native_cached.py')),
        prospective_plan_sha256=sha(Path(__file__).with_name('KV_ENGINEERING_PLAN.md')),results=results,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,pretrained_weights_loaded=False,
        formal_inferential_comparisons=0,BPD_transfer_attempted=False,sealed_test_accessed=False)
    write(out/'receipt.json',result);print({k:result[k] for k in ('status','peak_cuda_allocated_bytes','elapsed_seconds','pretrained_weights_loaded')})
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
