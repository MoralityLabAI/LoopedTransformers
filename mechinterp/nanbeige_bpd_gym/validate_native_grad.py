"""Small random Nanbeige GPU fixture; never loads pretrained weight tensors."""
import argparse
import copy
from pathlib import Path
import time
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_gym05.common import read,write,sha
from .native_components import load_bpd,attach_native_components,native_rollout_adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve an existing engineering attempt')
    out.mkdir(parents=True);started=time.monotonic();torch.manual_seed(2026112007)
    # This tiny random engineering process may overlap ONE real native worker.
    # It has a 10% allocation cap, no pretrained decoder and no stress loop.
    torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    framework=load_bpd(a.bpd_root);cfg=framework['config'].AblationConfig(mode='zero',sphere_rescale=False)
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':2,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,
            'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    sites={'model.layers.1.mlp.down_proj':8,'model.layers.1.self_attn.o_proj':8};results={}
    tokens=torch.randint(0,100,(1,19))
    try:
        for backend in ('bf16','nf4'):
            loops=[];instruments=[]
            for full in (False,True):
                loop=NativeLoop(copy.deepcopy(model),'cuda:0',quantized=backend=='nf4')
                if full and backend=='bf16':
                    for layer in loop.layers:layer.to(loop.device)
                    loop.n_gpu_layers=loop.n;loop.stage_layer=None
                instrument=attach_native_components(loop,sites,bpd_root=a.bpd_root,ablation=cfg)
                loops.append(loop);instruments.append(instrument)
            for name in sites:instruments[1].wrappers[name].load_state_dict(instruments[0].wrappers[name].state_dict())
            adapters=[native_rollout_adapter(loop,offload_saved_tensors=not full) for loop,full in zip(loops,(False,True))]
            Schedule=framework['components'].PassMaskSchedule
            def schedules(value,delta):return {name:Schedule(torch.full((1,2,19,rank),value,device='cuda:0'),torch.full((1,2,19),delta,device='cuda:0')) for name,rank in sites.items()}
            with torch.no_grad(),instruments[1].target_capture():
                clean,clean_states,_,_=adapters[1](tokens,2,0,None)
            with instruments[0].masked_components(schedules(1.,1.)):
                closure,closure_states,_,_=adapters[0](tokens,2,0,None)
                closure_error=max(float((closure[t].detach()-clean[t]).abs().max()) for t in (0,1))
                closure_state_error=max(float((closure_states[t].detach()-clean_states[t]).float().abs().max()) for t in (0,1))
            if closure_error or closure_state_error:raise AssertionError('Native all-one forward closure is not bit exact')
            del closure,closure_states
            losses=[];outputs=[]
            for instrument,adapter in zip(instruments,adapters):
                with instrument.masked_components(schedules(.35,.7)):
                    logits,states,_,_=adapter(tokens,2,0,None)
                    loss=sum(torch.nn.functional.kl_div(torch.log_softmax(logits[t],-1),torch.softmax(clean[t],-1),reduction='batchmean') for t in (0,1))/2
                    if not torch.isfinite(loss):raise AssertionError('Nonfinite native trajectory loss')
                    outputs.append({t:logits[t].detach().cpu() for t in (0,1)});losses.append(float(loss.detach()))
                    loss.backward()
                del logits,states,loss
            gradient_errors={};nonzero=True
            for name in sites:
                for label in ('V','U'):
                    ga=getattr(instruments[0].wrappers[name],label).grad;gb=getattr(instruments[1].wrappers[name],label).grad
                    if ga is None or gb is None or not torch.isfinite(ga).all() or not torch.isfinite(gb).all():raise AssertionError('Missing/nonfinite factor gradient')
                    nonzero=nonzero and bool(ga.norm()>0) and bool(gb.norm()>0)
                    torch.testing.assert_close(ga,gb,rtol=1e-4,atol=1e-7)
                    gradient_errors[name+'.'+label]={'max_abs':float((ga-gb).abs().max()),'relative_l2':float((ga-gb).norm()/gb.norm().clamp_min(1e-12))}
            if not nonzero:raise AssertionError('Factor gradients must be nonzero')
            forward_error=max(float((outputs[0][t]-outputs[1][t]).abs().max()) for t in (0,1))
            if forward_error:raise AssertionError('Saved-tensor offload changed the native masked forward')
            frozen=all(p.grad is None and not p.requires_grad for loop in loops for p in loop.model.parameters())
            frozen=frozen and all(p.grad is None and not p.requires_grad for loop in loops if loop.stage_layer is not None for p in loop.stage_layer.parameters())
            if not frozen:raise AssertionError('A pretrained/target parameter received a gradient')
            results[backend]={'all_one_head_max_abs':closure_error,'all_one_state_max_abs':closure_state_error,'masked_head_max_abs':forward_error,
                'trajectory_KL_losses':losses,'factor_gradient_errors':gradient_errors,'all_factor_gradients_finite_nonzero':nonzero,
                'target_parameters_frozen':frozen,'head_remained_CPU':all(loop.model.lm_head.weight.device.type=='cpu' for loop in loops),
                'saved_tensor_statistics':loops[0]._bpd_saved_tensors.statistics(),
                'offload_reference':'BF16 staged versus all resident; NF4 CPU-saved tensors versus resident graph'}
            for instrument in instruments:instrument.close()
            del instruments,loops,loop,instrument,adapters,adapter,ga,gb,outputs,clean,clean_states
            torch.cuda.empty_cache();print({'backend':backend,'engineering_status':'passed'},flush=True)
        root=Path(__file__).parent
        receipt={'status':'passed','evidence_class':'random-weight engineering parity, not a native trained-model or causal result',
            'pretrained_weights_loaded':False,'sealed_test_accessed':False,'BPD_transfer_attempted':False,
            'vocabulary_size':166144,'random_model_layers':2,'random_model_width':64,'loops':2,'source_contexts':1,'registered_sites':sites,
            'results':results,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'small_fixture_may_overlap_one_native_worker':True,'elapsed_seconds':time.monotonic()-started,
            'source_sha256':{n:sha(root/n) for n in ('native_components.py','native_saved_tensors.py','validate_native_grad.py','BPD_FRAMEWORK_LOCK.json')}}
        write(out/'receipt.json',receipt)
    except Exception as e:
        write(out/'receipt.json',{'status':'failed','failure_type':type(e).__name__,'failure_message':str(e),
            'pretrained_weights_loaded':False,'sealed_test_accessed':False,'elapsed_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
