"""Random native T2 BPD parity against the old all-resident decoder reference."""
import argparse
import copy
from dataclasses import asdict
import gc
import math
import os
from pathlib import Path
import time
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_gym05.common import sha,write
from .native_components import load_bpd,attach_native_components,native_rollout_adapter
from .resident_head import mount,retain_decoder_layers
from .recomputed_native import attach_resident_components,ResidentSavedTensors,rollout_adapter
from .validate_native_trainer import parameter_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve prior engineering attempts')
    out.mkdir(parents=True);cache=out/'HF_cache';cache.mkdir();os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    torch.set_num_threads(1);torch.manual_seed(2026112009);torch.cuda.set_per_process_memory_fraction(.05);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    framework=load_bpd(a.bpd_root);C=framework['config'];Trainer=framework['trainer'].BPDTrainer;results=[]
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    tokens=torch.randint(0,100,(1,19));sites={'model.layers.21.mlp.down_proj':8}
    try:
        for backend in ('bf16','nf4'):
            loops=[];instruments=[];trainers=[];owners=[];original=[]
            ablation=C.AblationConfig(mode='pass_mean',sphere_rescale=True)
            cfg=C.BPDConfig(c_per_module=sites,n_passes=2,n_steps=2,batch_size=1,seq_len=19,seed=2026112009,init_svd=False,
                faithfulness_warmup_steps=0,main_lr=.0001,coeff_faith=1e5,coeff_stoch=.5,coeff_adv=.5,rollout_mode='closed_loop',readout_policy='readout_only',ablation=ablation,
                ci=C.CIConfig(variant='state_only',d_model=32,n_blocks=0,n_heads=4,mlp_hidden=64,max_passes=2,max_seq_len=32),
                minimality=C.MinimalityConfig(pool_mode='max',coeff_switch=.0002),
                adversary=C.AdversaryConfig(scope='shared_batch_and_pass',inner_steps=2),null_calibration=C.NullCalibrationConfig(use_tolerance_weights=False))
            for candidate in (False,True):
                loop=NativeLoop(copy.deepcopy(model),'cuda:0',quantized=backend=='nf4')
                if backend=='bf16':
                    retain_decoder_layers(loop,4 if candidate else 22)
                    if not candidate:loop.stage_layer=None
                if candidate:
                    cpu=OperatorHead(loop.model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192);head=mount(loop,cpu);del cpu
                    instrument=attach_resident_components(loop,sites,bpd_root=a.bpd_root,ablation=ablation,max_passes=2)
                    owner=ResidentSavedTensors(loop,64*1024**2,constants=[instrument.wrappers[n].W_target for n in sites])
                    adapter=rollout_adapter(loop,owner,sites)
                else:
                    instrument=attach_native_components(loop,sites,bpd_root=a.bpd_root,ablation=ablation,max_passes=2);owner=None
                    adapter=native_rollout_adapter(loop,offload_saved_tensors=False)
                trainer=Trainer(instrument,cfg,adapter,is_causal_ci=True,device=loop.device)
                loops.append(loop);instruments.append(instrument);trainers.append(trainer);owners.append(owner);original.append(parameter_digest(loop.model))
            for name in sites:instruments[1].wrappers[name].load_state_dict(instruments[0].wrappers[name].state_dict())
            trainers[1].ci_net.load_state_dict(trainers[0].ci_net.state_dict())
            clean=[]
            for trainer in trainers:
                with torch.no_grad():clean.append(trainer.run_clean_rollout(tokens,update_means=True))
            assert all(torch.equal(clean[0].readouts[t],clean[1].readouts[t]) and torch.equal(clean[0].boundary_states[t],clean[1].boundary_states[t]) for t in (0,1))
            Schedule=framework['components'].PassMaskSchedule;schedules={n:Schedule(torch.full((1,2,19,8),.35,device='cuda'),torch.full((1,2,19),.7,device='cuda')) for n in sites}
            outputs=[]
            for trainer,owner in zip(trainers,owners):
                with owner.context() if owner is not None else torch.enable_grad():
                    changed=trainer.run_masked_rollout(tokens,schedules,n_passes=2)
                    loss=framework['losses'].trajectory_recon_loss(changed,clean[0],readout_policy='readout_only')[0];loss.backward()
                    outputs.append({t:changed.readouts[t].detach().cpu() for t in (0,1)})
                del changed,loss
            assert all(torch.equal(outputs[0][t],outputs[1][t]) for t in (0,1))
            errors={}
            for name in sites:
                for label in ('V','U'):
                    a_grad=getattr(instruments[0].wrappers[name],label).grad;b_grad=getattr(instruments[1].wrappers[name],label).grad
                    assert a_grad is not None and b_grad is not None and torch.isfinite(a_grad).all() and torch.isfinite(b_grad).all()
                    assert bool(a_grad.norm()>0) and bool(b_grad.norm()>0)
                    torch.testing.assert_close(b_grad,a_grad,rtol=2e-4,atol=2e-7);errors[label]=float((a_grad-b_grad).abs().max())
            before=[p.detach().clone() for p in instruments[1].component_parameters()]
            ci_before=[p.detach().clone() for p in trainers[1].ci_net.parameters()]
            metrics=[[],[]]
            for step in range(2):
                for i,(trainer,owner) in enumerate(zip(trainers,owners)):
                    with owner.context() if owner is not None else torch.enable_grad():value=asdict(trainer.train_step(tokens,step))
                    assert all(math.isfinite(v) for v in value.values() if isinstance(v,(float,int)));metrics[i].append(value)
                for x,y in zip(instruments[0].component_parameters(),instruments[1].component_parameters()):torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-7)
                for x,y in zip(trainers[0].ci_net.parameters(),trainers[1].ci_net.parameters()):torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-7)
            factor_delta=sum(float((p.detach()-old).square().sum()) for p,old in zip(instruments[1].component_parameters(),before))**.5
            ci_delta=sum(float((p.detach()-old).square().sum()) for p,old in zip(trainers[1].ci_net.parameters(),ci_before))**.5
            assert factor_delta>0 and ci_delta>0
            for loop,digest in zip(loops,original):assert parameter_digest(loop.model)==digest
            stats=owners[1].statistics();assert stats['checkpointed_frozen_blocks']>0 and stats['recomputed_frozen_blocks']>0 and stats['resident_GPU_weight_references']>0
            assert instruments[1].wrappers[next(iter(sites))].calls==2
            results.append(dict(backend=backend,passes=2,optimizer_steps_per_reference_and_candidate=2,masked_head_bit_exact=True,clean_boundaries_bit_exact=True,
                factor_gradient_max_abs_errors=errors,factor_update_l2=factor_delta,controller_update_l2=ci_delta,metrics=metrics,saved_tensor_statistics=stats))
            print({'backend':backend,'status':'passed','recomputed_blocks':stats['recomputed_frozen_blocks'],'peak_CPU_snapshots':stats['peak_added_cpu_bytes']},flush=True)
            for instrument in instruments:instrument.close()
            del loops,instruments,trainers,owners,loop,trainer,instrument,owner,adapter,head,clean,outputs,before,ci_before
            gc.collect();torch.cuda.empty_cache()
        write(out/'receipt.json',dict(status='passed',evidence_class='random width-64/22-layer native T2 training and gradient parity; no pretrained weights',pretrained_weights_loaded=False,
            results=results,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,
            source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('recomputed_native.py','validate_recomputed_trainer.py')},sealed_test_accessed=False))
    except Exception as error:
        write(out/'receipt.json',dict(status='failed',failure_type=type(error).__name__,failure_message=str(error),pretrained_weights_loaded=False,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
