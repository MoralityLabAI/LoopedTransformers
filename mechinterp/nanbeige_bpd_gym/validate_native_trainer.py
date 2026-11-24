"""Actual pinned BPDTrainer optimizer smoke on random native Nanbeige kernels.

This is engineering evidence only. No pretrained tensors, task gold, developer
corpora or held-out sources are opened. All four stock training terms run.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
import torch
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_gym05.common import write,sha
from .native_components import load_bpd,attach_native_components,native_rollout_adapter


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve earlier native trainer engineering attempts')
    out.mkdir(parents=True);start=time.monotonic();torch.manual_seed(2026112007)
    torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    framework=load_bpd(a.bpd_root);C=framework['config'];Trainer=framework['trainer'].BPDTrainer
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=2,random_init=True,
        config_overrides={'num_hidden_layers':2,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    sites={'model.layers.1.mlp.down_proj':8};tokens=torch.randint(0,100,(1,19));results=[]
    try:
        for backend in ('bf16','nf4'):
            for passes in (1,2):
                torch.manual_seed(2026112007)
                loop=NativeLoop(copy.deepcopy(model),'cuda:0',quantized=backend=='nf4')
                original=parameter_digest(loop.model)
                ablation=C.AblationConfig(mode='zero',sphere_rescale=False)
                instrument=attach_native_components(loop,sites,bpd_root=a.bpd_root,ablation=ablation,max_passes=passes)
                cfg=C.BPDConfig(c_per_module=sites,n_passes=passes,n_steps=2,batch_size=1,seq_len=19,seed=2026112007,
                    init_svd=False,faithfulness_warmup_steps=0,main_lr=.001,coeff_faith=1.,coeff_stoch=.5,coeff_adv=.5,
                    rollout_mode='closed_loop',readout_policy='readout_only',ablation=ablation,
                    ci=C.CIConfig(variant='state_only',d_model=16,n_blocks=1,n_heads=2,mlp_hidden=32,max_passes=passes,max_seq_len=32,clock_dim=8),
                    minimality=C.MinimalityConfig(pool_mode='mean' if passes==1 else 'max',coeff_switch=0. if passes==1 else .0002),
                    adversary=C.AdversaryConfig(scope='per_pass' if passes==1 else 'shared_batch_and_pass',inner_steps=2),
                    null_calibration=C.NullCalibrationConfig(use_tolerance_weights=False))
                trainer=Trainer(instrument,cfg,native_rollout_adapter(loop),is_causal_ci=True,device=loop.device)
                before=[p.detach().clone() for p in instrument.component_parameters()]
                ci_before=[p.detach().clone() for p in trainer.ci_net.parameters()]
                metrics=[asdict(trainer.train_step(tokens,step)) for step in range(2)]
                if not all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(float,int))):raise AssertionError('Nonfinite stock BPD training term')
                factor_change=sum(float((p.detach()-old).square().sum()) for p,old in zip(instrument.component_parameters(),before))**.5
                ci_change=sum(float((p.detach()-old).square().sum()) for p,old in zip(trainer.ci_net.parameters(),ci_before))**.5
                if factor_change<=0 or ci_change<=0:raise AssertionError('Both factors and the real CI network must update')
                if parameter_digest(loop.model)!=original:raise AssertionError('Native target parameters changed during factor training')
                if any(p.requires_grad or p.grad is not None for p in loop.model.parameters()):raise AssertionError('Target received a gradient')
                if loop.stage_layer is not None and any(p.requires_grad or p.grad is not None for p in loop.stage_layer.parameters()):raise AssertionError('Reusable target stage received a gradient')
                results.append({'backend':backend,'passes':passes,'steps':2,'metrics':metrics,'factor_update_l2':factor_change,'CI_update_l2':ci_change,
                    'target_parameter_sha256_unchanged':original,'head_remained_CPU':loop.model.lm_head.weight.device.type=='cpu',
                    'saved_tensor_statistics':loop._bpd_saved_tensors.statistics(),'training_contract':asdict(cfg),
                    'controller_training':'stock clean-trajectory-informed schedules; reached-state callback evaluation is a separate native condition',
                    'tolerance_scope':'uniform engineering weights; no mechanistic sham calibration or causal claim'})
                instrument.close();del trainer,instrument,loop,before,ci_before
                torch.cuda.empty_cache();print({'backend':backend,'passes':passes,'real_BPD_optimizer_steps':2,'status':'passed'},flush=True)
        root=Path(__file__).parent
        write(out/'receipt.json',{'status':'passed','evidence_class':'random-model genuine framework optimizer engineering, not trained pretrained-model factors',
            'pretrained_weights_loaded':False,'sealed_test_accessed':False,'BPD_transfer_attempted':False,'results':results,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':time.monotonic()-start,'source_sha256':{n:sha(root/n) for n in ('native_components.py','native_saved_tensors.py','validate_native_trainer.py','BPD_FRAMEWORK_LOCK.json')}})
    except Exception as e:
        write(out/'receipt.json',{'status':'failed','failure_type':type(e).__name__,'failure_message':str(e),
            'pretrained_weights_loaded':False,'sealed_test_accessed':False,'elapsed_seconds':time.monotonic()-start})
        raise


if __name__=='__main__':main()
