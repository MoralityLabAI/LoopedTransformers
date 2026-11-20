"""Pinned pretrained worker: public prompts, teacher targets, no host gold."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.nan_bpd01.backend import load_backend
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_comp03.prompt import aliases
from experiments.nan_diag04.telemetry import Telemetry
from experiments.nan_gym05.common import prompt_ids
from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
from mechinterp.nanbeige_bpd_gym.native_components import load_bpd,attach_native_components,native_rollout_adapter
from .io import read,rows,sha,write


class BudgetReached(RuntimeError):pass


def target_digest(loop):
    digest=hashlib.sha256()
    for name,p in loop.model.named_parameters():
        digest.update(name.encode());data=p.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(data))
    return digest.hexdigest()


def token_record(logits,tok):
    token=int(logits[0,-1].argmax());z=float(torch.logsumexp(logits[0,-1].double(),0))
    return {'free_token_id':token,'free_token_text':tok.tok.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False),
        'full_log_normalizer':z,'free_token_logprob':float(logits[0,-1,token])-z}


@contextmanager
def capture_module(loop,site):
    """Capture actual reached preweight inputs/output after component hooks."""
    records={};raw={};layer=int(site.split('.')[2]);suffix='.'.join(site.split('.')[3:]);handles=[]
    def before_component(module,args,output):
        if loop.active_layer==layer:raw[loop.phase]=output
    def hook(module,args,output):
        if loop.active_layer==layer:
            if loop.phase in records:raise RuntimeError('Unexpected repeated B1 module invocation')
            records[loop.phase]=(args[0],output,raw[loop.phase])
    modules=[loop.model.get_submodule(site)]
    if loop.stage_layer is not None and layer>=loop.n_gpu_layers:modules.append(loop.stage_layer.get_submodule(suffix))
    for module in modules:
        handles.append(module.register_forward_hook(before_component,prepend=True))
        handles.append(module.register_forward_hook(hook))
    try:yield records
    finally:
        for handle in handles:handle.remove()


def save_arrays(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def checkpoint(folder,name,trainer,core,step):
    arrays={'V':core.V.detach().cpu().numpy(),'U':core.U.detach().cpu().numpy()}
    for field in ('mean_comp_acts','mean_delta_out','mean_write_rms','calibrated_passes'):
        arrays[field]=getattr(core,field).detach().cpu().numpy()
    for key,value in trainer.ci_net.state_dict().items():arrays['ci__'+key]=value.detach().cpu().numpy()
    parameter_names={id(p):'factor_'+n for n,p in (('V',core.V),('U',core.U))}
    parameter_names.update({id(p):'ci_'+n for n,p in trainer.ci_net.named_parameters()})
    for p,state in trainer.optimizer.state.items():
        for key,value in state.items():
            if torch.is_tensor(value):arrays['optimizer__'+parameter_names[id(p)]+'__'+key]=value.detach().cpu().numpy()
    # Keep persistent PGD tensor state without executable pickle.
    for key,value in vars(trainer.pgd).items():
        if torch.is_tensor(value):arrays['pgd__'+key]=value.detach().cpu().numpy()
        elif isinstance(value,dict):
            for sub,item in value.items():
                if torch.is_tensor(item):arrays['pgd__'+key+'__'+sub]=item.detach().cpu().numpy()
    arrays['pgd__step_count']=np.array(trainer.pgd.step_count)
    arrays['trainer_rng_state']=trainer.rng.get_state().cpu().numpy()
    arrays['torch_CPU_rng_state']=torch.get_rng_state().numpy()
    arrays['torch_CUDA_rng_state']=torch.cuda.get_rng_state().cpu().numpy()
    file=folder/(name+'.npz');save_arrays(file,arrays)
    return {'file':file.name,'sha256':sha(file),'optimizer_steps_completed':step,'CI_config':asdict(trainer.cfg.ci)}


def make_config(C,cfg,passes):
    return C.BPDConfig(c_per_module={cfg['site']:cfg['rank']},n_passes=passes,n_steps=cfg['fit_steps'],batch_size=1,
        seq_len=cfg['maximum_prompt_tokens'],seed=cfg['source_seed'],init_svd=False,
        faithfulness_warmup_steps=0,main_lr=cfg['main_lr'],main_lr_final_frac=cfg['main_lr_final_frac'],
        coeff_faith=cfg['coeff_faith'],coeff_stoch=cfg['coeff_stoch'],coeff_adv=cfg['coeff_adv'],
        rollout_mode='closed_loop',readout_policy='readout_only',
        ci=C.CIConfig(variant='state_only',d_model=cfg['CI_d_model'],n_blocks=0,n_heads=4,mlp_hidden=64,max_passes=passes,max_seq_len=cfg['maximum_prompt_tokens']),
        ablation=C.AblationConfig(mode='zero' if passes==1 else 'pass_mean',sphere_rescale=passes==2),
        minimality=C.MinimalityConfig(pool_mode='mean' if passes==1 else 'max',coeff_switch=0. if passes==1 else .0002),
        adversary=C.AdversaryConfig(scope='per_pass' if passes==1 else 'shared_batch_and_pass',inner_steps=2),
        null_calibration=C.NullCalibrationConfig(use_tolerance_weights=passes==2,calibrate_every=0,n_rotations_train=4,perturbation_scale=.25))


def clean_masks(trainer,trajectory,site):
    # Keep evaluation GEMM shapes identical to each reached-state callback.
    x=trajectory.module_inputs[site]
    return torch.stack([trainer.ci_net({site:x[:,t:t+1]},pass_offset=t).lower_leaky[site][:,0] for t in range(x.shape[1])],dim=1)


@torch.no_grad()
def exact_means(trainer,loop,core,cal,seq,cfg,check_budget):
    per_root=[]
    for row in cal:
        check_budget()
        with capture_module(loop,cfg['site']) as actual:
            trainer.run_clean_rollout(seq[row['id']],update_means=False)
        mean=[]
        for phase in range(trainer.cfg.n_passes):
            x,y,_=actual[phase];acts=x.float()@core.V
            clean=y.float()-(core.bias.float() if core.bias is not None else 0.)
            mean.append((acts.mean((0,1)),(clean-acts@core.U).mean((0,1)),clean.square().mean(-1).add(core.ablation.eps).sqrt().mean()))
        per_root.append([[v.detach().cpu().double() for v in phase] for phase in mean])
    for phase in range(trainer.cfg.n_passes):
        for index,field in enumerate(('mean_comp_acts','mean_delta_out','mean_write_rms')):
            getattr(core,field)[phase].copy_(torch.stack([r[phase][index] for r in per_root]).mean(0).to(loop.device,dtype=torch.float32))
        core.calibrated_passes[phase]=True


@torch.no_grad()
def diagnose(trainer,loop,core,public,seq,tok,cfg,modes,check_budget):
    site=cfg['site'];passes=trainer.cfg.n_passes;Schedule=importlib.import_module('mechinterp.bpd.components').PassMaskSchedule
    main=[r for r in public if not r['copy_control']];static=[]
    for row in main:
        if row['split']!='calibration':continue
        check_budget();core.live_controller=None;clean=trainer.run_clean_rollout(seq[row['id']],update_means=False)
        g=clean_masks(trainer,clean,site)
        static.append(g.mean((0,2)))
    static=torch.stack(static).mean(0);records=[]
    for row in main:
        check_budget();core.live_controller=None
        with capture_module(loop,site) as observed:
            clean=trainer.run_clean_rollout(seq[row['id']],update_means=False)
        clean_last={t:(x[0,-1].float().cpu(),y[0,-1].float().cpu()) for t,(x,y,_) in observed.items()}
        clean_g=clean_masks(trainer,clean,site)
        length=len(seq[row['id']])
        for mode in modes:
            check_budget();core.live_controller=None
            if mode in ('all_one','component_only'):mask=torch.ones_like(clean_g)
            elif mode=='static_CAL_mean':mask=static[None,:,None,:].expand(1,passes,length,cfg['rank'])
            else:mask=clean_g
            residual=torch.ones((1,passes,length),device=loop.device) if mode=='all_one' else torch.zeros((1,passes,length),device=loop.device)
            if mode=='reached_state':
                core.live_controller=lambda x,tau:trainer.ci_net({site:x.unsqueeze(1)},pass_offset=tau).lower_leaky[site][:,0]
            try:
                with capture_module(loop,site) as actual:
                    changed=trainer.run_masked_rollout(seq[row['id']],{site:Schedule(mask,residual)},n_passes=passes)
                phases=[]
                for t in range(passes):
                    x,y,raw=actual[t];cx,cy=clean_last[t];nowx=x[0,-1].float().cpu();nowy=y[0,-1].float().cpu();native_y=raw[0,-1].float().cpu()
                    gap=float((nowx-cx).norm()/cx.norm().clamp_min(1e-12))
                    if mode=='all_one' and not torch.equal(changed.readouts[t],clean.readouts[t]):raise AssertionError('All-one diagnostic changed native logits')
                    mask_gap=None
                    if mode=='reached_state':
                        got=core.controller_observations[t];mask_gap=float((got['mask']-clean_g[:,t]).abs().max())
                        if passes==1 and (gap or mask_gap):raise AssertionError('T=1 reached controller must see the clean input')
                    phases.append({'pass_index_zero_based':t,**token_record(changed.readouts[t],tok),
                        'full_vocabulary_KL':float(importlib.import_module('mechinterp.bpd.losses').kl_divergence_logits(changed.readouts[t],clean.readouts[t])),
                        'actual_last_token_module_write_relative_l2':float((nowy-cy).norm()/cy.norm().clamp_min(1e-12)),
                        'local_intervention_l2_at_actual_reached_input':float((nowy-native_y).norm()),
                        'local_intervention_relative_l2_at_actual_reached_input':float((nowy-native_y).norm()/native_y.norm().clamp_min(1e-12)),
                        'reached_input_relative_l2':gap,'reached_vs_clean_mask_max_abs':mask_gap})
                records.append({'id':row['id'],'root':row['root'],'split':row['split'],'mode':mode,'phases':phases})
            finally:core.live_controller=None
    return {'records':records,'static_CAL_mask':static.cpu().tolist(),'scope':'unmatched-dose fidelity diagnostics, not a causal comparison'}


@torch.no_grad()
def operator_closure(core,cfg):
    generator=torch.Generator(device=core.V.device).manual_seed(cfg['source_seed'])
    x=torch.randn((4,core.d_in),device=core.V.device,generator=generator)
    target=core.W_target.float();approx=(core.V@core.U).T;residual=target-approx
    direct=x@target.T;closed=(x@core.V)@core.U+x@residual.T
    relative=float((direct-closed).norm()/direct.norm().clamp_min(1e-12))
    if relative>cfg['effective_operator_closure_relative_l2_tolerance']:raise AssertionError('Effective operator plus residual closure failed')
    return {'relative_l2':relative,'residual_Frobenius_ratio':float(residual.norm()/target.norm().clamp_min(1e-12)),
        'scope':'float32 backend-effective operator; distinct from native all-one kernel closure'}


@torch.no_grad()
def gauge_diagnostic(core,trainer):
    # Exact positive power-of-two scaling and coordinated component relabeling.
    # Test physical writes in float32, not unidentifiable hard-mask indices.
    x=torch.linspace(-1,1,core.d_in,device=core.V.device)[None,None,:]
    mask=torch.linspace(.1,.9,core.C,device=core.V.device)[None,None,:]
    initial=(x@core.V)*mask@core.U
    site=trainer.bpd_model.module_order[0]
    g=trainer.ci_net({site:x.unsqueeze(1)}).lower_leaky[site][:,0]
    controlled=(x@core.V*g)@core.U
    saved_v=core.V.clone();saved_u=core.U.clone();saved_mean=core.mean_comp_acts.clone()
    output_weight=trainer.ci_net.out_proj.weight.clone();output_bias=trainer.ci_net.out_proj.bias.clone()
    scales=torch.tensor([.5,1.,2.,4.,.25,8.,2.,1.],device=core.V.device);permutation=torch.arange(core.C-1,-1,-1,device=core.V.device)
    try:
        core.V.copy_((saved_v*scales)[:,permutation]);core.U.copy_((saved_u/scales[:,None])[permutation])
        core.mean_comp_acts.copy_((saved_mean*scales)[...,permutation])
        trainer.ci_net.out_proj.weight.copy_(output_weight[permutation]);trainer.ci_net.out_proj.bias.copy_(output_bias[permutation])
        changed=((x@core.V)*mask[...,permutation])@core.U
        relative=float((changed-initial).norm()/initial.norm().clamp_min(1e-12))
        changed_g=trainer.ci_net({site:x.unsqueeze(1)}).lower_leaky[site][:,0]
        control_error=float((changed_g-g[...,permutation]).abs().max())
        controlled_changed=(x@core.V*changed_g)@core.U
        controlled_error=float((controlled_changed-controlled).norm()/controlled.norm().clamp_min(1e-12))
    finally:
        core.V.copy_(saved_v);core.U.copy_(saved_u);core.mean_comp_acts.copy_(saved_mean)
        trainer.ci_net.out_proj.weight.copy_(output_weight);trainer.ci_net.out_proj.bias.copy_(output_bias)
    return {'positive_diagonal_and_permutation_physical_write_relative_l2':relative,
        'coordinated_controller_mask_max_abs':control_error,'controlled_write_relative_l2':controlled_error,
        'stable_at_1e_minus5':max(relative,control_error,controlled_error)<=1e-5,
        'scope':'finite float32 operator diagnostic; general rotations/hard-mask component identities are not certified'}


def fit_one(loop,tok,public,seq,cfg,backend,passes,out,initial,source,framework,check_budget,target_sha):
    cell=out/f't{passes}_layer21_down_proj_rank8';cell.mkdir(exist_ok=False)
    torch.manual_seed(cfg['source_seed']);C=framework['config'];contract=make_config(C,cfg,passes)
    instrument=attach_native_components(loop,contract.c_per_module,bpd_root=source['BPD_root'],ablation=contract.ablation,max_passes=passes)
    core=instrument.wrappers[cfg['site']];trainer=None;steps=[];cell_started=time.monotonic()
    receipt={'backend':backend,'passes':passes,'rank':cfg['rank'],'layer_zero_based':21,'module':'down_proj',
        'source_manifest_sha256':source['manifest_sha256'],'initial_svd_file_sha256':source['initial_svd_file_sha256'],
        'effective_operator_sha256':source['effective_operator_sha256'],'BPD_donor_transfer_attempted':False,'sealed_test_accessed':False}
    try:
        with torch.no_grad():core.V.copy_(torch.from_numpy(initial['V8']).to(loop.device));core.U.copy_(torch.from_numpy(initial['U8']).to(loop.device))
        if (core.d_in,core.d_out)!=(cfg['input_width'],cfg['output_width']):raise ValueError('Pinned physical module dimensions changed')
        if core.bias is not None:raise ValueError('The registered bias-free physical projection changed')
        receipt['initial_effective_operator_closure']=operator_closure(core,cfg)
        trainer=framework['trainer'].BPDTrainer(instrument,contract,native_rollout_adapter(loop,chunk_rows=cfg['vocabulary_chunk_rows']),is_causal_ci=True,device=loop.device)
        before_parameters=[p.detach().clone() for p in instrument.component_parameters()]
        before_ci=[p.detach().clone() for p in trainer.ci_net.parameters()]
        cal=[r for r in public if r['split']=='calibration' and not r['copy_control']];assert len(cal)==8
        first=seq[cal[0]['id']];Schedule=framework['components'].PassMaskSchedule;length=len(first)
        with torch.no_grad():reference=trainer.run_clean_rollout(first,update_means=False)
        with instrument.masked_components({cfg['site']:Schedule(torch.ones((1,passes,length,8),device=loop.device),torch.ones((1,passes,length),device=loop.device))}):
            closure=trainer.rollout_adapter(first,passes,0,None)[0]
            if any(not torch.equal(closure[t].detach(),reference.readouts[t]) for t in range(passes)):raise AssertionError('Autograd-enabled native all-one closure is not bit exact')
        del closure,reference;gc.collect();torch.cuda.empty_cache()
        # Full-size real pretrained gradient gate precedes optimizer changes.
        with torch.no_grad():teacher=trainer.run_clean_rollout(first,update_means=False)
        trainer.optimizer.zero_grad(set_to_none=True)
        probe=trainer.run_masked_rollout(first,{cfg['site']:Schedule(torch.full((1,passes,length,8),.35,device=loop.device),torch.full((1,passes,length),.7,device=loop.device))},n_passes=passes)
        probe_loss=framework['losses'].trajectory_recon_loss(probe,teacher,readout_policy='readout_only')[0];probe_loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() or not bool(p.grad.norm()>0) for p in (core.V,core.U)):raise AssertionError('Real pretrained factor gradient gate failed')
        receipt['pre_optimizer_gradient_gate_loss']=float(probe_loss.detach());receipt['native_all_one_full_head_bit_exact']=True
        del probe,teacher,probe_loss;trainer.optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        exact_means(trainer,loop,core,cal,seq,cfg,check_budget)
        nulls=importlib.import_module('mechinterp.bpd.nulls');calibration=[]
        with torch.no_grad():
            for i,row in enumerate(cal):
                check_budget();clean=trainer.run_clean_rollout(seq[row['id']],update_means=False)
                def run():
                    readouts,boundaries,vis,prompt=trainer.rollout_adapter(seq[row['id']],passes,0,None)
                    return framework['components'].RolloutTrajectory(readouts,boundaries,{},vis,prompt)
                tolerance=nulls.calibrate_pass_tolerances(instrument,clean,run,n_rotations=4,perturbation_scale=.25,
                    eval_at_final_readout=False,eps=cfg['tolerance_epsilon'],seed=cfg['source_seed']+i)
                calibration.append({'id':row['id'],'same_pass_tolerances':tolerance})
        trainer.pass_tolerances={t:sum(r['same_pass_tolerances'][t] for r in calibration)/8 for t in range(passes)} if passes==2 else {0:1.}
        receipt['training_tolerances']=trainer.pass_tolerances;receipt['calibration_unit_count']=8;receipt['calibration_rotations_per_unit_per_pass']=4
        write(cell/'calibration_local.json',calibration)
        receipt['initial_checkpoint']=checkpoint(cell,'initial',trainer,core,0)
        initial_diagnostic=diagnose(trainer,loop,core,public,seq,tok,cfg,['component_only'],check_budget)
        write(cell/'initial_diagnostic_local.json',initial_diagnostic)
        for step,row in enumerate(cal):
            check_budget();metrics=asdict(trainer.train_step(seq[row['id']],step))
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float))):raise AssertionError('Nonfinite native fit loss')
            if any(not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()) for p in instrument.component_parameters()+list(trainer.ci_net.parameters())):raise AssertionError('Nonfinite factor/controller parameter or gradient')
            steps.append({'step':step,'source_id':row['id'],**metrics});write(cell/'steps_local.json',steps)
            print({'backend':backend,'passes':passes,'step':step+1,'total_loss':metrics['total_loss']},flush=True)
        receipt['training_EMA_checkpoint']=checkpoint(cell,'training_EMA',trainer,core,len(steps))
        exact_means(trainer,loop,core,cal,seq,cfg,check_budget)
        receipt['factor_update_l2']=sum(float((p.detach()-old).square().sum()) for p,old in zip(instrument.component_parameters(),before_parameters))**.5
        receipt['controller_update_l2']=sum(float((p.detach()-old).square().sum()) for p,old in zip(trainer.ci_net.parameters(),before_ci))**.5
        if receipt['factor_update_l2']<=0 or receipt['controller_update_l2']<=0:raise AssertionError('Both factors and controller must actually update')
        receipt['final_checkpoint']=checkpoint(cell,'final',trainer,core,len(steps))
        with torch.no_grad():
            replay=trainer.run_clean_rollout(seq[cal[0]['id']],update_means=False)
            expected=clean_masks(trainer,replay,cfg['site'])
            save_arrays(cell/'CI_replay_local.npz',{'X':replay.module_inputs[cfg['site']][0,:,-1].float().cpu().numpy(),'masks':expected[0,:,-1].cpu().numpy()})
        receipt['CI_replay_sha256']=sha(cell/'CI_replay_local.npz')
        del replay,expected
        receipt['final_effective_operator_closure']=operator_closure(core,cfg)
        receipt['gauge_diagnostic']=gauge_diagnostic(core,trainer)
        diagnostic=diagnose(trainer,loop,core,public,seq,tok,cfg,cfg['diagnostic_modes'],check_budget)
        write(cell/'diagnostic_local.json',diagnostic)
        if target_digest(loop)!=target_sha:raise AssertionError('Pretrained target parameters changed')
        receipt|={'status':'completed','optimizer_steps_completed':len(steps),'target_parameter_sha256_unchanged':target_sha,'training_contract':asdict(contract),
            'saved_tensor_statistics':loop._bpd_saved_tensors.statistics(),'source_manifest_sha256':source['manifest_sha256'],
            'initial_svd_file_sha256':source['initial_svd_file_sha256'],'effective_operator_sha256':source['effective_operator_sha256'],
            'BPD_donor_transfer_attempted':False,'sealed_test_accessed':False}
        receipt['diagnostic_artifact_sha256']={name:sha(cell/name) for name in ('calibration_local.json','steps_local.json','initial_diagnostic_local.json','diagnostic_local.json')}
    except Exception as error:
        if trainer is not None:
            try:receipt['partial_checkpoint']=checkpoint(cell,'partial_preserved',trainer,core,len(steps))
            except Exception as save_error:receipt['partial_checkpoint_failure']={'type':type(save_error).__name__,'message':str(save_error)}
        receipt|={'status':'budget_reached' if isinstance(error,BudgetReached) else 'failed','failure_type':type(error).__name__,'failure_message':str(error),'optimizer_steps_completed':len(steps)}
        traceback.print_exc()
    finally:
        core.live_controller=None;instrument.close();receipt['elapsed_seconds']=time.monotonic()-cell_started
        receipt['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();receipt['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        write(cell/'receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--compactification-root',required=True);p.add_argument('--bpd-root',required=True);p.add_argument('--runs',default='runs/nan_fit07');a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');seal=read(root/'SOURCE_MANIFEST.json')
    for name,h in seal['worker_files_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Frozen fitting source changed: '+name)
    for name,h in seal['model_dependencies_sha256'].items():
        if sha(name)!=h:raise ValueError('Frozen model dependency changed: '+name)
    if sha(Path(a.snapshot)/'tokenizer_config.json')!=seal['tokenizer_config_sha256']:raise ValueError('Pinned tokenizer configuration changed')
    out=Path(a.runs)/a.backend;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def check_budget():
        if time.monotonic()-started>cfg['maximum_backend_seconds']:raise BudgetReached('Registered backend wall budget reached')
    torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
    peak={'allocated':0,'reserved':0}
    def keep_peak():
        peak['allocated']=max(peak['allocated'],torch.cuda.max_memory_allocated());peak['reserved']=max(peak['reserved'],torch.cuda.max_memory_reserved())
    public=rows(root/'fixtures/public.jsonl');assert len(public)==32 and all(set(r)=={'id','root','split','suite','prompt','schema','copy_control'} for r in public)
    status='failed';failure=None;cells=[]
    with Telemetry(out/'memory_samples.json'):
        try:
            assert_oracle_isolated();loop,old,tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
            if loop.stage_layer is not None:loop.stage_layer.requires_grad_(False).eval()
            target_sha=target_digest(loop);operator=OperatorHead(old.weight,aliases(tok),chunk_rows=cfg['vocabulary_chunk_rows'])
            seq={r['id']:prompt_ids(tok,a.snapshot,r['prompt']) for r in public}
            token_binding=read(root/'fixtures/tokenizer_receipt.json')
            if any(hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=token_binding['initial_prompt_ids_sha256'][key] for key,ids in seq.items()):raise ValueError('Frozen source prompt token IDs changed')
            if max(map(len,seq.values()))>cfg['maximum_prompt_tokens']:raise ValueError('No prompt truncation permitted')
            baseline=[];adapter=native_rollout_adapter(loop,chunk_rows=cfg['vocabulary_chunk_rows'])
            with torch.no_grad():
                for i,row in enumerate(public):
                    check_budget();logits,boundaries,_,_=adapter(seq[row['id']],2,0,None);phases=[]
                    for t in (0,1):
                        scored=operator.score(boundaries[t]);new=token_record(logits[t],tok)
                        if new['free_token_id']!=scored['free_top_token_id']:raise AssertionError('Full-head free argmax mismatch')
                        if abs(new['full_log_normalizer']-scored['full_log_normalizer'])>cfg['head_log_normalizer_absolute_tolerance']:raise AssertionError('Full-head normalizer mismatch')
                        if any(float(logits[t][0,-1,int(k)])!=v for k,v in scored['selected_logits'].items()):raise AssertionError('Selected head logits mismatch')
                        phases.append({'pass_index_zero_based':t,**new})
                    baseline.append({'id':row['id'],'root':row['root'],'split':row['split'],'copy_control':row['copy_control'],'phases':phases})
                    print({'backend':a.backend,'baseline_context':i+1},flush=True)
            write(out/'baseline.json',{'backend':a.backend,'records':baseline,'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                'public_sha256':sha(root/'fixtures/public.jsonl'),'head_parity_contexts':32,'target_parameter_sha256':target_sha,'host_gold_opened':False})
            while not (out/'host_gate.json').exists():check_budget();time.sleep(.5)
            gate=read(out/'host_gate.json')
            if gate['baseline_sha256']!=sha(out/'baseline.json') or gate['public_sha256']!=sha(root/'fixtures/public.jsonl') or gate['host_labels_sha256']!=seal['host_only_files_sha256']['fixtures/host_labels.jsonl']:raise ValueError('Unbound host gate')
            if not gate['allow_fits']:status='competence_gate_failed'
            else:
                bank=Path('runs/nan_gym05')/a.backend/'linear_baselines';initial_file=bank/'layer21_down_proj.npz'
                if sha(initial_file)!=seal['initialization'][a.backend]['file_sha256']:raise ValueError('Initial effective SVD changed')
                native=loop.model.get_submodule(cfg['site'])
                if getattr(native.weight,'quant_state',None) is not None:
                    from bitsandbytes.functional import dequantize_4bit
                    effective=dequantize_4bit(native.weight.data,quant_state=native.weight.quant_state).float()
                else:effective=native.weight.detach().float()
                effective_cpu=effective.cpu().contiguous().numpy()
                actual_hash=hashlib.sha256(memoryview(effective_cpu)).hexdigest()
                if actual_hash!=seal['initialization'][a.backend]['effective_operator_sha256']:raise ValueError('Current backend effective operator differs from its SVD bank')
                del effective,effective_cpu
                with np.load(initial_file,allow_pickle=False) as z:initial={k:z[k] for k in ('V8','U8')}
                framework=load_bpd(a.bpd_root);source={'BPD_root':a.bpd_root,'manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
                    'initial_svd_file_sha256':sha(initial_file),'effective_operator_sha256':seal['initialization'][a.backend]['effective_operator_sha256']}
                for passes in cfg['fit_passes']:
                    check_budget();keep_peak();torch.cuda.reset_peak_memory_stats();cells.append(fit_one(loop,tok,public,seq,cfg,a.backend,passes,out,initial,source,framework,check_budget,target_sha));keep_peak()
                    gc.collect();torch.cuda.empty_cache()
                    if target_digest(loop)!=target_sha:raise AssertionError('Frozen target changed between conditions')
                status='completed' if all(c['status']=='completed' for c in cells) else 'completed_with_failed_conditions'
            assert_oracle_isolated()
        except Exception as error:
            failure={'type':type(error).__name__,'message':str(error)};traceback.print_exc()
        finally:
            keep_peak()
            import transformers
            write(out/'receipt.json',{'backend':a.backend,'status':status,'failure':failure,'pid':os.getpid(),
                'source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'elapsed_seconds':time.monotonic()-started,
                'cells':[{k:c.get(k) for k in ('passes','status','optimizer_steps_completed','failure_type','failure_message')} for c in cells],
                'peak_cuda_allocated_bytes':peak['allocated'],'peak_cuda_reserved_bytes':peak['reserved'],
                'head_remained_CPU':bool('loop' in locals() and loop.model.lm_head.weight.device.type=='cpu'),
                'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'CUDA':torch.version.cuda,
                    'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
                'sealed_test_accessed':False,'BPD_donor_transfer_attempted':False})
    if status=='failed':raise SystemExit(2)


if __name__=='__main__':main()
