"""MOCK_ONLY random small pinned-class integration; no pretrained inference.

Exercises actual native/batched decoders, staged sources and BPD row hooks on
CPU. The synthetic factor receipt is local and explicitly mock; it is never
used by a real worker or a model grade. No frozen source is modified.
"""
import argparse
from dataclasses import asdict
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from .bounded_cpu_engineering import CPUBudget


def main():
    p=argparse.ArgumentParser()
    for name in ('snapshot','compactification-root','bpd-root','out-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out_dir).resolve();assert 'MOCK_ONLY' in out.name
    with CPUBudget(out) as budget:
        import os
        import torch
        import numpy as np
        from transformers import AutoConfig,AutoModelForCausalLM
        from experiments.nan_ram28.common import read,write,sha
        write(out/'attempt_sources.json',dict(MOCK_ONLY=True,source_files_sha256={name:sha(Path(__file__).with_name(name))
            for name in ('bounded_cpu_engineering.py','lazy_bf16_loop.py','lazy_bf16_backend.py','lazy_bf16_integration.py')}))
        from experiments.nan_recomp09.common import target_digest as original_digest
        from experiments.nan_bpd01.backend import NativeLoop
        from mechinterp.dark_fiber_gyms.arg03_nanbeige import _reinit_rotary_buffers
        from .resident_head import retain_decoder_layers
        from .native_components import load_bpd
        from .lazy_bf16_storage import CheckpointStore,logical_target_digest
        from .lazy_bf16_storage_tests import save_checkpoint
        from .lazy_bf16_loop import make_lazy_loop,digest_dispatch,clear_unused_row_observations
        from .lazy_bf16_backend import build_loop
        from .mlp_workspace27 import install as workspace_install
        from experiments.nan_full26.batch import install,BatchObserver,Embeddings,RowInstrument,logits
        torch.set_num_threads(1);torch.manual_seed(3232);assert not torch.cuda.is_initialized()
        compact=Path(a.compactification_root).resolve();sys.path.insert(0,str(compact))
        os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(a.snapshot).resolve())
        from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
        entry=load_lock()['nanbeige'];assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
        # Only model code/config is used. Checkpoint tensor files are not loaded.
        for name,h in entry['files_sha256'].items():
            if not name.endswith('.safetensors'):assert sha(Path(a.snapshot)/name)==h,name
        config=AutoConfig.from_pretrained(a.snapshot,trust_remote_code=True,local_files_only=True)
        for name,value in dict(hidden_size=64,intermediate_size=128,num_hidden_layers=22,num_attention_heads=4,
            num_key_value_heads=2,head_dim=16,vocab_size=256,max_position_embeddings=256,pad_token_id=0,bos_token_id=1,eos_token_id=2,
            attention_dropout=0.,pretraining_tp=1).items():setattr(config,name,value)
        config._attn_implementation='sdpa'
        def model_at(device):
            previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device(device):m=AutoModelForCausalLM.from_config(config,trust_remote_code=True,attn_implementation='sdpa')
            finally:torch.set_default_dtype(previous)
            return m.eval().requires_grad_(False)
        dense=model_at('cpu');meta=model_at('meta');root=out/'checkpoint_MOCK_ONLY';save_checkpoint(dense,root)
        store=CheckpointStore(root);store.validate(meta)
        expected=original_digest(SimpleNamespace(model=dense));assert logical_target_digest(meta,store)==expected
        selected=[name for name,p in meta.named_parameters() if name.startswith(tuple(f'model.layers.{i}.' for i in range(4)))]
        selected+=['model.embed_tokens.weight','model.norm.weight','lm_head.weight','model.layers.21.mlp.down_proj.weight']
        store.bind_parameters(meta,selected)
        assert sum(p.is_meta for p in meta.parameters())==161
        previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device('meta'):stage=type(dense.model.layers[0])(config,0)
        finally:torch.set_default_dtype(previous)
        stage.to_empty(device='cpu');_reinit_rotary_buffers(stage,torch.device('cpu'));stage.eval().requires_grad_(False)
        reference=NativeLoop(dense,'cpu',quantized=False);retain_decoder_layers(reference,4)
        uninitialized=[dict(layer=i,buffer=name,device=str(value.device)) for i in range(4)
            for name,value in meta.model.layers[i].named_buffers() if value.is_meta]
        assert uninitialized and all(v['buffer'].endswith('inv_freq') for v in uninitialized)
        try:make_lazy_loop(meta,store,'cpu',4,stage)
        except ValueError as error:assert str(error)=='Initialize resident prefix buffers on the execution device'
        else:raise AssertionError('Uninitialized resident buffers were admitted')
        for layer in meta.model.layers[:4]:_reinit_rotary_buffers(layer,torch.device('cpu'))
        write(out/'resident_buffer_fixture.json',dict(MOCK_ONLY=True,original_missing_buffers=uninitialized,
            missing_buffers_rejected_before_forward=True,initialization='Same native RoPE initialization already used by the pinned loaders.'))
        # The manual bad-buffer fixture above remains a rejection test. All
        # positive captures below now exercise the actual production builder.
        meta=model_at('meta')
        candidate=build_loop(meta,store,'cpu',n_resident=4,guard=budget.check)
        assert logical_target_digest(meta,store)==expected
        assert candidate.storage_layout['meta_parameters']==161
        sequences=[list(range(3,22)),list(range(3,20))]
        observations={};instruments={};workspaces={};embeddings={};capture_count=0
        for name,loop in (('reference',reference),('candidate',candidate)):
            workspaces[name]=workspace_install(loop,7);embeddings[name]=Embeddings(loop)
            observations[name]=BatchObserver(loop,[5,11,17,21])
        def capture(name,seq,modes,clear=False):
            nonlocal capture_count
            loop=reference if name=='reference' else candidate;obs=observations[name];embed=embeddings[name]
            embed.reset();h,ctx=embed.prepare(seq);obs.values={};obs.lengths=list(map(len,seq));obs.enabled=True
            instrument=instruments.get(name)
            from contextlib import nullcontext
            context=instrument.context(modes,obs.lengths) if instrument else nullcontext()
            boundaries=[];scores=[]
            with torch.inference_mode(),context:
                for phase in (1,2):
                    obs.phase=phase;loop.phase=phase;h=loop.run_pairs([(h,ctx)])[0]
                    last=torch.stack([h[b,n-1] for b,n in enumerate(obs.lengths)])
                    boundaries.append(last.detach().clone());scores.append(logits(SimpleNamespace(weight=loop.model.lm_head.weight,chunk=64),last))
            obs.enabled=False;assert len(obs.values)==40*len(seq)
            native=[torch.stack(v) for v in instrument.raw_by_phase] if instrument else []
            dropped=clear_unused_row_observations(instrument) if instrument and clear else 0
            capture_count+=1
            return dict(states=dict(obs.values),boundaries=boundaries,logits=scores,native=native,dropped=dropped)
        def equal(a,b):
            assert a['states'].keys()==b['states'].keys()
            entries=[(str(k),a['states'][k],b['states'][k]) for k in a['states']]
            entries += [(f'readout_or_native_{i}',x,y) for i,(x,y) in enumerate(zip(a['boundaries']+a['logits']+a['native'],b['boundaries']+b['logits']+b['native']))]
            for key,x,y in entries:
                if x.dtype!=y.dtype or not torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)):
                    write(out/'comparison_failure.json',dict(key=key,numerically_equal=bool(torch.equal(x,y)),maximum_abs=float((x.float()-y.float()).abs().max()),
                        left_dtype=str(x.dtype),right_dtype=str(y.dtype),capture_call=capture_count))
                    raise AssertionError('Exact bytes differ: '+key)
        # Original native B1, then the actual full-gym batch adapter at B1/B2.
        equal(capture('reference',sequences[:1],['native']),capture('candidate',sequences[:1],['native']))
        for loop in (reference,candidate):install(loop)
        for seq in (sequences[:1],sequences):equal(capture('reference',seq,['native']*len(seq)),capture('candidate',seq,['native']*len(seq)))
        for obs in observations.values():obs.close()
        framework=load_bpd(a.bpd_root);module=importlib.import_module('mechinterp.bpd.ci');site='model.layers.21.mlp.down_proj'
        ci_config=framework['config'].CIConfig(variant='state_only',n_blocks=0,d_model=32,mlp_hidden=64,max_passes=2,max_seq_len=32)
        ci=module.TiedCINetwork({site:128},{site:8},cfg=ci_config).eval()
        with torch.no_grad():ci.out_proj.bias.fill_(.4)
        rng=np.random.default_rng(3232)
        bank=dict(V=(rng.normal(size=(128,8))*.03).astype(np.float32),U=(rng.normal(size=(8,64))*.03).astype(np.float32),
            mean_comp_acts=np.zeros((2,8),np.float32),mean_delta_out=(rng.normal(size=(2,64))*.02).astype(np.float32),
            mean_write_rms=np.full(2,.07,np.float32),calibrated_passes=np.ones(2,dtype=bool))
        bank.update({'ci__'+name:value.detach().numpy() for name,value in ci.state_dict().items()})
        checkpoint=out/'factors_MOCK_ONLY.npz'
        with checkpoint.open('xb') as stream:np.savez(stream,**bank)
        asset=dict(MOCK_ONLY=True,checkpoint=str(checkpoint),static_CAL_mask=np.asarray([[.25]*8,[.75]*8]),
            receipt=dict(MOCK_ONLY=True,target_parameter_sha256_unchanged=expected,
                training_contract=dict(ablation=asdict(framework['config'].AblationConfig())),final_checkpoint=dict(CI_config=asdict(ci_config))))
        cfg=dict(learned_site=site,component_token_chunk_rows=7)
        import experiments.nan_story23.instrument as original_instrument
        dispatch_original=importlib.import_module('experiments.nan_recomp09.common').target_digest
        route_checks=[]
        try:
            with digest_dispatch() as digest,patch.object(original_instrument,'fit_asset',return_value=asset):
                assert digest(reference)==digest(candidate)==expected
                for name,loop in (('reference',reference),('candidate',candidate)):
                    instruments[name]=RowInstrument(loop,'bf16',a.bpd_root,cfg)
                    observations[name]=BatchObserver(loop,[5,11,17,21])
                # Reproduce legacy observer growth using the genuine constructor.
                growth=[]
                for repeat in range(3):
                    equal(capture('reference',sequences,['native','native']),capture('candidate',sequences,['native','native']))
                    growth.append(len(instruments['candidate'].base.raw))
                assert growth==[2,4,6]
                native_a=capture('reference',sequences,['native','native'])
                native_b=capture('candidate',sequences,['native','native'],clear=True);equal(native_a,native_b)
                assert len(instruments['candidate'].base.raw)==0 and len(native_b['native'])==2
                for route in ('all_one','component_only','static_CAL_mean','reached_state'):
                    r=capture('reference',sequences,['native',route]);c=capture('candidate',sequences,['native',route],clear=True);equal(r,c)
                    assert len(instruments['candidate'].base.raw)==0
                    assert all(torch.equal(c['states'][k],native_b['states'][k]) for k in c['states'] if k[0]==0)
                    if route=='all_one':equal(c,native_b)
                    else:assert any(not torch.equal(c['states'][k],native_b['states'][k]) for k in c['states'] if k[0]==1)
                    route_checks.append(dict(route=route,dense_lazy_bytes_exact=True,clean_row_unchanged=True,duplicate_observer_retained=0))
                assert digest(reference)==digest(candidate)==expected
        finally:
            for instrument in instruments.values():instrument.close()
            for obs in observations.values():obs.close()
            for workspace in workspaces.values():workspace.close()
        assert importlib.import_module('experiments.nan_recomp09.common').target_digest is dispatch_original
        assert not torch.cuda.is_initialized();budget.check()
        result=dict(MOCK_ONLY=True,status='passed_random_pinned_class_CPU_integration',
            architecture='Actual pinned decoder class with random small config:22 layers, width64, intermediate128, vocab256, four resident CPU layers plus a reusable CPU stage.',
            synthetic_factor_bank=True,synthetic_CI_weights=True,training_updates=0,qualified_real_factors_used=False,
            source_reference_weights_pretrained=False,pretrained_forwards=0,CUDA_initialized=False,
            CPU_capture_calls=capture_count,meta_parameters_retained=161,logical_target_before_after=expected,
            production_storage_builder_exercised=True,source_layout=candidate.storage_layout,
            uninitialized_resident_RoPE_buffers_rejected_before_forward=True,
            native_B1_and_batched_B1_B2_bytes_exact=True,all40_vectors_per_row_per_capture_checked=True,
            routes=route_checks,legacy_unused_observer_growth_per_native_capture=growth,
            clearing_duplicate_observer_changes_no_exported_states_logits_or_native_tensors=True,
            pinned_width_duplicate_observer_bytes_per_native_prefix=2*3072*4,
            duplicate_observer_byte_scope='Source-derived estimate for unused float32 vectors only, not a measured full-model heap-growth rate.',
            original_frozen_live_worker_modified=False,real_generated_text_or_states_read=False,sealed_test_accessed=False,
            BF16_inference_status='deferred_host_RAM_policy',full_new_condition_registered=False,
            candidate_stage_copies=candidate.stage_copies,largest_synthetic_source_tensor_bytes=candidate.maximum_source_tensor_bytes)
    from experiments.nan_ram28.common import read,write,sha
    resources=read(out/'resources.json');assert resources['status']=='passed'
    result['resources']=dict(resources_sha256=sha(out/'resources.json'),samples=len(resources['samples']),
        maximum_sampled_private_bytes=max(v['private_bytes'] for v in resources['samples']),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in resources['samples']))
    result['source_files_sha256']={name:sha(Path(__file__).with_name(name)) for name in ('bounded_cpu_engineering.py','lazy_bf16_loop.py','lazy_bf16_backend.py','lazy_bf16_integration.py')}
    write(out/'integration_receipt_MOCK_ONLY.json',result);print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':main()
