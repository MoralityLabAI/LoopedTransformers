"""MOCK_ONLY actual BPD hooks with chunk/phase binding; no pretrained inference.

Preset gates: exact cached clean-A/all-one closure; masks rtol2e-5/atol1e-5;
independent local-write relative L2 <=.03, inherited from the original host.
Full-prefix versus cached BF16 differences are recorded, not equivalence-tested.
"""
import argparse
from dataclasses import asdict
import copy
import gc
import importlib
from pathlib import Path
import sys
from unittest.mock import patch
from .bounded_cpu_engineering import CPUBudget


def main():
    p=argparse.ArgumentParser()
    for key in ('snapshot','compactification-root','bpd-root','mock-factor-fixture','out-dir'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.out_dir);assert 'MOCK_ONLY' in out.name and 'MOCK_ONLY' in a.mock_factor_fixture
    with CPUBudget(out) as budget:
        import os
        import torch
        import numpy as np
        from transformers import AutoConfig,AutoModelForCausalLM
        from experiments.nan_ram28.common import read,write,sha
        from experiments.nan_bpd01.backend import NativeLoop
        from experiments.nan_full26.batch import install,Embeddings,BatchObserver,RowInstrument
        from experiments.nan_recomp09.common import target_digest
        from experiments.nan_story23.host import local_replay
        from experiments.nan_fit07.pointwise import masks as shadow_masks
        from .native_components import load_bpd
        from .resident_head import retain_decoder_layers
        from .lazy_bf16_loop import clear_unused_row_observations
        from .chunked_cache import ChunkedTwoLoopCache
        from .chunked_instrument import ChunkedRowInstrument
        from .streamed_vocab_head import StreamedVocabularyHead
        torch.set_num_threads(1);torch.manual_seed(2026112232);assert not torch.cuda.is_initialized()
        sys.path.insert(0,str(Path(a.compactification_root).resolve()));os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(a.snapshot).resolve())
        from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
        entry=load_lock()['nanbeige'];assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
        for name,digest in entry['files_sha256'].items():
            if not name.endswith('.safetensors'):assert sha(Path(a.snapshot)/name)==digest,name
        sources={name:sha(Path(__file__).with_name(name)) for name in ('chunked_cache.py','chunked_instrument.py','chunked_bpd_engineering.py','streamed_vocab_head.py','bounded_cpu_engineering.py')}
        write(out/'source_at_start.json',sources)
        config=AutoConfig.from_pretrained(a.snapshot,trust_remote_code=True,local_files_only=True)
        for key,value in dict(hidden_size=64,intermediate_size=128,num_hidden_layers=22,num_attention_heads=8,num_key_value_heads=2,head_dim=16,kv_channels=16,
                              vocab_size=256,max_position_embeddings=256,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.,pretraining_tp=1).items():setattr(config,key,value)
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=True,attn_implementation='sdpa',torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        clone=copy.deepcopy(model);reference=NativeLoop(model,'cpu');candidate=NativeLoop(clone,'cpu')
        retain_decoder_layers(reference,4);retain_decoder_layers(candidate,4);install(reference)
        expected=target_digest(reference);assert target_digest(candidate)==expected
        framework=load_bpd(a.bpd_root);site='model.layers.21.mlp.down_proj'
        ci_cfg=framework['config'].CIConfig(variant='state_only',n_blocks=0,d_model=32,mlp_hidden=64,max_passes=2,max_seq_len=32)
        with np.load(a.mock_factor_fixture,allow_pickle=False) as data:bank={name:data[name].copy() for name in data.files}
        assert bank['V'].shape==(128,8) and bank['U'].shape==(8,64)
        checkpoint=out/'factors_MOCK_ONLY.npz'
        with checkpoint.open('xb') as stream:np.savez(stream,**bank)
        ablation=asdict(framework['config'].AblationConfig())
        asset=dict(MOCK_ONLY=True,checkpoint=str(checkpoint),static_CAL_mask=np.asarray([[.25]*8,[.75]*8]),
            receipt=dict(MOCK_ONLY=True,target_parameter_sha256_unchanged=expected,training_contract=dict(ablation=ablation),final_checkpoint=dict(CI_config=asdict(ci_cfg))))
        cfg=dict(learned_site=site,component_token_chunk_rows=7)
        import experiments.nan_story23.instrument as original_instrument
        head=StreamedVocabularyHead(candidate.model.lm_head.weight,'cpu',7)
        reference_head=StreamedVocabularyHead(reference.model.lm_head.weight,'cpu',7)
        results=[];paired=vector_pairs=readout_checks=0
        def exact(left,right,rows=None):
            for key,x in left['states'].items():
                if rows is None or key[0] in rows:assert torch.equal(x,right['states'][key]),key
            for x,y in zip(left['boundaries']+left['logits'],right['boundaries']+right['logits']):
                if rows is None:assert torch.equal(x,y)
                else:
                    for row in rows:assert torch.equal(x[row],y[row])
        with patch.object(original_instrument,'fit_asset',return_value=asset):
            for fixture,initial in [('duplicate_AB',[[2,3,5,7,11,13,17]]*2),('ragged_rows',[[2,3,5,7,11,13,17],[19,23,29,31]])]:
                native_ref=native_cache=None
                for route in ('native','all_one','component_only','static_CAL_mean','reached_state'):
                    ref_i=RowInstrument(reference,'bf16',a.bpd_root,cfg);obs=BatchObserver(reference,(5,11,17,21));embedding=Embeddings(reference)
                    cache=ChunkedTwoLoopCache(candidate,prefill_chunk=3,inplace=True,instrument_factory=lambda loop:ChunkedRowInstrument(loop,'bf16',a.bpd_root,cfg))
                    sequences=[list(ids) for ids in initial];route_modes=['native',route];trace_ref=[];trace_cache=[];checks=[]
                    def ref_capture():
                        h,ctx=embedding.prepare(sequences);obs.values={};obs.enabled=True;obs.lengths=list(map(len,sequences));boundaries=[]
                        with torch.inference_mode(),ref_i.context(route_modes,obs.lengths):
                            for phase in (1,2):
                                reference.phase=phase;obs.phase=phase;h=reference.run_pairs([(h,ctx)])[0]
                                boundaries.append(torch.stack([h[b,n-1] for b,n in enumerate(obs.lengths)]).clone())
                        obs.enabled=False;result=dict(states=dict(obs.values),boundaries=boundaries)
                        clear_unused_row_observations(ref_i);return result
                    try:
                        for step in range(4):
                            if step in (1,2):
                                tokens=[41+step]*len(sequences)
                                for ids,t in zip(sequences,tokens):ids.append(t)
                            elif step==3:
                                cache.select_rows([1]);sequences=[sequences[1]];route_modes=[route_modes[1]];embedding.reset();tokens=[53];sequences[0].append(53)
                            r=ref_capture();c=cache.prefill(sequences,modes=route_modes) if step==0 else cache.step(tokens)
                            r['logits']=[reference_head(h) for h in r['boundaries']];c['logits']=[head(h) for h in c['boundaries']]
                            for record,weight in ((r,reference.model.lm_head.weight),(c,candidate.model.lm_head.weight)):
                                for h,score in zip(record['boundaries'],record['logits']):
                                    expected_score=torch.cat([torch.nn.functional.linear(h,weight[start:start+7]).float() for start in range(0,len(weight),7)],dim=-1)
                                    assert torch.equal(score,expected_score);readout_checks+=1
                            readout_error=max(float((x-y).abs().max()) for x,y in zip(r['logits'],c['logits']))
                            assert r['states'].keys()==c['states'].keys();paired+=1;vector_pairs+=len(c['states'])
                            error=max(float((r['states'][k].float()-c['states'][k].float()).abs().max()) for k in r['states'])
                            inst=cache.instrument;assert len(inst.native)==len(inst.inputs)==2*len(sequences)
                            for phase in (1,2):
                                for row in range(len(sequences)):
                                    assert torch.equal(inst.inputs[phase,row],c['states'][row,phase,21,'down_proj_input'])
                                    if route_modes[row]=='native':assert torch.equal(inst.native[phase,row],c['states'][row,phase,21,'down_proj_output'])
                            mask_error=write_error=0.
                            if route!='native':
                                edited_row=len(sequences)-1;assert len(inst.masks)==2
                                assert inst.applied_tokens=={1:len(sequences[edited_row]) if step==0 else 1,2:len(sequences[edited_row]) if step==0 else 1}
                                for phase in (1,2):
                                    x=inst.inputs[phase,edited_row].float().numpy();native=inst.native[phase,edited_row].float().numpy();actual=inst.masks[phase,edited_row].numpy()
                                    predicted=shadow_masks(x[None],bank,asdict(ci_cfg))[0] if route=='reached_state' else asset['static_CAL_mask'][phase-1] if route=='static_CAL_mean' else np.ones(8)
                                    np.testing.assert_allclose(actual,predicted,rtol=2e-5,atol=1e-5);mask_error=max(mask_error,float(np.max(np.abs(actual-predicted))))
                                    if route!='all_one':
                                        estimate=local_replay(x,native,actual,bank,ablation,phase-1)
                                        output=c['states'][edited_row,phase,21,'down_proj_output'].float().numpy()
                                        relative=float(np.linalg.norm(estimate-output)/max(float(np.linalg.norm(output)),1e-12));assert relative<=.03;write_error=max(write_error,relative)
                            if native_cache is not None:
                                if route=='all_one':exact(native_cache[step],c);exact(native_ref[step],r)
                                elif step<3:exact(native_cache[step],c,rows=[0])
                            checks.append(dict(step=step,full_prefix_vs_cached_max_abs=error,full_prefix_vs_cached_logit_max_abs=readout_error,maximum_mask_abs_error=mask_error,
                                maximum_local_write_relative_L2_error=write_error,phase_bound_applied_tokens=dict(inst.applied_tokens),
                                active_rows=len(sequences),physical_cache_length=cache.physical_length,semantic_lengths=list(cache.semantic_lengths)))
                            trace_ref.append(r);trace_cache.append(c)
                        if route=='native':native_ref=trace_ref;native_cache=trace_cache
                        elif route!='all_one':assert any(not torch.equal(trace_cache[0]['states'][k],native_cache[0]['states'][k]) for k in trace_cache[0]['states'] if k[0]==1)
                        results.append(dict(fixture=fixture,route=route,checks=checks,all_one_exact=True if route=='all_one' else None,
                            clean_A_exact=True if route!='native' else None,duplicate_raw_observer_retained=len(cache.instrument.base.raw)))
                    finally:cache.close();obs.close();ref_i.close();embedding.reset();del cache,obs,ref_i,embedding;gc.collect()
                    assert target_digest(reference)==target_digest(candidate)==expected;budget.check()
        assert not torch.cuda.is_initialized();assert sources=={name:sha(Path(__file__).with_name(name)) for name in sources}
        result=dict(MOCK_ONLY=True,status='passed_chunked_actual_BPD_CPU_hooks',source_sha256=sources,paired_captures=paired,paired_physical_vectors=vector_pairs,
            full_vocabulary_readouts_direct_stage_parity_checked=readout_checks,readout_vocabulary_size=256,head_stage_bytes=head.staging_bytes,
            results=results,synthetic_factor_bank=True,synthetic_CI_weights=True,qualified_real_factors_used=False,optimizer_updates=0,
            phase_labels_explicit_not_chunk_call_counts=True,last_valid_masks_and_local_writes_independently_replayed=True,
            parameter_sha256_before_after=expected,pretrained_forwards=0,CUDA_initialized=False,
            full_new_condition_registered=False,BF16_cached_native_equivalence_claimed=False,sealed_test_accessed=False,formal_comparisons=0)
    head.close();reference_head.close()
    resources=read(out/'resources.json');assert resources['status']=='passed'
    result['resources']=dict(samples=len(resources['samples']),maximum_sampled_private_bytes=max(v['private_bytes'] for v in resources['samples']),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in resources['samples']))
    write(out/'receipt_MOCK_ONLY.json',result);print({k:result[k] for k in ('status','paired_captures','paired_physical_vectors','resources')},flush=True)


if __name__=='__main__':main()
