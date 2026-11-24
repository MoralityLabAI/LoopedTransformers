"""MOCK_ONLY native-class chunked-cache CPU checks, zero pretrained forwards.

Preset structural gate: FP64 max absolute physical/boundary error <=1e-10.
BF16 differences are retained without an equivalence gate. This first fixture
checks native rows only; BPD and full-checkpoint CUDA admission remain pending.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
from .bounded_cpu_engineering import CPUBudget


def main():
    p=argparse.ArgumentParser()
    for key in ('snapshot','compactification-root','out-dir'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.out_dir);assert 'MOCK_ONLY' in out.name
    with CPUBudget(out) as budget:
        import os
        import torch
        from transformers import AutoConfig,AutoModelForCausalLM
        from experiments.nan_ram28.common import read,write,sha
        from experiments.nan_bpd01.backend import NativeLoop
        from experiments.nan_full26.batch import install,Embeddings,BatchObserver
        from experiments.nan_recomp09.common import target_digest
        from .resident_head import retain_decoder_layers
        from .chunked_cache import ChunkedTwoLoopCache
        torch.set_num_threads(1);assert not torch.cuda.is_initialized()
        sys.path.insert(0,str(Path(a.compactification_root).resolve()))
        os.environ['ARG02_SNAPSHOT_NANBEIGE']=str(Path(a.snapshot).resolve())
        from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
        entry=load_lock()['nanbeige'];assert entry['revision']=='451ed48c3273ecef7ea8faaa43c31ce529763bb1'
        for name,digest in entry['files_sha256'].items():
            if not name.endswith('.safetensors'):assert sha(Path(a.snapshot)/name)==digest,name
        sources={name:sha(Path(__file__).with_name(name)) for name in ('chunked_cache.py','chunked_cache_engineering.py','bounded_cpu_engineering.py')}
        write(out/'source_at_start.json',sources)
        config=AutoConfig.from_pretrained(a.snapshot,trust_remote_code=True,local_files_only=True)
        for key,value in dict(hidden_size=64,intermediate_size=128,num_hidden_layers=22,num_attention_heads=8,num_key_value_heads=2,
                              head_dim=16,kv_channels=16,vocab_size=256,max_position_embeddings=256,pad_token_id=0,bos_token_id=1,eos_token_id=2,
                              attention_dropout=0.,pretraining_tp=1).items():setattr(config,key,value)
        def model_at(dtype):
            torch.manual_seed(2026112232)
            return AutoModelForCausalLM.from_config(config,trust_remote_code=True,attn_implementation='sdpa',torch_dtype=dtype).eval().requires_grad_(False)
        results=[];paired_captures=paired_vectors=0
        for dtype in (torch.float64,torch.bfloat16):
            model=model_at(dtype);clone=copy.deepcopy(model)
            assert model.model.layers[0].self_attn.o_proj.weight.shape==(64,128)
            reference=NativeLoop(model,'cpu');candidate=NativeLoop(clone,'cpu')
            retain_decoder_layers(reference,4);retain_decoder_layers(candidate,4);install(reference)
            before=target_digest(reference);assert target_digest(candidate)==before
            obs=BatchObserver(reference,(5,11,17,21));embedding=Embeddings(reference)
            def ref_capture(sequences):
                h,ctx=embedding.prepare(sequences);obs.values={};obs.enabled=True;obs.lengths=list(map(len,sequences));boundaries=[]
                with torch.inference_mode():
                    for phase in (1,2):
                        reference.phase=phase;obs.phase=phase;h=reference.run_pairs([(h,ctx)])[0]
                        boundaries.append(torch.stack([h[b,n-1] for b,n in enumerate(obs.lengths)]).clone())
                obs.enabled=False;assert len(obs.values)==40*len(sequences)
                return dict(states=dict(obs.values),boundaries=boundaries)
            try:
                for inplace in (False,True):
                    for initial in ([[2,3,5,7,11,13,17]],[[2,3,5,7,11,13,17],[19,23,29,31]],[[19,23,29,31],[2,3,5,7,11,13,17]]):
                        cache=ChunkedTwoLoopCache(candidate,prefill_chunk=3,inplace=inplace)
                        sequences=[list(ids) for ids in initial];embedding.reset();rows=[]
                        def compare(r,c,label):
                            nonlocal paired_captures,paired_vectors
                            assert r['states'].keys()==c['states'].keys()
                            errors=[float((r['states'][k]-c['states'][k]).abs().max()) for k in r['states']]
                            errors.extend(float((x-y).abs().max()) for x,y in zip(r['boundaries'],c['boundaries']))
                            assert all(torch.isfinite(x).all() for x in c['states'].values())
                            maximum=max(errors)
                            if dtype==torch.float64:assert maximum<=1e-10,dict(dtype=str(dtype),label=label,max_abs=maximum)
                            item=dict(label=label,max_abs_across_40_sites_and_boundaries=maximum,
                                every_vector_bit_exact=all(torch.equal(r['states'][k],c['states'][k]) for k in r['states']),
                                cache_physical_length=cache.physical_length,semantic_lengths=list(cache.semantic_lengths),
                                cache_bytes=cache.cache_bytes())
                            expected=2*22*2*2*16*torch.empty((),dtype=dtype).element_size()*len(sequences)*cache.physical_length
                            assert cache.cache_bytes()==expected
                            rows.append(item);paired_captures+=1;paired_vectors+=len(c['states'])
                        try:
                            compare(ref_capture(sequences),cache.prefill(sequences),'prefill')
                            for step in range(2):
                                tokens=[41+step+b for b in range(len(sequences))]
                                for ids,token in zip(sequences,tokens):ids.append(token)
                                compare(ref_capture(sequences),cache.step(tokens),'incremental_'+str(step))
                            # Drop the longer original row in both ragged orientations.
                            selected=[0] if len(sequences)==1 or len(initial[0])<len(initial[-1]) else [1]
                            cache.select_rows(selected);sequences=[sequences[i] for i in selected];embedding.reset()
                            sequences[0].append(53)
                            compare(ref_capture(sequences),cache.step([53]),'after_row_pruning')
                            assert cache.stage_materializations==4*2*(22-4)
                            results.append(dict(dtype=str(dtype),inplace=inplace,initial_lengths=list(map(len,initial)),rows=rows,
                                stage_materializations=cache.stage_materializations,layer_chunk_calls=cache.layer_chunk_calls))
                        finally:cache.close();del cache
                assert target_digest(reference)==target_digest(candidate)==before
            finally:obs.close();embedding.reset()
            del reference,candidate,model,clone,obs,embedding;gc.collect();budget.check()
        assert sources=={name:sha(Path(__file__).with_name(name)) for name in sources}
        assert not torch.cuda.is_initialized()
        result=dict(MOCK_ONLY=True,status='passed_native_chunked_cache_CPU_structure',source_sha256=sources,
            architecture='Actual pinned random small class:22 layers, hidden64, intermediate128, query width128, kv width32, vocab256',
            paired_captures=paired_captures,paired_physical_vectors=paired_vectors,results=results,
            FP64_max_abs_gate=1e-10,BF16_native_equivalence_claimed=False,BPD_hook_integration_checked=False,
            prototype_is_registered_for_pretrained_use=False,pretrained_forwards=0,CUDA_initialized=False,
            real_generated_text_or_states_read=False,sealed_test_accessed=False,formal_comparisons=0)
    resources=read(out/'resources.json');assert resources['status']=='passed'
    result['resources']=dict(samples=len(resources['samples']),maximum_sampled_private_bytes=max(v['private_bytes'] for v in resources['samples']),
        minimum_sampled_available_physical_bytes=min(v['available_physical_bytes'] for v in resources['samples']))
    write(out/'receipt_MOCK_ONLY.json',result)
    print({k:result[k] for k in ('status','paired_captures','paired_physical_vectors','resources')},flush=True)


if __name__=='__main__':main()
