"""Native full-prefix two-loop story generation; never reads host labels."""
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from .common import ROOT,append,read,sha,write
from .parser import parse
from .adapters import EmbeddingLookupCache


def ids_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


class Engine:
    def __init__(self,a,cfg,out,public):
        import torch
        from experiments.nan_bpd01.backend import load_backend
        from experiments.nan_comp02.head import OperatorHead
        from experiments.nan_comp03.prompt import aliases
        from experiments.nan_gym05.observe import TowerObserver
        from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers,check_parity
        from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
        from experiments.nan_recomp09.common import target_digest
        self.torch=torch;self.a=a;self.cfg=cfg;self.out=out;self.started=time.monotonic();self.bytes=0;self.instrument=None
        torch.set_num_threads(1);torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
        assert_oracle_isolated()
        self.loop,legacy,self.tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
        if self.loop.stage_layer is not None:self.loop.stage_layer.requires_grad_(False).eval()
        if a.backend=='bf16':retain_decoder_layers(self.loop,cfg['BF16_resident_decoder_layers'])
        cpu=OperatorHead(legacy.weight,aliases(self.tok),chunk_rows=cfg['vocabulary_chunk_rows']);self.head=mount(self.loop,cpu);del legacy
        assert self.loop.base.embed_tokens.weight.device.type=='cpu' and self.head.weight.device.type=='cuda'
        self.before=target_digest(self.loop);self.observer=TowerObserver(self.loop,cfg['layers_zero_based'])
        self.bindings=read(ROOT/'fixtures/tokenizer_receipt.json')
        for name,h in self.bindings['tokenizer_files_sha256'].items():assert sha(Path(a.snapshot)/name)==h
        cal=public[0];assert cal['split']=='calibration' and cal['arm']=='main' and cal['mode']=='native'
        initial=self.ids(cal)
        with torch.inference_mode():
            reference=self.capture(initial)
            parity=[check_parity(cpu,self.head,h) for h in reference['last_hidden']]
            token=reference['scores'][-1]['free_top_token_id']
            extension=self.capture(initial+[token])
            self.embedding_cache=EmbeddingLookupCache(self.loop.base.embed_tokens,self.loop.device,cfg['maximum_generated_tokens'])
            for ids,expected in ((initial,reference),(initial+[token],extension)):
                actual=self.capture(ids);self.exact(expected,actual)
        self.reference=reference;self.reference_ids=initial
        self.reference.pop('last_hidden');del cpu,extension,actual;gc.collect();self.embedding_cache.reset()
        write(out/'preflight.json',dict(source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            full_head_parity=parity,embedding_cache_initial_and_extension_exact=True,
            closure_scope='all40 physical vectors, normalized last-token boundaries, all166144 logits at both loops',
            CPU_embedding_weight_table=True,GPU_embedding_activation_cache=True,KV_cache=False,model_preflight_forwards=4))

    def ids(self,row):
        from experiments.nan_gym05.common import prompt_ids
        ids=prompt_ids(self.tok,self.a.snapshot,row['prompt']);bound=self.bindings['contexts'][row['id']]
        assert len(ids)==bound['tokens']<=self.cfg['maximum_prompt_tokens'] and ids_hash(ids)==bound['ids_sha256']
        return ids

    def capture(self,ids):
        from mechinterp.nanbeige_bpd_gym.resident_head import full_logits
        t=self.torch
        with t.inference_mode():
            two,states,one=self.observer.forward({'context':ids},capture=True)
            hidden=[one[0][:,-1:,:].clone(),two[0][:,-1:,:].clone()]
            boundaries=[h.detach().float().cpu().numpy().copy() for h in hidden]
            logits=[full_logits(self.head.weight,h,self.loop.device,self.cfg['vocabulary_chunk_rows']) for h in hidden]
            scores=[self.head.score(h) for h in hidden]
        return dict(states=states,boundaries=boundaries,logits=logits,scores=scores,last_hidden=hidden)

    def exact(self,a,b):
        import numpy as np
        assert a['states'].keys()==b['states'].keys()
        assert all(self.torch.equal(a['states'][k],b['states'][k]) for k in a['states'])
        assert all(np.array_equal(x,y) for x,y in zip(a['boundaries'],b['boundaries']))
        assert all(self.torch.equal(x,y) for x,y in zip(a['logits'],b['logits']))

    def install(self):
        from .instrument import Instrument
        from experiments.nan_gym05.observe import TowerObserver
        from experiments.nan_recomp09.common import target_digest
        self.observer.close();self.instrument=Instrument(self.loop,self.a.backend,self.a.bpd_root,self.cfg)
        self.observer=TowerObserver(self.loop,self.cfg['layers_zero_based']);self.embedding_cache.reset()
        with self.torch.inference_mode(),self.instrument.context('all_one',len(self.reference_ids)):
            actual=self.capture(self.reference_ids)
        self.instrument.snapshots();self.exact(self.reference,actual)
        assert target_digest(self.loop)==self.before
        write(self.out/'BPD_all_one_closure.json',dict(status='passed',all40_states_boundaries_and_full_logits_bit_exact=True,
            checkpoint_sha256=self.instrument.asset['checkpoint_sha256'],fit_audit_sha256=self.instrument.asset['audit_sha256'],
            target_unchanged=True,token_chunk_rows=self.cfg['component_token_chunk_rows'],edited_GEMM_bit_equivalence_claimed=False))
        self.reference=None;self.embedding_cache.reset();gc.collect()

    def budget(self,episode_started):
        if time.monotonic()-self.started>=self.cfg['maximum_backend_seconds']:raise RuntimeError('registered_backend_time_budget')
        if time.monotonic()-episode_started>=self.cfg['maximum_episode_seconds']:raise RuntimeError('registered_episode_time_budget')
        if self.bytes>=self.cfg['maximum_artifact_bytes_per_backend']:raise RuntimeError('registered_artifact_budget')
        if shutil.disk_usage(self.out).free<self.cfg['minimum_free_disk_bytes']:raise RuntimeError('registered_disk_free_floor')

    def generate(self,row,ordinal):
        import numpy as np
        from contextlib import nullcontext
        started=time.monotonic();self.budget(started);initial=self.ids(row);generated=[];fragments=[];text='';finish='length';stop='token_cap'
        if row['mode']!='native' and self.instrument is None:self.install()
        self.embedding_cache.reset()
        for step in range(self.cfg['maximum_generated_tokens']):
            self.budget(started);ids=initial+generated
            context=nullcontext() if row['mode']=='native' else self.instrument.context(row['mode'],len(ids))
            with self.torch.inference_mode(),context:value=self.capture(ids)
            arrays={};index=[]
            for i,((_,phase,layer,site),tensor) in enumerate(value['states'].items()):
                key='v'+str(i);arrays[key]=tensor.numpy();index.append(dict(loop=phase,layer_zero_based=layer,site=site,array_key=key))
            arrays['boundary']=np.asarray(value['boundaries'])
            if row['mode']!='native':
                arrays['actual_mask'],arrays['native_output_at_reached_input']=self.instrument.snapshots()
            else:
                arrays['native_output_at_reached_input']=np.asarray([value['states'][('context',p,21,'down_proj_output')].numpy() for p in (1,2)])
            file=self.out/f'request_{ordinal:04d}_prefix_{step:02d}.npz'
            with file.open('xb') as stream:np.savez_compressed(stream,**arrays)
            self.bytes+=file.stat().st_size
            token=value['scores'][-1]['free_top_token_id'];generated.append(token)
            eos=token==self.tok.tok.eos_token_id
            text=self.tok.tok.decode(generated[:-1] if eos else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            fragment=dict(step=step,source_id=row['id'],prefix_ids_sha256=ids_hash(ids),generated_history_sha256=ids_hash(generated[:-1]),
                generated_token_id=token,eos=eos,text_after=text,state_file=file.name,state_file_sha256=sha(file),state_index=index,
                scores=value['scores'],elapsed_seconds=time.monotonic()-started)
            closed=self.out/f'request_{ordinal:04d}_prefix_{step:02d}.json';write(closed,fragment)
            fragments.append(dict(file=closed.name,sha256=sha(closed)))
            append(self.out/'events.jsonl',dict(event='prefix_closed',ordinal=ordinal,step=step,mode=row['mode']))
            del value;gc.collect()
            if eos:finish='stop';stop='eos';break
            if parse(text,row['schema']) is not None:finish='stop';stop='first_complete_packet';break
            if len(text)>self.cfg['maximum_reply_characters']:stop='character_cap';break
        result=dict(text=text,finish_reason=finish,stop_reason=stop,initial_prompt_ids=initial,generated_ids=generated,
            fragments=fragments,elapsed_seconds=time.monotonic()-started,embedding_rows_gathered=self.embedding_cache.rows_gathered,
            usage=dict(prompt_tokens=len(initial),completion_tokens=len(generated),total_tokens=len(initial)+len(generated)))
        self.embedding_cache.reset()
        return result

    def close(self):
        from experiments.nan_recomp09.common import target_digest
        self.observer.close()
        if self.instrument:self.instrument.close()
        self.embedding_cache.close();after=target_digest(self.loop)
        return dict(target_parameter_sha256_before=self.before,target_parameter_sha256_after=after,target_unchanged=after==self.before,
            peak_cuda_allocated_bytes=self.torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=self.torch.cuda.max_memory_reserved(),
            head_device=str(self.head.weight.device),embedding_weight_device=str(self.loop.base.embed_tokens.weight.device),artifact_bytes=self.bytes)
