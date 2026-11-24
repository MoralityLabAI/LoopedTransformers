"""Full fixed cached schedule; model lane never opens current host gold labels."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from .common import ROOT,RUNS,THREADS,read,write,append,sha,verify,verify_worker,load_sources,parse
from .plan import build,digest


def quant_state_digest(loop):
    import torch
    value=hashlib.sha256();count=0
    for name,module in loop.model.named_modules():
        weight=getattr(module,'weight',None);state=getattr(weight,'quant_state',None)
        if state is None:continue
        count+=1;value.update(name.encode())
        for key,item in sorted(state.as_dict(packed=False).items()):
            value.update(key.encode())
            if isinstance(item,torch.Tensor):
                value.update(str((str(item.dtype),list(item.shape))).encode());raw=item.detach().reshape(-1).view(torch.uint8)
                for start in range(0,len(raw),1024**2):value.update(memoryview(raw[start:start+1024**2].cpu().numpy()))
            else:value.update(str(item).encode())
    return dict(modules=count,sha256=value.hexdigest(),scope='Actual new-condition quantization state before/after only; not independent identity with older quantization metadata')


class Engine:
    def __init__(self,a,cfg,out,started):
        import torch
        from .loader import load
        from mechinterp.nanbeige_bpd_gym.streamed_vocab_head import StreamedVocabularyHead
        self.t=torch;self.a=a;self.cfg=cfg;self.out=out;self.started=started;self.cache=self.gym=None
        self.captures=0;self.scored_capture_start=None;self.preflight_bytes=0;self.last_gpu_check=0;self.maximum_allocated=self.maximum_reserved=0
        torch.set_num_threads(1);torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction'])
        props=torch.cuda.get_device_properties(0);assert '3050' in props.name and 3.5*2**30<=props.total_memory<=4.5*2**30
        torch.cuda.reset_peak_memory_stats();self.loop,self.tok,self.entry,self.before=load(a,cfg)
        self.head=StreamedVocabularyHead(self.loop.model.lm_head.weight,self.loop.device,cfg['vocabulary_chunk_rows'])
        self.bindings=read('experiments/nan_full26/tokenizer_receipt.json')['contexts']
        self.quant_before=quant_state_digest(self.loop);self.gpu_check()
    def observe_peaks(self):
        self.maximum_allocated=max(self.maximum_allocated,self.t.cuda.max_memory_allocated())
        self.maximum_reserved=max(self.maximum_reserved,self.t.cuda.max_memory_reserved())
    def gpu_check(self):
        from .resources import check_gpu
        from experiments.nan_ram28.resources import soft_guard
        soft_guard(self.cfg);self.observe_peaks();value=check_gpu(self.cfg,self.a.backend);self.last_gpu_check=time.monotonic();return value
    def ids(self,row):
        from experiments.nan_gym05.common import prompt_ids
        from mechinterp.nanbeige_bpd_gym.chunked_gym import digest_ids
        ids=prompt_ids(self.tok,self.a.snapshot,row['prompt']);pin=self.bindings[row['id']]
        assert len(ids)==pin['tokens'] and digest_ids(ids)==pin['ids_sha256'],'Frozen tokenizer/prompt mismatch'
        return ids
    def admitted(self,count,physical):
        return any(x['batch']==count and x['tokens']>=physical for x in self.cfg['required_capacity'])
    def budget(self,start,artifact_bytes):
        from experiments.nan_ram28.resources import soft_guard
        soft_guard(self.cfg);self.observe_peaks();now=time.monotonic()
        if now-self.started>self.cfg['maximum_backend_seconds']:raise RuntimeError('backend_time_budget')
        if now-start>self.cfg['maximum_pair_seconds']:raise RuntimeError('pair_time_budget')
        if artifact_bytes+self.preflight_bytes>self.cfg['maximum_artifact_bytes_per_backend']:raise RuntimeError('artifact_budget')
        if shutil.disk_usage(self.out).free<self.cfg['minimum_free_disk_bytes']:raise RuntimeError('disk_free_floor')
        if now-self.last_gpu_check>=self.cfg['GPU_allocation_sampling_interval_seconds']:self.gpu_check()
    def reset_runtime(self,instrumented):
        from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache
        from mechinterp.nanbeige_bpd_gym.chunked_instrument import ChunkedRowInstrument
        from mechinterp.nanbeige_bpd_gym.chunked_gym import CachedGym
        if self.cache is not None:self.cache.close()
        factory=(lambda loop:ChunkedRowInstrument(loop,self.a.backend,self.a.bpd_root,self.cfg)) if instrumented else None
        self.cache=ChunkedTwoLoopCache(self.loop,prefill_chunk=self.cfg['prefill_token_chunk_rows'],inplace=True,instrument_factory=factory)
        self.gym=CachedGym(self.cache,self.head,self.out,ids=self.ids,
            decode=lambda ids:self.tok.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
            parse=parse,eos_token_id=self.tok.tok.eos_token_id,budget=self.budget,admitted=self.admitted)
    def prefill(self,sequences,modes):
        self.budget(time.monotonic(),0);value=self.gym.capture(sequences,modes,list(range(len(sequences))),None,[[] for _ in sequences])
        self.captures+=1;self.observe_peaks();return value
    def increment(self,sequences,tokens):
        self.budget(time.monotonic(),0);indices=list(range(len(tokens)))
        value=self.gym.capture(sequences,list(self.cache.modes),indices,indices,[[token] for token in tokens])
        self.captures+=1;self.observe_peaks();return value


def execute_schedule(engine,public,plan,completed,event):
    for job in plan['jobs']:
        rows=[public[m['source_index']] for m in job['members']];modes=[m['mode'] for m in job['members']]
        engine.budget(time.monotonic(),engine.gym.bytes)
        entries=engine.gym.generate(rows,modes,job['ordinal'],job['lane']);completed.append(job['execution_index'])
        event(dict(event='fixed_job_closed',execution_index=job['execution_index'],lane=job['lane'],ordinal=job['ordinal'],closed_rows=len(entries)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('nf4','bf16'))
    for name in ('snapshot','compactification-root','bpd-root','artifact-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();seal=verify('model');cfg=read(ROOT/'config.json');started=time.monotonic()
    for name in THREADS:os.environ[name]='1'
    os.environ.update(HF_HUB_OFFLINE='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false')
    os.environ.pop('TRANSFORMERS_CACHE',None)
    run=RUNS/a.backend;run.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HOME']=str((run/'HF_cache').resolve());os.environ['HF_MODULES_CACHE']=str((run/'HF_modules').resolve())
    from experiments.nan_ranksite13.resources import below_normal
    from experiments.nan_storyaudit30.common import Handle
    below_normal();own=Handle(os.getpid());creation=str(own.creation);own.close()
    write(run/'runtime.json',dict(pid=os.getpid(),creation=creation,backend=a.backend,condition='NAN-CACHE-32',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json')))
    verify_worker(a.backend);(run/'runtime_ready.signal').touch(exist_ok=False)
    assert Path(a.compactification_root).resolve()==Path('../Compactification-Theory-arg02').resolve()
    assert Path(a.bpd_root).resolve()==Path('../BPD').resolve()
    for folder,expected in ((a.compactification_root,seal['adapter_commit']),(a.bpd_root,seal['BPD_commit'])):
        assert subprocess.check_output(['git','-C',folder,'rev-parse','HEAD'],text=True).strip()==expected
    public,pins=load_sources();assert pins==seal['public_source_pins']
    plan=build(public,read('experiments/nan_full26/tokenizer_receipt.json')['contexts']);assert digest(plan)==seal['plan_sha256']
    out=Path(a.artifact_root)/a.backend;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',dict(experiment='NAN-CACHE-32',backend=a.backend,pid=os.getpid(),creation=creation,
        source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),plan_sha256=digest(plan),baseline_registered=1549,AB_pairs_registered=216,
        raw_replies_reused=0,CPU_threads=1,priority='BelowNormal',sealed_test_accessed=False))
    dispatch=None;engine=None;status='failed';failure=None;completed=[]
    try:
        from importlib.metadata import version
        for name,expected in cfg['GPU_packages'].items():assert version(name)==expected,name
        from mechinterp.nanbeige_bpd_gym.lazy_bf16_loop import digest_dispatch
        dispatch=digest_dispatch();dispatch.__enter__();engine=Engine(a,cfg,out,started)
        from .preflight import run as preflight
        admission=preflight(engine,public[0]);assert admission['status']=='passed'
        engine.scored_capture_start=engine.gym.captures
        append(out/'events.jsonl',dict(event='all_preflight_gates_passed'))
        engine.preflight_bytes+=(out/'preflight.json').stat().st_size+(out/'manifest.json').stat().st_size
        execute_schedule(engine,public,plan,completed,lambda event:append(out/'events.jsonl',event))
        status='completed'
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));traceback.print_exc()
        if type(error).__name__=='OutOfMemoryError':status='capacity_stopped'
        elif type(error).__name__ in ('HostMemoryBudgetReached','ResourceAdmissionError'):status='resource_budget_stopped'
    finally:
        terminal=dict(status=status,failure=failure,backend=a.backend,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
            plan_sha256=digest(plan),completed_execution_indices=completed,total_registered_jobs=len(plan['jobs']),
            elapsed_seconds_including_load=time.monotonic()-started,neural_retries=0,sealed_test_accessed=False)
        if engine is not None:
            engine.observe_peaks()
            if engine.cache is not None:engine.cache.close()
            engine.head.close();gc.collect();engine.t.cuda.empty_cache()
            from experiments.nan_recomp09.common import target_digest
            from experiments.nan_bf31.loader import verify_checkpoint
            after=target_digest(engine.loop);quant_after=quant_state_digest(engine.loop)
            verify_checkpoint(a.snapshot,engine.entry)
            terminal.update(target_before=engine.before,target_after=after,target_unchanged=engine.before==after,
                quant_state_before=engine.quant_before,quant_state_after=quant_after,quant_state_unchanged=engine.quant_before==quant_after,
                checkpoint_files_verified_after=True,normal_terminal_integrity_verified=engine.before==after and engine.quant_before==quant_after,
                decoder_captures=engine.captures+(engine.gym.captures-engine.scored_capture_start if engine.scored_capture_start is not None else 0),
                artifact_bytes=engine.preflight_bytes+(engine.gym.bytes if engine.gym is not None else 0),
                CUDA_maximum_observed_peak_allocated_bytes=engine.maximum_allocated,CUDA_maximum_observed_peak_reserved_bytes=engine.maximum_reserved)
            if not terminal['normal_terminal_integrity_verified']:terminal['status']='target_integrity_failed'
        if dispatch is not None:dispatch.__exit__(None,None,None)
        write(out/'receipt.json',terminal);append(out/'events.jsonl',dict(event='worker_terminal',status=terminal['status']))
    if terminal['status']!='completed':raise SystemExit(2)


if __name__=='__main__':main()
