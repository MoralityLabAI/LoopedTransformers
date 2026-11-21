"""Public/source-only generation; oracle and host grades are never loaded here."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback
from .common import ROOT,THREADS,append,read,write,sha,verify,load_sources,intervention_rows,parse

def digest_ids(ids):
    import json
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

class Engine:
    def __init__(self,a,cfg,out):
        import torch
        from experiments.nan_bpd01.backend import load_backend
        from experiments.nan_comp02.head import OperatorHead
        from experiments.nan_comp03.prompt import aliases
        from experiments.nan_recomp09.common import target_digest
        from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers
        from .batch import install,BatchObserver,Embeddings
        self.t=torch;self.cfg=cfg;self.out=out;self.a=a;self.instrument=None;self.bytes=0;self.forward_count=0
        self.bindings=read(ROOT/'tokenizer_receipt.json')
        for name,h in self.bindings['tokenizer_files_sha256'].items():assert sha(Path(a.snapshot)/name)==h
        torch.set_num_threads(1);torch.manual_seed(cfg['source_seed'])
        torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
        self.loop,old,self.tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=8192)
        if self.loop.stage_layer is not None:self.loop.stage_layer.requires_grad_(False).eval()
        if a.backend=='bf16':retain_decoder_layers(self.loop,4)
        cpu=OperatorHead(old.weight,aliases(self.tok),chunk_rows=8192);self.head=mount(self.loop,cpu)
        del old,cpu;gc.collect()
        self.before=target_digest(self.loop);self.old=install(self.loop)
        self.observer=BatchObserver(self.loop,cfg['layers_zero_based']);self.embedding=Embeddings(self.loop)
        self.started=time.monotonic()

    def ids(self,row):
        from experiments.nan_gym05.common import prompt_ids
        ids=prompt_ids(self.tok,self.a.snapshot,row['prompt'])
        bound=self.bindings['contexts'][row['id']]
        assert len(ids)==bound['tokens'] and digest_ids(ids)==bound['ids_sha256']
        return ids

    def capture(self,sequences,modes,observe=True):
        from contextlib import nullcontext
        from .batch import logits
        t=self.t;lengths=list(map(len,sequences));h,ctx=self.embedding.prepare(sequences)
        self.observer.enabled=observe;self.observer.lengths=lengths;self.observer.values={}
        context=self.instrument.context(modes,lengths) if self.instrument else nullcontext()
        begin=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True);start=time.monotonic()
        boundaries=[];scores=[]
        with t.inference_mode(),context:
            begin.record()
            for phase in (1,2):
                self.observer.phase=phase;h=self.loop.run_pairs([(h,ctx)])[0]
                last=t.stack([h[b,n-1] for b,n in enumerate(lengths)])
                boundaries.append(last.detach().clone());scores.append(logits(self.head,last))
            end.record();end.synchronize()
        self.forward_count+=1;self.observer.enabled=False
        # One contiguous state transfer, rather than 80 per-site CPU syncs.
        observed=self.observer.values;values={}
        if observed:
            flat=t.cat([x.flatten() for x in observed.values()]).cpu();offset=0
            for key,x in observed.items():
                values[key]=flat[offset:offset+x.numel()].reshape(x.shape);offset+=x.numel()
            self.observer.values={}
        boundaries=[x.cpu() for x in boundaries]
        if observe:assert len(values)==40*len(sequences)
        result=dict(states=values,boundaries=boundaries,logits=scores,
            cuda_elapsed_ms=begin.elapsed_time(end),wall_seconds=time.monotonic()-start,
            lengths=lengths,batch_size=len(sequences),padding_tokens=max(lengths)*len(lengths)-sum(lengths))
        if self.instrument:
            result['native']=list(self.instrument.raw_by_phase);result['masks']=list(self.instrument.mask_by_phase)
        return result

    def install_instrument(self):
        from .batch import RowInstrument,BatchObserver
        self.observer.close();self.instrument=RowInstrument(self.loop,self.a.backend,self.a.bpd_root,self.cfg)
        # Observe AFTER intervention hook, including the actual edited output.
        self.observer=BatchObserver(self.loop,self.cfg['layers_zero_based'])

    def preflight(self,row):
        t=self.t;ids=self.ids(row);self.embedding.reset()
        # B1 old/new closure under identical contexts, every captured vector and
        # every full-vocabulary logit. New B2 arithmetic is separately labelled.
        new=self.capture([ids],['native'])
        from mechinterp.nanbeige_bpd_gym.resident_head import full_logits
        for boundary,score in zip(new['boundaries'],new['logits']):
            reference=full_logits(self.head.weight,boundary[:,None,:],self.loop.device,8192)
            assert t.equal(reference,score[0].cpu()),'B1 head chunk shape parity'
        self.loop.run_pairs=self.old;self.embedding.reset()
        old=self.capture([ids],['native'])
        assert all(t.equal(new['states'][k],old['states'][k]) for k in new['states'])
        assert all(t.equal(a,b) for a,b in zip(new['logits'],old['logits']))
        from .batch import install
        install(self.loop);self.embedding.reset();pair=self.capture([ids,ids],['native','native'])
        row_exact=all(t.equal(pair['states'][(0,*k[1:])],pair['states'][(1,*k[1:])]) for k in pair['states'])
        assert row_exact and all(t.equal(x[0],x[1]) for x in pair['logits'])
        differences=[float((x[0]-y[0]).abs().max()) for x,y in zip(pair['logits'],new['logits'])]
        self.install_instrument();self.embedding.reset();closed=self.capture([ids,ids],['native','all_one'])
        assert all(t.equal(pair['states'][k],closed['states'][k]) for k in pair['states'])
        assert all(t.equal(x,y) for x,y in zip(pair['logits'],closed['logits']))
        del old,new,pair,closed;gc.collect();self.embedding.reset()
        capacity=[]
        for batch,length in self.cfg['capacity_shapes']:
            repeated=(ids*((length+len(ids)-1)//len(ids)))[:length]
            try:
                v=self.capture([repeated]*batch,['native']*batch,observe=False)
                capacity.append(dict(batch=batch,length=length,status='passed',cuda_elapsed_ms=v['cuda_elapsed_ms'],
                    peak_allocated_bytes=t.cuda.max_memory_allocated(),peak_reserved_bytes=t.cuda.max_memory_reserved()))
                del v
            except t.cuda.OutOfMemoryError:
                capacity.append(dict(batch=batch,length=length,status='capacity_failed'))
            finally:
                self.embedding.reset();gc.collect();t.cuda.empty_cache()
        self.capacity=capacity
        # Every scored input must fit a previously passed shape. OOM on a
        # scored input stops the backend, with no replay, truncation or retry.
        write(self.out/'preflight.json',dict(status='passed',B1_native_closure=True,B2_clean_rows_exact=True,
            B2_all_one_closure=True,B1_B2_max_logit_abs_difference=differences,capacity=capacity,
            selection_uses_only_capacity=True,pretrained_preflight_forwards=self.forward_count))

    def admitted(self,count,length):
        return any(p['status']=='passed' and p['batch']>=count and p['length']>=length for p in self.capacity)

    def budget(self,start):
        if time.monotonic()-self.started>self.cfg['maximum_backend_seconds']:raise RuntimeError('backend_time_budget')
        if time.monotonic()-start>self.cfg['maximum_pair_seconds']:raise RuntimeError('pair_time_budget')
        if self.bytes>self.cfg['maximum_artifact_bytes_per_backend']:raise RuntimeError('artifact_budget')
        if shutil.disk_usage(self.out).free<self.cfg['minimum_free_disk_bytes']:raise RuntimeError('disk_free_floor')

    def generate(self,rows,modes,ordinal,lane):
        import numpy as np
        t=self.t;initial=[self.ids(r) for r in rows];generated=[[] for r in rows];text=['' for r in rows]
        stopped=[None for r in rows];fragments=[[] for r in rows];totals=[dict(cuda_elapsed_ms=0.,forward_wall_seconds=0.) for r in rows]
        caps=[64 if r['stage']=='nan_story23' else 256 if r['schema']['type']=='grid' else 128 for r in rows]
        start=time.monotonic();self.embedding.reset();last_active=None
        for step in range(max(caps)):
            self.budget(start);active=[i for i in range(len(rows)) if stopped[i] is None]
            if not active:break
            if active!=last_active:self.embedding.reset();last_active=active
            sequences=[initial[i]+generated[i] for i in active]
            assert self.admitted(len(active),max(map(len,sequences))),'Unadmitted input capacity; preserve and stop'
            value=self.capture(sequences,[modes[i] for i in active])
            for b,i in enumerate(active):
                arrays={};index=[]
                for (batch,phase,layer,site),vector in value['states'].items():
                    if batch!=b:continue
                    key='v'+str(len(index));assert vector.dtype==t.bfloat16
                    arrays[key]=vector.view(t.uint16).numpy();index.append(dict(loop=phase,layer=layer,site=site,key=key))
                arrays['boundaries']=t.stack([x[b] for x in value['boundaries']]).view(t.uint16).numpy()
                if 'native' in value:
                    arrays['native_at_reached_input']=t.stack([x[b] for x in value['native']]).view(t.uint16).numpy()
                if modes[i]!='native':
                    arrays['actual_masks']=np.asarray([x.float().numpy() if hasattr(x,'float') else x for x in value['masks']])
                file=self.out/f'{lane}_{ordinal:04d}_r{i}_p{step:03d}.npz'
                with file.open('xb') as f:np.savez(f,**arrays)
                self.bytes+=file.stat().st_size
                token=int(value['logits'][-1][b].argmax());eos=token==self.tok.tok.eos_token_id
                history=digest_ids(generated[i]);generated[i].append(token)
                text[i]=self.tok.tok.decode(generated[i][:-1] if eos else generated[i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                record=dict(step=step,source_id=rows[i]['id'],mode=modes[i],prefix_ids_sha256=digest_ids(sequences[b]),
                    history_sha256=history,token=token,state_file=file.name,state_sha256=sha(file),state_index=index,
                    batch_size=len(active),batch_row=b,padded_length=max(map(len,sequences)),actual_length=len(sequences[b]),
                    cuda_elapsed_ms_for_entire_batch=value['cuda_elapsed_ms'],wall_seconds_for_entire_batch=value['wall_seconds'],
                    padding_tokens=value['padding_tokens'],full_vocabulary_size=value['logits'][-1].shape[-1],
                    top_logit=float(value['logits'][-1][b,token]),log_normalizer=float(t.logsumexp(value['logits'][-1][b].double(),0)))
                path=self.out/f'{lane}_{ordinal:04d}_r{i}_p{step:03d}.json';write(path,record);self.bytes+=path.stat().st_size
                fragments[i].append(dict(file=path.name,sha256=sha(path)))
                totals[i]['cuda_elapsed_ms']+=value['cuda_elapsed_ms']/len(active)
                totals[i]['forward_wall_seconds']+=value['wall_seconds']/len(active)
                if eos:stopped[i]='eos'
                elif parse(text[i],rows[i]) is not None:stopped[i]='complete_packet'
                elif len(text[i])>(512 if rows[i]['stage']=='nan_story23' else 4096):stopped[i]='character_cap'
                elif len(generated[i])>=caps[i]:stopped[i]='token_cap'
            append(self.out/'events.jsonl',dict(event='prefix_batch_closed',lane=lane,ordinal=ordinal,step=step,batch_size=len(active),
                CUDA_allocated_bytes=t.cuda.memory_allocated(),CUDA_peak_allocated_bytes=t.cuda.max_memory_allocated(),
                cuda_elapsed_ms=value['cuda_elapsed_ms'],wall_seconds=value['wall_seconds']))
            del value
        for i,r in enumerate(rows):
            path=self.out/f'{lane}_{ordinal:04d}_r{i}_closed.json'
            write(path,dict(source_id=r['id'],root=r['root'],stage=r['stage'],split=r['split'],mode=modes[i],lane=lane,
                initial_ids=initial[i],generated_ids=generated[i],text=text[i],stop_reason=stopped[i],fragments=fragments[i],
                prefix_count=len(fragments[i]),wall_seconds=time.monotonic()-start,attributed_batch_timing=totals[i]))
            self.bytes+=path.stat().st_size
            append(self.out/'closed.jsonl',dict(file=path.name,sha256=sha(path)))
        self.embedding.reset()

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=['nf4','bf16'])
    for name in ('snapshot','compactification-root','bpd-root','artifact-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    for key in THREADS:os.environ[key]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();verify();cfg=read(ROOT/'config.json');out=Path(a.artifact_root)/a.backend;out.mkdir(parents=True,exist_ok=False)
    public,pins=load_sources();assert pins==read(ROOT/'SOURCE_MANIFEST.json')['source_pins']
    engine=None;status='failed';failure=None;baseline=ab=0;started=time.monotonic()
    write(out/'manifest.json',dict(pid=os.getpid(),backend=a.backend,seal_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        baseline_registered=1549,AB_pairs_registered=216,CPU_threads=1,priority='BelowNormal',oracle_opened=False,sealed_test_accessed=False))
    try:
        engine=Engine(a,cfg,out);engine.preflight(public[0]);append(out/'events.jsonl',dict(event='preflight_closed'))
        index=0
        while index<len(public):
            selected=[public[index]]
            first=engine.ids(selected[0]);cap=256
            if index+1<len(public):
                nxt=engine.ids(public[index+1])
                if engine.admitted(2,max(len(first),len(nxt))+cap):selected.append(public[index+1])
            engine.generate(selected,['native']*len(selected),index,'baseline');index+=len(selected);baseline=index
            append(out/'events.jsonl',dict(event='baseline_batch_closed',completed=baseline,registered=1549))
        for route in cfg['AB_routes']:
            for row in intervention_rows(public):
                ids=engine.ids(row);assert engine.admitted(2,len(ids)+256),'A/B capacity unavailable; no serial replacement'
                engine.generate([row,row],['native',route],ab,'AB');ab+=1
                append(out/'events.jsonl',dict(event='AB_pair_closed',completed=ab,registered=216,route=route))
        status='completed'
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));traceback.print_exc()
        status='capacity_stopped' if type(error).__name__=='OutOfMemoryError' else 'failed'
    finally:
        terminal=dict(status=status,failure=failure,baseline_completed=baseline,baseline_registered=1549,AB_pairs_completed=ab,AB_pairs_registered=216,
            source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),elapsed_seconds_including_load=time.monotonic()-started,
            neural_retries=0,sealed_test_accessed=False)
        if engine:
            from experiments.nan_recomp09.common import target_digest
            engine.observer.close()
            if engine.instrument:engine.instrument.close()
            after=target_digest(engine.loop)
            terminal.update(target_before=engine.before,target_after=after,target_unchanged=engine.before==after,
                CUDA_peak_allocated_bytes=engine.t.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=engine.t.cuda.max_memory_reserved(),
                artifact_bytes=engine.bytes,forwards=engine.forward_count,loaded_model_elapsed_seconds=time.monotonic()-engine.started)
        write(out/'receipt.json',terminal);append(out/'events.jsonl',dict(event='worker_terminal',status=status))
    if status!='completed':raise SystemExit(2)

if __name__=='__main__':main()
