"""Unregistered full-gym cached trajectory writer, not a model-loading CLI.

Uses the original bounded packet parser and token caps via caller bindings.
The future frozen worker must supply capacity/resource/time/disk admission,
tokenizer pins, complete rosters, model identity and post-GPU host grading.
All 40 BF16 site vectors and the full active batch/cache context are exported;
only complete, immutable episodes enter the closed journal. No neural retry.
"""
import hashlib
import json
from pathlib import Path
import time


def digest_ids(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def mask_runs(row):
    """Lossless canonical half-open intervals of actual ones, including holes."""
    result=[];start=None
    for position,value in enumerate([*row,0]):
        if type(value) is not int or value not in (0,1):raise ValueError('Binary cache attention required')
        if value and start is None:start=position
        elif not value and start is not None:result.append([start,position]);start=None
    return result


def batch_signature(record):
    """Validate complete cached-batch metadata before proposing a geometry pair.

    This key is necessary, not sufficient: independent immutable token/state,
    parameter, native-write and terminal audits must still pass. It certifies
    no actual cache bytes, argmax, causal effect or numerical equivalence.
    """
    import re
    if record['schema']!='cached-prefix-v2':raise ValueError('Separate cached arithmetic condition required')
    context=record['batch_context'];B=record['batch_size'];row=record['batch_row'];step=record['step']
    if type(B) is not int or not 1<=B<=2 or type(row) is not int or not 0<=row<B:raise ValueError('Invalid batch row')
    if type(step) is not int or not 0<=step<256:raise ValueError('Invalid bounded generation step')
    fields=('source_ids','modes','prefix_ids_sha256','history_sha256','semantic_lengths','initial_semantic_lengths','attention_mask_runs','original_row_indices')
    if any(len(context[k])!=B for k in fields):raise ValueError('Missing companion context')
    P=context['physical_cache_length'];initial=context['original_prefill_canvas']
    if type(P) is not int or type(initial) is not int or not 0<initial<=P or P!=initial+step:raise ValueError('Physical cache position changed')
    indices=context['original_row_indices']
    if indices!=sorted(set(indices)) or any(type(i) is not int or not 0<=i<2 for i in indices):raise ValueError('Original cache row order changed')
    if context['attention_mask_encoding']!='half-open-runs-v1':raise ValueError('Unknown lossless mask encoding')
    if not re.fullmatch('[0-9a-f]{64}',context['parent_cache_context_sha256']):raise ValueError('Missing cache lineage')
    if step==0 and context['parent_cache_context_sha256']!='0'*64:raise ValueError('Prefill must start a fresh cache lineage')
    masks=[]
    for i in range(B):
        n=context['initial_semantic_lengths'][i]
        if type(n) is not int or not 0<n<=initial or context['semantic_lengths'][i]!=n+step:raise ValueError('Semantic cache position changed')
        mask=[1]*n+[0]*(initial-n)+[1]*step;masks.append(mask)
        if context['attention_mask_runs'][i]!=mask_runs(mask):raise ValueError('Cached padding hole changed')
        if context['modes'][i] not in ('native','all_one','component_only','static_CAL_mean','reached_state'):raise ValueError('Unknown route')
        if not isinstance(context['source_ids'][i],str) or not context['source_ids'][i]:raise ValueError('Missing source identity')
        for name in ('prefix_ids_sha256','history_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',context[name][i]):raise ValueError('Invalid complete history hash')
    if context['attention_mask_sha256']!=digest_ids(masks):raise ValueError('Cache mask checksum changed')
    for field,actual in (('source_id',context['source_ids'][row]),('mode',context['modes'][row]),
                         ('prefix_ids_sha256',context['prefix_ids_sha256'][row]),('history_sha256',context['history_sha256'][row]),
                         ('actual_length',context['semantic_lengths'][row]),('padded_length',P)):
        if record[field]!=actual:raise ValueError('Row disagrees with complete batch context')
    if record['padding_tokens']!=sum(P-n for n in context['semantic_lengths']):raise ValueError('Padding count changed')
    return hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class CachedGym:
    def __init__(self,cache,head,out,*,ids,decode,parse,eos_token_id,budget,admitted):
        self.cache=cache;self.head=head;self.out=Path(out);self.ids=ids;self.decode=decode;self.parse=parse
        self.eos=eos_token_id;self.budget=budget;self.admitted=admitted;self.bytes=0;self.captures=0
        self.out.mkdir(parents=True,exist_ok=True)
        if tuple(cache.observer.layers)!=(5,11,17,21):raise ValueError('Full-gym last-token site coverage required')

    def capture(self,sequences,modes,active,previous,generated):
        import torch
        cuda=self.cache.loop.device.type=='cuda';begin=end=None
        if cuda:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        start=time.monotonic()
        with torch.inference_mode():
            if previous is None:value=self.cache.prefill(sequences,modes=modes)
            else:
                if active!=previous:self.cache.select_rows([previous.index(i) for i in active])
                if tuple(modes)!=self.cache.modes:raise RuntimeError('Cached route changed within a trajectory')
                value=self.cache.step([generated[i][-1] for i in active])
            value['logits']=[self.head(h) for h in value['boundaries']]
            if any(not torch.isfinite(x).all() for x in value['logits']):raise RuntimeError('Nonfinite full-vocabulary logits')
            expected=list(map(len,sequences))
            if expected!=self.cache.semantic_lengths:raise RuntimeError('Semantic cache history diverged from actual prefix')
            if any(x.shape[0]!=len(active) for x in value['logits']):raise RuntimeError('Full-vocabulary batch changed')
            observed=value['states']
            if len(observed)!=40*len(active):raise RuntimeError('Missing full-gym physical state')
            flat=torch.cat([x.flatten() for x in observed.values()]).cpu();offset=0;states={}
            for key,x in observed.items():
                if x.dtype!=torch.bfloat16:raise RuntimeError('Lossless BF16 site export required')
                states[key]=flat[offset:offset+x.numel()].reshape(x.shape);offset+=x.numel()
            value['states']=states;self.cache.observer.values={}
            value['boundaries']=[x.cpu() for x in value['boundaries']]
            mask=self.cache.attention.cpu().tolist()
            if any(sum(row)!=n for row,n in zip(mask,expected)):raise RuntimeError('Cache attention does not match semantic history')
            value['attention_mask']=mask;value['physical_cache_length']=self.cache.physical_length
            value['cache_bytes']=self.cache.cache_bytes()
            if end is not None:end.record();end.synchronize()
        value['cuda_elapsed_ms']=begin.elapsed_time(end) if cuda else None
        value['wall_seconds']=time.monotonic()-start;self.captures+=1
        return value

    def generate(self,rows,modes,ordinal,lane):
        """One genuine B1/B2 trajectory batch, retaining padding after pruning.

        No answer/label/oracle is read here. Greedy tokens use the complete
        vocabulary; bounded parser closure and EOS remain independent stop
        reasons. A failed batch's fragments are preserved without closed-row
        promotion, replay or extraction into an official score.
        """
        import numpy as np
        import torch
        from experiments.nan_ram28.common import append,write,sha
        if lane not in ('baseline','AB') or type(ordinal) is not int or ordinal<0:raise ValueError('Invalid immutable slot')
        if not 1<=len(rows)<=2 or len(rows)!=len(modes):raise ValueError('One or two explicit route-bound rows required')
        if any(mode!='native' for mode in modes) and self.cache.instrument is None:raise ValueError('Actual BPD instrument required')
        initial=[list(self.ids(row)) for row in rows];generated=[[] for _ in rows];text=['' for _ in rows]
        caps=[64 if r['stage']=='nan_story23' else 256 if r['schema']['type']=='grid' else 128 for r in rows]
        stopped=[None for _ in rows];fragments=[[] for _ in rows];previous=None
        start=time.monotonic();self.cache.reset();closed=[];parent_context='0'*64
        try:
            for step in range(max(caps)):
                active=[i for i in range(len(rows)) if stopped[i] is None]
                if not active:break
                sequences=[initial[i]+generated[i] for i in active]
                physical=max(map(len,sequences)) if previous is None else self.cache.physical_length+1
                self.budget(start,self.bytes)
                if not self.admitted(len(active),physical):raise RuntimeError('Unadmitted physical cache canvas')
                value=self.capture(sequences,[modes[i] for i in active],active,previous,generated);previous=list(active)
                batch=dict(source_ids=[rows[i]['id'] for i in active],modes=[modes[i] for i in active],
                    prefix_ids_sha256=[digest_ids(x) for x in sequences],history_sha256=[digest_ids(generated[i]) for i in active],
                    semantic_lengths=list(map(len,sequences)),physical_cache_length=value['physical_cache_length'],
                    initial_semantic_lengths=[len(initial[i]) for i in active],
                    attention_mask_encoding='half-open-runs-v1',attention_mask_runs=[mask_runs(x) for x in value['attention_mask']],
                    attention_mask_sha256=digest_ids(value['attention_mask']),parent_cache_context_sha256=parent_context,
                    original_row_indices=list(active),original_prefill_canvas=max(map(len,initial)))
                for b,i in enumerate(active):
                    arrays={};index=[]
                    for (row,phase,layer,site),vector in value['states'].items():
                        if row!=b:continue
                        key='v'+str(len(index));arrays[key]=vector.view(torch.uint16).numpy()
                        index.append(dict(loop=phase,layer=layer,site=site,key=key))
                    arrays['boundaries']=torch.stack([x[b] for x in value['boundaries']]).view(torch.uint16).numpy()
                    instrument=self.cache.instrument
                    if instrument is not None:
                        arrays['native_at_reached_input']=torch.stack([instrument.native[p,b] for p in (1,2)]).view(torch.uint16).numpy()
                        arrays['instrument_input_last_token']=torch.stack([instrument.inputs[p,b] for p in (1,2)]).view(torch.uint16).numpy()
                        if modes[i]!='native':arrays['actual_masks']=np.stack([instrument.masks[p,b].numpy() for p in (1,2)])
                    file=self.out/f'{lane}_{ordinal:04d}_r{i}_p{step:03d}.npz'
                    with file.open('xb') as stream:np.savez(stream,**arrays)
                    self.bytes+=file.stat().st_size
                    score=value['logits'][-1][b];token=int(score.argmax());eos=token==self.eos
                    history=digest_ids(generated[i]);generated[i].append(token)
                    text[i]=self.decode(generated[i][:-1] if eos else generated[i])
                    record=dict(schema='cached-prefix-v2',step=step,source_id=rows[i]['id'],mode=modes[i],
                        prefix_ids_sha256=digest_ids(sequences[b]),history_sha256=history,token=token,
                        state_file=file.name,state_sha256=sha(file),state_index=index,batch_context=batch,
                        batch_size=len(active),batch_row=b,actual_length=len(sequences[b]),
                        padded_length=value['physical_cache_length'],padding_tokens=sum(len(x)-sum(x) for x in value['attention_mask']),
                        full_vocabulary_size=len(score),top_logit=float(score[token]),
                        log_normalizer=float(torch.logsumexp(score.double(),0)),cache_bytes_for_entire_batch=value['cache_bytes'],
                        cuda_elapsed_ms_for_entire_batch=value['cuda_elapsed_ms'],wall_seconds_for_entire_batch=value['wall_seconds'],
                        full_logits_independently_reconstructible_from_export=False)
                    record['complete_batch_context_sha256']=batch_signature(record)
                    path=file.with_suffix('.json');write(path,record);self.bytes+=path.stat().st_size
                    fragments[i].append(dict(file=path.name,sha256=sha(path)))
                    if eos:stopped[i]='eos'
                    elif self.parse(text[i],rows[i]) is not None:stopped[i]='complete_packet'
                    elif len(text[i])>(512 if rows[i]['stage']=='nan_story23' else 4096):stopped[i]='character_cap'
                    elif len(generated[i])>=caps[i]:stopped[i]='token_cap'
                parent_context=record['complete_batch_context_sha256']
                append(self.out/'events.jsonl',dict(event='cached_prefix_batch_closed',lane=lane,ordinal=ordinal,step=step,
                    batch_size=len(active),physical_cache_length=value['physical_cache_length'],
                    semantic_lengths=batch['semantic_lengths'],cache_bytes=value['cache_bytes'],
                    cuda_elapsed_ms=value['cuda_elapsed_ms'],wall_seconds=value['wall_seconds']))
                del value
            for i,row in enumerate(rows):
                path=self.out/f'{lane}_{ordinal:04d}_r{i}_closed.json'
                write(path,dict(schema='cached-episode-v2',source_id=row['id'],root=row['root'],stage=row['stage'],split=row['split'],
                    mode=modes[i],lane=lane,initial_ids=initial[i],generated_ids=generated[i],text=text[i],stop_reason=stopped[i],
                    fragments=fragments[i],prefix_count=len(fragments[i]),wall_seconds=time.monotonic()-start))
                self.bytes+=path.stat().st_size;entry=dict(file=path.name,sha256=sha(path));append(self.out/'closed.jsonl',entry);closed.append(entry)
            return closed
        finally:self.cache.reset()
