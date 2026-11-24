"""Independent, stdlib-only audit of immutable cached-v2 trajectories.

Does not import the producer, Torch, NumPy, model code or an oracle. Verifies
lossless NPY/BF16 arrays and actual token/stop history. Complete geometry keys
require audited companions AND every parent batch in the cache lineage. Top
logit/normalizer metadata cannot independently prove a full-vocabulary argmax.
Numerical SAE/BPD replay and before/after model/stage/terminal checks remain
separate requirements; this module does not register or launch a worker.
"""
import array
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import zipfile

LAYERS=(5,11,17,21)
SITES=('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')
MODES=('native','all_one','component_only','static_CAL_mean','reached_state')


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def ids_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def context_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked(condition,message):
    if not condition:raise ValueError(message)


def read_json(path,digest):
    checked(Path(path).stat().st_size<=2*1024**2,'Oversized immutable JSON')
    checked(file_hash(path)==digest,'Immutable JSON checksum changed')
    def unique(pairs):
        value={}
        for key,item in pairs:
            checked(key not in value,'Duplicate JSON key');value[key]=item
        return value
    def nonfinite(value):raise ValueError('Nonfinite JSON number')
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=nonfinite)


@dataclass(frozen=True)
class Layout:
    widths:dict
    hidden:int
    vocabulary:int
    cache_bytes_per_row_token:int
    components:int=8


def pinned_layout(snapshot):
    """Read config and safetensors headers only; materialize no weight tensor."""
    from .lazy_bf16_storage import CheckpointStore
    from .ram28_work_budget import capture_layout
    snapshot=Path(snapshot);store=CheckpointStore(snapshot);cfg=json.loads((snapshot/'config.json').read_bytes())
    checked(cfg['num_hidden_layers']==22 and cfg['num_loops']==2,'Unexpected pinned native architecture')
    widths,_=capture_layout(store.spans,LAYERS);H=cfg['hidden_size'];V=cfg['vocab_size']
    checked(store.spans['model.embed_tokens.weight'].shape==store.spans['lm_head.weight'].shape==(V,H),'Vocabulary shape changed')
    head=cfg['head_dim'];kv=cfg['num_key_value_heads'];query=cfg['num_attention_heads']
    for i in range(22):
        checked(store.spans[f'model.layers.{i}.self_attn.k_proj.weight'].shape==(kv*head,H),'KV width changed')
        checked(store.spans[f'model.layers.{i}.self_attn.q_proj.weight'].shape==(query*head,H),'Query width changed')
    checked(all(x['residual_after_layer']==H for x in widths.values()),'Residual width changed')
    return Layout(widths,H,V,2*22*2*kv*head*2)


def read_npy(archive,name,shape,kind):
    """Bounded little-endian numeric NPY reader; object/pickle data is refused."""
    count=math.prod(shape);size=2 if kind=='BF16' else 4;info=archive.getinfo(name+'.npy')
    checked(info.file_size<=count*size+16384,'Oversized NPY payload')
    checked(not info.flag_bits&1 and info.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED),'Unsupported ZIP member')
    with archive.open(info) as stream:
        checked(stream.read(6)==b'\x93NUMPY','Invalid NPY magic');version=stream.read(2)
        checked(version in (b'\x01\x00',b'\x02\x00',b'\x03\x00'),'Unknown NPY version')
        n=2 if version==b'\x01\x00' else 4;length=stream.read(n);checked(len(length)==n,'Short NPY header')
        length=int.from_bytes(length,'little');checked(0<length<=16384,'Unbounded NPY header')
        header=ast.literal_eval(stream.read(length).decode('utf-8' if version==b'\x03\x00' else 'latin1'))
        checked(set(header)=={'descr','fortran_order','shape'} and header['fortran_order'] is False,'Invalid NPY layout')
        checked(header['shape']==shape and all(type(x) is int for x in header['shape']),'NPY shape changed')
        checked(header['descr']==('<u2' if kind=='BF16' else '<f4'),'NPY dtype changed')
        data=stream.read(count*size+1);checked(len(data)==count*size and stream.read(1)==b'','NPY extent changed')
    values=array.array('H' if kind=='BF16' else 'f');values.frombytes(data)
    if sys.byteorder!='little':values.byteswap()
    if kind=='BF16':checked(all((value&0x7f80)!=0x7f80 for value in values),'Nonfinite BF16 state')
    else:checked(all(math.isfinite(value) and 0<=value<=1 for value in values),'Invalid BPD mask')
    return data


def audit_states(path,prefix,layout,mode,require_instrumented):
    index=prefix['state_index'];expected={(p,l,s) for p in (1,2) for l in LAYERS for s in SITES}
    checked(all(type(x['loop']) is int and type(x['layer']) is int for x in index),'Noninteger physical coordinate')
    coords={(x['loop'],x['layer'],x['site']) for x in index};keys=[x['key'] for x in index]
    checked(len(index)==40 and coords==expected and len(set(keys))==40,'Missing or duplicate physical site')
    checked(all(re.fullmatch(r'v\d+',key) for key in keys),'Invalid state-array key')
    expected_shapes={x['key']:(layout.widths[x['layer']][x['site']],) for x in index}
    expected_shapes['boundaries']=(2,layout.hidden)
    if require_instrumented:
        expected_shapes['native_at_reached_input']=(2,layout.hidden)
        expected_shapes['instrument_input_last_token']=(2,layout.widths[21]['down_proj_input'])
    checked(mode=='native' or require_instrumented,'Edited condition lacks required BPD snapshots')
    if mode!='native':expected_shapes['actual_masks']=(2,layout.components)
    checked(Path(path).stat().st_size<=sum(math.prod(s)*4+16384 for s in expected_shapes.values())+65536,'Oversized NPZ container')
    checked(file_hash(path)==prefix['state_sha256'],'State checksum changed')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();checked(len(names)==len(expected_shapes) and set(names)=={k+'.npy' for k in expected_shapes},'Unexpected NPZ member')
        needed={}
        for item in index:
            p,l,s=item['loop'],item['layer'],item['site'];raw=read_npy(archive,item['key'],expected_shapes[item['key']],'BF16')
            if l==21 and s in ('down_proj_input','down_proj_output'):needed[p,s]=raw
        read_npy(archive,'boundaries',expected_shapes['boundaries'],'BF16')
        if require_instrumented:
            native=read_npy(archive,'native_at_reached_input',expected_shapes['native_at_reached_input'],'BF16')
            inputs=read_npy(archive,'instrument_input_last_token',expected_shapes['instrument_input_last_token'],'BF16')
            for p in (1,2):
                size=2*layout.widths[21]['down_proj_input']
                checked(inputs[(p-1)*size:p*size]==needed[p,'down_proj_input'],'BPD input coordinate differs from physical input')
                if mode in ('native','all_one'):
                    size=2*layout.hidden
                    checked(native[(p-1)*size:p*size]==needed[p,'down_proj_output'],'Native/all-one local-write identity failed')
        if mode!='native':
            raw=read_npy(archive,'actual_masks',expected_shapes['actual_masks'],'F32')
            if mode in ('all_one','component_only'):checked(raw==struct.pack('<f',1.)*(2*layout.components),'Nonadaptive route mask changed')


def audit_episode(base,entry,source,*,members,ordinal,lane,original_row,decode,parse,eos_token_id,layout,
                  token_bound,require_instrumented=True,allow_CPU_fixture_timing=False):
    """Audit one closed row without labels; companions are a separate proof."""
    base=Path(base);checked(lane in ('baseline','AB') and type(ordinal) is int and ordinal>=0,'Invalid scheduled slot')
    checked(1<=len(members)<=2 and type(original_row) is int and 0<=original_row<len(members),'Invalid original row')
    checked(all(m['mode'] in MODES and m['initial_ids'] for m in members),'Invalid scheduled members')
    if lane=='baseline':checked(all(m['mode']=='native' for m in members),'Edited baseline is not registered')
    else:
        checked(len(members)==2 and members[0]['mode']=='native' and members[1]['mode'] in MODES[2:],'Genuine clean/edit B2 required')
        checked(members[0]['source_id']==members[1]['source_id'] and members[0]['initial_ids']==members[1]['initial_ids'],'A/B source mismatch')
    stem=f'{lane}_{ordinal:04d}_r{original_row}';checked(entry['file']==stem+'_closed.json','Closed filename differs from scheduled slot')
    episode=read_json(base/entry['file'],entry['sha256']);checked(episode['schema']=='cached-episode-v2','Cached schema not admitted')
    member=members[original_row];initial=member['initial_ids'];mode=member['mode']
    checked(member['source_id']==source['id'],'Scheduled source differs')
    checked(all(episode[k]==source[k] for k in ('root','stage','split')),'Closed source metadata changed')
    checked(episode['source_id']==source['id'] and episode['mode']==mode and episode['lane']==lane,'Closed route changed')
    checked(episode['initial_ids']==initial and len(initial)==token_bound['tokens'] and ids_hash(initial)==token_bound['ids_sha256'],'Prompt/token pin changed')
    generated=episode['generated_ids'];cap=64 if source['stage']=='nan_story23' else 256 if source['schema']['type']=='grid' else 128
    checked(all(type(t) is int and 0<=t<layout.vocabulary for m in members for t in m['initial_ids']),'Invalid initial token')
    checked(1<=len(generated)<=cap and len(generated)==episode['prefix_count']==len(episode['fragments']),'Invalid generated extent')
    checked(all(type(t) is int and 0<=t<layout.vocabulary for t in generated),'Invalid generated token')
    initial_canvas=max(len(m['initial_ids']) for m in members);parent='0'*64;proofs=[]
    batch_identity=context_hash([dict(source_id=m['source_id'],mode=m['mode'],initial_ids_sha256=ids_hash(m['initial_ids'])) for m in members])
    for step,fragment in enumerate(episode['fragments']):
        name=f'{stem}_p{step:03d}.json';checked(fragment['file']==name,'Fragment slot changed')
        p=read_json(base/name,fragment['sha256']);checked(p['schema']=='cached-prefix-v2','Prefix schema not admitted')
        checked(p['step']==step and p['source_id']==source['id'] and p['mode']==mode and p['token']==generated[step],'Prefix identity changed')
        checked(p['history_sha256']==ids_hash(generated[:step]) and p['prefix_ids_sha256']==ids_hash(initial+generated[:step]),'Token/history checksum changed')
        context=p['batch_context'];B=p['batch_size'];row=p['batch_row'];P=initial_canvas+step
        checked(type(B) is int and 1<=B<=len(members) and type(row) is int and 0<=row<B,'Invalid active batch')
        fields=('source_ids','modes','prefix_ids_sha256','history_sha256','semantic_lengths','initial_semantic_lengths','attention_mask_runs','original_row_indices')
        checked(set(context)==set(fields)|{'physical_cache_length','attention_mask_encoding','attention_mask_sha256','parent_cache_context_sha256','original_prefill_canvas'},'Unexpected cache context fields')
        checked(all(len(context[k])==B for k in fields),'Missing companion metadata')
        indices=context['original_row_indices'];checked(indices==sorted(set(indices)) and all(type(i) is int and 0<=i<len(members) for i in indices),'Invalid cache row lineage')
        checked(indices[row]==original_row and (step>0 or indices==list(range(len(members)))),'Serial substitute or original row changed')
        checked(context['physical_cache_length']==p['padded_length']==P and context['original_prefill_canvas']==initial_canvas,'Cache canvas reset or compacted')
        checked(context['parent_cache_context_sha256']==parent,'Cache ancestry changed')
        checked(context['attention_mask_encoding']=='half-open-runs-v1','Lossless mask encoding changed')
        masks=[]
        for b,index in enumerate(indices):
            m=members[index];n=len(m['initial_ids']);length=n+step
            checked(context['source_ids'][b]==m['source_id'] and context['modes'][b]==m['mode'],'Companion identity/route changed')
            checked(context['semantic_lengths'][b]==length and context['initial_semantic_lengths'][b]==n,'Semantic position changed')
            runs=[[0,n]]
            if step:
                if n==initial_canvas:runs[0][1]=P
                else:runs.append([initial_canvas,P])
            checked(context['attention_mask_runs'][b]==runs,'Cached attention hole changed')
            masks.append([1]*n+[0]*(initial_canvas-n)+[1]*step)
            for key in ('prefix_ids_sha256','history_sha256'):checked(re.fullmatch('[0-9a-f]{64}',context[key][b]) is not None,'Invalid companion hash')
        checked(context['attention_mask_sha256']==ids_hash(masks),'Actual attention checksum changed')
        for field,key in (('source_id','source_ids'),('mode','modes'),('prefix_ids_sha256','prefix_ids_sha256'),('history_sha256','history_sha256'),('actual_length','semantic_lengths')):
            checked(p[field]==context[key][row],'Own prefix disagrees with batch metadata')
        checked(p['padding_tokens']==sum(P-n for n in context['semantic_lengths']),'Padding count changed')
        checked(p['cache_bytes_for_entire_batch']==layout.cache_bytes_per_row_token*P*B,'Actual cache allocation extent changed')
        parent=context_hash(context);checked(p['complete_batch_context_sha256']==parent,'Cache context checksum changed')
        checked(p['full_vocabulary_size']==layout.vocabulary and p['full_logits_independently_reconstructible_from_export'] is False,'Unsupported vocabulary/argmax claim')
        checked(math.isfinite(p['top_logit']) and math.isfinite(p['log_normalizer']) and -1e-10<=p['log_normalizer']-p['top_logit']<=math.log(layout.vocabulary)+1e-8,'Invalid logit metadata')
        checked(math.isfinite(p['wall_seconds_for_entire_batch']) and p['wall_seconds_for_entire_batch']>=0,'Invalid wall time')
        cuda=p['cuda_elapsed_ms_for_entire_batch']
        checked((cuda is None and allow_CPU_fixture_timing) or (isinstance(cuda,(int,float)) and math.isfinite(cuda) and cuda>=0),'Missing CUDA time')
        checked(p['state_file']==f'{stem}_p{step:03d}.npz','State filename changed')
        audit_states(base/p['state_file'],p,layout,mode,require_instrumented)
        eos=generated[step]==eos_token_id;text=decode(generated[:step] if eos else generated[:step+1])
        stop='eos' if eos else 'complete_packet' if parse(text,source) is not None else 'character_cap' if len(text)>(512 if source['stage']=='nan_story23' else 4096) else 'token_cap' if step+1>=cap else None
        checked(stop is None if step<len(generated)-1 else stop==episode['stop_reason'] and text==episode['text'],'Generation violated bounded stop rule')
        proofs.append(dict(file=name,sha256=fragment['sha256'],step=step,batch_size=B,batch_row=row,original_row=original_row,
            original_rows=indices,context_sha256=parent,parent_context_sha256=context['parent_cache_context_sha256']))
    return dict(status='episode_audited',source_id=source['id'],mode=mode,lane=lane,ordinal=ordinal,original_row=original_row,
        original_batch_sha256=batch_identity,closed_file=entry['file'],closed_sha256=entry['sha256'],prefixes=proofs,physical_vectors=40*len(proofs),
        instrumented_snapshots_verified=require_instrumented,BPD_shadow_and_local_write_replay_still_required=mode not in ('native','all_one'),
        full_vocabulary_argmax_independently_reconstructed=False,model_target_stage_and_terminal_checks_still_required=True)


def complete_batch_signatures(audits):
    """One original scheduled batch at a time; memory stays bounded by B2.

    A missing/invalid companion excludes that physical batch and every cache
    descendant. Independently audited own behavior need not be discarded;
    excluded geometry/control outcomes stay explicitly unknown.
    """
    checked(1<=len(audits)<=2,'One scheduled B1/B2 group required')
    checked(len({(a['lane'],a['ordinal'],a['original_batch_sha256']) for a in audits})==1,'Different scheduled cache batches')
    checked(len({a['original_row'] for a in audits})==len(audits),'Duplicate audited row')
    groups={};result={};previous='0'*64;parent_verified=True
    for audit in audits:
        checked(audit['status']=='episode_audited','Unaudited episode')
        for p in audit['prefixes']:groups.setdefault(p['step'],[]).append(p)
    for step,rows in sorted(groups.items()):
        counts={p['batch_size'] for p in rows};contexts={p['context_sha256'] for p in rows}
        good=len(counts)==1 and len(rows)==next(iter(counts)) and {p['batch_row'] for p in rows}==set(range(len(rows)))
        reason='missing_or_inconsistent_audited_companion'
        if good:
            good=len(contexts)==1 and all(p['original_rows']==[q['original_row'] for q in sorted(rows,key=lambda q:q['batch_row'])] for p in rows)
            if not good:reason='companion_cache_context_mismatch'
        if good and (not parent_verified or any(p['parent_context_sha256']!=previous for p in rows)):
            good=False;reason='unverified_parent_cache_context'
        for p in rows:result[p['file']]=dict(signature=p['context_sha256'] if good else None,reason=None if good else reason)
        previous=next(iter(contexts)) if len(contexts)==1 else None;parent_verified=good
    return result
