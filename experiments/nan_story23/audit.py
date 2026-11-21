"""Independent terminal replay of raw tokens, world grades and all physical states."""
import argparse
from collections import defaultdict
from pathlib import Path
from .common import ROOT,RUNS,SDK_COMMIT,eligible_rows,public_rows,read,rows,sha,verify,write
from .oracle import grade,trace_packet


def coverage(public,outcomes,eligible):
    groups=[];keys=('split','query_type','event_count','arm','mode')
    grouped=defaultdict(list)
    for r in public:grouped[tuple(r[k] for k in keys)].append(r)
    for key,sources in sorted(grouped.items()):
        done=[r for r in sources if r['id'] in outcomes];n=len(sources)
        values=dict(zip(keys,key))|dict(registered_trajectories=n,eligible_trajectories=sum(r['id'] in eligible for r in sources),
            graded_trajectories=len(done),missing_graded_trajectories=n-len(done),registered_roots=len({r['root'] for r in sources}))
        for endpoint in ('format_valid','final_correct','path_correct','trace_exact'):
            correct=sum(outcomes[r['id']][endpoint] for r in done)
            pair=defaultdict(list)
            for r in done:pair[r['root']].append((r['world'],outcomes[r['id']][endpoint]))
            complete=[v for v in pair.values() if len(v)==2 and {x[0] for x in v}=={'A','B'}]
            values[endpoint]=dict(correct=correct,whole_roster_lower_bound=correct/n,whole_roster_upper_bound=(correct+n-len(done))/n,
                complete_world_pairs=len(complete),both_world_roots_correct=sum(all(x[1] for x in v) for v in complete))
        values['intermediate_correct']=[sum(outcomes[r['id']]['intermediate_correct'][i] for r in done) for i in range(3)]
        groups.append(values)
    return groups


def run(backend,snapshot):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal();seal=verify();cfg=read(ROOT/'config.json');public=public_rows();eligible=eligible_rows(backend)
    import numpy as np
    from transformers import PreTrainedTokenizerFast
    from types import SimpleNamespace
    from experiments.nan_gym05.common import prompt_ids
    from experiments.nan_prime20.wire import OrderedSlots
    from mechinterp.nanbeige_bpd_gym.prime_evidence import successful_episode,raw_reply_bound
    from .engine import ids_hash
    events=rows(RUNS/'queue_events.jsonl')
    for name in ('model_worker_terminal','official_runner_terminal_or_skipped'):
        assert len([e for e in events if e['event']==name and e['backend']==backend])==1
    base=RUNS/backend;terminal=read(base/'receipt.json');assert not terminal['mock_only']
    count=terminal['requests'];assert 0<=count<=len(eligible)
    if 'target_unchanged' in terminal:assert terminal['target_unchanged']
    if count:
        checks=read(base/'preflight.json');assert checks['embedding_cache_initial_and_extension_exact']
        assert all(c['vocabulary_logits_bit_exact'] and c['all_scorer_fields_exact'] and c['full_vocabulary_logits_checked']==166144 for c in checks['full_head_parity'])
    tok=SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(snapshot)/'tokenizer.json')))
    bindings=read(ROOT/'fixtures/tokenizer_receipt.json')
    for name,h in bindings['tokenizer_files_sha256'].items():assert sha(Path(snapshot)/name)==h
    gold={r['id']:r for r in rows(ROOT/'fixtures/host_labels.jsonl')}
    fence=OrderedSlots(eligible);records={};prefixes=[];vectors=0
    for ordinal,source in enumerate(eligible):
        file=base/f'request_{ordinal:04d}.json'
        fragments=sorted(base.glob(f'request_{ordinal:04d}_prefix_*.json'))
        if not file.exists() and not fragments:continue
        initial=prompt_ids(tok,snapshot,source['prompt']);assert ids_hash(initial)==bindings['contexts'][source['id']]['ids_sha256']
        generated=[]
        for step,fragmentfile in enumerate(fragments):
            r=read(fragmentfile);assert r['step']==step and r['source_id']==source['id']
            assert r['prefix_ids_sha256']==ids_hash(initial+generated) and r['generated_history_sha256']==ids_hash(generated)
            assert r['generated_token_id']==r['scores'][-1]['free_top_token_id']
            assert len(r['scores'])==2
            token=r['generated_token_id'];generated.append(token)
            assert r['eos']==(token==tok.tok.eos_token_id)
            text=tok.tok.decode(generated[:-1] if r['eos'] else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert text==r['text_after']
            if step<len(fragments)-1:
                assert not r['eos'] and trace_packet(text,source['schema']) is None and len(text)<=512
            state=base/r['state_file'];assert state.name==state.relative_to(base).as_posix() and sha(state)==r['state_file_sha256']
            with np.load(state,allow_pickle=False) as z:
                expected={(p,l,s) for p in (1,2) for l in cfg['layers_zero_based'] for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
                coords={(x['loop'],x['layer_zero_based'],x['site']) for x in r['state_index']};assert len(r['state_index'])==40 and coords==expected
                for item in r['state_index']:
                    v=z[item['array_key']];width=10752 if item['site']=='down_proj_input' else 3072
                    assert v.shape==(width,) and v.dtype==np.float32 and np.isfinite(v).all();vectors+=1
                assert z['boundary'].shape==(2,1,1,3072) and np.isfinite(z['boundary']).all()
                assert z['native_output_at_reached_input'].shape==(2,3072) and np.isfinite(z['native_output_at_reached_input']).all()
                if source['mode']!='native':
                    assert z['actual_mask'].shape==(2,8) and np.isfinite(z['actual_mask']).all() and z['actual_mask'].min()>=0 and z['actual_mask'].max()<=1
            prefixes.append(dict(source_id=source['id'],ordinal=ordinal,step=step,record_file=fragmentfile.name,
                record_sha256=sha(fragmentfile),state_file=state.name,state_sha256=sha(state),closed_neural_reply=file.exists(),
                prefix_ids_sha256=r['prefix_ids_sha256']))
        if file.exists():
            r=read(file);i,key,matched=fence.claim(r['request'],backend,cfg)
            assert i==ordinal==r['source_ordinal'] and matched['id']==source['id']==r['source_id'] and not r['mock_only']
            assert r['prompt_sha256']==key and r['initial_prompt_ids']==initial and r['generated_ids']==generated and 1<=len(generated)<=64
            assert len(r['fragments'])==len(fragments)
            assert all(v['file']==f.name and v['sha256']==sha(f) for v,f in zip(r['fragments'],fragments))
            assert r['text']==text and r['embedding_rows_gathered']==len(initial)+len(generated)-1
            reason='eos' if read(fragments[-1])['eos'] else 'first_complete_packet' if trace_packet(text,source['schema']) is not None else 'character_cap' if len(text)>512 else 'token_cap'
            assert r['stop_reason']==reason and (reason!='token_cap' or len(generated)==64)
            assert r['finish_reason']==('stop' if reason in ('eos','first_complete_packet') else 'length')
            assert r['usage']==dict(prompt_tokens=len(initial),completion_tokens=len(generated),total_tokens=len(initial)+len(generated))
            records[source['id']]=r;fence.commit(source['id'])
        else:assert ordinal==count,'Only a final attempted orphan can have partial prefix artifacts'
    assert len(records)==count
    tracefile=RUNS/f'prime/nan-story23-{backend}-model/traces.jsonl';episodes=rows(tracefile) if tracefile.exists() else []
    assert len(episodes)<=len(eligible);outcomes={};bound=set()
    for ordinal,(source,episode) in enumerate(zip(eligible,episodes)):
        data=episode['task']['data'];assert data['idx']==ordinal and data['id']==source['id'] and data['prompt']==source['prompt']
        assert data['gold_packet']==gold[source['id']] and data['schema_packet']==source['schema'] and data['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
        for trace in episode['traces']:assert trace['verifiers']['commit']==SDK_COMMIT and trace['agent']['runtime']['type']=='subprocess' and trace['agent']['config']['harness']['id']=='null'
        if not successful_episode(episode):continue
        trace=episode['traces'][0];record=records[source['id']];assert source['id'] not in bound;bound.add(source['id'])
        assert raw_reply_bound(record,trace) and len(trace['calls'])==trace['num_turns']==1 and len(trace['nodes'])==2
        assert trace['nodes'][0]['message']==dict(role='user',content=source['prompt'])
        result=grade(record['text'],source['schema'],gold[source['id']])
        assert trace['info']['endpoints']==result and trace['reward']==float(result['trace_exact'])
        assert trace['num_input_tokens']==record['usage']['prompt_tokens'] and trace['num_output_tokens']==len(record['generated_ids'])
        outcomes[source['id']]=result
    write(RUNS/'audits'/(backend+'_prefixes_local.json'),dict(prefixes=prefixes,outcomes=outcomes))
    result=dict(status='passed',backend=backend,worker_status=terminal['status'],registered_trajectories=144,eligible_trajectories=len(eligible),
        neural_replies=count,official_episodes_recorded=len(episodes),graded_trajectories=len(outcomes),missing_graded_trajectories=144-len(outcomes),
        orphan_neural_replies=count-len(bound),captured_prefixes=len(prefixes),physical_vectors_checked=vectors,
        local_index_sha256=sha(RUNS/'audits'/(backend+'_prefixes_local.json')),
        groups=coverage(public,outcomes,{r['id'] for r in eligible}),worker_receipt_sha256=sha(base/'receipt.json'),
        source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),official_trace_sha256=sha(tracefile) if tracefile.exists() else None,
        formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False)
    write(RUNS/'audits'/(backend+'.json'),result);print(dict(backend=backend,captured_prefixes=len(prefixes),graded=len(outcomes)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=('bf16','nf4'));p.add_argument('--snapshot',required=True);a=p.parse_args()
    run(a.backend,a.snapshot)
