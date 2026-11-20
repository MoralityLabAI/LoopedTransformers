"""Independent SDK/worker/native-prefix audit; no orphan answer rescue."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.nan_prime06.wire import prompt_sha,validate_request
from mechinterp.nanbeige_bpd_gym.audit_native import final_answer
from mechinterp.nanbeige_bpd_gym.prime_evidence import successful_episode,raw_reply_bound
from .common import public_rows,verify_source,read,rows,sha,write,SDK_COMMIT


def data_matches(data,r,gold):
    expected=dict(id=r['id'],prompt=r['prompt'],source_root=r['root'],context_budget=r['context_budget'],position=r['position'],
        task_kind=r['task'],world=r['world'],copy_control=r['copy_control'],answer_schema=r['schema'],private_answer=gold[r['id']],
        source_manifest_sha256=sha('experiments/nan_ctx08/SOURCE_MANIFEST.json'))
    assert all(data[k]==v for k,v in expected.items())


def audit(runs,backend,seal,cfg,public,expected,gold,eos_id):
    folder=runs/backend;terminal=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None
    progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {'requests':0}
    count=terminal['requests'] if terminal else progress['requests'];by={prompt_sha(r['prompt']):r for r in expected}
    indices={r['id']:i for i,r in enumerate(public)};records={};vectors=0
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json');assert not m['mock_only'] and not m['host_labels_opened']
        assert m['source_manifest_sha256']==sha(Path(__file__).parent/'SOURCE_MANIFEST.json') and m['maximum_generated_tokens']==1
        parity=read(folder/'resident_head_parity.json');assert len(parity['checks'])==2
        assert all(c['vocabulary_logits_bit_exact'] and c['all_scorer_fields_exact'] for c in parity['checks'])
    for ordinal in range(count):
        r=read(folder/f'request_{ordinal:04d}.json');assert not r['mock_only'] and r['completion_id']==f'nan-prime-ctx10-{backend}-{ordinal}'
        key,source=validate_request(r['request'],by,backend,cfg);assert key==r['prompt_sha256'] and key not in records
        native_file=Path('runs/nan_ctx08')/backend/f"case_{indices[source['id']]:04d}.json";native=read(native_file)
        assert native['id']==source['id'] and native['initial_prompt_ids']==r['initial_prompt_ids'] and r['phases']==native['phases']
        token=native['phases'][1]['score']['free_top_token_id'];assert r['generated_ids']==[token]
        assert r['text']==('' if token==eos_id else native['phases'][1]['free_token_text'])
        assert r['finish_reason']==('stop' if token==eos_id else 'length')
        u=r['usage'];assert u==dict(prompt_tokens=len(r['initial_prompt_ids']),completion_tokens=1,total_tokens=len(r['initial_prompt_ids'])+1)
        state=folder/r['state_file'];oldstate=native_file.parent/native['state_file']
        assert sha(state)==r['state_file_sha256'] and sha(oldstate)==native['state_file_sha256']
        logical=lambda record:{(x['loop'],x['layer_zero_based'],x['site']):x['array_key'] for x in record['state_index']}
        new,old=logical(r),logical(native);assert len(r['state_index'])==len(native['state_index'])==40 and new.keys()==old.keys()
        with np.load(state,allow_pickle=False) as z,np.load(oldstate,allow_pickle=False) as nz:
            for k,v in new.items():
                a,b=z[v],nz[old[k]];assert a.dtype==b.dtype==np.float32 and a.shape==b.shape and a.tobytes()==b.tobytes();vectors+=1
        records[key]=r
    traces=runs/f'prime/nan-prime-ctx10-{backend}-model/traces.jsonl';episodes=rows(traces) if traces.exists() else []
    assert len(episodes)<=288;outcomes={};trimmed=0;bound=set()
    for source,e in zip(expected,episodes):
        data_matches(e['task']['data'],source,gold)
        for t in e['traces']:assert t['verifiers']['commit']==SDK_COMMIT and t['agent']['runtime']['type']=='subprocess' and t['agent']['config']['harness']['id']=='null'
        if not successful_episode(e):continue
        t=e['traces'][0];data_matches(t['task']['data'],source,gold)
        assert len(t['calls'])==t['num_turns']==1 and len(t['nodes'])==2
        assert t['nodes'][0]['message']==dict(role='user',content=source['prompt'])
        key=prompt_sha(source['prompt']);assert key in records and key not in bound;bound.add(key);r=records[key]
        assert raw_reply_bound(r,t) and t['num_input_tokens']==r['usage']['prompt_tokens'] and t['num_output_tokens']==1
        assert t['calls'][0]['finish_reason']==r['finish_reason']
        value=final_answer(r['text'],source['schema']);correct=int(value==gold[source['id']]);assert t['reward']==correct and t['info']['format_valid']==(value is not None)
        outcomes[source['id']]=dict(correct=correct,format_valid=value is not None)
        trimmed+=int(r['text']!=t['last_reply'])
    summary=[];groups=defaultdict(list)
    for r in expected:groups[r['context_budget'],r['position'],r['task'],r['copy_control']].append(r)
    for key,sources in sorted(groups.items()):
        done=[r for r in sources if r['id'] in outcomes];correct=sum(outcomes[r['id']]['correct'] for r in done)
        summary.append(dict(context_budget=key[0],position=key[1],task=key[2],copy_control=key[3],registered_contexts=len(sources),graded_contexts=len(done),
            missing_graded_contexts=len(sources)-len(done),correct=correct,format_valid=sum(outcomes[r['id']]['format_valid'] for r in done),
            registered_root_families=len({r['root'] for r in sources}),graded_root_families=len({r['root'] for r in done}),
            whole_registered_correctness_lower_bound=correct/len(sources),whole_registered_correctness_upper_bound=(correct+len(sources)-len(done))/len(sources)))
    events=rows(runs/'queue_events.jsonl') if (runs/'queue_events.jsonl').exists() else []
    runner=[e for e in events if e.get('backend')==backend and e['event'] in ('official_runner_terminal','official_runner_skipped')]
    assert len(runner)<=1
    status='completed' if terminal and terminal['status']=='completed' and runner and runner[0].get('exit_code')==0 and len(outcomes)==288 else 'terminal_with_missing_coverage' if runner else 'partial_evidence' if count else 'queued'
    return dict(status=status,registered_contexts=288,neural_requests=count,successful_official_episodes=len(outcomes),official_episodes_recorded=len(episodes),
        missing_graded_contexts=288-len(outcomes),failed_official_episodes=len(episodes)-len(outcomes),orphan_or_unbound_neural_replies=count-len(bound),
        matched_native_prefixes=count,physical_prefix_vectors_byte_exact=vectors,SDK_outer_whitespace_trimmed=trimmed,groups=summary,
        worker_terminal_status=terminal['status'] if terminal else 'worker_exited_without_receipt' if runner else None,worker_receipt_sha256=sha(folder/'receipt.json') if terminal else None,
        official_runner_exit_code=runner[0].get('exit_code') if runner else None,official_traces_sha256=sha(traces) if traces.exists() else None)


def report(runs,out,snapshot):
    root=Path(__file__).parent;seal=verify_source(root);cfg=read(root/'config.json');public,dev=public_rows()
    for name,h in seal['host_only_files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['host_dependencies_sha256'].items():assert sha(name)==h,name
    for name,h in seal['native_evidence_sha256'].items():assert sha(name)==h,name
    ctx=Path('experiments/nan_ctx08');labels=ctx/'fixtures/host_labels.jsonl'
    assert sha(labels)==read(ctx/'SOURCE_MANIFEST.json')['host_only_files_sha256']['fixtures/host_labels.jsonl']
    gold={r['id']:r['answer'] for r in rows(labels)}
    assert sha(snapshot/'tokenizer_config.json')==seal['tokenizer_config_sha256']
    tok=read(snapshot/'tokenizer_config.json');eos=tok['eos_token'];eos=eos['content'] if isinstance(eos,dict) else eos
    assert sha(snapshot/'tokenizer.json')==seal['tokenizer_json_sha256']
    eos_id=next(r['id'] for r in read(snapshot/'tokenizer.json')['added_tokens'] if r['content']==eos)
    backends={b:audit(runs,b,seal,cfg,public,dev,gold,eos_id) for b in cfg['backends']}
    result=dict(experiment='NAN-PRIME-CTX-10',updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
        backends=backends,evidence_class='reused exposed development context task set; official local SDK endpoint/accuracy/physical-prefix audit',
        unit='four source-generator families, one checkpoint; contexts/positions/worlds/loops/backends are repeated measures',
        n_units=4,n_cases=288,primary_contrast='same-backend official one-token final reply versus CTX08 native loop-2 first token',
        physical_dose=0,scientific_variation_floor=None,matched_sham_floor=None,estimated_causal_resolution='not calibrated; exploratory only',
        readout_selection_exposure='CTX08 native outcomes and completed FIT07/RECOMP09 fits and probes exposed before registration',
        planned_formal_comparisons=0,actual_formal_comparisons=0,BPD_transfer_attempted=False,sealed_test_accessed=False,
        original_PRIME06_missing_coverage_unchanged=True,privacy='aggregate counts/root summaries/hashes only; raw traces, replies, tokens and states stay local')
    write(out/'nan_prime_ctx10_progress.json',result)
    lines=['# NAN-PRIME-CTX-10: official local SDK context evaluation','',f"Updated {result['updated_UTC']}",'',
        'All 288 CTX08 DEV contexts are registered: lookup/join, early/middle/late records, 1K/2K/4K, both source '+
        'worlds and all copy controls. This uses the same already exposed source families; it is not an independent '+
        'replication. The complete official final reply is capped at one unconstrained token, matching the native '+
        'CTX08 first-token endpoint. PRIME06 multi-token/grid/plan outcomes and its missing BF16 coverage are unchanged.','',
        '| Backend | Status | Fresh model replies | Successful SDK grades | Missing grades | Prefix vectors byte exact |',
        '|---|---|---:|---:|---:|---:|']
    for b,r in backends.items():lines.append(f"| {b.upper()} | {r['status']} | {r['neural_requests']} | {r['successful_official_episodes']} | {r['missing_graded_contexts']} | {r['physical_prefix_vectors_byte_exact']} |")
    lines+=['','Successful SDK episodes must bind to a distinct fresh neural response, its exact raw assistant node, '+
        'usage and finish reason. The SDK accessor may trim outer whitespace; raw text remains exact. Native prompt '+
        'IDs, both phase scores and all 40 physical prefix vectors must match CTX08 byte for byte. Orphan responses '+
        'never rescue failed SDK episodes. All registered strata, invalid answers and missing-coverage bounds are in the JSON.','',
        'The persistent official SDK uses local subprocess runtime and null harness, one turn and one concurrent '+
        'episode. No hosted inference or paid runtime. The model retains the unchanged BF16 head on GPU, all NF4 '+
        'decoder layers or four BF16 layers plus reusable stage; one CPU thread/Below Normal priority. One large '+
        'model at a time, 7200-second backend budget, no fresh neural retries.','',
        'The exact physical-prefix audit permits reuse of the fixed CTX08 SAE/VPD observations for matched states; '+
        'it does not turn observational geometry into causal evidence. New RECOMP09 BPD asset analysis, if eligible, '+
        'is a separately declared CTX10 cached-input probe. No feature-index subtraction, fitting or online donor '+
        'intervention is performed. The original 168-case test stays sealed.','',
        '[Protocol](../../experiments/nan_prime_ctx10/PROTOCOL.md) · [Source seal](../../experiments/nan_prime_ctx10/SOURCE_MANIFEST.json) · '+
        '[Aggregate evidence](nan_prime_ctx10_progress.json)','']
    (out/'NAN_PRIME_CTX10_PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({b:{k:r[k] for k in ('status','neural_requests','successful_official_episodes')} for b,r in backends.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--runs',default='runs/nan_prime_ctx10');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out),Path(a.snapshot))


if __name__=='__main__':main()
