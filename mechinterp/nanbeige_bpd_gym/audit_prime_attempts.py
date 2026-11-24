"""Audit primary/tail SDK attempts without converting timeouts to answers.

Post-registration CPU analysis of exposed development evidence only. Original
receipts remain intact. Equivalent duplicate input/reply occurrences receive a
distinct deterministic matching certificate, not an invented request identity.
"""
import argparse
from collections import Counter,defaultdict,deque
import math
import os
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from experiments.nan_prime06.wire import prompt_sha
from .audit_fit import dependency_matches
from .audit_native import final_answer,correctness
from .audit_prime import sudoku_valid
from .prime_evidence import raw_reply_bound,successful_episode
from .suite_rows import load_suite_rows


def request_key(record):
    u=record['usage']
    return (record['prompt_sha256'],record['text'],u['prompt_tokens'],u['completion_tokens'],record['finish_reason'])


def trace_key(trace,prompt):
    message=trace['nodes'][-1]['message'];assert message['role']=='assistant' and isinstance(message['content'],str)
    return (prompt_sha(prompt),message['content'],trace['num_input_tokens'],trace['num_output_tokens'],trace['calls'][0]['finish_reason'])


def distinct_occurrences(worker_keys,successful_keys):
    available=defaultdict(deque)
    for i,key in enumerate(worker_keys):available[key].append(i)
    interchangeable={key for key,values in available.items() if len(values)>1}
    assigned=[];used_signatures=set()
    for key in successful_keys:
        if not available[key]:raise ValueError('A successful SDK reply has no unused matching fresh neural record')
        assigned.append(available[key].popleft());used_signatures.add(key)
    assert len(assigned)==len(set(assigned))
    return assigned,len(interchangeable & used_signatures)


def condition(row):
    label=row['private_label']
    return 'copy' if label['copy_control'] else 'hop'+str(label['hops']) if label.get('hops') else 'main'


def check_task_data(data,source):
    for field in ('id','root_id','suite','prompt','private_label','source_manifest_sha256'):assert data[field]==source[field],field
    assert data['answer_schema']==source['schema']


def summaries(expected,outcomes):
    groups=defaultdict(list)
    for row in expected:groups[row['suite'],condition(row)].append(row)
    result=[]
    for (suite,arm),source in sorted(groups.items()):
        observed=[outcomes[r['id']] for r in source if r['id'] in outcomes]
        total=len(source);graded=len(observed);correct=sum(r['correct'] for r in observed)
        result.append(dict(suite=suite,condition=arm,registered_contexts=total,graded_contexts=graded,
            missing_graded_contexts=total-graded,correct=correct,format_valid=sum(r['format_valid'] for r in observed),
            registered_source_roots=len({r['root_id'] for r in source}),
            graded_source_roots=len({r['root_id'] for r in source if r['id'] in outcomes}),
            whole_registered_correctness_lower_bound=correct/total,
            whole_registered_correctness_upper_bound=(correct+total-graded)/total))
    return result


def native_records(backend):
    folder=Path('runs/nan_gym05')/backend;audit_file=Path('runs/nan_gym05/audit')/(backend+'_independent.json')
    checked=read(audit_file);assert checked['status']=='passed' and checked['execution_receipt_sha256']==sha(folder/'receipt.json')
    result={}
    for file in folder.glob('root_*.json'):
        assert sha(file)==checked['artifact_sha256'][file.name];result.update(read(file)['records'])
    return result,sha(audit_file)


def audit_attempt(runs,backend,attempt,expected,cfg,tokenizer,native):
    primary=attempt=='primary'
    folder=runs/backend if primary else runs/'continuation'/backend
    traces=runs/('prime' if primary else 'prime_tail1')/('nan-prime06-'+backend+('' if primary else '-tail1'))/'traces.jsonl'
    receipt_file=folder/'receipt.json'
    if not receipt_file.exists():return {'status':'pending','attempt':attempt},{}
    receipt=read(receipt_file);episodes=rows(traces) if traces.exists() else []
    # Require the controller's actual runner-terminal observation before a final
    # attempt audit. A response/receipt file alone is not process liveness proof.
    event_file=runs/('queue_events.jsonl' if primary else 'continuation/events.jsonl')
    events=rows(event_file) if event_file.exists() else []
    terminal=[r for r in events if r.get('backend')==backend and r['event']==('official_runner_exited' if primary else 'tail_runner_terminal')]
    if not terminal:return {'status':'awaiting_official_runner_terminal','attempt':attempt,'worker_receipt_sha256':sha(receipt_file)},{}
    assert len(terminal)==1 and len(episodes)<=len(expected)
    assert [e['task']['data']['id'] for e in episodes]==[r['id'] for r in expected[:len(episodes)]]
    assert len({r['id'] for r in expected})==len(expected)
    allowed={prompt_sha(r['prompt']):r for r in expected};files=sorted(folder.glob('request_*.json'))
    if (folder/'manifest.json').exists():
        manifest=read(folder/'manifest.json');assert not manifest['mock_only'] and not manifest['host_labels_opened']
        assert manifest['source_registration_sha256']==sha('experiments/nan_prime06/SOURCE_MANIFEST.json')
    else:assert not files and not any(successful_episode(e) for e in episodes)
    assert receipt.get('requests',0)==len(files)
    records=[];vectors=0;tokens=0;native_matches=0;complete_native_matches=0
    for ordinal,file in enumerate(files):
        r=read(file);assert r['completion_id']=='nan-prime06-'+backend+'-'+str(ordinal) and not r['mock_only']
        source=allowed[r['prompt_sha256']];assert r['request']['messages']==[dict(role='user',content=source['prompt'])]
        assert r['request']['model']=='nanbeige-451ed48c-'+backend and r['request'].get('temperature',0)==0
        assert r['initial_prompt_ids']==native[source['id']]['initial_prompt_ids']
        generated=r['generated_ids'];cap=cfg['maximum_'+source['schema']['type']+'_tokens']
        assert 1<=len(generated)<=cap and tokenizer.eos_token_id not in generated[:-1]
        ended=generated[-1]==tokenizer.eos_token_id;assert ended or len(generated)==cap
        assert r['finish_reason']==('stop' if ended else 'length')
        assert tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)==r['text']
        assert len(r['scores'])==len(generated)
        for token,score in zip(generated,r['scores']):
            assert token==score['free_top_token_id'] and type(token) is int and 0<=token<166144
            assert math.isfinite(score['full_log_normalizer'])
        u=r['usage'];assert u['prompt_tokens']==len(r['initial_prompt_ids']) and u['completion_tokens']==len(generated)
        assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
        statefile=folder/r['state_file'];assert sha(statefile)==r['state_file_sha256']
        want={(t,l,s) for t in (1,2) for l in (5,11,17,21) for s in
            ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
        assert len(r['state_index'])==40 and {(o['loop'],o['layer_zero_based'],o['site']) for o in r['state_index']}==want
        with np.load(statefile,allow_pickle=False) as z:
            assert set(z.files)=={o['array_key'] for o in r['state_index']}
            for o in r['state_index']:
                width={'down_proj_input':10752,'o_proj_input':6144}.get(o['site'],3072);v=z[o['array_key']]
                assert v.shape==(width,) and v.dtype==np.float32 and np.isfinite(v).all();vectors+=1
        old=native[source['id']]['generated_ids'];common=min(len(old),len(generated))
        native_matches+=int(old[:common]==generated[:common]);complete_native_matches+=int(len(generated)>=len(old) and generated[:len(old)]==old)
        tokens+=len(generated);records.append(r)
    successful=[];keys=[];failed=Counter();outcomes={};sdk_trimmed=0
    for source,e in zip(expected,episodes):
        check_task_data(e['task']['data'],source)
        for t in e['traces']:
            assert t['verifiers']['commit']==cfg['verifiers_commit']
            assert t['agent']['runtime']['type']=='subprocess' and t['agent']['config']['harness']['id']=='null'
        if not successful_episode(e):failed[source['suite'],condition(source)]+=1;continue
        t=e['traces'][0];assert len(t['calls'])==t['num_turns']==1 and len(t['nodes'])==2
        check_task_data(t['task']['data'],source)
        assert t['nodes'][0]['message']==dict(role='user',content=source['prompt'])
        keys.append(trace_key(t,source['prompt']));successful.append((source,e))
    assigned,interchangeable=distinct_occurrences([request_key(r) for r in records],keys)
    local=[]
    for (source,e),ordinal in zip(successful,assigned):
        t=e['traces'][0];r=records[ordinal];assert raw_reply_bound(r,t)
        sdk_trimmed+=int(r['text']!=t['last_reply'])
        value=final_answer(r['text'],source['schema']);label=source['private_label'];correct=correctness(value,label)
        assert correct==t['reward'] and (value is not None)==t['info']['format_valid']
        if source['suite']=='sudoku' and not label['copy_control']:assert sudoku_valid(value,label['givens'])==t['info']['valid_sudoku_solution']
        outcomes[source['id']]=dict(correct=correct,format_valid=value is not None,attempt=attempt)
        local.append(dict(id=source['id'],neural_request_ordinal=ordinal,neural_record_sha256=sha(files[ordinal]),**outcomes[source['id']]))
    local_file=runs/'audit/attempts'/f'{backend}_{attempt}_matches_local.json';write(local_file,local)
    return dict(status='passed',attempt=attempt,worker_terminal_status=receipt['status'],
        official_runner_exit_code=terminal[0]['exit_code'],selected_contexts=len(expected),official_episodes_recorded=len(episodes),
        successful_official_episodes=len(successful),failed_or_incomplete_official_episodes=len(episodes)-len(successful),
        selected_contexts_without_recorded_official_episode=len(expected)-len(episodes),fresh_neural_requests=len(records),
        neural_requests_without_successful_SDK_binding=len(records)-len(successful),
        distinct_neural_reply_bindings=len(assigned),interchangeable_duplicate_reply_signature_groups=interchangeable,
        binding_scope='distinct occurrence certificate for identical prompt/raw reply/usage/finish; equivalent duplicate event identities remain unresolved',
        SDK_last_reply_outer_whitespace_trimmed_episodes=sdk_trimmed,physical_vectors_checked=vectors,generated_tokens=tokens,
        native_common_prefix_token_agreements=native_matches,complete_native_prefix_token_agreements=complete_native_matches,
        failed_episode_groups=[dict(suite=k[0],condition=k[1],episodes=v) for k,v in sorted(failed.items())],
        raw_failure_receipts=len(list(folder.glob('failure_*.json'))),summaries=summaries(expected,outcomes),
        worker_receipt_sha256=sha(receipt_file),official_traces_sha256=sha(traces) if traces.exists() else None,
        local_matching_certificate_sha256=sha(local_file)),outcomes


def policy_unfinished(expected,episodes):
    finished=set()
    for e in episodes:
        if e['ok'] and len(e['traces'])==1:
            t=e['traces'][0]
            if t['ok'] and t['is_completed'] and not t['is_timeout'] and len(t['calls'])==1:finished.add(e['task']['data']['id'])
    return [r for r in expected if r['id'] not in finished]


def audit(runs,backend,snapshot):
    root=Path('experiments/nan_prime06');seal=read(root/'SOURCE_MANIFEST.json');cfg=read(root/'config.json')
    assert sha(root/'SOURCE_MANIFEST.json')=='98973e25cfa188d0a7f67bd2ebcc30433efd8145e9bb8bd358ef52aa589a7b47'
    for name,h in seal['files_sha256'].items():assert sha(root/name)==h,name
    for name,h in seal['dependencies_sha256'].items():assert dependency_matches(Path(name).read_bytes(),h),name
    expected=load_suite_rows(scope='local',include_copy_controls=True);assert len(expected)==162
    cache=runs/'tokenizer_cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    from transformers import PreTrainedTokenizerFast
    assert sha(snapshot/'tokenizer_config.json')==read('experiments/nan_gym05/SOURCE_MANIFEST.json')['tokenizer_config_sha256']
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(snapshot/'tokenizer.json'));config=read(snapshot/'tokenizer_config.json')
    tokenizer.bos_token=config['bos_token'];tokenizer.eos_token=config['eos_token']
    native,native_audit_sha=native_records(backend)
    primary,first=audit_attempt(runs,backend,'primary',expected,cfg,tokenizer,native)
    outcomes=dict(first);tail={'status':'not_recorded'};selection=runs/'continuation'/(backend+'_selection_receipt.json')
    if selection.exists():
        registration=Path('mechinterp/nanbeige_bpd_gym/PRIME_CONTINUATION_REGISTRATION.json');registered=read(registration)
        for name,h in registered['source_files_sha256'].items():assert sha(name)==h
        primary_traces=runs/'prime'/('nan-prime06-'+backend)/'traces.jsonl';episodes=rows(primary_traces) if primary_traces.exists() else []
        selected=policy_unfinished(expected,episodes);selected_receipt=read(selection)
        assert selected_receipt['remaining_tasks']==len(selected)
        assert selected_receipt['primary_trace_sha256']==(sha(primary_traces) if primary_traces.exists() else None)
        if selected:
            ids_file=runs/'continuation'/(backend+'_pending_ids.txt')
            assert ids_file.read_text(encoding='utf-8').splitlines()==[r['id'] for r in selected]
            tail,second=audit_attempt(runs,backend,'tail1',selected,cfg,tokenizer,native)
            assert not (set(first)&set(second))
            outcomes.update(second)
        else:tail={'status':'not_required','selected_contexts':0}
    return dict(experiment='NAN-PRIME-06',status='passed',backend=backend,auditor_source_sha256=sha(__file__),
        source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),native_independent_audit_sha256=native_audit_sha,
        primary=primary,tail1=tail,unique_successfully_graded_contexts=len(outcomes),registered_contexts=162,
        missing_graded_contexts=162-len(outcomes),union_complete=len(outcomes)==162,unique_task_summaries=summaries(expected,outcomes),
        scope='primary and at most one outcome-blind operational tail kept separate; no failed SDK episode rescued with an orphan neural reply',
        bounds='fixed-cohort missing-outcome bounds, not confidence intervals or independent model replications',
        formal_inferential_comparisons=0,BPD_donor_transfer_enabled=False,sealed_test_accessed=False,
        privacy='only aggregates/hashes; task IDs, raw replies, tokens, vectors and per-record occurrence certificates stay local')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_prime06');p.add_argument('--backend',choices=('nf4','bf16'),required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.runs),a.backend,Path(a.snapshot));write(a.out,result)
    print({k:result[k] for k in ('status','backend','unique_successfully_graded_contexts','registered_contexts','missing_graded_contexts')})


if __name__=='__main__':main()
