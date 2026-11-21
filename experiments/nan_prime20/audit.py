"""Terminal SDK/source-slot/native-prefix audit; no orphan reply rescue."""
import argparse
from collections import defaultdict
from pathlib import Path
from .common import ROOT, RUNS, COMP14_SEAL, SDK_COMMIT, public_rows, read, rows, sha, verify, write


def run(backend, snapshot):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json'); _, public = public_rows()
    import numpy as np
    from .wire import OrderedSlots
    from experiments.nan_comp14.audit import answer
    from mechinterp.nanbeige_bpd_gym.prime_evidence import successful_episode, raw_reply_bound
    terminal_events = [e for e in rows(RUNS / 'queue_events.jsonl') if e.get('backend') == backend and e['event'] == 'model_worker_terminal']
    runner_events = [e for e in rows(RUNS / 'queue_events.jsonl') if e.get('backend') == backend and e['event'] in ('official_runner_terminal', 'official_runner_skipped')]
    assert len(terminal_events) == len(runner_events) == 1
    base = RUNS / backend; terminal = read(base / 'receipt.json')
    count = terminal['requests']; assert 0 <= count <= 96 and not terminal['mock_only']
    labels = {r['id']: r['answer'] for r in rows('experiments/nan_comp14/fixtures/host_labels.jsonl')}
    token_cfg = read(Path(snapshot) / 'tokenizer_config.json'); eos = token_cfg['eos_token']; eos = eos['content'] if isinstance(eos, dict) else eos
    eos_id = next(r['id'] for r in read(Path(snapshot) / 'tokenizer.json')['added_tokens'] if r['content'] == eos)
    assert sha(Path(snapshot) / 'tokenizer_config.json') == seal['tokenizer_config_sha256']
    assert sha(Path(snapshot) / 'tokenizer.json') == seal['tokenizer_json_sha256']
    fence = OrderedSlots(public); records = {}; vectors = 0
    if (base / 'manifest.json').exists():
        manifest = read(base / 'manifest.json')
        assert not manifest['host_labels_opened'] and not manifest['mock_only'] and manifest['registered_contexts'] == 96
        assert manifest['source_manifest_sha256'] == sha(ROOT / 'SOURCE_MANIFEST.json') and manifest['maximum_generated_tokens'] == 1
        parity = read(base / 'resident_head_parity.json')['checks']
        assert len(parity) == 2 and all(r['full_vocabulary_logits_checked'] == 166144 and r['vocabulary_logits_bit_exact'] and r['all_scorer_fields_exact'] for r in parity)
    for i in range(count):
        record = read(base / f'request_{i:04d}.json'); ordinal, key, source = fence.claim(record['request'], backend, cfg)
        assert ordinal == i == record['source_ordinal'] and record['source_id'] == source['id'] and not record['mock_only']
        assert record['prompt_sha256'] == key and record['completion_id'] == f'nan-prime20-{backend}-{i}'
        nativefile = Path('runs/nan_comp14') / backend / f"case_{source['upstream_ordinal']:04d}.json"
        locked = seal['native_evidence'][backend][source['id']]
        assert sha(nativefile) == locked['record_sha256']; native = read(nativefile)
        assert native['id'] == source['id'] and native['initial_prompt_ids'] == record['initial_prompt_ids']
        for now, old in zip(record['phases'], native['initial_phases']):
            assert now['loop'] == old['loop'] and now['score'] == old['score'] and now['free_token_text'] == old['token_text']
        assert len(record['phases']) == len(native['initial_phases']) == 2
        token = native['initial_phases'][1]['score']['free_top_token_id']
        assert record['generated_ids'] == [token] and record['text'] == ('' if token == eos_id else native['initial_phases'][1]['token_text'])
        assert record['finish_reason'] == ('stop' if token == eos_id else 'length')
        assert record['usage'] == dict(prompt_tokens=len(record['initial_prompt_ids']), completion_tokens=1, total_tokens=len(record['initial_prompt_ids'])+1)
        state, oldstate = base / record['state_file'], nativefile.parent / native['state_file']
        assert sha(state) == record['state_file_sha256'] and sha(oldstate) == native['state_file_sha256'] == locked['state_sha256']
        logical = lambda value: {(r['loop'], r['layer_zero_based'], r['site']): r['array_key'] for r in value['state_index']}
        new, old = logical(record), logical(native); assert len(new) == len(record['state_index']) == 40 and new.keys() == old.keys()
        with np.load(state, allow_pickle=False) as z, np.load(oldstate, allow_pickle=False) as reference:
            for coordinate, name in new.items():
                a, b = z[name], reference[old[coordinate]]
                assert a.dtype == b.dtype == np.float32 and a.shape == b.shape and a.tobytes() == b.tobytes(); vectors += 1
        records[source['id']] = record; fence.commit(source['id'])
    file = RUNS / f'prime/nan-prime20-{backend}-model/traces.jsonl'; episodes = rows(file) if file.exists() else []
    assert len(episodes) <= 96; outcomes = {}; bound = set(); trimmed = 0
    for ordinal, (source, episode) in enumerate(zip(public, episodes)):
        expected = dict(id=source['id'], prompt=source['prompt'], source_root=source['root'], arm=source['arm'], world=source['world'], private_answer=labels[source['id']], source_manifest_sha256=COMP14_SEAL)
        assert all(episode['task']['data'][k] == v for k, v in expected.items())
        assert episode['task']['data']['idx'] == ordinal
        for trace in episode['traces']:
            assert trace['verifiers']['commit'] == SDK_COMMIT and trace['agent']['runtime']['type'] == 'subprocess' and trace['agent']['config']['harness']['id'] == 'null'
        if not successful_episode(episode):
            continue
        trace = episode['traces'][0]; assert all(trace['task']['data'][k] == v for k, v in expected.items())
        assert len(trace['calls']) == trace['num_turns'] == 1 and len(trace['nodes']) == 2
        assert trace['nodes'][0]['message'] == dict(role='user', content=source['prompt'])
        assert source['id'] in records and source['id'] not in bound; record = records[source['id']]; bound.add(source['id'])
        assert raw_reply_bound(record, trace) and trace['calls'][0]['finish_reason'] == record['finish_reason']
        assert trace['num_input_tokens'] == record['usage']['prompt_tokens'] and trace['num_output_tokens'] == 1
        parsed = answer(record['text']); correct = parsed == labels[source['id']]
        assert trace['reward'] == float(correct) and trace['info']['format_valid'] == (parsed is not None)
        outcomes[source['id']] = dict(correct=correct, valid=parsed is not None)
        trimmed += int(record['text'] != trace['last_reply'])
    groups = []
    for arm in ('focused', 'copy', 'focused_erased'):
        sources = [r for r in public if r['arm'] == arm]; done = [r for r in sources if r['id'] in outcomes]
        paired = defaultdict(list)
        for source in done:
            paired[source['root']].append((source['world'], outcomes[source['id']]['correct']))
        complete = [v for v in paired.values() if len(v) == 2 and {r[0] for r in v} == {'A', 'B'}]
        correct = sum(outcomes[r['id']]['correct'] for r in done)
        groups.append(dict(arm=arm, registered_contexts=32, graded_contexts=len(done), missing_graded_contexts=32-len(done),
            correct=correct, format_valid=sum(outcomes[r['id']]['valid'] for r in done), registered_source_roots=16,
            graded_source_roots=len(paired), complete_world_pairs=len(complete), both_world_roots_correct=sum(all(v[1] for v in pair) for pair in complete),
            whole_registered_correctness_lower_bound=correct/32, whole_registered_correctness_upper_bound=(correct+32-len(done))/32))
    if terminal['status'] == 'completed':
        assert terminal['target_unchanged'] and terminal['target_parameter_sha256_before'] == terminal['target_parameter_sha256_after']
    status = 'completed' if terminal['status'] == 'completed' and runner_events[0].get('exit_code') == 0 and len(outcomes) == count == 96 else 'terminal_with_missing_coverage'
    write(RUNS / 'audits' / (backend + '.json'), dict(status='passed', backend=backend, execution_status=status, registered_contexts=96,
        neural_requests=count, successful_official_episodes=len(outcomes), official_episodes_recorded=len(episodes), missing_graded_contexts=96-len(outcomes),
        failed_official_episodes=len(episodes)-len(outcomes), orphan_or_unbound_neural_replies=count-len(bound),
        matched_native_prefixes=count, physical_prefix_vectors_byte_exact=vectors, SDK_outer_whitespace_trimmed=trimmed, groups=groups,
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), worker_receipt_sha256=sha(base / 'receipt.json'),
        official_trace_sha256=sha(file) if file.exists() else None, BPD_donor_transfer_attempted=False, sealed_test_accessed=False))
    print(dict(backend=backend, status=status, neural_requests=count, successful_episodes=len(outcomes)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True); p.add_argument('--snapshot', required=True); a = p.parse_args()
    run(a.backend, a.snapshot)
