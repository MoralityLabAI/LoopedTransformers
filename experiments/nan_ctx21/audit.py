"""Independent terminal grades, tokenizer/score/state coverage; no neural replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from .common import ROOT, RUNS, read, rows, sha, verify, write
from .source_audit import parse

FIELDS = ('id', 'root', 'split', 'context_budget', 'position', 'world', 'arm', 'graph_records', 'target_graph_index')


def grade(text):
    # Independent anchored implementation of COMP14 grammar, not worker output.
    if not isinstance(text, str) or len(text) > 64:
        return None
    value = text.strip()
    if value.endswith('.'):
        value = value[:-1]
    for a, b in [('**','**'), ('*','*'), ('`','`'), ('[',']')]:
        if len(value) >= len(a)+len(b) and value.startswith(a) and value.endswith(b):
            value = value[len(a):-len(b)]; break
    return value if len(value) == 1 and value in 'ABCDEFGHIJKLMNOP' else None


def run(backend, snapshot, compactification_root):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT/'config.json'); base = RUNS/backend
    assert any(v['event']=='worker_terminal' and v['backend']==backend for v in rows(RUNS/'queue_events.jsonl'))
    journal = rows(base/'events.jsonl'); assert journal[-1]['event'] == 'worker_receipt_closed'
    terminal = read(base/'receipt.json'); assert terminal['source_manifest_sha256'] == sha(ROOT/'SOURCE_MANIFEST.json')
    assert terminal['backend'] == backend and terminal['registered_contexts'] == 624
    assert terminal['status'] in ('completed', 'capacity_stopped', 'budget_reached', 'failed')
    count = terminal['completed_contexts']; assert isinstance(count, int) and 0 <= count <= 624
    assert [e['ordinal'] for e in journal if e['event']=='case_closed'] == list(range(count))
    if terminal['status'] == 'completed':
        assert count == 624 and terminal['target_parameter_sha256_after'] == terminal['target_parameter_sha256_before'] and terminal['target_parameter_sha256_after']
        assert terminal['failure'] is None
    else:
        assert terminal['failure'] and terminal['failed_or_next_ordinal'] == count
    for name, h in seal['host_files_sha256'].items():
        assert sha(ROOT/name) == h
    public = rows(ROOT/'fixtures/public.jsonl'); gold = {r['id']:r for r in rows(ROOT/'fixtures/host_labels.jsonl')}
    tokens = read(ROOT/'fixtures/tokenizer_receipt.json'); graded = []
    import numpy as np
    sys.path.insert(0, str(Path(compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    from experiments.nan_gym05.common import prompt_ids
    tok = load_tokenizer(Path(snapshot), 166100)
    for name, h in tokens['tokenizer_files_sha256'].items():
        assert sha(Path(snapshot)/name) == h
    widths = dict(residual_after_layer=3072, down_proj_input=10752, down_proj_output=3072, o_proj_input=6144, o_proj_output=3072)
    expected_sites = {(t,l,s) for t in (1,2) for l in cfg['layers_zero_based'] for s in widths}
    if count:
        manifest = read(base/'manifest.json'); assert manifest['backend']==backend and not manifest['host_gold_opened']
        assert manifest['source_manifest_sha256'] == sha(ROOT/'SOURCE_MANIFEST.json') and manifest['CPU_threads']==1
        assert manifest['decoder_GPU_layers'] == (22 if backend=='nf4' else 4)
        assert manifest['embedding_device']=='cpu' and manifest['head_device']=='cuda:0' and manifest['head_dtype']=='torch.bfloat16'
        assert manifest['target_parameter_sha256'] == terminal['target_parameter_sha256_before']
        parity = read(base/'head_parity.json')['checks']; assert len(parity)==2
        assert all(p['vocabulary_logits_bit_exact'] and p['all_scorer_fields_exact'] and p['full_vocabulary_logits_checked']==166144 and not p['head_quantized'] for p in parity)
    expected_files = set()
    for ordinal, source in enumerate(public[:count]):
        casefile = base/f'case_{ordinal:04d}.json'; record = read(casefile); target = gold[source['id']]
        assert record['status']=='completed' and all(record[k]==source[k] for k in FIELDS)
        name, start, graphs, _ = parse(source)
        assert target['target_graph_id']==name and target['query']==start and target['target_edges'][start]==target['answer']
        if source['arm'] in ('main','copy'):
            assert graphs[name][start]==target['answer']
        else:
            assert set(graphs[name].values())=={'?'}
        ids = prompt_ids(tok, snapshot, source['prompt']); bound = tokens['contexts'][source['id']]
        assert ids == record['initial_prompt_ids'] and len(ids)==bound['tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == record['initial_prompt_ids_sha256'] == bound['ids_sha256']
        assert len(record['phases'])==2
        predictions = []; token_ids = []
        for t, phase in enumerate(record['phases'], 1):
            score = phase['score']; assert phase['loop']==t and score['vocabulary_size']==166144
            token = score['free_top_token_id']; assert isinstance(token,int) and 0 <= token < 166144
            assert math.isfinite(score['full_log_normalizer']) and math.isfinite(score['free_top_token_logit'])
            assert score['full_log_normalizer'] >= score['free_top_token_logit']-1e-10
            assert all(math.isfinite(v) for v in score['selected_logits'].values())
            assert all(v <= score['free_top_token_logit'] for v in score['selected_logits'].values())
            assert 0 <= score['grammar_mass'] <= 1+1e-8 and set(score['class_logprobs'])==set('INSR')
            assert math.isclose(sum(math.exp(v) for v in score['class_logprobs'].values()), score['grammar_mass'], rel_tol=1e-12, abs_tol=1e-14)
            text = tok.tok.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert text==phase['token_text']; predictions.append(grade(text)); token_ids.append(token)
        archive = base/record['state_file']; assert archive.name==f'case_{ordinal:04d}_states.npz' and sha(archive)==record['state_file_sha256']
        sites = set()
        with np.load(archive, allow_pickle=False) as arrays:
            assert len(arrays.files)==len(record['state_index'])==40
            assert set(arrays.files)=={v['array_key'] for v in record['state_index']}
            for item in record['state_index']:
                site = (item['loop'], item['layer_zero_based'], item['site']); assert site not in sites; sites.add(site)
                value = arrays[item['array_key']]
                assert value.shape==(widths[item['site']],) and value.dtype==np.float32 and np.isfinite(value).all()
        assert sites==expected_sites
        graded.append({k:source[k] for k in FIELDS} | dict(correct=[int(v==target['answer']) for v in predictions],
            format_valid=[int(v is not None) for v in predictions], token_ids=token_ids,
            case_sha256=sha(casefile), state_sha256=sha(archive)))
        expected_files.update((casefile.name, archive.name))
    orphans = sorted(p.name for p in base.glob('case_*') if p.name not in expected_files)
    result = dict(status='passed', backend=backend, worker_status=terminal['status'], records=graded,
        contexts_checked=count, registered_contexts=624, missing_contexts=624-count, physical_vectors_checked=40*count,
        orphan_case_artifacts_not_scored=orphans, source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'), receipt_sha256=sha(base/'receipt.json'),
        scope='Independent anchored grammar, textual graph oracle, exact pinned re-tokenization, greedy-score bindings, archive hashes/shapes/finite coverage; no neural replay or missing-record rescue.',
        pretrained_model_loaded=False, sealed_test_accessed=False)
    write(RUNS/'audits'/f'{backend}.json', result)
    print({k:result[k] for k in ('status','backend','worker_status','contexts_checked','missing_contexts','physical_vectors_checked')}, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4','bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True); a = p.parse_args()
    run(a.backend, a.snapshot, a.compactification_root)


if __name__ == '__main__':
    main()
