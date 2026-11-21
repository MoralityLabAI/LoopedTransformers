"""Independent anchored grammar, tokenizer, graph and archived-state checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from .common import ROOT, RUNS, events, read, rows, sha, verify, write_new


def answer(text):
    if not isinstance(text, str) or len(text) > 64:
        return None
    text = text.strip()
    if text.endswith('.'):
        text = text[:-1]
    for left, right in [('**', '**'), ('*', '*'), ('`', '`'), ('[', ']')]:
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            text = text[len(left):-len(right)]
            break
    return text if len(text) == 1 and text in 'ABCDEFGHIJKLMNOP' else None


def run(backend, snapshot, compactification_root, runs=RUNS):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ.pop('TRANSFORMERS_CACHE', None)
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json')
    for name, expected in seal['host_files_sha256'].items():
        assert sha(ROOT / name) == expected
    import numpy as np
    sys.path.insert(0, str(Path(compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok = load_tokenizer(Path(snapshot), 166100).tok
    base = Path(runs) / backend
    assert any(v['event'] == 'worker_receipt_closed' for v in events(base / 'events.jsonl'))
    terminal = read(base / 'receipt.json')
    assert terminal['source_manifest_sha256'] == sha(ROOT / 'SOURCE_MANIFEST.json')
    source = rows(ROOT / 'fixtures/public.jsonl'); gold = {r['id']: r for r in rows(ROOT / 'fixtures/host_labels.jsonl')}
    tokens = read(ROOT / 'fixtures/tokenizer_receipt.json'); graded = []
    widths = dict(residual_after_layer=3072, down_proj_input=10752, down_proj_output=3072, o_proj_input=6144, o_proj_output=3072)
    expected_sites = {(t, l, s) for t in (1, 2) for l in cfg['layers_zero_based'] for s in widths}
    count = terminal['completed_contexts']; assert 0 <= count <= len(source) == 240
    for ordinal, public in enumerate(source[:count]):
        record = read(base / f'case_{ordinal:04d}.json'); host = gold[public['id']]
        assert record['status'] == 'completed'
        assert all(record[k] == public[k] for k in ('id', 'root', 'split', 'world', 'arm'))
        if public['arm'] in cfg['templates']:
            last = public['prompt'].split('\n\n')[-1]
            graph = dict(re.findall(r'([A-P])->([A-P])', last))
            query = re.search(r'\nStart: ([A-P])\n', last).group(1)
            assert len(graph) == len(set(graph.values())) == 16 and graph[query] == host['answer']
        elif public['arm'] == 'copy':
            assert public['prompt'].endswith('\n' + host['answer'])
        else:
            assert 'Edges: [not provided]' in public['prompt'].split('\n\n')[-1]
        initial = record['initial_prompt_ids']; bound = tokens['contexts'][public['id']]
        assert len(initial) == bound['tokens'] and hashlib.sha256(json.dumps(initial).encode()).hexdigest() == bound['ids_sha256'] == record['initial_prompt_ids_sha256']
        generated = record['generated_ids']; steps = record['steps']
        assert 1 <= len(generated) == len(steps) <= cfg['maximum_generated_tokens']
        for i, step in enumerate(steps):
            assert step['step'] == i and step['history_tokens'] == len(initial) + i
            assert step['generated_token_id'] == generated[i] == step['score']['free_top_token_id']
            assert 0 <= generated[i] < 166144
            assert step['score']['vocabulary_size'] == 166144
            assert math.isfinite(step['score']['full_log_normalizer']) and math.isfinite(step['score']['free_top_token_logit'])
            assert tok.decode(generated[:i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False) == step['decoded_prefix']
            if i < len(steps) - 1:
                assert answer(step['decoded_prefix']) is None and generated[i] != tok.eos_token_id
        assert record['bounded_reply'] == steps[-1]['decoded_prefix']
        value = answer(record['bounded_reply']); assert value == record['parsed_label']
        if value is None and len(generated) < cfg['maximum_generated_tokens']:
            assert generated[-1] == tok.eos_token_id or len(record['bounded_reply']) > cfg['maximum_reply_characters']
        assert len(record['initial_phases']) == 2
        first = []
        for phase, item in enumerate(record['initial_phases'], 1):
            assert item['loop'] == phase
            text = tok.decode([item['score']['free_top_token_id']], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert text == item['token_text']; first.append(answer(text))
        assert generated[0] == record['initial_phases'][1]['score']['free_top_token_id']
        archive = base / record['state_file']; assert sha(archive) == record['state_file_sha256']
        sites = set()
        with np.load(archive, allow_pickle=False) as z:
            assert len(z.files) == len(record['state_index']) == 40
            assert set(z.files) == {v['array_key'] for v in record['state_index']}
            for item in record['state_index']:
                site = (item['loop'], item['layer_zero_based'], item['site']); assert site not in sites; sites.add(site)
                array = z[item['array_key']]
                assert array.shape == (widths[item['site']],) and array.dtype == np.float32 and np.isfinite(array).all()
        assert sites == expected_sites
        graded.append({k: public[k] for k in ('id', 'root', 'split', 'world', 'arm')} | dict(
            first_token_correct=[int(v == host['answer']) for v in first], first_token_valid=[int(v is not None) for v in first],
            generated_ids_sha256=hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
            bounded_correct=int(value == host['answer']), bounded_valid=int(value is not None), generated_tokens=len(generated),
            case_sha256=sha(base / f'case_{ordinal:04d}.json'), state_sha256=sha(archive)))
    if terminal['status'] == 'completed':
        assert count == len(source)
    else:
        assert terminal['status'] == 'failed'
    result = dict(status='passed', backend=backend, worker_status=terminal['status'],
        source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), receipt_sha256=sha(base / 'receipt.json'),
        contexts_checked=count, registered_contexts=240, missing_contexts=240 - count,
        physical_vectors_checked=40 * count, records=graded, pretrained_model_loaded=False,
        scope='Independent grammar, pinned-tokenizer decoding, graph oracle, prefix/score-ID bindings, archive hashes/dimensions/finite coverage; not a second model forward replay.',
        sealed_test_accessed=False)
    write_new(Path(runs) / 'audits' / (backend + '.json'), result)
    return {k: result[k] for k in ('status', 'backend', 'contexts_checked', 'physical_vectors_checked', 'missing_contexts')}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True)
    p.add_argument('--runs', default=str(RUNS)); a = p.parse_args()
    print(run(a.backend, a.snapshot, a.compactification_root, Path(a.runs)), flush=True)


if __name__ == '__main__':
    main()
