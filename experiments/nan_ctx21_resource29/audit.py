"""Preserve the original scientific checks; separately attest the native stop."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT, RUNS, OLD, OLD_RUNS, read, rows, sha, write, select_prefix, no_torch
from experiments.nan_ctx21.audit import FIELDS, grade
from experiments.nan_ctx21.source_audit import parse


def run(snapshot, guard):
    cfg = read(OLD/'config.json'); backend = 'bf16'; base = OLD_RUNS/backend
    registration = read(ROOT/'registration.json')
    count = select_prefix(rows(base/'events.jsonl'), rows(OLD_RUNS/'queue_events.jsonl'),
                          read('experiments/nan_ram28/RESOURCE_OUTCOME.json'), (base/'receipt.json').exists())
    public = rows(OLD/'fixtures/public.jsonl'); gold = {r['id']:r for r in rows(OLD/'fixtures/host_labels.jsonl')}
    tokens = read(OLD/'fixtures/tokenizer_receipt.json'); graded = []
    import numpy as np
    import transformers
    import tokenizers
    from transformers import PreTrainedTokenizerFast
    from experiments.nan_gym05.common import prompt_ids
    # The original audit uses only .tok. Avoid its sibling wrapper's module-level Torch import.
    tok = SimpleNamespace(tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(snapshot)/'tokenizer.json')))
    no_torch()
    for name, h in tokens['tokenizer_files_sha256'].items():
        assert sha(Path(snapshot)/name) == h
    widths = dict(residual_after_layer=3072, down_proj_input=10752, down_proj_output=3072, o_proj_input=6144, o_proj_output=3072)
    expected_sites = {(t,l,s) for t in (1,2) for l in cfg['layers_zero_based'] for s in widths}
    manifest = read(base/'manifest.json'); assert manifest['backend']==backend and not manifest['host_gold_opened']
    assert manifest['source_manifest_sha256'] == sha(OLD/'SOURCE_MANIFEST.json') and manifest['CPU_threads']==1
    assert manifest['decoder_GPU_layers'] == 4
    assert manifest['embedding_device']=='cpu' and manifest['head_device']=='cuda:0' and manifest['head_dtype']=='torch.bfloat16'
    assert manifest['target_parameter_sha256'] == registration['BF16_target_parameter_sha256_before']
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
    no_torch(); guard.check()
    result = dict(status='passed_closed_prefix_audit', backend=backend, records=graded,
        original_terminal_audit='failed_missing_normal_receipt', worker_status='externally_stopped_for_host_RAM',
        contexts_checked=count, registered_contexts=624, missing_contexts=624-count, physical_vectors_checked=40*count,
        orphan_case_artifacts_not_scored=orphans, source_manifest_sha256=sha(OLD/'SOURCE_MANIFEST.json'),
        resource_audit_source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        stop_attestation_sha256=sha('experiments/nan_ram28/RESOURCE_OUTCOME.json'),
        target_parameter_sha256_before=manifest['target_parameter_sha256'], target_parameter_sha256_after=None,
        original_receipt_synthesized=False, original_gate_changed=False, pretrained_model_loaded=False,
        packages=dict(numpy=np.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__),
        sealed_test_accessed=False)
    write(RUNS/'bf16_closed_prefix_audit.json', result)
    return result
