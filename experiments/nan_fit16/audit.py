"""Independent terminal checkpoint/controller/grade audit; no neural replay."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import re
import sys
from .common import ROOT, RUNS, read, rows, sha, verify, write


def summarize(records, public, gold, passes, modes, tokenizer, baseline=None):
    from experiments.nan_comp14.audit import answer
    from mechinterp.nanbeige_bpd_gym.audit_fit import check_phase
    source = {r['id']: r for r in public}; seen = set(); groups = defaultdict(list)
    base = {r['id']: r for r in baseline or []}
    for r in records:
        key, mode = r['id'], r.get('mode', 'native')
        assert key in source and mode in modes and (key, mode) not in seen; seen.add((key, mode))
        row = source[key]; assert r['root'] == row['root'] and r['split'] == row['split']
        assert len(r['phases']) == passes
        for t, phase in enumerate(r['phases']):
            check_phase(phase, t, tokenizer)
            if mode != 'native':
                for name in ('full_vocabulary_KL', 'actual_last_token_module_write_relative_l2',
                    'local_intervention_l2_at_actual_reached_input', 'local_intervention_relative_l2_at_actual_reached_input', 'reached_input_relative_l2'):
                    assert math.isfinite(phase[name]) and phase[name] >= (-1e-6 if name == 'full_vocabulary_KL' else 0)
                if mode == 'all_one':
                    assert phase['actual_last_token_module_write_relative_l2'] == phase['local_intervention_l2_at_actual_reached_input'] == phase['reached_input_relative_l2'] == 0
                    assert abs(phase['full_vocabulary_KL']) <= 1e-6
                    for name in ('free_token_id', 'free_token_text', 'full_log_normalizer', 'free_token_logprob'):
                        assert phase[name] == base[key]['phases'][t][name]
                if mode == 'reached_state':
                    assert math.isfinite(phase['reached_vs_clean_mask_max_abs']) and phase['reached_vs_clean_mask_max_abs'] >= 0
                    if passes == 1:
                        assert phase['reached_input_relative_l2'] == phase['reached_vs_clean_mask_max_abs'] == 0
                else:
                    assert phase['reached_vs_clean_mask_max_abs'] is None
            groups[row['split'], row['arm'], mode, t + 1].append((row['root'], row['world'], answer(phase['free_token_text']) == gold[key], answer(phase['free_token_text']) is not None, phase))
    assert seen == {(key, mode) for key in source for mode in modes}
    result = []
    for (split, arm, mode, loop), values in sorted(groups.items()):
        pairs = defaultdict(list)
        for root, world, correct, _, _phase in values:
            pairs[root].append((world, correct))
        assert all(len(v) == 2 and {x[0] for x in v} == {'A', 'B'} for v in pairs.values())
        group = dict(split=split, arm=arm, mode=mode, loop=loop, contexts=len(values), source_roots=len(pairs),
            correct=sum(v[2] for v in values), format_valid=sum(v[3] for v in values),
            both_world_roots_correct=sum(all(x[1] for x in pair) for pair in pairs.values()))
        if mode != 'native':
            for name in ('full_vocabulary_KL', 'actual_last_token_module_write_relative_l2', 'local_intervention_l2_at_actual_reached_input',
                         'local_intervention_relative_l2_at_actual_reached_input', 'reached_input_relative_l2', 'reached_vs_clean_mask_max_abs'):
                byroot = defaultdict(list)
                for root, _world, _correct, _valid, phase in values:
                    if phase[name] is not None:
                        byroot[root].append(phase[name])
                means = [sum(v)/len(v) for v in byroot.values()]
                group[name] = dict(equal_root_mean=sum(means)/len(means) if means else None,
                    minimum_root_mean=min(means) if means else None, maximum_root_mean=max(means) if means else None,
                    defined_roots=len(means), undefined_contexts=sum(v[4][name] is None for v in values))
        result.append(group)
    return result


def check_steps(steps, public, cfg):
    cal = [r['id'] for r in public if r['split'] == 'calibration' and not r['copy_control']]
    assert len(steps) == cfg['fit_steps'] == len(cal) == 16
    assert len({r['root'] for r in public if r['id'] in cal}) == 8
    for i, step in enumerate(steps):
        assert step['step'] == i and step['source_id'] == cal[i] and step['mode'] == 'closed_loop'
        assert all(math.isfinite(v) for v in step.values() if isinstance(v, (int, float)))
        total = cfg['coeff_faith'] * step['faith_loss'] + step['min_loss'] + cfg['coeff_stoch'] * step['stoch_loss'] + cfg['coeff_adv'] * step['adv_loss']
        assert math.isclose(total, step['total_loss'], rel_tol=2e-6, abs_tol=1e-6)
    fields = ('total_loss', 'faith_loss', 'min_loss', 'stoch_loss', 'adv_loss')
    return dict(optimizer_steps=16, CAL_roots=8, CAL_worlds=16, DEV_or_control_training_uses=0, four_term_loss_accounting_checked=True,
        first_step_losses={k: steps[0][k] for k in fields}, final_step_losses={k: steps[-1][k] for k in fields})


def cell_audit(folder, backend, passes, cfg, public, gold, seal, baseline, tok):
    import numpy as np
    from mechinterp.nanbeige_bpd_gym.audit_fit import checkpoint
    from experiments.nan_fit07.pointwise import masks
    r = read(folder / 'receipt.json')
    if r['status'] != 'completed':
        return dict(status=r['status'], optimizer_steps_completed=r['optimizer_steps_completed'],
            failure_type=r.get('failure_type'), failure_message=r.get('failure_message'), receipt_sha256=sha(folder / 'receipt.json'))
    assert r['backend'] == backend and r['passes'] == passes and r['rank'] == 8 and r['module'] == 'down_proj'
    assert r['source_manifest_sha256'] == sha(ROOT / 'SOURCE_MANIFEST.json')
    assert r['optimizer_steps_completed'] == 16 and r['target_parameter_sha256_unchanged'] == baseline['target_parameter_sha256']
    assert r['initial_svd_file_sha256'] == seal['initialization'][backend]['file_sha256']
    assert r['effective_operator_sha256'] == seal['initialization'][backend]['effective_operator_sha256']
    assert r['native_all_one_full_head_bit_exact'] and r['native_all_one_boundary_states_bit_exact']
    assert math.isfinite(r['pre_optimizer_gradient_gate_loss']) and r['pre_optimizer_gradient_gate_loss'] > 0
    assert r['gauge_diagnostic']['stable_at_1e_minus5']
    assert not r['BPD_donor_transfer_attempted'] and not r['sealed_test_accessed']
    mem = r['saved_tensor_statistics']; assert mem['maximum_added_cpu_bytes'] == cfg['maximum_added_saved_tensor_CPU_bytes']
    assert 0 <= mem['peak_added_cpu_bytes'] <= mem['maximum_added_cpu_bytes']
    assert set(r['diagnostic_artifact_sha256']) == {'calibration_local.json', 'steps_local.json', 'initial_diagnostic_local.json', 'diagnostic_local.json'}
    for name, h in r['diagnostic_artifact_sha256'].items():
        assert sha(folder / name) == h
    training = check_steps(read(folder / 'steps_local.json'), public, cfg)
    banks = {}
    for label in ('initial_checkpoint', 'training_EMA_checkpoint', 'final_checkpoint'):
        description = r[label]; path = folder / description['file']; assert sha(path) == description['sha256']
        bank = checkpoint(path); banks[label] = bank
        assert bank['V'].shape == (10752, 8) and bank['U'].shape == (8, 3072)
        for field, shape in [('mean_comp_acts', (passes, 8)), ('mean_delta_out', (passes, 3072)), ('mean_write_rms', (passes,)), ('calibrated_passes', (passes,))]:
            assert bank[field].shape == shape
        assert bank['calibrated_passes'].all()
        if label != 'initial_checkpoint':
            for key in ('optimizer__factor_V__step', 'optimizer__factor_U__step', 'optimizer__ci_proj_in.weight__step', 'optimizer__ci_out_proj.weight__step'):
                assert bank[key].item() == 16
    initial, final = banks['initial_checkpoint'], banks['final_checkpoint']
    svdfile = Path('runs/nan_gym05') / backend / 'linear_baselines/layer21_down_proj.npz'
    assert sha(svdfile) == seal['initialization'][backend]['file_sha256']; svd = checkpoint(svdfile)
    np.testing.assert_array_equal(initial['V'], svd['V8']); np.testing.assert_array_equal(initial['U'], svd['U8'])
    for keys, field in [(('V', 'U'), 'factor_update_l2'), ([k for k in initial if k.startswith('ci__')], 'controller_update_l2')]:
        delta = sum(float(np.sum((final[k].astype(np.float64) - initial[k])**2)) for k in keys)**.5
        assert delta > 0 and math.isclose(delta, r[field], rel_tol=2e-5, abs_tol=1e-8)
    replayfile = folder / 'CI_replay_local.npz'; assert sha(replayfile) == r['CI_replay_sha256']; replay = checkpoint(replayfile)
    np.testing.assert_allclose(masks(replay['X'], final, r['final_checkpoint']['CI_config']), replay['masks'], rtol=2e-5, atol=1e-5)
    calibration = read(folder / 'calibration_local.json'); cal = [row['id'] for row in public if row['split'] == 'calibration' and not row['copy_control']]
    assert [row['id'] for row in calibration] == cal and len(calibration) == r['calibration_unit_count'] == 16
    for t in range(passes):
        values = [row['same_pass_tolerances'][str(t)] for row in calibration]
        assert all(math.isfinite(v) and v >= cfg['tolerance_epsilon'] for v in values)
        assert math.isclose(r['training_tolerances'][str(t)], float(np.mean(values)) if passes == 2 else 1., rel_tol=1e-12, abs_tol=1e-12)
    contract = r['training_contract']; assert contract['n_passes'] == passes and contract['n_steps'] == 16
    assert contract['c_per_module'] == {cfg['site']: 8} and contract['ci']['n_blocks'] == 0
    assert contract['adversary']['inner_steps'] == 2 and contract['coeff_faith'] == 1e5 and contract['coeff_stoch'] == contract['coeff_adv'] == .5
    assert contract['ablation']['mode'] == ('zero' if passes == 1 else 'pass_mean') and contract['ablation']['sphere_rescale'] == (passes == 2)
    for name in ('initial_effective_operator_closure', 'final_effective_operator_closure'):
        assert r[name]['relative_l2'] <= cfg['effective_operator_closure_relative_l2_tolerance']
    main = [row for row in public if not row['copy_control']]
    initial_groups = summarize(read(folder / 'initial_diagnostic_local.json')['records'], main, gold, passes, ['component_only'], tok)
    diagnostic = read(folder / 'diagnostic_local.json')
    static = np.asarray(diagnostic['static_CAL_mask']); assert static.shape == (passes, 8) and np.isfinite(static).all() and static.min() >= 0 and static.max() <= 1
    final_groups = summarize(diagnostic['records'], main, gold, passes, cfg['diagnostic_modes'], tok, baseline['records'])
    return dict(status='passed', passes=passes, receipt_sha256=sha(folder / 'receipt.json'), checkpoint_sha256=r['final_checkpoint']['sha256'],
        training=training, factor_update_l2=r['factor_update_l2'], controller_update_l2=r['controller_update_l2'],
        initial_groups=initial_groups, final_groups=final_groups, saved_tensor_statistics=mem,
        CUDA_peak_allocated_bytes=r['peak_cuda_allocated_bytes'], CUDA_peak_reserved_bytes=r['peak_cuda_reserved_bytes'],
        gauge_diagnostic=r['gauge_diagnostic'], training_tolerances=r['training_tolerances'])


def run(backend, snapshot, compactification_root):
    from experiments.nan_ranksite13.resources import below_normal
    below_normal(); seal = verify(); cfg = read(ROOT / 'config.json'); base = RUNS / backend
    assert any(json['event'] == 'worker_receipt_closed' for json in rows(base / 'events.jsonl'))
    terminal = read(base / 'receipt.json'); public = rows(ROOT / 'fixtures/public.jsonl')
    host = {r['id']: r for r in rows(ROOT / 'fixtures/host_labels.jsonl')}; gold = {k: r['answer'] for k, r in host.items()}
    sys.path.insert(0, str(Path(compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok = load_tokenizer(Path(snapshot), 166100).tok
    for row in public:
        if row['arm'] == 'focused':
            query = row['prompt'].split('\n\n')[-1]; graph = dict(re.findall(r'([A-P])->([A-P])', query))
            start = re.search(r'\nStart: ([A-P])\n', query).group(1)
            assert len(graph) == len(set(graph.values())) == 16 and graph[start] == gold[row['id']]
        elif row['arm'] == 'copy':
            assert row['prompt'].endswith('\n' + gold[row['id']])
        else:
            assert 'Edges: [not provided]' in row['prompt'].split('\n\n')[-1]
    baselinefile = base / 'baseline.json'; baseline_groups = []; results = []
    if baselinefile.exists():
        baseline = read(baselinefile); assert baseline['contexts'] == len(baseline['records']) == 144 and not baseline['host_gold_parsed']
        assert [r['id'] for r in baseline['records']] == [r['id'] for r in public]
        assert len(baseline['full_head_parity']) == 2 and all(r['vocabulary_logits_bit_exact'] and r['all_scorer_fields_exact'] and r['full_vocabulary_logits_checked'] == 166144 for r in baseline['full_head_parity'])
        for record, row in zip(baseline['records'], public):
            upstream = Path('runs/nan_comp14') / backend / f"case_{row['upstream_ordinal']:04d}.json"
            assert sha(upstream) == record['upstream_case_sha256']; old = read(upstream)
            assert record['initial_prompt_ids_sha256'] == old['initial_prompt_ids_sha256']
            for t, p in enumerate(record['phases']):
                score = old['initial_phases'][t]['score']; assert p['free_token_id'] == score['free_top_token_id']
                assert abs(p['full_log_normalizer'] - score['full_log_normalizer']) <= 1e-9
        baseline_groups = summarize(baseline['records'], public, gold, 2, ['native'], tok)
        for passes in cfg['fit_passes']:
            folder = base / f't{passes}_layer21_down_proj_rank8'
            result = cell_audit(folder, backend, passes, cfg, public, gold, seal, baseline, tok) if (folder / 'receipt.json').exists() else dict(status='unrun_or_missing_terminal_cell')
            results.append(dict(result, passes=passes))
    if terminal['status'] == 'completed':
        assert len(results) == 2 and all(r['status'] == 'passed' for r in results)
    write(RUNS / 'audits' / (backend + '.json'), dict(status='passed', backend=backend, worker_status=terminal['status'], cells=results,
        baseline_groups=baseline_groups, source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'), receipt_sha256=sha(base / 'receipt.json'),
        limitation='Archive/checkpoint/controller replay and first-token grade checks; no independent full-model backward or CAL-mean replay. No SAE or donor claim.',
        sealed_test_accessed=False))
    print(dict(backend=backend, worker_status=terminal['status'], cells=[(r['passes'], r['status']) for r in results]), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--backend', choices=('nf4', 'bf16'), required=True)
    p.add_argument('--snapshot', required=True); p.add_argument('--compactification-root', required=True); a = p.parse_args()
    run(a.backend, a.snapshot, a.compactification_root)
