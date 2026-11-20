"""Readable projection of committed rank/site audits; no model or raw-run reads."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

INPUT = 'reports/nanbeige_bpd_gym/nan_ranksite13_progress.json'
OUTPUT = Path('reports/nanbeige_bpd_gym')
SEAL = '27b3e5e636abcdf9217f88a5f19999c46b56d9723c76e6f5263c781e93239ddc'
AMENDMENT = 'e93a1df940cc8c263caff6a5c7dfa3acf7395d9a20e01cdcb07017753a129a7a'
EXPECTED = [('down_proj', 16, 1), ('down_proj', 16, 2), ('o_proj', 8, 1),
            ('o_proj', 8, 2), ('o_proj', 16, 1), ('o_proj', 16, 2)]
MODES = ('all_one', 'component_only', 'static_CAL_mean', 'clean_informed', 'reached_state')


def finite(value):
    assert isinstance(value, (int, float)) and math.isfinite(value)
    return value


def project(data):
    assert data['source_manifest_sha256'] == SEAL
    assert not data['sealed_test_accessed'] and not data['BPD_donor_transfer_attempted']
    assert data['planned_formal_comparisons'] == data['actual_formal_comparisons'] == 0
    assert data['n_CAL_roots'] == data['n_DEV_roots'] == 8
    assert [(c['module'], c['rank'], c['fit_passes']) for c in data['cells']] == EXPECTED
    cells, correctness = [], []
    for cell in data['cells']:
        module, rank, passes = cell['module'], cell['rank'], cell['fit_passes']
        assert cell['condition'] == f't{passes}_layer21_{module}_rank{rank}'
        label = f'{module} r{rank} T{passes}'
        fit, physical, cached = (cell[k] for k in ('fit_independent_audit', 'physical_write_SAE_probe', 'cached_suite_probe'))
        row = dict(condition=cell['condition'], label=label, status=cell['status'],
                   failure_type=cell.get('failure_type'), fit_audited=fit is not None,
                   audited_steps=None, factor_update_l2=None, controller_update_l2=None,
                   saved_CPU_peak_GiB=None, finite_writes=None, finite_feature_groups=None,
                   derivative_groups=None, cached_vectors=None, cached_groups=None)
        if fit:
            assert fit['status'] == 'passed' and fit['source_manifest_sha256'] == SEAL
            train = fit['training']
            assert train['optimizer_steps'] == 8 and train['independent_CAL_roots_used'] == 8
            assert train['DEV_or_copy_training_uses'] == 0
            assert fit['gauge_diagnostic']['stable_at_1e_minus5']
            row.update(audited_steps=8, factor_update_l2=finite(fit['factor_update_l2']),
                       controller_update_l2=finite(fit['controller_update_l2']),
                       saved_CPU_peak_GiB=finite(fit['saved_tensor_statistics']['peak_added_cpu_bytes']) / 2**30)
            assert 0 < row['saved_CPU_peak_GiB'] <= 2
            groups = fit['final_groups']
            actual = {(g['split'], g['loop'], g['mode']) for g in groups}
            expected = {(split, loop, mode) for split in ('calibration', 'development')
                        for loop in range(1, passes + 1) for mode in MODES}
            assert len(groups) == len(actual) and actual == expected
            for split in ('calibration', 'development'):
                for loop in range(1, passes + 1):
                    record = dict(condition=cell['condition'], label=label, split=split, loop=loop,
                                  roots=8, contexts=8, correct={})
                    for mode in MODES:
                        g = next(g for g in groups if (g['split'], g['loop'], g['mode']) == (split, loop, mode))
                        assert g['contexts'] == g['independent_source_roots'] == 8
                        assert 0 <= g['correct'] <= 8
                        record['correct'][mode] = g['correct']
                    correctness.append(record)
        if physical:
            assert fit and physical['status'] == 'passed' and physical['contexts'] == 16
            assert physical['signed_finite_writes_verified'] == 16 * 2 * 3 * rank * 3 * 2
            assert len(physical['geometry_groups']) == 72 and len(physical['derivative_diagnostics']) == 36
            assert not physical['causal_claim'] and not physical['derivative_finite_equivalence_asserted']
            row.update(finite_writes=physical['signed_finite_writes_verified'], finite_feature_groups=72, derivative_groups=36)
        if cached:
            assert fit and cached['status'] == 'passed' and cached['source_manifest_sha256'] == SEAL
            assert (cached['module'], cached['rank'], cached['fit_passes']) == (module, rank, passes)
            assert cached['checkpoint_sha256'] == fit['checkpoint_sha256']
            assert cached['fit_receipt_sha256'] == fit['fit_receipt_sha256']
            assert cached['analysis_amendment']['registration_sha256'] == AMENDMENT
            assert len(cached['groups']) == 28 and sum(g['vectors'] for g in cached['groups']) == 644
            assert len(cached['geometry']) == (6 if module == 'down_proj' else 0)
            row.update(cached_vectors=644, cached_groups=28)
        if cell['status'] == 'completed_and_audited':
            assert fit and physical and cached and cell['optimizer_steps_completed'] == 8
            assert cell['worker_receipt_sha256']
        cells.append(row)
    pairs = []
    for pair in data['learned_VPD_BPD_pairs']:
        module, rank = pair['module'], pair['rank']
        assert (module, rank) in {('down_proj', 16), ('o_proj', 8), ('o_proj', 16)}
        assert all(next(c for c in cells if c['label'] == f'{module} r{rank} T{t}')['fit_audited'] for t in (1, 2))
        values = pair['output_subspace_principal_cosines']
        assert len(values) == rank and all(-1e-8 <= finite(v) <= 1 + 1e-6 for v in values)
        pairs.append(dict(module=module, rank=rank,
                          relative_operator_l2=finite(pair['physical_component_operator_relative_l2']),
                          operator_cosine=finite(pair['physical_component_operator_cosine']),
                          principal_cosine_min=min(values), principal_cosine_max=max(values)))
    assert len({(p['module'], p['rank']) for p in pairs}) == len(pairs)
    return dict(cells=cells, correctness=correctness, learned_pairs=pairs,
                completed_conditions=sum(c['status'] == 'completed_and_audited' for c in cells),
                registered_conditions=6, n_CAL_roots=8, n_DEV_roots=8,
                formal_comparisons=0, sealed_test_accessed=False)


def number(value):
    return '—' if value is None else f'{value:.6g}'


def render(data):
    lines = ['# Native BF16 learned rank/site results', '',
             f"{data['completed_conditions']}/6 conditions are fully completed and audited in this committed snapshot. "
             'All six remain listed. Blank metrics mean no corresponding closed audit, not a measured zero.', '',
             'The fits reuse eight exposed CAL and eight exposed DEV roots from FIT07 on one checkpoint. '
             'Training uses each CAL main root once; DEV/copy examples are not training inputs. '
             'NF4 remains excluded by its failed fixed fitting screen. Original algebraic gates remain failed.', '',
             '| Condition | Closed status | Audited steps | Factor update L2 | Controller update L2 | Added saved-tensor CPU peak GiB |',
             '|---|---|---:|---:|---:|---:|']
    for c in data['cells']:
        lines.append(f"| {c['label']} | {c['status']} | {number(c['audited_steps'])} | {number(c['factor_update_l2'])} | {number(c['controller_update_l2'])} | {number(c['saved_CPU_peak_GiB'])} |")
    lines += ['', 'Update magnitudes depend on factor parameterization. The CPU column measures added saved tensors under the unchanged 2-GiB cap; it is not whole-process RAM.', '']
    for split, title in (('development', 'DEV'), ('calibration', 'CAL')):
        lines += ['', f'## All {title} routing controls', '',
                  '| Condition | Loop | All-one | Component only | Static CAL mean | Clean-informed | Actual reached-state |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        selected = [r for r in data['correctness'] if r['split'] == split]
        for r in selected:
            values = ' | '.join(f"{r['correct'][m]}/8" for m in MODES)
            lines.append(f"| {r['label']} | {r['loop']} | {values} |")
        if not selected:
            lines += ['| No closed fit audit | — | — | — | — | — | — |']
    lines += ['', 'Each fraction uses eight main contexts from eight roots at that split. Invalid outputs stay in the denominator. '
              'The same roots recur across routes, loops, ranks and sites. Static controls retaining correctness limit behavioral specificity. '
              'T1/T2 differ in fit depth, ablation and adversary rules; differences do not isolate a BPD advantage.', '',
              '## Audited SAE cross-reference coverage', '',
              '| Condition | Signed finite writes | Finite feature groups | Derivative groups | Cached vectors | Cached groups |',
              '|---|---:|---:|---:|---:|---:|']
    for c in data['cells']:
        values = ' | '.join(number(c[k]) for k in ('finite_writes', 'finite_feature_groups', 'derivative_groups', 'cached_vectors', 'cached_groups'))
        lines.append(f"| {c['label']} | {values} |")
    lines += ['', 'Finite probes cover all 16 main fitting prefixes, both native loops, three bases, every component, '
              'requested relative doses .001/.01/.1 and both signs. Groups retain all 128 features, including rounded-away writes. '
              'o_proj responses use the actual residual/norm/MLP path; they have no identity-coordinate geometry cells. '
              'JVPs remain derivative surrogates, with all finite-write discrepancies preserved. '
              'The CPU audit checks archive/dose/coverage/native additions; native MLP/JVP outputs are source-bound worker measurements.', '',
              'The cached suite retains 644 vectors/89 older source groups, all 28 suite/split/copy/loop strata and all 128-by-rank correlations. '
              'These clean-input probes are observational. T1 application at loop2 is outside fit depth. '
              'Full feature arrays, source groups and derivative diagnostics remain in the linked aggregate; none are selected by outcome.', '',
              '## Registered physical operator comparisons', '',
              '| Module/rank | Relative V@U difference | V@U cosine | Output-subspace principal cosine range |',
              '|---|---:|---:|---|']
    for p in data['learned_pairs']:
        lines.append(f"| {p['module']} r{p['rank']} | {number(p['relative_operator_l2'])} | {number(p['operator_cosine'])} | {number(p['principal_cosine_min'])}–{number(p['principal_cosine_max'])} |")
    if not data['learned_pairs']:
        lines += ['| No two audited matching fits | — | — | — |']
    lines += ['', 'These compare physical module-output operators/subspaces, not individual factor columns or an unbridged o_proj-to-SAE identity. '
              'Shared SVD initialization and small updates can produce high alignment. No pooled accuracy, inferential comparison, '
              'equivalence, semantic mediation or donor transfer is claimed. Causal effect/sham floors and power remain uncalibrated. '
              'The original 168-case test stays sealed.', '',
              f"Source commit: `{data['source_commit']}`. Aggregate SHA-256: `{data['source_sha256']}`. "
              f"Producer SHA-256: `{data['producer_sha256']}`. Experiment seal: `{SEAL}`. HOST01 registration: `{AMENDMENT}`.", '',
              f"[Exact source aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/{data['source_commit']}/{INPUT}) · [Latest worker/audit status](NAN_RANKSITE13_PROGRESS.md) · "
              '[HOST01 amendment](../../experiments/nan_ranksite13_host01/PROTOCOL.md) · '
              '[Compact machine-readable summary](nan_ranksite13_results.json)', '',
              'This is a post hoc presentation of committed audited endpoints, prepared after the first three completed cells were exposed. '
              'It reads no raw runs or live progress files and performs zero model calls or new grading. '
              'Statuses describe closed artifacts, not current process liveness.', '']
    return '\n'.join(lines)


def main():
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    producer = Path(__file__).resolve().relative_to(Path.cwd().resolve()).as_posix()
    code = Path(__file__).read_bytes()
    assert subprocess.check_output(['git', 'show', 'HEAD:' + producer]) == code, 'Commit the producer before publishing output'
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', INPUT], text=True).strip()
    raw = subprocess.check_output(['git', 'show', commit + ':' + INPUT])
    assert hashlib.sha256(subprocess.check_output(['git', 'show', commit + ':experiments/nan_ranksite13/SOURCE_MANIFEST.json'])).hexdigest() == SEAL
    assert hashlib.sha256(subprocess.check_output(['git', 'show', commit + ':experiments/nan_ranksite13_host01/REGISTRATION.json'])).hexdigest() == AMENDMENT
    data = project(json.loads(raw))
    data.update(source_commit=commit, source_sha256=hashlib.sha256(raw).hexdigest(), producer_sha256=hashlib.sha256(code).hexdigest())
    (OUTPUT / 'NAN_RANKSITE13_RESULTS.md').write_text(render(data), encoding='utf-8', newline='\n')
    (OUTPUT / 'nan_ranksite13_results.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print({'completed_conditions': data['completed_conditions'], 'source_commit': commit, 'scope': 'committed aggregates only'})


if __name__ == '__main__':
    main()
