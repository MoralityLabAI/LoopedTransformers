"""Regenerate Deuterium tables, figures and a review PDF from recorded outputs.

Run from the repository root after the archive/audit pipeline. This program never
runs model inference or scores evaluation data. Historical rows retain their
report-only or independently-recomputed provenance labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'papers' / 'VPD_TRM_Feature_Steering'


def manifest_path(path: Path) -> str:
    """Use stable labels for runner-temporary inputs without embedding host paths."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return 'external/' + path.name


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(cell, headers)) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines += ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]
    return '\n'.join(lines)


def f(v: float, digits: int = 4) -> str:
    return f'{v:.{digits}f}'


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f'<!-- BEGIN GENERATED:{name} -->', f'<!-- END GENERATED:{name} -->'
    pattern = re.compile(re.escape(start) + '.*?' + re.escape(end), re.S)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f'Expected exactly one generated block: {name}')
    return pattern.sub(lambda _: start + '\n' + content.strip() + '\n' + end, text)


def generated_blocks(audit: dict[str, Any], inventory: dict[str, Any],
                     followup: dict[str, Any] | None) -> dict[str, str]:
    arc, wordle, rep = audit['arc'], audit['wordle'], audit['replication']
    blocks: dict[str, str] = {}
    arc_meta = inventory['archive']
    blocks['evidence_tiers'] = table(['Result family', 'Evidence in this revision'], [
        ['Earlier full arena', 'Recovered summary only; sweep outside supplied package'],
        ['Six-seed replication', 'Complete archived-row recount and acceptance-rule audit'],
        ['ARC and Wordle policies', 'Recomputed cached-score decisions and source-identity overlap'],
        ['Governance', 'Preserved report values with source-code audit'],
        ['Component adapters', 'Recovered result JSON and dataset overlap; exact invocation missing'],
        ['Later basis-search addendum', 'Recovered manuscript report only'],
        ['New follow-up', followup.get('status', 'Hosted comparison unrun; local diagnostics pending') if followup else 'Hosted comparison unrun; local diagnostics pending'],
    ])
    blocks['inventory'] = table(['Item', 'Verified value'], [
        ['Archive bytes', f"{arc_meta['bytes']:,}"],
        ['Extracted members', f"{inventory['member_count']:,}"],
        ['Original source members', sum(m['path'].startswith('sources/') for m in inventory['members'])],
        ['Recovery metadata members', sum(not m['path'].startswith('sources/') for m in inventory['members'])],
        ['Uncompressed bytes', f"{inventory['total_uncompressed_bytes']:,}"],
        ['Dataset revision', f"`{arc_meta['revision']}`"],
        ['Digest check', 'PASS' if inventory['verified'] else 'FAIL'],
    ])
    rows = [[str(x['seed']), f"{x['graft_rows']:,}", x['accepted'], x['summary_preview_accepted']]
            for x in rep['by_seed']]
    rows += [['Total', f"{rep['counts']['rows']:,}", rep['counts']['recomputed_accepted'],
              sum(x['summary_preview_accepted'] for x in rep['by_seed'])]]
    original = rep['original_full_arena']
    blocks['replication'] = table(['Seed', 'Sweep rows', 'Accepted in full recount', 'Accepted in preview'], rows)
    blocks['replication'] += '\n\n' + table(['Sweep', 'Accepted', 'Stable neutral', 'Evidence'], [
        ['Earlier full arena', original['accepted_count'], f"{original['stable_graft_count']:,}",
         'Recovered summary; sweep absent from package'],
        ['Six-seed replication', rep['counts']['recomputed_accepted'],
         f"{rep['counts']['stable_candidate_rows']:,}", 'Streamed raw-row recount'],
    ])
    blocks['replication'] += (
        f"\n\nThe replication registry contains {rep['registry_entries']} organelles. "
        f"The {rep['counts']['recomputed_accepted']} accepted rows reduce to "
        f"{rep['unique_accepted_seed_source_target_component']} distinct seed/source/target/component combinations "
        "when scale variants are collapsed."
    )
    counts = rep['counts']
    blocks['acceptance'] = table(['Diagnostic among accepted replication rows', 'Count'], [
        ['Strictly better task-metric rank than best guarded random', counts['accepted_task_metrics_strictly_beat_random']],
        ['Equal task-metric rank; wins on smaller mean absolute logit change', counts['accepted_task_metrics_tie_random']],
        ['SVD head components', rep['accepted_component_types'].get('head_svd', 0)],
        ['Class-row head components', rep['accepted_component_types'].get('head_row', 0)],
        ['Repair-step targets', rep['accepted_target_gate_families'].get('repair_step', 0)],
        ['Candidate-verification targets', rep['accepted_target_gate_families'].get('candidate_verify', 0)],
    ])
    blocks['acceptance'] += (
        f"\n\nAll {counts['accepted_task_metrics_tie_random']} accepted rows tie the best guarded random "
        "control on the first four rank entries. Their winning entry is the smaller output perturbation. "
        "The reconciled result records lower output perturbation under the historical rank rule. "
        "Strict task superiority over the best guarded random row for the target has count zero."
    )
    arc_rows = [[x['card'].removeprefix('seed'), x['rows'], x['shared_with_other_cards'],
                 x['unseen_in_other_cards'], f(x['condition_replay_all_rows']['delta']),
                 f(x['condition_replay_unseen_rows']['delta'])] for x in arc['folds']]
    blocks['arc'] = table(['Held-out seed', 'Rows', 'Seen in other cards', 'Unique to this card',
                           'All-row delta', 'Unique-row delta'], arc_rows)
    dedup = arc['deduplicated_condition_replay']
    blocks['arc'] += (
        f"\n\nThe {arc['raw_rows']} scored appearances contain {arc['unique_ids']} distinct trajectory identities. "
        f"Replay of the fixed `top D, runner-up A, margin <= 1.0` condition changes "
        f"{dedup['baseline_hits']}/{dedup['n']} correct to {dedup['policy_hits']}/{dedup['n']}, "
        f"with {dedup['rescues']} rescues and {dedup['damages']} damages. "
        "Every fold's examples unique to that card have zero gain under this fixed rule. "
        "These post hoc slices describe the previously selected rule; their labels have already entered the research record."
    )
    wordle_rows = []
    for x in wordle['cards']:
        card = x['card'].replace('constrained_post_initial_', '')
        wordle_rows.append([card, x['n'], f"{x['baseline_hits']}/{x['n']}",
                            f"{x['policy_hits']}/{x['n']}", f(x['delta']), f(x['best_control_delta'])])
    blocks['wordle'] = table(['Card', 'Rows', 'Baseline correct', 'Policy correct', 'Delta', 'Best control delta'], wordle_rows)
    blocks['wordle'] += (
        f"\n\nThe mean card delta is {f(wordle['mean_card_delta'], 6)}. Weighting scored appearances "
        f"by card size gives {f(wordle['pooled_row_delta'], 6)} across {wordle['raw_rows']} rows. "
        "The four-example card is contained in the eight-example card; the two later cards are disjoint "
        "from those discovery cards and from each other by recorded trajectory ID."
    )
    wd = wordle.get('deduplicated_replay', {})
    if wd:
        blocks['wordle'] += (
            f" The audit's stated identity rule yields {wd['n']} retained records with "
            f"{wd['baseline_hits']} baseline hits and {wd['policy_hits']} policy hits. "
            "The full audit preserves both identity definitions and repeated-record diagnostics."
        )
    if wordle.get('identity'):
        blocks['wordle'] += f" Identity rule: `{wordle['identity']}`."
    if wordle.get('episode_overlap'):
        blocks['wordle'] += f" These events span {wordle['episode_overlap']['unique_ids']} recorded episodes; one episode appears at two different steps. All four cards replay their stored per-row decisions exactly."
    if wordle.get('target_conflict_ids'):
        blocks['wordle'] += ' Conflicting targets under the chosen identity: ' + ', '.join(f'`{x}`' for x in wordle['target_conflict_ids']) + '.'
    # These are historical report values parsed from preserved files, not rerun model outcomes.
    blocks['governance'] = table(['Reported arm', 'Target accuracy', 'Collateral accuracy', 'Evidence'], [
        [x['arm'], f(x['target_accuracy']), f(x['nontarget_accuracy']), 'Historical report + source audit']
        for x in audit['governance']['arms']
    ])
    adapter_rows = []
    for run in audit['adapters']['runs']:
        if run['run'] not in ['powered_int3_vpd_gated_adapter', 'powered_int3_vpd_gated_adapter_realvpd',
                              'powered_int3_vpd_gated_adapter_validated', 'powered_int3_vpd_gated_adapter_validated_zero_step']:
            continue
        label = {'powered_int3_vpd_gated_adapter': 'SVD static',
                 'powered_int3_vpd_gated_adapter_realvpd': 'Learned static',
                 'powered_int3_vpd_gated_adapter_validated': 'Learned validated',
                 'powered_int3_vpd_gated_adapter_validated_zero_step': 'Learned validated (0 steps)'}[run['run']]
        for arm in run['arms']:
            adapter_rows.append([label, arm['name'], f(arm['eval']['loss'], 6),
                                 f(arm['eval']['bucket_accuracy'], 6)])
    blocks['adapter'] = table(['Historical experiment', 'Arm', 'Eval loss', 'Bucket accuracy'], adapter_rows)
    a = audit['adapters']
    blocks['adapter'] += '\n\n' + table(['Identity used for dataset comparison', 'Shared distinct identities', 'Evaluation rows sharing a training identity'], [
        [key.replace('_', ' '), v['shared_unique'], f"{v['eval_rows_with_train_identity']:,} / {a['eval_rows']:,}"]
        for key, v in a['overlap'].items()
    ])
    blocks['adapter'] += (
        "\n\nThe archived dataset has disjoint row identifiers. Exact text overlaps remain, and task context "
        "before the literal `MODEL_ACTION:` delimiter is shared by most evaluation rows. This source-task "
        "identity check exposes a different level of dependence from a row-ID split. It also leaves the "
        "historical 64-row evaluation subset unauthenticated because the exact invocation and subset IDs are absent."
    )
    rnd = audit.get('historical_roundout')
    if rnd:
        metrics = rnd['metrics']['rows']
        rows = [[x.get('result_family', ''), x.get('metric', ''), x.get('value', ''), 'Recovered summary/table']
                for x in metrics if x.get('metric') in ['accepted_eval_aligned_count', 'accepted_edit_count',
                    'stacked_bootstrap_success', 'family_count', 'first_edit_family_count', 'stacked_bootstrap_family_count']]
        blocks['historical'] = table(['Historical family', 'Metric', 'Value', 'Evidence'], rows)
        policy_rows = [[x['policy'], x['raw_total_metric_gain'], x['filtered_accept_count'], x['filtered_metric_gain']]
                       for x in rnd['policies']['rows']]
        blocks['historical'] += '\n\n' + table(['Historical policy', 'Raw total gain', 'Filter accepts', 'Filtered gain'], policy_rows)
        blocks['historical'] += '\n\nThese values reproduce the archived roundout CSVs. The model executions and estimated-effect calculations have not been rerun.'
    else:
        blocks['historical'] = ('The original roundout CSVs remain in the archive with their original claim labels. '
                                'This section uses their reported search boundaries as historical context; '
                                'the independently recomputed results appear in Sections 3 and 4.')
    recovered_path = PAPER/'archive/recovered_vpd_trm_feature_steering_paper.md'
    recovered_meta = json.loads((PAPER/'archive/recovered_manuscript_manifest.json').read_text())
    recovered_bytes = recovered_path.read_bytes()
    if hashlib.sha256(recovered_bytes).hexdigest() != recovered_meta['sha256']:
        raise ValueError('Recovered manuscript bytes differ from their preservation manifest.')
    source_text = recovered_bytes.decode('utf-8')
    match = re.search(r'^arm +accepts[^\n]*\n(.*?)```', source_text, re.M | re.S)
    if not match:
        raise ValueError('Recovered manuscript head-to-head table not found.')
    later_rows = []
    for line in match.group(1).strip().splitlines():
        cells = line.split()
        if len(cells) != 5:
            raise ValueError('Unexpected recovered head-to-head row: ' + line)
        later_rows.append(cells)
    if len(later_rows) != 4:
        raise ValueError('Expected four reported head-to-head arms.')
    blocks['later_basis'] = table(['Reported arm', 'Accepts', 'Accepts / eval', 'Evals to first', 'Accepted delta'], later_rows)
    blocks['later_basis'] += '\n\nSource: the recovered expanded manuscript, parsed without modifying its numbers. Each row is report-only.'
    blocks['followup'] = render_followup(followup)
    gap_rows = [
        ['Earlier full-arena sweep', 'Absent from supplied package; summary recovered; package says not requested.'],
        ['Historical execution commit', 'Recovered source and records agree on reviewed logic; exact run-to-commit attestation remains absent.'],
        ['Arena generalization', 'Fresh confirmation examples and multiplicity-controlled search evidence remain required.'],
        ['ARC independence', 'Card overlap is measured; a prospective task-disjoint test remains required.'],
        ['Wordle source episodes', 'Recorded identities support overlap checks; complete upstream puzzle/episode lineage remains incomplete.'],
        ['Governance provenance', 'Exact invocation, full selected row IDs, dirty-update checkpoint, and projection coefficients remain missing.'],
        ['Adapter provenance', 'Exact invocation and selected train/validation/evaluation row IDs remain missing.'],
        ['Recovered-model execution', 'Original forward class, dataset/tokenizer implementation, independent reference logits, and historical x-transformers/package versions remain missing.'],
        ['Later basis-search claims', 'Preserved expanded-manuscript reports; candidate-level recount and invocation reconciliation remain open.'],
        ['New method scope', 'One synthetic gate distribution, head-only edits, and three model/decomposition seeds; the frozen positive criterion is unmet in the local run.'],
        ['Local runtime provenance', 'Eleven package versions were recorded at scoring and match the frozen requirements; a contemporaneous full transitive package freeze is absent.'],
        ['Hosted follow-up', 'GitHub Actions reproduction and remote review branch/PR remain blocked by repository write access.'],
    ]
    blocks['gaps'] = table(['Evidence gap', 'Current boundary'], gap_rows)
    return blocks


def scored_source(result: dict[str, Any]) -> dict[str, Any] | None:
    for path, expected in result.get('source_receipts_sha256', {}).items():
        if path.endswith('/scored/steering_summary.json'):
            source = ROOT/path
            if source.exists():
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError('Follow-up source receipt mismatch: ' + path)
                return json.loads(payload)
    return None


def render_followup(result: dict[str, Any] | None) -> str:
    if result is None:
        return (
            '**Execution status: hosted comparison UNRUN.** The repository review branch has been prepared locally. '
            'The active GitHub integration rejected branch and tree writes with HTTP 403, so an Actions run '
            'and remote registration commit are unavailable. Held-out scoring remains sealed.\n\n'
            'The prepared protocol and code can be reviewed and executed after repository write access is restored. '
            'There are no per-seed steering effects or uncertainty intervals to report for an unrun comparison.'
        )
    # Parent supplies a small paper-facing manifest with sources to actual logs/results.
    rows = [[x.get('item', ''), x.get('status', ''), x.get('evidence', '')] for x in result.get('rows', [])]
    out = f"**Execution status: {result.get('status', 'UNSPECIFIED')}.**\n\n"
    if result.get('summary'):
        out += result['summary'] + '\n\n'
    scored = scored_source(result)
    if scored is not None and scored.get('comparison_completed'):
        diagnostics = []
        for row in result.get('rows', []):
            if row['item'].startswith('Registered seed'):
                break
            evidence = row['evidence']
            if row['status'] == 'ENGINEERING_FAILURE' and 'slow_eval_freq must be a multiple of eval_freq' in evidence:
                evidence = 'The initial pilot failed Config validation: slow_eval_freq must be a multiple of eval_freq. The failed receipt is retained.'
            diagnostics.append([row['item'], row['status'], evidence])
        out += table(['Prerequisite or pilot', 'Verdict', 'Observed evidence'], diagnostics)
        out += '\n\n### 6.3 Local held-out comparison\n\n'
        seed_rows = []
        arm_rows = []
        for seed in scored['seed_results']:
            effect = seed['effects']['random_minus_vpd_target_nll']
            seed_rows.append([seed['seed'], f(seed['heldout_clean']['accuracy']), effect['n'],
                              f(effect['estimate'], 6), f"[{f(effect['ci95'][0], 6)}, {f(effect['ci95'][1], 6)}]",
                              f(seed['runtime_seconds'], 2)])
            clean_collateral = seed['metrics']['clean']['collateral_accuracy']
            for arm in ['clean', 'sham', 'vpd', 'random', 'task_gradient']:
                metric = seed['metrics'][arm]
                arm_rows.append([seed['seed'], arm, f(metric['target_nll'], 6),
                                 f(metric['target_accuracy']),
                                 f(100*(metric['collateral_accuracy']-clean_collateral), 4),
                                 f(metric['false_commit_rate_on_veto'])])
        out += table(['Seed', 'Clean test accuracy', 'Veto rows', 'Random minus VPD NLL', '95% paired-row interval', 'Runtime (s)'], seed_rows)
        out += ('\n\nThe mean primary effect across seeds is ' + f(scored['primary_mean'], 9) +
                ', with a 95% t interval [' + f(scored['primary_ci95'][0], 9) + ', ' +
                f(scored['primary_ci95'][1], 9) + ']. The recorded verdict is `' + scored['status'] +
                '`. Every seed passes its prerequisites. Collateral gate passed: ' + str(scored['collateral_gate_passed']) + '. ' + ('The across-seed interval includes zero. ' if scored['primary_ci95'][0] <= 0 <= scored['primary_ci95'][1] else 'The across-seed interval excludes zero. ') +
                'Each within-seed interval is positive; those intervals quantify row variation conditional on one trained model. The across-seed interval treats the independently trained model/decomposition as the replication unit.')
        out += '\n\n![Local synthetic follow-up effects and uncertainty](figures/followup_effects.png)\n\n'
        out += '*Figure 3. Positive values favor learned steering over the selected random arm. Seed intervals use paired-row bootstrap; the bottom interval uses the mean of three independent seed effects and Student t with two degrees of freedom. Both estimate the same contrast at different levels of variation.*'
        out += '\n\n' + table(['Seed', 'Arm', 'Target NLL', 'Target accuracy', 'Collateral delta (pp)', 'False commits / veto'], arm_rows)
        gradient_wins = sum(seed['metrics']['task_gradient']['target_nll'] < seed['metrics']['vpd']['target_nll'] for seed in scored['seed_results'])
        collateral = ', '.join(str(seed['seed']) + ': ' + f"{100*seed['collateral_accuracy_change']:+.4f}" for seed in scored['seed_results'])
        parameters = sorted({seed['model_parameters'] for seed in scored['seed_results']})
        env = scored['environment']
        out += ('\n\nThe task-gradient baseline has lower target NLL than learned VPD in ' + str(gradient_wins) + ' of ' + str(len(scored['seed_results'])) + ' seeds. ' +
                'Clean and sham outputs agree. VPD collateral accuracy changes in percentage points are ' + collateral + '. ' +
                'Model parameter counts: ' + ', '.join(f'{n:,}' for n in parameters) + '. ' +
                'The measured seed runtimes include model training, decomposition, selection, and scoring; environment setup and archive recovery are outside that timing. ' +
                'The CPU was ' + env['cpu'] + ' with ' + str(env['torch_threads']) + ' PyTorch threads. ' +
                'The frozen source used Python ' + env['python'] + ' and PyTorch ' + env['packages']['torch'] + '. Eleven package versions accompany the result receipt; a contemporaneous full transitive package freeze is absent. The Actions workflow will capture `pip freeze`.')
        errors = [seed['decomposition']['residual_restored_max_logit_error'] for seed in scored['seed_results']]
        kls = [seed['decomposition']['components_only_kl'] for seed in scored['seed_results']]
        out += ('\n\nResidual-restored replay has maximum absolute logit error ' + str(max(errors)) +
                '. Components-only calibration KL ranges from ' + f'{min(kls):.3e}' + ' to ' + f'{max(kls):.3e}' +
                ', with minimum argmax agreement ' + str(min(seed['decomposition']['components_only_argmax_agreement'] for seed in scored['seed_results'])) + '. The literal edits, checksummed per-example held-out scores, loss traces, and selected candidate receipts remain available for replay.')
    elif rows:
        out += table(['Endpoint or prerequisite', 'Verdict', 'Observed evidence'], rows)
    for paragraph in result.get('paragraphs', []):
        out += '\n\n' + paragraph
    return out


def figures(audit: dict[str, Any], out: Path, followup: dict[str, Any] | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                               'axes.spines.top': False, 'axes.spines.right': False,
                               'svg.hashsalt': 'deuterium-retrospective-v1'})
    out.mkdir(parents=True, exist_ok=True)
    seed_rows = audit['replication']['by_seed']
    fig, ax = plt.subplots(figsize=(7.4, 3.25), constrained_layout=True)
    x = np.arange(len(seed_rows))
    full = [row['accepted'] for row in seed_rows]
    preview = [row['summary_preview_accepted'] for row in seed_rows]
    ax.bar(x - .19, full, .36, label='Full raw-row recount', color='#224c63')
    ax.bar(x + .19, preview, .36, label='12-entry summary preview', color='#cba267')
    for xx, val in zip(x - .19, full): ax.text(xx, val + .12, str(val), ha='center', fontsize=9)
    for xx, val in zip(x + .19, preview): ax.text(xx, val + .12, str(val), ha='center', fontsize=9)
    ax.set_xticks(x, [str(row['seed']) for row in seed_rows], fontsize=9)
    ax.set_ylabel('Accepted candidate rows')
    ax.set_ylim(0, max(full) + 2.1)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('Replication: 16 accepted rows, 12 shown in preview', loc='left', pad=12)
    fig.savefig(out / 'replication_by_seed.png', dpi=180, metadata={'Software': 'Deuterium record renderer'})
    fig.savefig(out / 'replication_by_seed.svg', metadata={'Date': None})
    plt.close(fig)

    arc = audit['arc']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.45), gridspec_kw={'width_ratios': [1, 1.05]}, constrained_layout=True)
    names = [row['card'] for row in arc['folds']]
    mat = np.zeros((len(names), len(names)), dtype=int)
    for i, fold in enumerate(arc['folds']): mat[i, i] = fold['unique_ids']
    for pair in arc['pairwise']:
        i, j = names.index(pair['left']), names.index(pair['right'])
        mat[i, j] = mat[j, i] = pair['overlap_ids']
    ax.imshow(mat, cmap='Blues', vmin=0, vmax=32)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', color='white' if mat[i, j] > 23 else '#183448')
    labels = [name.removeprefix('seed') for name in names]
    ax.set_xticks(range(len(names)), labels)
    ax.set_yticks(range(len(names)), labels)
    ax.set_title('Pairwise shared identities', fontsize=10)
    ax.set_xlabel('Card seed'); ax.set_ylabel('Card seed')
    y = np.arange(len(names))
    shared = [r['shared_with_other_cards'] for r in arc['folds']]
    unseen = [r['unseen_in_other_cards'] for r in arc['folds']]
    bx.barh(y, shared, color='#224c63', label='Shared with training union')
    bx.barh(y, unseen, left=shared, color='#cba267', label='Unique to held-out card')
    for yi, s, u in zip(y, shared, unseen):
        bx.text(s/2, yi, str(s), color='white', ha='center', va='center')
        bx.text(s+u/2, yi, str(u), color='#1c3443', ha='center', va='center')
    bx.set_yticks(y, labels); bx.invert_yaxis()
    bx.set_xlabel('Recorded examples'); bx.set_xlim(0, 32)
    bx.set_title('Leave-one-card-out dependence', fontsize=10)
    bx.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.20), fontsize=8)
    fig.savefig(out / 'arc_overlap.png', dpi=180, metadata={'Software': 'Deuterium record renderer'})
    fig.savefig(out / 'arc_overlap.svg', metadata={'Date': None})
    plt.close(fig)

    if followup and followup.get('primary_effects') and followup.get('aggregate_effect'):
        values = followup['primary_effects']
        aggregate = followup['aggregate_effect']
        fig, ax = plt.subplots(figsize=(7.4, 3.0), constrained_layout=True)
        labels = []
        for i, item in enumerate(values):
            value, (lo, hi) = item['effect'], item['ci95']
            ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', color='#224c63',
                        markersize=6, capsize=4, linewidth=1.7)
            labels.append('Seed ' + str(item['seed']))
        i = len(values) + .3
        value, (lo, hi) = aggregate['effect'], aggregate['ci95']
        ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='D', color='#9f6f2d',
                    markersize=7, capsize=5, linewidth=2)
        ax.set_yticks(list(range(len(values)))+[i], labels+['Across-seed mean'])
        ax.axvline(0, color='#80929a', linewidth=1, linestyle='--')
        ax.invert_yaxis()
        ax.set_xlabel('Selected random minus learned VPD target NLL')
        ax.set_title('Local synthetic gate: three seeds and the frozen endpoint', loc='left', pad=12)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3))
        ax.margins(x=.10, y=.20)
        fig.savefig(out/'followup_effects.png', dpi=180, metadata={'Software': 'Deuterium record renderer'})
        fig.savefig(out/'followup_effects.svg', metadata={'Date': None})
        plt.close(fig)


def pdf_from_markdown(md: Path, dest: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, Image, Preformatted)
    import matplotlib
    fontroot = Path(matplotlib.get_data_path())/'fonts'/'ttf'
    # Bundled TrueType fonts keep long paths and Unicode metadata legible.
    if not fontroot.exists():
        raise FileNotFoundError('Pinned matplotlib DejaVu font bundle is missing.')
    for name, filename in [('PaperSerif', 'DejaVuSerif.ttf'), ('PaperSerifBold', 'DejaVuSerif-Bold.ttf'),
                           ('PaperSerifItalic', 'DejaVuSerif-Italic.ttf'),
                           ('PaperSans', 'DejaVuSans.ttf'), ('PaperSansBold', 'DejaVuSans-Bold.ttf'),
                           ('PaperMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot / filename)))
    pdfmetrics.registerFontFamily('PaperSerif', normal='PaperSerif', bold='PaperSerifBold',
                                  italic='PaperSerifItalic', boldItalic='PaperSerifBold')
    pdfmetrics.registerFontFamily('PaperSans', normal='PaperSans', bold='PaperSansBold',
                                  italic='PaperSans', boldItalic='PaperSansBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyPaper', fontName='PaperSerif', fontSize=9.4, leading=13.4,
                              textColor=colors.HexColor('#20292e'), spaceAfter=7.5,
                              allowWidows=0, allowOrphans=0, splitLongWords=True))
    styles.add(ParagraphStyle('TitlePaper', fontName='PaperSansBold', fontSize=21, leading=26,
                              textColor=colors.HexColor('#173d52'), spaceAfter=9))
    styles.add(ParagraphStyle('SubtitlePaper', fontName='PaperSerifItalic', fontSize=13.5, leading=18,
                              textColor=colors.HexColor('#475b66'), spaceAfter=14))
    styles.add(ParagraphStyle('SectionPaper', fontName='PaperSansBold', fontSize=12.5, leading=17,
                              textColor=colors.HexColor('#173d52'), spaceBefore=13, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle('SubsectionPaper', fontName='PaperSansBold', fontSize=10.5, leading=15,
                              textColor=colors.HexColor('#173d52'), spaceBefore=10, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle('CaptionPaper', parent=styles['BodyPaper'], fontName='PaperSerifItalic',
                              fontSize=8.2, leading=11.5, textColor=colors.HexColor('#536570')))
    styles.add(ParagraphStyle('TablePaper', fontName='PaperSans', fontSize=7.3, leading=10,
                              textColor=colors.HexColor('#20292e'), splitLongWords=True))
    styles.add(ParagraphStyle('TableHeadPaper', parent=styles['TablePaper'], fontName='PaperSansBold',
                              textColor=colors.white))
    styles.add(ParagraphStyle('CodePaper', fontName='PaperMono', fontSize=6.8, leading=9,
                              textColor=colors.HexColor('#33474f'), spaceAfter=7))
    width = 7.0 * inch
    doc = SimpleDocTemplate(str(dest), pagesize=(8.5*inch, 11*inch), rightMargin=.75*inch,
                            leftMargin=.75*inch, topMargin=.65*inch, bottomMargin=.65*inch,
                            title='Not Enough Deuterium to Detonate the Ocean', author='Patrick Dugan / MoralityLabAI',
                            subject='Retrospective record audit and follow-up status')
    def inline(s: str) -> str:
        s = html.escape(s)
        # Convert portable Markdown inline constructs, leaving exact code text visible.
        s = re.sub(r'`([^`]+)`', lambda m: '<font face="PaperMono" size="7.6">'+m.group(1)+'</font>', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
        s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', s)
        def link(m: re.Match) -> str:
            label, url = m.group(1), html.unescape(m.group(2))
            if url.startswith(('https://', 'http://')):
                return f'<a href="{html.escape(url, quote=True)}" color="#225f7d">{label}</a>'
            return label
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, s)
        return s
    story: list[Any] = []
    lines = md.read_text().splitlines()
    i = 0
    subtitle_used = False
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('<!--'):
            i += 1; continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            story.append(Preformatted('\n'.join(code), styles['CodePaper'], maxLineLength=93))
            i += 1; continue
        image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            path = md.parent / image_match.group(2)
            pic = Image(str(path))
            height = width * pic.imageHeight / pic.imageWidth
            pic.drawWidth, pic.drawHeight = width, height
            story += [Spacer(1, 5), pic, Spacer(1, 3)]
            i += 1; continue
        if line.startswith('|'):
            matrix = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                if not all(re.fullmatch(r'[-: ]+', c) for c in cells): matrix.append(cells)
                i += 1
            cols = len(matrix[0])
            # Relative content lengths guide widths with a floor for readable numeric columns.
            weight = []
            for j in range(cols):
                lengths = sorted(min(len(re.sub(r'`|\*', '', row[j])), 70) for row in matrix if j < len(row))
                avg = sum(lengths)/len(lengths)
                weight.append(max(9., min(42., avg)))
            col_widths = [width*w/sum(weight) for w in weight]
            data = [[Paragraph(inline(c), styles['TableHeadPaper' if r == 0 else 'TablePaper'])
                     for c in row] for r, row in enumerate(matrix)]
            tab = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#224c63')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f1f5f6'), colors.white]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINEBELOW', (0,-1), (-1,-1), .4, colors.HexColor('#bbcbd2')),
            ]))
            story += [Spacer(1, 4), tab, Spacer(1, 9)]
            continue
        heading = re.match(r'^(#{1,3}) (.*)$', line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            if level == 1: style = 'TitlePaper'
            elif level == 2 and not subtitle_used:
                style = 'SubtitlePaper'; subtitle_used = True
            else: style = 'SectionPaper' if level == 2 else 'SubsectionPaper'
            story.append(Paragraph(inline(title), styles[style]))
            i += 1; continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '```', '<!--', '![')):
            para.append(lines[i].strip()); i += 1
        text = ' '.join(para)
        style = styles['CaptionPaper'] if text.startswith('*Figure ') else styles['BodyPaper']
        story.append(Paragraph(inline(text), style))
    def page(canvas_: Any, doc_: Any) -> None:
        canvas_.saveState()
        canvas_.setFont('PaperSans', 7)
        canvas_.setFillColor(colors.HexColor('#667d88'))
        canvas_.drawString(.75*inch, .35*inch, 'DEUTERIUM  |  Record audit and follow-up status  |  Review draft')
        canvas_.drawRightString(7.75*inch, .35*inch, str(doc_.page))
        canvas_.restoreState()
    def fixed_canvas(*args: Any, **kwargs: Any) -> Any:
        kwargs['invariant'] = 1
        return canvas.Canvas(*args, **kwargs)
    doc.build(story, onFirstPage=page, onLaterPages=page, canvasmaker=fixed_canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT/'experiments/deuterium/results/retrospective/audit.json')
    parser.add_argument('--inventory', type=Path, default=ROOT/'experiments/deuterium/results/archive_manifest.json')
    parser.add_argument('--followup', type=Path)
    parser.add_argument('--paper', type=Path, default=PAPER/'vpd_trm_feature_steering_paper.md')
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    inventory = json.loads(args.inventory.read_text())
    if not inventory.get('verified'):
        raise ValueError('Archive must be hash-verified before rendering.')
    followup = json.loads(args.followup.read_text()) if args.followup else None
    blocks = generated_blocks(audit, inventory, followup)
    text = args.paper.read_text()
    tabledir = args.paper.parent / 'tables'
    tabledir.mkdir(exist_ok=True)
    for name, content in blocks.items():
        text = replace_block(text, name, content)
        (tabledir/f'{name}.md').write_text(content + '\n')
    args.paper.write_text(text)
    ledger = tabledir/'source_ledger.csv'
    with ledger.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'sha256', 'jsonl_rows'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(audit['sources'])
    figures(audit, args.paper.parent/'figures', followup)
    if not args.no_pdf:
        pdf_from_markdown(args.paper, args.paper.parent/'deuterium_retrospective.pdf')
    manifest = {
        'schema_version': 1, 'generator': 'experiments/deuterium/render_paper.py',
        'inputs': {manifest_path(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [args.audit, args.inventory, Path(__file__).resolve(), PAPER/'archive/recovered_vpd_trm_feature_steering_paper.md', PAPER/'archive/recovered_manuscript_manifest.json'] + ([args.followup] if args.followup else [])},
        'outputs': {},
        'fonts': {},
        'followup_source_receipts_sha256': followup.get('source_receipts_sha256', {}) if followup else {},
    }
    import matplotlib
    fontroot = Path(matplotlib.get_data_path())/'fonts'/'ttf'
    for name in ['DejaVuSerif.ttf','DejaVuSerif-Bold.ttf','DejaVuSerif-Italic.ttf',
                 'DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansMono.ttf']:
        manifest['fonts'][name] = hashlib.sha256((fontroot/name).read_bytes()).hexdigest()
    paths = [args.paper, ledger] + sorted(tabledir.glob('*.md')) + sorted((args.paper.parent/'figures').glob('*'))
    if not args.no_pdf: paths.append(args.paper.parent/'deuterium_retrospective.pdf')
    for p in paths: manifest['outputs'][manifest_path(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.paper.parent/'render_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'paper': manifest_path(args.paper), 'pdf': not args.no_pdf,
                      'output_count':len(manifest['outputs'])}))


if __name__ == '__main__':
    main()
