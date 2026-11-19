"""Recompute paper summaries from preserved BPD receipts without model execution.

This arithmetic audit conditions on the archived measurements. It cannot recover
unrecorded logits, authenticate sample independence, or repair the legacy top-k
mask statistic. Its only output population is the exposed engineering cohort.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same(observed: float, expected: float, label: str) -> None:
    require(math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
            f'Cross-study arithmetic disagrees: {label}: {observed} != {expected}')


def audit(paper: Path) -> dict[str, Any]:
    paths: dict[str, str] = {}

    def read(family: str, name: str) -> Any:
        path = paper / 'evidence' / family / name
        paths[str(path.relative_to(paper))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    old = read('bpd_dt04', 'DT04_ORIGINAL_SUMMARY.json')
    old_audit = read('bpd_dt04', 'DT04_ORIGINAL_AUDIT.json')
    post = read('bpd_dt04', 'POSTLOCK_59ROOT_AND_COMPETENT_DISSECTION.json')
    require(old['verdict'] == 'INCOMPLETE_OR_INELIGIBLE', 'DT04 original verdict changed')
    require(old_audit['loaded_roots'] == len(old_audit['per_root_audit']) == 59,
            'DT04 loaded-root roster changed')
    clean_roots = sum(r['passed'] for r in old_audit['per_root_audit'].values())
    require(clean_roots == old['roots_completed'] == 56, 'DT04 original audit-clean count changed')
    require(old['roots_admitted'] == old_audit['admitted_roots_count'] == 16,
            'DT04 strict eligibility count changed')
    require(post['all_59_roots_completed'] and post['all_472_signed_jacobian_shams_certified'],
            'DT04 post-lock scope changed')
    require(post['original_summary_sha256'] == paths['evidence/bpd_dt04/DT04_ORIGINAL_SUMMARY.json'],
            'DT04 post-lock summary link disagrees')
    competent19 = post['competent_sudoku4_subset_n19']
    require(len(set(competent19['roots'])) == 19, 'DT04 selected subset changed')
    competent_effect = (competent19['steering_by_depth']['8']['task_gramian_r8']['gold_whole_board_exact']
                        - competent19['steering_by_depth']['8']['signed_jac_sham_mean8']['gold_whole_board_exact'])
    same(competent_effect, post['evalmath_competent19_descriptive']['mean_pilot_effect'],
         'DT04 selected19 task minus sham')
    dt04 = {
        'unit': 'independently trained model root',
        'loaded_roots': old_audit['loaded_roots'],
        'original_audit_clean_roots': clean_roots,
        'original_strictly_admitted_roots': old['roots_admitted'],
        'original_verdict': old['verdict'],
        'postlock_evidence_class': post['analysis_class'],
        'postlock_all59_effect': post['evalmath_all59_certified_sham_roots']['mean_pilot_effect'],
        'postlock_decision': post['evalmath_all59_certified_sham_roots']['decision'],
        'postselected_competent_roots': competent19['roots'],
        'postselected_competent19_effect': competent_effect,
        'teacher_conditioned_sparse_exact_by_depth': competent19['bpd_sparse_top12_exact_by_depth'],
        'geometry_original_audit_clean56': old['geometry_and_bpd_factors'],
        'arithmetic_scope': 'Original audit counts and selected19 arm subtraction; all59 effect retained from post-lock summary.',
    }

    routing = read('bpd_dt05', '08_autonomous_routing_study.json')
    units = routing['units']
    expected_ids = [f'excluded_pilot_root_{seed}' for seed in range(2605900, 2605908)]
    require([u['unit_id'] for u in units] == expected_ids, 'DT05 routing root roster changed')
    require(routing['n_independent_roots'] == len(units) == 8, 'DT05 routing root count changed')
    require(all(u['shared_vu_verified'] and u['n_cases'] == 64 for u in units),
            'DT05 routing factor/case contract changed')
    arm_names = ['fixed', 'online', 'teacher_scheduled', 'random']
    per_root = []
    for u in units:
        exact = {'dense_teacher': u['dense_teacher']['by_depth']['8']['exact_board_accuracy']}
        for arm in arm_names:
            exact[arm] = u['arms'][arm]['by_depth']['8']['exact_board_accuracy']
            for site in u['arms'][arm]['active_component_telemetry']['sites'].values():
                require(site['components'] == 24 and site['declared_top_k'] == 12,
                        'DT05 routing active factor budget changed')
            same(exact[arm] * u['n_cases'], round(exact[arm] * u['n_cases']),
                 f'{u["unit_id"]}/{arm} exact count')
        per_root.append({'unit_id': u['unit_id'], 'n_cases': u['n_cases'], 'exact_board': exact})
    selected = [u for u in units if u['dense_teacher']['by_depth']['8']['exact_board_accuracy'] >= .5]
    require(len(selected) == routing['n_competent_roots_ge_50pct'] == 3,
            'DT05 postselected competent count changed')

    def aggregate(items: list[dict[str, Any]], saved: dict[str, Any]) -> dict[str, float]:
        values = {'dense_teacher_exact_mean': statistics.mean(
            u['dense_teacher']['by_depth']['8']['exact_board_accuracy'] for u in items)}
        for arm in arm_names:
            values[f'{arm}_exact_mean'] = statistics.mean(
                u['arms'][arm]['by_depth']['8']['exact_board_accuracy'] for u in items)
        values['online_minus_fixed_exact_mean'] = values['online_exact_mean'] - values['fixed_exact_mean']
        values['teacher_scheduled_minus_online_gap_mean'] = (
            values['teacher_scheduled_exact_mean'] - values['online_exact_mean'])
        if 'online_minus_random_exact_mean' in saved:
            values['online_minus_random_exact_mean'] = values['online_exact_mean'] - values['random_exact_mean']
            for arm in arm_names:
                values[f'{arm}_kl_nats_mean'] = statistics.mean(
                    u['arms'][arm]['by_depth']['8']['kl_from_teacher_nats'] for u in items)
            for arm in ['online', 'fixed']:
                values[f'{arm}_forward_seconds_mean'] = statistics.mean(u['arms'][arm]['forward_seconds'] for u in items)
                values[f'{arm}_unique_active_components_mean'] = statistics.mean(
                    u['arms'][arm]['active_component_telemetry']['mean_unique_active_components'] for u in items)
        for key, value in values.items():
            same(value, saved[key], f'routing/{len(items)}roots/{key}')
        return values

    route_summary = {
        'unit': 'excluded engineering model root',
        'n_roots': len(units), 'n_cases_per_root': 64, 'scored_root_case_pairs_per_arm': 512,
        'depth': 8, 'factors': 24, 'active_factors': 12,
        'all8': aggregate(units, routing['aggregate_all_8_roots_depth8']),
        'postselected3': aggregate(selected, routing['aggregate_competent_roots_depth8']),
        'postselection_rule': 'Dense teacher exact board fraction at depth eight at least 0.5.',
        'postselected_unit_ids': [u['unit_id'] for u in selected],
        'per_root': per_root,
        'teacher_schedule': 'Privileged reference; no mathematical upper-bound guarantee.',
        'timing_scope': 'Online arm includes teacher-mask capture hooks and overlap arithmetic inside timing region; no isolated deployment-overhead estimate.',
        'legacy_topk_overlap': {
            'status': 'INVALID_AS_UNORDERED_SET_OVERLAP',
            'measured_quantity': 'ordered position-match fraction divided by k',
            'maximum_at_k12': 1 / 12,
            'true_set_overlap_recomputed': False,
            'corrected_eight_root_replay_available': False,
        },
    }

    transport = read('bpd_dt05', '07_dynamic_transport_and_evalmath.json')
    t_units = transport['pilot_units']
    require([u['unit_id'] for u in t_units] == expected_ids, 'DT05 transport roster changed')
    paired = []
    for u in t_units:
        arms = u['by_depth']['8']
        difference = (arms['transported_task_basis']['exact_board_accuracy']
                      - arms['static_task_basis_dose_matched']['exact_board_accuracy'])
        same(difference, u['depth8_contrasts']['transported_minus_static_exact'], 'Transport per-root contrast')
        paired.append(difference)
    gate = transport['evalmath_preflight']['evalmath_result']
    inputs = transport['evalmath_preflight']['input_spec']
    require([p['unit_id'] for p in inputs['pilot']] == expected_ids, 'Transport preflight roster changed')
    for computed, entered in zip(paired, inputs['pilot']):
        same(computed, float(Fraction(entered['paired_effect'])), 'Transport preflight paired effect')
    same(statistics.mean(paired), gate['mean_pilot_effect'], 'Transport all-root mean')
    require(gate['decision'] == 'REDESIGN_OR_PILOT_ONLY' and gate['upper_sham_quantile'] is None,
            'Transport gate or uncalibrated sham status changed')
    require(transport['confirmation_roots_consumed'] == transport['reserved_dt05_development_roots_consumed'] == 0,
            'DT05 reserved cohort exposure changed')
    transport_summary = {
        'n_roots': len(t_units), 'per_root_effects': dict(zip(expected_ids, paired)),
        'mean_transported_minus_static_exact': statistics.mean(paired),
        'scientific_variation_floor': gate['mechanistic_variation_floor'],
        'observed_worst_sham': gate['observed_worst_sham'],
        'population_sham_tail_calibrated': False,
        'registered_comparisons': gate['planned_comparisons'],
        'decision': gate['decision'],
        'reserved_development_roots_consumed': transport['reserved_dt05_development_roots_consumed'],
        'confirmation_roots_consumed': transport['confirmation_roots_consumed'],
    }

    rank = read('bpd_dt05', '10_causal_rank_study.json')
    require(rank['n_independent_roots'] == len(rank['units']) == 3, 'Causal-rank root count changed')
    require(all(u['n_eval_cases'] == 32 and u['n_discovery_cases'] == 64 and u['horizon_depth'] == 12
                for u in rank['units']), 'Causal-rank case/horizon contract changed')
    rank_means = {}
    for depth in ['1', '2', '4', '6', '8']:
        rows = [u['by_depth'][depth] for u in rank['units']]
        features = {
            'stat_full_pr_mean': [d['statistical_information_dimension']['full_dual_state']['participation_ratio'] for d in rows],
            'stat_dark_pr_mean': [d['statistical_information_dimension']['protected_dark_state']['participation_ratio'] for d in rows],
            'decodable_full_op_pr_mean': [d['linearly_decodable_dimension']['full_dual_state']['operator_spectrum']['participation_ratio'] for d in rows],
            'decodable_dark_r8_blank_acc_mean': [d['linearly_decodable_dimension']['protected_dark_state']['heldout_by_rank']['8']['heldout_blank_accuracy'] for d in rows],
            'causal_gramian_pr_mean': [d['native_future_causal_dimension']['protected_gramian_spectrum']['participation_ratio'] for d in rows],
            'causal_gramian_entropy_rank_mean': [d['native_future_causal_dimension']['protected_gramian_spectrum']['entropy_effective_rank'] for d in rows],
        }
        for label in ['oracle_exact_headroom_mean', 'oracle_kl_headroom_nats_mean',
                      'rank8_exact_advantage_over_sham_mean', 'rank8_kl_advantage_over_sham_nats_mean']:
            features[label] = []
        for d in rows:
            native = d['native_future_causal_dimension']
            head = native['behavioral_headroom']
            exact = head['full_dark_donor_oracle_exact_board'] - head['corrupted_no_edit_exact_board']
            kl = head['corrupted_no_edit_kl_nats'] - head['full_dark_donor_oracle_kl_nats']
            same(exact, head['oracle_exact_board_headroom'], 'Causal-rank exact headroom')
            same(kl, head['oracle_kl_headroom_nats'], 'Causal-rank KL headroom')
            features['oracle_exact_headroom_mean'].append(exact)
            features['oracle_kl_headroom_nats_mean'].append(kl)
            r8 = native['interventions_by_rank']['8']
            features['rank8_exact_advantage_over_sham_mean'].append(r8['candidate_exact_board'] - r8['haar_sham_mean_exact_board'])
            features['rank8_kl_advantage_over_sham_nats_mean'].append(r8['haar_sham_mean_horizon_kl_nats'] - r8['candidate_horizon_kl_nats'])
        rank_means[depth] = {key: statistics.mean(values) for key, values in features.items()}
        for key, value in rank_means[depth].items():
            same(value, rank['aggregate_by_depth'][depth][key], f'Rank depth{depth}/{key}')
    require(rank_means['4']['oracle_exact_headroom_mean'] < 0, 'Negative depth-four headroom was lost')

    stop = read('bpd_dt05', '09_first_valid_stopping_7m_and_dt04.json')
    cases = stop['cases_9x9_receipt']
    require(len(set(cases['case_ids'])) == len(cases['case_ids']) == 64 and cases['scope'] == 'engineering',
            'First-valid exposed cohort changed')
    first_valid = {}
    for checkpoint in ['yagizdevre', 'gaoxin492']:
        saved = stop[f'{checkpoint}_replay_summary']
        require(sum(saved['selected_depths_histogram'].values()) == 64, 'First-valid histogram case count')
        spent = sum(sum(chunk) for chunk in saved['active_cases_per_step_by_chunk'])
        require(1024 - spent == saved['act_case_steps_saved'], 'First-valid saved case steps disagree')
        same(spent / 1024, saved['case_step_fraction'], 'First-valid case-step fraction')
        same(saved['raw_depth16_exact_board'], saved['by_depth_exact_board']['16'], 'First-valid raw depth16')
        require(saved['compact_disagreement_count'] == 1 and not saved['compact_matches_trajectory_policy'],
                'BF16 compaction mismatch must remain visible')
        first_valid[checkpoint] = {
            'checkpoint_sha256': saved['checkpoint_sha256'], 'n_exposed_cases': 64,
            'raw_depth16_correct': round(saved['raw_depth16_exact_board'] * 64),
            'first_valid_correct': round(saved['first_valid_exact_board'] * 64),
            'fixed_case_steps': 1024, 'compact_case_steps': spent,
            'act_case_steps_saved': saved['act_case_steps_saved'],
            'raw_fixed_seconds': saved['raw_fixed_seconds'], 'compact_policy_seconds': saved['compact_policy_seconds'],
            'compact_disagreement_count': saved['compact_disagreement_count'],
            'compact_matches_trajectory_policy': saved['compact_matches_trajectory_policy'],
        }
    micro = stop['batch_composition_dt04_64cases_fp32']
    require(len(micro['runs']) == 9 and not micro['selection_uses_gold'], 'FP32 stopping design changed')
    require(all(r['prediction_disagreements_vs_reference'] == r['selected_depth_disagreements_vs_reference'] == 0
                for r in micro['runs']), 'FP32 stopping batch-composition disagreements changed')

    train = read('bpd_dt05', '06_ab_training_factors_and_probes.json')
    require(train['scientific_roots_consumed'] == 0 and train['excluded_engineering_root'] == 2605900,
            'Engineering training scope changed')
    training = {}
    for name, key in [('fp32_480', 'ab_fp32_mb128_ref_vs_opt_graphed'),
                      ('bf16_480', 'ab_bf16_mb128_ref_vs_opt_graphed'),
                      ('compound_120', 'ab_default_fp32_mb16_vs_autotuned_bf16_mb128_graphed_120steps')]:
        row = train[key]
        before, after = row['reference'], row['optimized']
        ratio = before['elapsed_seconds'] / after['elapsed_seconds']
        same(ratio, row['speedup_ratio'], f'{name} speedup')
        same(after['ema_depth8_exact_board'] - before['ema_depth8_exact_board'],
             row['numerical_parity']['depth8_exact_board_delta'], f'{name} exact accuracy difference')
        training[name] = {
            'root': row['root'], 'steps': row['steps'],
            'reference_config': before['config'], 'optimized_config': after['config'],
            'reference_seconds': before['elapsed_seconds'], 'optimized_seconds': after['elapsed_seconds'],
            'speedup_ratio': ratio, 'reference_depth8_exact': before['ema_depth8_exact_board'],
            'optimized_depth8_exact': after['ema_depth8_exact_board'],
            'recorded_numerical_parity': row['numerical_parity'],
            'arithmetic_scope': 'Runtime ratio and task delta recomputed; maximum trace differences retained from source summary.',
        }
    require(training['fp32_480']['reference_depth8_exact'] == training['fp32_480']['optimized_depth8_exact'] == 0,
            'FP32 zero-competence result must remain visible')
    require(training['bf16_480']['recorded_numerical_parity']['trajectory_difference_flagged'],
            'BF16 changed optimization trajectory must remain visible')
    return {
        'schema': 'deuterium-cross-study-arithmetic-v1', 'status': 'PASS',
        'model_inference_performed': False, 'new_scientific_data_generated': False,
        'scope': 'Recomputed arithmetic on archived engineering records; no population confirmation or cross-study pooling.',
        'source_sha256': paths, 'dt04': dt04, 'dt05_routing': route_summary,
        'dt05_transport': transport_summary,
        'dt05_causal_rank': {'n_roots': 3, 'n_eval_cases_per_root': 32, 'horizon_depth': 12,
                            'aggregate_by_depth': rank_means},
        'dt05_first_valid': first_valid,
        'dt04_fp32_stopping': {'n_exposed_cases': 64, 'batch_and_order_comparisons': 9,
                              'prediction_disagreements': 0, 'selected_depth_disagreements': 0},
        'dt05_training': training,
    }


def routing_figure(summary: dict[str, Any], paper: Path) -> list[Path]:
    """Display every archived routing root, retaining weak dense teachers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'deuterium-cross-study-20261119',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False})
    roots = summary['dt05_routing']['per_root']
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.8), sharex=True, sharey=True)
    for row, root in enumerate(roots):
        y = len(roots) - row - 1
        v = root['exact_board']
        axes[0].plot([v['fixed'], v['online']], [y, y], color='#b7cbd4', linewidth=1.8, zorder=1)
        axes[0].scatter(v['fixed'], y, marker='s', s=25, color='#7b8990', label='Fixed 12' if row == 0 else None, zorder=3)
        axes[0].scatter(v['online'], y, marker='o', s=34, color='#226886', label='Online 12' if row == 0 else None, zorder=4)
        axes[1].plot([v['online'], v['teacher_scheduled']], [y, y], color='#d8bfac', linewidth=1.8, zorder=1)
        axes[1].scatter(v['online'], y, marker='o', s=34, color='#226886', label='Online 12' if row == 0 else None, zorder=4)
        axes[1].scatter(v['teacher_scheduled'], y, marker='^', s=34, color='#bd622c', label='Privileged 12' if row == 0 else None, zorder=3)
        for ax in axes:
            ax.scatter(v['dense_teacher'], y, marker='|', s=110, linewidths=1.6, color='#173d52',
                       label='Dense teacher' if row == 0 else None, zorder=2)
    for ax, title in zip(axes, ['Online versus fixed factors', 'Online versus privileged routing']):
        ax.set_xlim(-.025, 1.025)
        ax.set_ylim(-.7, len(roots) - .3)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis='x', color='#e3e9eb', linewidth=.6, zorder=0)
        ax.tick_params(axis='y', length=0, pad=7)
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold', color='#173d52', pad=10)
        ax.set_xlabel('Exact boards at depth 8', fontsize=9)
        ax.legend(loc='upper left', bbox_to_anchor=(0, -.18), ncol=1, frameon=False,
                  fontsize=8, handletextpad=.4, borderpad=0, labelspacing=.6)
    axes[0].set_yticks(list(range(len(roots))), [r['unit_id'].rsplit('_', 1)[1] for r in reversed(roots)])
    fig.suptitle('DT05: all eight excluded engineering roots', x=.14, y=.97,
                 ha='left', fontsize=12, fontweight='bold', color='#173d52')
    fig.text(.14, .915, '64 boards per root; common V/U factors; 12 of 24 active components',
             fontsize=8, color='#526670')
    fig.subplots_adjust(left=.14, right=.98, top=.80, bottom=.29, wspace=.13)
    stem = paper / 'figures/cross_study_routing'
    paths = [stem.with_suffix('.png'), stem.with_suffix('.svg')]
    fig.savefig(paths[0], dpi=220, metadata={'Software': 'Deuterium cross-study evidence'})
    fig.savefig(paths[1], metadata={'Date': None, 'Creator': 'Deuterium cross-study evidence'})
    paths[1].write_text('\n'.join(line.rstrip() for line in paths[1].read_text().splitlines()) + '\n')
    plt.close(fig)
    return paths
