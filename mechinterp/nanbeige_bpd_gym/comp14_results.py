"""Present the immutable, independently audited COMP14 aggregate; no raw-run reads."""
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '9db0d4c973fe27662c6b75805d2cabdff9a10bc9'
SOURCE = 'reports/nanbeige_bpd_gym/nan_comp14_progress.json'
SOURCE_SHA = '67dd9d6601ff800e334f7a901d1948620d5159ccc180fb627be07e1a0547e6a6'
OUTPUT = Path('reports/nanbeige_bpd_gym/NAN_COMP14_RESULTS.md')


def render(data):
    backends = data['backends']
    for item in backends.values():
        assert item['status'] == 'completed_audited'
        assert item['audited_contexts'] == 240 and item['missing_contexts'] == 0
        assert item['physical_vectors_checked'] == 9600
        assert len(item['competence_screens']) == 4
        assert all(g['passed'] for g in item['competence_screens'])
    geo = data['paired_initial_prefix_geometry']
    assert geo['status'] == 'completed' and geo['matched_initial_prefixes'] == 240
    assert len(geo['groups']) == 40 and data['sealed_test_accessed'] is False
    def group(b, split, arm, endpoint):
        rows = [g for g in backends[b]['groups'] if (g['split'], g['arm'], g['endpoint']) == (split, arm, endpoint)]
        assert len(rows) == 1
        return rows[0]
    # Equal scores are observed, never assumed to mean identical model histories.
    for b in backends:
        for split in ('calibration', 'development'):
            for arm in ('legacy', 'focused', 'copy', 'legacy_erased', 'focused_erased'):
                a, z = [group(b, split, arm, e) for e in ('first_token_loop2', 'bounded_loop2')]
                for key in ('correct', 'contexts', 'format_valid', 'both_world_roots_correct', 'complete_world_pairs'):
                    assert a[key] == z[key], (b, split, arm, key)
    lines = ['# Fresh competence pilot: matched BF16 and NF4 results', '',
        'Both backends completed all 240 contexts and independent audits. BF16 solved 32/32 DEV worlds under each fixed recipe; NF4 solved 29/32 under each. Both passed all four new-condition screens. These fresh source-conditioned pointer tasks establish a usable development condition; the earlier algebraic and FIT07 failures remain unchanged. No donor transfer was attempted. The original 168-case test stayed sealed.', '',
        'The protocol, public synthetic sources, host-only oracle labels, bounded parser and criteria were frozen and pushed at `9bcfec1557775376e8ad300c5ffaf916aa616a13` before pretrained inference. Both synthetic source and oracle fixtures are committed publicly; host-only means isolated from the model worker, not secret. Source seal: `b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f`. Eight fresh CAL roots and sixteen fresh DEV roots each have two graph worlds with distinct correct answers. Each world has legacy/focused prompts, a copy control and both source-erasure controls. The recipes jointly change instructions and demonstrations, so their effects cannot be separated.', '',
        'Full-vocabulary greedy loop-2 generation used an anchored A–P parser and a six-token/64-character bound. First-token and bounded endpoints have the same aggregate counts in every split/arm here; both remain separately recorded in the complete aggregate. Loop-1 first-token results are shown separately below.', '',
        '![DEV accuracy, paired-root success and physical recurrence displacement](figures/nan_comp14_accuracy_recurrence.png)', '',
        '[Download SVG](figures/nan_comp14_accuracy_recurrence.svg) · [Figure values and provenance](figures/nan_comp14_figure_receipt.json). The figure summarizes the committed audit; it introduces no new grading or model calls.', '',
        '## Correctness and controls', '',
        'Each cell gives correct worlds, then roots with both counterfactual worlds correct. Worlds, recipes and repeated controls from one root are correlated.', '',
        '| Split | Arm | BF16 worlds; both-correct roots | NF4 worlds; both-correct roots |',
        '|---|---|---:|---:|']
    for split in ('calibration', 'development'):
        for arm in ('legacy', 'focused', 'copy', 'legacy_erased', 'focused_erased'):
            cells = []
            for b in ('bf16', 'nf4'):
                g = group(b, split, arm, 'first_token_loop2')
                cells.append(f"{g['correct']}/{g['contexts']}; {g['both_world_roots_correct']}/{g['complete_world_pairs']}")
            lines.append(f"| {split} | {arm} | {' | '.join(cells)} |")
    lines += ['', 'Each recipe/readout screen required at least 6/8 both-correct CAL roots and 12/16 DEV roots, every copy pair correct, byte-identical erased repeats and zero erased both-world successes. All eight backend × recipe × endpoint screens passed. Erased single-world successes remain in the table; identical source-free replies cannot solve both deliberately distinct targets. This is a development screen, with zero formal comparisons and no calibrated causal floor, sham floor or power claim.', '',
        'The two NF4 recipes tie on world accuracy but differ in both-correct DEV roots (legacy 14/16, focused 13/16). This supplies no evidence that the focused recipe improves competence. Bounded success cannot substitute for a first-token gate in future fitting.', '',
        '| Split | Arm | BF16 loop-1 first-token correct | NF4 loop-1 first-token correct |', '|---|---|---:|---:|']
    for split in ('calibration', 'development'):
        for arm in ('legacy', 'focused', 'copy', 'legacy_erased', 'focused_erased'):
            cells = []
            for b in ('bf16', 'nf4'):
                g = group(b, split, arm, 'first_token_loop1')
                cells.append(f"{g['correct']}/{g['contexts']}")
            lines.append(f"| {split} | {arm} | {' | '.join(cells)} |")
    lines += ['', '## Matched physical recurrence', '',
        'All 240 identical initial prefixes are paired across backends. Each backend audit checked 9,600 physical vectors. The table shows every DEV main-recipe/layer group: 32 worlds from 16 roots per row, average worlds within root then average roots equally. All metrics have 16 defined root means and zero undefined contexts. CAL, copy and erasure groups remain in the linked JSON (40 groups total). Layers use zero-based indices.', '',
        'Relative L2 is ||NF4 − BF16|| / ||BF16||. Displacement uses physical residual(loop2) − residual(loop1), before final loop RMSNorm. Values below are descriptive means, without confidence intervals. No independently fitted SAE feature indices are subtracted.', '',
        '| Recipe | Layer | Loop1 rel L2 | Loop1 cosine | Loop2 rel L2 | Loop2 cosine | Displacement rel L2 | Displacement cosine |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('loop1_relative_l2', 'loop1_cosine', 'loop2_relative_l2', 'loop2_cosine', 'recurrence_displacement_relative_l2', 'recurrence_displacement_cosine')
    selected = [g for g in geo['groups'] if g['split'] == 'development' and g['arm'] in ('legacy', 'focused')]
    assert len(selected) == 8
    for g in selected:
        assert g['source_roots'] == 16 and g['contexts'] == 32
        for key in metrics:
            assert g[key]['defined_root_means'] == 16 and g[key]['undefined_contexts'] == 0
        values = ' | '.join(f"{g[k]['equal_root_mean']:.6f}" for k in metrics)
        lines.append(f"| {g['arm']} | {g['layer_zero_based']} | {values} |")
    lines += ['', 'High state cosine similarity does not imply preserved recurrence geometry or mechanism equivalence. Layer-11 recurrence-displacement relative L2 is about 34–36% across the two recipes. Large coordinates in the original residual basis can dominate state cosine. Later generated histories are not paired; accuracy differences and initial-state geometry alone establish no causal mediation.', '',
        '## Resources and audit scope', '',
        '| Backend | Peak allocated GiB | Peak reserved GiB | Audited contexts | Audited physical vectors |', '|---|---:|---:|---:|---:|']
    for b in ('bf16', 'nf4'):
        g = backends[b]
        lines.append(f"| {b} | {g['CUDA_peak_allocated_bytes']/2**30:.6f} | {g['CUDA_peak_reserved_bytes']/2**30:.6f} | {g['audited_contexts']}/240 | {g['physical_vectors_checked']} |")
    lines += ['', 'NF4 retains all 22 decoder layers and the unchanged BF16 output head on GPU; BF16 retains four layers plus a reusable stage and the same head. CPU embeddings save about .951 GiB. Allocator peaks differ from driver dedicated/shared-memory samples and do not prove physical residency or absence of WDDM paging. One large model, one numerical CPU thread and Below Normal priority protect the separate math workload; shared-machine timing is not an isolated throughput benchmark.', '',
        'Independent terminal audits reparsed graph answers, decoded every generated prefix with the pinned tokenizer, used an independent bounded grammar, verified greedy-score bindings and checked every physical-vector archive hash, shape, finite value and expected coverage. Both pretrained target hashes were unchanged; full-vocabulary CPU/resident-head parity passed. Synthetic prompts and oracle fixtures are public source artifacts. Generated replies, token IDs, vectors and per-record run grades remain local.', '',
        'The initial Linux checkout-byte failures are preserved in the [CI byte-policy note](../../.github/scripts/COMP14_CHECKOUT_BYTES.md). The later CI-only checker correction changed no frozen inference source and caused no model replay. Both corrected COMP14 workflows passed, followed by the successful COMP14/GRAD15/gym CPU checks on the next freeze.', '',
        'The continuous goal remains active. The separately [frozen GRAD15 gradient/resource preflight](../../experiments/nan_grad15/PROTOCOL.md) follows these completed results; it performs zero optimizer steps and cannot establish trained NF4 VPD/BPD or authorize donor transfer.', '',
        '## Reproducible presentation', '',
        f'This host-only presentation reads the committed aggregate at `{COMMIT}`, SHA256 `{SOURCE_SHA}`. It performs no inference, fitting, grading, raw-run analysis, outcome selection or new formal comparisons. Regenerate the text from the repository root with `python -m mechinterp.nanbeige_bpd_gym.comp14_results`; the standalone Matplotlib figure uses `python -m mechinterp.nanbeige_bpd_gym.plot_comp14_comparison` and existing reporting packages.', '',
        f'[Immutable aggregate](https://github.com/MoralityLabAI/LoopedTransformers/blob/{COMMIT}/{SOURCE}) · [Complete scores and screens](NAN_COMP14_PROGRESS.md) · [Frozen protocol](../../experiments/nan_comp14/PROTOCOL.md) · [Source manifest](../../experiments/nan_comp14/SOURCE_MANIFEST.json) · [Broader research report](NAN_GYM_RESEARCH_REPORT.md)', '']
    return '\n'.join(lines)


def main():
    raw = subprocess.check_output(['git', 'show', f'{COMMIT}:{SOURCE}'])
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    OUTPUT.write_text(render(json.loads(raw)), encoding='utf-8', newline='\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
