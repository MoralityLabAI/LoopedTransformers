"""Standalone Matplotlib figure from one immutable COMP14 aggregate only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = '9db0d4c973fe27662c6b75805d2cabdff9a10bc9'
SOURCE = 'reports/nanbeige_bpd_gym/nan_comp14_progress.json'
SOURCE_SHA = '67dd9d6601ff800e334f7a901d1948620d5159ccc180fb627be07e1a0547e6a6'


def main():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from .evidence_index import below_normal
    below_normal()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    raw = subprocess.check_output(['git', 'show', COMMIT + ':' + SOURCE])
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA; data = json.loads(raw)
    assert data['source_manifest_sha256'] == 'b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f'
    assert data['sealed_test_accessed'] is False and data['actual_formal_comparisons'] == 0 and not data['BPD_donor_transfer_attempted']
    for value in data['backends'].values():
        assert value['status'] == 'completed_audited' and value['audited_contexts'] == 240 and value['missing_contexts'] == 0
    arms = ('legacy', 'focused', 'copy', 'legacy_erased', 'focused_erased')
    labels = ('Legacy', 'Focused', 'Copy', 'Legacy\nerased', 'Focused\nerased')
    colors = {'bf16': '#1768ac', 'nf4': '#c66a19'}
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'svg.hashsalt': 'nan-comp14-immutable-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=.09, right=.975, bottom=.205, top=.845, hspace=.64, wspace=.29)
    fig.suptitle('Nanbeige 3B: fresh source-conditioned competence', x=.09, y=.975, ha='left', fontsize=18, weight='bold')
    fig.text(.09, .928, 'Frozen COMP14 · 16 DEV roots · 2 counterfactual worlds per root and arm', fontsize=11)
    fig.text(.09, .898, 'Native loop-2 first token · one checkpoint · all 240 contexts audited per backend', fontsize=10, color='#475569')
    plotted = []
    for ax, field, denom, title in [(axes[0, 0], 'correct', 32, 'A  Correct DEV worlds'),
                                    (axes[0, 1], 'both_world_roots_correct', 16, 'B  Both-world-correct DEV roots')]:
        x = np.arange(5)
        for offset, backend in ((-.18, 'bf16'), (.18, 'nf4')):
            groups = [next(g for g in data['backends'][backend]['groups'] if g['split'] == 'development' and g['arm'] == arm and g['endpoint'] == 'first_token_loop2') for arm in arms]
            assert all(g['contexts'] == 32 and g['complete_world_pairs'] == 16 for g in groups)
            values = [g[field] for g in groups]; bars = ax.bar(x+offset, values, width=.33, color=colors[backend], label=backend.upper(), zorder=3)
            assert [bar.get_height() for bar in bars] == values
            for arm, bar, value in zip(arms, bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.025*denom, str(value), ha='center', va='bottom', fontsize=9)
                plotted.append(dict(panel=title[0], backend=backend, arm=arm, metric=field, value=value, denominator=denom))
        ax.set_title(title, loc='left', fontsize=12.5, weight='bold', pad=12)
        ax.set_xticks(x, labels); ax.set_ylim(0, denom*1.30); ax.set_yticks(np.arange(0, denom+1, denom//4))
        ax.set_ylabel('Correct worlds / 32' if denom == 32 else 'Both worlds correct / 16 roots')
        ax.grid(axis='y', color='#dbe2ea', zorder=0); ax.set_axisbelow(True)
        ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=9)
    geom = data['paired_initial_prefix_geometry']; assert geom['status'] == 'completed' and geom['matched_initial_prefixes'] == 240
    layers = [5, 11, 17, 21]
    for ax, field, title, ylabel in [
        (axes[1, 0], 'recurrence_displacement_relative_l2', 'C  Recurrence-displacement difference', 'Relative L2 (BF16 denominator)'),
        (axes[1, 1], 'recurrence_displacement_cosine', 'D  Recurrence-displacement alignment', 'Cosine similarity')]:
        for arm, color, marker, linestyle in [('legacy', '#7758a6', 'o', '-'), ('focused', '#228365', 's', '--')]:
            groups = [next(g for g in geom['groups'] if g['split'] == 'development' and g['arm'] == arm and g['layer_zero_based'] == layer) for layer in layers]
            assert all(g['source_roots'] == 16 and g['contexts'] == 32 and g[field]['defined_root_means'] == 16 and g[field]['undefined_contexts'] == 0 for g in groups)
            values = [g[field]['equal_root_mean'] for g in groups]
            line = ax.plot(layers, values, color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=6, label=arm.capitalize()+' recipe')[0]
            assert list(line.get_ydata()) == values
            for layer, value in zip(layers, values):
                plotted.append(dict(panel=title[0], arm=arm, layer_zero_based=layer, metric=field, equal_root_mean=value, roots=16, worlds=32))
        ax.set_title(title, loc='left', fontsize=12.5, weight='bold', pad=12)
        ax.set_xlabel('Layer (zero-based)'); ax.set_xticks(layers); ax.set_ylabel(ylabel)
        ax.grid(color='#dbe2ea', zorder=0); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9, loc='best')
    axes[1, 0].set_ylim(0, .43); axes[1, 1].set_ylim(.90, 1.008)
    fig.text(.09, .142, 'Descriptive counts and equal-root means; no confidence intervals or formal comparisons.', fontsize=10)
    fig.text(.09, .112, 'C/D compare BF16 vs NF4: physical residual displacement h2 − h1 before final loop RMSNorm.', fontsize=9.5)
    fig.text(.09, .085, 'Two worlds averaged within each root. CAL/control geometry and state metrics remain in the full report.', fontsize=9.5)
    fig.text(.09, .058, 'No donor transfer, mechanism-specificity or equivalence claim. Original 168-case test sealed.', fontsize=9.5)
    fig.text(.09, .028, 'Immutable aggregate: 9db0d4c · both first-token and bounded endpoints remain separately recorded.', fontsize=8.5, color='#475569')
    out = Path('reports/nanbeige_bpd_gym/figures'); out.mkdir(exist_ok=True)
    png, svg = out / 'nan_comp14_accuracy_recurrence.png', out / 'nan_comp14_accuracy_recurrence.svg'
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib; immutable COMP14 presentation'})
    fig.savefig(svg, metadata={'Date': None}); plt.close(fig)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = dict(source_commit=COMMIT, source_file=SOURCE, source_sha256=SOURCE_SHA,
        producer_sha256=sha(__file__), matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
        source_manifest_sha256=data['source_manifest_sha256'], CPU_numerical_threads=1, priority='BelowNormal on Windows',
        evidence_class='Post hoc presentation of committed audits; zero model calls, grading, fitting or formal comparisons.',
        plotted_values=plotted, image_size_pixels=[2160, 1800], files={png.name:sha(png), svg.name:sha(svg)}, sealed_test_accessed=False)
    (out / 'nan_comp14_figure_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(dict(figure=str(png), plotted_values=len(plotted), source_sha256=SOURCE_SHA), flush=True)


if __name__ == '__main__':
    main()
