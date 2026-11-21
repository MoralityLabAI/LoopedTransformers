"""Plot immutable CTX08 aggregates; no model, raw records or new grading."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

INPUT_COMMIT = 'ad8a39dbd1cee76e08cf4f88a3eb01e4ff6668d8'
INPUT_PATH = 'reports/nanbeige_bpd_gym/nan_ctx08_comparison.json'
INPUT_SHA256 = '6ba8068957476cebf1ca1507a538d50d0afc08d3299d0a9c1cb07037ad2073ee'
SOURCE_SEAL = '7f729d76bd0b37eabbeaf4a2fd8dc0faa7621c01f8830554bbf424a63dd4d9ec'


def main():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    raw = subprocess.check_output(['git', 'show', INPUT_COMMIT + ':' + INPUT_PATH])
    assert hashlib.sha256(raw).hexdigest() == INPUT_SHA256
    data = json.loads(raw)
    assert data['source_manifest_sha256'] == SOURCE_SEAL
    assert data['matched_DEV_contexts'] == 288 and data['independent_DEV_source_families'] == 4
    assert data['formal_inferential_comparisons'] == 0
    assert not data['BPD_donor_transfer_attempted'] and not data['sealed_test_accessed']
    rows = data['paired_correctness']
    assert len(rows) == 24
    assert len({(r['context_budget'], r['task'], r['copy_control'], r['loop']) for r in rows}) == 24
    assert all(r['contexts'] == 24 and r['source_root_groups'] == 4 for r in rows)
    for r in rows:
        assert r['both_correct'] + r['BF16_only_correct'] == r['BF16_correct']
        assert r['both_correct'] + r['NF4_only_correct'] == r['NF4_correct']
        assert sum(r[k] for k in ('both_correct', 'BF16_only_correct', 'NF4_only_correct', 'neither_correct')) == 24
    geometry = sorted(data['paired_physical_geometry_ranges'], key=lambda r: r['layer_zero_based'])
    assert [r['layer_zero_based'] for r in geometry] == [5, 11, 17, 21]
    assert all(v['defined_strata'] == v['registered_strata'] == 18 for r in geometry for k, v in r.items() if k != 'layer_zero_based')

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'nan-ctx08-audited-figure-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(left=.085, right=.975, bottom=.19, top=.85, hspace=.48, wspace=.27)
    fig.suptitle('Nanbeige 3B: BF16 and NF4 under longer contexts', x=.085, y=.973, ha='left', fontsize=19, weight='bold')
    fig.text(.085, .925, 'Frozen CTX08 development study · 288 correlated contexts from 4 source families', fontsize=11)
    fig.text(.085, .895, 'One checkpoint · native loop-2 free first token · all registered task/control groups', fontsize=10, color='#475569')
    colors = {'BF16': '#1768ac', 'NF4': '#c66a19'}
    displayed = []
    pairs = [(length, task) for length in (1024, 2048, 4096) for task in ('lookup', 'join')]
    for ax, copy, title in zip(axes[0], (False, True), ('A  Main task accuracy', 'B  Explicit answer-copy controls')):
        ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=12)
        x = np.arange(6)
        for offset, backend in ((-.19, 'BF16'), (.19, 'NF4')):
            selected = [next(r for r in rows if (r['context_budget'], r['task'], r['copy_control'], r['loop']) == (length, task, copy, 2)) for length, task in pairs]
            values = [r[backend + '_correct'] for r in selected]
            bars = ax.bar(x + offset, values, width=.34, color=colors[backend], label=backend, zorder=3)
            assert [b.get_height() for b in bars] == values
            ax.bar_label(bars, labels=[str(v) for v in values], padding=3, fontsize=9)
            displayed.append({'panel': 'copy' if copy else 'main', 'backend': backend, 'correct': values, 'denominator_per_bar': 24})
        ax.set_xticks(x, [f'{length // 1024}K\n{task}' for length, task in pairs], fontsize=9)
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_ylabel('Correct / 24 contexts')
        ax.set_xlabel('Context budget and task', labelpad=8)
        ax.grid(axis='y', alpha=.18, zorder=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .915), ncol=2, frameon=False, fontsize=10)

    for ax, field, title, ylabel, ylim in (
        (axes[1, 0], 'loop2_state_cosine', 'C  Physical state agreement', 'BF16–NF4 residual cosine (loop 2)', (.94, 1.002)),
        (axes[1, 1], 'loop_displacement_relative_l2', 'D  Recurrence displacement difference', 'Relative L2 difference / BF16 displacement norm', (0, .45)),
    ):
        ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=12)
        for x, row in enumerate(geometry):
            low, high = row[field]['minimum'], row[field]['maximum']
            line, = ax.plot([x, x], [low, high], color='#4c5685', linewidth=5, solid_capstyle='round', zorder=3)
            assert list(line.get_ydata()) == [low, high]
            ax.plot([x - .10, x + .10], [low, low], color='#4c5685', linewidth=1.3)
            ax.plot([x - .10, x + .10], [high, high], color='#4c5685', linewidth=1.3)
            ax.annotate(f'{low:.3f}–{high:.3f}', (x, high), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(range(4), [f'L{r["layer_zero_based"]}' for r in geometry])
        ax.set_xlim(-.45, 3.45)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Physical decoder layer (zero based)', labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=.18)
        ax.text(.5, -.22, 'Min–max across all 18 main-DEV strata; four-root means', transform=ax.transAxes, ha='center', fontsize=9, color='#475569')
    fig.text(.085, .085, 'Geometry ranges cover every length × position × task stratum; they are not confidence intervals.', fontsize=10)
    fig.text(.085, .055, 'Residuals precede final loop RMSNorm. No SAE feature-index subtraction or causal/equivalence claim.', fontsize=10)
    fig.text(.085, .025, 'Original 168-case test sealed · Source aggregate: ad8a39d · Post hoc visualization; no new grading', fontsize=9, color='#475569')

    out = Path('reports/nanbeige_bpd_gym/figures')
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ('png', 'svg'):
        path = out / ('nan_ctx08_behavior_geometry.' + suffix)
        fig.savefig(path, dpi=180, facecolor='white', metadata={'Date': None} if suffix == 'svg' else {})
        files.append({'file': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    receipt = dict(source_commit=INPUT_COMMIT, source_file=INPUT_PATH, source_sha256=INPUT_SHA256,
                   source_manifest_sha256=SOURCE_SEAL, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                   CPU_numerical_threads=1, priority='BelowNormal on Windows',
                   scope='post hoc visualization of committed aggregates; zero model calls, new grades or formal comparisons',
                   n_DEV_source_families=4, n_correlated_DEV_contexts=288, displayed_accuracy=displayed,
                   displayed_physical_geometry_ranges=geometry, figure_files=files, sealed_test_accessed=False)
    (out / 'nan_ctx08_figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print({'status': 'plotted_committed_aggregates', 'figures': [v['file'] for v in files]})


if __name__ == '__main__':
    main()
