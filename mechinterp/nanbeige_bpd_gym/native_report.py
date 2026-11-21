"""Publish aggregate native BF16/NF4 results, geometry and sampling limits."""
import argparse
from pathlib import Path
from experiments.nan_gym05.common import read,write,sha


def report(runs,out):
    out.mkdir(parents=True,exist_ok=True)
    progress=read(out/'nan_gym05_progress.json');paired=read(runs/'paired_aggregate.json')
    assert not progress['partial'] and paired['status']=='passed' and len(progress['tower_diagnostics'])==2
    paired|={'state_site':'residual_after_layer','observed_states_are_before_loop_final_norm':True}
    write(out/'nan_gym05_paired_geometry.json',paired)
    memory={}
    for backend in ('bf16','nf4'):
        receipt=read(runs/backend/'receipt.json');samples=read(runs/backend/'memory_samples.json');windows=[]
        for sample in samples:
            row=sample['windows_process_gpu_memory']
            if row:windows.extend(row if isinstance(row,list) else [row])
        memory[backend]={k:receipt[k] for k in ('elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}
        memory[backend]|={'sampled_driver_global_peak_mib':max(s['driver_device_global_mib'][0] for s in samples if s['driver_device_global_mib']),
            'sampled_windows_process_dedicated_peak_bytes':max(s['DedicatedUsage'] for s in windows),
            'sampled_windows_process_shared_peak_bytes':max(s['SharedUsage'] for s in windows),'samples':len(samples),
            'receipt_sha256':sha(runs/backend/'receipt.json'),'memory_samples_sha256':sha(runs/backend/'memory_samples.json')}
    write(out/'nan_gym05_memory.json',{'measurements':memory,'scope':'CUDA per-process peaks plus sampled Windows process and device-global driver counters; sampling may miss transient residency/shared spikes'})
    by={b:{(r['suite'],r['condition']):r for r in rr if r['split']=='development'} for b,rr in progress['summaries'].items()}
    main={b:[r for r in rr.values() if r['condition']!='copy'] for b,rr in by.items()}
    totals={b:sum(r['correct'] for r in rr) for b,rr in main.items()}
    lines=['# NAN-GYM-05: complete native BF16/NF4 development pilot','',
        f"Both backends completed 322 contexts and 89 source-root groups. On the 105 DEV main contexts, BF16 scored **{totals['bf16']}/105** and NF4 **{totals['nf4']}/105**. Both scored **57/57** on DEV copy controls. All eight calibration-only SAE dictionaries were fitted, independently checked and probed. The original 168-case test stayed sealed.",'',
        'Registration: `18fb185dcda884473d51fe9c420654418c6a9f49`; source seal `96294903b1be52021725f2b839a7ad79fa635fd75aa4d5752cb3d53af0d132b0`; checkpoint `451ed48c3273ecef7ea8faaa43c31ce529763bb1`. [Frozen protocol](../../experiments/nan_gym05/PROTOCOL.md), [full aggregate grades and SAE groups](nan_gym05_progress.json), [paired geometry and discordances](nan_gym05_paired_geometry.json).','',
        '## Native task performance','',
        '| DEV main task | BF16 correct | NF4 correct | BF16 format | NF4 format |','|---|---:|---:|---:|---:|']
    for key,a in sorted(by['bf16'].items()):
        if key[1]=='copy':continue
        b=by['nf4'][key];n=a['records'];assert n==b['records']
        lines.append(f"| {key[0]} / {key[1]} | {a['correct']}/{n} | {b['correct']}/{n} | {a['format_valid']}/{n} | {b['format_valid']}/{n} |")
    lines+=['','Sudoku primary grading is the exact upstream reference: both backends also produced 1/3 valid givens-preserving solutions. ARC-1/2 are local synthetic grid microbenches. Storyworld verifies complete plans under the native modeled rival; both scored 0/16 despite valid formatting. Routing classifies eight environment categories; it is not source-benchmark QA accuracy. The bounded anchored parser does not repair answers or scan explanations.','',
        'Both pass the prespecified new one-hop pointer readout screen (16/16 main and 16/16 copy). Multi-hop performance remains weak. This nominates a source-dependent retrieval task for a new causal design; it does not pass the original failed algebraic competence gate or establish reasoning transfer.','',
        '## Paired recurrence geometry','',
        'States are physical residual_after_layer outputs before the end-of-loop final RMSNorm. They are not normalized loop-boundary states; a boundary transfer additionally requires the actual normalization map. Equal accuracy can also hide different successes: both backends score 2/16 at three hops, but only one success is shared.','',
        'The table includes all 16 DEV pointer roots and all four non-copy hop prompts per root. Values are equal-root means, with each phase paired at the identical first prompt prefix. NF4 relative distance is ||h_NF4-h_BF16|| / ||h_BF16||. Sparse feature indices from separately fitted dictionaries are not equated.','',
        '| Layer (zero based) | Phase 1 cosine | Phase 1 relative L2 | Phase 2 cosine | Phase 2 relative L2 | Loop-displacement cosine | Loop-displacement relative L2 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for layer in paired['layers']:
        states={g['loop']:g for g in layer['states'] if g['split']=='development' and g['suite']=='pointer' and not g['copy_control']}
        gap=next(g for g in layer['loop_displacements'] if g['split']=='development' and g['suite']=='pointer' and not g['copy_control'])
        a,b=states[1],states[2]
        lines.append(f"| {layer['layer_zero_based']} | {a['root_macro_cosine']:.6f} | {a['root_macro_NF4_minus_BF16_relative_l2']:.4f} | {b['root_macro_cosine']:.6f} | {b['root_macro_NF4_minus_BF16_relative_l2']:.4f} | {gap['root_macro_cosine']:.6f} | {gap['root_macro_NF4_minus_BF16_relative_l2']:.4f} |")
    lines+=['','![Matched native state and recurrence displacement](figures/nan_gym05_quantization_geometry.png)','',
        'Layer-21/channel-2279 remains a large phase-associated coordinate on these exposed DEV pointer prompts:']
    for backend in ('bf16','nf4'):
        probe=read(runs/'probes'/(backend+'_aggregate_probes.json'))
        groups=probe['layers'][-1]['groups'];values={g['loop']:g['root_macro_channel2279'] for g in groups if g['split']=='development' and g['suite']=='pointer' and not g['copy_control']}
        lines.append(f"{backend.upper()} phase 1 {values[1]:.3f}, phase 2 {values[2]:.3f}.")
    lines+=['','A dominant phase coordinate can inflate whole-vector cosine. High cosine and matching one-hop answers do not establish identical recurrence mechanisms. No channel intervention or causal mediation is measured in this pilot. All-suite physical drift, cross-dictionary reconstruction and both module-output SVD subspaces are retained in the paired JSON; SVD is a weight baseline, not trained VPD/BPD.','',
        '## SAE towers','',
        'The weight-only rank-8/16 SVD residual Frobenius ratios range from 0.9808 to 0.9961 across these matrices, so the residual retains most operator magnitude. These algebraic baselines do not substitute for learned VPD/BPD.','',
        'Each backend/layer has one 128-feature, top-4 dictionary shared by both loops, fitted on exactly 256 vectors from 32 CAL pointer roots. No DEV or copy-control vector was used in fitting or checkpoint selection.','',
        '| Layer | BF16 CAL normalized MSE | NF4 CAL normalized MSE | BF16 active / 128 | NF4 active / 128 |','|---|---:|---:|---:|---:|']
    fits={b:{r['layer_zero_based']:r['fit_receipt'] for r in d['layers']} for b,d in progress['tower_diagnostics'].items()}
    for layer in (5,11,17,21):
        a,b=fits['bf16'][layer],fits['nf4'][layer]
        lines.append(f"| {layer} | {a['normalized_CAL_MSE']:.6f} | {b['normalized_CAL_MSE']:.6f} | {a['features_ever_active_on_CAL']} | {b['features_ever_active_on_CAL']} |")
    lines+=['','Normalized MSE uses each dictionary’s own CAL scale; compare cross-backend reconstruction using the reported physical relative errors. Random initial dictionaries, all features and all registered source/loop groups were retained. These small dictionaries are observational bottlenecks, not evidence that their feature identities mediate correct reasoning.','',
        '## Memory and execution','',
        '| Backend | Elapsed minutes | CUDA allocated peak GiB | CUDA reserved peak GiB | Sampled process dedicated MiB | Sampled process shared MiB | Sampled driver global MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for b,m in memory.items():
        lines.append(f"| {b.upper()} | {m['elapsed_seconds']/60:.2f} | {m['peak_cuda_allocated_bytes']/2**30:.3f} | {m['peak_cuda_reserved_bytes']/2**30:.3f} | {m['sampled_windows_process_dedicated_peak_bytes']/2**20:.1f} | {m['sampled_windows_process_shared_peak_bytes']/2**20:.1f} | {m['sampled_driver_global_peak_mib']} |")
    lines+=['','The full 166,144 × 3,072 BF16 head and embeddings stayed on CPU; vocabulary chunks were 8192 rows. CUDA allocator peaks are per process; driver counters are sampled and can miss short spikes. Shared usage is reported rather than assumed zero. Device-global samples include small overlapping BF16 SAE fitting / random-weight engineering checks; native BF16 and NF4 large model workers never overlapped. Elapsed time includes loading, baselines and scoring, and is not an isolated throughput benchmark. [Memory aggregates](nan_gym05_memory.json).','',
        '## Evidence boundary and next stages','',
        'Unit: source-generator/root group, with prompts/hops/loops/features repeated within it; one checkpoint and two quantization conditions. Counts: 32 CAL roots, 57 DEV roots, 322 total contexts/backend. Independence across all fixture roots is not asserted. Primary causal contrast, physical dose, scientific causal-effect floor and matched-sham tolerance floor: not calibrated here. Planned and actual formal inferential comparisons: zero. Resolution/power: no causal or equivalence test. Readout exposure: development sources and BF16 results were visible during post-registration analysis-helper design. Causal controls: none applied; copy/random-dictionary/Haar controls address instrument diagnostics only. Verdict: suggestive observational development evidence.','',
        'Each backend audit checked 322 final answers, all 12,880 captured physical state/module vectors, source bindings, full-vocabulary greedy score tapes and effective-operator closure. Paired analysis matched 2576 residual states, eight SAE fits and all 16 module/rank subspace cells. Local routing text, per-record predictions, identifiers, tokens, states and codes remain outside publication.','',
        'Actual local Prime full-reply rollouts are a separate frozen condition (NAN-PRIME-06). NF4 is producing fresh model replies through the official CLI/null harness on the local GPU; BF16 is queued next. Their scores must not be pooled with native first-complete-prefix results. The native-kernel BPD engineering bridge now has genuine-framework CPU checks and random-model GPU gradient/closure checks. Trained Nanbeige VPD/BPD fits, reached-state causal routing and donor-specific behavioral transfer remain pending their own prospective protocols. Further causal evidence requires independent sham calibration and a new untouched readout, preserving the original protected 168 cases.','']
    (out/'NAN_GYM05_NATIVE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    plot(paired,out/'figures/nan_gym05_quantization_geometry.png')
    print({'report':'NAN_GYM05_NATIVE_REPORT.md','contexts_per_backend':322,'DEV_main_correct':totals,'paired_states':2576,'trained_SAE_dictionaries':8})


def plot(paired,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layers=[r['layer_zero_based'] for r in paired['layers']]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for phase,color in ((1,'#2777b9'),(2,'#dc7d25')):
        values=[next(g['root_macro_NF4_minus_BF16_relative_l2'] for g in r['states'] if g['split']=='development' and g['suite']=='pointer' and not g['copy_control'] and g['loop']==phase) for r in paired['layers']]
        axes[0].plot(layers,values,'o-',label=f'Loop {phase}',color=color)
    gaps=[next(g['root_macro_NF4_minus_BF16_relative_l2'] for g in r['loop_displacements'] if g['split']=='development' and g['suite']=='pointer' and not g['copy_control']) for r in paired['layers']]
    axes[1].plot(layers,gaps,'o-',color='#7762a8')
    for ax,title in zip(axes,('Matched physical state difference','Loop-displacement difference')):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Layer (zero based)');ax.set_xticks(layers);ax.set_ylim(bottom=0)
        ax.set_ylabel('NF4 − BF16 relative L2');ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('DEV pointer prompts: 16 roots × 4 hops; equal-root means',fontsize=11)
    target.parent.mkdir(parents=True,exist_ok=True);fig.savefig(target,dpi=180);fig.savefig(target.with_suffix('.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
