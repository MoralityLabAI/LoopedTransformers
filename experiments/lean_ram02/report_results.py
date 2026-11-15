"""Render completed measured receipts into the repository's RESULTS.md."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    data=json.loads((args.analysis/'metrics.json').read_text(encoding='utf8'))
    audit=json.loads(args.audit.read_text(encoding='utf8'))
    models=data['training_models']; policies=data['regression_models'];cal=data['calibration']
    native=policies[0]
    lines=['# RAM-02 measured results | 2026-11-14 (Asia/Taipei)','',
      'The local campaign completed all six required recurrent fits, then three flat controls. '
      f"All nine used finite FP32 losses and passed exact CPU checkpoint reload. Total: {data['total_optimizer_steps']:,} optimizer updates and {data['sum_fit_seconds']:.3f} summed fit seconds, including in-fit validation/checkpoint work.",'',
      f"The post-hoc FC-13 comparison produced {data['certified_instances']} independently replayed policy/theorem instances across eleven policies, covering {len(data['distinct_certified_theorems'])} distinct original theorems. Native certified {native['certified']}/16.",'',
      '## Hardware and device choice','',
      'Native Windows, RTX 3050 Laptop GPU (4096 MiB, driver610.62), approximately24 GiB installed system RAM. Python3.13.7 and existing PyTorch2.6.0+cu124; two CPU threads. Existing environments were not upgraded. All repositories, dependency launcher, datasets and checkpoints were stored on D: after C: was found nearly full.','',
      '| Device | Measured updates | Seconds | Updates/s |','| --- | ---: | ---: | ---: |']
    for trial in cal['trials']:
        lines.append(f"| {trial['device']} | {trial['updates']} | {trial['seconds']:.4f} | {trial['updates_per_second']:.2f} |")
    tr=data['training_resources'];er=data['regression_resources']
    lines += ['',f"CUDA/CPU speed ratio was {cal['cuda_speedup']:.3f}; the frozen1.15x threshold selected **{cal['selected_device']}**. Both trials used the identical198,148-parameter four-pass model, microbatch16/accumulation4, ten warmup and forty measured updates. This is one short calibration, not a repeated hardware benchmark.",'',
      f"Training/calibration peak sampled process-tree RSS: {tr['peaks']['tree_rss_bytes']/1024**3:.3f} GiB. Exact torch CUDA calibration peak: {max(t['cuda_peak_allocated_bytes'] for t in cal['trials'])/1024**2:.2f} MiB; peak sampled dedicated GPU use: {tr['peaks']['dedicated_used_bytes']/1024**2:.1f} MiB. Minimum available RAM: {tr['minimum_available_ram_bytes']/1024**3:.2f} GiB. Lean build/regression peak tree RSS: {er['peaks']['tree_rss_bytes']/1024**3:.3f} GiB. No resource abort occurred.",'',
      'Caps were6 GiB tree RSS and2.5 GiB allocated VRAM, with2 GiB available RAM/commit and512 MiB dedicated GPU headroom floors. Monitoring includes all child Lean processes. RSS and hardware peaks are sampled; torch separately reports its allocator peak. FP16 and torch.compile were not used.','',
      '## Frozen input and RAM-01 reproduction','',
      f"The original training ZIP and the original RAM-01 model ZIP matched their registered SHA256 values. All{len(audit['checkpoints'])} RAM-01 checkpoint byte hashes were verified, and all eight original development top1 executable counts reproduced with the existing2.6 runtime. The51-state /961-probe corpus and39-state/766-row training versus12-state/195-row development split were unchanged. Native cached top1 executable was3/12; original training-only family frequency was6/12.",'',
      'Only actual goal/context text, candidate script and family enter features. No original reference proof command, future action or held-out theorem label enters training. Compiler labels were reused without regeneration or relabeling. The independently recompiled FC-13 source-goal file matched the frozen SHA256 after LF normalization.','',
      '## Selected development checkpoints','',
      '| Model | Seed | Parameters | Updates run | Best update | Train CE | Dev CE | Cached top1 usable | Fit seconds |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in models:
        lines.append(f"| {r['name']} | {r['seed']} | {r['parameters']:,} | {r['optimizer_steps']} | {r['best_step']} | {r['train']['loss']:.4f} | {r['dev']['loss']:.4f} | {r['dev']['choices']['multigoal']['top1_usable']}/12 | {r['elapsed_seconds']:.3f} |")
    lines += ['',
      'Each fit used AdamW(.002,.01), clipping1.0 and effective batch64. The loss normalizes accumulated weighted CE over the entire effective batch. Maximum2,000 updates, validation every100; stop after four validations without a0.0001 development CE improvement. The best checkpoint is chosen exclusively by development CE. Last and best optimizer/RNG checkpoints are preserved. No all51-state refit was performed.','',
      'Cached usable means the selected fixed probe recorded progress, including branching or scratch closure. It is not a complete proof. Development contains zero positive multi-goal outcomes. Best-checkpoint counts vary substantially across seeds; no NaN/Inf or hardware memory failure occurred. Flat64 has173,252 parameters and is a useful control, but is not exactly parameter-matched to loop64.','',
      'The frozen development grammar has at least one progressing/executable candidate for8/12 states and a scratch-closing candidate for5/12. Four states therefore have no observed progressing action in this fixed candidate set.','',
      '| Family | Positive train probes | Positive dev probes | Dev positive recall, loop32 seeds17/29/43 | Dev positive recall, loop64 seeds17/29/43 | Dev positive recall, flat64 seeds17/29/43 |',
      '| --- | ---: | ---: | --- | --- | --- |']
    for f in ('exact_local','rw_local','rcases_local','induction_list'):
        first=models[0]
        cells=[]
        for name in ('loop32','loop64','flat64'):
            vals=[r['dev']['per_family'][f]['positive_recall'] for r in models if r['name']==name]
            cells.append('/'.join('N/A' if v is None else f'{100*v:.0f}%' for v in vals))
        lines.append(f"| {f} | {first['train']['per_family'][f]['positives']} | {first['dev']['per_family'][f]['positives']} | {' | '.join(cells)} |")
    lines += ['',
      'Positive recall here is binary predicted usability; class-wise precision/recall, supports, full confusion matrices, train/dev losses and every seed are in `receipts/metrics.json`. Zero-support recall is N/A in analysis; raw training receipts retain their zero-division=0 convention. Exact and rcases have only one successful training probe each. Development therefore cannot validate generalization for those families or for induction/branching.','',
      '## Complete original-theorem certification (post-hoc FC-13)','',
      '| Policy | Certified /16 | Search Lean calls | Replay calls | Search seconds | Search+replay seconds/certificate | Model-only vs native | Native-only misses |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in policies:
        cost=r['total_seconds_per_independently_certified_proof']
        cost_text='N/A' if cost is None else f'{cost:.3f}'
        lines.append(f"| {r['model']} | {r['certified']} | {r['actual_lean_calls']} | {r['replay_lean_calls']} | {r['search_seconds']:.3f} | {cost_text} | {len(r['model_only_vs_native'])} | {len(r['native_only_vs_model'])} |")
    lines += ['', '| Original theorem | Policies certifying it /11 |', '| --- | ---: |']
    for theorem in data['distinct_certified_theorems']:
        count=sum(theorem in r['solved'] for r in policies)
        lines.append(f'| {theorem} | {count} |')
    lines += ['',
      'All policies used the unchanged FC-12 candidate grammar,20-second search deadline,12 actual Lean calls including separate final compilation, and at most three successful transitions per path, including closure. All branches must close. No proof completed after the deadline received credit. Deadline termination has small process-kill overhead, recorded in raw trajectories. Fresh independent replay is subsequent verification and cannot improve the score; its time/calls are reported separately. In this run, every accepted instance also remained within12 calls when its replay call was included.','',
      'Every accepted theorem was separately compiled in its original preceding declaration/import context. Printed dependencies allowed only propext, Classical.choice and Quot.sound. The exact final source bytes were freshly replayed and matched SHA256. Accepted sources are deduplicated by byte hash under `certificates/`; `certificates/index.json` maps every model/theorem/script to its certificate. Original source-oracle proofs and synthetic smoke tests are excluded from these counts.','',
      'All16 reference proofs compiled in the separate source oracle. This is an already published FC-13 regression, not a pristine holdout. New theorem families were not evaluated. Comparison timing is one sequential local run per policy; scheduling/cache variation precludes a robust speed claim. Per-instance fit+search+replay costs are also in the JSON receipt. Cost per model-only theorem is undefined when no new theorem was solved.','',
      '### Family and branching observations','',
      '| Policy | Accepted multigoal transitions | Certified proofs using exact | Using rw | Using rcases | Using induction |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in policies:
        counts=[r['per_family'].get(f,{}).get('certified_proofs_using_family',0) for f in ('exact_local','rw_local','rcases_local','induction_list')]
        lines.append(f"| {r['model']} | {r['successful_multigoal_transitions']} | {' | '.join(map(str,counts))} |")
    lines += ['',
      'The prototype stores complete ordered goal lists, remaining counts, goal hashes and canonical visible-state signatures, and preserves siblings after induction/rcases. The real Lean smoke suite exercises complete constructor and induction branching, plus rcases continuation; the custom-axiom negative compiles successfully and is rejected by the release gate. Unit tests cover partial-branch rejection, deduplication, deadline/final-call accounting and candidate injection.','',
      'Training remains the original four-outcome supervision on single-goal contexts. Frozen probe records contain outcome classes and remaining counts, but not complete post-transition branch text. No extra branch-state training labels were fabricated. The extension changes state tracking/search and training/selection duration, not the frozen supervision corpus.','',
      'Failures are preserved by theorem and policy with every source/log hash. Hard-deadline failures do not establish grammar impossibility: not every candidate was tested. Multi-goal trajectories that remain open demonstrate branch/budget pressure, while the three-transition cap prevents unrestricted continuation. Frozen cached probes identify states without a usable candidate in the registered grammar; no new outcome labels were generated to repair expressivity.','',
      '## Additional recurrent passes versus compute','',
      'Inference-only ablation on each selected four-pass checkpoint, using pre-encoded development batches, five warmups and three timing trials of twenty repetitions. CPU forward time excludes tokenization and metric bookkeeping. These are not separately trained one/two/eight-pass models, and no proof-coverage conclusion is drawn from cached scores.','',
      '| Model/seed | Passes | Cached top1 usable /12 | Dev CE | Median forward ms /195 rows |',
      '| --- | ---: | ---: | ---: | ---: |']
    for r in models:
        for a in r['inference_only_pass_ablations']:
            lines.append(f"| {r['name']}/{r['seed']} | {a['passes']} | {a['metrics']['choices']['multigoal']['top1_usable']} | {a['metrics']['loss']:.4f} | {a['median_forward_seconds_per_dev_set']*1000:.3f} |")
    recurrent=[r for r in models if r['inference_only_pass_ablations']]
    for fewer,more in ((2,4),(4,8)):
        changes=[];ratios=[]
        for r in recurrent:
            ab={a['passes']:a for a in r['inference_only_pass_ablations']}
            changes.append(ab[more]['metrics']['choices']['multigoal']['top1_usable']-ab[fewer]['metrics']['choices']['multigoal']['top1_usable'])
            ratios.append(ab[more]['median_forward_seconds_per_dev_set']/ab[fewer]['median_forward_seconds_per_dev_set'])
        lines += ['',f"{more} versus {fewer} inference passes: cached top1 improved for {sum(c>0 for c in changes)}/6 checkpoints, declined for {sum(c<0 for c in changes)}/6, and tied for {sum(c==0 for c in changes)}/6. Median forward-cost ratio was {statistics.median(ratios):.3f}x. These are fixed-checkpoint development diagnostics, not additional certified proof comparisons."]
    lines += ['',
      '## Preservation and reproducibility','',
      'The campaign runs are under `D:\\Research Engine\\RAM-02\\LoopedTransformers\\experiments\\lean_ram02\\runs\\`. `campaign-01` contains all18 best/last checkpoints and complete training receipts; `fc13-regression-01` contains all176 theorem searches, compiler sources/logs and independent replays. Full hash manifests and checkpoint locations accompany the output deliverables. Large artifacts remain gitignored. Previous FC-12/FC-13/RAM-01 results and checkpoints were not changed.','',
      'Training and protocol freeze: local commit1d86ea5. Search freeze before regression:0b44a15. Subsequent implementation additions provide memory backoff/recovery, portable checkpoint resolution, evidence audits and reporting; they do not change the recorded training tensors or frozen scoring policy. No memory backoff was needed in the measured campaign. The initial CLI encoding typo was corrected before calibration/fitting and before any checkpoint existed.','',
      'Commands and environment setup are in README.md. `reproduce_local.ps1` reuses this machine’s installed environment and creates a new timestamped output tree. `audit_ram01.py`, `smoke_branches.py`, `analyze.py` and `report_results.py` reproduce the audits and summaries. No merge was performed.','']
    text='\n'.join(lines).replace('D:\\\\Research Engine\\\\RAM-02\\\\LoopedTransformers\\\\experiments\\\\lean_ram02\\\\runs\\\\','D:\\Research Engine\\RAM-02\\LoopedTransformers\\experiments\\lean_ram02\\runs\\')
    replacements={'driver610.62':'driver 610.62','approximately24':'approximately 24',
      'Python3.13.7':'Python 3.13.7','PyTorch2.6.0':'PyTorch 2.6.0','frozen1.15x':'frozen 1.15x',
      'identical198,148':'identical 198,148','microbatch16':'microbatch 16','accumulation4':'accumulation 4',
      'were6 GiB':'were 6 GiB','and2.5 GiB':'and 2.5 GiB','with2 GiB':'with 2 GiB','and512 MiB':'and 512 MiB',
      'All16':'All 16','all16':'all 16','All18':'All 18','all18':'all 18','all176':'all 176',
      'The51':'The 51','and39-state':'and 39-state','versus12-state':'versus 12-state','top1 executable was3':'top1 executable was 3',
      'frequency was6':'frequency was 6','existing2.6':'existing 2.6','for8/12':'for 8/12','for5/12':'for 5/12','seeds17':'seeds 17','clipping1.0':'clipping 1.0','batch64':'batch 64','Maximum2,000':'Maximum 2,000','every100':'every 100',
      'a0.0001':'a 0.0001','all51-state':'all 51-state','has173,252':'has 173,252',
      'grammar,20-second':'grammar, 20-second','deadline,12':'deadline, 12','within12':'within 12',
      'commit1d86ea5':'commit 1d86ea5','regression:0b44a15':'regression: 0b44a15'}
    for old,new in replacements.items():text=text.replace(old,new)
    args.out.write_text(text,encoding='utf8',newline='\n')

if __name__=='__main__':main()
