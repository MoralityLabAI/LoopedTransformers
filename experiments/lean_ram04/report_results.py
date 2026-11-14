"""Render measured RAM-04 results with authored support and real proof counts separate."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=json.loads((a.analysis/'metrics.json').read_text(encoding='utf8'));t=d['training'];e=d['evaluation'];data=t['data']
    rows=e['summaries'];get=lambda budget,policy:next(r for r in rows if r['experiment']==budget and r['model']==policy)
    extra=set().union(*(set(get('primary',m)['model_only_vs_native']) for m in ('loop32','loop64','flat64')))
    lines=['# RAM-04 measured results','',
      'RAM-04 addresses exact/rw/rcases support using typed local contexts, explicit authored curricula, family/label-balanced sampling and position-sensitive token pooling. RAM-02/03 and their 36 checkpoints remain preserved.','',
      f"The primary comparison produced {len(extra)} distinct new prospective declarations certified by a neural policy beyond matched native search. Only complete original MathCan theorem compiles with independent exact-source replay count below; authored tasks and scratch closures never count.",'',
      '| Budget | Policy | Certified / 12 | Model-only vs native | Native-only misses | Logical evaluations | Lean processes | Seconds per certificate |',
      '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for budget,policies in [('primary',('native','frequency','loop32','loop64','flat64')),('secondary',('native','loop64'))]:
        for name in policies:
            r=get(budget,name);cost=r['seconds_per_independent_certificate'];cost='N/A' if cost is None else f'{cost:.3f}'
            lines.append(f"| {budget} | {name} | {r['certified']} | {len(r['model_only_vs_native'])} | {len(r['native_only_vs_model'])} | {r['logical_evaluations']} | {r['actual_lean_processes']} | {cost} |")
    lines += ['','Primary: 30 seconds, 48 logical prefix evaluations, 8 transitions per path. Secondary: 60 seconds, 128 evaluations, 16 transitions. Both use at most 8 goals and 3 seconds per request. Startup (maximum 10 seconds), final compilation and independent replay are charged. Actual Lean processes and logical evaluations are different quantities.','']
    for budget,policies in [('primary',('frequency','loop32','loop64','flat64')),('secondary',('loop64',))]:
        for name in policies:
            r=get(budget,name);lines += [f"**{budget}/{name}:** extra vs native: "+(', '.join(r['model_only_vs_native']) or 'none')+'. Native-only: '+(', '.join(r['native_only_vs_model']) or 'none')+'.','']
    lines += ['The frequency and Flat64 controls are included to prevent a rank-order improvement being mistaken for evidence that recurrence itself is necessary. These are twelve new prospective targets, not the published FC-13 or RAM-03 suites; their percentages cannot be compared as if the tasks were identical.','',
      '## Supervision support','',
      f"The complete new corpus has {d['data_audit']['checked']:,} compiler-observed transitions at {sum(r['states'] for r in d['collection']['roots'])} states from 27 natural TRAIN roots and 52 authored roots. All observations were independently replayed, including parent/after complete goals and typed local metadata, with clean Lean exits.",'',
      '| Split | Theorem roots | Unique states | Multi-goal states | Rows | Excluded overlap rows |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split in ('train','development'):
        r=data[split];lines.append(f"| {split} | {r['theorems']} | {r['states']} | {r['multi_input_states']} | {r['rows']} | {data['excluded_training_rows'] if split=='train' else 0} |")
    lines += ['','| Split / family | Natural positives | Authored positives | Total positives | Negatives |','| --- | ---: | ---: | ---: | ---: |']
    for split in ('train','development'):
        for family in ('exact_local','rw_local','rcases_local'):
            r=data[split];f=r['target_families'][family];lines.append(f"| {split} / {family} | {r['origins']['natural']['family_positive'][family]} | {r['origins']['authored']['family_positive'][family]} | {f['positive']} | {f['negative']} |")
    lines += ['',
      'Authored sources are statement-only teaching tasks with compiler-probed outcomes. Complete template groups are held out. Their support is not evidence of natural theorem coverage or independent sample count. The eight-positive/eight-negative gate passed for all three families in both splits after removing exact cross-split state overlap. No original reference command or teacher prefix was executed for collection, and the old 961/4,973 records remain unchanged.','',
      'Each effective batch contains 8 positive and 8 negative rows from each target family plus 8 single- and 8 multi-goal rows. Sampling is from optimization data only. Model inputs exclude theorem/file/template IDs, source origin, action prefixes and future outcomes. Referenced local types and pending contexts are visible features. Fixed position gates preserve argument order; hashing and truncation remain lossy.','',
      '## Nine fits and development selection','',
      f"Nine fits completed {d['total_updates']:,} optimizer updates in {d['fit_seconds']:.3f} summed fit seconds, including validation and checkpoint work. Losses stayed finite; last-checkpoint CPU reloads were exact. All 18 best/last optimizer and RNG checkpoints are retained.",'',
      '| Model | Seed | Parameters | Updates | Selected step | Train loss | Dev loss | Cached dev top1 usable |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in t['models']:lines.append(f"| {r['name']} | {r['seed']} | {r['parameters']:,} | {r['updates']} | {r['best_step']} | {r['train']['selection_loss']:.4f} | {r['dev']['selection_loss']:.4f} | {r['dev']['top1_usable']}/{r['dev']['states']} |")
    lines += ['',
      'Selection uses the mean of five dev weighted-CE + .2 SmoothL1 losses: single, multi, exact, rw, rcases. Training-only class weights; AdamW .002/.01, clip 1, microbatch 16 x4, at most 1,600 updates, validation every 100, patience 4. Seed and checkpoint selection use no prospective outcomes. Selected: '+', '.join(f"{k} seed {v['seed']} step {v['best_step']}" for k,v in t['selected'].items())+'.','',
      '| Family | Looped32 dev positive recall | Looped64 | Flat64 |','| --- | ---: | ---: | ---: |']
    for family in ('exact_local','rw_local','rcases_local','induction_list'):
        values=[]
        for name in ('loop32','loop64','flat64'):
            v=t['selected'][name]['dev']['families'][family]['positive_recall'];values.append('N/A' if v is None else f'{100*v:.1f}%')
        lines.append('| '+family+' | '+' | '.join(values)+' |')
    lines += ['','These are cached positive-outcome recalls, not proof success rates. Full per-class precision/recall, confusion matrices, goal-delta error, per-origin family recalls and all seed metrics are in receipts/metrics.json. Authored support can dominate this proxy; the complete prospective proof comparison determines gains.','',
      '## Search limits and certificates','',
      '| Budget / policy | Stop reasons | Multi-goal transitions | Goal-cap events | Path-cap events | Final compile/replay seconds |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in d['search_diagnostics']:
        lines.append(f"| {r['experiment']} / {r['policy']} | {r['reasons']} | {r['events'].get('multi_goal_transitions',0)} | {r['events'].get('goal_cap',0)} | {r['events'].get('transition_budget',0)} | {r['verification_seconds']:.3f} |")
    lines += ['',f"The final audit checked {d['trajectories']} scored trajectories, {d['certified_instances']} successful policy/theorem instances and {len(d['unique_theorems'])} distinct original theorems. Duplicate policies and source variants are not extra theorems. All accepted sources excluded ProbeServer/sorry, compiled independently and passed explicit propext/Classical.choice/Quot.sound gates with byte-identical replay. The artifact manifest fingerprints {d['artifact_files']:,} files ({d['artifact_bytes']:,} bytes). Deduplicated complete scripts and the policy/theorem/action/hash index are under certificates/.",'']
    for identity in sorted(extra):
        example=next(r for r in d['certificates'] if r['experiment']=='primary' and r['theorem']==identity and r['policy'] in ('loop32','loop64','flat64'))
        lines += [f"Extra primary certificate: `{identity}` ({example['policy']}).",'','```lean',*example['actions'],'```','',f"Source SHA256: `{example['source_sha256']}`.",'']
    lines += ['Unsolved cases describe a bounded search and fixed grammar, not mathematical impossibility. Pool exhaustion, goal/path caps and evaluation limits remain visible in the receipts. No scratch closure alone receives credit.','',
      '## Hardware, caps and implementation history','',
      '| Calibration | Timed updates | Seconds | Updates/s |','| --- | ---: | ---: | ---: |']
    for r in t['calibration']['trials']:lines.append(f"| {r['device']} | {r['updates']} | {r['seconds']:.4f} | {r['updates_per_second']:.2f} |")
    lines += ['',f"Calibration selected {t['calibration']['selected_device']} using the frozen 1.15x CUDA speedup threshold. CUDA/CPU ratio: {t['calibration']['cuda_speedup']}. This short measurement is not a repeated hardware benchmark.",'',
      f"Peak sampled process-tree RSS: training/calibration {t['resources']['peaks']['tree_rss_bytes']/1024**3:.3f} GiB; evaluation {e['resources']['peaks']['tree_rss_bytes']/1024**3:.3f} GiB. Exact torch CUDA calibration peak: {max(r['peak_cuda_allocated_bytes'] for r in t['calibration']['trials'])/1024**2:.2f} MiB. Resource aborts: training {t['resources']['abort']}, evaluation {e['resources']['abort']}.",'',
      'Caps remained 6 GiB process-tree RSS, 2.5 GiB CUDA allocation, 2 GiB available RAM/commit and 512 MiB dedicated GPU headroom; .1-second memory monitoring and 2-second nvidia-smi sampling. Python 3.13.7/PyTorch 2.6.0+cu124/psutil 7.1.3; Lean 4.35.0. No upgrades, torch.compile or GPU compilation.','',
      'Pre-collection broad Lean imports repeatedly stalled. A byte-identical toolchain copy was recovered from the preserved archive; narrower registered tactic imports passed. A first collection attempt failed clean shutdown under asynchronous single-worker probing. Synchronous elaboration with two Lean workers passed the real-root pilot and full search/replay smoke before the fresh completed dataset. No root cause for the broad-import stall is claimed. Preflight/failed collection artifacts remain preserved, and no prospective goal was observed to guide these repairs.','',
      'A separate pre-fitting review added position gates because plain token means discard exact-argument order. This amendment was tested and committed after collection began and before fitting or prospective observation. The final implementation and development-selected weight hashes were frozen before reading prospective goals or compiling references. The two amendments explicitly document the timeline.','',
      '## Limits and reproduction','',
      'The prospective set is only twelve source-order-selected declarations from four modules; it is not a random sample. Authored templates and descendant states are correlated. The study combines grammar, representation, sampling and runtime changes and cannot isolate their causal effects. A gain shared by frequency/Flat64 does not establish a recurrence advantage. Per-family natural support, not only overall synthetic support, matters for transfer. Syntactic deduplication is not semantic equivalence; full-context truncation and token collisions remain.','',
      'Run run_windows.ps1 or the stage commands in README.md to reproduce into a fresh directory on D:. All 18 checkpoint paths and SHA256s are in receipts/checkpoints.json. Complete JSONL labels, metadata, mailbox messages, logs, optimizer/RNG states and the artifact manifest remain under D:\\Research Engine\\RAM-04\\LoopedTransformers\\experiments\\lean_ram04\\runs. RAM-02/03 are preserved. The separate feature PR is never merged automatically.','']
    a.out.write_text('\n'.join(lines),encoding='utf8',newline='\n')
if __name__=='__main__':main()
