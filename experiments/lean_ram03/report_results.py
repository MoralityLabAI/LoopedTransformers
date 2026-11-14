"""Render audited RAM-03 metrics without conflating historical/native gains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=json.loads((a.analysis/'metrics.json').read_text(encoding='utf8'))
    t=d['training'];e=d['evaluation'];data=t['data'];rows=e['summaries']
    def find(experiment,suite,model):return next(r for r in rows if (r['experiment'],r['suite'],r['model'])==(experiment,suite,model))
    def counts(experiment,suite,names):
        return ', '.join(f"{name} {find(experiment,suite,name)['certified']}/{find(experiment,suite,name)['cases']}" for name in names)
    lines=['# RAM-03 measured results | 2026-11-14 (Asia/Taipei)','',
      'RAM-03 tested genuine branch-state supervision, a goal-count auxiliary head and a persistent Lean search backend. RAM-02 remains preserved. All coverage below requires complete original-theorem compilation plus independent exact-source replay.','',
      'Primary prospective coverage: '+counts('primary','prospective',('native','loop32','loop64','flat64'))+'. Primary post-hoc FC-13 coverage: '+counts('primary','posthoc_fc13',('native','frequency','loop32','loop64','flat64'))+'.','',
      'Secondary prospective coverage: '+counts('secondary','prospective',('native','loop64'))+'. Secondary post-hoc FC-13 coverage: '+counts('secondary','posthoc_fc13',('native','loop64'))+'. The matched model-only and native-only sets below determine the scope of every gain.','',
      '## Coverage against contemporary native search','',
      '| Budget | Suite | Policy | Certified | Model-only vs native | Native-only misses | Logical evaluations | Lean processes | Seconds / certificate |',
      '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for experiment in ('primary','secondary'):
        for suite in ('prospective','posthoc_fc13'):
            order=('native','frequency','loop32','loop64','flat64') if experiment=='primary' else ('native','loop64')
            for name in order:
                r=find(experiment,suite,name);cost=r['seconds_per_independent_certificate']
                text='N/A' if cost is None else f'{cost:.3f}'
                lines.append(f"| {experiment} | {suite} | {name} | {r['certified']}/{r['cases']} | {len(r['model_only_vs_native'])} | {len(r['native_only_vs_model'])} | {r['logical_evaluations']} | {r['actual_lean_processes']} | {text} |")
    lines += ['',
      'Primary uses30 seconds,48 logical evaluations and8 successful transitions per path. Secondary uses60 seconds,128 evaluations and16 transitions. Both budgets were registered before scoring. Every policy starts a fresh persistent probe process, with startup charged. Final compile and replay are separate processes charged to the same deadline. All sibling goals must close. Actual process counts and logical evaluations are different measures.','']
    for experiment in ('primary','secondary'):
        for suite in ('prospective','posthoc_fc13'):
            models=('loop32','loop64','flat64') if experiment=='primary' else ('loop64',)
            for name in models:
                r=find(experiment,suite,name)
                lines += [f"**{experiment}/{suite}/{name}:** model-only declarations: "+(', '.join(r['model_only_vs_native']) or 'none')+'. Native-only declarations: '+(', '.join(r['native_only_vs_model']) or 'none')+'.','']
    lines += ['RAM-02 native coverage on the published FC-13 suite was6/16. Changes in grammar, backend and budget mean an increase over that historical number is a system-level result, not evidence that a learned ranker beats native search. The matched RAM-03 native rows above determine learned-policy gains.','',
      '## Search limits observed','',
      '| Budget / suite / policy | Evaluation-budget stops | Deadline stops | Path-cap events | Goal-cap events | Multi-goal transitions | Unknown requests | Final compile + replay seconds |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in d['search_diagnostics']:
        reasons=r['reasons'];deadlines=sum(v for k,v in reasons.items() if 'deadline' in k or 'budget' in k and k!='evaluation_budget' and k!='transition_budget')
        lines.append(f"| {r['experiment']} / {r['suite']} / {r['model']} | {reasons.get('evaluation_budget',0)} | {deadlines} | {r.get('transition_budget',0)} | {r.get('goal_cap',0)} | {r.get('multi_goal_transitions',0)} | {r.get('unknown_requests',0)} | {r['verification_seconds']:.3f} |")
    lines += ['',
      'Stops are per theorem; path/goal caps and transitions are search events and can repeat within one theorem. Unsolved cases are bounded-search failures, not impossibility results. A wall/verification budget stop is recorded without treating an incomplete compile as a proof. Full per-family attempted/progress/branch/reduction counts are in metrics.json; no timeout is used as a fabricated negative label.','',
      '## Certified branching example','']
    example=next((r for r in d['certificates'] if r['experiment']=='primary' and r['suite']=='prospective' and r['model']=='native' and any(s.startswith('induction ') for s in r['actions'])),None)
    if example:
        lines += [f"Native search independently certified `{example['theorem']}` with the following complete action sequence, including continuation through multiple goals. The model policies receive the same branching grammar. This demonstrates functioning branch continuation; it is not a learned-only result.",'','```lean',*example['actions'],'```','']
    else:lines += ['No primary prospective native induction certificate was found.','']
    lines += [
      '## Genuine branch supervision','',
      f"The completed corpus contains {d['data_audit']['checked']:,} compiler-observed action transitions from {sum(r['states'] for r in d['collection']['roots'])} visible states at 27 original TRAIN theorem roots. A fresh Lean process independently replayed every observation before fitting. Acceptance and canonical complete-goal-list results agreed with zero discrepancies.",'',
      '| Split | Theorems | States | Multi-goal states | Candidate rows | Multi-goal input rows |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ('train','development'):
        r=data[name];lines.append(f"| {name} | {r['theorems']} | {r['states']} | {r['multi_input_states']} | {r['rows']} | {r['multi_input_rows']} |")
    lines += ['',
      'All descendants of five named development theorems remain outside optimization. Exact cross-split state hashes are excluded from training; this run excluded'+str(data['excluded_training_rows'])+' rows. The original51-state/961-probe corpus was not overwritten or relabeled. New collection starts at theorem roots and never executes the original target reference proof or a teacher prefix.','',
      'Outcome labels are failed/unchanged, reduced goal count, other observed progress (including branch creation), and scratch all-goal closure. The auxiliary target is clipped net goal-count decrease. Only current active context/goal, pending conclusions, observed goal count, candidate and family are features. Future states, file/theorem IDs, original commands and action-prefix metadata are excluded and tested. Half of every effective training batch is drawn from actual multi-goal inputs.','',
      '## Training, seed stability and selection','',
      f"Nine fits completed {d['total_training_updates']:,} updates in {d['summed_fit_seconds']:.3f} summed fit seconds, including validation/checkpoint work. All losses remained finite and saved last checkpoints passed exact CPU output reload. The complete last/best optimizer/RNG checkpoints remain in gitignored storage.",'',
      '| Architecture | Seed | Parameters | Updates run | Selected update | Train selection loss | Dev selection loss | Dev top1 usable |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in t['models']:
        lines.append(f"| {r['name']} | {r['seed']} | {r['parameters']:,} | {r['updates']} | {r['best_step']} | {r['train']['selection_loss']:.4f} | {r['dev']['selection_loss']:.4f} | {r['dev']['top1_usable']}/{r['dev']['states']} |")
    lines += ['',
      'Selection loss balances single/multi-goal development weighted CE plus0.2 SmoothL1 goal-delta loss. Training uses AdamW(.002,.01), gradient cap1.0, microbatch16/accumulation4, at most1,600 updates, validation every100 and patience4. Class weights use training counts only. Selected seeds/checkpoints were fixed before any evaluation: '+', '.join(f"{name} seed{r['seed']} update{r['best_step']}" for name,r in t['selected'].items())+'.','',
      'The early selections and worsening later development loss are retained as evidence of overfitting/seed sensitivity. Cached action usability is not theorem certification. Per-class precision/recall, confusion matrices, delta MAE, branch/single losses and all family counts are in receipts/metrics.json.','',
      '| Family | Positive training probes | Positive development probes | Selected loop32 dev positive recall | Selected loop64 recall | Flat64 recall |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for family in ('exact_local','rw_local','rcases_local','induction_list','unfold_simp','all_simp'):
        r=t['models'][0];values=[]
        for name in ('loop32','loop64','flat64'):
            value=t['selected'][name]['dev']['families'][family]['positive_recall'];values.append('N/A' if value is None else f'{100*value:.1f}%')
        lines.append(f"| {family} | {r['train']['families'][family]['positive']} | {r['dev']['families'][family]['positive']} | {' | '.join(values)} |")
    lines += ['',
      'Branch inputs and positive induction outcomes now occur in both splits. Coverage is still uneven: exact_local has no positive probes, and development contains no positive rw_local or rcases_local examples. Their unavailable recalls are reported as N/A, not zero or improvement. The fixed collection limits were not enlarged after these gaps were measured.','',
      '## Hardware and resource enforcement','',
      'Native Windows, RTX3050 Laptop GPU with4096 MiB, existing Python3.13.7/PyTorch2.6.0+cu124/psutil7.1.3, two CPU threads. No dependency upgrades, FP16 or torch.compile.','',
      '| Calibration device | Timed updates | Seconds | Updates/s |',
      '| --- | ---: | ---: | ---: |']
    for r in t['calibration']['trials']:lines.append(f"| {r['device']} | {r['updates']} | {r['seconds']:.4f} | {r['updates_per_second']:.2f} |")
    lines += ['',f"The identical goal-aware loop64 calibration selected {t['calibration']['selected_device']} at the frozen1.15x CUDA speedup threshold. CUDA/CPU ratio was {t['calibration']['cuda_speedup']:.3f}. This was one short calibration, not a robust repeated hardware speed comparison.",'',
      f"Sampled peak process-tree RSS was {t['resources']['peaks']['tree_rss_bytes']/1024**3:.3f} GiB during training/calibration and {e['resources']['peaks']['tree_rss_bytes']/1024**3:.3f} GiB during proof evaluation. Exact torch CUDA calibration peak was {max(r['peak_cuda_allocated_bytes'] for r in t['calibration']['trials'])/1024**2:.2f} MiB. No resource abort occurred.",'',
      'Caps remained6 GiB tree RSS and2.5 GiB allocated VRAM, with2 GiB available RAM/commit and512 MiB dedicated GPU headroom. RSS/availability was sampled every0.1s; nvidia-smi every2s. Large artifacts and mailbox traffic stayed on D:. Resource peaks are sampled except the separately reported torch allocator peak.','',
      '## Certification and repair audit','',
      f"The final audit checked {d['trajectories']} scored policy/theorem trajectories and {d['certified_instances']} successful instances. These correspond to {len(d['unique_theorems'].get('prospective',[]))} distinct prospective theorems and {len(d['unique_theorems'].get('posthoc_fc13',[]))} distinct FC-13 theorems. Duplicate policies/proof variants are not new theorems.",'',
      'Every accepted result compiled the complete original theorem in its original preceding context, with no ProbeServer or sorry in the release script. Printed dependencies allowed only propext, Classical.choice and Quot.sound. The identical final source bytes were independently recompiled. Deduplicated sources and a model/theorem/script/hash index are under certificates/. Synthetic smoke proofs and original reference oracles are excluded from research coverage.','',
      'An initial evaluation attempt was invalidated after a source-scaffolding bug inserted a heartbeat option between some documentation comments and their theorem declarations. This caused valid tactic prefixes to fail final source parsing. The wrapper was repaired and fresh-mailbox and clean probe-exit checks were added. That audit also caught heartbeat usage accumulating across long interactive batches; the outer session now has no cumulative heartbeat cap, and each request independently receives the same 100,000-heartbeat limit and 3-second wall timeout. Every unchanged training observation was then independently re-audited using corrected complete sources and clean process exits. The dataset bytes, all checkpoint weights, seed selections, candidate grammar and registered budgets were kept frozen. evaluation-01 and its logs remain preserved. These implementation repairs occurred after initial prospective outcomes were visible and are explicitly disclosed; no claim of an untouched implementation is made.','',
      'The complete primary comparison comes from evaluation-02. Its secondary phase later experienced repeated initial-goal startup timeouts, independently reproduced by synthetic smoke-07. The root cause was not established. Fresh import/session checks and full smoke-08 subsequently passed with unchanged Lean code and timeouts. All failed attempts are preserved. Both policies and all 28 secondary cases were rerun as an entire comparison in evaluation-03-secondary; individual failures were not selectively retried. An explicit assembly receipt verifies matching fingerprints and complete case counts. The final artifact manifest includes both raw component runs and their initial startup failures.','',
      'Earlier interrupted collections were preserved after fixing Lean isolated streams and transient Windows file-sharing errors. A Python import-path collision was fixed before any evaluation. Neither transport failures nor timeouts were fabricated into negative training labels.','',
      'All 17 integrity tests pass. Fault injection additionally reproduced and fixed missing child/log cleanup during a startup resource exception after the scored evaluator had loaded its frozen implementation. No scored resource abort occurred; this exception-path cleanup does not alter successful observations or policies. A final real-Lean smoke checks the published runtime.','',
      '## Limits and reproducibility','',
      'There are only27 training theorem roots, five development groups and twelve prospective declarations. Descendant states are correlated, not independent mathematical problems. The native comparator is stronger than RAM-02 because it receives the same new grammar and backend. The study bundles data, representation and search changes; it cannot isolate their causal effects without further ablations. Syntactic signatures are not semantic equivalence. The model sees pending conclusions rather than every sibling context, and token hashing/truncation lose information. Sparse family support, the fixed grammar, goal cap and bounded search remain limitations.','',
      'All18 best/last checkpoints and complete JSONL data, mailbox messages, failures, compiler logs and replays are under D:\\Research Engine\\RAM-03\\LoopedTransformers\\experiments\\lean_ram03\\runs. Checkpoint locations and SHA256s are in checkpoints.json. The full artifact manifest accompanies the local outputs; compact receipts and accepted sources are tracked in Git. RAM-02 artifacts remain unchanged. README.md and run_windows.ps1 provide reproducible commands. No merge was performed.','']
    text='\n'.join(lines)
    for old,new in {'contains4973':'contains 4973','contains4,973':'contains 4,973','at27':'at 27','original51':'original 51','excluded0':'excluded 0',
       'plus0.2':'plus 0.2','cap1.0':'cap 1.0','microbatch16':'microbatch 16','accumulation4':'accumulation 4','most1,600':'most 1,600',
       'every100':'every 100','patience4':'patience 4','seed17':'seed 17','seed29':'seed 29','seed43':'seed 43','update100':'update 100','update300':'update 300',
       'uses30':'uses 30','seconds,48':'seconds, 48','and8':'and 8','uses60':'uses 60','seconds,128':'seconds, 128','and16':'and 16',
       'native6/16':'native 6/16','was6/16':'was 6/16','RTX3050':'RTX 3050','with4096':'with 4096','Python3.13':'Python 3.13','PyTorch2.6':'PyTorch 2.6','psutil7.1':'psutil 7.1',
       'frozen1.15':'frozen 1.15','remained6':'remained 6','and2.5':'and 2.5','with2':'with 2','and512':'and 512','every0.1':'every 0.1','every2s':'every 2s',
       'only27':'only 27','All18':'All 18'}.items():text=text.replace(old,new)
    a.out.write_text(text,encoding='utf8',newline='\n')
if __name__=='__main__':main()
