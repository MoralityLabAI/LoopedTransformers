"""RAM-15 independent third Lean4.35 original-source certificate audit.

This job has no model-scoring role. It reads immutable first-run case receipts,
reconstructs each accepted original complete theorem source and recompiles it
in a third fresh Lean process. All originally failed cases retain their score.
"""
from __future__ import annotations
import collections,hashlib,json,os,subprocess,time,sys
from pathlib import Path
import harness
from native_replay import render_source,axiom_inventory

OUT=harness.OUT
MAN=harness.MANIFEST
PHASES={'primary':harness.POLICIES,'secondary':harness.SECOND,
        'ablation':['native','loop64']}
BUDGETS={'primary':harness.PRIMARY,'secondary':harness.SECONDARY,
         'ablation':harness.PRIMARY}
ORIGINAL_CERTS=0


def audit():
    src=harness.module_sources()
    cases=[(m,t) for m,(_,names) in harness.LOCKED.items() for t in names]
    if len(cases)!=14:raise AssertionError('missing originally registered cases')
    full=[]
    receipt_dir=OUT/'independent_third'
    if receipt_dir.exists():raise FileExistsError('independent audit is single-use')
    receipt_dir.mkdir(parents=True)
    start=time.perf_counter()
    for phase,policies in PHASES.items():
        for index,(module,theorem) in enumerate(cases):
            for policy in policies:
                case_dir=OUT/phase/f'{index:02d}'/policy
                path=case_dir/'case.json'
                if not path.is_file():raise AssertionError('missing first scored result '+str(path))
                rec=json.loads(path.read_text())
                budget=BUDGETS[phase]
                if rec['budget']!=budget or rec['policy']!=policy or rec['theorem']!=theorem:
                    raise AssertionError('first score metadata does not match preregistration')
                if rec['original_source_sha256']!=harness.sha(src[module].encode()):
                    raise AssertionError('original theorem source hash differs')
                if rec['logical_evaluations']>budget['evaluations'] or len(rec['actions'])>budget['transitions']:
                    raise AssertionError('case exceeded registered tactic/transition limit')
                if rec['elapsed_seconds']>budget['seconds']+0.0000001 and rec['accepted']:
                    raise AssertionError('late original theorem cannot score')
                record=dict(phase=phase,case_index=index,module=module,
                            theorem=theorem,policy=policy,accepted=rec['accepted'],
                            original_reason=rec['reason'],original_budget=budget,
                            original_logical_evaluations=rec['logical_evaluations'])
                if rec['accepted']:
                    if rec['reason']!='independently_certified' or not rec['actions']:
                        raise AssertionError('accepted case lacks paired source certificate')
                    harness.grammar.set_context(src[module],theorem,ablation=(phase=='ablation'))
                    if any(not harness.grammar.valid_script(x) for x in rec['actions']):
                        raise AssertionError('unregistered executable proof action')
                    first=rec['certificate'];second=rec['replay']
                    if not first['accepted'] or not second['accepted']:
                        raise AssertionError('first two full source compilers did not accept')
                    raw=(case_dir/'certificate.lean').read_bytes()
                    replay=(case_dir/'independent_replay.lean').read_bytes()
                    exact=render_source(src[module],theorem,rec['actions']).encode()
                    if raw!=replay or raw!=exact:
                        raise AssertionError('source reference proof or script differs')
                    if not (harness.sha(raw)==first['source_sha256']==second['source_sha256']):
                        raise AssertionError('original compiled source SHA mismatch')
                    target=receipt_dir/phase/f'{index:02d}'/policy
                    target.mkdir(parents=True)
                    third=target/'third.lean';third.write_bytes(raw)
                    p=subprocess.run([str(harness.LEAN),str(third.resolve())],cwd=harness.SRC,
                        env=harness.make_env(),capture_output=True,text=True,timeout=30)
                    log=p.stdout+p.stderr
                    (target/'third.log').write_text(log)
                    good,axioms,why=axiom_inventory(log)
                    if p.returncode!=0 or not good or 'error:' in log or 'sorryAx' in log:
                        raise AssertionError(f'new independent proof failed: {phase}/{index}/{policy} {why} {log[-500:]}')
                    if axioms!=first['axioms'] or axioms!=second['axioms']:
                        raise AssertionError('third compiler transitive axiom disagreement')
                    record.update(third_compiler_accepted=True,
                        original_complete_source_sha256=harness.sha(raw),
                        third_compiler_log_sha256=harness.sha(log.encode()),
                        axioms=axioms,
                        script_actions=rec['actions'],
                        new_structural_grammar_used=any(not harness.grammar.ORIGINAL.valid_script(x)
                                                        for x in rec['actions']))
                    (target/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
                full.append(record)
                print('RAM15_AUDIT',phase,index,policy,int(rec['accepted']),flush=True)
    expected=14*(len(PHASES['primary'])+len(PHASES['secondary'])+len(PHASES['ablation']))
    if len(full)!=expected:raise AssertionError('incomplete registered case matrix')
    summary={}
    for phase,policies in PHASES.items():
        summary[phase]={p:sum(int(x['accepted']) for x in full
                      if x['phase']==phase and x['policy']==p)
                      for p in policies}
    primary=summary['primary']
    symbolic=set()
    recurrent=set()
    flat=set()
    for rec in full:
        if rec['phase']!='primary' or not rec['accepted']:continue
        item=f"{rec['module']}:{rec['theorem']}"
        if rec['policy'] in ('native','frequency','shape_frequency'):symbolic.add(item)
        if rec['policy'] in ('loop32','loop64'):recurrent.add(item)
        if rec['policy']=='flat64':flat.add(item)
    criterion=sorted(recurrent-symbolic)
    recurrence_specific=sorted(recurrent-symbolic-flat)
    retained=dict(schema='RAM15-first-matched-source-family-results',
        original_source_commit=MAN['source_commit'],
        number_selected_theorems=14,full_case_counts={p:14*len(v) for p,v in PHASES.items()},
        total_scored_cases=len(full),
        third_compiled_cases=sum(x['accepted'] for x in full),
        full_source_accepted_compilations=3*sum(x['accepted'] for x in full),
        false_certificates=0,policy_certified_counts=summary,
        registered_recurrent_only_theorems=criterion,
        recurrence_specific_theorems=recurrence_specific,
        primary_criterion_passed=bool(criterion),
        original_source_grammars_evaluated=['expanded_shared_grammar','original_20_family_ablation'],
        proof_sources_with_new_structural_macro=sum(x.get('new_structural_grammar_used',False)
                                                      for x in full),
        source_grammar_cap=32,original_RAM08_PEF_cases_scored=0,
        RAM10_first_original_heldout_cases_scored=0,
        scientific_status='first fresh source-family theorem transfer; source public, theorem roots nested in modules',
        benchmark_was_pinned_before_source_inspection=True,
        elapsed_audit_seconds=round(time.perf_counter()-start,3))
    (OUT/'RAM15_RESULT.json').write_text(json.dumps(retained,indent=2)+'\n')
    with (OUT/'RAM15_CASES.jsonl').open('w') as f:
        for row in full:f.write(json.dumps(row,sort_keys=True)+'\n')
    print('RAM15_FINAL',json.dumps(retained,sort_keys=True),flush=True)


if __name__=='__main__':audit()
