"""Independently verify registered RAM13 primary/secondary tables and third source compiles."""
import collections,hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,'/mnt/data/ram13_local')
import harness,grammar
OUT=Path('/mnt/data/ram13_local')

def sha(b):return hashlib.sha256(b).hexdigest()
def run():
 source=harness.module_sources()
 cases=[(mod,t) for mod,(_,names) in harness.LOCKED.items() for t in names]
 rows=[];bycase=[];third=[]
 for phase,policies,budget in [('primary',harness.POLICIES,harness.PRIMARY),('secondary',harness.SECOND,harness.SECONDARY)]:
  for i,(mod,t) in enumerate(cases):
   solved=[];failures=[]
   for policy in policies:
    f=OUT/'results'/phase/f'{i:02d}'/policy/'case.json'
    if not f.is_file():raise AssertionError('missing source case '+str(f))
    record=json.loads(f.read_text())
    assert record['policy']==policy and record['budget']==budget and record['theorem']==t
    assert record['logical_evaluations']<=budget['evaluations']
    assert record['elapsed_seconds']<=budget['seconds']
    assert record['original_source_sha256']==sha(source[mod].encode())
    reason=record['reason']
    auditpath=OUT/'third_audit'/phase/f'{i:02d}'/policy/'third_audit.json'
    cert=record['certificate'];replay=record['replay']
    if record['accepted']:
     assert reason=='independently_certified' and cert['accepted'] and replay['accepted']
     assert cert['source_sha256']==replay['source_sha256']
     assert len(record['actions'])<=budget['transitions']
     assert all(grammar.valid_script(s) for s in record['actions'])
     assert auditpath.is_file(),'missing independent third Lean compile'
     check=json.loads(auditpath.read_text())
     assert check['accepted'] and check['score_unchanged']
     assert check['compiler_sha256']==cert['source_sha256']
     assert check['axioms']==cert['axioms']==replay['axioms']
     assert check['original_source_sha256']==record['original_source_sha256']
     third.append(check)
     solved.append(policy)
    else:
     assert not auditpath.exists()
     failures.append(dict(policy=policy,reason=reason))
    rows.append(dict(phase=phase,case=i,module=mod,theorem=t,policy=policy,accepted=record['accepted'],reason=reason,
                     actions=record['actions'],tactic_calls=record['logical_evaluations'],seconds=record['elapsed_seconds'],
                     source_sha256=record['original_source_sha256'],certificate_sha256=cert['source_sha256'] if record['accepted'] else None,
                     axioms=cert['axioms'] if record['accepted'] else None))
   bycase.append(dict(phase=phase,index=i,module=mod,theorem=t,certified=solved,failures=failures))
 assert len(rows)==156 and len(third)==114
 assert len({(r['phase'],r['case'],r['policy']) for r in rows})==156
 assert len({(r['phase'],r['case_index'],r['policy']) for r in third})==114
 accepted=sum(r['accepted'] for r in rows)
 assert accepted==114
 policycounts={}
 for phase,policies,_ in [('primary',harness.POLICIES,harness.PRIMARY),('secondary',harness.SECOND,harness.SECONDARY)]:
  for p in policies:
   pp=[r for r in rows if r['phase']==phase and r['policy']==p]
   assert len(pp)==12
   policycounts[f'{phase}/{p}']=dict(certified=sum(r['accepted'] for r in pp),cases=12,
         total_tactic_calls=sum(r['tactic_calls'] for r in pp),total_wall_seconds=round(sum(r['seconds'] for r in pp),3),
         failure_reasons=dict(collections.Counter(r['reason'] for r in pp if not r['accepted'])))
 primary=[r for r in bycase if r['phase']=='primary']
 symbolic={'native','frequency','shape_frequency'}
 recurrent={'loop32_residual','loop64_residual'}
 model_only=[r['module']+'.'+r['theorem'] for r in primary if set(r['certified'])&recurrent and not set(r['certified'])&symbolic]
 recurrence_only=[r['module']+'.'+r['theorem'] for r in primary if set(r['certified'])&recurrent and not set(r['certified'])&symbolic and not set(r['certified'])&{'flat64','flat64_residual'}]
 assert model_only==[] and recurrence_only==[]
 for name,digest in harness.HEADS.items():
  with zipfile.ZipFile(harness.RAM12ZIP) as z:assert sha(z.read(f'weights/first_{name}.pt'))==digest
 with zipfile.ZipFile(harness.MODELZIP) as z:
  assert sha(z.read('data/transitions.jsonl'))==harness.old.DATA_SHA
  assert all(sha(z.read(f'training/{n}_s{seed}/best.pt'))==d for n,(seed,d) in harness.old.EXPECTED.items())
 report={
  'experiment':'RAM13-v1-first-fresh-theorem-family-transfer',
  'source_commit':'f427e0ef4a61093304d12ca2c0dc943d8d2429e5',
  'registration_commit':'5935f96ea4fe0dc06ef1a115a1e4a55491b00d98',
  'locked_source_manifest_commit':'d7324f48c179e6a6c3968e2c01b72181661cba36',
  'source_pin_correction_amendment':'AMENDMENT_A.md (pre-scoring missing final c in 40-digit blob hash)',
  'public_native_lean435_runtime_run':43721176783,
  'lean_version':'4.35.0',
  'source_selector':'first 2 top-level original theorem/lemma after ignoring Lean comments',
  'test_modules':{k:dict(git_blob_sha=v[0],theorems=list(v[1]),normalized_source_sha256=sha(source[k].encode())) for k,v in harness.LOCKED.items()},
  'ram08_train_archive_sha256':harness.old.ARCHIVE_SHA,
  'ram08_training_effects_sha256':harness.old.DATA_SHA,
  'first_ram12_residual_head_sha256':harness.HEADS,
  'original_selected_weights_sha256':{k:v[1] for k,v in harness.old.EXPECTED.items()},
  'original_grammar_old_train_state_pools_exact_match':'714/714, ordered 20,930 candidate observations',
  'primary_policies':harness.POLICIES,'secondary_policies':harness.SECOND,
  'primary_budget':harness.PRIMARY,'secondary_budget':harness.SECONDARY,
  'full_case_counts':dict(primary=108,secondary=48,total=156),
  'accepted_policy_theorem_instances':accepted,'three_compile_verified_cases':len(third),
  'full_source_lean_compilations_for_accepted_cases':3*len(third),
  'false_accepted_certificates':0,'registered_model_only_theorems':model_only,
  'registered_recurrence_specific_theorems':recurrence_only,
  'primary_recurrent_only_criterion_met':bool(model_only),
  'secondary_additional_theorems':0,'policy_summary':policycounts,
  'per_theorem_summary':bycase,
  'unknown_outcomes':dict(collections.Counter(r['reason'] for r in rows if r['reason'] in ('request_timeout','startup_or_request_timeout','deadline_shutdown_overrun','unexpected_error'))),
  'original_RAM08_PEF_cases_scored':0,
  'RAM10_original_heldout_cases_scored':0,
  'limitations':'Matched first new-family native Lean proof search of frozen RAM08/RAM12 models, but public theorem statements may share MathCan definitions; 12 original theorem targets, no independence across theorems in the same module. Failure to find a proof within a budget is not an unprovability claim. A 3-compile certificate is for original complete-source proof of the selected theorem. Original RAM08 PEF and RAM10 independent test registrations remain pending.'
 }
 (OUT/'RAM13_RESULT_20261115.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
 (OUT/'RAM13_CASES_20261115.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows))
 print('REGISTERED_COUNT',len(rows),'ACCEPTED',accepted,'THIRD_AUDIT',len(third),'FALSE_CERTIFICATES',0)
 print('REGISTERED_GAIN',model_only)
 for p,v in policycounts.items():print(p, v['certified'],'/12')
 return report
if __name__=='__main__':run()
