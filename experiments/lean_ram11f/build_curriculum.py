"""Build a sequence-level positive curriculum from independently checked RAM11F source proofs."""
from __future__ import annotations
import collections,hashlib,json
from pathlib import Path
import zipfile
import sys

OLD_COMMIT='c84e2025334e9a967c1297425a490f98d5b3d189'

def sha(b):return hashlib.sha256(b).hexdigest()
def dump_jsonl(path,items):
    path.write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
    return sha(path.read_bytes())

def build(artifact:Path,worklist:Path,audit:Path,out:Path):
    if out.exists():raise FileExistsError('Do not overwrite previous curriculum')
    a=json.loads(audit.read_text())
    if a['status']!='passed' or a['double_compiled_cases']!=76 or a['distinct_original_train_theorems']!=22:
        raise AssertionError('native compiler-source evidence was not independently verified')
    if sha(artifact.read_bytes())!=a['archive_sha256']:
        raise AssertionError('immutable source archive differs from audit')
    jobs=[json.loads(s) for s in worklist.read_text().splitlines() if s.strip()]
    if len(jobs)!=76 or sha(worklist.read_bytes())!=a['worklist_sha256']:
        raise AssertionError('source script list differs from independently audited original')
    with zipfile.ZipFile(artifact) as z:
        proof=json.loads(z.read('receipt.json'))
    if proof['status']!='complete' or proof['processed']!=76 or proof['certified_original_train_sources']!=76:
        raise AssertionError('incomplete native original source replay')
    cases=[];steps={}
    for i,(case,job) in enumerate(zip(proof['cases'],jobs)):
        if case['index']!=i or case['classification']!='double_kernel_compiled_train_theorem':
            raise AssertionError('uncertified input proof')
        key=dict(root_file=case['source_file'],theorem=case['theorem'])
        actions=case['actions']
        name=f'RAM11F-{i:03d}'
        cases.append(dict(proof_id=name,**key,train_original_commit=OLD_COMMIT,
             root_original_source_sha256=job['original_source_sha256'],
             actions=actions,length=len(actions),candidate_family=job['candidate_family'],
             original_reference_target_proof_replaced=True,complete_source_compiles=2,
             native_lean_version='4.35.0',proof_source_sha256=case['certificate_sha256'],
             axioms=case['axioms'],historical_train_only=True,
             source_worklist_sha256=a['worklist_sha256']))
        for ix,action in enumerate(actions):
            label_key=(case['source_file'],case['theorem'],tuple(actions[:ix]),action)
            row=steps.setdefault(label_key,dict(**key,prefix_actions=actions[:ix],
                    next_action=action,min_remaining_actions=len(actions)-ix,
                    supporting_proof_ids=[]))
            row['min_remaining_actions']=min(row['min_remaining_actions'],len(actions)-ix)
            row['supporting_proof_ids'].append(name)
    out.mkdir(parents=True)
    all_steps=[]
    for row in steps.values():
        row['supporting_proof_ids']=sorted(set(row['supporting_proof_ids']))
        row['label']='source_verified_prefix_has_explicit_complete_tactic_suffix'
        row['labels_from_kernel_completed_original_theorem']=True
        row['per_step_goal_state_observed_in_fresh_Lean']=False
        all_steps.append(row)
    all_steps.sort(key=lambda x:(x['root_file'],x['theorem'],x['prefix_actions'],x['next_action']))
    proof_sha=dump_jsonl(out/'certified_complete_scripts.jsonl',cases)
    step_sha=dump_jsonl(out/'certified_positive_prefix_action_labels.jsonl',all_steps)
    rootset={(x['root_file'],x['theorem']) for x in cases}
    prefixset={(x['root_file'],x['theorem'],tuple(x['prefix_actions'])) for x in all_steps}
    receipt=dict(schema='RAM11F-v1-native-certified-train-sequence-curriculum',
       original_source_commit=OLD_COMMIT,artifact_sha256=a['archive_sha256'],
       worklist_sha256=a['worklist_sha256'],scripts_compiled_twice=len(cases),
       unique_original_theorems=len(rootset),verified_prefix_next_action_labels=len(all_steps),
       unique_prefix_contexts=len(prefixset),
       total_script_actions=sum(len(c['actions']) for c in cases),
       repeated_path_edges=sum(len(c['actions']) for c in cases)-len(all_steps),
       longest_script=max(len(c['actions']) for c in cases),
       script_sha256=proof_sha,positive_prefix_sha256=step_sha,
       target_reference_proofs_copied=0,
       per_step_Lean_goal_observations_added=0,
       original_PEF_test_scores=0,RAM10_new_source_scores=0,
       negative_action_labels_added=0,
       trust_boundary='Full source was independently compiled twice per original theorem, and all generated source/script/log bytes were separately audited. Prefix rows are existentially extendable script-prefix labels, not newly captured Lean proof states.',
       caveat='Retrospective selected original TRAIN cohort; duplicate script prefixes share proof roots. Not a new held-out capability measurement.')
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    return receipt

if __name__=='__main__':
    a=build(*(Path(s) for s in sys.argv[1:]))
    print(json.dumps(a,indent=2))
