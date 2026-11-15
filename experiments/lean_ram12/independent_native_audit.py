"""Independent read-only RAM-12 artifact audit. Native Lean output is immutable."""
from __future__ import annotations
import argparse,hashlib,json,re,zipfile
from pathlib import Path
OLD='c84e2025334e9a967c1297425a490f98d5b3d189'
WL='f5876539862a0f4f913dc3af620be0d038555efcdd3611754e002dc353c3b2aa'
ARTIFACT='327dbd1382aca1a1c3c790e434341d582aafb85e5ad1b4f9b70ca84f21cae924'
MARK='RAM12_STATE_JSON:'
def sha(b):return hashlib.sha256(b).hexdigest()

def audit(archive:Path,worklist:Path)->dict:
    if sha(archive.read_bytes())!=ARTIFACT:raise AssertionError('native state archive SHA mismatch')
    if sha(worklist.read_bytes())!=WL:raise AssertionError('historical worklist SHA mismatch')
    jobs=[json.loads(s) for s in worklist.read_text().splitlines() if s.strip()]
    with zipfile.ZipFile(archive) as z:
        receipt=json.loads(z.read('receipt.json'))
        if (receipt['status'],receipt['source_commit'],receipt['worklist_sha256'])!=('complete',OLD,WL):
            raise AssertionError('native state receipt incomplete')
        if receipt.get('full_source_compilations')!=152 or len(receipt['cases'])!=76:
            raise AssertionError('missing original full-source proofs')
        raw=z.read('certified_state_action_labels.jsonl')
        if sha(raw)!=receipt['labels_sha256']:raise AssertionError('positive state-label byte SHA differs')
        labels=[json.loads(s) for s in raw.decode().splitlines() if s.strip()]
        original_keys={(j['source_file'],j['theorem'],tuple(j['proposed_new_actions'][:i]),j['proposed_new_actions'][i])
                       for j in jobs for i in range(len(j['proposed_new_actions']))}
        label_keys={(r['source_file'],r['theorem'],tuple(r['prefix_actions']),r['next_action']) for r in labels}
        if len(labels)!=230 or label_keys!=original_keys:
            raise AssertionError('native state/action labels differ from historical original scripts')
        same_prefix={}
        verified_compiles=0
        markers=0
        for i,(case,job) in enumerate(zip(receipt['cases'],jobs)):
            if (case['index']!=i or case['proof_id']!=f'RAM11F-{i:03d}' or
                case['actions']!=job['proposed_new_actions'] or case['status']!='two_complete_source_compiles_agree'):
                raise AssertionError('native case permuted or skipped')
            prefix=f'cases/{i:03d}/'
            first=z.read(prefix+'first.lean');second=z.read(prefix+'second.lean')
            if first!=second or sha(first)!=case['instrumented_source_sha256']:
                raise AssertionError('independently compiled source bytes disagree')
            source=first.decode()
            if source.count('ram12_observe "RAM11F-')!=len(case['actions']):
                raise AssertionError('native instrumentation point count differs')
            if re.search(r'^\s*(?:sorry|admit)\s*$',source,re.M):
                raise AssertionError('instrumentation inserted an admitted tactic')
            states=[]
            for run in ('first','second'):
                entry=case[run]
                if entry['status']!='accepted' or entry['source_sha256']!=sha(first):
                    raise AssertionError('full original theorem did not compile')
                log=z.read(prefix+run+'.log')
                if sha(log)!=entry['log_sha256'] or b'error:' in log or b'sorryAx' in log:
                    raise AssertionError('native Lean compiler log invalid or modified')
                data=[json.loads(line.split(MARK,1)[1].strip())
                      for line in log.decode().splitlines() if MARK in line]
                if [x['id'] for x in data]!=[f'RAM11F-{i:03d}:{k:02d}' for k in range(len(case['actions']))]:
                    raise AssertionError('one or more original tactic-prefix states missing')
                states.append(data);verified_compiles+=1
            if states[0]!=states[1]:raise AssertionError('fresh Lean state replay mismatch')
            markers+=len(states[0])
            for k,st in enumerate(states[0]):
                key=(job['source_file'],job['theorem'],tuple(job['proposed_new_actions'][:k]))
                state=(st['goals'],st['definitions'])
                if key in same_prefix and same_prefix[key]!=state:
                    raise AssertionError('duplicate exact prefix has conflicting original Lean goals')
                same_prefix[key]=state
        if (verified_compiles,markers,len(same_prefix))!=(152,282,176):
            raise AssertionError('independent prefix-state counts differ')
        for row in labels:
            expected=same_prefix[(row['source_file'],row['theorem'],tuple(row['prefix_actions']))]
            if (row['goals'],row['definitions'])!=expected or row['full_source_compiles']!=2:
                raise AssertionError('positive label not backed by two native complete proof sources')
            if row['label']!='positive_from_twice_compiled_complete_original_Lean_source':
                raise AssertionError('nonpositive or unknown label injected into proof curriculum')
    return dict(status='passed',original_compiles=verified_compiles,
                original_scripts=len(jobs),raw_tactic_states=markers,
                distinct_exact_prefix_states=len(same_prefix),
                distinct_positive_prefix_actions=len(labels),
                artifact_sha256=ARTIFACT,
                native_state_labels_sha256=receipt['labels_sha256'],
                no_original_PEF_or_RAM10_heldout_scoring=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--worklist',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('preserve original independent source audit')
    result=audit(a.artifact,a.worklist)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))
