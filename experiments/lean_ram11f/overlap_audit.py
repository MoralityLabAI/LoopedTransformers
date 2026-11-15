"""Compare certified historical prefix-action positive rows with RAM08 old TRAIN compiler observations."""
import collections, hashlib, json, sys, zipfile
from pathlib import Path

SHA_OLD='f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'

def sha(b):return hashlib.sha256(b).hexdigest()

def audit(original_archive,proof_rows,output):
    with zipfile.ZipFile(original_archive) as z: raw=z.read('data/transitions.jsonl')
    if sha(raw)!=SHA_OLD:raise ValueError('RAM08 original labels checksum mismatch')
    old=[json.loads(s) for s in raw.decode().splitlines() if s]
    if len(old)!=20930:raise ValueError('Original Lean effects missing')
    source={ (r['source_file'],r['theorem'],tuple(r['prefix']),r['script']):r for r in old if not r['development']}
    devkeys={ (r['source_file'],r['theorem']) for r in old if r['development']}
    rows=[json.loads(s) for s in proof_rows.read_text().splitlines() if s]
    expected=230
    if len(rows)!=expected:raise ValueError('Certified source proof corpus changed')
    scores=collections.Counter();roots=set();newkeys=[]
    for r in rows:
        root=(r['root_file'],r['theorem'])
        if root in devkeys:raise AssertionError('Historical RAM08 dev theorem leaked into training curriculum')
        roots.add(root)
        k=(root[0],root[1],tuple(r['prefix_actions']),r['next_action'])
        earlier=source.get(k)
        if earlier:
            if earlier['label'] not in (1,2,3):raise AssertionError('Certified next tactic was previously Lean rejected')
            scores['observed_original_compiler_positive']+=1
            scores['original_label_'+str(earlier['label'])]+=1
        else:
            scores['absent_from_original_compiler_collector']+=1
            newkeys.append(k)
    assert len(roots)==22 and scores['observed_original_compiler_positive']==138 and scores['absent_from_original_compiler_collector']==92
    result=dict(status='passed',old_transitions_sha256=SHA_OLD,old_original_effects=20930,
        original_train_compiler_effects=sum(not r['development'] for r in old),
        unique_certified_full_proof_prefix_next_action_pairs=len(rows),
        verified_pairs_in_old_collector=scores['observed_original_compiler_positive'],
        fully_certified_pairs_absent_from_old_collector=scores['absent_from_original_compiler_collector'],
        historical_accepted_not_closed=scores['original_label_1']+scores['original_label_2'],
        originally_closed_actions=scores['original_label_3'],
        source_roots=len(roots),RAM08_DEV_source_root_overlap=0,
        PEF_and_RAM10_test_scores=0,
        caveat='The 92 unobserved exact-prefix actions occur in successfully compiled original proof scripts. They are valid existential completion witnesses but their intermediate Lean printed states were not freshly captured, and no prospective theorem test was performed.')
    output.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])),indent=2))
