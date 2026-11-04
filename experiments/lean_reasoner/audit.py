"""Replay committed neural receipts and emit proofs for independent Lean checking."""
import gzip
import hashlib
import json
from pathlib import Path
from proof_data import check_trace, legal, lean_theorem, structure_id

ROOT=Path(__file__).parent


def audit():
    artifacts=ROOT/'artifacts'
    tasks=[json.loads(x) for x in (artifacts/'tasks.jsonl').read_text().splitlines()]
    expected=json.loads((ROOT/'results'/'corpus_manifest.json').read_text())
    for name,key in [('tasks.jsonl','tasks_sha256'),('Corpus.lean','lean_source_sha256')]:
        actual=hashlib.sha256((artifacts/name).read_bytes()).hexdigest()
        if actual!=expected[key]: raise ValueError(f'{name}: corpus changed')
    fingerprints=[structure_id(t) for t in tasks]
    if len(fingerprints)!=len(set(fingerprints)): raise ValueError('duplicate structural group')
    data=json.loads(gzip.decompress((ROOT/'results'/'prediction_receipts.json.gz').read_bytes()))
    by_id={t['id']:t for t in tasks}; examples=[]; seen=set()
    for ident, solved, trace in data['rows']:
        if ident in seen: raise ValueError('duplicate receipt')
        seen.add(ident); t=by_id[ident]; known=set(t['facts'])
        if len(trace)>t['min_steps']: raise ValueError('over budget')
        for action in trace:
            if action not in legal(t,known): raise ValueError('illegal action')
            known.add(t['rules'][action][1])
        if bool(solved)!=(t['goal'] in known): raise ValueError('incorrect success label')
        if solved:
            check_trace(t,trace)
            examples.append(lean_theorem(t,'receipt_'+ident,trace))
    expected_ids={t['id'] for t in tasks if t['split'] in ['test','depth_ood']}
    if seen!=expected_ids: raise ValueError('missing evaluation receipts')
    text='set_option linter.unusedVariables false\n\n'+'\n\n'.join(examples)+'\n'
    (artifacts/'Predictions_receipts.lean').write_text(text)
    print(json.dumps({'receipt_tasks':len(seen),'successful_certificates':len(examples),
                      'lean_source_sha256':hashlib.sha256(text.encode()).hexdigest()},indent=2))


if __name__=='__main__': audit()
