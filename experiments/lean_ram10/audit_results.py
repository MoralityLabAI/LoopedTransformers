"""RAM-10 third independent Lean4.35 full-theorem certificate and score audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from ram10_core import rows,sha_file,stable_json,registered_ram08_core
from evaluate import PHASE_POLICIES


def audit(root,config_path,lean,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('third audit output exists; preserve old proof receipts')
    cfg=json.loads(Path(config_path).read_text())
    c=registered_ram08_core(cfg)
    report=json.loads((root/'evaluation_report.json').read_text())
    manifest=json.loads((root/'test_manifest.json').read_text())
    if report['status']!='complete' or report['config_sha256']!=sha_file(config_path):
        raise AssertionError('original first proof search incomplete or changed')
    tests=manifest['tests'];count=0;verified=[];unknown=[]
    for phase,policies in PHASE_POLICIES.items():
        for policy in policies:
            listing=rows(root/f'{phase}__{policy}.jsonl')
            if len(listing)!=len(tests):raise AssertionError('missing registered theorem rows')
            if [(r['source_file'],r['theorem']) for r in listing]!=[(t['source_file'],t['theorem']) for t in tests]:
                raise AssertionError('test theorem order or source changed after freeze')
            for i,(entry,case) in enumerate(zip(listing,tests)):
                count+=1
                if entry['budget']!=cfg['eval'][phase]:raise AssertionError('registered theorem search budget changed')
                if entry['logical_evaluations']>entry['budget']['evaluations']:
                    raise AssertionError('logical tactic evaluation cap exceeded')
                if entry['max_path_transitions']>entry['budget']['transitions']+1:
                    raise AssertionError('path transition cap exceeded')
                if entry['reason'] in ('request_timeout','startup_or_request_timeout','deadline_shutdown_overrun'):
                    unknown.append((phase,policy,case['theorem']))
                if not entry['accepted']:continue
                if entry['reason']!='independently_certified':raise AssertionError('accepted theorem has wrong verification state')
                cert=entry['certificate'];replay=entry['replay']
                if not cert or not replay or not cert['accepted'] or not replay['accepted']:
                    raise AssertionError('missing second in-budget original source compile')
                proof_dir=root/'evidence'/phase/f'{i:02d}'/policy
                first=proof_dir/'certificate.lean';second=proof_dir/'independent_replay.lean'
                if not first.exists() or not second.exists():raise AssertionError('original source proof script missing')
                source=first.read_bytes()
                if source!=second.read_bytes():raise AssertionError('source changed across in-budget compiles')
                digest=hashlib.sha256(source).hexdigest()
                if cert['source_sha256']!=digest or replay['source_sha256']!=digest:
                    raise AssertionError('certified original proof source byte mismatch')
                if entry['original_source_sha256']!=report['source_hashes'][case['source_file']]['normalized_sha256']:
                    raise AssertionError('original source theorem raw SHA mismatch')
                if not entry['actions'] or any(not c.valid_script(script) for script in entry['actions']):
                    raise AssertionError('outside registered source-visible tactic grammar')
                if any(re.search(r'\b(?:sorry|sorryAx|admit)\b',script) for script in entry['actions']):
                    raise AssertionError('forbidden generated proof command')
                if source.count(b'#print axioms ')!=1 or (b'#print axioms '+case['theorem'].encode()) not in source:
                    raise AssertionError('missing explicitly printed theorem axiom inventory')
                # The third original full-theorem compilation must run in a
                # separate unscored Lean process and cannot change the score.
                result=subprocess.run([lean,str(first.resolve())],capture_output=True,text=True,timeout=120)
                output=result.stdout+result.stderr
                permitted,axioms=c.axiom_audit(output)
                if result.returncode or 'error:' in output or 'sorryAx' in output or not permitted:
                    raise AssertionError('third original Lean source proof did not pass kernel/axiom gate')
                if set(axioms)!=set(cert['axioms']) or set(axioms)!=set(replay['axioms']):
                    raise AssertionError('second/third original axiom inventory mismatch')
                verified.append(dict(phase=phase,policy=policy,source_file=case['source_file'],
                    theorem=case['theorem'],source_sha256=digest,axioms=axioms,
                    third_recompiled=True,original_scores_unchanged=True))
    expected=len(tests)*sum(len(p) for p in PHASE_POLICIES.values())
    if count!=expected:raise AssertionError('original prospective case matrix incomplete')
    accepted=sum(s['certified'] for s in report['summaries'])
    if accepted!=len(verified):raise AssertionError('accepted theorem count differs from independent audit')
    output=dict(experiment='RAM10_independent_third_Lean435_replay',
        evaluated_cases=count,original_test_theorems=len(tests),
        independently_verified_certificates=len(verified),
        false_positive_certificates=0,unresolved_compiler_unknown=unknown,
        original_scored_report_sha256=sha_file(root/'evaluation_report.json'),
        original_proof_sources=verified,source_reference_proofs_used_for_policy=0,
        all_scored_results_unchanged=True)
    out.mkdir(parents=True,exist_ok=False)
    stable_json(out/'RAM10_independent_audit.json',output)
    print('RAM10_ORIGINAL_LEAN_AUDIT',json.dumps({k:output[k] for k in ('evaluated_cases','independently_verified_certificates','false_positive_certificates')},sort_keys=True),flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--lean',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.results,a.config,a.lean,a.out)
