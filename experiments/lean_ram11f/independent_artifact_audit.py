"""Independent byte/proof-source verifier for a completed public RAM11F CI archive.

Reads the archived compiler results but never invokes Lean. Does NOT replace
native Lean; confirms the trusted source replay and case accounting are
consistent with the immutable 76 candidate list and original allowlist.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

EXPECTED_WORKLIST_SHA='f5876539862a0f4f913dc3af620be0d038555efcdd3611754e002dc353c3b2aa'
ALLOWED={'propext','Classical.choice','Quot.sound'}
BEGIN='RAM11F_AXIOMS_BEGIN_20261115'
END='RAM11F_AXIOMS_END_20261115'

def sha(data):return hashlib.sha256(data).hexdigest()

def parse_ax(log):
    assert log.count(BEGIN)==log.count(END)==1
    middle=log.split(BEGIN,1)[1].split(END,1)[0]
    if 'does not depend on any axioms' in middle:return []
    m=re.search(r'depends on axioms:\s*\[([^\]]*)\]',middle)
    assert m is not None
    result=[x.strip() for x in m.group(1).split(',') if x.strip()]
    assert set(result).issubset(ALLOWED)
    return result

def proof_region(src,theorem):
    lines=src.splitlines()
    starts=[i for i,l in enumerate(lines) if re.match(r'^(?:theorem|lemma)\s+'+re.escape(theorem)+r'(?=$|\s|\(|\{|\[|:)',l)]
    assert len(starts)==1, (theorem,starts)
    marker=next(i for i in range(starts[0],len(lines)) if ':= by' in lines[i])
    end=next(i for i in range(marker+1,len(lines)) if lines[i].startswith('#eval IO.println "'+BEGIN+'"'))
    return [l.strip() for l in lines[marker+1:end] if l.strip()]

def audit(archive,worklist):
    old=worklist.read_bytes()
    assert sha(old)==EXPECTED_WORKLIST_SHA
    jobs=[json.loads(s) for s in old.decode().splitlines() if s.strip()]
    assert len(jobs)==76
    with zipfile.ZipFile(archive) as z:
        receipt=json.loads(z.read('receipt.json'))
        assert receipt['status']=='complete',receipt['status']
        assert receipt['expected']==len(jobs)==receipt['processed']==len(receipt['cases'])
        assert receipt['worklist_sha256']==EXPECTED_WORKLIST_SHA
        certified=[];rejections=[];unknown=[];family=collections.Counter()
        for i,(job,case) in enumerate(zip(jobs,receipt['cases'])):
            assert case['index']==i
            assert case['source_file']==job['source_file']
            assert case['theorem']==job['theorem']
            assert case['actions']==job['proposed_new_actions']
            assert case['recorded_original_source_sha256']==job['original_source_sha256']
            assert case['historical_proof_score']==0
            label=case['classification']
            if label!='double_kernel_compiled_train_theorem':
                (unknown if 'unknown' in label or 'timeout' in label or 'unavailable' in label else rejections).append(i)
                continue
            first=z.read(f'cases/{i:03d}/first.lean');second=z.read(f'cases/{i:03d}/second.lean')
            assert first==second
            assert case['first']['source_sha256']==case['second']['source_sha256']==case['certificate_sha256']==sha(first)
            assert case['first']['exitcode']==case['second']['exitcode']==0
            assert case['first']['axiom_gate'] and case['second']['axiom_gate']
            a=z.read(f'cases/{i:03d}/first.log').decode(errors='replace')
            b=z.read(f'cases/{i:03d}/second.log').decode(errors='replace')
            assert 'error:' not in a+b and 'sorryAx' not in a+b
            assert case['first']['log_sha256']==sha(z.read(f'cases/{i:03d}/first.log'))
            assert case['second']['log_sha256']==sha(z.read(f'cases/{i:03d}/second.log'))
            ax=parse_ax(a)
            assert ax==parse_ax(b)==case['axioms']==case['first']['axioms']==case['second']['axioms']
            actions=proof_region(first.decode('utf8'),job['theorem'])
            assert actions==job['proposed_new_actions'],(i,actions,job['proposed_new_actions'])
            assert 'sorry' not in '\n'.join(actions) and 'admit' not in '\n'.join(actions)
            assert case['seconds']<=30.0
            certified.append(i)
            family[job['candidate_family']]+=1
        assert len(certified)==receipt['certified_original_train_sources']
        return dict(status='passed',archive_sha256=sha(archive.read_bytes()),worklist_sha256=EXPECTED_WORKLIST_SHA,
                    expected=76,processed=receipt['processed'],double_compiled_cases=len(certified),
                    distinct_original_train_theorems=len({(jobs[i]['source_file'],jobs[i]['theorem']) for i in certified}),
                    certified_case_indices=certified,negative_indices=rejections,unknown_indices=unknown,
                    approved_axioms=sorted(ALLOWED),PEF_heldout_scores=0,RAM10_heldout_scores=0,
                    certified_candidate_family=dict(sorted(family.items())))

if __name__=='__main__':
    archive=Path(sys.argv[1]);worklist=Path(sys.argv[2]);out=Path(sys.argv[3]);out.parent.mkdir(parents=True,exist_ok=True)
    result=audit(archive,worklist)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
