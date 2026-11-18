"""Supplemental source uniqueness and byte-index checks; no fitting or grading."""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return json.dumps(sorted(x),separators=(',',':'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args();runs=Path(a.runs)
    prior=Path('experiments/nan_comp02/fixtures/public.jsonl')
    old=[json.loads(s) for s in prior.read_text().splitlines()];public=[json.loads(s) for s in (runs/'cohort/public.jsonl').read_text().splitlines()]
    pairs=read(runs/'cohort/pairs.json');rows={r['case_id']:r for r in public};source=read(runs/'cohort/source_receipt.json')
    assert sha(prior)==source['previous_public_sha256'];oldkeys={key(r['inputs']) for r in old};newkeys={key(rows[p['recipient']]['inputs']) for p in pairs}
    assert len(oldkeys)==len(newkeys)==10 and not oldkeys&newkeys
    for r in public:assert not any(x in source['tutorial_inputs_excluded'] for x in r['inputs'])
    indexed=[];model_tapes=0
    for b in ('bf16','nf4'):
        byvariant={v:set() for v in ('curriculum','balanced','erased')}
        for pair in pairs:
            tapes=[read(runs/b/(pair[k].replace(':','_')+'.json')) for k in ('recipient','donor')]
            for role,t in zip(('recipient','donor'),tapes):
                assert (t['case_id'],t['root'],t['split'])==(pair[role],pair['root'],pair['split']);model_tapes+=1
            for arm in ('curriculum','balanced'):
                assert len(tapes[0]['records'][arm]['initial_prompt_ids'])==len(tapes[1]['records'][arm]['initial_prompt_ids'])
                for tape in tapes:
                    ids=tuple(tape['records'][arm]['initial_prompt_ids']);assert ids not in byvariant[arm];byvariant[arm].add(ids)
            erased=[tuple(t['records']['erased']['initial_prompt_ids']) for t in tapes]
            assert erased[0]==erased[1] and erased[0] not in byvariant['erased'];byvariant['erased'].add(erased[0])
        assert {k:len(v) for k,v in byvariant.items()}=={'curriculum':20,'balanced':20,'erased':10}
    bf,nf=[read(runs/b/'receipt.json') for b in ('bf16','nf4')]
    for k in ('critical_files_sha256','source_manifest_sha256','public_sha256','pairs_sha256','config_sha256','decoder_source_sha256','head_source_sha256'):
        assert bf[k]==nf[k]
    for file in sorted(runs.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts and file.resolve()!=Path(a.out).resolve():
            indexed.append({'path':file.relative_to(runs).as_posix(),'bytes':file.stat().st_size,'sha256':sha(file)})
    result={'status':'passed','prior_exposed_input_pairs_excluded':10,'new_input_pairs':10,'CAL_DEV_token_prompts_disjoint':True,
        'clean_prompts_per_format':20,'erased_root_prompts':10,'model_tape_files_verified':model_tapes,'matched_backend_source_hashes':True,
        'sealed_test_accessed':False,'runtime_files':indexed}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({k:v for k,v in result.items() if k!='runtime_files'}))

if __name__=='__main__':main()
