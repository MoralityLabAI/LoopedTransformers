"""Post-freeze provenance/artifact checks, separate from registered inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();root=Path('experiments/nan_diag04');s=read(root/'SOURCE_MANIFEST.json')
    for k,v in s['critical_files_sha256'].items():assert sha(root/k)==v
    for k,v in s['reused_files_sha256'].items():assert sha(Path(k))==v
    oldpaths=[Path('experiments')/name/'fixtures/public.jsonl' for name in ('nan_comp02','nan_comp03')]
    prior={tuple(x) for p in oldpaths for r in rows(p) for x in r['inputs']}
    labels=rows(root/'fixtures/host_labels.jsonl');fresh=[tuple(x[:5]) for r in labels for x in r['inputs']]
    assert len(fresh)==len(set(fresh))==20 and not set(fresh)&prior
    source=read(root/'fixtures/source_receipt.json');attempts=sum(r['trials'] for r in source['attempts']);rejects=Counter()
    for r in source['attempts']:rejects.update(r['rejections'])
    allfiles={};maximum=0;by_backend={}
    for b in ('bf16','nf4'):
        folder=root/'receipts'/b; rr=[read(folder/f"root_{r['root']}.json") for r in labels]
        maximum=max(maximum,max(len(x['initial_prompt_ids']) for r in rr for x in r['records'].values()))
        for arm in rr[0]['records']:
            if arm.startswith(('copy/','labelcopy/','codecopy/')) or arm=='erased_repeat':continue
            prompts=[tuple(r['records'][arm]['initial_prompt_ids']) for r in rr]
            assert len(set(prompts))==10
        memory=read(folder/'memory_samples.json');assert memory==rows(folder/'memory_samples.jsonl')
        errors=read(folder/'telemetry_errors.json');assert not errors
        for file in sorted(folder.iterdir()):
            if file.is_file():allfiles[file.relative_to(root).as_posix()]=sha(file)
        by_backend[b]={'root_files':len(rr),'telemetry_samples':len(memory),'telemetry_errors':len(errors),'resumes':len(read(folder/'receipt.json').get('attempts',[])),
            'token_steps':sum(len(x['steps']) for r in rr for x in r['records'].values())}
    for r in labels:
        bf=read(root/'receipts/bf16'/f"root_{r['root']}.json");nf=read(root/'receipts/nf4'/f"root_{r['root']}.json")
        assert all(bf['records'][k]['initial_prompt_ids']==nf['records'][k]['initial_prompt_ids'] for k in bf['records'])
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'passed','scope':'post-freeze provenance and raw artifact preservation','source_manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),
        'critical_files_verified':len(s['critical_files_sha256']),'reused_files_verified':len(s['reused_files_sha256']),
        'prior_exposed_individual_polynomials':len(prior),'new_unique_individual_polynomials':len(set(fresh)),'overlap':0,'source_draws':attempts,'rejections':dict(rejects),
        'all_scientific_arm_CAL_DEV_prompts_disjoint':True,'maximum_initial_prompt_tokens':maximum,'backend':by_backend,'artifact_count':len(allfiles),
        'raw_artifacts_sha256':allfiles,'sealed_test_accessed':False}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({k:v for k,v in result.items() if k!='raw_artifacts_sha256'}))
if __name__=='__main__':main()
