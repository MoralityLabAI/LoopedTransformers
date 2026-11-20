"""Independent final-answer/state-tape audit. Detailed routing evidence stays local."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from experiments.nan_gym05.common import read,rows,sha,write
from .audit_suite_sources import story_step,story_rival

def final_answer(text,schema):
    if len(text)>schema.get('max_characters',2048):return None
    s=text.strip();tick=chr(96)
    if s.startswith(tick*3):
        m=re.fullmatch(tick*3+r'(?:json)?\s*\n(.*?)'+tick*3,s,flags=re.S)
        if not m:return None
        s=m[1].strip()
    elif s.startswith('**'):
        if len(s)<=4 or not s.endswith('**'):return None
        s=s[2:-2].strip()
    elif s.startswith(tick):
        if len(s)<=2 or not s.endswith(tick):return None
        s=s[1:-1].strip()
    if schema['type']=='label':
        for answer in schema['allowed']:
            if s in (answer,answer+'.'):return answer
        return None
    try:v=json.loads(s)
    except (json.JSONDecodeError,ValueError):return None
    if schema['type']=='grid':
        if re.search(r'[^\[\],0-9\s]',s):return None
        good=isinstance(v,list) and 1<=len(v)<=12 and all(type(r) is list and 1<=len(r)<=12 for r in v)
        if good:good=len({len(r) for r in v})==1 and all(type(x) is int and 0<=x<=9 for r in v for x in r)
    else:
        good=(not re.search(r'[^\[\],"a-z_\s]',s) and isinstance(v,list) and 1<=len(v)<=6 and all(type(x) is str and x in schema['allowed'] for x in v))
    return v if good else None

def correctness(value,label):
    if label.get('copy_control') or label['suite']!='storyworld':return int(value==label['answer'])
    if not isinstance(value,list) or not 1<=len(value)<=6:return 0
    d=label['start_state'];s=tuple(d[k] for k in ('trust','evidence','heat','scene'))
    for action in value:
        if s[3]>=5:break
        s=story_step(s,action,story_rival(s))
    return int(s[3]>=5 and s[0]>=1 and s[1]>=3 and s[2]<=2)

def audit(runs,backend):
    seal=read('experiments/nan_gym05/SOURCE_MANIFEST.json');cfg=read('experiments/nan_gym05/config.json')
    for n,h in seal['critical_files_sha256'].items():assert sha(Path('experiments/nan_gym05')/n)==h
    assert sha(runs/'cohort/public.jsonl')==seal['combined_public_sha256'] and sha(runs/'cohort/host_labels.jsonl')==seal['combined_host_sha256']
    public={r['id']:r for r in rows(runs/'cohort/public.jsonl')};labels={r['id']:r for r in rows(runs/'cohort/host_labels.jsonl')}
    folder=runs/backend;receipt=read(folder/'receipt.json');assert receipt['status']=='completed'
    assert receipt['source_manifest_sha256']==sha('experiments/nan_gym05/SOURCE_MANIFEST.json') and receipt['backend']==backend
    assert set(receipt['completed_roots'])=={r['root'] for r in public.values()}
    linear=read(folder/'linear_baselines/manifest.json');shapes={};artifacts={};closure=[]
    for r in linear['receipts']:
        p=folder/'linear_baselines'/r['file'];assert sha(p)==r['file_sha256'];shapes[r['layer_zero_based'],r['module']]=r['shape'];artifacts[p.name]=sha(p)
        assert not r['trained_VPD'] and not r['trained_BPD']
        for rank,v in r['ranks'].items():assert v['closure_relative_l2']<=1e-5;closure.append(v['closure_relative_l2'])
    assert len(shapes)==8
    graded={r['id']:r for r in read(runs/'graded/local_graded_records.json') if r['backend']==backend}
    seen=set();source_roots=set();prefixes=0;states=0;formats=Counter();eos=receipt['eos_token_id']
    def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    for file in sorted(folder.glob('root_*.json')):
        tape=read(file);source_roots.add(tape['root']);assert set(tape['records'])=={k for k,r in public.items() if r['root']==tape['root']}
        for key,r in tape['records'].items():
            assert key not in seen;seen.add(key);assert r['schema']==public[key]['schema']
            assert digest(r['initial_prompt_ids'])==r['initial_prompt_sha256'];ids=[]
            for i,s in enumerate(r['steps']):
                assert s['prefix_sha256']==digest(r['initial_prompt_ids']+ids);score=s['score'];ids.append(score['free_top_token_id'])
                assert 0<=ids[-1]<166144 and math.isfinite(score['full_log_normalizer'])
                assert score['free_top_token_logit']>=max(score['selected_logits'].values())
                if i+1<len(r['steps']):assert final_answer(s['parse_text'],r['schema']) is None
                prefixes+=1
            value=final_answer(r['generated_text'],r['schema']);assert value==r['parser']['value'] and ids==r['generated_ids']
            assert (value is not None)==(r['parser']['status']=='accepted')
            g=graded[key];assert (g['prediction'],g['correct'])==(value,correctness(value,labels[key]));formats['accepted']+=value is not None
        expected={(key,t,l,s) for key in tape['records'] for t in (1,2) for l in cfg['layers_zero_based']
            for s in ('residual_after_layer','down_proj_input','down_proj_output','o_proj_input','o_proj_output')}
        got={(o['task_id'],o['loop'],o['layer_zero_based'],o['site']) for o in tape['observations']};assert len(got)==len(tape['observations']) and got==expected
        statefile=folder/tape['state_file'];assert sha(statefile)==tape['state_file_sha256']
        with np.load(statefile,allow_pickle=False) as z:
            assert set(z.files)=={o['array_key'] for o in tape['observations']}
            for o in tape['observations']:
                if o['site'].endswith('_input'):
                    module=o['site'][:-6];width=shapes[o['layer_zero_based'],module][1]
                else:width=3072
                v=z[o['array_key']];assert v.shape==(width,) and v.dtype==np.float32 and np.isfinite(v).all();states+=1
        artifacts[file.name]=sha(file);artifacts[statefile.name]=sha(statefile)
    assert len(seen)==322 and len(source_roots)==89
    telemetry=[]
    for file in folder.glob('memory_samples*.json'):
        sampled=read(file);journal=file.with_suffix('.jsonl')
        telemetry.append({'file':file.name,'samples':len(sampled),'append_journal_matches':journal.exists() and rows(journal)==sampled})
    for name in ('manifest.json','receipt.json','memory_samples.json','memory_samples.jsonl','memory_samples_resume.json','memory_samples_resume.jsonl','telemetry_errors.json'):
        file=folder/name
        if file.exists():artifacts[name]=sha(file)
    return {'status':'passed','backend':backend,'scope':'independent anchored final answers, root grades and physical state dimensions; no primary parser/grader import',
        'source_manifest_sha256':sha('experiments/nan_gym05/SOURCE_MANIFEST.json'),'execution_receipt_sha256':sha(folder/'receipt.json'),
        'task_records':len(seen),'root_groups':len(source_roots),'prefix_scores':prefixes,'state_vectors':states,'accepted_generations':formats['accepted'],
        'maximum_float32_linear_closure_relative_l2':max(closure),'telemetry':telemetry,'telemetry_errors':read(folder/'telemetry_errors.json'),
        'artifact_sha256':artifacts,'sealed_test_accessed':False,'BPD_transfer_attempted':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.runs),a.backend);write(a.out,result);print({k:v for k,v in result.items() if k!='artifact_sha256'})

if __name__=='__main__':main()
