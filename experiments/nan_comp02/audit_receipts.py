"""Independent stdlib audit of NAN-COMP-02 published source and model tapes."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as Q
from itertools import product
import hashlib
import json
import math
from pathlib import Path
from experiments.nan_bpd01.verify_sources import unary
from experiments.nan_comp02.prompt import body

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(c, op):
    return int(op in ('I','N','S','R') and all(unary(x,op)==y for x,y in zip(c['inputs'],c['outputs'])) and unary(c['query'],op)==c['future'])

def audit(root):
    fixtures=root/'fixtures'; receipts=root/'receipts'
    labels={c['case_id']:c for c in map(json.loads,(fixtures/'host_labels.jsonl').read_text().splitlines())}
    public={c['case_id']:c for c in map(json.loads,(fixtures/'public.jsonl').read_text().splitlines())}
    pairs=read(fixtures/'pairs.json'); source=read(receipts/'source_receipt.json')
    for n,h in source['files_sha256'].items(): assert sha(fixtures/n)==h
    for cid,c in labels.items():
        assert checked(c,c['operator'])==1
        assert [g for g in 'INSR' if checked(c,g)]==[c['operator']]
        r=public[cid]
        assert r['inputs']==[x[:5] for x in c['inputs']] and r['outputs']==[x[:5] for x in c['outputs']]
        assert r['query']==c['query'][:5] and (r['root'],r['split'])==(c['root'],c['split'])
        assert len(c['future'])==33
    devpairs=[p for p in pairs if p['split']=='development']
    dev=[cid for p in devpairs for cid in (p['recipient'],p['donor'])]
    assert len(labels)==20 and len(devpairs)==8 and len(dev)==16
    assert Counter(labels[cid]['operator'] for cid in dev)==dict.fromkeys('INSR',4)
    clean=defaultdict(list); erased=defaultdict(set)
    for cid,r in public.items(): clean[body(r,'worked')].append(cid); erased[body(r,'erased')].add(r['root'])
    overlaps=[v for v in clean.values() if len(v)>1]
    assert overlaps==[['nan-comp02:182601:recipient','nan-comp02:182705:recipient']]
    assert len({body(public[cid],'worked') for cid in dev})==16
    assert len({body(public[p['recipient']],'erased') for p in devpairs})==8
    alias=source['aliases']; flat=read(receipts/'graded_cases.json'); results=read(receipts/'results.json')
    bykey={}; raw={}; cfg=read(root/'config.json')
    for backend in ('bf16','nf4'):
        raw[backend]={}
        receipt=read(receipts/backend/'receipt.json')
        assert receipt['status']=='completed' and set(receipt['completed_cases'])==set(labels)
        for cid in labels:
            tape=read(receipts/backend/(cid.replace(':','_')+'.json')); raw[backend][cid]=tape
            assert tape['case_id']==cid and tape['candidate_rows']==1
            for arm,r in tape['records'].items():
                free=next((op for op,ids in alias.items() if r['free_top_token_id'] in ids),None)
                assert free==r['free_operator']
                lp={}
                for op,ids in alias.items():
                    vals=[r['selected_logits'][str(i)] for i in ids]; mx=max(vals)
                    lp[op]=mx+math.log(sum(math.exp(x-mx) for x in vals))-r['full_log_normalizer']
                    assert abs(lp[op]-r['class_logprobs'][op])<1e-12
                assert abs(sum(math.exp(x) for x in lp.values())-r['grammar_mass'])<1e-12
                assert r['free_top_token_logit']>=max(r['selected_logits'].values())
                if free is not None: assert r['selected_logits'][str(r['free_top_token_id'])]==r['free_top_token_logit']
                restricted=max('INSR',key=lambda op:lp[op]); assert restricted==r['constrained_operator']
                bykey[backend,cid,arm]={'future_correct':checked(labels[cid],free),'constrained_future_correct':checked(labels[cid],restricted),
                    'free_operator':free,'constrained_operator':restricted,'format_valid':int(free is not None),'grammar_mass':r['grammar_mass'],'source_nll':-lp[labels[cid]['operator']]}
            bykey[backend,cid,'constant_I']={'future_correct':checked(labels[cid],'I'),'constrained_future_correct':checked(labels[cid],'I'),'format_valid':1}
        for p in pairs:
            c,d=[raw[backend][p[role]] for role in ('recipient','donor')]
            assert c['prompt_token_hashes']['erased']==d['prompt_token_hashes']['erased']
            assert c['prompt_lengths']==d['prompt_lengths']
            for arm in ('erased_native2','erased_repeat_native2'): assert c['records'][arm]==d['records'][arm]
            assert labels[p['recipient']]['operator']!=labels[p['donor']]['operator']
            assert body(public[p['recipient']],'erased')==body(public[p['donor']],'erased')
        for split in ('calibration','development'):
            ids=[cid for cid,c in labels.items() if c['split']==split]
            for arm in cfg['arms']+['constant_I']:
                summary=results['summaries'][backend][split][arm]; rr=[bykey[backend,cid,arm] for cid in ids]
                assert len(rr)==summary['cases']
                for k in ('future_correct','constrained_future_correct','format_valid'): assert sum(r[k] for r in rr)==summary[k]
        g=results['gates'][backend]
        main=sum(bykey[backend,cid,'worked_native2']['future_correct'] for cid in dev)
        null=sum(bykey[backend,cid,'erased_native2']['future_correct'] for cid in dev)
        both=sum(all(bykey[backend,p[role],'worked_native2']['future_correct'] for role in ('recipient','donor')) for p in devpairs)
        blindboth=sum(all(bykey[backend,p[role],'erased_native2']['future_correct'] for role in ('recipient','donor')) for p in devpairs)
        assert (main,null,both,blindboth)==(g['free_future_correct'],g['erased_future_correct'],g['both_correct_pairs'],g['erased_both_correct_pairs'])
        assert blindboth==0 and Q(g['source_use_difference_exact'])==Q(main-null,16)
        assert g['passed']==(main>=12 and both>=6 and Q(main-null,16)>=Q(1,4))
        power=read(receipts/f'{backend}_power_input.json')
        for p,u in zip(devpairs,power['pilot']):
            assert u['unit_id']==str(p['root'])
            ids=[p[role] for role in ('recipient','donor')]
            effect=sum(Q(bykey[backend,cid,'worked_native2']['future_correct']-bykey[backend,cid,'erased_native2']['future_correct']) for cid in ids)/2
            sham=sum(Q(bykey[backend,cid,'erased_native2']['future_correct']-bykey[backend,cid,'erased_repeat_native2']['future_correct']) for cid in ids)/2
            assert Q(u['paired_effect'])==effect and list(map(Q,u['sham_effects']))==[sham]
    assert len(flat)==240
    for r in flat:
        assert r['root']==labels[r['case_id']]['root'] and r['split']==labels[r['case_id']]['split']
        expected=bykey[r['backend'],r['case_id'],r['arm']]
        assert all(r[k]==v for k,v in expected.items())
    tests=read(receipts/'fsi_contrasts.json'); assert len(tests)==9
    ps=[]
    for t in tests:
        left,right=t['contrast'].split(' minus '); cb,ca=left.split('/'); bb,ba=right.split('/')
        groups=defaultdict(list)
        for c in t['manifest']['cases']:
            cid=c['case_id']; assert c['cluster_id']==str(labels[cid]['root'])
            assert c['candidate']==bykey[cb,cid,ca]['future_correct'] and c['baseline']==bykey[bb,cid,ba]['future_correct']
            groups[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
        effects=[sum(v,Q(0))/len(v) for v in groups.values()]; assert len(effects)==8
        observed=abs(sum(effects,Q(0)))
        p=Q(sum(abs(sum((s*x for s,x in zip(signs,effects)),Q(0)))>=observed for signs in product((-1,1),repeat=8)),256)
        assert Q(t['conditional_p_exact'])==p and t['fsi']['n_clusters']==8 and t['fsi']['p_unadjusted'] is None
        assert t['fsi']['cluster_effect_exact']==str(sum(effects,Q(0))/8)
        ps.append(p)
    adjusted=[Q(1)]*9; running=Q(0)
    for rank,i in enumerate(sorted(range(9),key=lambda i:ps[i])):
        running=max(running,min(Q(1),(9-rank)*ps[i])); adjusted[i]=running
    assert all(Q(t['conditional_holm_exact'])==p for t,p in zip(tests,adjusted))
    assert not results['BPD_transfer_attempted'] and not results['sealed_168_case_test_accessed']
    assert results['joint_competence_gate']==all(g['passed'] for g in results['gates'].values())
    critical=['config.json','prompt.py','head.py','cohort.py','run.py','grade.py','validate_gpu.py','PROTOCOL.md','SOURCE_MANIFEST.json']
    manifests={b:read(receipts/b/'manifest.json') for b in ('bf16','nf4')}
    registered=read(receipts/'registration_file_hashes.json')
    assert registered['commit']=='2ec61819220d39db2dc3561d59708124e5703cf9'
    bound={r['name']:r for r in registered['files']}
    for name in critical:
        assert manifests['bf16']['source_sha256'][name]==manifests['nf4']['source_sha256'][name]==bound[name]['working_sha256']
        assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==bound[name]['git_sha256']
    return {'status':'passed','evidence_class':'independent stdlib source/grade audit; no Torch, model or EvalMath import',
        'worlds_verified':20,'independent_development_roots':8,'development_worlds':16,'graded_rows_verified':240,
        'exact_root_signflip_tests_recomputed':9,'holm_family_recomputed':9,'critical_inference_files_unchanged_between_backends':critical,
        'calibration_development_prompt_overlap':overlaps,'development_clean_prompts_distinct':16,'development_erased_root_prompts_distinct':8,
        'calibration_used_for_selection':False,'sealed_168_case_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',default='experiments/nan_comp02');p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.experiment_root));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result))

if __name__=='__main__': main()
