"""Independent stdlib published-receipt audit, including parser and exact grades."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
from experiments.nan_bpd01.verify_sources import unary

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
def answer(text,final):
    if len(text)>64:return 'rejected',None
    value=''.join(ch for ch in text if not ch.isspace())
    m=re.fullmatch(r'(?:([INSR])|(\*\*|\*|`)([INSR])\2)\.?',value)
    if m:return 'accepted',m[1] or m[3]
    words=[f'{left}{op}{right}{period}' for left,right in (('',''),('**','**'),('*','*'),('`','`')) for op in 'INSR' for period in ('','.')]
    return ('pending' if not final and any(word.startswith(value) for word in words) else 'rejected'),None
def correct(c,op):return int(op in ('I','N','S','R') and all(unary(x,op)==y for x,y in zip(c['inputs'],c['outputs'])) and unary(c['query'],op)==c['future'])

def audit(root):
    fixtures=root/'fixtures';receipts=root/'receipts';frozen=read(root/'SOURCE_MANIFEST.json');cfg=read(root/'config.json')
    labels={r['case_id']:r for r in map(json.loads,(fixtures/'host_labels.jsonl').read_text().splitlines())}
    public={r['case_id']:r for r in map(json.loads,(fixtures/'public.jsonl').read_text().splitlines())};pairs=read(fixtures/'pairs.json')
    for name,h in frozen['cohort_files_sha256'].items():assert sha(fixtures/name)==h
    assert sha(fixtures/'host_labels.jsonl')==frozen['host_only_labels_sha256']
    for name,h in frozen['critical_files_sha256'].items():assert sha(root/name)==h
    assert len(labels)==20 and len(pairs)==10
    dev=[cid for cid,c in labels.items() if c['split']=='development'];pp=[p for p in pairs if p['split']=='development']
    assert len(dev)==16 and len(pp)==8 and Counter(labels[cid]['operator'] for cid in dev)==dict.fromkeys('INSR',4)
    for cid,c in labels.items():
        assert correct(c,c['operator']) and len(c['future'])==33
        assert [op for op in 'INSR' if all(unary(x,op)[:5]==y[:5] for x,y in zip(c['inputs'],c['outputs']))]==[c['operator']]
        assert public[cid]['inputs']==[x[:5] for x in c['inputs']] and public[cid]['outputs']==[x[:5] for x in c['outputs']]
    alias=read(receipts/'source_receipt.json')['aliases'];results=read(receipts/'results.json');flat=read(receipts/'graded_cases.json');g={};tapes={};stepcount=0
    for backend in ('bf16','nf4'):
        receipt=read(receipts/backend/'receipt.json');assert receipt['status']=='completed' and set(receipt['completed_cases'])==set(labels)
        assert receipt['critical_files_sha256']==frozen['critical_files_sha256'] and receipt['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
        tapes[backend]={}
        for cid,c in labels.items():
            tape=read(receipts/backend/(cid.replace(':','_')+'.json'));tapes[backend][cid]=tape;assert tape['case_id']==cid
            for arm,r in tape['records'].items():
                prefix=list(r['initial_prompt_ids']);assert digest(prefix)==r['initial_prompt_sha256'];assert len(prefix)<=768
                generated=[];assert 1<=len(r['steps'])<=8
                for i,s in enumerate(r['steps']):
                    assert s['step']==i+1 and s['prefix_sha256']==digest(prefix+generated)
                    score=s['score'];token=score['free_top_token_id'];generated.append(token);assert 0<=token<166144
                    op=next((op for op,ids in alias.items() if token in ids),None);assert op==score['free_operator']
                    lp={}
                    for op,ids in alias.items():
                        vals=[score['selected_logits'][str(k)] for k in ids];mx=max(vals)
                        lp[op]=mx+math.log(sum(math.exp(v-mx) for v in vals))-score['full_log_normalizer']
                        assert math.isfinite(lp[op]) and abs(lp[op]-score['class_logprobs'][op])<1e-12
                    assert abs(sum(math.exp(v) for v in lp.values())-score['grammar_mass'])<1e-12
                    assert score['free_top_token_logit']>=max(score['selected_logits'].values())
                    assert max('INSR',key=lambda op:lp[op])==score['constrained_operator']
                    status,op=answer(s['decoded_prefix'],token==receipt['eos_token_id'] or i+1==8)
                    if i+1<len(r['steps']):assert status=='pending'
                    stepcount+=1
                assert status!='pending' and (status,op)==(r['parser']['status'],r['parser']['operator'])
                assert generated==r['generated_ids'] and len(generated)==r['tokens_generated'] and r['generated_text']==r['steps'][-1]['decoded_prefix']
                g[backend,cid,arm]=(correct(c,op),correct(c,r['steps'][0]['score']['constrained_operator']),op)
            strict=tape['records']['curriculum']['steps'][0]['score']['free_operator']
            g[backend,cid,'strict_first']=(correct(c,strict),g[backend,cid,'curriculum'][1],strict)
            g[backend,cid,'constant_I']=(correct(c,'I'),correct(c,'I'),'I')
        for pair in pairs:
            a,b=[tapes[backend][pair[k]] for k in ('recipient','donor')]
            for arm in ('erased','erased_repeat'):assert a['records'][arm]==b['records'][arm]
            assert labels[pair['recipient']]['operator']!=labels[pair['donor']]['operator']
        for split in ('calibration','development'):
            ids=[cid for cid,c in labels.items() if c['split']==split]
            for arm in cfg['arms']:
                s=results['summaries'][backend][split][arm]
                assert len(ids)==s['cases'] and sum(g[backend,cid,arm][0] for cid in ids)==s['future_correct'] and sum(g[backend,cid,arm][1] for cid in ids)==s['restricted_future_correct']
        main=sum(g[backend,cid,'curriculum'][0] for cid in dev);null=sum(g[backend,cid,'erased'][0] for cid in dev)
        both=sum(all(g[backend,p[k],'curriculum'][0] for k in ('recipient','donor')) for p in pp)
        reciprocal=sum(g[backend,cid,'curriculum'][0] for cid in dev if labels[cid]['operator']=='R')
        gate=results['gates'][backend];assert (main,null,both,reciprocal)==(gate['future_correct'],gate['erased_correct'],gate['both_correct_pairs'],gate['reciprocal_correct'])
        assert gate['passed']==(main>=12 and both>=6 and reciprocal>=3 and Q(main-null,16)>=Q(1,4))
        power=read(receipts/f'{backend}_power_input.json')
        for pair,u in zip(pp,power['pilot']):
            ids=[pair[k] for k in ('recipient','donor')];effect=sum(Q(g[backend,cid,'curriculum'][0]-g[backend,cid,'erased'][0]) for cid in ids)/2
            sham=sum(Q(g[backend,cid,'erased'][0]-g[backend,cid,'erased_repeat'][0]) for cid in ids)/2
            assert str(pair['root'])==u['unit_id'] and Q(u['paired_effect'])==effect and list(map(Q,u['sham_effects']))==[sham]
    assert len(flat)==240
    for r in flat:assert (r['future_correct'],r['restricted_future_correct'],r['operator'])==g[r['backend'],r['case_id'],r['arm']]
    tests=read(receipts/'fsi_contrasts.json');assert len(tests)==9;ps=[]
    for t in tests:
        left,right=t['contrast'].split(' minus ');cb,ca=left.split('/');bb,ba=right.split('/');groups=defaultdict(list)
        for c in t['manifest']['cases']:
            cid=c['case_id'];assert c['cluster_id']==str(labels[cid]['root'])
            assert (c['candidate'],c['baseline'])==(g[cb,cid,ca][0],g[bb,cid,ba][0]);groups[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
        effects=[sum(v,Q(0))/len(v) for v in groups.values()];assert len(effects)==8;observed=abs(sum(effects,Q(0)))
        p=Q(sum(abs(sum((s*x for s,x in zip(signs,effects)),Q(0)))>=observed for signs in product((-1,1),repeat=8)),256)
        assert Q(t['conditional_p_exact'])==p and t['fsi']['p_unadjusted'] is None and t['fsi']['n_clusters']==8;ps.append(p)
    adjusted=[Q(1)]*9;running=Q(0)
    for rank,i in enumerate(sorted(range(9),key=lambda i:ps[i])):running=max(running,min(Q(1),(9-rank)*ps[i]));adjusted[i]=running
    assert all(Q(t['conditional_holm_exact'])==v for t,v in zip(tests,adjusted))
    assert not results['BPD_transfer_attempted'] and not results['sealed_168_case_test_accessed']
    return {'status':'passed','scope':'independent stdlib checker and regex grammar; no Torch, model, EvalMath or primary parser/grader import',
        'source_worlds_verified':20,'scalar_grades_verified':240,'generation_steps_verified':stepcount,'conditional_root_tests_verified':9,'Holm_family_verified':9,'sealed_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',default='experiments/nan_comp03');p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(Path(a.experiment_root));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))

if __name__=='__main__':main()
