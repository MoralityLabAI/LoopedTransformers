"""Independent stdlib source/answer/metric/statistical receipt audit."""
import argparse
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
def digest(x):return hashlib.sha256(json.dumps(x).encode()).hexdigest()

def final_value(text,kind,eos):
    if len(text)>64:return None
    if kind!='integer':
        s=''.join(text.split());chars='INSR' if kind=='operator' else 'ABCD'
        m=re.fullmatch(r'(?:('+f'[{chars}]'+r')|(\*\*|\*|`)('+f'[{chars}]'+r')\2)\.?',s)
        return (m[1] or m[3]) if m else None
    if re.search(r'[0-9]\s+[0-9]',text):return None
    s=''.join(text.replace('\u2212','-').split());num=r'(?:-?(?:0|[1-9][0-9]{0,2})|\+(?:0|[1-9][0-9]{0,2}))'
    patterns=[r'\[('+num+r')\]',r'\*\*('+num+r')\*\*',r'\*('+num+r')\*',r'`('+num+r')`',r'\*\*\[('+num+r')\]\*\*',r'`\[('+num+r')\]`']
    for pat in patterns:
        m=re.fullmatch(pat+r'\.?',s)
        if m:return int(m[1])
    if eos or bool(text and text[-1].isspace()) or s.endswith('.'):
        m=re.fullmatch('('+num+r')\.?',s)
        if m:return int(m[1])
    return None

def audit(root):
    fixtures=root/'fixtures';receipts=root/'receipts';frozen=read(root/'SOURCE_MANIFEST.json');cfg=read(root/'config.json')
    for n,h in frozen['critical_files_sha256'].items():assert sha(root/n)==h
    for n,h in frozen['model_input_files_sha256'].items():assert sha(fixtures/n)==h
    assert sha(fixtures/'host_labels.jsonl')==frozen['host_labels_sha256']
    labels={r['root']:r for r in map(json.loads,(fixtures/'host_labels.jsonl').read_text().splitlines())}
    public={r['root']:r for r in map(json.loads,(fixtures/'public.jsonl').read_text().splitlines())};source=read(fixtures/'source_receipt.json')
    seen=set()
    for k,c in labels.items():
        for x in c['inputs']:
            assert len(x)==33 and x[0]==1 and tuple(x[:5]) not in seen;seen.add(tuple(x[:5]))
        assert {g:[unary(x,g)[:5] for x in c['inputs']] for g in 'INSR'}==c['refs']
        assert all(unary(c['query'],g)==c['futures'][g] for g in c['worlds'])
    assert len(seen)==20 and len(labels)==10
    result=read(receipts/'results.json');flat=read(receipts/'graded_records.json');saved=read(receipts/'root_metrics.json');goldrows={};metrics={};steps=0
    for b in ('bf16','nf4'):
        receipt=read(receipts/b/'receipt.json');assert receipt['status']=='completed' and set(receipt['completed_roots'])==set(labels)
        assert receipt['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json') and receipt['critical_files_sha256']==frozen['critical_files_sha256']
        metrics[b]={}
        for k,c in labels.items():
            tape=read(receipts/b/f'root_{k}.json');assert tape['root']==k;pred={}
            assert set(tape['records'])==set(public[k]['prompts']) and len(tape['records'])==24
            for arm,r in tape['records'].items():
                assert r['kind']==public[k]['prompts'][arm]['kind'];assert digest(r['initial_prompt_ids'])==r['initial_prompt_sha256'];ids=[]
                cap=12 if r['kind']=='integer' else 8;assert len(r['initial_prompt_ids'])<=768 and 1<=len(r['steps'])<=cap
                for i,s in enumerate(r['steps']):
                    assert s['step']==i+1 and s['prefix_sha256']==digest(r['initial_prompt_ids']+ids);score=s['score'];ids.append(score['free_top_token_id'])
                    assert math.isfinite(score['full_log_normalizer']) and 0<=ids[-1]<166144 and score['free_top_token_logit']>=max(score['selected_logits'].values())
                    for op,aliases in receipt['aliases'].items():
                        vals=[score['selected_logits'][str(v)] for v in aliases];mx=max(vals);lp=mx+math.log(sum(math.exp(v-mx) for v in vals))-score['full_log_normalizer']
                        assert abs(lp-score['class_logprobs'][op])<1e-12
                    if i+1<len(r['steps']):assert final_value(s['parse_text'],r['kind'],ids[-1]==receipt['eos_token_id']) is None
                    steps+=1
                v=final_value(r['steps'][-1]['parse_text'],r['kind'],ids[-1]==receipt['eos_token_id'])
                assert v==r['parser']['value'] and (v is not None)==(r['parser']['status']=='accepted') and ids==r['generated_ids'] and len(ids)==r['tokens_generated']
                pred[arm]=v;prefix,n=arm.split('/') if '/' in arm else (arm,None)
                if prefix in ('free','scaffold','copy'):expected=c['refs']['R'][0][int(n[1:])];scorevalue=int(v==expected)
                elif n is not None:
                    op=c['worlds'][int(n)];expected=c['codes'][op] if prefix in ('coded','codecopy') else op;scorevalue=int(v==expected)
                else:expected=c['worlds'];scorevalue=sum(v==op for op in expected)/2
                goldrows[b,k,arm]=(v,expected,scorevalue)
            m={}
            for arm in ('free','scaffold','copy'):
                vector=[1]+[pred[f'{arm}/b{n}'] for n in range(1,5)]
                for n in range(1,5):m[f'{arm}_b{n}']=int(vector[n]==c['refs']['R'][0][n])
                # Independent multiplicative certificate, not just a copied target vector.
                m[arm+'_full']=int(all(v is not None for v in vector) and all(sum(c['inputs'][0][j]*vector[n-j] for j in range(n+1))==(1 if n==0 else 0) for n in range(5)))
            for i,op in enumerate(c['worlds']):
                suffix='R' if op=='R' else 'alt'
                for arm in ('select','table','coded','labelcopy','codecopy'):
                    exp=c['codes'][op] if arm in ('coded','codecopy') else op;m[arm+'_'+suffix]=int(pred[f'{arm}/{i}']==exp)
                for arm in ('erased','erased_repeat'):m[arm+'_'+suffix]=int(pred[arm]==op)
            for arm in ('select','table','coded','labelcopy','codecopy','erased','erased_repeat'):
                m[arm+'_both']=m[arm+'_R']*m[arm+'_alt'];m[arm+'_macro']=(m[arm+'_R']+m[arm+'_alt'])/2
            assert m['erased_both']==m['erased_repeat_both']==0
            m['controls']=m['copy_full']*m['labelcopy_both']*m['codecopy_both'];assert m==saved[b][str(k)];metrics[b][k]=m
        for split in ('calibration','development'):
            rr=[m for k,m in metrics[b].items() if labels[k]['split']==split];s=result['summaries'][b][split]
            assert s['roots']==len(rr) and all(s[key]==sum(v[key] for v in rr) for key in rr[0])
    assert len(flat)==480
    for r in flat:assert (r['prediction'],r['expected'],r['correct'])==goldrows[r['backend'],r['root'],r['arm']]
    tests=read(receipts/'fsi_contrasts.json');assert len(tests)==15;ps=[]
    for t in tests:
        left,right=t['contrast'].split(' minus ');cb,ca=left.split('/');bb,ba=right.split('/');effects=[]
        for c in t['manifest']['cases']:
            k=int(c['case_id']);assert c['cluster_id']==str(k);assert (c['candidate'],c['baseline'])==(metrics[cb][k][ca],metrics[bb][k][ba]);effects.append(Q(c['candidate']-c['baseline']))
        assert len(effects)==8;observed=abs(sum(effects,Q(0)))
        pv=Q(sum(abs(sum((s*x for s,x in zip(signs,effects)),Q(0)))>=observed for signs in product((-1,1),repeat=8)),256)
        assert Q(t['conditional_p_exact'])==pv and t['fsi']['p_unadjusted'] is None;ps.append(pv)
    adjusted=[Q(1)]*15;running=Q(0)
    for rank,i in enumerate(sorted(range(15),key=lambda i:ps[i])):running=max(running,min(Q(1),(15-rank)*ps[i]));adjusted[i]=running
    assert all(Q(t['conditional_holm_exact'])==pv for t,pv in zip(tests,adjusted))
    assert not result['BPD_transfer_attempted'] and not result['sealed_168_case_test_accessed']
    return {'status':'passed','scope':'independent regex/final-answer, convolution-certificate, scalar/root and exact-statistics audit; no model, Torch, EvalMath or primary parser/grader import',
        'source_roots':10,'new_individual_polynomials':20,'generation_records':480,'prefix_score_records':steps,'conditional_tests':15,'Holm_family':15,'sealed_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',default='experiments/nan_diag04');p.add_argument('--out',required=True);a=p.parse_args();r=audit(Path(a.experiment_root))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(r))

if __name__=='__main__':main()
