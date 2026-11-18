"""Separate host-only exact-source grading and frozen nine-contrast readout."""
import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from experiments.nan_comp02.grade import verify_sources,correct
from .parser import parse

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

def validate_record(r,cfg,eos,alias):
    ids=r['initial_prompt_ids'];generated=[];assert token_hash(ids)==r['initial_prompt_sha256']
    assert len(ids)<=cfg['maximum_prompt_tokens'] and 1<=len(r['steps'])<=cfg['maximum_new_tokens']
    for i,step in enumerate(r['steps']):
        assert step['step']==i+1 and step['prefix_sha256']==token_hash(ids+generated)
        score=step['score'];token=score['free_top_token_id'];generated.append(token)
        free=next((op for op,values in alias.items() if token in values),None);assert free==score['free_operator']
        lp={}
        for op,values in alias.items():
            selected=[score['selected_logits'][str(v)] for v in values];mx=max(selected)
            lp[op]=mx+math.log(sum(math.exp(v-mx) for v in selected))-score['full_log_normalizer']
            assert math.isfinite(lp[op]) and abs(lp[op]-score['class_logprobs'][op])<1e-12
        assert max('INSR',key=lambda op:lp[op])==score['constrained_operator']
        assert abs(sum(math.exp(v) for v in lp.values())-score['grammar_mass'])<1e-12
        assert score['free_top_token_logit']>=max(score['selected_logits'].values())
        result=parse(step['decoded_prefix'],final=token==eos or i+1==cfg['maximum_new_tokens'])
        if i+1<len(r['steps']):assert result['status']=='pending','Generation continued after stop'
    assert result['status']!='pending' and result==r['parser']
    assert generated==r['generated_ids'] and len(generated)==r['tokens_generated'] and r['generated_text']==r['steps'][-1]['decoded_prefix']
    return result['operator']

def gate(g,pairs,labels,cfg):
    pp=[p for p in pairs if p['split']=='development'];ids=[p[k] for p in pp for k in ('recipient','donor')]
    main=sum(g[cid]['curriculum']['future_correct'] for cid in ids);null=sum(g[cid]['erased']['future_correct'] for cid in ids)
    both=sum(all(g[p[k]]['curriculum']['future_correct'] for k in ('recipient','donor')) for p in pp)
    reciprocal=sum(g[cid]['curriculum']['future_correct'] for cid in ids if labels[cid]['operator']=='R')
    blindboth=sum(all(g[p[k]]['erased']['future_correct'] for k in ('recipient','donor')) for p in pp);assert blindboth==0
    effect=Q(main-null,len(ids));requirements={'free_parsed_future':main>=cfg['minimum_free_future_correct'],'both_world_pairs':both>=cfg['minimum_both_correct_pairs'],
        'reciprocal':reciprocal>=cfg['minimum_reciprocal_correct'],'source_use_difference':effect>=Q(cfg['minimum_source_use_difference'])}
    return {'passed':all(requirements.values()),'requirements':requirements,'future_correct':main,'cases':16,'both_correct_pairs':both,'roots':8,
        'reciprocal_correct':reciprocal,'reciprocal_cases':4,'erased_correct':null,'erased_both_correct_pairs':blindboth,'source_use_difference_exact':str(effect)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--evalmath-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);cfg=json.loads(Path(__file__).with_name('config.json').read_text());labels,pairs=verify_sources(runs/'cohort')
    alias=json.loads((runs/'cohort/source_receipt.json').read_text())['aliases'];graded={};flat=[];summaries={};gates={};repeats={}
    for backend in ('bf16','nf4'):
        receipt=json.loads((runs/backend/'receipt.json').read_text());assert receipt['status']=='completed' and set(receipt['completed_cases'])==set(labels)
        g={};raw={}
        for cid,c in labels.items():
            tape=json.loads((runs/backend/(cid.replace(':','_')+'.json')).read_text());assert tape['case_id']==cid;raw[cid]=tape;g[cid]={}
            for arm,r in tape['records'].items():
                op=validate_record(r,cfg,receipt['eos_token_id'],alias);score=r['steps'][0]['score'];restricted=score['constrained_operator']
                g[cid][arm]={'future_correct':correct(c,op),'restricted_future_correct':correct(c,restricted),'operator':op,'restricted_operator':restricted,
                    'format_valid':int(op is not None),'tokens_generated':r['tokens_generated'],'stop_reason':r['stop_reason'],
                    'first_grammar_mass':score['grammar_mass'],'source_nll':-score['class_logprobs'][c['operator']]}
            first=tape['records']['curriculum']['steps'][0]['score'];strict=first['free_operator']
            g[cid]['strict_first']=g[cid]['curriculum']|{'future_correct':correct(c,strict),'operator':strict,'format_valid':int(strict is not None),'tokens_generated':1,'stop_reason':'strict_first_token'}
            g[cid]['constant_I']={'future_correct':correct(c,'I'),'restricted_future_correct':correct(c,'I'),'operator':'I','restricted_operator':'I','format_valid':1,'tokens_generated':0,'stop_reason':'fixed_prior','first_grammar_mass':None,'source_nll':None}
            for arm in cfg['arms']:flat.append({'backend':backend,'case_id':cid,'root':c['root'],'split':c['split'],'target_operator':c['operator'],'arm':arm,**g[cid][arm]})
        rr=[]
        for pair in pairs:
            c,d=[raw[pair[k]] for k in ('recipient','donor')]
            for arm in ('erased','erased_repeat'):assert c['records'][arm]==d['records'][arm]
            original,repeat=[c['records'][arm] for arm in ('erased','erased_repeat')]
            matching=original['generated_ids']==repeat['generated_ids'];errors=[];zs=[]
            if matching:
                for x,y in zip(original['steps'],repeat['steps']):
                    errors.extend(abs(x['score']['selected_logits'][i]-y['score']['selected_logits'][i]) for i in x['score']['selected_logits'])
                    zs.append(abs(x['score']['full_log_normalizer']-y['score']['full_log_normalizer']))
            rr.append({'root':pair['root'],'split':pair['split'],'same_generated_ids':matching,'max_selected_logit_error':max(errors) if errors else None,'max_log_normalizer_error':max(zs) if zs else None})
        graded[backend]=g;repeats[backend]=rr;gates[backend]=gate(g,pairs,labels,cfg);summaries[backend]={}
        for split in ('calibration','development'):
            ids=[cid for cid,c in labels.items() if c['split']==split];summaries[backend][split]={}
            for arm in cfg['arms']:
                vals=[g[cid][arm] for cid in ids];s={k:sum(r[k] for r in vals) for k in ('future_correct','restricted_future_correct','format_valid','tokens_generated')}
                s|={'cases':len(ids),'by_operator':{op:{'correct':sum(g[cid][arm]['future_correct'] for cid in ids if labels[cid]['operator']==op),'cases':sum(labels[cid]['operator']==op for cid in ids)} for op in 'INSR'},
                    'mean_first_grammar_mass':sum(r['first_grammar_mass'] for r in vals)/len(vals) if arm!='constant_I' else None,'mean_source_nll':sum(r['source_nll'] for r in vals)/len(vals) if arm!='constant_I' else None}
                summaries[backend][split][arm]=s
    sys.path.insert(0,str(Path(a.evalmath_root).resolve()/'src'))
    from spiral_ln.evaluation_finite_sample import evaluate_manifest,signflip_cluster_p,holm_adjust
    specs=[(b,'curriculum',b,base) for b in ('bf16','nf4') for base in ('erased','balanced','strict_first','constant_I')]+[('nf4','curriculum','bf16','curriculum')]
    tests=[];pvalues=[];dev=[cid for cid,c in labels.items() if c['split']=='development']
    for cb,ca,bb,ba in specs:
        name=f'{cb}/{ca} minus {bb}/{ba}';cases=[{'case_id':cid,'cluster_id':str(labels[cid]['root']),'candidate':graded[cb][cid][ca]['future_correct'],'baseline':graded[bb][cid][ba]['future_correct']} for cid in dev]
        manifest={'schema_version':1,'experiment_id':'NAN-COMP-03/'+name,'target_population':'fresh unary source roots, source-only deduplicated, one checkpoint','evaluation_mode':'sampled',
            'alpha':'0.05','minimum_effect':'0.25','planned_comparisons':9,'min_independent_clusters':20,'hypothesis_preregistered':True,'sealed_readout':False,
            'clusters_independent':False,'exchangeable_pair_labels':False,'cases':cases}
        fsi=evaluate_manifest(manifest);groups=defaultdict(list)
        for c in cases:groups[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
        effects=[sum(v,Q(0))/len(v) for v in groups.values()];pv,method=signflip_cluster_p(effects);pvalues.append(pv)
        tests.append({'contrast':name,'manifest':manifest,'fsi':fsi,'conditional_p_exact':str(pv),'conditional_method':method,
            'conditional_scope':'Sensitivity under an imposed independent-root signed-symmetry law; not an established inferential null',
            'root_effects_exact':{k:str(sum(v,Q(0))/len(v)) for k,v in groups.items()}})
    for t,pv in zip(tests,holm_adjust(pvalues)):t['conditional_holm_exact']=str(pv)
    for backend in ('bf16','nf4'):
        pilot=[]
        for pair in pairs:
            if pair['split']!='development':continue
            ids=[pair[k] for k in ('recipient','donor')]
            effect=sum(Q(graded[backend][cid]['curriculum']['future_correct']-graded[backend][cid]['erased']['future_correct']) for cid in ids)/2
            sham=sum(Q(graded[backend][cid]['erased']['future_correct']-graded[backend][cid]['erased_repeat']['future_correct']) for cid in ids)/2
            pilot.append({'unit_id':str(pair['root']),'paired_effect':str(effect),'sham_effects':[str(sham)]})
        spec={'schema_version':1,'experiment_id':f'NAN-COMP-03/{backend}/source-use','independent_unit':'source root (deduplicated; independence unestablished)','alpha':'0.05',
            'scientific_min_effect':'0.25','target_effect':'0.50','measurement_resolution':'1/16','sham_quantile':'0.95','sham_confidence':'0.95','planned_comparisons':9,
            'confirm_units_planned':60,'pilot_from_development_only':True,'heldout_sealed':True,'units_independent':False,'pilot':pilot,
            'sham_scope':'Repeated erased generation on identical labels; numerical/format repeat noise only, not semantic-intervention tail'}
        write(out/f'{backend}_power_input.json',spec);tool=Path(a.evalmath_root).resolve()/'skills/evalmath-mechinterp-power/scripts/preflight.py'
        proc=subprocess.run([sys.executable,str(tool),'--pilot',str(out/f'{backend}_power_input.json'),'--strict'],capture_output=True,text=True,encoding='utf-8')
        write(out/f'{backend}_power_result.json',{'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'planner_sha256':sha(tool)})
    write(out/'graded_cases.json',flat);write(out/'fsi_contrasts.json',tests)
    result={'experiment':'NAN-COMP-03','evidence_class':'prospective development-only competence pilot, power-limited; no BPD transfer','sampling_unit':'source generator root',
        'development_roots':8,'development_worlds':16,'calibration_roots':2,'calibration_worlds':4,'checkpoint_count':1,'summaries':summaries,'gates':gates,
        'joint_competence_gate':all(g['passed'] for g in gates.values()),'repeated_erasure':repeats,'comparisons':9,'BPD_transfer_attempted':False,
        'sealed_168_case_test_accessed':False,'original_compound_gate_changed':False,'provenance':{'grader_sha256':sha(__file__),'public_sha256':sha(runs/'cohort/public.jsonl'),'labels_sha256':sha(runs/'cohort/host_labels.jsonl')}}
    write(out/'results.json',result);print(json.dumps({'gates':gates,'development':{b:summaries[b]['development'] for b in summaries}},indent=2))

if __name__=='__main__':main()
