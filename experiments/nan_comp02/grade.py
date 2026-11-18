"""Independent integer verification, source-use gate and root-aware evidence."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
from experiments.nan_bpd01.verify_sources import unary

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_sources(cohort):
    cohort=Path(cohort);labels={c['case_id']:c for c in map(json.loads,(cohort/'host_labels.jsonl').read_text().splitlines())}
    public={c['case_id']:c for c in map(json.loads,(cohort/'public.jsonl').read_text().splitlines())};pairs=json.loads((cohort/'pairs.json').read_text())
    for cid,c in labels.items():
        r=public[cid];assert [unary(x,c['operator']) for x in c['inputs']]==c['outputs'];assert unary(c['query'],c['operator'])==c['future']
        assert [x[:5] for x in c['inputs']]==r['inputs'] and [y[:5] for y in c['outputs']]==r['outputs'] and c['query'][:5]==r['query']
        signature=hashlib.sha256(json.dumps({k:v for k,v in r.items() if k not in ['case_id','root','split','family','public_sha256']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert signature==r['public_sha256']
        valid=[op for op in 'INSR' if all(unary(x,op)[:5]==y[:5] for x,y in zip(c['inputs'],c['outputs']))];assert valid==[c['operator']]
    for p in pairs:
        c,d=labels[p['recipient']],labels[p['donor']];assert c['inputs']==d['inputs'] and c['query']==d['query'] and c['future']!=d['future']
        assert c['operator']!=d['operator']
    assert Counter(c['operator'] for c in labels.values() if c['split']=='development')==dict.fromkeys('INSR',4)
    return labels,pairs

def correct(c,op):
    if op not in ('I','N','S','R'):return 0
    return int(all(unary(x,op)[:5]==y[:5] for x,y in zip(c['inputs'],c['outputs'])) and unary(c['query'],op)==c['future'])

def gate(grades,pairs,cfg):
    dev=[p for p in pairs if p['split']=='development'];ids=[p[r] for p in dev for r in ['recipient','donor']]
    main=sum(grades[cid]['worked_native2']['future_correct'] for cid in ids)
    null=sum(grades[cid]['erased_native2']['future_correct'] for cid in ids)
    both=sum(all(grades[p[r]]['worked_native2']['future_correct'] for r in ['recipient','donor']) for p in dev)
    blindboth=sum(all(grades[p[r]]['erased_native2']['future_correct'] for r in ['recipient','donor']) for p in dev)
    assert blindboth==0,'Identical blinded inputs cannot recover both distinct targets'
    difference=Q(main-null,len(ids))
    requirements={'free_verified_future':main>=cfg['minimum_free_future_correct'],'both_world_pairs':both>=cfg['minimum_both_correct_pairs'],
                  'source_use_difference':difference>=Q(cfg['minimum_source_use_difference'])}
    return {'passed':all(requirements.values()),'requirements':requirements,'free_future_correct':main,'cases':len(ids),'both_correct_pairs':both,'roots':len(dev),
            'erased_future_correct':null,'erased_both_correct_pairs':blindboth,'source_use_difference_exact':str(difference),'source_use_difference':float(difference)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--evalmath-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);cfg=json.loads(Path(__file__).with_name('config.json').read_text());labels,pairs=verify_sources(runs/'cohort')
    sys.path.insert(0,str(Path(a.evalmath_root).resolve()/'src'))
    from spiral_ln.evaluation_finite_sample import evaluate_manifest,signflip_cluster_p,holm_adjust
    alias=json.loads((runs/'cohort/source_receipt.json').read_text())['aliases'];graded={};summaries={};gates={};flat=[];repeats={}
    for backend in ['bf16','nf4']:
        receipt=json.loads((runs/backend/'receipt.json').read_text());assert receipt['status']=='completed' and len(receipt['completed_cases'])==20
        g={};raw={};root_repeat=[]
        for cid,c in labels.items():
            record=json.loads((runs/backend/(cid.replace(':','_')+'.json')).read_text());raw[cid]=record;g[cid]={}
            for arm,r in record['records'].items():
                free=next((op for op,ids in alias.items() if r['free_top_token_id'] in ids),None);assert free==r['free_operator']
                values={op:float(v) for op,v in r['class_logprobs'].items()};assert all(math.isfinite(x) for x in values.values())
                predicted=max('INSR',key=lambda op:values[op]);assert predicted==r['constrained_operator']
                result={'future_correct':correct(c,free),'constrained_future_correct':correct(c,predicted),'free_operator':free,
                    'constrained_operator':predicted,'format_valid':int(free is not None),'grammar_mass':r['grammar_mass'],'source_nll':-values[c['operator']]}
                g[cid][arm]=result
                flat.append({'backend':backend,'case_id':cid,'root':c['root'],'split':c['split'],'operator':c['operator'],'arm':arm,**result})
            g[cid]['constant_I']={'future_correct':correct(c,'I'),'constrained_future_correct':correct(c,'I'),'free_operator':'I','format_valid':1}
        for pair in pairs:
            c,d=raw[pair['recipient']],raw[pair['donor']]
            assert c['prompt_token_hashes']['erased']==d['prompt_token_hashes']['erased']
            for arm in ['erased_native2','erased_repeat_native2']:assert c['records'][arm]==d['records'][arm]
            original=c['records']['erased_native2'];repeat=c['records']['erased_repeat_native2']
            root_repeat.append({'root':pair['root'],'split':pair['split'],'max_selected_logit_error':max(abs(original['selected_logits'][i]-repeat['selected_logits'][i]) for i in original['selected_logits']),
                'full_log_normalizer_error':abs(original['full_log_normalizer']-repeat['full_log_normalizer']),'same_free_token':original['free_top_token_id']==repeat['free_top_token_id']})
        graded[backend]=g;repeats[backend]=root_repeat;gates[backend]=gate(g,pairs,cfg);summaries[backend]={}
        for split in ['calibration','development']:
            ids=[cid for cid,c in labels.items() if c['split']==split];summary={}
            for arm in cfg['arms']+['constant_I']:
                rr=[g[cid][arm] for cid in ids]
                summary[arm]={'cases':len(rr),'future_correct':sum(r['future_correct'] for r in rr),'constrained_future_correct':sum(r['constrained_future_correct'] for r in rr),
                    'format_valid':sum(r['format_valid'] for r in rr),'mean_grammar_mass':sum(r['grammar_mass'] for r in rr)/len(rr) if arm!='constant_I' else None,
                    'mean_source_nll':sum(r['source_nll'] for r in rr)/len(rr) if arm!='constant_I' else None}
            summaries[backend][split]=summary
    tests=[];pvalues=[];dev_ids=[cid for cid,c in labels.items() if c['split']=='development']
    specs=[(b,'worked_native2',b,base) for b in ['bf16','nf4'] for base in ['erased_native2','direct_native2','worked_native1','constant_I']]
    specs.append(('nf4','worked_native2','bf16','worked_native2'))
    for cb,ca,bb,ba in specs:
        name=f'{cb}/{ca} minus {bb}/{ba}'
        cases=[{'case_id':cid,'cluster_id':str(labels[cid]['root']),'candidate':graded[cb][cid][ca]['future_correct'],'baseline':graded[bb][cid][ba]['future_correct']} for cid in dev_ids]
        manifest={'schema_version':1,'experiment_id':'NAN-COMP-02/'+name,'target_population':'fresh unary development roots on one checkpoint',
            'evaluation_mode':'sampled','alpha':'0.05','minimum_effect':'0.25','planned_comparisons':9,'min_independent_clusters':20,
            'hypothesis_preregistered':True,'sealed_readout':False,'clusters_independent':True,'exchangeable_pair_labels':False,'cases':cases}
        fsi=evaluate_manifest(manifest);groups=defaultdict(list)
        for c in cases:groups[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
        effects=[sum(x,Q(0))/len(x) for x in groups.values()];conditional,method=signflip_cluster_p(effects)
        tests.append({'contrast':name,'manifest':manifest,'fsi':fsi,'conditional_p_exact':str(conditional),'conditional_method':method,'root_effects_exact':{k:str(sum(v,Q(0))/len(v)) for k,v in groups.items()}});pvalues.append(conditional)
    for r,q in zip(tests,holm_adjust(pvalues)):r['conditional_holm_exact']=str(q)
    for backend in ['bf16','nf4']:
        pilot=[]
        for pair in pairs:
            if pair['split']!='development':continue
            ids=[pair[r] for r in ['recipient','donor']]
            effect=sum(Q(graded[backend][cid]['worked_native2']['future_correct']-graded[backend][cid]['erased_native2']['future_correct']) for cid in ids)/2
            sham=sum(Q(graded[backend][cid]['erased_native2']['future_correct']-graded[backend][cid]['erased_repeat_native2']['future_correct']) for cid in ids)/2
            pilot.append({'unit_id':str(pair['root']),'paired_effect':str(effect),'sham_effects':[str(sham)]})
        spec={'schema_version':1,'experiment_id':f'NAN-COMP-02/{backend}/source-use','independent_unit':'source root','alpha':'0.05','scientific_min_effect':'0.25','target_effect':'0.50',
            'measurement_resolution':'1/16','sham_quantile':'0.95','sham_confidence':'0.95','planned_comparisons':9,'confirm_units_planned':60,
            'pilot_from_development_only':True,'heldout_sealed':True,'units_independent':True,'pilot':pilot,'sham_scope':'fresh repeated erased forward on identical labels; not semantic-intervention tail calibration'}
        write(out/f'{backend}_power_input.json',spec);tool=Path(a.evalmath_root).resolve()/'skills/evalmath-mechinterp-power/scripts/preflight.py'
        run=subprocess.run([sys.executable,str(tool),'--pilot',str(out/f'{backend}_power_input.json'),'--strict'],capture_output=True,text=True,encoding='utf-8')
        write(out/f'{backend}_power_result.json',{'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'planner_sha256':sha(tool)})
    write(out/'graded_cases.json',flat);write(out/'fsi_contrasts.json',tests)
    write(out/'results.json',{'schema_version':1,'experiment':'NAN-COMP-02','evidence_class':'development-only competence pilot; no BPD transfer or confirmation',
        'independent_unit':'fresh source root','development_roots':8,'development_cases':16,'calibration_roots':2,'calibration_cases':4,'checkpoint_count':1,
        'summaries':summaries,'gates':gates,'joint_competence_gate':all(r['passed'] for r in gates.values()),'repeated_erasure':repeats,
        'comparisons':9,'BPD_transfer_attempted':False,'original_compound_gate_changed':False,'sealed_168_case_test_accessed':False,
        'provenance':{'grader_sha256':sha(__file__),'independent_checker_sha256':sha(Path(unary.__code__.co_filename)),'public_sha256':sha(runs/'cohort/public.jsonl'),'labels_sha256':sha(runs/'cohort/host_labels.jsonl')}})
    print(json.dumps({'gates':gates,'development':{b:r['development'] for b,r in summaries.items()}},indent=2))

if __name__=='__main__':main()
