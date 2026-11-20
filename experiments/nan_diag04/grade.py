"""Frozen host-only arithmetic and label-component readout, no training."""
import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from experiments.nan_bpd01.verify_sources import unary
from .parser import parse
from .prompt import make_prompts

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x).encode()).hexdigest()

def sources(cohort):
    labels={r['root']:r for r in map(json.loads,(cohort/'host_labels.jsonl').read_text().splitlines())}
    public={r['root']:r for r in map(json.loads,(cohort/'public.jsonl').read_text().splitlines())}
    for root,c in labels.items():
        refs={g:[unary(x,g)[:5] for x in c['inputs']] for g in 'INSR'};assert refs==c['refs']
        assert all(unary(c['query'],g)==c['futures'][g] for g in c['worlds'])
        assert make_prompts([x[:5] for x in c['inputs']],refs,c['worlds'],c['codes'],c['table_order'])==public[root]['prompts']
        assert len({tuple(y[0]) for y in refs.values()})==4
    assert len(labels)==10 and sum(c['split']=='development' for c in labels.values())==8
    return labels,public

def validate(r,cfg,eos):
    ids=r['initial_prompt_ids'];assert digest(ids)==r['initial_prompt_sha256'];generated=[]
    cap=cfg['maximum_integer_tokens'] if r['kind']=='integer' else cfg['maximum_operator_tokens']
    assert 1<=len(r['steps'])<=cap and len(ids)<=cfg['maximum_prompt_tokens']
    for i,s in enumerate(r['steps']):
        assert s['prefix_sha256']==digest(ids+generated);sc=s['score'];generated.append(sc['free_top_token_id'])
        assert math.isfinite(sc['full_log_normalizer']) and sc['free_top_token_logit']>=max(sc['selected_logits'].values())
        ended=generated[-1]==eos
        result=parse(s['parse_text'],r['kind'],eos=ended,limit=i+1==cap)
        if i+1<len(r['steps']):assert result['status']=='pending'
    assert result==r['parser'] and result['status']!='pending' and generated==r['generated_ids']
    assert len(generated)==r['tokens_generated'] and r['generated_text']==r['steps'][-1]['parse_text']
    return result['value']

def root_metrics(c,pred):
    m={};gold=c['refs']['R'][0]
    for arm in ('free','scaffold','copy'):
        for n in range(1,5):m[f'{arm}_b{n}']=int(pred[f'{arm}/b{n}']==gold[n])
        m[arm+'_full']=int(all(m[f'{arm}_b{n}'] for n in range(1,5)))
    for i,op in enumerate(c['worlds']):
        suffix='R' if op=='R' else 'alt'
        for arm in ('select','table','coded','labelcopy','codecopy'):
            expected=c['codes'][op] if arm in ('coded','codecopy') else op
            m[arm+'_'+suffix]=int(pred[f'{arm}/{i}']==expected)
        for arm in ('erased','erased_repeat'):m[arm+'_'+suffix]=int(pred[arm]==op)
    for arm in ('select','table','coded','labelcopy','codecopy','erased','erased_repeat'):
        m[arm+'_both']=m[arm+'_R']*m[arm+'_alt'];m[arm+'_macro']=(m[arm+'_R']+m[arm+'_alt'])/2
    assert m['erased_both']==m['erased_repeat_both']==0
    m['controls']=m['copy_full']*m['labelcopy_both']*m['codecopy_both']
    return m

def diagnose(s):
    # Fixed descriptive patterns, not mechanism confirmation or a BPD gate.
    lookup=min(s[k] for k in ('table_R','table_alt','coded_R','coded_alt'))>=6
    if s['controls']<8:return 'readout_control_failure'
    if s['free_full']<=2 and s['scaffold_full']<=2 and lookup and s['select_R']<=2:return 'arithmetic_bottleneck_compatible'
    if s['free_full']<=2 and s['scaffold_full']>=6 and lookup:return 'scaffold_sensitive_computation'
    if s['free_full']>=6 and s['select_R']<=2 and lookup:return 'implicit_selection_gap_despite_explicit_computation'
    if s['free_full']>=6 and s['table_R']<=2 and s['coded_R']>=6:return 'operator_label_interference_compatible'
    if s['free_full']>=6 and s['table_R']<=2 and s['coded_R']<=2:return 'matching_or_instruction_difficulty_compatible'
    if s['free_full']>=6 and min(s['select_R'],s['select_alt'])>=6:return 'competent_on_separated_subtests'
    return 'mixed_or_unresolved'

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--evalmath-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    runs=Path(a.runs);out=Path(a.out);labels,public=sources(runs/'cohort');cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    allmetrics={};summary={};flat=[];formatstats={};shams={};tapes={}
    for b in ('bf16','nf4'):
        receipt=json.loads((runs/b/'receipt.json').read_text());assert receipt['status']=='completed' and set(receipt['completed_roots'])==set(labels)
        metrics={};fs={'valid':0,'records':0};repeat=[];raw={}
        for root,c in labels.items():
            tape=json.loads((runs/b/f'root_{root}.json').read_text());raw[root]=tape;pred={}
            assert tape['root']==root and set(tape['records'])==set(public[root]['prompts'])
            for arm,r in tape['records'].items():
                value=validate(r,cfg,receipt['eos_token_id']);pred[arm]=value;fs['records']+=1;fs['valid']+=r['parser']['status']=='accepted'
                prefix,n=arm.split('/') if '/' in arm else (arm,None)
                if prefix in ('free','scaffold','copy'):expected=c['refs']['R'][0][int(n[1:])];correct=int(value==expected)
                elif n is not None:
                    op=c['worlds'][int(n)];expected=c['codes'][op] if prefix in ('coded','codecopy') else op;correct=int(value==expected)
                else:expected=c['worlds'];correct=sum(value==op for op in expected)/2
                flat.append({'backend':b,'root':root,'split':c['split'],'family':c['family'],'arm':arm,'kind':r['kind'],'prediction':value,'expected':expected,
                    'correct':correct,'format_valid':int(value is not None),'tokens_generated':r['tokens_generated'],'text':r['generated_text']})
            metrics[root]=root_metrics(c,pred)
            x,y=[tape['records'][k] for k in ('erased','erased_repeat')];same=x['generated_ids']==y['generated_ids'];errors=[]
            if same:
                for u,v in zip(x['steps'],y['steps']):errors.extend(abs(u['score']['selected_logits'][k]-v['score']['selected_logits'][k]) for k in u['score']['selected_logits'])
            repeat.append({'root':root,'split':c['split'],'same_generated_ids':same,'max_selected_logit_error':max(errors) if errors else None})
        summary[b]={}
        for split in ('calibration','development'):
            rr=[m for root,m in metrics.items() if labels[root]['split']==split]
            summary[b][split]={'roots':len(rr),**{k:sum(r[k] for r in rr) for k in rr[0]}}
        allmetrics[b]=metrics;formatstats[b]=fs;shams[b]=repeat;tapes[b]=raw
    sys.path.insert(0,str(Path(a.evalmath_root).resolve()/'src'))
    from spiral_ln.evaluation_finite_sample import evaluate_manifest,signflip_cluster_p,holm_adjust
    within=[('scaffold_full','free_full'),('table_R','select_R'),('coded_R','table_R'),('table_alt','select_alt'),('select_R','select_alt'),('free_full','select_R')]
    specs=[(b,x,b,y) for b in ('bf16','nf4') for x,y in within]+[('nf4',x,'bf16',x) for x in ('free_full','select_R','table_R')]
    tests=[];ps=[];roots=[root for root,c in labels.items() if c['split']=='development']
    for cb,ca,bb,ba in specs:
        name=f'{cb}/{ca} minus {bb}/{ba}';cases=[{'case_id':str(root),'cluster_id':str(root),'candidate':allmetrics[cb][root][ca],'baseline':allmetrics[bb][root][ba]} for root in roots]
        manifest={'schema_version':1,'experiment_id':'NAN-DIAG-04/'+name,'evaluation_mode':'sampled','target_population':'fresh expanded-coefficient development source roots',
            'alpha':'0.05','minimum_effect':'0.25','planned_comparisons':15,'min_independent_clusters':20,'hypothesis_preregistered':True,'sealed_readout':False,
            'clusters_independent':False,'exchangeable_pair_labels':False,'cases':cases}
        effects=[Q(c['candidate']-c['baseline']) for c in cases];pv,method=signflip_cluster_p(effects);ps.append(pv)
        tests.append({'contrast':name,'manifest':manifest,'fsi':evaluate_manifest(manifest),'root_effects_exact':{str(root):str(x) for root,x in zip(roots,effects)},
            'conditional_p_exact':str(pv),'conditional_method':method,'scope':'Imposed independent-root signed-law sensitivity only; assumptions unestablished'})
    for t,pv in zip(tests,holm_adjust(ps)):t['conditional_holm_exact']=str(pv)
    for b in ('bf16','nf4'):
        pilot=[{'unit_id':str(root),'paired_effect':str(Q(allmetrics[b][root]['table_R']-allmetrics[b][root]['select_R'])),
            'sham_effects':[str(Q(allmetrics[b][root]['erased_macro'])-Q(allmetrics[b][root]['erased_repeat_macro']))]} for root in roots]
        spec={'schema_version':1,'experiment_id':f'NAN-DIAG-04/{b}/reference_assistance','independent_unit':'source generator root; dependence after global source rejection unestablished',
            'alpha':'0.05','scientific_min_effect':'0.25','target_effect':'0.50','measurement_resolution':'1/8','sham_quantile':'0.95','sham_confidence':'0.95','planned_comparisons':15,
            'confirm_units_planned':60,'pilot_from_development_only':True,'heldout_sealed':True,'units_independent':False,'pilot':pilot,'sham_scope':'Repeated erased generation, numeric/format repeat only, not semantic-assistance tail'}
        write(out/f'{b}_power_input.json',spec);tool=Path(a.evalmath_root).resolve()/'skills/evalmath-mechinterp-power/scripts/preflight.py'
        proc=subprocess.run([sys.executable,str(tool),'--pilot',str(out/f'{b}_power_input.json'),'--strict'],capture_output=True,text=True,encoding='utf-8')
        write(out/f'{b}_power_result.json',{'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'planner_sha256':sha(tool)})
    write(out/'graded_records.json',flat);write(out/'root_metrics.json',allmetrics);write(out/'fsi_contrasts.json',tests)
    result={'experiment':'NAN-DIAG-04','evidence_class':'prospective development-only component diagnostic; no BPD gate or mechanism confirmation','sampling_unit':'source generator root',
        'development_roots':8,'calibration_roots':2,'checkpoint_count':1,'planned_and_actual_comparisons':15,'summaries':summary,'format':formatstats,'shams':shams,
        'diagnostic_pattern':{b:diagnose(summary[b]['development']) for b in summary},'BPD_transfer_attempted':False,'original_competence_gates_changed':False,'sealed_168_case_test_accessed':False}
    write(out/'results.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
