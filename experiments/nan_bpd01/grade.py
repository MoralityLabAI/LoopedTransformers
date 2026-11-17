"""Isolated exact-oracle grading and cluster-aware descriptive evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from .head import scores_at_temperatures

def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--evalmath-root',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.evalmath_root).resolve()/'src'))
    from spiral_ln import algebraic_reasoning_oracle as O
    from spiral_ln.evaluation_finite_sample import evaluate_manifest,signflip_cluster_p,holm_adjust
    def forbid(*a,**kw):raise RuntimeError('Existing roster access forbidden')
    O.roster=forbid
    runs=Path(a.runs);out=Path(a.out);cohort=runs/'cohort';cfg=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    from .verify_sources import verify
    write(out/'independent_source_verification.json',verify(cohort))
    labels={}
    for line in (cohort/'host_labels.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);r['demos']=tuple((tuple(x),tuple(y)) for x,y in r['demos'])
        for key in ['query','future','source_graph']:r[key]=tuple(r[key])
        labels[r['identifier']]=O.Case(**r)
    pairs=json.loads((cohort/'pairs.json').read_text(encoding='utf-8'))
    def grade(c,scores):
        order=sorted(range(64),key=lambda i:(-scores[i],i));programs=[O.PROGRAMS[i] for i in order]
        g=O.grade(c,O.search(c,programs,12))
        top=O.grade(c,O.search(c,programs[:1],1))
        return g|{'top1_future_correct':top['future_correct'],'source_rank':order.index(O.PROGRAMS.index(c.source_graph))+1,
                  'top1_program':list(programs[0]),'top12_programs':[list(g) for g in programs[:12]]}
    tapes={};calibrations={}
    # Only CAL scores and CAL labels are touched during temperature fitting.
    for backend in ['bf16','nf4']:
        path=runs/backend
        calrows=[]
        for pair in pairs:
            if pair['split']!='calibration':continue
            for role in ['recipient','donor']:
                caseid=pair[role];file=path/(caseid.replace(':','_')+'.json')
                if file.exists():calrows.append((labels[caseid],json.loads(file.read_text())['records']['native2']))
        if len(calrows)!=4:raise RuntimeError(f'{backend}: incomplete calibration')
        grid=[]
        for temps in product(cfg['temperature_grid'],repeat=3):
            # Fixed CAL source-program NLL, with lexicographic temperature tie-break.
            loss=-sum(scores_at_temperatures(r,temps)[O.PROGRAMS.index(c.source_graph)] for c,r in calrows)/len(calrows)
            grid.append((loss,temps))
        loss,temps=min(grid)
        counts=[[1]*4 for _ in range(3)]
        for c,_ in calrows:
            for pos,op in enumerate(c.source_graph):counts[pos][O.OPS.index(op)]+=1
        import math
        frequency=[sum(math.log(counts[pos][O.OPS.index(op)]/sum(counts[pos])) for pos,op in enumerate(g)) for g in O.PROGRAMS]
        calibrations[backend]={'temperatures':list(temps),'cal_source_nll':loss,'cal_cases':4,'development_labels_used':0,'selection':'CAL source NLL; lexicographic ties','position_counts_with_laplace1':counts,'frequency_scores':frequency,'all_grid_losses':[{'temperatures':list(t),'loss':v} for v,t in grid]}
    write(out/'calibration_frozen.json',calibrations)
    backend_results={};flat=[];contrasts=[];conditional_ps=[]
    for backend in ['bf16','nf4']:
        path=runs/backend;graded={};native=[];allrecords={}
        for pair in pairs:
            for role in ['recipient','donor']:
                cid=pair[role];file=path/(cid.replace(':','_')+'.json')
                if not file.exists():continue
                r=json.loads(file.read_text());allrecords[cid]=r
                if pair['split']!='development':continue
                c=labels[cid];dr=labels[pair['donor']]
                arms={n:grade(c,s['scores']) for n,s in r['records'].items()}
                arms['calibrated']=grade(c,scores_at_temperatures(r['records']['native2'],calibrations[backend]['temperatures']))
                arms['frequency']=grade(c,calibrations[backend]['frequency_scores'])
                edits={}
                for name,record in r['interventions'].items():
                    edits[name]={'recipient':grade(c,record['delayed']['scores']),'donor':grade(dr,record['delayed']['scores'])}
                graded[cid]={'root':pair['root'],'role':role,'arms':arms,'edits':edits}
                for name,g in arms.items():flat.append({'backend':backend,'case_id':cid,'root':pair['root'],'arm':name,'target':'recipient',**g})
                for name,g in edits.items():
                    for target,v in g.items():flat.append({'backend':backend,'case_id':cid,'root':pair['root'],'arm':name,'target':target,**v})
        stats={}
        names=sorted({name for r in graded.values() for name in r['arms']})
        for name in names:
            values=[r['arms'][name] for r in graded.values() if name in r['arms']]
            stats[name]={'n_cases':len(values),'future12_correct':sum(g['future_correct'] for g in values),'top1_future_correct':sum(g['top1_future_correct'] for g in values),'mean_source_rank':sum(g['source_rank'] for g in values)/len(values)}
        both=both_top=0
        for pair in pairs:
            if pair['split']!='development' or not all(pair[r] in graded for r in ['recipient','donor']):continue
            values=[graded[pair[r]]['arms']['native2'] for r in ['recipient','donor']]
            both+=int(all(g['future_correct'] for g in values));both_top+=int(all(g['top1_future_correct'] for g in values))
        measured=stats.get('native2',{});gate=(measured.get('n_cases')==8 and measured['future12_correct']/8>=cfg['clean_native_future12_minimum'] and both>=2 and both_top>=1)
        closures=[v['state_closure_max_error'] for r in allrecords.values() for name,v in r['interventions'].items() if name=='full_donor_state']
        closure_gate=len(closures)==2 and max(closures)==0
        # FSI-01 provenance stays honest: algorithm labels were not randomized.
        comparison_names=['native1','calibrated','open_loop','equal_compute','bare_native2']
        comparison_names+=sorted({name for r in graded.values() for name in r['edits']})
        comparison_names+=['phase_identity_activation']
        for name in comparison_names:
            cases=[]
            for cid,r in graded.items():
                if name in r['arms']:candidate=r['arms'][name]['future_correct']
                elif name in r['edits']:candidate=r['edits'][name]['recipient']['future_correct']
                else:continue
                cases.append({'case_id':cid,'cluster_id':str(r['root']),'candidate':candidate,'baseline':r['arms']['native2']['future_correct']})
            if not cases:
                contrasts.append({'backend':backend,'contrast':name,'status':'continuous diagnostic; no binary test','holm_placeholder':'1'});conditional_ps.append(Q(1));continue
            manifest={'schema_version':1,'experiment_id':f'NAN-BPD-01/{backend}/{name}','target_population':'fresh development source worlds; one pinned checkpoint',
                'evaluation_mode':'sampled','alpha':'0.05','minimum_effect':'0.15','planned_comparisons':40,'min_independent_clusters':20,
                'hypothesis_preregistered':True,'sealed_readout':False,'clusters_independent':True,'exchangeable_pair_labels':False,'cases':cases}
            result=evaluate_manifest(manifest)
            roots=defaultdict(list)
            for c in cases:roots[c['cluster_id']].append(Q(c['candidate']-c['baseline']))
            effects=[sum(vals)/len(vals) for vals in roots.values()]
            conditional,method=signflip_cluster_p(effects)
            contrasts.append({'backend':backend,'contrast':name,'manifest':manifest,'fsi':result,
                'conditional_signflip_p_exact':str(conditional),'conditional_method':method,
                'conditional_assumption':'requires rootwise symmetric sign null, unestablished here; sensitivity calculation only'})
            conditional_ps.append(conditional)
        backend_results[backend]={'task':stats,'both_native_future12_pairs':both,'both_native_top1_pairs':both_top,
            'clean_competence_gate':gate,'checkpoint_state_closure_errors':closures,'state_closure_gate':closure_gate,
            'primary_BPD_vs_random_dark':{'status':'not_run','reason':'clean competence gate failed' if not gate else 'eligible for next instrument stage; fitting required'},
            'graded_cases':graded,'receipt':json.loads((path/'receipt.json').read_text()),'phase_calibration':json.loads((path/'calibration_phase.json').read_text())}
        # Empirical preflight describes the tested phase contrast only, not BPD.
        pilot=[]
        for cid,r in graded.items():
            if 'phase_context_replace' not in r['edits']:continue
            e=r['edits'];base=r['arms']['native2']['future_correct']
            pilot.append({'unit_id':str(r['root']),'paired_effect':str(Q(e['phase_context_replace']['recipient']['future_correct']-e['random_dense_dark_741']['recipient']['future_correct'])),
                'sham_effects':[str(Q(e[n]['recipient']['future_correct']-base)) for n in ['random_dense_dark_741','random_dense_dark_947']]})
        planner={'schema_version':1,'experiment_id':f'NAN-BPD-01/{backend}/phase-context-diagnostic-NOT-BPD-primary','independent_unit':'source generator world',
            'alpha':'0.05','scientific_min_effect':'0.15','target_effect':'0.20','measurement_resolution':'0.125','sham_quantile':'0.95','sham_confidence':'0.95',
            'planned_comparisons':40,'confirm_units_planned':60,'pilot_from_development_only':True,'heldout_sealed':True,'units_independent':True,'pilot':pilot}
        write(out/f'{backend}_power_input.json',planner)
        tool=Path(a.evalmath_root).resolve()/'skills/evalmath-mechinterp-power/scripts/preflight.py'
        run=subprocess.run([sys.executable,str(tool),'--pilot',str(out/f'{backend}_power_input.json'),'--strict'],capture_output=True,text=True,encoding='utf-8')
        write(out/f'{backend}_power_result.json',{'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'planner_sha256':sha(tool),'contrast_is_BPD_primary':False})
    # Reserve all forty comparisons even if no endpoint can be computed.
    if len(conditional_ps)>40:raise AssertionError('Registered comparison budget exceeded')
    adjusted=holm_adjust(conditional_ps+[Q(1)]*(40-len(conditional_ps)))
    for r,q in zip(contrasts,adjusted):r['conditional_holm_p_exact']=str(q)
    write(out/'graded_cases.json',flat);write(out/'fsi_contrasts.json',contrasts)
    write(out/'results.json',{'schema_version':1,'evidence_class':'development-only, power-limited; no confirmation',
        'unit':'source generator world','n_units_development':4,'n_cases_development':8,'n_calibration_roots':2,'n_phase_intervention_roots':2,
        'backend_results':backend_results,'planned_exploratory_comparisons':40,'actual_binary_or_continuous_comparisons':len(contrasts),
        'proposed_primary_comparisons':2,'scientific_minimum_effect':.15,'physical_relative_dose':.02,'original_168_case_test_accessed':False,
        'provenance':{'oracle_sha256':sha(O.__file__),'grader_sha256':sha(__file__),'calibration_frozen_sha256':sha(out/'calibration_frozen.json'),
                      'host_labels_sha256':sha(cohort/'host_labels.jsonl'),'source_receipt_sha256':sha(cohort/'source_receipt.json')}})
    print(json.dumps({b:{k:v for k,v in r.items() if k in ['task','both_native_future12_pairs','both_native_top1_pairs','clean_competence_gate','state_closure_gate']} for b,r in backend_results.items()},indent=2))

if __name__=='__main__':main()
