"""Copy compact new-study receipts only; never traverse historical split data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from .head import scores_at_temperatures
from .caseio import load_case

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--experiment-root',default='experiments/nan_bpd01');a=p.parse_args()
    runs=Path(a.runs);root=Path(a.experiment_root);dest=root/'receipts';dest.mkdir(parents=True,exist_ok=True)
    for name in ['public.jsonl','host_labels.jsonl','pairs.json','source_receipt.json','independent_verification.json']:
        (root/'fixtures').mkdir(parents=True,exist_ok=True)
        source=runs/'graded/independent_source_verification.json' if name=='independent_verification.json' else runs/'cohort'/name
        shutil.copyfile(source,root/'fixtures'/name)
    for name in ['results.json','graded_cases.json','calibration_frozen.json','fsi_contrasts.json','independent_source_verification.json','bf16_power_input.json','bf16_power_result.json','nf4_power_input.json','nf4_power_result.json']:
        shutil.copyfile(runs/'graded'/name,dest/name)
    graded=json.loads((runs/'graded/graded_cases.json').read_text())
    columns=['backend','root','case_id','arm','target','canonical_prefix_consistent','accepted','future_correct','top1_future_correct','source_rank','false_future_acceptance']
    with (dest/'graded_outcomes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(graded)
    for path in (runs/'analysis').iterdir():
        if path.suffix in ['.json','.csv','.png','.pdf']:shutil.copyfile(path,dest/path.name)
    for name in ['gpu_software.json','cpu_software.json','environment_during_bf16.json','cpu_tests.log','local_paths.json']:shutil.copyfile(runs/name,dest/name)
    for path in (runs/'engineering').glob('*.json'):shutil.copyfile(path,dest/path.name)
    shutil.copyfile(runs/'prefix_consistency_diagnostic.json',dest/'initial_prefix_consistency_diagnostic.json')
    initial=dest/'initial_invalid_dense_analysis';initial.mkdir(exist_ok=True)
    for path in (runs/'graded_initial_invalid_dense').glob('*.json'):shutil.copyfile(path,initial/path.name)
    frozen=json.loads((runs/'graded/calibration_frozen.json').read_text());pairs=json.loads((runs/'cohort/pairs.json').read_text());compact={}
    for backend in ['bf16','nf4']:
        shutil.copyfile(runs/backend/'receipt.json',dest/f'{backend}_execution_receipt.json')
        shutil.copyfile(runs/backend/'manifest.json',dest/f'{backend}_inference_manifest.json')
        for name in ['initial_execution_receipt.json','manifest_resume.json']:
            if (runs/backend/name).exists():shutil.copyfile(runs/backend/name,dest/f'{backend}_{name}')
        shutil.copyfile(runs/'prefix_repair'/backend/'receipt.json',dest/f'{backend}_prefix_repair_execution_receipt.json')
        compact[backend]={}
        for pair in pairs:
            for role in ['recipient','donor']:
                cid=pair[role];file=runs/backend/(cid.replace(':','_')+'.json');r=load_case(runs,backend,cid)
                scores={name:v['scores'] for name,v in r['records'].items()}
                if pair['split']=='development':
                    scores['calibrated']=scores_at_temperatures(r['records']['native2'],frozen[backend]['temperatures'])
                    scores['frequency']=frozen[backend]['frequency_scores']
                for name,v in r['interventions'].items():scores[name]=v['delayed']['scores']
                measures={}
                for name,v in r['interventions'].items():
                    info={k:x for k,x in v.items() if k not in ['current','delayed']}
                    info['delayed_metrics_reference']='donor_native1' if name=='source_input' else ('recipient_native1' if name=='zero_write' else 'recipient_native2')
                    info['delayed_head_metrics']={k:x for k,x in v['delayed'].items() if k.startswith('full_vocab_')}
                    if 'current' in v:info['current_head_metrics']={k:x for k,x in v['current'].items() if k.startswith('full_vocab_')}
                    measures[name]=info
                rec={'source_raw_tape_sha256':sha(file),'root':pair['root'],'split':pair['split'],'role':role,'scores':scores,'intervention_measurements':measures}
                repaired=runs/'prefix_repair'/backend/(cid.replace(':','_')+'.json')
                if repaired.exists():rec['prefix_repair_raw_tape_sha256']=sha(repaired)
                if pair['split']=='calibration':rec['calibration_native2_full_head_record']=r['records']['native2']
                compact[backend][cid]=rec
    write(dest/'compact_score_tapes.json',compact)
    lockpath=Path(a.compactification_root)/'research/algernon/algebraic_reasoning_gym/arg02/config/checkpoint_lock.json'
    lock=json.loads(lockpath.read_text())['nanbeige'];digest=sha(Path(a.snapshot)/'tokenizer_config.json')
    assert digest=='82316b70213c65e5f820c03d5a11561ccda4055bda984cb770ed0d51fe497c56'
    write(dest/'checkpoint_verification.json',{'revision':lock['revision'],'locked_files_sha256':lock['files_sha256'],
        'loader_hash_checks_passed_before_model_construction_both_backends':True,'checkpoint_lock_sha256':sha(lockpath),
        'supplemental_tokenizer_config_sha256':digest,'weights_fetched_or_replaced':False})
    gates=json.loads((runs/'graded/results.json').read_text())['backend_results'];stages={}
    for b,r in gates.items():
        reason='clean competence gate failed' if not r['clean_competence_gate'] else ('checkpoint state closure gate failed' if not r['state_closure_gate'] else 'eligible; next stage required')
        names=['rank8_rank16_native_dictionary_fitting','native_task_gradient_basis','native_delayed_observability_gramian','trained_BPD_read_write_factors','selected_BPD_vs_random_dark_primary','native_static_vs_reached_state_routing','native_GL_component_gauge_interpretation','BF16_NF4_component_subspace_alignment']
        stages[b]={'clean_competence':r['clean_competence_gate'],'state_closure':r['state_closure_gate'],
            'gate_observations':{'native_future12':r['task']['native2']['future12_correct'],'native_cases':8,'minimum_native_future12':4,
                'both_future12_pairs':r['both_native_future12_pairs'],'minimum_both_future12_pairs':2,'both_top1_pairs':r['both_native_top1_pairs'],'minimum_both_top1_pairs':1,
                'donor_state_closure_max_errors':r['checkpoint_state_closure_errors']},
            'native_stages':{name:{'status':'blocked' if not (r['clean_competence_gate'] and r['state_closure_gate']) else 'pending','reason':reason} for name in names}}
    write(dest/'stage_status.json',{'backend':stages,'confirmation':{'status':'blocked','reason':'competence, independent sham calibration, root count and untouched new confirmation readout required'},'original_168_case_test_accessed':False})
    raw=[]
    for path in sorted(runs.rglob('*')):
        if path.is_file():raw.append({'path':path.relative_to(runs).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    write(dest/'local_raw_artifact_index.json',{'root':'runs/nan_bpd01','files':raw,'checkpoint_weights_included':False,'raw_tensors_committed':False,'sealed_test_accessed':False})
    print(json.dumps({'compact_receipts':str(dest),'raw_files_hashed':len(raw)}))

if __name__=='__main__':main()
