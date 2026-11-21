"""Host-only grading, separate from model worker; safe aggregate rolling reports."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from .common import read,rows,write,sha
from .parser import parse

def score_value(value,label):
    exact=int(value==label['answer']);result={'correct':exact,'reference_exact':exact}
    if label.get('copy_control'):return result
    if label['suite']=='sudoku':
        from research_gym.envs.sudoku import grid_from_rows,is_solved
        givens=label['givens']
        valid=(isinstance(value,list) and len(value)==4 and all(isinstance(r,list) and len(r)==4 for r in value)
               and all(type(v) is int and 1<=v<=4 for r in value for v in r))
        valid=valid and is_solved(grid_from_rows(value)) and all(not g or value[i][j]==g for i,row in enumerate(givens) for j,g in enumerate(row))
        result['valid_givens_preserving_solution']=int(valid)
    if label['suite']=='storyworld':
        from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv,StoryState
        env=CoupledStoryworldEnv();state=StoryState(**label['start_state']);trace=[]
        valid=isinstance(value,list) and 1<=len(value)<=label['horizon'] and all(type(v) is str and v in env.self_actions for v in value)
        if valid:
            for action in value:
                if env.target(state) or env.terminal(state):break
                other=env.other_policy(state);state=env.step(state,action,other)
                trace.append({'action':action,'rival':other,'state':state.to_dict()})
        result|={'correct':int(valid and env.target(state)),'rollout':trace,'final_state':state.to_dict()}
    return result

def validate(record,eos):
    def digest(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    initial=record['initial_prompt_ids'];assert digest(initial)==record['initial_prompt_sha256'];generated=[]
    for i,step in enumerate(record['steps']):
        assert step['step']==i+1 and step['prefix_sha256']==digest(initial+generated)
        s=step['score'];generated.append(s['free_top_token_id'])
        assert math.isfinite(s['full_log_normalizer']) and s['free_top_token_logit']>=max(s['selected_logits'].values())
        parsed=parse(step['parse_text'],record['schema'],final=generated[-1]==eos or i+1==len(record['steps']))
        if i+1<len(record['steps']):assert parsed['status']=='pending'
    assert generated==record['generated_ids'] and parsed==record['parser'] and parsed['status']!='pending'
    assert record['generated_text']==record['steps'][-1]['parse_text']
    return parsed['value']

def grade(runs,*,allow_partial=False):
    labels={r['id']:r for r in rows(runs/'cohort/host_labels.jsonl')};public={r['id']:r for r in rows(runs/'cohort/public.jsonl')}
    seal=read(Path(__file__).with_name('SOURCE_MANIFEST.json'))
    for name,h in seal['grading_dependencies_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('Native task grader source changed: '+name)
    assert sha(runs/'cohort/host_labels.jsonl')==seal['combined_host_sha256'] and sha(runs/'cohort/public.jsonl')==seal['combined_public_sha256']
    full=[];summaries={};coverage={}
    for backend in ('bf16','nf4'):
        folder=runs/backend
        if not (folder/'manifest.json').exists():coverage[backend]={'roots':0,'status':'not_loaded'};continue
        manifest=read(folder/'manifest.json');receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else {'status':'running'}
        if not allow_partial:assert receipt['status']=='completed'
        done=[]
        for file in sorted(folder.glob('root_*.json')):
            try:tape=read(file)
            except json.JSONDecodeError:
                if allow_partial:continue
                raise
            assert sha(folder/tape['state_file'])==tape['state_file_sha256']
            for task_id,record in tape['records'].items():
                label=labels[task_id];assert record['schema']==public[task_id]['schema']
                value=validate(record,manifest['eos_token_id']);score=score_value(value,label)
                full.append({'backend':backend,'id':task_id,'root':label['root'],'suite':label['suite'],'split':label['split'],
                    'copy_control':label.get('copy_control',False),'hops':label.get('hops'),'format_valid':int(record['parser']['status']=='accepted'),
                    'prediction':value,'expected':label['answer'],'generated_text':record['generated_text'],'tokens':record['tokens_generated'],**score})
            done.append(tape['root'])
        coverage[backend]={'roots':len(done),'status':receipt['status'],'root_ids':done}
        groups=defaultdict(list)
        for r in full:
            if r['backend']!=backend:continue
            condition='copy' if r['copy_control'] else ('hop'+str(r['hops']) if r['hops'] else 'main')
            groups[(r['split'],r['suite'],condition)].append(r)
        summaries[backend]=[{'split':k[0],'suite':k[1],'condition':k[2],'roots':len({r['root'] for r in rr}),
            'records':len(rr),'correct':sum(r['correct'] for r in rr),'format_valid':sum(r['format_valid'] for r in rr),
            **({'valid_sudoku_solutions':sum(r.get('valid_givens_preserving_solution',0) for r in rr)} if k[1]=='sudoku' and k[2]=='main' else {})}
            for k,rr in sorted(groups.items())]
    return {'experiment':'NAN-GYM-05','evidence_class':'descriptive development, no inferential testing','coverage':coverage,'summaries':summaries,
        'partial':any(v['status']!='completed' for v in coverage.values()),'formal_comparisons':0,'BPD_transfer_attempted':False,'sealed_test_accessed':False},full

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    result,full=grade(Path(a.runs),allow_partial=a.allow_partial);write(Path(a.out)/'summary.json',result)
    # Detailed routing records stay LOCAL; only summary.json is publishable.
    write(Path(a.out)/'local_graded_records.json',full)
    print(json.dumps(result))

if __name__=='__main__':main()
