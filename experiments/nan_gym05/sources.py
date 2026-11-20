"""Outcome-blind suite materialization. Local routing text is never Git evidence."""
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
import sys
from .common import read,write,write_rows,sha,prompt_ids

NODES='ABCDEFGHIJKLMNOP'
ROUTES=('alphabet_sort','arc_challenge','arc_easy','gsm8k','intellect_3_logic','intellect_3_math','mbpp','wiki_search')
GRID={'type':'grid','max_characters':2048}
ACTIONS=('befriend','investigate','defuse','rush','wait')
PLAN={'type':'plan','allowed':list(ACTIONS),'max_characters':512}
STORY_RULES='''State = (trust,evidence,heat,scene). Bounds: trust -3..3, evidence 0..4, heat 0..4, scene 0..5. Clamp each field after every complete turn.
Before a turn choose the rival using the CURRENT state: if evidence>=2 and heat<=3, sabotage; else if trust<=-1 and evidence>=1, flatter; else rival_wait.
Apply self action first: befriend adds 1 trust and replaces heat by max(0,heat-1); investigate adds 1 evidence and 1 heat, then subtracts 1 trust if the NEW heat>=2; defuse subtracts 2 heat; rush adds 1 scene and 1 heat; wait replaces heat by max(0,heat-1).
Then apply rival: sabotage subtracts 1 trust and adds 1 heat; flatter adds 1 trust and replaces evidence by max(0,evidence-1); rival_wait does nothing.
Finally add 1 scene unless the self action was wait, then clamp all fields. Thus rush advances scene twice. Stop at scene>=5. Success: scene>=5 AND trust>=1 AND evidence>=3 AND heat<=2.
Give a plan of one to six self actions that reaches success under this deterministic rival. Output only a JSON list of action strings; no explanation.'''

def pointer_case(rng,hops):
    while True:
        permutation=list(NODES);rng.shuffle(permutation);start=rng.choice(NODES);walk=[start]
        for _ in range(hops):walk.append(permutation[NODES.index(walk[-1])])
        if len(set(walk))==len(walk):break
    order=list(range(16));rng.shuffle(order)
    text='Edges: '+', '.join(NODES[i]+'->'+permutation[i] for i in order)+f'\nStart: {start}\nHops: {hops}\nAnswer:'
    return text,walk[-1]

def pointer_root(seed,hops):
    rng=random.Random(seed);demos=[]
    for h in (1,2,3):
        t,a=pointer_case(rng,h);demos.append(t+' '+a)
    out=[]
    for h in hops:
        t,a=pointer_case(rng,h)
        out.append((h,'Follow the directed edges exactly the requested number of hops. Reply only the final node A through P.\n\n'+'\n\n'.join(demos+[t]),a))
    return out

def materialize(config):
    # These are explicit PUBLIC microbenchmarks, not held-out ARG rosters.
    from research_gym.envs.arc_tasks import default_arc1_tasks,default_arc2_tasks,primitive_rules
    from research_gym.envs.sudoku import default_sudoku_puzzles,is_solved
    from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv
    from research_gym.benchmarks.storyworld_bench import sample_start_states
    public=[];host=[]
    def add(root,split,suite,prompt,schema,answer,**extra):
        task_id=root+':'+str(extra.pop('variant','main'))
        public.append({'id':task_id,'root':root,'split':split,'suite':suite,'prompt':prompt,'schema':schema,'publication':'public'})
        host.append({'id':task_id,'root':root,'split':split,'suite':suite,'answer':answer,**extra})
    def copy(root,split,suite,schema,answer):
        text=answer if type(answer) is str else json.dumps(answer,separators=(',',':'))
        add(root,split,suite,'Copy the following verified answer exactly, with no explanation:\n'+text,schema,answer,variant='copy',copy_control=True)
    for split,count,offset in [('calibration',config['calibration_pointer_roots'],0),('development',config['development_pointer_roots'],100000)]:
        for i in range(count):
            seed=config['source_seed']+offset+i;root=f'pointer:{seed}';made=pointer_root(seed,config['pointer_hops'])
            schema={'type':'label','allowed':list(NODES),'max_characters':64}
            for h,prompt,answer in made:
                add(root,split,'pointer',prompt,schema,answer,variant='hop'+str(h),hops=h,copy_control=False)
            copy(root,split,'pointer',schema,made[0][2])
    rules={r.name:r for r in primitive_rules()}
    for suite, tasks in [('arc1',default_arc1_tasks()),('arc2',default_arc2_tasks())]:
        for t in tasks:
            names=t.rule_names or (t.rule_name,)
            predicted=t.test_input
            for name in names:predicted=rules[name].apply(predicted)
            assert predicted==t.test_output
            text='Infer the grid transformation from BOTH examples and apply it to the query. Output only the complete output grid as a JSON list of rows. No explanation.\n'
            for j,e in enumerate(t.train):text+=f'Example {j+1} input: {json.dumps(e.input_grid)}\nExample {j+1} output: {json.dumps(e.output_grid)}\n'
            text+='Query input: '+json.dumps(t.test_input)
            root=suite+':'+t.task_id;answer=[list(r) for r in t.test_output]
            add(root,'development',suite,text,GRID,answer,copy_control=False,public_upstream_task=t.task_id)
            copy(root,'development',suite,GRID,answer)
    for p in default_sudoku_puzzles():
        assert is_solved(p.solution)
        root='sudoku:'+p.puzzle_id;answer=[list(r) for r in p.solution]
        text='Complete this 4 by 4 Sudoku. Digits 1,2,3,4 appear once in every row, column and 2 by 2 box. Preserve all nonzero givens; zeros are blanks. Output only the entire grid as a JSON list of rows.\n'+json.dumps(p.givens)
        add(root,'development','sudoku',text,GRID,answer,copy_control=False,givens=p.givens,
            primary='upstream_exact_reference',secondary='any_valid_givens_preserving_solution')
        copy(root,'development','sudoku',GRID,answer)
    env=CoupledStoryworldEnv()
    starts=sample_start_states(env,n=config['story_roots'],horizon=config['story_horizon'],seed=config['source_seed'])
    for i,s in enumerate(starts):
        success,witness=env.reachable(s,config['story_horizon'],'model');assert success
        answer=[v.split('/')[0] for v in witness];assert 1<=len(answer)<=6
        root='story:'+str(config['source_seed'])+':'+str(i)
        add(root,'development','storyworld',STORY_RULES+'\nInitial state: '+json.dumps(s.to_dict()),PLAN,answer,
            copy_control=False,start_state=s.to_dict(),horizon=config['story_horizon'],primary='native_environment_rollout_success')
        copy(root,'development','storyworld',PLAN,answer)
    return public,host

def main():
    p=argparse.ArgumentParser();p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--routing-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).parent;cfg=read(root/'config.json');sys.path.insert(0,str(Path(a.compactification_root).resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok=load_tokenizer(Path(a.snapshot),166100);public,host=materialize(cfg)
    public_repo=list(public);host_repo=list(host);routing_receipt={}
    schema={'type':'label','allowed':list(ROUTES),'max_characters':96}
    for env in ROUTES:
        path=Path(a.routing_root)/'normalized_trajectories'/f'{env}.jsonl';taken=[];rejections=Counter();seen=set();examined=0
        with path.open(encoding='utf-8') as f:
            for index,line in enumerate(f):
                if index>=cfg['routing_max_rows_examined_per_env'] or len(taken)>=cfg['routing_roots_per_env']:break
                examined+=1;r=json.loads(line);trajectory=str(r['trajectory_id'])
                if trajectory in seen:rejections['same_trajectory']+=1;continue
                prompt='Route this task to ONE environment. Allowed environment IDs: '+', '.join(ROUTES)+'. Reply only the environment ID, with no explanation.\nTask:\n'+r['state_prompt']
                ids=prompt_ids(tok,a.snapshot,prompt)
                if len(ids)>cfg['maximum_prompt_tokens']:rejections['prompt_bound']+=1;continue
                seen.add(trajectory);key=hashlib.sha256((env+'\0'+trajectory).encode()).hexdigest()[:20];rid='routing:'+env+':'+key
                row={'id':rid+':main','root':rid,'split':'development','suite':'routing','prompt':prompt,'schema':schema,'publication':'local_only'}
                label={'id':row['id'],'root':rid,'split':'development','suite':'routing','answer':env,'copy_control':False,'source_file_sha256':sha(path),'source_row':index,'trajectory_sha256':hashlib.sha256(trajectory.encode()).hexdigest()}
                public.append(row);host.append(label);taken.append({'row':index,'root':rid,'prompt_tokens':len(ids)})
                cp=dict(row,id=rid+':copy',prompt='The verified route is '+env+'. Copy only that environment ID.')
                public.append(cp);host.append(dict(label,id=cp['id'],copy_control=True))
        routing_receipt[env]={'source_file_sha256':sha(path),'examined':examined,'admitted':taken,'rejections':dict(rejections)}
        if len(taken)!=cfg['routing_roots_per_env']:raise RuntimeError('Insufficient fixed source-eligible trajectories: '+env)
    lengths={r['id']:len(prompt_ids(tok,a.snapshot,r['prompt'])) for r in public}
    if max(lengths.values())>cfg['maximum_prompt_tokens']:raise RuntimeError('An unmodified public-suite prompt exceeds the registered bound')
    assert len({r['id'] for r in public})==len(public)
    assert not {r['prompt'] for r in public if r['split']=='calibration' and not r['id'].endswith(':copy')} & {r['prompt'] for r in public if r['split']=='development'}
    out=Path(a.out);write_rows(out/'public.jsonl',public);write_rows(out/'host_labels.jsonl',host)
    write_rows(out/'public_repo.jsonl',public_repo);write_rows(out/'host_repo.jsonl',host_repo)
    write(out/'source_receipt.json',{'source_seed':cfg['source_seed'],'sources_outcome_blind':True,'sealed_test_accessed':False,
        'task_rows':len(public),'source_roots':len({r['root'] for r in public}),'by_suite':dict(Counter(r['suite'] for r in public)),
        'by_split':dict(Counter(r['split'] for r in public)),'maximum_prompt_tokens':max(lengths.values()),'prompt_lengths':lengths,
        'routing':routing_receipt,'publication_policy':'Only public_repo/host_repo and aggregate/hash receipts may enter Git; local routing prompts/tapes/states remain local'})
    print(json.dumps({'tasks':len(public),'maximum_prompt_tokens':max(lengths.values()),'routing_roots':sum(len(v['admitted']) for v in routing_receipt.values())}))

if __name__=='__main__':main()
