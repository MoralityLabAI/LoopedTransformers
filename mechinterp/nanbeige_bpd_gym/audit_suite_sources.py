"""Post-registration source-only independent checks; no neural result access."""
import argparse
from itertools import product
import json
from pathlib import Path
import re
from experiments.nan_gym05.common import rows,read,sha,write

def pointer_answer(text):
    matches=list(re.finditer(r'Edges: ([^\n]+)\nStart: ([A-P])\nHops: ([1-4])\nAnswer:(?: ([A-P]))?',text))
    assert len(matches)==4
    for i,m in enumerate(matches):
        edges=dict(re.findall(r'([A-P])->([A-P])',m[1]));assert len(edges)==len(set(edges.values()))==16
        node=m[2]
        for _ in range(int(m[3])):node=edges[node]
        if i<3:assert m[4]==node
    return node

def sudoku_completions(givens):
    board=[list(row) for row in givens];answers=[]
    def solve():
        choices=[]
        for r,c in product(range(4),repeat=2):
            if board[r][c]:continue
            used=set(board[r])|{row[c] for row in board}|{board[i][j] for i in range(r//2*2,r//2*2+2) for j in range(c//2*2,c//2*2+2)}
            options=set(range(1,5))-used
            if not options:return
            choices.append((len(options),r,c,options))
        if not choices:answers.append(tuple(tuple(r) for r in board));return
        _,r,c,options=min(choices,key=lambda x:x[:3])
        for value in sorted(options):board[r][c]=value;solve();board[r][c]=0
    solve();return answers

def story_step(s,action,other):
    t,e,h,c=s
    if action=='befriend':t+=1;h=max(0,h-1)
    elif action=='investigate':e+=1;h+=1;t-=int(h>=2)
    elif action=='defuse':h-=2
    elif action=='rush':c+=1;h+=1
    elif action=='wait':h=max(0,h-1)
    else:raise ValueError(action)
    if other=='rival_sabotage':t-=1;h+=1
    elif other=='rival_flatter':t+=1;e=max(0,e-1)
    elif other!='rival_wait':raise ValueError(other)
    c+=int(action!='wait')
    return max(-3,min(3,t)),max(0,min(4,e)),max(0,min(4,h)),max(0,min(5,c))

def story_rival(s):
    t,e,h,c=s
    return 'rival_sabotage' if e>=2 and h<=3 else ('rival_flatter' if t<=-1 and e>=1 else 'rival_wait')

def grid_transform(g,name):
    if name.startswith('map_'):
        _,src,_,dst=name.split('_');return tuple(tuple(int(dst) if v==int(src) else v for v in r) for r in g)
    if name.startswith('fill_zero_with_'):
        dst=int(name[-1]);return tuple(tuple(dst if v==0 else v for v in r) for r in g)
    if name=='flip_h':return tuple(tuple(reversed(r)) for r in g)
    if name=='flip_v':return tuple(reversed(g))
    if name=='rotate_180':return tuple(tuple(reversed(r)) for r in reversed(g))
    raise ValueError(name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_gym05');p.add_argument('--routing-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.runs)/'cohort';seal=read('experiments/nan_gym05/SOURCE_MANIFEST.json')
    assert sha(root/'public.jsonl')==seal['combined_public_sha256'] and sha(root/'host_labels.jsonl')==seal['combined_host_sha256']
    public={r['id']:r for r in rows(root/'public.jsonl')};labels={r['id']:r for r in rows(root/'host_labels.jsonl')}
    pointer=0;sudoku=[];story=0;route=0
    for key,label in labels.items():
        if label['copy_control']:continue
        row=public[key]
        if label['suite']=='pointer':assert pointer_answer(row['prompt'])==label['answer'];pointer+=1
        elif label['suite']=='sudoku':
            solutions=sudoku_completions(label['givens']);reference=tuple(tuple(r) for r in label['answer'])
            assert reference in solutions
            sudoku.append({'root':label['root'],'legal_completions':len(solutions),'reference_is_legal':True})
        elif label['suite']=='storyworld':
            d=label['start_state'];s=tuple(d[k] for k in ('trust','evidence','heat','scene'))
            for action in label['answer']:
                if s[3]>=5:break
                s=story_step(s,action,story_rival(s))
            assert s[3]>=5 and s[0]>=1 and s[1]>=3 and s[2]<=2;story+=1
        elif label['suite']=='routing':
            env=label['answer'];file=Path(a.routing_root)/'normalized_trajectories'/(env+'.jsonl')
            assert sha(file)==label['source_file_sha256']
            with file.open(encoding='utf-8') as f:
                for i,line in enumerate(f):
                    if i==label['source_row']:source=json.loads(line);break
            assert source['env_id']==env and row['prompt'].split('\nTask:\n',1)[1]==source['state_prompt'];route+=1
    # Exhaustive finite transition verification, a distinct evidence class.
    from research_gym.envs.coupled_storyworld import CoupledStoryworldEnv,StoryState
    env=CoupledStoryworldEnv();transition_checks=0
    for s in product(range(-3,4),range(5),range(5),range(6)):
        state=StoryState(*s);assert env.other_policy(state)==story_rival(s)
        for action,other in product(env.self_actions,env.other_actions):
            native=env.step(state,action,other);assert tuple(native.to_dict().values())==story_step(s,action,other);transition_checks+=1
    from research_gym.envs.arc_tasks import default_arc1_tasks,default_arc2_tasks
    primitives=['map_1_to_2','map_2_to_3','fill_zero_with_1','fill_zero_with_2','flip_h','flip_v','rotate_180'];arc=[]
    for suite,tasks,programs in [('arc1',default_arc1_tasks(),[(x,) for x in primitives]),('arc2',default_arc2_tasks(),list(product(primitives,repeat=2)))]:
        def apply(g,program):
            for name in program:g=grid_transform(g,name)
            return g
        for task in tasks:
            fitting=[p for p in programs if all(apply(e.input_grid,p)==e.output_grid for e in task.train)]
            outputs={apply(task.test_input,p) for p in fitting};assert task.test_output in outputs
            arc.append({'root':suite+':'+task.task_id,'fitting_primitive_programs':len(fitting),'distinct_query_outputs':len(outputs),'reference_is_consistent':True})
    result={'status':'passed','scope':'source-only independent graph walks, Sudoku enumeration, story transition/witness checks, finite ARC hypotheses and routing label/prompt bindings; no model outcomes opened',
        'pointer_queries':pointer,'sudoku':sudoku,'ARC':arc,'story_witnesses':story,'finite_story_transitions':transition_checks,
        'routing_source_bindings':route,'neural_inference_or_mechanism_proved':False,'sealed_test_accessed':False}
    write(a.out,result);print(json.dumps(result))

if __name__=='__main__':main()
