"""New individual polynomials; only explicitly exposed prior DEV fixtures excluded."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from experiments.nan_bpd01.verify_sources import unary
from .prompt import make_prompts,tokens,OPS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8',newline='\n')
def poly(rng,family):
    a=[1,rng.choice([-1,1])]+[0]*3
    if family=='sparse2':a[2+rng.randrange(3)]=rng.choice([-2,2])
    else:
        a[2:]=[rng.choice([-2,-1,1,2]) for _ in range(3)]
        if not any(abs(x)==2 for x in a[2:]):return None
    return a

def main():
    p=argparse.ArgumentParser();p.add_argument('--evalmath-root',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path[:0]=[str(Path(a.evalmath_root)/'src'),a.compactification_root]
    from spiral_ln import algebraic_reasoning_oracle as O
    O.roster=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Existing rosters forbidden'))
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok=load_tokenizer(Path(a.snapshot),166100);cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    priorpaths=[Path('experiments')/name/'fixtures/public.jsonl' for name in ('nan_comp02','nan_comp03')]
    seen={tuple(x) for path in priorpaths for r in map(json.loads,path.read_text().splitlines()) for x in r['inputs']};old_count=len(seen)
    admitted=[];public=[];attempts=[];prompt_seen={};source_seen=set()
    for split in ('calibration','development'):
        for i,root in enumerate(cfg[split+'_roots']):
            family=cfg['families'][i%2];rng=random.Random(root);rejects=Counter()
            for trial in range(1,cfg['maximum_source_trials']+1):
                inputs=[poly(rng,family) for _ in range(2)]
                if None in inputs:rejects['dense_requires_coefficient_two']+=1;continue
                if any(tuple(x) in seen for x in inputs) or inputs[0]==inputs[1]:rejects['previous_or_current_individual_input']+=1;continue
                full=[x+[0]*28 for x in inputs];refs={op:[list(O.unary(x,op))[:5] for x in full] for op in OPS}
                assert all(refs[op]==[unary(x,op)[:5] for x in full] for op in OPS)
                if len({digest(v) for v in refs.values()})!=4:rejects['ambiguous_operator']+=1;continue
                if max(abs(v) for v in refs['R'][0])>cfg['integer_magnitude_bound']:rejects['integer_bound']+=1;continue
                alt=cfg['alternative_operators'][i%8];worlds=['R',alt] if i%2==0 else [alt,'R']
                codes={op:'ABCD'[(j+i)%4] for j,op in enumerate(OPS)};order=list(OPS[i%4:]+OPS[:i%4]);rng.shuffle(order)
                prompts=make_prompts(inputs,refs,worlds,codes,order);ids={k:tokens(tok,a.snapshot,v) for k,v in prompts.items()}
                if max(map(len,ids.values()))>cfg['maximum_prompt_tokens']:rejects['prompt_bound']+=1;continue
                unique={name:digest(value) for name,value in ids.items() if not name.startswith(('copy/','labelcopy/','codecopy/')) and name!='erased_repeat'}
                if any(h in prompt_seen.get(k,set()) for k,h in unique.items()):rejects['actual_prompt_overlap']+=1;continue
                assert ids['erased']==ids['erased_repeat']
                query=[1,rng.choice([-1,1])]+[rng.choice([-1,0,1]) for _ in range(31)]
                futures={op:list(O.unary(query,op)) for op in worlds};assert all(futures[op]==unary(query,op) for op in worlds)
                assert futures[worlds[0]]!=futures[worlds[1]]
                row={'root':root,'split':split,'family':family,'prompts':prompts};row['public_sha256']=digest(row)
                label={'root':root,'split':split,'family':family,'inputs':full,'refs':refs,'worlds':worlds,'codes':codes,'table_order':order,'query':query,'futures':futures}
                break
            else:raise RuntimeError(f'Unfillable root {root}: {rejects}')
            public.append(row);admitted.append(label);seen.update(map(tuple,inputs));source_seen.update(map(tuple,inputs))
            for k,h in unique.items():prompt_seen.setdefault(k,set()).add(h)
            attempts.append({'root':root,'split':split,'trials':trial,'rejections':dict(rejects),'prompt_lengths':{k:len(v) for k,v in ids.items()}})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('public.jsonl',public),('host_labels.jsonl',admitted)]:
        (out/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8',newline='\n')
    write(out/'source_receipt.json',{'source_only':True,'model_outcomes_used':False,'sealed_test_accessed':False,'existing_roster_called':False,
        'previous_public_files_sha256':{str(p):sha(p) for p in priorpaths},'prior_individual_polynomials_excluded':old_count,'new_individual_polynomials':len(source_seen),
        'actual_prompt_counts':{k:len(v) for k,v in prompt_seen.items()},'copy_controls_may_repeat':True,'attempts':attempts,
        'oracle_sha256':sha(Path(O.__file__)),'independent_checker_sha256':sha(Path(unary.__code__.co_filename)),
        'files_sha256':{n:sha(out/n) for n in ('public.jsonl','host_labels.jsonl')}})
    print(json.dumps({'roots':len(public),'individual_polynomials':len(source_seen),'attempts':[{'root':r['root'],'trials':r['trials'],'rejections':r['rejections']} for r in attempts]}))

if __name__=='__main__':main()
