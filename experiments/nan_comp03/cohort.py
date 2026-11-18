"""Fresh independent unary worlds; no existing roster or split is called."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from .prompt import body,tokens,aliases,RECIPROCAL_EXAMPLES
from experiments.nan_bpd01.verify_sources import unary as independent_unary

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def digest(x):return sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def poly(rng,family):
    a=rng.choice([-1,1]);rest=[rng.choice([-1,1]) for _ in range(3)] if family=='dense_p4' else [0,0,0]
    if family=='sparse_p4':rest[rng.randrange(3)]=rng.choice([-1,1])
    return [1,a]+rest+[0]*28

def main():
    p=argparse.ArgumentParser();p.add_argument('--evalmath-root',required=True);p.add_argument('--compactification-root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);p.add_argument('--previous-public',default='experiments/nan_comp02/fixtures/public.jsonl');a=p.parse_args()
    sys.path[:0]=[str(Path(a.evalmath_root)/'src'),a.compactification_root]
    from spiral_ln import algebraic_reasoning_oracle as O
    def forbid(*a,**k):raise RuntimeError('Existing roster access forbidden')
    O.roster=forbid
    from research.algernon.algebraic_reasoning_gym.arg02.nanbeige import load_tokenizer
    tok=load_tokenizer(Path(a.snapshot),166100);cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    previous=Path(a.previous_public);prior=[json.loads(s) for s in previous.read_text().splitlines()]
    input_key=lambda demos:digest(sorted(demos))
    seen_inputs={input_key(r['inputs']) for r in prior};prior_count=len(seen_inputs)
    seen_tokens={v:set() for v in ['balanced','curriculum','erased']}
    tutorials=[x for x,y in RECIPROCAL_EXAMPLES]
    assert all(independent_unary(x,'R')==y and list(O.unary(x,'R'))==y for x,y in RECIPROCAL_EXAMPLES)
    cases=[];public=[];pairs=[];attempts=[];seen=set()
    for split in ['calibration','development']:
        roots=cfg[split+'_roots'];ops=cfg[split+'_pairs']
        for index,(root,operators) in enumerate(zip(roots,ops)):
            rng=random.Random(root);family=cfg['families'][index%2];counts=Counter();accepted=None
            for trial in range(1,cfg['maximum_source_trials_per_root']+1):
                demos=[poly(rng,family) for _ in range(2)]
                if demos[0]==demos[1]:counts['duplicate_demo']+=1;continue
                if any(x[:5] in tutorials for x in demos):counts['tutorial_input_overlap']+=1;continue
                ikey=input_key([x[:5] for x in demos])
                if ikey in seen_inputs:counts['previous_or_current_input_pair_overlap']+=1;continue
                query=[1,rng.choice([-1,1])]+[rng.choice([-1,0,1]) for _ in range(31)]
                candidate=[]
                for role,op in zip(['recipient','donor'],operators):
                    outputs=[list(O.unary(x,op)) for x in demos];future=list(O.unary(query,op))
                    assert outputs==[independent_unary(x,op) for x in demos] and future==independent_unary(query,op)
                    if max(abs(v) for y in outputs for v in y[:5])>cfg['maximum_observed_absolute_coefficient']:counts['large_public_coefficient']+=1;break
                    valid=[g for g in 'INSR' if all(list(O.unary(x,g))[:5]==y[:5] for x,y in zip(demos,outputs))]
                    if valid!=[op]:counts['ambiguous_operator']+=1;break
                    row={'case_id':f'nan-comp03:{root}:{role}','root':root,'split':split,'family':family,'inputs':[x[:5] for x in demos],'outputs':[y[:5] for y in outputs],'query':query[:5]}
                    row['public_sha256']=digest({k:v for k,v in row.items() if k not in ['case_id','root','split','family']})
                    label={'case_id':row['case_id'],'root':root,'split':split,'operator':op,'inputs':demos,'outputs':outputs,'query':query,'future':future}
                    lengths={v:len(tokens(tok,a.snapshot,row,v)) for v in ['balanced','curriculum','erased']}
                    candidate.append((row,label,lengths))
                if len(candidate)!=2:continue
                if candidate[0][1]['future']==candidate[1][1]['future']:counts['target_collision']+=1;continue
                if any(candidate[0][2][v]!=candidate[1][2][v] for v in ['balanced','curriculum','erased']):counts['length_mismatch']+=1;continue
                if max(candidate[0][2].values())>cfg['maximum_prompt_tokens']:counts['prompt_too_long']+=1;continue
                if any(c[0]['public_sha256'] in seen for c in candidate):counts['duplicate_public']+=1;continue
                assert tokens(tok,a.snapshot,candidate[0][0],'erased')==tokens(tok,a.snapshot,candidate[1][0],'erased')
                fingerprints={v:[digest(tokens(tok,a.snapshot,c[0],v)) for c in candidate] for v in seen_tokens}
                if any(any(h in seen_tokens[v] for h in hs) for v,hs in fingerprints.items()):counts['rendered_prompt_overlap']+=1;continue
                assert all(len(set(fingerprints[v]))==2 for v in ['balanced','curriculum'])
                accepted=candidate;break
            if accepted is None:raise RuntimeError(f'Unfillable source root {root}: {counts}')
            seen_inputs.add(ikey)
            for v,hs in fingerprints.items():seen_tokens[v].update(hs)
            for row,label,lengths in accepted:public.append(row);cases.append(label);seen.add(row['public_sha256'])
            pairs.append({'root':root,'split':split,'family':family,'recipient':accepted[0][0]['case_id'],'donor':accepted[1][0]['case_id'],'prompt_tokens':accepted[0][2],
                          'distinct_verified_futures':True,'erased_token_ids_identical':True})
            attempts.append({'root':root,'source_trials':trial,'rejections':dict(counts)})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('public.jsonl',public),('host_labels.jsonl',cases)]:
        (out/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    write(out/'pairs.json',pairs)
    write(out/'source_receipt.json',{'source_only':True,'checkpoint_outcomes_used':False,'existing_roster_called':False,'sealed_test_accessed':False,
        'previous_public_sha256':sha256(previous.read_bytes()).hexdigest(),'previous_exposed_input_pairs_excluded':prior_count,
        'rendered_prompt_counts':{v:len(hs) for v,hs in seen_tokens.items()},'CAL_DEV_actual_token_prompts_disjoint':True,
        'reciprocal_tutorials_verified':RECIPROCAL_EXAMPLES,'tutorial_inputs_excluded':tutorials,
        'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'oracle_sha256':sha256(Path(O.__file__).read_bytes()).hexdigest(),
        'independent_checker_sha256':sha256(Path(independent_unary.__code__.co_filename).read_bytes()).hexdigest(),'aliases':aliases(tok),
        'tutorial_verified':{op:independent_unary([1,1,0,0,0],op) for op in 'INSR'},'pairs':pairs,'attempts':attempts,
        'files_sha256':{n:sha256((out/n).read_bytes()).hexdigest() for n in ['public.jsonl','host_labels.jsonl','pairs.json']}})
    print(json.dumps({'worlds':len(cases),'roots':len(pairs),'attempts':attempts,'aliases':aliases(tok)}))

if __name__=='__main__':main()
