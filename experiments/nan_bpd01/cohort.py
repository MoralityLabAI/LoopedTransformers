"""Source-only fresh development worlds; never calls an existing roster."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,replace
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import sys

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--evalmath-root',required=True);p.add_argument('--compactification-root',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path[:0]=[str(Path(a.evalmath_root)/'src'),a.compactification_root]
    from spiral_ln import algebraic_reasoning_oracle as O
    def forbidden(*a,**k):raise RuntimeError('Existing roster access forbidden')
    O.roster=forbidden
    from research.algernon.algebraic_reasoning_gym.arg02 import nanbeige
    from .backend import prompt_ids
    # Backend helpers are imported, but no model is created in this source process.
    tok=nanbeige.load_tokenizer(Path(a.snapshot),166100)
    cfg=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    cases=[];pairs=[];attempts=[];seen=set()
    for split,roots,multisets in [('calibration',cfg['calibration_roots'],cfg['calibration_program_multisets']),
                                  ('development',cfg['development_roots'],cfg['development_program_multisets'])]:
        for index,(root,multiset) in enumerate(zip(roots,multisets)):
            rng=random.Random(root);family=cfg['families'][index];counts=Counter();accepted=None
            for trial in range(1,1001):
                topology=O.TOPOLOGIES[(index+trial-1)%4]
                graphs=sorted(set(permutations(multiset)));rng.shuffle(graphs)
                base=O.make_case(rng,split=split,family=family,topology=topology,graph=graphs[0],identifier=f'nan-bpd01:{root}:recipient',pair_id=str(root))
                options=[]
                for graph in graphs:
                    c=replace(base,source_graph=graph,demos=tuple((f,O.execute(f,topology,graph)) for f,_ in base.demos),future=O.execute(base.query,topology,graph))
                    accepted_programs=[g for g in O.PROGRAMS if O.check_candidate(c,g)['accepted']]
                    futures={O.execute(c.query,topology,g) for g in accepted_programs}
                    if len(futures)!=1 or c.future not in futures:counts['ambiguous_future']+=1;continue
                    if not O.grade(c,O.search(c,[graph],1))['future_correct']:raise AssertionError('Oracle positive control failed')
                    public=c.public|{'public_sha256':c.signature}
                    length=len(prompt_ids(tok,a.snapshot,public))
                    if length>cfg['maximum_prompt_tokens']:counts['prompt_too_long']+=1;continue
                    options.append((c,length))
                for i,(c,length) in enumerate(options):
                    for d,dl in options[i+1:]:
                        if c.future==d.future:counts['target_collision']+=1;continue
                        if length!=dl:counts['length_mismatch']+=1;continue
                        if c.signature in seen or d.signature in seen:counts['duplicate_public']+=1;continue
                        d=replace(d,identifier=f'nan-bpd01:{root}:donor')
                        accepted=(c,d,length);break
                    if accepted:break
                if accepted:break
            if not accepted:raise RuntimeError(f'No source-only matched pair for root {root}: {counts}')
            c,d,length=accepted;cases.extend([c,d]);seen.update([c.signature,d.signature])
            pairs.append({'root':root,'split':split,'family':family,'recipient':c.identifier,'donor':d.identifier,
                          'prompt_tokens':length,'same_operator_multiset':True,'different_verified_future':True})
            attempts.append({'root':root,'source_trials':trial,'rejections':dict(counts),'retained_pairs':1})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('public.jsonl',[c.public|{'public_sha256':c.signature} for c in cases]),('host_labels.jsonl',[asdict(c) for c in cases])]:
        (out/name).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    write(out/'pairs.json',pairs)
    write(out/'source_receipt.json',{'schema_version':1,'source_only':True,'model_outcomes_used':False,'sealed_test_accessed':False,
        'existing_roster_called':False,'oracle_path':str(Path(O.__file__).resolve()),'oracle_sha256':sha(O.__file__),
        'config_sha256':sha(Path(__file__).with_name('config.json')),'generator_sha256':sha(__file__),
        'pairs':pairs,'attempts':attempts,'files_sha256':{n:sha(out/n) for n in ['public.jsonl','host_labels.jsonl','pairs.json']}})
    print(json.dumps({'pairs':len(pairs),'worlds':len(cases),'attempts':attempts}))

if __name__=='__main__':main()
