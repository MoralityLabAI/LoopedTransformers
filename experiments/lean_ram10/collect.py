"""RAM-10 fresh MathCan TRAIN/DEV compiler miner and independent replay.

The original TEST modules must be absent from the collector's checkout.
No teacher prefix, source reference script or PEF theorem is used for labels.
"""
from __future__ import annotations
import argparse
from collections import Counter,deque,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import traceback
from ram10_core import (sha_file,stable_json,append_json,rows,source_order_roots,
                        verify_checkout,registered_ram08_core,gate)


def old_runtime():
    from ram10_core import RAM08
    if str(RAM08) not in sys.path:sys.path.insert(0,str(RAM08))
    from runtime import Session,ResourceGuard,ProbeTimeout
    return Session,ResourceGuard,ProbeTimeout


def source_text(mathcan,item):
    source=(Path(mathcan)/item['source_file']).read_text(encoding='utf8')
    from ram10_core import sha_bytes
    if sha_bytes(source.encode('utf8'))!=item['source_sha256']:
        raise AssertionError('original MathCan source bytes changed')
    return source


def bounded_successors(successors,prioritize):
    named=[s for s in successors if s[0][-1].startswith('intro ram07_arg')]
    constructors=[s for s in successors if s[0][-1]=='constructor' and len(s[1]['goals'])>1]
    branch=[s for s in successors if len(s[1]['goals'])>1 and s not in constructors]
    single=[s for s in successors if len(s[1]['goals'])==1 and s not in named]
    prioritize.extendleft(reversed(constructors[:1]+named[:1]))
    return branch[:3]+single[:3]


def collect_root(mathcan,root,index,root_out,out,lean,core,Session,guard,config):
    source=source_text(mathcan,root)
    root_hash=core.sha(source)
    visited=set();queue=deque();high=deque()
    generated=0;truncated=0;states=0;observations=Counter();total=0
    with Session(lean,source,root['theorem'],root_out,guard) as server:
        state=server.initial
        if not state['accepted'] or not state['goals']:
            raise AssertionError('original theorem has no observable pending Lean goal')
        queue.append(([],state));visited.add(core.signature(state['goals']))
        while (high or queue) and states<config['collector']['states_per_root']:
            actions,before=(high if high else queue).popleft()
            states+=1
            if len(actions)>=config['collector']['max_transitions']:continue
            state_sig=core.signature(before['goals'])
            append_json(out/'states.jsonl',dict(**root,actions=actions,state=before,state_sha256=state_sig))
            pool=core._ram10_original_candidates(before)
            selected=core.candidates(before)
            if not set(selected).issubset(set(pool)):raise AssertionError('changed candidate grammar')
            generated+=len(pool);truncated+=len(pool)-len(selected)
            next_states=[]
            for family,script in selected:
                guard.check()
                try:
                    after=server.query(actions+[script],config['collector']['per_request_seconds'])
                except Exception as err:
                    append_json(out/'unknown.jsonl',dict(**root,prefix=actions,script=script,
                             classification='unknown',exception=repr(err)))
                    raise
                label=core.label(before,after)
                entry=dict(**root,prefix=actions,before=before,after=after,family=family,script=script,
                           label=label,delta=max(-3,min(3,len(before['goals'])-len(after.get('goals',[])))) if after['accepted'] else 0,
                           state_sha256=state_sig)
                append_json(out/'transitions.jsonl',entry)
                observations[family+'/'+str(label)]+=1;total+=1
                if label>0 and after['goals'] and len(actions)+1<config['collector']['max_transitions']:
                    if len(after['goals'])>config['collector']['max_goals']:continue
                    sig=core.signature(after['goals'])
                    if sig not in visited:
                        visited.add(sig);next_states.append((actions+[script],after))
            rest=bounded_successors(next_states,high)
            queue.extend(rest)
    report=dict(**root,observed_states=states,candidate_records=total,
                generated_candidates=generated,censored_unqueried_candidates=truncated,
                outcomes=dict(observations),fresh_source=True)
    print('RAM10_COLLECT',json.dumps({k:report[k] for k in ('source_file','theorem','partition','observed_states','candidate_records','censored_unqueried_candidates')},sort_keys=True),flush=True)
    return report


def independent_audit(mathcan,out,lean,core,Session,guard,config):
    records=rows(out/'transitions.jsonl')
    by_root=defaultdict(list)
    for row in records:by_root[row['source_file'],row['theorem']].append(row)
    counts=0;disagreements=[]
    for i,((file,name),group) in enumerate(by_root.items()):
        original=source_text(mathcan,group[0])
        if core.sha(original)!=group[0]['source_sha256']:raise AssertionError('original source SHA changed at audit')
        with Session(lean,original,name,out/'independent_replay'/f'{i:03d}',guard) as server:
            cache={():server.initial}
            for entry in group:
                prefix=tuple(entry['prefix'])
                if prefix not in cache:
                    cache[prefix]=server.query(list(prefix),config['collector']['per_request_seconds'])
                before=cache[prefix]
                if before['goals']!=entry['before']['goals'] or sorted(before.get('definitions',[]))!=sorted(entry['before'].get('definitions',[])):
                    disagreements.append((file,name,entry['script'],'before state'))
                # Independently recompute every original compiler effect, including failed ones.
                effect=server.query(list(prefix)+[entry['script']],config['collector']['per_request_seconds'])
                prior=entry['after']
                same=effect['accepted']==prior['accepted']
                if same and effect['accepted']:
                    same=effect['goals']==prior['goals'] and sorted(effect.get('definitions',[]))==sorted(prior.get('definitions',[]))
                same=same and core.label(before,effect)==entry['label']
                if not same:disagreements.append((file,name,entry['script'],'compiler effect'))
                counts+=1
        print('RAM10_INDEPENDENT_AUDIT',i+1,len(by_root),counts,len(disagreements),flush=True)
    report=dict(status='passed' if not disagreements else 'failed',replayed=counts,source_roots=len(by_root),
                disagreements=disagreements[:30],dataset_sha256=sha_file(out/'transitions.jsonl'),
                independent_sessions=len(by_root),test_cases_seen=0,source_reference_proofs_used=0)
    stable_json(out/'independent_audit.json',report)
    if disagreements:raise AssertionError('independent original Lean action replay mismatch')
    return report


def run(mathcan,config_path,lean,out,*,audit_only=False):
    mathcan=Path(mathcan);out=Path(out);config=json.loads(Path(config_path).read_text())
    verify_checkout(mathcan,config,'train')
    core=registered_ram08_core(config)
    Session,ResourceGuard,_=old_runtime()
    if not audit_only:
        out.mkdir(parents=True,exist_ok=False)
    with ResourceGuard(out/'resources.jsonl') as guard:
        if not audit_only:
            selected=[]
            for partition in ('train','dev'):selected.extend(source_order_roots(mathcan,config,partition))
            assert {x['source_file'] for x in selected}.isdisjoint({f'MathCan/{m}.lean' for m in config['modules']['test']})
            stable_json(out/'source_selection.json',dict(schema='RAM10-source-selection-v1',
                mathcan_commit=config['source_mathcan'],config_sha256=sha_file(config_path),
                roots=selected,source_test_filenames_not_opened=True))
            collected=[]
            for i,root in enumerate(selected):
                rec=collect_root(mathcan,root,i,out/'collection'/f'{i:03d}',out,lean,core,Session,guard,config)
                collected.append(rec)
                stable_json(out/'collection_report.json',dict(status='collecting',completed=collected))
            stable_json(out/'collection_report.json',dict(status='complete',roots=collected,
                 train_roots=sum(r['partition']=='train' for r in collected),dev_roots=sum(r['partition']=='dev' for r in collected),
                 dataset_sha256=sha_file(out/'transitions.jsonl'),config_sha256=sha_file(config_path),
                 source_test_filenames_not_opened=True,source_reference_proofs_used=0,
                 censored_scripts=sum(r['censored_unqueried_candidates'] for r in collected)))
        checked=independent_audit(mathcan,out,lean,core,Session,guard,config)
        scored=rows(out/'transitions.jsonl')
        verdict=gate(scored,config)
        verdict.update(dataset_sha256=sha_file(out/'transitions.jsonl'),config_sha256=sha_file(config_path),
                       independently_replayed=checked['replayed'],train_effects=sum(r['partition']=='train' for r in scored),
                       dev_effects=sum(r['partition']=='dev' for r in scored),test_theorem_effects=0)
        stable_json(out/'training_gate.json',verdict)
        stable_json(out/'resource_receipt.json',guard.receipt())
        print('RAM10_GENUINE_TACTIC_GATE',json.dumps(verdict,sort_keys=True),flush=True)
        if not verdict['admitted']:
            raise RuntimeError('RAM-10 original Lean source TRAIN has insufficient independently verified exact/rw/branch diversity; corpus preserved, TEST forbidden')
        return verdict

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mathcan',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--lean',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    run(args.mathcan,args.config,args.lean,args.out,audit_only=args.audit_only)
