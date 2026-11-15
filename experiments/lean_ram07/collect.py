"""Collect new train-only branch transitions, then audit in fresh Lean sessions."""
from __future__ import annotations
import argparse
from collections import Counter,deque
import json
import re
from pathlib import Path
from runtime import Session,ResourceGuard,ProbeTimeout
import ram07_core as c

def roots(data):
    state=data/'train_source/actual_goal_states.jsonl'
    if c.digest(state)!='403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0':raise ValueError('old states changed')
    observations=c.read_rows(data/'train_probes/train_candidate_outcomes.jsonl')
    keys=('stage_id','family','script','accepted','progress','closed_in_scratch','remaining_goal_count','compiler_returncode','candidate_source_sha256')
    text=''.join(json.dumps({k:r.get(k) for k in keys},sort_keys=True,ensure_ascii=False)+'\\n' for r in observations)
    if c.sha(text)!='bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec':raise ValueError('old observations changed')
    result=[dict(source_file=r['source_file'],theorem=r['theorem']) for r in c.read_rows(state) if r['stage_index']==0]
    assert len(result)==27
    return result

EXTRA_TRAIN_SOURCE_FILES=(
    "MathCan/Hodge.lean",
    "MathCan/FundedHooks.lean",
    "MathCan/TOEJordan.lean",
    "MathCan/TOETensorConstraints.lean",
)

def extended_roots(data,mathcan,config):
    original=roots(data)
    result=list(original)
    selected={(x["source_file"],x["theorem"]) for x in original}
    duplicates=[]
    for filename in EXTRA_TRAIN_SOURCE_FILES:
        text=(mathcan/filename).read_text(encoding="utf8")
        names=re.findall(r"^theorem\s+([A-Za-z_0-9']+)",text,re.M)[:5]
        if len(names)!=5:
            raise AssertionError("changed first-five extension selector: "+filename)
        for name in names:
            key=(filename,name)
            if key in selected:
                duplicates.append(filename+":"+name)
                continue
            if filename+":"+name in config["development"]:
                raise AssertionError("extended root enters the five-theorem dev split")
            result.append(dict(source_file=filename,theorem=name))
            selected.add(key)
    assert len(original)==27 and len(result)==47-len(duplicates)
    return result,duplicates


def collect(args,config,guard):
    all_roots,duplicates=extended_roots(args.data,args.mathcan,config);receipt=[]
    for i,root in enumerate(all_roots):
        key=root['source_file']+':'+root['theorem']
        source=(args.mathcan/root['source_file']).read_text(encoding='utf8')
        out=args.out/'collection'/f'{i:02d}'
        rows=[];visited=set();branches=deque();ordinary=deque();states=0
        with Session(args.lean,source,root['theorem'],out,guard) as server:
            ordinary.append(([],server.initial));visited.add(c.signature(server.initial['goals']))
            while (branches or ordinary) and states<config['collector']['states_per_theorem']:
                chosen=branches if branches and (states%2==1 or not ordinary) else ordinary
                actions,before=chosen.popleft();states+=1
                c.append_json(args.out/'states.jsonl',dict(**root,actions=actions,state=before,
                    state_sha256=c.signature(before['goals']),development=key in config['development']))
                successors=[]
                for family,script in c.candidates(before):
                    guard.check()
                    try:after=server.query(actions+[script],config['collector']['request_seconds'])
                    except ProbeTimeout:
                        c.append_json(args.out/'timeouts.jsonl',dict(**root,actions=actions+[script],label=None))
                        raise RuntimeError('collection timeout requires an independent audit; no failure label fabricated')
                    outcome=c.label(before,after)
                    record=dict(**root,development=key in config['development'],prefix=actions,
                        before=before,after=after,family=family,script=script,label=outcome,
                        delta=max(-3,min(3,len(before['goals'])-len(after.get('goals',[])))) if after['accepted'] else 0,
                        state_sha256=c.signature(before['goals']),source_sha256=c.sha(source))
                    rows.append(record);c.append_json(args.out/'transitions.jsonl',record)
                    if outcome>0 and after['goals'] and len(actions)+1<config['collector']['depth'] and len(after['goals'])<=config['collector']['max_goals']:
                        sig=c.signature(after['goals'])
                        if sig not in visited:
                            visited.add(sig);successors.append((actions+[script],after))
                # Keep genuine branching states even when easier one-goal progress exists.
                multi=[s for s in successors if len(s[1]['goals'])>1]
                single=[s for s in successors if len(s[1]['goals'])==1]
                multi.sort(key=lambda s:(len(s[1]['goals']),len(s[0])))
                # Training-only prereg amendment A: visit a legitimate
                # source-bound named introduction immediately, so its next
                # compiler state is mined before breadth-first expansion
                # exhausts the unchanged 12-state per-theorem cap.
                # No gold-prefix action, extra hypothesis, goal duplication,
                # new tactic or heldout source is introduced.
                named=[item for item in single if item[0] and
                    item[0][-1].startswith('intro ram07_arg')]
                branches.extend(multi[:3])
                if named:
                    ordinary.appendleft(named[0])
                    ordinary.extend([item for item in single
                                     if item is not named[0]][:2])
                else:
                    ordinary.extend(single[:3])
        summary=dict(**root,states=states,rows=len(rows),multi_input_rows=sum(len(r['before']['goals'])>1 for r in rows),
          outcomes=dict(Counter(r['label'] for r in rows)),development=key in config['development'])
        receipt.append(summary);c.write_json(args.out/'collection_report.json',dict(roots=receipt,status='collecting'))
        print('RAM07_COLLECT',json.dumps(summary),flush=True)
    result=dict(status='complete',roots=receipt,extension_duplicate_roots=duplicates,
                source_selection="original_27_plus_first_five_from_four_old_modules",
                dataset_sha256=c.digest(args.out/'transitions.jsonl'),
                state_file_sha256=c.digest(args.out/'states.jsonl'),config_sha256=c.digest(args.config))
    c.write_json(args.out/'collection_report.json',result)
    return result

def audit(args,config,guard):
    records=c.read_rows(args.out/'transitions.jsonl');groups={}
    for record in records:groups.setdefault((record['source_file'],record['theorem']),[]).append(record)
    checked=0;failures=[]
    for i,((filename,theorem),rows) in enumerate(groups.items()):
        source=(args.mathcan/filename).read_text(encoding='utf8')
        audit_dir=Path(args.audit_name).stem.replace('_','-') if args.audit_name!='independent_audit.json' else 'independent-audit'
        with Session(args.lean,source,theorem,args.out/audit_dir/f'{i:02d}',guard) as server:
            parents={}
            for row in rows:
                if row['source_sha256']!=c.sha(source):raise ValueError('original source hash differs during audit')
                prefix=tuple(row['prefix'])
                if prefix not in parents:
                    parents[prefix]=server.initial if not prefix else server.query(list(prefix),config['collector']['request_seconds'])
                    if c.signature(parents[prefix]['goals'])!=row['state_sha256']:failures.append(dict(file=filename,theorem=theorem,reason='before_state'))
                result=server.query(row['prefix']+[row['script']],config['collector']['request_seconds'])
                agree=result['accepted']==row['after']['accepted']
                if result['accepted']:
                    agree=agree and c.signature(result['goals'])==c.signature(row['after']['goals']) and sorted(result['definitions'])==sorted(row['after']['definitions'])
                agree=agree and c.label(parents[prefix],result)==row['label']
                if not agree:failures.append(dict(file=filename,theorem=theorem,script=row['script'],reason='outcome_disagreement'))
                checked+=1
        print('RAM07_AUDIT',i+1,len(groups),checked,len(failures),flush=True)
    report=dict(status='passed' if not failures else 'failed',checked=checked,failures=failures,
      dataset_sha256=c.digest(args.out/'transitions.jsonl'),config_sha256=c.digest(args.config),
      independent_fresh_lean_sessions=len(groups),old_labels_modified=False)
    c.write_json(args.out/args.audit_name,report)
    if failures:raise AssertionError('new dataset failed independent compiler replay')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--mathcan',type=Path,required=True)
    p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--audit-name',default='independent_audit.json')
    a=p.parse_args();config=json.loads(a.config.read_text())
    if not a.audit_only:a.out.mkdir(parents=True,exist_ok=False)
    with ResourceGuard(a.out/'resources.jsonl') as guard:
        if not a.audit_only:collect(a,config,guard)
        audit(a,config,guard)
        c.write_json(a.out/'resource_receipt.json',guard.receipt())
if __name__=='__main__':main()
