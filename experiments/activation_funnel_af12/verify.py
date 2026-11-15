"""AF-12 independent Horn semantics, all-logit replay, and checkpoint audit."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af11.core import replay
from experiments.activation_funnel_af10.core import latch_suffix
from . import data
from . import experiment as ex
from .core import trace_upstream


def independent_layers(case,steps):
    # Reconstruct from explicit rules, not the AF-12 training-label function.
    edges=gym.logical_rules(case)[0]
    state=set(gym.initial_facts(case))
    phys,_=data.physical_permutations(case.permutation)
    result=[]
    for _ in range(steps):
        new=[]
        for premises,head in edges:
            if all(x in state for x in premises):new.append(head)
        state.update(new)
        result.append([int(i in state) for i in sorted(range(gym.ATOMS),key=phys.__getitem__)])
    return result


def argmax(pair):
    if len(pair)!=2 or not all(math.isfinite(float(x)) for x in pair):
        raise ValueError('nonfinite/invalid logits')
    return int(pair[1]>pair[0])


def close(a,b,tol=3e-5):
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return all(close(x,y,tol) for x,y in zip(a,b))
    try:return abs(float(a)-float(b))<=tol and math.isfinite(float(a))
    except (ValueError,TypeError):return False


def audit(training,evaluation,seed,out,smoke=False):
    training,evaluation,out=map(Path,(training,evaluation,out))
    receipt,models=ex.load_models(training,seed,smoke)
    score=json.loads((evaluation/'evaluation.json').read_text())
    if (score['experiment'],score['seed'],score['smoke'])!=('AF-12',seed,smoke):
        raise ValueError('AF12 evaluation identity mismatch')
    if score['train_receipt_sha256']!=ex.sha(training/'train_receipt.json') or score['manifest_sha256']!=ex.verify_manifest() or score['source_hashes']!=ex.source_hashes():
        raise ValueError('AF12 source/fit receipt was altered')
    if score['clean_sha256']!=ex.sha(evaluation/'clean.jsonl') or score['patched_sha256']!=ex.sha(evaluation/'patched.jsonl'):
        raise ValueError('AF12 evidence digest mismatch')
    clean=[json.loads(line) for line in (evaluation/'clean.jsonl').read_text().splitlines()]
    patch=[json.loads(line) for line in (evaluation/'patched.jsonl').read_text().splitlines()]
    splits=('dev',) if smoke else ex.FAMILIES
    datasets={k:data.Dataset(k) for k in splits}
    expected_clean=len(ex.ARMS)*sum(len(datasets[k].cases) for k in splits)
    if len(clean)!=expected_clean or len(patch)!=2*expected_clean or (score['clean_rows'],score['patched_rows'])!=(expected_clean,2*expected_clean):
        raise ValueError('AF12 record counts incomplete')
    ids={split:{c.id:(i,c) for i,c in enumerate(ds.cases)} for split,ds in datasets.items()}
    cache={};seen_clean=set();calc={}
    for arm in ex.ARMS:
        for split,ds in datasets.items():
            calc[arm,split]={'n':len(ds.cases),'joint_q_correct':0,'goal_correct':0,
                'relay_final_correct':0,'relay_final_total':0,'trajectory_correct':0,
                'informative':0,'informative_correct':0,'unaffected':0,'unaffected_correct':0,
                'false_goal':0,'patches':0,'patch_other_correct':0}
    for row in clean:
        arm,split,cid=row['arm'],row['split'],row['case_id']
        key=(arm,split,cid)
        if key in seen_clean or arm not in ex.ARMS or split not in datasets or cid not in ids[split]:
            raise ValueError('AF12 duplicate/unknown clean trial')
        seen_clean.add(key)
        i,c=ids[split][cid];truth=gym.oracle(c)[0]
        physical=independent_layers(c,ex.PRE[c.depth])
        # independent_layer indexing yields physical order by inverse permutation.
        if tuple(row['truth'])!=truth or row['depth']!=c.depth or len(row['closure_stage_logits'])!=len(physical):
            raise ValueError('AF12 clean target/trajectory mismatch')
        if argmax(row['clean_goal_logits'])==truth[2]:calc[arm,split]['goal_correct']+=1
        qp=[argmax(x) for x in row['prefix_q_logits']]
        calc[arm,split]['joint_q_correct']+=int(tuple(qp)==truth[:2])
        correct=True
        for stage,labels in zip(row['closure_stage_logits'],physical):
            if len(stage)!=gym.ATOMS:raise ValueError('AF12 stage atom roster incomplete')
            for logits,t in zip(stage,labels):
                pred=argmax(logits)
                correct &= (pred==t)
        calc[arm,split]['trajectory_correct']+=int(correct)
        for logits,t in zip(row['closure_stage_logits'][-1],physical[-1]):
            calc[arm,split]['relay_final_total']+=1
            calc[arm,split]['relay_final_correct']+=int(argmax(logits)==t)
        cache[key]=row
    if len(seen_clean)!=expected_clean:raise ValueError('AF12 missing clean trials')
    seen_patch=set()
    for row in patch:
        arm,split,cid,j=row['arm'],row['split'],row['case_id'],row['selected']
        key=(arm,split,cid,j)
        if key in seen_patch or (arm,split,cid) not in seen_clean or j not in (0,1):
            raise ValueError('AF12 duplicate/unknown patch case')
        seen_patch.add(key)
        i,c=ids[split][cid];ds=datasets[split]
        donor_idx,composed=ds.donors[i,j]
        orig=gym.oracle(c)[0]
        if row['donor_case_id']!=ds.cases[donor_idx].id or tuple(row['recipient'])!=orig or tuple(row['composed'])!=composed:
            raise ValueError('AF12 patched donor/Horn target mismatch')
        other=1-j;inf=orig[other] if c.gate=='and' else 1-orig[other]
        if row['influence']!=inf or row['untouched_q_code_exact'] is not True:
            raise ValueError('AF12 influence/causal-cut proof violated')
        pred=argmax(row['patch_goal_logits']);other_pred=argmax(row['patch_q_logits'][other])
        z=calc[arm,split]
        z['informative']+=inf;z['informative_correct']+=inf*int(pred==composed[2])
        z['unaffected']+=1-inf;z['unaffected_correct']+=(1-inf)*int(pred==composed[2])
        z['false_goal']+=int(pred==1 and composed[2]==0)
        z['patches']+=1;z['patch_other_correct']+=int(other_pred==orig[other])
    if len(seen_patch)!=2*expected_clean:raise ValueError('AF12 missing patched pairs')
    for (arm,split),z in calc.items():
        z['joint_q_accuracy']=z['joint_q_correct']/z['n'];z['goal_accuracy']=z['goal_correct']/z['n']
        z['relay_final_accuracy']=z['relay_final_correct']/z['relay_final_total']
        z['trajectory_exact_rate']=z['trajectory_correct']/z['n']
        z['informative_flip_accuracy']=z['informative_correct']/max(1,z['informative'])
        z['irrelevant_retention']=z['unaffected_correct']/max(1,z['unaffected'])
        z['patched_untouched_q_accuracy']=z['patch_other_correct']/z['patches']
        expected=score['scores'][arm][split]
        for k,v in z.items():
            if k not in expected or abs(float(expected[k])-float(v))>1e-7:
                raise ValueError(f'AF12 scoring mismatch: {arm}/{split}/{k}')
    # Numerical replay of both q-propagation and addressed goal outcomes from
    # model weights, which cannot be satisfied by merely rehashing fake logits.
    numeric=0
    cp={(x['arm'],x['split'],x['case_id'],x['selected']):x for x in patch}
    torch.set_num_threads(2)
    for arm,model in models.items():
        for split,ds in datasets.items():
            for i in sorted({0,len(ds.cases)-1}):
                rx=ds.get([i]);c=ds.cases[i];row=cache[arm,split,c.id]
                with torch.no_grad():
                    t=trace_upstream(model,rx,ex.PRE[c.depth])
                    q,_=model._read(t.final.a,rx.read_slots)
                    g=latch_suffix(model,rx,t.final,3).goal
                if not all((close(saved,actual) for saved,actual in
                      [(row['closure_stage_logits'],t.logits[0].tolist()),
                       (row['prefix_q_logits'],q[0].tolist()),
                       (row['clean_goal_logits'],g[0].tolist())])):
                    raise ValueError('AF12 source-bound clean forward replay mismatch')
                numeric+=1
                for j in (0,1):
                    donor,_=ds.pair([i],[j]);selected=torch.tensor([j])
                    with torch.no_grad():pair=replay(model,rx,donor,selected,ex.PRE[c.depth],cut=True)
                    rowp=cp[arm,split,c.id,j]
                    if not close(rowp['patch_goal_logits'],pair.patched.goal[0].tolist()) or not close(rowp['patch_q_logits'],pair.patched.q[0].tolist()):
                        raise ValueError('AF12 source-bound patch replay mismatch')
                    numeric+=1
    result={'experiment':'AF-12','seed':seed,'passed':True,'clean_rows':len(clean),'patched_rows':len(patch),
            'sampled_forward_replays':numeric,'training_sha256':ex.sha(training/'train_receipt.json'),
            'evaluation_sha256':ex.sha(evaluation/'evaluation.json'),
            'fresh_manifest_sha256':score['manifest_sha256']}
    write_json=ex.write_json
    write_json(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=ex.SEEDS)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.training,a.evaluation,a.seed,a.out,smoke=a.smoke)))
if __name__=='__main__':main()
