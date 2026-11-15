"""AF-11 independent semantics, saved-logit recount, and native replay audit.

Checks every row, source/train hashes, tensor exports and per-case Boolean
influence, before any scientific aggregate verdict is accepted. A successful
program exit certifies integrity; it does not certify the scientific hypothesis.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from experiments.activation_funnel_af05 import gym
from . import data, experiment as ex
from .core import replay


def exact_boolean(case):
    x=case.facts
    q=[]
    for i,k in enumerate(case.kinds):
        a,b=x[2*i:2*i+2]
        q.append(int(bool(a and b)) if k=='and' else int(bool(a or b)))
    goal=int(bool(q[0] and q[1])) if case.gate=='and' else int(bool(q[0] or q[1]))
    return tuple(q)+(goal,)


def equal_logits(a,b,tol=2e-5):
    if len(a)!=len(b):return False
    return all(abs(float(x)-float(y))<=tol and math.isfinite(float(x)) for x,y in zip(a,b))


def audit(training,evaluation,seed,out,replay_count=2,smoke=False):
    training,evaluation,out=Path(training),Path(evaluation),Path(out)
    rec,models=ex.load_models(training,seed,smoke)
    s=json.loads((evaluation/'evaluation.json').read_text())
    if s['experiment']!='AF-11' or (s['seed'],s['smoke'])!=(seed,smoke):
        raise ValueError('AF-11 evaluation identity changed')
    if s['fresh_manifest_sha256']!=ex.verify_fresh_manifest() or s['source_train_receipt_sha256']!=ex.sha_file(training/'train_receipt.json'):
        raise ValueError('AF-11 source or corpus mismatch')
    if ex.sha_file(evaluation/'counterfactuals.jsonl')!=s['counterfactual_sha256']:
        raise ValueError('AF11 altered counterfactual logit rows')
    entries=[json.loads(x) for x in (evaluation/'counterfactuals.jsonl').read_text().splitlines()]
    allowed_splits=('dev',) if smoke else ex.FAMILIES
    lookup={(a,f):data.Dataset(f) for a in ex.ARMS for f in allowed_splits}
    case_index={(a,f,c.id):(i,c) for (a,f),ds in lookup.items()
                                     for i,c in enumerate(ds.cases)}
    expected=sum(len(ds.cases)*2 for ds in (lookup[ex.ARMS[0],f] for f in allowed_splits))*len(ex.ARMS)
    if len(entries)!=expected or s['counterfactual_rows']!=expected:
        raise ValueError('AF11 expected prediction roster incomplete')
    seen=set();calc={}
    for arm in ex.ARMS:
        for split in allowed_splits:
            calc[arm,split]={k:0 for k in ['patch_rows','informative','correct_flip',
                 'unaffected_n','unaffected_goal_retention','patched_goal_correct',
                 'prefix_q_exact','clean_goal_correct','all_patched_q_exact',
                 'prefix_correct_informative','prefix_correct_informative_success',
                 'false_goal','patched_other_q_correct']}
            calc[arm,split]['signed_delta_abs_error_sum']=0.
    for r in entries:
        a,f,j=r['arm'],r['split'],r['selected']
        if a not in ex.ARMS or f not in allowed_splits or j not in (0,1):
            raise ValueError('AF-11 unexpected model/split/selector')
        key=(a,f,r['case_id'],j)
        if key in seen or (a,f,r['case_id']) not in case_index:
            raise ValueError('AF11 repeated/unexpected episode identity')
        seen.add(key)
        i,c=case_index[a,f,r['case_id']]
        ds=lookup[a,f]
        donor_id,composed=ds.donors[i,j]
        if r['donor_case_id']!=ds.cases[donor_id].id:
            raise ValueError('changed matched donor ID')
        orig=exact_boolean(c);donor=exact_boolean(ds.cases[donor_id])
        original_oracle=gym.oracle(c)[0]
        if orig!=original_oracle or donor!=gym.oracle(ds.cases[donor_id])[0]:
            raise ValueError('independent Boolean-Horn oracle contradiction')
        comp=list(orig);comp[j]=donor[j]
        comp[2]=int(all(comp[:2])) if c.gate=='and' else int(any(comp[:2]))
        if (r['recipient'],r['composed'])!=(list(orig),comp) or tuple(comp)!=composed:
            raise ValueError('AF11 independent composed-target mismatch')
        other=1-j
        inf=orig[other] if c.gate=='and' else 1-orig[other]
        delta=comp[2]-orig[2]
        if r['influence']!=inf or r['goal_delta']!=delta or delta!=(comp[j]-orig[j])*inf or r['gate']!=int(c.gate=='or'):
            raise ValueError('AF11 Boolean influence identity mismatch')
        # Independent raw-logit decoding: no stored model prediction trusted.
        import math
        def soft(x):
            assert len(x)==2 and all(math.isfinite(float(v)) for v in x)
            d=float(x[0])-float(x[1])
            if d>=0:return math.exp(-d)/(1+math.exp(-d))
            return 1./(1.+math.exp(d))
        def cls(x):return int(x[1]>x[0])
        pred=cls(r['patch_goal_logits']); clean=cls(r['clean_goal_logits'])
        qprefix=[cls(x) for x in r['prefix_q_logits']]
        qp=[cls(x) for x in r['patched_q_logits']]
        clean_ok=int(clean==orig[2]);good=int(pred==comp[2]);qcorrect=int(qprefix==list(orig[:2]))
        x=calc[a,f];x['patch_rows']+=1;x['informative']+=inf
        x['correct_flip']+=inf*good;x['unaffected_n']+=1-inf;x['unaffected_goal_retention']+=(1-inf)*good
        x['patched_goal_correct']+=good;x['prefix_q_exact']+=qcorrect;x['clean_goal_correct']+=clean_ok
        x['all_patched_q_exact']+=int(qp==comp[:2]);x['prefix_correct_informative']+=qcorrect*inf
        x['prefix_correct_informative_success']+=qcorrect*inf*good
        x['false_goal']+=int(pred==1 and comp[2]==0)
        x['patched_other_q_correct']+=int(qp[other]==orig[other])
        x['signed_delta_abs_error_sum']+=abs((soft(r['patch_goal_logits'])-soft(r['clean_goal_logits']))-delta)
        if a!='influence_ordinary' and (r['protected_cut_code_difference'] is None or abs(r['protected_cut_code_difference'])>0):
            raise ValueError('AF-11 structural cut mismatch in saved codes')
    if len(seen)!=expected:raise ValueError('AF-11 missing retained counterfactual records')
    for (arm,split),x in calc.items():
        y=s['scores'][arm][split]
        for k,v in x.items():
            if abs(float(y[k])-float(v))>2e-4:raise ValueError(f'AF-11 metric tampered: {arm}/{split}/{k}')
        ratios={
         'informative_flip_accuracy':x['correct_flip']/max(1,x['informative']),
         'irrelevant_patch_goal_retention':x['unaffected_goal_retention']/max(1,x['unaffected_n']),
         'overall_patched_goal_accuracy':x['patched_goal_correct']/x['patch_rows'],
         'clean_goal_accuracy':x['clean_goal_correct']/x['patch_rows'],
         'prefix_joint_q_accuracy':x['prefix_q_exact']/x['patch_rows'],
         'prefix_conditioned_flip_accuracy':x['prefix_correct_informative_success']/max(1,x['prefix_correct_informative']),
         'patched_other_q_accuracy':x['patched_other_q_correct']/x['patch_rows'],
         'signed_delta_mae':x['signed_delta_abs_error_sum']/x['patch_rows']}
        for k,v in ratios.items():
            if abs(y[k]-v)>2e-4:raise ValueError(f'AF11 derived metric altered: {arm}/{split}/{k}')
    # Source-verified checkpoint replay with exact tensors (sampled after
    # all per-case oracle and numerical audits); sample both truth classes.
    by_key={(r['arm'],r['split'],r['case_id'],r['selected']):r for r in entries}
    replays=0
    for arm in ex.ARMS:
        model=models[arm]
        for split in allowed_splits:
            ds=lookup[arm,split]
            ix=sorted({0,len(ds.cases)-1})[:replay_count]
            for i in ix:
                selected=torch.tensor([0,1],dtype=torch.long)
                rx=ds.get([i,i]);dx,composed=ds.pair([i,i],[0,1])
                with torch.no_grad():
                    pair=replay(model,rx,dx,selected,ex.PRE[ds.cases[i].depth],cut=arm!='influence_ordinary')
                for k in (0,1):
                    row=by_key[arm,split,ds.cases[i].id,k]
                    for key,values in [('clean_goal_logits',pair.clean.goal[k].tolist()),
                         ('patch_goal_logits',pair.patched.goal[k].tolist()),
                         ('control_goal_logits',pair.control.goal[k].tolist())]:
                        if not equal_logits(row[key],values):
                            raise ValueError('AF11 independent source-checkpoint forward mismatch: '+key)
                    replays+=1
    result={'experiment':'AF-11','seed':seed,'verified':True,'model_arms':list(ex.ARMS),
            'scored_rows':len(entries),'independent_replays':replays,
            'source_receipt_sha256':ex.sha_file(training/'train_receipt.json'),
            'evaluation_sha256':ex.sha_file(evaluation/'evaluation.json'),
            'counterfactual_sha256':s['counterfactual_sha256'],
            'fresh_manifest_sha256':s['fresh_manifest_sha256']}
    ex.json_write(out,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True,choices=ex.SEEDS)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    x=p.parse_args();print(json.dumps(audit(x.training,x.evaluation,x.seed,x.out,smoke=x.smoke)))
if __name__=='__main__':main()
