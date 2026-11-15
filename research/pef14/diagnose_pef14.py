"""PEF-14 posthoc same-input Jacobian audit, excluded from registered decision."""
import argparse
import json
import math
from pathlib import Path
import torch
from research.pef14 import pef14 as p


def run(root):
    root=Path(root)
    source,manifest=p.split_contexts()
    training,ledger=p.validate_frozen(root,source,manifest)
    result=json.loads((root/'summary.json').read_text())
    scores=[]
    torch.set_num_threads(1)
    for rec in training['checkpoints']:
        model=p.checked_model(root,rec)
        seed,task=rec['seed'],rec['task']
        row=next(x for x in ledger['models'] if (x['seed'],x['task'])==(seed,task))
        scored=next(x for x in result['models'] if (x['seed'],x['task'])==(seed,task))
        bits=source['heldout'];truth=p.class_truth(bits,task)
        h8=p.at_eight(model,bits)
        mu=torch.tensor(row['center'],dtype=torch.float64)
        for cond in row['conditions']:
            v=torch.tensor(cond['projector_v'],dtype=torch.float64)
            w=torch.tensor(cond['retained_w'],dtype=torch.float64)
            fit=tuple(torch.tensor(cond['fit'][k],dtype=torch.float64) for k in ('mean','std','beta'))
            z=p.project(h8,mu,v)
            f,j=p.probe_margin_j(model,bits,z,fit,w)
            q=(z.double()-mu)@w
            for forecast in cond['forecasts']:
                rel=forecast['relative_step']; step=rel*row['pc_scale']
                d=p.quant_error(q,step)
                lin_correct=int((((f+j*d)>=0).long()==truth).sum())
                actual=next(x for x in scored['conditions'] if x['tangent']==cond['tangent'] and
                    x['angle_deg']==cond['angle_deg'] and x['relative_step']==rel)
                test_lin=lin_correct/768
                scores.append({'seed':seed,'task':task,'tangent':cond['tangent'],
                   'angle_deg':cond['angle_deg'],'relative_step':rel,
                   'linear_same_heldout_correct':lin_correct,
                   'observed_nonlinear_correct':actual['correct'],
                   'same_input_nonlinearity_abs_error':abs(test_lin-actual['accuracy']),
                   'probe_mix_to_test_linear_abs_error':abs(forecast['distribution_mixture']-test_lin),
                   'probe_exactphase_to_test_linear_abs_error':abs(forecast['exact_probe_phase']-test_lin),
                   'probe_uniform_to_test_linear_abs_error':abs(forecast['uniform_phase']-test_lin),
                   'mean_abs_linear_margin_perturbation':float((j*d).abs().mean()),
                   'heldout_phase_mean':float(d.mean()),
                   'probe_phase_mean':forecast['probe_phase_mean']})
        print('PEF14_POSTHOC',seed,task,flush=True)
    stats={}
    for task in p.TASKS:
        selected=[r for r in scores if r['task']==task]
        stats[task]={'conditions':len(selected)}
        for key in ('same_input_nonlinearity_abs_error','probe_mix_to_test_linear_abs_error',
                    'probe_exactphase_to_test_linear_abs_error','probe_uniform_to_test_linear_abs_error'):
            stats[task][key]=sum(v[key] for v in selected)/len(selected)
    output={'campaign':'PEF-14','analysis':'posthoc_only',
            'registered_decision_unchanged':result['primary_decision'],
            'aggregate':stats,'conditions':scores}
    (root/'posthoc_jacobian_diagnostic.json').write_bytes(p.canonical(output))
    print(json.dumps(stats,indent=2),flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    run(args.out)

if __name__=='__main__':main()
