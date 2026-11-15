"""Standalone standard-library audit of frozen PEF-11 JSON receipts.

Does not use PyTorch, execute neural networks, or trust summary 'decision'.
This checks evidence shape, the registered decision from raw score rows,
selected geometry, and the original frozen-ledger SHA256 reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

TANGENTS=('pair01','pair23','pca2','random')
ANGLES=(0,.25,1,4,16)
CHECKPOINTS={103,131,167}
TASKS={'coalition','threshold'}
NOISE=(0,.001,.003,.01,.03,.1)
QUANT=(0,.003,.01,.03,.1)
PEF10_HASH='3ef6ce605540a5009f6896cf02a938c200bde23c7a5104d378aace7388430177'

def same_sha(content):
    return hashlib.sha256(content).hexdigest()


def scores(cond,kind,site,level,readout):
    rows=[s for s in cond['scores'] if s['kind']==kind and s['site']==site and
          s['relative_level']==level]
    if not rows:
        raise AssertionError(f'missing {kind}/{site}/{level}')
    nums=[r['scores'][readout]['correct'] for r in rows]
    if any(not 0 <= x <= 768 for x in nums):
        raise AssertionError('invalid accuracy counts')
    return sum(nums)/len(nums)


def audit(folder):
    folder=Path(folder)
    precision_raw=(folder/'frozen_precision_ledger.json').read_bytes()
    precision=json.loads(precision_raw)
    outcome=json.loads((folder/'summary.json').read_bytes())
    if outcome['frozen_precision_ledger_sha256']!=same_sha(precision_raw):
        raise AssertionError('precision ledger bytes mutated')
    if precision['pef10_angle_ledger_sha256']!=PEF10_HASH:
        raise AssertionError('incorrect parent')
    if not precision['all_conditions_and_noise_seeds_frozen_before_heldout']:
        raise AssertionError('selection not frozen')
    if (tuple(precision['tangents'])!=TANGENTS or
        tuple(precision['angles_deg'])!=ANGLES or
        tuple(precision['noise_relative_levels'])!=NOISE or
        tuple(precision['quant_relative_levels'])!=QUANT):
        raise AssertionError('registration changed')
    if (len(precision['model_selections'])!=6 or len(outcome['models'])!=6 or
        {(m['seed'],m['task']) for m in outcome['models']}!=
        {(s,t) for s in CHECKPOINTS for t in TASKS}):
        raise AssertionError('missing model runs')
    model_decisions=[]
    total_conditions=total_rows=0
    for m in outcome['models']:
        meta=next(x for x in precision['model_selections'] if
                  (x['seed'],x['task'])==(m['seed'],m['task']))
        if meta['checkpoint_sha256']!=m['checkpoint_sha256']:
            raise AssertionError('checkpoint digest differs')
        if len(m['conditions'])!=20 or len(meta['conditions'])!=20:
            raise AssertionError('missing angle grid')
        selection={(c['tangent'],c['angle_deg']):c for c in meta['conditions']}
        scored={(c['tangent'],c['angle_deg']):c for c in m['conditions']}
        expected={(t,a) for t in TANGENTS for a in ANGLES}
        if set(selection)!=expected or set(scored)!=expected:
            raise AssertionError('incomplete angle/tangent grid')
        for key in expected:
            md=selection[key]
            cond=scored[key]
            v,w,o=(md[k] for k in ('direction','retained_direction','offplane_direction'))
            for vec in (v,w,o):
                if abs(sum(x*x for x in vec)-1)>1e-8:
                    raise AssertionError('nonunit direction')
            if abs(sum(x*y for x,y in zip(v,w)))>1e-8:
                raise AssertionError('not orthogonal projector')
            if len(cond['scores'])!=63:
                raise AssertionError('expected 63 conditions per angle')
            # 2 identities, 5 noise levels * 3 replicas * 3 sites, 4 quant levels x 2 sites x 2 modes.
            kinds={(r['kind'],r['site'],r['relative_level'],r['replicate']) for r in cond['scores']}
            if len(kinds)!=63:
                raise AssertionError('duplicated condition identity')
            for site in ('ablated','natural'):
                s=scores(cond,'baseline',site,0,'native_h12')
                if site=='natural' and s!=768:
                    raise AssertionError('natural classifier has changed')
            for r in cond['scores']:
                for record in r['scores'].values():
                    if sum(record['errors_by_focal_mask']) != 768-record['correct']:
                        raise AssertionError('focal-mask counts do not sum')
            total_rows+=len(cond['scores'])
            total_conditions+=1
        if m['task']=='coalition':
            small=scored[('random',.25)]
            large=scored[('random',4)]
            target='frozen_pef10_ridge_h12'
            clean=scores(small,'baseline','ablated',0,target)
            n_small=scores(small,'gaussian','transverse',.01,target)
            control=scores(small,'gaussian','natural_transverse',.01,'native_h12')
            q_small=scores(small,'transverse_quant','ablated',.03,target)
            n_large=scores(large,'gaussian','transverse',.01,target)
            q_large=scores(large,'transverse_quant','ablated',.03,target)
            frag=clean>=720 and n_small<=clean-96 and control>=720 and q_small<=clean-96
            robust=n_large>=720 and q_large>=720
            off=scores(small,'gaussian','offplane',.01,target)
            model_decisions.append({'seed':m['seed'],'fragility':frag,'robustness':robust,
                                    'both_met':frag and robust,'transverse_more_damaging':off>n_small,
                                    'clean':clean,'noise_0_01':n_small,'quant_0_03':q_small})
    success=sum(v['both_met'] for v in model_decisions)
    if outcome['decision']['coalition_both_seed_count']!=success:
        raise AssertionError('stored decision differs from independent recomputation')
    return {'models':6,'conditions':total_conditions,'score_rows':total_rows,
            'model_decisions':model_decisions,'coalition_met':success,
            'decision':'criterion_met' if success>=2 else 'criterion_not_met'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True,type=Path)
    x=parser.parse_args()
    print(json.dumps(audit(x.results),indent=2))

if __name__=='__main__':
    main()
