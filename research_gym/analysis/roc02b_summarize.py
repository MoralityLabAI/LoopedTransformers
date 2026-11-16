"""Public, standard-library audit and aggregation for ROC-02B calibration-only receipts."""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEEDS=(2026111601,2026111603,2026111607,2026111609,2026111613,2026111619)
RANKS=(1,2,3,4,7,8,12,16,32)
DEPTHS=(8,16)

def load(src):
    cells=[];observed=[];nulls=[];basenorms=[];artifacts=[]
    for depth in DEPTHS:
        for seed in SEEDS:
            path=src/f'roc02b_R{depth}_S{seed}.json.gz'
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path,'rt',encoding='utf8') as f:r=json.load(f)
            if (r['seed'],r['depth'])!=(seed,depth) or not r['calibration_only'] or r['shadow_indices']!=[0,1]:raise ValueError('metadata')
            if (len(r['cells']),len(r['observed_states']),len(r['nulls']))!=(18,36,2304):raise ValueError('card count')
            cells.extend(r['cells']);observed.extend(r['observed_states']);nulls.extend(r['nulls']);basenorms.extend(r['base_norms'])
            artifacts.append({'depth':depth,'seed':seed,'checkpoint_sha256':r['checkpoint_sha256'],
                              'file':path.name,'receipt_sha256':h})
    return cells,observed,nulls,basenorms,artifacts

def audit(cells,observed,nulls,basenorms):
    assert len(cells)==216 and len(observed)==432 and len(nulls)==27648 and len(basenorms)==216
    null_by=defaultdict(list)
    for r in nulls:null_by[(r['depth'],r['seed'],r['state_index'],r['rank'])].append(r)
    assert len(null_by)==216
    for k,group in null_by.items():
        assert len(group)==128 and len({n['null_seed'] for n in group})==128
        assert len({n['subspace_sha256'] for n in group})==128
        assert all(n['D_frob']>0 and 0<=n['s']<=1 for n in group)
    tagged=defaultdict(int)
    for x in observed:
        assert (x['depth'],x['seed'],x['state_index'],x['rank']) in null_by
        assert 0<=x['s']<=1
        assert abs(x['B_per_sqrtk']-x['B']/math.sqrt(x['rank']))<1e-9
        assert all(p is None or 1/129.-1e-12<=p<=1 for p in x['p'])
        assert all(q is None or 0<=q<=1 for q in x['q'])
        for tag in x['tags']:
            tagged[tag]+=1
            assert tag in ('closed_quotient','invariant_subspace','latched_memory','irrelevant','output_register','unclassified')
    for c in cells:
        assert c['calibration_mse']>=0 and 0<=c['calibration_accuracy']<=1
    return dict(tag_counts=dict(tagged),null_groups=len(null_by),all_128_distinct=True)

def aggregate(cells,observed,artifacts):
    groups=[]
    for depth in DEPTHS:
        for arm in ('supervised','pca'):
            for rank in RANKS:
                g=[x for x in cells if x['depth']==depth and x['rank']==rank and x['arm']==arm]
                assert len(g)==6
                b=[x for x in observed if x['depth']==depth and x['rank']==rank and x['arm']==arm]
                def avg(name):return mean(float(x[name]) for x in g)
                def avg_b(name):return mean(float(x[name]) for x in b)
                groups.append({'depth':depth,'arm':arm,'rank':rank,'seeds':len(g),'n_states':len(b),
                               'mean_calibration_mse':avg('calibration_mse'),
                               'mean_calibration_accuracy':avg('calibration_accuracy'),
                               'mean_original_network_accuracy':avg('original_network_accuracy'),
                               'mean_B_per_sqrtk':avg('B_per_sqrtk'),
                               'mean_B_to_rank_null':avg('B_to_null'),
                               'mean_causal_future_logit_capture':avg('H_capture'),
                               'mean_outgoing_delta_logits':avg_b('observed_delta_logits'),
                               'mean_incoming_delta_logits':avg_b('dark_delta_logits'),
                               'mean_isotropic_delta_logits':avg_b('isotropic_delta_logits'),
                               'mean_tangent_residual_observed':avg_b('observed_tangent_residual_logits'),
                               'mean_tangent_residual_dark':avg_b('dark_tangent_residual_logits'),
                               'significant_a_lower':sum(x['q'][0] is not None and x['q'][0]<=.05 for x in b),
                               'significant_b_lower':sum(x['q'][1] is not None and x['q'][1]<=.05 for x in b),
                               'significant_b_upper':sum(x['q'][3] is not None and x['q'][3]<=.05 for x in b),
                               'significant_s_upper':sum(x['q'][4] is not None and x['q'][4]<=.05 for x in b)})
    # Three objective Pareto: rank, MSE and rank-relative B. Source-only dev, no test selection.
    pareto=[]
    for depth in DEPTHS:
        for seed in SEEDS:
            c=[x for x in cells if x['depth']==depth and x['seed']==seed]
            for x in c:
                xyz=(x['rank'],x['calibration_mse'],x['B_to_null'])
                dominates=lambda y: (y['rank']<=xyz[0] and y['calibration_mse']<=xyz[1] and y['B_to_null']<=xyz[2]
                          and (y['rank'],y['calibration_mse'],y['B_to_null'])!=xyz)
                pareto.append({**x,'pareto':not any(dominates(y) for y in c if y is not x)})
    return groups,pareto

def write_csv(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]))
        w.writeheader();w.writerows(items)

def run(src,out):
    cells,obs,nulls,bases,artifacts=load(src)
    checks=audit(cells,obs,nulls,bases)
    groups,pareto=aggregate(cells,obs,artifacts)
    out.mkdir(exist_ok=True,parents=True)
    write_csv(out/'roc02b_rank_by_seed.csv',cells)
    write_csv(out/'roc02b_observer_states.csv',obs)
    write_csv(out/'roc02b_nulls.csv',nulls)
    write_csv(out/'roc02b_rank_summary.csv',groups)
    write_csv(out/'roc02b_3d_pareto.csv',pareto)
    with (out/'roc02b_nulls.csv.gz').open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,compresslevel=9,mtime=0) as gz:
            with io.TextIOWrapper(gz,encoding='utf8',newline='') as txt:
                w=csv.DictWriter(txt,fieldnames=list(nulls[0]));w.writeheader();w.writerows(nulls)
    (out/'roc02b_nulls.csv').unlink()
    summary={'study':'ROC-02B-calibration-only-full-observer-card-v1',
             'stage':'calibration; NEVER sealed test','seeds':list(SEEDS),'depths':list(DEPTHS),'rank_grid':list(RANKS),
             'n_models':12,'n_cards':len(cells),'n_observed_state_rows':len(obs),
             'n_null_rows':len(nulls),'n_null_groups':checks['null_groups'],
             'all_128_distinct':checks['all_128_distinct'],'tag_counts':checks['tag_counts'],
             'basis_types':['supervised','pca'],'no_equal_budget_80_step_optimizer':True,
             'no_test_claim':True,'artifact_hashes':artifacts,
             'rank8':{str(depth):{arm:next(x for x in groups if x['depth']==depth and x['arm']==arm and x['rank']==8)
                                  for arm in ['supervised','pca']} for depth in DEPTHS}}
    (out/'roc02b_summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();v=run(a.input_dir,a.output_dir)
    print(json.dumps({k:v[k] for k in ('n_models','n_cards','n_null_rows','rank8','tag_counts')},sort_keys=True))