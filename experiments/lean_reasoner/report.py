"""Condense full local receipts; retain seed counts and per-theorem outcome bits."""
import gzip
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT=Path(__file__).parent


def group(r):
    if r['condition']=='baseline': return r['tag']
    trained=r['tag'].split('_')[0]
    return f'{trained}_u{r["steps"]}_{r["condition"]}'


def summarize():
    source=ROOT/'results'/'evaluation.json'
    raw=json.loads(source.read_text()); groups={}
    for r in raw: groups.setdefault(group(r),[]).append(r)
    summary={'experiment':'LR-01', 'full_evaluation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'parameters_per_model':20161, 'groups':{}}
    for key, rows in groups.items():
        g={'seeds':[r['seed'] for r in rows]}
        for split in ['test','depth_ood']:
            rates=[r[split]['rate'] for r in rows]
            g[split]={'counts':[r[split]['solved'] for r in rows], 'n_per_seed':192,
                      'mean':statistics.mean(rates), 'seed_sd':statistics.stdev(rates),
                      'outcome_bits':[''.join('1' if z['solved'] else '0' for z in r[split]['rows']) for r in rows]}
        summary['groups'][key]=g
    # Conditional-on-seeds paired theorem bootstrap, with identical theorem draws.
    contrasts={}
    for split in ['test','depth_ood']:
        left=sorted(groups['l4_u300_trained_depth'],key=lambda r:r['seed'])
        right=sorted(groups['l1_u1200_trained_depth'],key=lambda r:r['seed'])
        delta=[statistics.mean(int(a[split]['rows'][i]['solved'])-int(b[split]['rows'][i]['solved'])
                               for a,b in zip(left,right)) for i in range(192)]
        rng=random.Random(704); draws=sorted(statistics.mean(rng.choices(delta,k=192)) for _ in range(5000))
        contrasts[split]={'mean_difference':statistics.mean(delta),
                          'paired_theorem_bootstrap_95':[draws[124],draws[4874]],
                          'interpretation':'conditional on the three trained seeds; not seed-population uncertainty'}
    summary['four_loop_vs_long_one_loop']=contrasts
    training=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('l*_s*_u*.json'))]
    summary['training']=training
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Complete rollout receipts for prespecified seed 17, four-loop primary arm.
    r=next(r for r in raw if r['tag']=='l4_s17_u300' and r['condition']=='trained_depth')
    receipts={'tag':r['tag'],'condition':r['condition'],'selection':'all 384 evaluation tasks, first prespecified seed',
              'rows':[[row['id'],int(row['solved']),row['trace']] for s in ['test','depth_ood'] for row in r[s]['rows']]}
    receipt_bytes=(json.dumps(receipts,separators=(',',':'))+'\n').encode()
    (ROOT/'results'/'prediction_receipts.json').write_bytes(receipt_bytes)
    (ROOT/'results'/'prediction_receipts.json.gz').write_bytes(gzip.compress(receipt_bytes,mtime=0))
    print(json.dumps({k:{s:[v[s]['counts'],v[s]['mean']] for s in ['test','depth_ood']}
                      for k,v in summary['groups'].items()},indent=2))
    return summary


if __name__=='__main__': summarize()
