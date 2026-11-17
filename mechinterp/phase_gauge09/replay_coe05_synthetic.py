"""COE-05 separate frozen 2411..2413 synthetic cohort heldout-digit decoding.

Replays previously selected fit-only order-sensitive readouts from original
COE-05 documents. This source was part of the trained 24D synthetic study,
not a native 7M pretrained inference run.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .label_alignment import score_fit_heldout
except ImportError:
    from label_alignment import score_fit_heldout


def replay(activation_file:Path,receipt_file:Path,*,permutations:int=3000):
    with gzip.open(activation_file,'rt',encoding='utf8') as f:source=json.load(f)
    receipt=json.loads(receipt_file.read_text(encoding='utf8'))
    if source['schema']!='COE-04-native-tied-loop-frozen-v1' or receipt['schema']!='COE-05-fit-only-phase-probe-exploratory-v1':
        raise ValueError('source schema mismatch')
    if source['probe_fit_contexts']!=[0,1,2,3] or source['probe_heldout_contexts']!=[4,5,6,7]:
        raise ValueError('split changed')
    if receipt['split']!={'fit':[0,1,2,3],'heldout':[4,5,6,7]}:
        raise ValueError('receipt split changed')
    source_rows={int(x['model_seed']):x for x in source['items']}
    receipt_rows={int(x['model_seed']):x for x in receipt['records']}
    if sorted(source_rows)!=[2411,2412,2413] or sorted(receipt_rows)!=sorted(source_rows):
        raise ValueError('seeds changed')
    result=[]
    for seed,source_row in sorted(source_rows.items()):
        row=receipt_rows[seed]
        if source_row['model_checkpoint_sha256']!=row['source_checkpoint_sha256']:
            raise ValueError('changed checkpoint')
        if source_row['train_steps']!=160 or source_row['train_examples']!=64:
            raise ValueError('changed training source')
        if any(source_row['model_predictions'][8*a+b]!=a for a in range(8) for b in range(8)):
            raise ValueError('source failed admission')
        head=row['primary_stable_order_head']
        data=np.array([[source_row['activations'][f'{a}:{b}'] for a in range(8)]
                       for b in range(8)],dtype=np.float64)
        if data.shape!=(8,8,4,24) or not np.isfinite(data).all() or len(head['weight'])!=24:
            raise ValueError('invalid frozen source tensors')
        codes=np.sum((np.einsum('catd,d->cat',data,np.array(head['weight']))>head['threshold']).astype(np.int64)
                     *(1<<np.arange(4,dtype=np.int64)),axis=2)
        heldout_total=sum(codes[b,a]!=codes[b,c] for b in range(4,8) for a in range(8) for c in range(a+1,8))
        count_total=sum(codes[b,a].bit_count()!=codes[b,c].bit_count()
                        for b in range(4,8) for a in range(8) for c in range(a+1,8))
        if heldout_total!=head['heldout_total_edge_hits'] or count_total!=head['heldout_count_only_hits']:
            raise ValueError('previous selected-head score mismatch')
        scores=score_fit_heldout(codes[:4],codes[4:],width=4,repetitions=permutations,seed=20261115+seed)
        result.append({'seed':seed,'checkpoint_sha256':source_row['model_checkpoint_sha256'],
                       'original_heldout_pair_hits':int(heldout_total),
                       'original_count_only_pair_hits':int(count_total),
                       'label_alignment':scores})
    return {'schema':'COE-09-separate-synthetic-2411-2413-v1',
            'explicitly_not_pretrained':True,
            'source_activation_sha256':hashlib.sha256(activation_file.read_bytes()).hexdigest(),
            'source_receipt_sha256':hashlib.sha256(receipt_file.read_bytes()).hexdigest(),
            'fit_contexts':[0,1,2,3],'heldout_contexts':[4,5,6,7],
            'readout':'previous COE-05 primary_stable_order_head; no head refitting',
            'rows':result}


def main():
    p=argparse.ArgumentParser();p.add_argument('--activations',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--permutations',type=int,default=3000)
    p.add_argument('--out',type=Path)
    args=p.parse_args();r=replay(args.activations,args.receipt,permutations=args.permutations)
    if args.out:args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'schema':r['schema'],'accuracy':{row['seed']:{m:s['observed_correct']
          for m,s in row['label_alignment']['modes'].items()} for row in r['rows']}},indent=2))
if __name__=='__main__':main()
