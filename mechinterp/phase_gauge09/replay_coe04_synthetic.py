"""Read-only COE-04 synthetic-frozen control for COE-09 label alignment.

This is a control on three *locally trained toy tied-attention* checkpoints,
not a native pretrained TRM result. Consumes existing COE-04 frozen artifacts.
No retraining, no head search, and no heldout threshold calibration.
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

EXPECTED_SCHEMA = 'COE-04-native-tied-loop-frozen-v1'
EXPECTED_RECEIPT = 'COE-04-temporal-quantization-receipt-v1'
FIT=(0,1,2,3)
HOLDOUT=(4,5,6,7)
N=8;T=4;D=24


def audit_item(source:dict,receipt:dict):
    if source.get('schema')!=EXPECTED_SCHEMA or receipt.get('schema')!=EXPECTED_RECEIPT:
        raise ValueError('unexpected source schema')
    if source['probe_fit_contexts']!=list(FIT) or source['probe_heldout_contexts']!=list(HOLDOUT):
        raise ValueError('changed probe split')
    if receipt['split']!={'fit':list(FIT),'holdout':list(HOLDOUT)}:raise ValueError('changed selection receipt')
    a={int(x['model_seed']):x for x in source['items']}
    b={int(x['seed']):x for x in receipt['rows']}
    if sorted(a)!=[2406,2407,2408] or sorted(b)!=sorted(a):raise ValueError('seed mismatch')
    for seed,item in a.items():
        if b[seed]['checkpoint_sha256']!=item['model_checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
        if item['train_steps']!=160 or len(item['model_predictions'])!=64:
            raise ValueError('unexpected base model')
        if any(item['model_predictions'][8*d+c]!=d for d in range(N) for c in range(N)):
            raise ValueError('base task failed')
        if len(item['activations'])!=64:raise ValueError('activation shape')
        head=b[seed]['heads']['temporal_shared_1bit_x4']
        if len(head['weight'])!=D:raise ValueError('head shape')
        data=np.array([[item['activations'][f'{d}:{c}'] for d in range(N)] for c in range(8)],dtype=float)
        if data.shape!=(8,N,T,D) or not np.isfinite(data).all():raise ValueError('activation values')
        words=np.sum((np.einsum('cdtp,p->cdt',data,np.asarray(head['weight'],dtype=float))>head['threshold']).astype(np.int64)*(1<<np.arange(T,dtype=np.int64)),axis=2)
        calc=sum(int(words[c,d]!=words[c,e]) for c in HOLDOUT for d in range(N) for e in range(d+1,N))
        if calc!=head['heldout_edge_hits']:raise ValueError('source score mismatch')
        yield seed,words,item['model_checkpoint_sha256']


def replay(activations:Path,receipt:Path,*,permutations:int=3000)->dict:
    with gzip.open(activations,'rt',encoding='utf8') as f:doc=json.load(f)
    info=json.loads(receipt.read_text(encoding='utf8'))
    rows=[]
    for seed,words,checkpoint_sha in audit_item(doc,info):
        scores=score_fit_heldout(words[list(FIT)],words[list(HOLDOUT)],width=T,
                                 repetitions=permutations,seed=seed+20261115)
        rows.append({'seed':seed,'checkpoint_sha256':checkpoint_sha,
                     'label_alignment':scores})
    return {'schema':'COE-09-synthetic-label-alignment-v1',
            'source_name':'COE-04 synthetic frozen 24D tied-attention checkpoints',
            'explicitly_not_pretrained':True,'original_head_selection':'COE-04 fit-only maximal pair separation',
            'source_activation_file_sha256':hashlib.sha256(activations.read_bytes()).hexdigest(),
            'source_head_receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
            'fit_contexts':list(FIT),'heldout_contexts':list(HOLDOUT),
            'evaluation':'nearest two fit-stratum code prototypes per label, min distance',
            'results':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--activations',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path);p.add_argument('--permutations',type=int,default=3000)
    p.add_argument('--out',type=Path)
    opt=p.parse_args();res=replay(opt.activations,opt.receipt,permutations=opt.permutations)
    msg=json.dumps(res,indent=2,sort_keys=True)+'\n'
    if opt.out:opt.out.write_text(msg,encoding='utf8')
    print(json.dumps({'schema':res['schema'],'results':[
      {'seed':r['seed'],
       'accuracy':{k:v['observed_correct'] for k,v in r['label_alignment']['modes'].items()},
       'shuffle_p':{k:v['upper_tail_p_plus_one'] for k,v in r['label_alignment']['modes'].items()}}
       for r in res['results']]},indent=2))

if __name__=='__main__':main()
