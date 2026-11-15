"""COE-09 read-only label-alignment audit of COE-07 native frozen activations.

This script *requires* COE-07's SHA-verified full-pretrained checkpoint receipt.
It never fits heads on test states or treats truth-conditioned telemetry as a
probe. The sole supervision is fit-stratum target digits in the frozen COE-07
protocol. No model inference is performed by this replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .label_alignment import projected_words,score_fit_heldout
except ImportError:
    from label_alignment import projected_words,score_fit_heldout

SOURCE_SCHEMA = 'COE-07-native-pretrained-TRM-phase-gauge-v1'
SOURCE_MODEL_SHA = '932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3'
FIT_STRATA=('S1_naked1','S3_prop3plus')
TEST_STRATA=('S2_prop2','S4_stalled')
REPS=('visible','dark','full','scratch')
HEADS=('cyclic_gain','cyclic','count','ordered')


def load_native(path:Path,receipt:dict)->dict[str,np.ndarray]:
    if receipt.get('schema')!=SOURCE_SCHEMA or receipt.get('source_checkpoint_sha256')!=SOURCE_MODEL_SHA:
        raise ValueError('wrong pretrained source model')
    if receipt.get('fit_strata')!=list(FIT_STRATA) or receipt.get('heldout_strata')!=list(TEST_STRATA):
        raise ValueError('strata were changed')
    if receipt.get('visits')!=8 or receipt.get('num_digits')!=9 or receipt.get('pairs_per_stratum')!=36:
        raise ValueError('source experiment changed')
    with np.load(path,allow_pickle=False) as loaded:
        arrays={rep:np.asarray(loaded[rep]) for rep in REPS}
    for rep,arr in arrays.items():
        if arr.shape[:3]!=(4,9,8) or (rep=='visible' and arr.shape[3]!=11) or (rep!='visible' and arr.shape[3]!=512):
            raise ValueError('wrong native activation shape')
        if arr.dtype not in (np.float32,np.float64) or not np.isfinite(arr).all():
            raise ValueError('invalid native activation value')
        digest=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if digest!=receipt['metadata']['activation_sha256'][rep]:
            raise ValueError('native activation digest mismatch')
    return arrays


def replay(arrays:dict[str,np.ndarray],receipt:dict,*,permutations:int=2000)->dict:
    if permutations<1:raise ValueError('require permutation samples')
    output={}
    for idx,rep in enumerate(REPS):
        selected=receipt['representations'][rep]
        output[rep]={}
        for j,head_name in enumerate(HEADS):
            h=selected[head_name]
            word=projected_words(arrays[rep],np.asarray(h['direction']),h['threshold'])
            if word[2:].tolist()!=h['heldout_codes']:
                raise ValueError(f'changed COE-07 selected readout or heldout codes: {rep}/{head_name}')
            audit=score_fit_heldout(word[:2],word[2:],width=8,
                                    repetitions=permutations,seed=20261115+29*idx+j)
            output[rep][head_name]=audit
    return {'schema':'COE-09-native-label-alignment-v1',
            'source_schema':SOURCE_SCHEMA,'model_sha256':SOURCE_MODEL_SHA,
            'manifest_sha256':receipt['source_benchmark_sha256'],
            'source_label_order':'digits 1..9 in each stratum; frozen COE-07 manifest',
            'fit_strata':list(FIT_STRATA),'test_strata':list(TEST_STRATA),
            'heads':'unaltered COE-07 fit-selected heads',
            'label_shuffle':'consistent training label reassignment, heldout assignment fixed',
            'denominator_per_head':18,'results':output}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--activations',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--permutations',type=int,default=2000)
    opt=p.parse_args()
    receipt=json.loads(opt.receipt.read_text(encoding='utf8'))
    arrays=load_native(opt.activations,receipt)
    result=replay(arrays,receipt,permutations=opt.permutations)
    content=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if opt.out:opt.out.write_text(content,encoding='utf8')
    print(json.dumps({'schema':result['schema'],
        'observed_correct':{rep:{head:{mode:d['observed_correct'] for mode,d in data['modes'].items()}
                      for head,data in heads.items()}for rep,heads in result['results'].items()}},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
