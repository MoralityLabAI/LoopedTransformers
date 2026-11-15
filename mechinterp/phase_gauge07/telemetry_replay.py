"""Exact native LoopScope-02 output telemetry replay for COE-07.

Reads the *previously committed* 16-step correctness traces of the original
SHA-pinned pretrained model. This is a truth-conditioned output-oracle
observation. It is never represented as a fitted latent activation probe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

try:
    from .score import (VISITS, SOURCE_BUNDLE_SHA256, MODEL_SHA256, STRATA,
                        select_manifest, rotate, necklace)
except ImportError:
    from score import (VISITS, SOURCE_BUNDLE_SHA256, MODEL_SHA256, STRATA,
                       select_manifest, rotate, necklace)

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'mechinterp/loopscope02/results/benchmark_bundle.json'
TRACE=ROOT/'mechinterp/loopscope02/results/trm_results.json'
BUNDLE_BLOB_SHA='01ec63b4b2a4ae5399f783c6036a7fc018825a87'
TRACE_BLOB_SHA='3c35daeb71da32f5c9ebbf03da539b8ee718b0ee'


def git_blob_digest(path:Path) -> str:
    data=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()


def verify_source(bundle:dict,trace:dict):
    if bundle.get('bundle_sha256')!=SOURCE_BUNDLE_SHA256 or trace.get('bundle_sha256')!=SOURCE_BUNDLE_SHA256:
        raise ValueError('source benchmark SHA mismatch')
    if trace.get('checkpoint_sha256')!=MODEL_SHA256 or trace.get('steps')!=16:
        raise ValueError('source checkpoint/visit mismatch')
    outcomes=trace.get('items_target_correct_by_step')
    if (len(outcomes)!=16 or any(len(row)!=len(bundle['items']) for row in outcomes)
        or any(a not in (0,1) for row in outcomes for a in row)):
        raise ValueError('corrupt output truth-conditioned traces')
    return True


def trace_word(data:dict,idx:int,width:int) -> int:
    if not 1<=width<=16:raise ValueError('wrong width')
    return sum(int(data['items_target_correct_by_step'][t][idx])<<t for t in range(width))


def pair_stats(words:list[int],labels:list[int],width:int)->dict:
    if len(words)!=len(labels):raise ValueError('shapes differ')
    totals=dict(eligible=0,ordered=0,cyclic=0,count=0,final=0)
    for i,j in combinations(range(len(labels)),2):
        if labels[i]==labels[j]:continue
        a,b=words[i],words[j]
        totals['eligible']+=1
        totals['ordered']+=a!=b
        totals['cyclic']+=necklace(a,width)!=necklace(b,width)
        totals['count']+=a.bit_count()!=b.bit_count()
        totals['final']+=((a>>(width-1))&1)!=((b>>(width-1))&1)
    totals['order_with_unknown_clock_loss']=totals['ordered']-totals['cyclic']
    totals['cyclic_beyond_count']=totals['cyclic']-totals['count']
    if not (totals['ordered']>=totals['cyclic']>=totals['count']):
        raise AssertionError('quotient nesting violated')
    return totals


def analyze(bundle:dict,trace:dict) -> dict:
    verify_source(bundle,trace)
    manifest=select_manifest(bundle)
    per_stratum={}; first_nine={}
    for T in (4,8,16):
        per_stratum[str(T)]={};first_nine[str(T)]={}
        for stratum in STRATA:
            idx=[i for i,r in enumerate(bundle['items']) if r['stratum']==stratum]
            labels=[int(bundle['items'][i]['target_digit']) for i in idx]
            words=[trace_word(trace,i,T) for i in idx]
            score=pair_stats(words,labels,T)
            score['code_hist']={str(w):n for w,n in sorted(Counter(words).items())}
            per_stratum[str(T)][stratum]=score
            chosen=set(r['item_id'] for r in manifest['strata'][stratum])
            chosen_idx=[i for i in idx if bundle['items'][i]['item_id'] in chosen]
            first_nine[str(T)][stratum]={
                **pair_stats([trace_word(trace,i,T) for i in chosen_idx],
                             [int(bundle['items'][i]['target_digit']) for i in chosen_idx],T),
                'items':[bundle['items'][i]['item_id'] for i in chosen_idx],
                'words':[trace_word(trace,i,T) for i in chosen_idx]}
    return dict(schema='COE-07-native-output-oracle-replay-v1',
                classification='pretrained checkpoint *truth-conditioned output telemetry*, not activation probe',
                checkpoint_sha256=trace['checkpoint_sha256'],
                source_bundle_sha256=bundle['bundle_sha256'],
                source_github_blobs={'benchmark':BUNDLE_BLOB_SHA,'trm_results':TRACE_BLOB_SHA},
                group_manifest_sha256=manifest['manifest_sha256'],
                windows=per_stratum,first_nine=first_nine)


def load_and_verify() -> dict:
    for path,digest in ((BUNDLE,BUNDLE_BLOB_SHA),(TRACE,TRACE_BLOB_SHA)):
        if git_blob_digest(path)!=digest:
            raise RuntimeError(f'Git blob SHA mismatch in {path.name}')
    return analyze(json.loads(BUNDLE.read_text()),json.loads(TRACE.read_text()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    result=load_and_verify()
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.out:a.out.write_text(text,encoding='utf8')
    summary={T:{s:{k:result['windows'][T][s][k] for k in ('eligible','ordered','cyclic','count','final','cyclic_beyond_count','order_with_unknown_clock_loss')}
                for s in STRATA} for T in ('4','8','16')}
    print(json.dumps({'model_sha256':MODEL_SHA256,'summary':summary},sort_keys=True))
if __name__=='__main__':main()
