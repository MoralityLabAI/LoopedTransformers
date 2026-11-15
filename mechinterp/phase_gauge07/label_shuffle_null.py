"""COE-08: exact and permutation nulls for *truth-conditioned* loop telemetry.

Warning: correctness bits require the ground truth label and are not an
implementable unsupervised activation monitor. This module audits whether a
reported label-pair distinction exceeds what the trace histogram alone buys.

Python 3.10+, standard library. Inputs are pinned LoopScope-02 JSON receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from random import Random

try:
    from .dihedral_replay import descriptor
    from .telemetry_replay import (BUNDLE, TRACE, BUNDLE_BLOB_SHA, TRACE_BLOB_SHA,
                                   git_blob_digest, verify_source, trace_word)
except ImportError:
    from dihedral_replay import descriptor
    from telemetry_replay import (BUNDLE, TRACE, BUNDLE_BLOB_SHA, TRACE_BLOB_SHA,
                                  git_blob_digest, verify_source, trace_word)


def eligible_count(labels: list[int]) -> int:
    """All unordered pairs of points with distinct evaluator labels."""
    if not labels:
        raise ValueError("empty label list")
    n = len(labels)
    hist = Counter(labels)
    return (n*n - sum(v*v for v in hist.values()))//2


def class_histogram(words: list[int], width: int, mode: str) -> Counter:
    if not words or width < 1:
        raise ValueError("empty or invalid words")
    return Counter(descriptor(w, width, mode) for w in words)


def expected_distinctions_under_shuffle(words: list[int], labels: list[int],
                                         width: int, mode: str) -> Fraction:
    """Exact expectation conditional on BOTH multisets under uniform random relabeling.

    Permute whole trace histories among fixed label positions. Every distinct
    eligible endpoint pair receives two distinct uniformly sampled traces.
    """
    if len(words) != len(labels) or len(words) < 2:
        raise ValueError("require >=2 equally many words and labels")
    n = len(words)
    h = class_histogram(words,width,mode)
    e = eligible_count(labels)
    ordered_distinct = n*n - sum(x*x for x in h.values())
    return Fraction(e*ordered_distinct,n*(n-1))


def expected_refinement_gain(words: list[int], labels: list[int], width: int,
                             fine: str = "dihedral", coarse: str = "count") -> Fraction:
    """Exact finite permutation-null expectation of fine-minus-coarse hits."""
    if len(words) != len(labels) or len(words) < 2:
        raise ValueError("require >=2 equally many words and labels")
    f = [descriptor(x,width,fine) for x in words]
    c = [descriptor(x,width,coarse) for x in words]
    if len({(v,w) for v,w in zip(f,c)}) != len(set(f)):
        raise ValueError("fine descriptor does not determine coarse descriptor")
    hf, hc = Counter(f), Counter(c)
    e = eligible_count(labels)
    return Fraction(e*(sum(x*x for x in hc.values())-
                       sum(x*x for x in hf.values())),len(words)*(len(words)-1))


def observed_refinement_gain(words: list[int], labels: list[int], width: int,
                             fine: str = "dihedral", coarse: str = "count") -> int:
    if len(words)!=len(labels) or not words:
        raise ValueError("invalid inputs")
    f=[descriptor(x,width,fine) for x in words]
    c=[descriptor(x,width,coarse) for x in words]
    if len(set(zip(f,c)))!=len(set(f)):
        raise ValueError("not a nested quotient")
    return sum(labels[i]!=labels[j] and c[i]==c[j] and f[i]!=f[j]
               for i in range(len(words)) for j in range(i+1,len(words)))



def exact_refinement_null_variance(words: list[int], labels: list[int], width: int,
                                   fine: str = "dihedral", coarse: str = "count") -> dict:
    """Exact first two moments for a fixed-label-count shuffle on an edge graph.

    Let H join trace indices when they have equal coarse code but different
    fine code. Under a uniform permutation of the observed label multiset,
    D counts H edges crossing labels. The result includes every dependency
    from edge pairs sharing zero or one endpoint, without assuming iid pairs.
    """
    if len(words) != len(labels) or len(words) < 4:
        raise ValueError("second moment formula requires at least four states")
    n = len(words)
    f = [descriptor(w,width,fine) for w in words]
    c = [descriptor(w,width,coarse) for w in words]
    if len(set(zip(f,c)))!=len(set(f)):
        raise ValueError("not a nested quotient")
    edges=[(i,j) for i in range(n) for j in range(i+1,n)
           if c[i]==c[j] and f[i]!=f[j]]
    degrees=[0]*n
    for i,j in edges:
        degrees[i]+=1;degrees[j]+=1
    e=len(edges)
    adjacent=sum(d*(d-1)//2 for d in degrees)
    disjoint=e*(e-1)//2-adjacent
    m=list(Counter(labels).values())
    def fall(x:int,k:int)->int:
        v=1
        for j in range(k):
            if x <= j: return 0
            v*=x-j
        return v
    same2=sum(fall(x,2) for x in m)
    p_diff=1-Fraction(same2,fall(n,2))
    p_adj=Fraction(sum(x*fall(n-x,2) for x in m),fall(n,3))
    p_double_same=(sum(fall(x,4) for x in m)+
                   sum(fall(x,2)*fall(y,2) for i,x in enumerate(m)
                       for j,y in enumerate(m) if i!=j))
    p_disjoint=1-2*Fraction(same2,fall(n,2))+Fraction(p_double_same,fall(n,4))
    mean=e*p_diff
    second=e*p_diff+2*adjacent*p_adj+2*disjoint*p_disjoint
    variance=second-mean*mean
    assert variance>=0
    return {'edges':e,'adjacent_edge_pairs':adjacent,'disjoint_edge_pairs':disjoint,
            'mean':mean,'variance':variance}

def permutation_null(words: list[int], labels: list[int], width: int,
                     *, repetitions: int = 10000, seed: int = 20261115) -> dict:
    """Fixed-multiset label permutation, with deterministic +1 tail probabilities."""
    if repetitions < 1:
        raise ValueError("require positive permutations")
    if len(words)!=len(labels) or len(words)<2:
        raise ValueError("invalid shapes")
    fine=[descriptor(x,width,'dihedral') for x in words]
    count=[x.bit_count() for x in words]
    if len(set(zip(fine,count)))!=len(set(fine)):
        raise ValueError("not a nested quotient")
    indices=[(i,j) for i in range(len(words)) for j in range(i+1,len(words))
             if count[i]==count[j] and fine[i]!=fine[j]]
    rng=Random(seed)
    observed=sum(labels[i]!=labels[j] for i,j in indices)
    expected=expected_refinement_gain(words,labels,width)
    shuffled=labels.copy()
    samples=[]
    for _ in range(repetitions):
        rng.shuffle(shuffled)
        samples.append(sum(shuffled[i]!=shuffled[j] for i,j in indices))
    # Exact shuffled-label histogram is held fixed at every draw.
    samples.sort()
    # Prechosen summary uses inclusive empirical order statistics.
    q025=samples[max(0,int(0.025*repetitions)-1)]
    q975=samples[min(repetitions-1,int(0.975*repetitions)-1)]
    return {
       'width':width,'n':len(words),'eligible':eligible_count(labels),
       'observed_extra':observed,'exact_null_expectation':str(expected),
       'exact_null_float':float(expected),
       'shuffle_mean':sum(samples)/repetitions,
       'shuffle_q025':q025,'shuffle_median':samples[repetitions//2],
       'shuffle_q975':q975,
       'upper_tail_p_plus_one':(1+sum(x>=observed for x in samples))/(repetitions+1),
       'lower_tail_p_plus_one':(1+sum(x<=observed for x in samples))/(repetitions+1),
       'repetitions':repetitions,'seed':seed,
       'classification':'posthoc negative control on truth-conditioned outputs, not a latent-probe efficacy p-value'
    }


def analyze(bundle:dict, trace:dict, *, repetitions:int=10000, seed:int=20261115)->dict:
    # Reuse source provenance and checkpoint/dataset validators.
    if trace.get('checkpoint_sha256')!='932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3':
        raise ValueError('checkpoint pin mismatch')
    verify_source(bundle,trace)
    if trace.get('bundle_sha256') != bundle.get('bundle_sha256'):
        raise ValueError('bundle pin mismatch')
    indices=[i for i,v in enumerate(bundle['items']) if v['stratum']=='S4_stalled']
    if len(indices)!=64:
        raise ValueError('expected exactly 64 stalled boards')
    labels=[int(bundle['items'][i]['target_digit']) for i in indices]
    rows=[]
    for width in (4,8,16):
        words=[trace_word(trace,i,width) for i in indices]
        row=permutation_null(words,labels,width,repetitions=repetitions,seed=seed+width)
        row['observed_count_separations'] = sum(labels[i]!=labels[j] and
             words[i].bit_count()!=words[j].bit_count()
             for i in range(len(words)) for j in range(i+1,len(words)))
        row['observed_dihedral_separations']=row['observed_count_separations']+row['observed_extra']
        rows.append(row)
    return {'schema':'COE-08-null-v1','source':'LoopScope-02 pretrained S4 truth-conditioned correctness telemetry',
            'checkpoint_sha256':trace['checkpoint_sha256'],'bundle_sha256':bundle['bundle_sha256'],
            'label_histogram':dict(sorted(Counter(labels).items())),
            'windows':rows}


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=BUNDLE)
    parser.add_argument('--trace',type=Path,default=TRACE)
    parser.add_argument('--repetitions',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261115)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if git_blob_digest(args.bundle)!=BUNDLE_BLOB_SHA or git_blob_digest(args.trace)!=TRACE_BLOB_SHA:
        raise ValueError("pinned original Git blob SHA mismatch")
    bundle=json.loads(args.bundle.read_text(encoding="utf8"))
    trace=json.loads(args.trace.read_text(encoding="utf8"))
    result=analyze(bundle,trace,repetitions=args.repetitions,seed=args.seed)
    content=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.out:args.out.write_text(content,encoding='utf8')
    print(content)
    return 0

if __name__=='__main__':raise SystemExit(main())
