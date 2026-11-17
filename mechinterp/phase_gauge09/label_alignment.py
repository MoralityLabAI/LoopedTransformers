"""COE-09: held-out *label alignment* for frozen one-bit temporal probes.

Pair separation is invariant to class renaming on a K_q evaluator; it cannot
establish digit accuracy. For each fixed readout, fit labeled prototypes from
training strata and evaluate independently labeled held-out strata. A shuffle
of FIT label-to-code association is the conditional null. All probe weights,
thresholds and codewords must be frozen before this module sees test labels.

Python >=3.10 + numpy. No checkpoint downloads and no training performed here.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from random import Random
from typing import Mapping

import numpy as np

MODES = ('ordered', 'cyclic', 'dihedral', 'count', 'final')


def rot(word: int, shift: int, width: int) -> int:
    if width < 1 or not 0 <= word < (1 << width):
        raise ValueError('invalid cyclic code')
    shift %= width
    return ((word << shift) | (word >> (width-shift))) & ((1 << width)-1)


def rev(word: int, width: int) -> int:
    if width < 1 or not 0 <= word < 1 << width:
        raise ValueError('invalid code')
    return sum(((word >> i) & 1) << (width-1-i) for i in range(width))


def distance(a: int, b: int, mode: str, width: int) -> int:
    if mode not in MODES or width < 1 or not (0 <= a < 1 << width and 0 <= b < 1 << width):
        raise ValueError('invalid distance request')
    if mode == 'ordered': return (a^b).bit_count()
    if mode == 'final': return ((a>>(width-1))^(b>>(width-1)))&1
    if mode == 'count': return abs(a.bit_count()-b.bit_count())
    if mode == 'cyclic': return min((a^rot(b,t,width)).bit_count() for t in range(width))
    return min(min((a^rot(c,t,width)).bit_count() for t in range(width)) for c in (b,rev(b,width)))


def validate_codes(fit_codes: np.ndarray, test_codes: np.ndarray, n_labels: int | None = None) -> int:
    """Both arrays [groups,labels], with integer words (not float projections)."""
    if fit_codes.ndim != 2 or test_codes.ndim != 2 or fit_codes.shape[0] < 1 or test_codes.shape[0] < 1:
        raise ValueError('code grid must contain fit and test groups')
    if fit_codes.shape[1] < 2 or fit_codes.shape[1] != test_codes.shape[1]:
        raise ValueError('mismatched label dimension')
    if n_labels is not None and fit_codes.shape[1] != n_labels:
        raise ValueError('changed class count')
    for arr in (fit_codes,test_codes):
        if arr.dtype.kind not in 'iu' or np.any(arr < 0):
            raise ValueError('require nonnegative integer code words')
    return fit_codes.shape[1]


def projected_words(states: np.ndarray, direction: np.ndarray, threshold: float) -> np.ndarray:
    """Activation array [groups,labels,visits,features]; readout is fixed."""
    if states.ndim != 4 or states.shape[2] not in range(1,33):
        raise ValueError('invalid state tensor')
    vec=np.asarray(direction,dtype=np.float64)
    if vec.ndim!=1 or vec.shape[0]!=states.shape[3] or not np.isfinite(vec).all() or not np.isfinite(threshold):
        raise ValueError('readout mismatch')
    if not np.isfinite(states).all():raise ValueError('nonfinite state')
    bits=np.einsum('gltp,p->glt',states.astype(np.float64),vec)>threshold
    return np.sum(bits.astype(np.int64)*(1<<np.arange(states.shape[2],dtype=np.int64)),axis=2)


def nearest_label_costs(fit_codes: np.ndarray, test_codes: np.ndarray, mode: str, width: int) -> list[list[int]]:
    """One cost row per held-out (stratum,label), one column per fit label.

    Each label has all FIT-stratum prototypes. Distances are minimum over
    training prototypes, so the fit label association is the only supervision.
    """
    n=validate_codes(fit_codes,test_codes)
    if any(int(w)>=1<<width for arr in (fit_codes,test_codes) for w in arr.flat):
        raise ValueError('code does not fit declared width')
    return [[min(distance(int(test_codes[g,l]),int(fit_codes[f,label]),mode,width)
                 for f in range(fit_codes.shape[0]))
             for label in range(n)]
            for g in range(test_codes.shape[0]) for l in range(n)]


def predict_from_costs(costs:list[list[int]], permutation: tuple[int,...]) -> list[int]:
    """Permutation maps each original training label (0-based) to decoded label.

    Explicit deterministic tie: smallest permuted label among minimal costs.
    """
    n=len(permutation)
    if sorted(permutation)!=list(range(n)):
        raise ValueError('labels must be a permutation')
    if any(len(row)!=n for row in costs):
        raise ValueError('invalid cost matrix')
    return [min(range(n),key=lambda j:(row[j],permutation[j])) for row in costs]


def score_permutation(costs:list[list[int]], permutation: tuple[int,...]) -> tuple[int,list[int]]:
    pred_idx=predict_from_costs(costs,permutation)
    n=len(permutation)
    decoded=[permutation[j] for j in pred_idx]
    right=sum(decoded[i]==i%n for i in range(len(decoded)))
    return right,decoded


def conditional_shuffle_null(costs:list[list[int]], *, permutations:int=3000, seed:int=20261115) -> dict:
    if permutations<1:raise ValueError('require positive null samples')
    n=len(costs[0]); rng=Random(seed)
    base=tuple(range(n)); observed,pred=score_permutation(costs,base)
    scores=[]
    for _ in range(permutations):
        labels=list(base);rng.shuffle(labels)
        scores.append(score_permutation(costs,tuple(labels))[0])
    return {'observed_correct':observed,'total':len(costs),
            'null_mean':sum(scores)/permutations,
            'null_q025':sorted(scores)[max(0,int(.025*permutations)-1)],
            'null_q975':sorted(scores)[max(0,int(.975*permutations)-1)],
            'upper_tail_p_plus_one':(1+sum(s>=observed for s in scores))/(permutations+1),
            'permutations':permutations,'seed':seed,'predictions':pred,
            'note':'conditional permutation of fit label assignment, heldout labels fixed; scores are board-level'}


def score_fit_heldout(fit_codes:np.ndarray,test_codes:np.ndarray,*,width:int,
                      repetitions:int=3000,seed:int=20261115)->dict:
    n=validate_codes(fit_codes,test_codes)
    if n != 9:
        # The codebook accepts synthetic q other than 9, but the native plan is q=9.
        pass
    result={'n_labels':n,'fit_groups':fit_codes.shape[0],'test_groups':test_codes.shape[0],
            'width':width,'modes':{}}
    for m in MODES:
        costs=nearest_label_costs(fit_codes,test_codes,m,width)
        s=conditional_shuffle_null(costs,permutations=repetitions,seed=seed)
        # One count per held-out group, exposing any stratum-specific collapse.
        s['correct_per_group']=[sum(s['predictions'][g*n+i]==i for i in range(n))
                                for g in range(test_codes.shape[0])]
        s['unique_test_codes_per_group']=[len({int(x) for x in test_codes[g]}) for g in range(test_codes.shape[0])]
        result['modes'][m]=s
    return result


def separation_invariant_under_label_permutation(words:list[int], permutation:list[int]) -> bool:
    """A K_q evaluator has all pairs: code separation ignores label names."""
    n=len(words)
    if sorted(permutation)!=list(range(n)):
        raise ValueError('invalid permutation')
    base=sum(words[a]!=words[b] for a,b in combinations(range(n),2))
    changed=sum(words[permutation[a]]!=words[permutation[b]] for a,b in combinations(range(n),2))
    return base==changed
