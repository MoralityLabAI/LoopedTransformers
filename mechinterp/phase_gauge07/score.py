"""COE-07 score library for cyclically invariant native recurrent probes.

No model weights or held-out labels are used to select candidate thresholds.
The only API crossing the fit/test boundary is ``score_heldout``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from typing import Any

import numpy as np

SOURCE_BUNDLE_SHA256 = "59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811"
STRATA = ("S1_naked1", "S3_prop3plus", "S2_prop2", "S4_stalled")
FIT_STRATA = STRATA[:2]
HOLDOUT_STRATA = STRATA[2:]
DIGITS = tuple(range(1, 10))
VISITS = 8
MODEL_SHA256 = "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"


def select_manifest(bundle: dict) -> dict:
    """Select first available board per target digit and stratum.

    This is label-stratified, deterministic, and evaluated on frozen benchmark
    metadata. No model trajectory, probe success, or target predictions enter.
    """
    if bundle.get("bundle_sha256") != SOURCE_BUNDLE_SHA256:
        raise ValueError("changed LoopScope-02 benchmark bundle")
    selection = {}
    for stratum in STRATA:
        rows = [row for row in bundle["items"] if row["stratum"] == stratum]
        by_digit = {}
        for row in rows:
            digit = int(row["target_digit"])
            if digit in DIGITS and digit not in by_digit:
                if len(row["question"]) != 81 or not (0 <= int(row["target_idx"]) < 81):
                    raise ValueError("invalid Sudoku benchmark row")
                if str(row["answer"])[int(row["target_idx"])] != str(digit):
                    raise ValueError("label mismatch with target cell")
                by_digit[digit] = {
                    "item_id": row["item_id"],
                    "question": row["question"],
                    "target_idx": int(row["target_idx"]),
                    "digit": digit,
                }
        if sorted(by_digit) != list(DIGITS):
            raise ValueError(f"source lacks all nine digits in {stratum}")
        selection[stratum] = [by_digit[d] for d in DIGITS]
    ids = [r["item_id"] for records in selection.values() for r in records]
    if len(set(ids)) != 36:
        raise ValueError("duplicate board across strata")
    return {"strata": selection, "fit_strata": list(FIT_STRATA),
            "heldout_strata": list(HOLDOUT_STRATA),
            "manifest_sha256": hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(",", ":")).encode()).hexdigest()}


def rotate(word: int, shift: int, width: int = VISITS) -> int:
    mask = (1 << width) - 1
    shift %= width
    return ((word << shift) | (word >> (width - shift))) & mask


def necklace(word: int, width: int = VISITS) -> int:
    return min(rotate(word,shift,width) for shift in range(width))


def code_class(word: int, kind: str, width: int = VISITS) -> int:
    if kind == "ordered": return word
    if kind == "cyclic": return necklace(word, width)
    if kind == "count": return word.bit_count()
    if kind == "final": return (word >> (width-1)) & 1
    raise ValueError(kind)


def score_codes(codes: dict[str, list[int]], width: int = VISITS) -> dict:
    """All K9 edges per stratum; exact integer numerator and denominator."""
    breakdown = {}
    for group, words in codes.items():
        if len(words) != 9 or any(w < 0 or w >= (1<<width) for w in words):
            raise ValueError("one valid word for each of 9 digits is required")
        tally = {name: 0 for name in ("ordered","cyclic","count","final")}
        for u,v in combinations(range(9),2):
            for kind in tally:
                tally[kind] += code_class(words[u],kind,width) != code_class(words[v],kind,width)
        assert 0 <= tally["count"] <= tally["cyclic"] <= tally["ordered"] <= 36
        tally.update({"order_with_unknown_clock_loss": tally["ordered"]-tally["cyclic"],
                      "recoverable_order_only":tally["cyclic"]-tally["count"],
                      "denominator":36,
                      "unique_ordered":len(set(words)),
                      "unique_cyclic":len(set(necklace(w,width) for w in words)),
                      "unique_counts":len(set(w.bit_count() for w in words))})
        breakdown[group] = tally
    pooled = {key:sum(v[key] for v in breakdown.values()) for key in next(iter(breakdown.values()))} if breakdown else {}
    return {"groups":breakdown, "pooled":pooled}


def codes_from_projection(values: np.ndarray, threshold: float) -> np.ndarray:
    """values: [groups,9,visits] => binary words [groups,9]."""
    if values.ndim != 3 or values.shape[1:] != (9,VISITS):
        raise ValueError("invalid projection grid")
    hits = values > float(threshold)
    return np.sum(hits.astype(np.int64) * (1 << np.arange(VISITS,dtype=np.int64)),axis=2)


def candidate_heads(fit: np.ndarray, seed: int = 20261115, n_random: int = 96):
    """A deterministic pool derived solely from fit states.

    ``fit``: [2,9,T,D]; signed centroid differences and seeded random directions.
    Thresholds are fit-only projected quantiles, with no test calibration.
    """
    if fit.ndim != 4 or fit.shape[:3] != (2,9,VISITS):
        raise ValueError("fit shape must [2,9,8,D]")
    width = fit.shape[-1]
    rng=np.random.default_rng(seed)
    means=fit.mean(axis=2).mean(axis=0) # [9,D], 2 strata and T visits
    directions=[]
    for a,b in combinations(range(9),2):
        directions.append(means[a]-means[b])
    directions.extend(rng.normal(size=(n_random,width)))
    arr=fit.reshape(-1,width)
    centered=arr-arr.mean(axis=0)
    # Principal directions add unsupervised fit-only candidate heads.
    _,_,vh=np.linalg.svd(centered,full_matrices=False)
    directions.extend(vh[:min(12,len(vh))])
    for i,d in enumerate(directions):
        norm=np.linalg.norm(d)
        if norm < 1e-10:continue
        direction=np.asarray(d,dtype=np.float64)/norm
        projected=arr@direction
        q=np.unique(np.quantile(projected,[.15,.25,.4,.5,.6,.75,.85]))
        for threshold in q:
            yield (i, direction, float(threshold))


def select_heads(fit: np.ndarray, seed:int=20261115) -> dict[str,dict]:
    """Select three independent objectives by fit-only K9 edge scores."""
    if not np.all(np.isfinite(fit)):
        raise ValueError("nonfinite fit states")
    best={}
    for candidate_id,vec,theta in candidate_heads(fit,seed=seed):
        codes=codes_from_projection(np.einsum('gtpd,d->gtp',fit,vec),theta)
        score=score_codes({str(g):list(codes[g]) for g in range(2)})["pooled"]
        for objective,priority in {
            "ordered":(score["ordered"],score["cyclic"],score["count"]),
            "cyclic":(score["cyclic"],score["recoverable_order_only"],score["count"]),
            "cyclic_gain":(score["recoverable_order_only"],score["cyclic"],score["ordered"]),
            "count":(score["count"],score["cyclic"],score["ordered"]),
        }.items():
            existing=best.get(objective)
            if existing is None or tuple(priority)>tuple(existing["priority"]):
                best[objective]={"direction":vec.tolist(),"threshold":theta,
                                "candidate_id":candidate_id,"priority":list(priority),
                                "fit_score":score}
    if set(best)!={"ordered","cyclic","cyclic_gain","count"}:
        raise ValueError("incomplete head search")
    return best


def evaluate_heads(fit: np.ndarray, test: np.ndarray, seed:int=20261115) -> dict:
    """An explicit fit/test firewall: fit-only selection, then test score."""
    chosen=select_heads(fit,seed=seed)
    results={}
    for name,p in chosen.items():
        vec=np.asarray(p["direction"])
        proj=np.einsum('gtpd,d->gtp',test,vec)
        codes=codes_from_projection(proj,p["threshold"])
        results[name]={"fit_priority":p["priority"],"candidate_id":p["candidate_id"],
                       "threshold":p["threshold"],"direction":p["direction"],
                       "fit_score":p["fit_score"],
                       "heldout_scores":score_codes({str(g):list(codes[g]) for g in range(test.shape[0])}),
                       "heldout_codes":codes.tolist()}
    return results


def audit_phases_equal(weighted: list[int]) -> bool:
    return all(necklace(rotate(word,j))==necklace(word) for word in weighted for j in range(VISITS))
