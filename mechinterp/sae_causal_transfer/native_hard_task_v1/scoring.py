"""LS-SAE07 fixed 52/54-hole ACT6 blank flips, ACT12 recovery and bootstrap.

Uses exact original LS-SAE06 per-cell Sudoku scorer with stronger physical doses.
"""
from __future__ import annotations
import random

from mechinterp.sae_causal_transfer.native_task_transfer_v1 import scoring as SCORE

RADII=(0.20,2.00,8.00)
HORIZONS=(3,6,12)
ARMS=SCORE.ARMS
BOOTSTRAPS=2000


def family_score(rows,family):
    if family not in ("test_52","test_54") or len(rows)==0:
        raise ValueError("registered hard Sudoku families only")
    ds=[];correct_pairs=[];events=0
    clean_inexact=0;power_best_sign_improved=0
    for row in rows:
        at6=row["task_score"]["ACT6"]
        power=at6["state_power"]["8.00"]
        random_arm=at6["random"]["8.00"]
        if random_arm is None:
            raise ValueError("equal-dose random source control missing")
        original=power["clean"] if power is not None else random_arm["clean"]
        if original["blank_correct"]<original["blank_total"]:
            clean_inexact+=1
            if power is not None and any(
                signed["blank_correct"]>original["blank_correct"]
                and signed["clue_disagreements"]==0
                for signed in (power["plus"],power["minus"])
            ):
                power_best_sign_improved+=1
        fp=power["symmetric_blank_prediction_flip_score"] if power is not None else 0.
        fr=random_arm["symmetric_blank_prediction_flip_score"]
        effect=float(fp-fr)
        ds.append(effect);events+=int(effect>=1.0)
        cp=power["symmetric_blank_correctness_gain"] if power is not None else 0.
        correct_pairs.append(float(cp-random_arm["symmetric_blank_correctness_gain"]))
    rng=random.Random(202627130+len(family))
    draws=[]
    for _ in range(BOOTSTRAPS):
        ids=[rng.randrange(len(rows)) for _ in rows]
        draws.append(sum(ds[i] for i in ids)/len(ds))
    draws.sort()
    ci=[draws[50],draws[1949]]
    return {
        "n_puzzles":len(rows),
        "power_minus_random_flip_pairs":ds,
        "power_flip_advantage_events_at_least_one":events,
        "mean_paired_blank_prediction_flip_effect":sum(ds)/len(ds),
        "puzzle_bootstrap95":ci,
        "bootstrap_draws":len(draws),
        "mean_paired_blank_correctness_gain":sum(correct_pairs)/len(correct_pairs),
        "clean_inexact_puzzles_ACT6":clean_inexact,
        "power_best_sign_clue_consistent_improvement_on_inexact":power_best_sign_improved,
        "registered_success":bool(len(rows)==8 and events>=6 and ci[0]>0)
    }
