"""LS-SAE06 model-independent 81-cell Sudoku scoring and fixed paired bootstrap.

All discrete correctness and Sudoku-validity calculations are exact integer
checks on the argmax digits produced by the frozen native checkpoint.
"""
from __future__ import annotations
import math
import random

import numpy as np

RADII=(0.05,0.20)
HORIZONS=(3,6)
ARMS=("state_power","logit_power","sae","random")
BOOTSTRAPS=2000


def center(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.shape[-1]!=9:
        raise ValueError("must have nine Sudoku digit logits")
    return x-x.mean(axis=-1,keepdims=True)


def valid_sudoku(grid,givens):
    if len(grid)!=81 or len(givens)!=81 or any(not 1<=v<=9 for v in grid):
        return False
    if any(givens[i] and grid[i]!=givens[i] for i in range(81)):
        return False
    need=set(range(1,10))
    for j in range(9):
        if set(grid[j*9:j*9+9])!=need or {grid[9*i+j] for i in range(9)}!=need:
            return False
    for br in (0,3,6):
        for bc in (0,3,6):
            if {grid[9*(br+dr)+bc+dc] for dr in range(3)
                for dc in range(3)}!=need:
                return False
    return True


def unit_conflicts(grid):
    if len(grid)!=81 or any(not 1<=v<=9 for v in grid):
        raise ValueError("requires 81 predicted digits")
    groups=[]
    for r in range(9):
        groups.append(grid[9*r:9*r+9])
        groups.append([grid[9*c+r] for c in range(9)])
    for br in (0,3,6):
        for bc in (0,3,6):
            groups.append([grid[9*(br+dr)+bc+dc] for dr in range(3) for dc in range(3)])
    return sum(len(x)-len(set(x)) for x in groups)


def digit_metrics(logits,puzzle,target,goal_position,clean_prediction):
    arr=np.asarray(logits,dtype=np.float64)
    if arr.shape!=(81,9) or not np.all(np.isfinite(arr)):
        raise ValueError("source task head must produce finite 81×9 digit logits")
    if len(puzzle)!=81 or len(target)!=81 or len(clean_prediction)!=81:
        raise ValueError("mismatched Sudoku manifest")
    preds=(arr.argmax(axis=-1)+1).tolist()
    missing=[j for j in range(81) if puzzle[j]==0]
    clues=[j for j in range(81) if puzzle[j]!=0]
    correct=sum(preds[j]==target[j] for j in missing)
    flips=[j for j in range(81) if preds[j]!=clean_prediction[j]]
    changed_blanks=sum(puzzle[j]==0 for j in flips)
    correct_to_wrong=sum(clean_prediction[j]==target[j] and preds[j]!=target[j] for j in missing)
    wrong_to_correct=sum(clean_prediction[j]!=target[j] and preds[j]==target[j] for j in missing)
    return {
      "predicted_digits_81":preds,
      "blank_correct":int(correct),
      "blank_total":len(missing),
      "board_exact":bool(preds==target),
      "sudoku_valid_and_clue_consistent":bool(valid_sudoku(preds,puzzle)),
      "unit_conflicts":int(unit_conflicts(preds)),
      "clue_disagreements":sum(preds[j]!=puzzle[j] for j in clues),
      "all_81_prediction_changes":len(flips),
      "blank_prediction_changes":int(changed_blanks),
      "goal_prediction_changed":bool(preds[goal_position]!=clean_prediction[goal_position]),
      "other_cell_prediction_changes":sum(j!=goal_position for j in flips),
      "blank_correct_to_wrong":int(correct_to_wrong),
      "blank_wrong_to_correct":int(wrong_to_correct)
    }


def margin_receipt(clean_logits,changed_logits,target,puzzle):
    """Freeze each blank's strongest CLEAN nontruth rival for this horizon."""
    clean=np.asarray(clean_logits,dtype=np.float64)
    edited=np.asarray(changed_logits,dtype=np.float64)
    if clean.shape!=(81,9) or edited.shape!=(81,9):
        raise ValueError("missing complete 81×9 source logits")
    idx=[j for j,x in enumerate(puzzle) if x==0]
    rival=[]
    original=[]
    new=[]
    for i in idx:
        y=target[i]-1
        ls=clean[i].copy();ls[y]=-np.inf
        r=int(np.argmax(ls))
        rival.append(r)
        original.append(float(clean[i,y]-clean[i,r]))
        new.append(float(edited[i,y]-edited[i,r]))
    return {"blank_indices":idx,"clean_rival_zero_based":rival,
            "clean_margins":original,"edited_margins":new,
            "mean_abs_margin_change":float(np.mean(np.abs(np.subtract(new,original))))}


def signed_pair_effect(clean_logits,plus_logits,minus_logits,puzzle,target,goal):
    clean=np.asarray(clean_logits,dtype=np.float64)
    plus=np.asarray(plus_logits,dtype=np.float64)
    minus=np.asarray(minus_logits,dtype=np.float64)
    if any(x.shape!=(81,9) for x in (clean,plus,minus)):
        raise ValueError("incomplete 81×9 logits for signed effects")
    original=(clean.argmax(axis=-1)+1).tolist()
    p=digit_metrics(plus,puzzle,target,goal,original)
    m=digit_metrics(minus,puzzle,target,goal,original)
    c=digit_metrics(clean,puzzle,target,goal,original)
    dplus=center(plus-clean);dminus=center(minus-clean)
    C=(float(np.linalg.norm(dplus))+float(np.linalg.norm(dminus)))/2
    symmetric=float(np.linalg.norm(center(plus-minus)))/2
    return {
       "clean":c,"plus":p,"minus":m,
       "mean_centered_logit_effect_unscaled":C,
       "centered_logit_sym_secant_unscaled":symmetric,
       "symmetric_blank_prediction_flip_score":(
            p["blank_prediction_changes"]+m["blank_prediction_changes"])/2,
       "symmetric_blank_correctness_gain":(
            p["blank_correct"]+m["blank_correct"])/2-c["blank_correct"],
       "plus_margins":margin_receipt(clean,plus,target,puzzle),
       "minus_margins":margin_receipt(clean,minus,target,puzzle)
    }


def score_families(rows,family):
    if family not in ("test_30","test_34") or not rows:
        raise ValueError("registered nonempty fresh Sudoku family required")
    diffs=[]
    correct=[]
    events=0
    for row in rows:
        a=row["task_score"]["ACT6"]["state_power"]["0.20"]
        b=row["task_score"]["ACT6"]["random"]["0.20"]
        power=a["symmetric_blank_prediction_flip_score"] if a is not None else 0.0
        random_score=b["symmetric_blank_prediction_flip_score"]
        d=power-random_score
        diffs.append(float(d))
        correct.append((a["symmetric_blank_correctness_gain"] if a is not None else 0.0)-
                       b["symmetric_blank_correctness_gain"])
        events+=int(d>=1.0)
    rng=random.Random(202626130+len(family))
    boot=[]
    for _ in range(BOOTSTRAPS):
        idx=[rng.randrange(len(diffs)) for _ in diffs]
        boot.append(sum(diffs[i] for i in idx)/len(idx))
    boot.sort()
    return {
     "n_puzzles":len(rows),"power_flip_events_at_least_one":events,
     "paired_flip_differences":diffs,
     "mean_power_minus_random_flip_score":sum(diffs)/len(diffs),
     "bootstrap95":[boot[50],boot[1949]],
     "bootstrap_draws":len(boot),
     "mean_power_minus_random_blank_correctness_gain":sum(correct)/len(correct),
     "registered_success":bool(len(rows)==8 and events>=6 and boot[50]>0)
    }
