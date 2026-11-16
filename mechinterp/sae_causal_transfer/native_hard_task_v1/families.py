"""LS-SAE07 source-registered exact-unique 52/54-hole noncanonical Sudoku.

Uses original source MRV full solutions, exact two-solution cutoff, and same
permutation digger for harder numbers of holes. No model access here.
"""
from __future__ import annotations
import random

from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as S3
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as S4
from mechinterp.sae_causal_transfer.native_singular_v1 import families as S5
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as S6
from mechinterp.loopscope02.scripts import sudoku_prop as PROP

BASES={
 "pilot_52":202627020000,"pilot_54":202627021000,
 "test_52":202627030000,"test_54":202627040000
}
SIZES={name:(2 if name.startswith("pilot") else 8) for name in BASES}
HOLES={name:54 if name.endswith("_54") else 52 for name in BASES}


def dig_hard(solution,seed,holes):
    if holes not in (52,54) or not S2.valid_solution(solution):
        raise ValueError("LS-SAE07 registered unique hard-Sudoku dig only")
    rng=random.Random(seed)
    x=list(solution);n=0
    for i in rng.sample(range(81),81):
        if n==holes:break
        original=x[i];x[i]=0
        if S2.count_solutions(x,2)==1:n+=1
        else:x[i]=original
    if n!=holes or S2.count_solutions(x,2)!=1:
        raise ValueError(f"registered hard source admission failed for seed={seed}, holes={holes}, reached={n}")
    return x


def roster(family):
    if family not in BASES:
        raise ValueError("unknown registered LS-SAE07 source family")
    result=[]
    for i in range(SIZES[family]):
        seed=BASES[family]+i
        target=S2.full_solution(seed)
        grid=dig_hard(target,seed+1000000,HOLES[family])
        if S2.rectangles(target)<=0 or not S2.valid_solution(target):
            raise AssertionError("canonical or incorrect source solution")
        visible_only=PROP.trace_propagation("".join("." if x==0 else str(x) for x in grid))
        progress=sum(1 for i in range(81) if grid[i]==0 and visible_only.cell_round[i]>=1)
        first_round=sum(bool(x) for x in visible_only.naked_round1)
        result.append({
          "seed":seed,"x":grid,"target":target,
          "holes":HOLES[family],"rectangles":S2.rectangles(target),
          "visible_structural_difficulty":{
             "propagation_solved":bool(visible_only.solved),
             "propagation_rounds":int(visible_only.num_rounds),
             "deduced_missing_cells":progress,
             "first_round_naked_singles":first_round,
             "stalled_without_progress":bool(visible_only.num_rounds==0 and not visible_only.solved)
          }
        })
    return result


def median_blank(x):
    blank=[i for i,v in enumerate(x) if v==0]
    if len(blank) not in (52,54):
        raise ValueError("registered hard Sudoku 52/54 blanks required")
    return blank[len(blank)//2]


def full_manifest():
    older2,_=S2.build_rosters(pilot=False)
    older3,_=S3.build_new(pilot=False)
    known=[*older2.values(),*older3.values()]
    known.extend(S4.roster(k) for k in S4.BASES)
    known.extend(S5.roster(k) for k in S5.SEEDS)
    known.extend(S6.roster(k) for k in S6.SEEDS)
    targets={tuple(row["target"]) for group in known for row in group}
    inputs={tuple(row["x"]) for group in known for row in group}
    results={}
    for family in BASES:
        records=roster(family)
        for row in records:
            a,b=tuple(row["target"]),tuple(row["x"])
            if a in targets or b in inputs:
                raise ValueError("registered LS-SAE07 source solution/input duplicated across campaigns")
            targets.add(a);inputs.add(b)
        results[family]=S2.fingerprint(records)
    return results
