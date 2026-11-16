"""LS-SAE08 model-blind deterministic first-admissible 52/54-hole Sudoku.

Each source slot searches fixed candidates j=0..31, seed=base+100*slot+j,
using original source MRV and exact two-solution uniqueness checking.
The first geometrically admissible unique puzzle is frozen before model access.
"""
from __future__ import annotations

from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as S3
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as S4
from mechinterp.sae_causal_transfer.native_singular_v1 import families as S5
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as S6
from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as S7

BASES={"pilot_52":202628020000,"pilot_54":202628021000,
       "test_52":202628030000,"test_54":202628040000}
SIZES={k:2 if k.startswith("pilot") else 8 for k in BASES}
HOLES={k:54 if k.endswith("_54") else 52 for k in BASES}
MAX_CANDIDATES=32


def roster(family):
    if family not in BASES:
        raise ValueError("unregistered LS-SAE08 source family")
    rows=[]
    for slot in range(SIZES[family]):
        failures=[]
        winner=None
        for attempt in range(MAX_CANDIDATES):
            seed=BASES[family]+100*slot+attempt
            target=S2.full_solution(seed)
            if S2.rectangles(target)<=0 or not S2.valid_solution(target):
                raise AssertionError("original noncanonical unique Sudoku source invariant failed")
            try:
                puzzle=S7.dig_hard(target,seed+1000000,HOLES[family])
            except ValueError as err:
                failures.append({"grid_seed":seed,"failure":str(err)})
                continue
            if puzzle.count(0)!=HOLES[family] or S2.count_solutions(puzzle,2)!=1:
                raise AssertionError("first candidate accepted without exact uniqueness")
            winner={
              "slot":slot,"grid_seed":seed,"seed":seed,
              "candidate_attempt_index":attempt,
              "rejected_model_blind_candidates":failures,
              "x":puzzle,"target":target,"holes":HOLES[family],
              "rectangles":S2.rectangles(target),
              "visible_structural_difficulty":S7.visible_structure(puzzle)
            }
            break
        if winner is None:
            raise ValueError(f"frozen source admission failed for {family} slot {slot}: all 32 candidates insufficient")
        rows.append(winner)
    return rows


def median_blank(x):
    return S7.median_blank(x)


def full_manifest():
    old2,_=S2.build_rosters(pilot=False)
    old3,_=S3.build_new(pilot=False)
    old=[*old2.values(),*old3.values()]
    old.extend(S4.roster(k) for k in S4.BASES)
    old.extend(S5.roster(k) for k in S5.SEEDS)
    old.extend(S6.roster(k) for k in S6.SEEDS)

    complete={tuple(row["target"]) for rows in old for row in rows}
    puzzle_inputs={tuple(row["x"]) for rows in old for row in rows}

    # Every fixed LS-SAE07 grid seed is considered reserved even if its 54-hole
    # dig later failed, so the new cohort remains independent of that campaign.
    for family,base in S7.BASES.items():
        for i in range(S7.SIZES[family]):
            seed=base+i
            target=S2.full_solution(seed)
            complete.add(tuple(target))
            try:puzzle=S7.dig_hard(target,seed+1000000,S7.HOLES[family])
            except ValueError:continue
            puzzle_inputs.add(tuple(puzzle))

    manifest={}
    for family in BASES:
        items=roster(family)
        for item in items:
            a,b=tuple(item["target"]),tuple(item["x"])
            if a in complete or b in puzzle_inputs:
                raise ValueError("source-admitted LS-SAE08 puzzle duplicated another research cohort")
            complete.add(a);puzzle_inputs.add(b)
        manifest[family]=S2.fingerprint(items)
    return manifest
