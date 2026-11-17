"""LS-SAE03 locked fresh Sudoku families; the original LS-SAE02 fit is reused."""
from __future__ import annotations

from mechinterp.sae_causal_transfer.native_trm_v1 import families as previous

NEW_BASES={"test_30":202622030000,"test_34":202622040000}
NEW_COUNTS={"test_30":12,"test_34":12}
NEW_HOLES={"test_30":30,"test_34":34}


def median_blank(puzzle: list[int]) -> int:
    blank=[i for i,v in enumerate(puzzle) if v==0]
    if len(blank) not in (30,34):
        raise ValueError("expected registered 30 or 34 blank Sudoku")
    return blank[len(blank)//2]


def build_new(*,pilot:bool=False):
    """Generate entirely new source-disjoint full-grid and visible-input rosters.

    Independent original LS-SAE02 80-puzzle roster is constructed for
    source-split collision checks. New data is never used for PCA/SAE fits.
    """
    original,_=previous.build_rosters(pilot=False)
    known_complete={tuple(v["target"]) for rows in original.values() for v in rows}
    known_input={tuple(v["x"]) for rows in original.values() for v in rows}
    out={}
    for family,base in NEW_BASES.items():
        rows=[]
        for j in range(2 if pilot else NEW_COUNTS[family]):
            seed=base+j
            t=previous.full_solution(seed)
            x=previous.dig_unique(t,seed+1000000,NEW_HOLES[family])
            if (tuple(t) in known_complete or tuple(x) in known_input
                or previous.rectangles(t)<=0 or previous.count_solutions(x)!=1):
                raise ValueError("registered LS-SAE03 Sudoku is invalid or overlaps prior fit/test")
            known_complete.add(tuple(t));known_input.add(tuple(x))
            rows.append({"seed":seed,"x":x,"target":t,
                         "holes":NEW_HOLES[family],
                         "rectangles":previous.rectangles(t)})
        out[family]=rows
    return out,{fam:previous.fingerprint(v) for fam,v in out.items()}
