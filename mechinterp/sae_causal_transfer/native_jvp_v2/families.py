"""LS-SAE04 source-fit, pilot and unseen full-test noncanonical Sudoku splits."""
from __future__ import annotations
from mechinterp.sae_causal_transfer.native_trm_v1 import families as OLD
from mechinterp.sae_causal_transfer.native_jvp_v1.families import median_blank

BASES={
  "fit_30":202624000000,
  "pilot_30":202624020000,
  "pilot_34":202624021000,
  "test_30":202624030000,
  "test_34":202624040000
}
SIZES={"fit_30":32,"pilot_30":2,"pilot_34":2,"test_30":12,"test_34":12}
HOLES={key:34 if key.endswith("_34") else 30 for key in BASES}


def roster(name):
    if name not in BASES:
        raise ValueError("unknown LS-SAE04 split")
    output=[]
    for i in range(SIZES[name]):
        seed=BASES[name]+i
        t=OLD.full_solution(seed)
        x=OLD.dig_unique(t,seed+1000000,HOLES[name])
        if (not OLD.valid_solution(t) or OLD.rectangles(t)<=0
            or OLD.count_solutions(x)!=1 or x.count(0)!=HOLES[name]):
            raise ValueError("invalid registered Sudoku solution/puzzle")
        output.append({"seed":seed,"x":x,"target":t,
                       "holes":HOLES[name],"rectangles":OLD.rectangles(t)})
    return output


def verify_disjoint():
    old,_=OLD.build_rosters(pilot=False)
    old_solutions={tuple(r["target"]) for rows in old.values() for r in rows}
    old_puzzles={tuple(r["x"]) for rows in old.values() for r in rows}
    seen_solutions=set();seen_puzzles=set()
    manifests={}
    for name in BASES:
        records=roster(name)
        for x in records:
            target=tuple(x["target"]);puzzle=tuple(x["x"])
            if (target in seen_solutions or puzzle in seen_puzzles
                or target in old_solutions or puzzle in old_puzzles):
                raise ValueError("LS-SAE04 shares Sudoku with LS-SAE02 or own other splits")
            seen_solutions.add(target);seen_puzzles.add(puzzle)
        manifests[name]=OLD.fingerprint(records)
    return manifests
