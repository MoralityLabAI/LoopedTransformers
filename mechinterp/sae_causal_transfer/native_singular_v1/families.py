"""LS-SAE05 new pilot/fresh complete-grid Sudoku splits, source disjoint."""
from __future__ import annotations
from mechinterp.sae_causal_transfer.native_trm_v1 import families as BASE
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as JVP3
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as JVP4

SEEDS={"pilot_30":202625020000,"pilot_34":202625021000,
       "test_30":202625030000,"test_34":202625040000}
SIZES={k:(2 if k.startswith("pilot") else 8) for k in SEEDS}
HOLES={k:(34 if k.endswith("_34") else 30) for k in SEEDS}


def roster(family):
    if family not in SEEDS:
        raise ValueError("only registered LS-SAE05 cohort")
    out=[]
    for i in range(SIZES[family]):
        seed=SEEDS[family]+i
        target=BASE.full_solution(seed)
        inputs=BASE.dig_unique(target,seed+1000000,HOLES[family])
        if not BASE.valid_solution(target) or BASE.rectangles(target)<=0 or BASE.count_solutions(inputs)!=1:
            raise ValueError("new Sudoku invalid, canonical or not uniquely solvable")
        out.append({"seed":seed,"x":inputs,"target":target,
                    "holes":HOLES[family],"rectangles":BASE.rectangles(target)})
    return out


def median_blank(x):
    zeros=[i for i,n in enumerate(x) if n==0]
    if len(zeros) not in (30,34):
        raise ValueError("unregistered Sudoku hole count")
    return zeros[len(zeros)//2]


def manifest():
    old,_=BASE.build_rosters(pilot=False)
    other3,_=JVP3.build_new(pilot=False)
    known=list(old.values())+list(other3.values())+[JVP4.roster(k) for k in JVP4.BASES]
    complete={tuple(r["target"]) for rows in known for r in rows}
    inputs={tuple(r["x"]) for rows in known for r in rows}
    sha={}
    for family in SEEDS:
        samples=roster(family)
        for r in samples:
            a,b=tuple(r["target"]),tuple(r["x"])
            if a in complete or b in inputs:
                raise ValueError("LS-SAE05 puzzle overlaps LS-SAE02,03,04 or this cohort")
            complete.add(a);inputs.add(b)
        sha[family]=BASE.fingerprint(samples)
    return sha
