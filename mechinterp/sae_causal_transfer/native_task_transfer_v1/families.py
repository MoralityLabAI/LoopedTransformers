"""LS-SAE06 fresh noncanonical unique Sudoku families; no neural model imports."""
from __future__ import annotations

from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as S3
from mechinterp.sae_causal_transfer.native_jvp_v2 import families as S4
from mechinterp.sae_causal_transfer.native_singular_v1 import families as S5

SEEDS={"pilot_30":202626020000,"pilot_34":202626021000,
       "test_30":202626030000,"test_34":202626040000}
COUNT={key:(2 if key.startswith("pilot") else 8) for key in SEEDS}
HOLES={key:34 if key.endswith("_34") else 30 for key in SEEDS}


def roster(family):
    if family not in SEEDS:
        raise ValueError("only original registered LS-SAE06 split allowed")
    out=[]
    for i in range(COUNT[family]):
        seed=SEEDS[family]+i
        target=S2.full_solution(seed)
        x=S2.dig_unique(target,seed+1000000,HOLES[family])
        if not S2.valid_solution(target) or S2.rectangles(target)<=0 or S2.count_solutions(x)!=1:
            raise ValueError("new Sudoku is canonical, invalid or lacks exact uniqueness")
        out.append({"seed":seed,"x":x,"target":target,
                    "holes":HOLES[family],"rectangles":S2.rectangles(target)})
    return out


def goal_from_visible_mask(x):
    missing=[i for i,v in enumerate(x) if v==0]
    if len(missing) not in (30,34):
        raise ValueError("requires exactly registered 30 or 34 blanks")
    return missing[len(missing)//2]


def manifest():
    old2,_=S2.build_rosters(pilot=False)
    old3,_=S3.build_new(pilot=False)
    old=[*old2.values(),*old3.values()]
    old.extend(S4.roster(name) for name in S4.BASES)
    old.extend(S5.roster(name) for name in S5.SEEDS)
    complete={tuple(r["target"]) for family in old for r in family}
    inputs={tuple(r["x"]) for family in old for r in family}
    result={}
    for family in SEEDS:
        rows=roster(family)
        for r in rows:
            a,b=tuple(r["target"]),tuple(r["x"])
            if a in complete or b in inputs:
                raise ValueError("LS-SAE06 source solution overlaps LS-SAE02..05 or same campaign")
            complete.add(a);inputs.add(b)
        result[family]=S2.fingerprint(rows)
    return result
