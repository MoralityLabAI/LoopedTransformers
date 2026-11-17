"""LS-SAE02 source-registered noncanonical Sudoku generation.

The randomized MRV and exact uniqueness procedures are adapted from
MoralityLabAI/Compactification-Theory research/real_recurrence_sae_v1/
smc13_family.py. New seed ranges are reserved for LS-SAE02, with canonical
JSON SHA256 receipts. No PyTorch or trained model needed.
"""
import hashlib
import json
import random

BASES={
    "fit_30":202619000000,
    "map_30":202619010000,
    "test_30":202619030000,
    "test_34":202619040000,
}
COUNTS={"fit_30":32,"map_30":16,"test_30":16,"test_34":16}
HOLES={name:34 if name=="test_34" else 30 for name in BASES}
PILOT={"fit_30":4,"map_30":4,"test_30":2,"test_34":2}


def valid_solution(g):
    if len(g)!=81 or any(not isinstance(v,int) or not 1<=v<=9 for v in g):
        return False
    want=set(range(1,10))
    for r in range(9):
        if set(g[9*r:9*r+9])!=want or {g[9*i+r] for i in range(9)}!=want:
            return False
    for br in (0,3,6):
        for bc in (0,3,6):
            if {g[9*(br+dr)+bc+dc] for dr in range(3)
                for dc in range(3)}!=want:
                return False
    return True


def rectangles(g):
    """2×2 intercalate invariant; positive excludes cyclic canonical orbit."""
    t=0
    for r in range(9):
        for s in range(r+1,9):
            for c in range(9):
                a,b=g[9*r+c],g[9*s+c]
                for d in range(c+1,9):
                    if a!=b and a==g[9*s+d] and b==g[9*r+d]:
                        t+=1
    return t


def full_solution(seed):
    rng=random.Random(seed)
    g=[0]*81;rows=[0]*9;cols=[0]*9;boxes=[0]*9
    for box in (0,4,8):
        nums=list(range(1,10));rng.shuffle(nums)
        for j,v in enumerate(nums):
            r=3*(box//3)+j//3;c=3*(box%3)+j%3
            g[9*r+c]=v;bit=1<<v
            rows[r]|=bit;cols[c]|=bit;boxes[box]|=bit
    def dfs():
        best=-1;opts=[];n=10
        for i in range(81):
            if g[i]:continue
            r,c=divmod(i,9);box=3*(r//3)+c//3
            mask=0x3fe & ~(rows[r]|cols[c]|boxes[box])
            k=mask.bit_count()
            if not k:return False
            if k<n:
                best=i;n=k;opts=[v for v in range(1,10) if mask&(1<<v)]
                if k==1:break
        if best<0:return True
        rng.shuffle(opts)
        r,c=divmod(best,9);box=3*(r//3)+c//3
        for v in opts:
            bit=1<<v
            g[best]=v;rows[r]|=bit;cols[c]|=bit;boxes[box]|=bit
            if dfs():return True
            g[best]=0;rows[r]^=bit;cols[c]^=bit;boxes[box]^=bit
        return False
    if not dfs() or not valid_solution(g) or rectangles(g)<=0:
        raise ArithmeticError("noncanonical Sudoku generator failed")
    return g


def possible(g,i):
    r,c=divmod(i,9);used=0
    for j in range(9):
        used|=1<<g[9*r+j]
        used|=1<<g[9*j+c]
    for rr in range(3*(r//3),3*(r//3)+3):
        for cc in range(3*(c//3),3*(c//3)+3):
            used|=1<<g[9*rr+cc]
    return 0x3fe & ~used


def count_solutions(grid,limit=2):
    if len(grid)!=81 or limit!=2:
        raise ValueError("registered two-solution cutoff only")
    g=list(grid)
    def dfs(remaining):
        best=-1;mask=0;n=10
        for i in range(81):
            if g[i]:continue
            m=possible(g,i);k=m.bit_count()
            if not k:return 0
            if k<n:
                best=i;mask=m;n=k
                if k==1:break
        if best<0:return 1
        hits=0
        while mask:
            bit=mask&-mask;g[best]=bit.bit_length()-1
            hits+=dfs(remaining-hits)
            g[best]=0
            if hits>=remaining:return hits
            mask^=bit
        return hits
    return dfs(2)


def dig_unique(solution,seed,holes):
    if not valid_solution(solution) or holes not in (30,34):
        raise ValueError("registered hole count and solution required")
    rng=random.Random(seed)
    x=list(solution);count=0
    for i in rng.sample(range(81),81):
        if count==holes:break
        old=x[i];x[i]=0
        if count_solutions(x)==1:count+=1
        else:x[i]=old
    if count!=holes or count_solutions(x)!=1:
        raise ArithmeticError("failed exact unique puzzle dig")
    return x


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def visible_goals(puzzle):
    missing=[i for i,v in enumerate(puzzle) if v==0]
    if len(missing)<3:
        raise ValueError("not enough visible missing cells")
    result=(missing[len(missing)//3],missing[2*len(missing)//3])
    if result[0]==result[1]:
        raise ValueError("duplicate visible goal")
    return result


def build_rosters(pilot=False):
    rosters={}
    seen_targets=set();seen_inputs=set()
    for family,base in BASES.items():
        n=(PILOT if pilot else COUNTS)[family]
        group=[]
        for i in range(n):
            seed=base+i
            target=full_solution(seed)
            x=dig_unique(target,seed+1000000,HOLES[family])
            if tuple(target) in seen_targets or tuple(x) in seen_inputs:
                raise AssertionError("cross-split duplicate Sudoku")
            seen_targets.add(tuple(target));seen_inputs.add(tuple(x))
            group.append({"seed":seed,"x":x,"target":target,
                          "holes":HOLES[family],"rectangles":rectangles(target)})
        rosters[family]=group
    return rosters,{f:fingerprint(items) for f,items in rosters.items()}
