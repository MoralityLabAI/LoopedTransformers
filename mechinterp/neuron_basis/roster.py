"""Deterministic engineering roster; two independent unique-solution checks.

Generated puzzles are for engineering only. The prospective manifest requires
prior-roster exclusions and independent source audit before model access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path

UNITS = ([[9*r+c for c in range(9)] for r in range(9)] +
         [[9*r+c for r in range(9)] for c in range(9)] +
         [[9*(r+i)+c+j for i in range(3) for j in range(3)]
          for r in (0,3,6) for c in (0,3,6)])
PEERS = [set().union(*(u for u in UNITS if i in u))-{i} for i in range(81)]
FULL = (1<<9)-1


def solve_mrv(grid):
    """Bitmask DFS, capped at two solutions."""
    g = list(grid)
    found = []
    def dfs():
        if len(found) >= 2:
            return
        best, mask = -1, 0
        for i,v in enumerate(g):
            if v:
                continue
            m = FULL
            for j in PEERS[i]:
                if g[j]:
                    m &= ~(1<<(g[j]-1))
            if not m:
                return
            if best < 0 or m.bit_count() < mask.bit_count():
                best, mask = i, m
                if m.bit_count() == 1:
                    break
        if best < 0:
            found.append(tuple(g))
            return
        while mask and len(found)<2:
            b = mask & -mask
            mask ^= b
            g[best] = b.bit_length()
            dfs()
            g[best] = 0
    dfs()
    return found


def solve_exact(grid):
    """Independent Algorithm X, immutable column sets, two-solution cap."""
    def keys(i,d):
        r,c=divmod(i,9)
        return (i,81+9*r+d,162+9*c+d,243+9*(3*(r//3)+c//3)+d)
    rows = {(i,d): keys(i,d) for i,v in enumerate(grid)
            for d in range(9) if not v or v==d+1}
    cols = {k:set() for k in range(324)}
    for row, ks in rows.items():
        for k in ks:
            cols[k].add(row)
    found=[]
    def dfs(c, picked):
        if len(found)>=2:
            return
        if not c:
            a=[0]*81
            for i,d in picked:
                a[i]=d+1
            found.append(tuple(a))
            return
        k=min(c,key=lambda k:(len(c[k]),k))
        for row in sorted(c[k]):
            ks=rows[row]
            conflicts=set().union(*(c[j] for j in ks))
            nxt={j:v-conflicts for j,v in c.items() if j not in ks}
            dfs(nxt,picked+[row])
            if len(found)>=2:
                return
    dfs(cols,[])
    return found


def orbit_invariant(g):
    """Necessary invariants for Sudoku symmetry; distinct values rule out orbit
    equivalence. Collisions are conservatively rejected, not canonicalized."""
    def axis(x):
        return tuple(sorted(tuple(sorted(sum(bool(x[9*r+c]) for c in range(9))
                    for r in range(b,b+3))) for b in (0,3,6)))
    t=[g[9*c+r] for r in range(9) for c in range(9)]
    boxes=tuple(sorted(sum(bool(g[i]) for i in u) for u in UNITS[18:]))
    counts=tuple(sorted(g.count(d) for d in range(1,10)))
    return tuple(sorted((axis(g),axis(t)))),boxes,counts


def new_grid(rng):
    """Randomized full-grid search, avoiding fixed Latin-square symmetry class."""
    g=[0]*81
    def dfs():
        best=-1
        opts=None
        for i,v in enumerate(g):
            if not v:
                q=set(range(1,10))-{g[j] for j in PEERS[i]}
                if not q:
                    return False
                if opts is None or len(q)<len(opts):
                    best,opts=i,q
        if best<0:
            return True
        a=sorted(opts); rng.shuffle(a)
        for d in a:
            g[best]=d
            if dfs():
                return True
        g[best]=0
        return False
    assert dfs()
    return g


def make(count, seed):
    rng=random.Random(seed)
    rows=[]; inv=set(); solutions=set()
    for split in ("cal","eval","rep"):
        for k in range(count):
            for attempt in range(200):
                g=new_grid(rng); q=g[:]
                order=list(range(81)); rng.shuffle(order)
                for i in order:
                    old=q[i]; q[i]=0
                    if len(solve_mrv(q))!=1:
                        q[i]=old
                    if q.count(0)>=52:
                        break
                key=orbit_invariant(q)
                if q.count(0)<48 or key in inv or tuple(g) in solutions:
                    continue
                a,b=solve_mrv(q),solve_exact(q)
                if a!=[tuple(g)] or b!=[tuple(g)]:
                    raise RuntimeError("independent exact solvers disagree")
                inv.add(key); solutions.add(tuple(g))
                rows.append({"id":f"{split}-{k:03d}","split":split,
                    "question":"".join(str(d) if d else "." for d in q),
                    "answer":"".join(map(str,g))})
                break
            else:
                raise RuntimeError("source admission exhausted; no replacement policy")
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--count",type=int,default=6,help="boards per split")
    p.add_argument("--seed",type=int,default=2026112507)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.count<1:
        p.error("require new output and positive count")
    rows=make(a.count,a.seed)
    raw=(json.dumps(rows,indent=2)+"\n").encode()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("ENGINEERING roster",len(rows),hashlib.sha256(raw).hexdigest())


if __name__=="__main__":
    main()
