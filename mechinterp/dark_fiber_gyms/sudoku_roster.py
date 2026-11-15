#!/usr/bin/env python3
"""Deterministic new unique-solution Sudoku puzzles for a *fresh* local LOOPSCOPE-04 roster.

Uses a permuted canonical solved grid, random clue deletion with an exact
count-up-to-two backtracker, and rejects exact/digit-relabel overlaps with
excluded previously evaluated datasets. Retains a solver-verified answer.
Provenance notes: these puzzles share the same solved-grid isomorphism class;
they are a local pilot gym, not a claim of broad-distribution evaluation.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import random

from .loopscope04_trm import canonical_digit_relabel, read_boards
from .core import sha256_file, write_json

FULL = (1 << 9) - 1


def generate_solution(rng: random.Random) -> str:
    bands = rng.sample(range(3), 3)
    stacks = rng.sample(range(3), 3)
    rows = [3*b+i for b in bands for i in rng.sample(range(3),3)]
    cols = [3*b+i for b in stacks for i in rng.sample(range(3),3)]
    digits = rng.sample(range(1,10),9)
    grid = [[digits[(r*3+r//3+c)%9] for c in cols] for r in rows]
    if rng.randrange(2):
        grid = [[grid[c][r] for c in range(9)] for r in range(9)]
    return "".join(str(v) for row in grid for v in row)


def count_solutions(puzzle: str, limit: int = 2) -> int:
    grid = [0 if c in ".0" else int(c) for c in puzzle]
    rmask,cmask,bmask = [0]*9,[0]*9,[0]*9
    for i,v in enumerate(grid):
        if not v: continue
        r,c=divmod(i,9)
        b=(r//3)*3+c//3
        bit=1<<(v-1)
        if (rmask[r]|cmask[c]|bmask[b])&bit:
            return 0
        rmask[r]|=bit;cmask[c]|=bit;bmask[b]|=bit
    answers=0
    def search():
        nonlocal answers
        if answers>=limit:
            return
        best=-1
        best_mask=0
        best_count=10
        for i,v in enumerate(grid):
            if v: continue
            r,c=divmod(i,9);b=(r//3)*3+c//3
            mask=FULL&~(rmask[r]|cmask[c]|bmask[b])
            n=mask.bit_count()
            if n==0:return
            if n<best_count:
                best,best_mask,best_count=i,mask,n
                if n==1:break
        if best<0:
            answers+=1
            return
        r,c=divmod(best,9);b=(r//3)*3+c//3
        mask=best_mask
        while mask and answers<limit:
            bit=mask&-mask;mask-=bit
            grid[best]=bit.bit_length()
            rmask[r]|=bit;cmask[c]|=bit;bmask[b]|=bit
            search()
            rmask[r]^=bit;cmask[c]^=bit;bmask[b]^=bit
            grid[best]=0
    search()
    return answers


def delete_clues(solution: str, target: int, rng: random.Random) -> str | None:
    grid=list(solution)
    indexes=list(range(81))
    rng.shuffle(indexes)
    for i in indexes:
        if sum(c!="." for c in grid)<=target:
            break
        old=grid[i]
        grid[i]="."
        if count_solutions("".join(grid))!=1:
            grid[i]=old
    return "".join(grid) if sum(c!="." for c in grid)==target else None


def run(args):
    rng=random.Random(args.seed)
    banned=set()
    excluded={}
    for path in args.exclude_csv:
        excluded[str(path)]=sha256_file(path)
        banned.update(canonical_digit_relabel(r["question"]) for r in read_boards(path))
    out={}
    used=set(banned)
    total=0
    for name,n in (("train",args.n_train),("eval",args.n_eval)):
        rows=[]
        for _ in range(args.max_attempts):
            if len(rows)>=n:
                break
            total+=1
            answer=generate_solution(rng)
            target=rng.randint(args.min_clues,args.max_clues)
            question=delete_clues(answer,target,rng)
            if question is None:
                continue
            signature=canonical_digit_relabel(question)
            if signature in used:
                continue
            if count_solutions(question)!=1:
                raise AssertionError("generator uniqueness regression")
            used.add(signature)
            rows.append({"question":question,"answer":answer})
        if len(rows)<n:
            raise RuntimeError(f"generator exhausted ({name}: {len(rows)}/{n})")
        out[name]=rows
    if not args.dry_run:
        args.out_dir.mkdir(parents=True,exist_ok=True)
        for name,rows in out.items():
            path=args.out_dir/f"{name}.csv"
            with path.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=("question","answer"))
                w.writeheader();w.writerows(rows)
    manifest={"schema":"darkfibers.sudoku.fresh_pilot.v1","seed":args.seed,
              "n_train":len(out["train"]),"n_eval":len(out["eval"]),
              "clue_range":[args.min_clues,args.max_clues],
              "exact_unique_solution_certified":True,"excluded_sha256":excluded,
              "collision_audit":"digit relabel + exact; broader Sudoku isomorphisms not checked",
              "source_distribution":"permuted base Latin solved grid, unique clue deletion",
              "generator_attempts":total,
              "puzzle_sha256":{k:sha256("".join(row["question"] for row in v).encode()).hexdigest()
                               for k,v in out.items()}}
    if not args.dry_run:
        manifest["csv_sha256"]={k:sha256_file(args.out_dir/f"{k}.csv") for k in out}
        write_json(args.out_dir/"roster_manifest.json",manifest)
    print(json.dumps(manifest,indent=2))
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--exclude-csv",type=Path,action="append",default=[],help="Previously evaluated puzzles")
    p.add_argument("--n-train",type=int,default=24)
    p.add_argument("--n-eval",type=int,default=12)
    p.add_argument("--min-clues",type=int,default=32)
    p.add_argument("--max-clues",type=int,default=38)
    p.add_argument("--max-attempts",type=int,default=5000)
    p.add_argument("--seed",type=int,default=20261115)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(argv)
    if not args.exclude_csv:
        p.error("at least one --exclude-csv is required")
    if args.n_train<1 or args.n_eval<1 or not(17<=args.min_clues<=args.max_clues<=81):
        p.error("invalid puzzle sizes/clues")
    return run(args)


if __name__=="__main__":
    raise SystemExit(main())
