"""LS-SAE11 frozen, externally sourced hard Sudoku and exact uniqueness receipts.

The public Champagne inputs are pinned to exact Git blobs. No checkpoint, model
prediction, readout margin or intervention result enters source admission.
"""
from __future__ import annotations
import hashlib
import json
import urllib.request

from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as S7
from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2

REPO = "Snow-Hoe/sudoku-database"
COMMIT = "72f77030afd36ac3a31bc0708ec9179f5d1b8fc4"
ROOT = "sudoku-champagne-collection"
SOURCE_FILES = (
    ("100153.txt", "39cafe696597c056894bc7fd8a0d465810b57cf7"),
    ("100371.txt", "3ce8098d30aef823820e8a118c2b344cce51787c"),
    ("100587.txt", "762cb7bb28e8b7faca31e8d5fc64e7da6e4bff1d"),
    ("10080.txt", "f794e70295ff313f8ce8657bef0bb34bf619a1ff"),
    ("100804.txt", "adf6da58e8ae87ae475e02cce012f99b3ada5ab1"),
    ("101034.txt", "52e1fc43f73f4c1d9e277d4d3f428f5fbc0c323c"),
    ("101250.txt", "c50f4ab0a21a7cb940afeb217052d3fcf74166fc"),
    ("101465.txt", "d4b5f47f3fa8b98161a7a33502c5ee93778b6fb8"),
    ("101676.txt", "ce36e94ebcc3da30136e391c7fa57e0f4aec5a6d"),
    ("101887.txt", "4be06403b3f92591f5edf7e22690e386f6adef68"),
    ("102107.txt", "677ed63aad5d93b2839d3d36b1d2dc7f5d1e6ccf"),
    ("102318.txt", "f28104827120ea56a8cfcf0ebc21ad6d387f57ea"),
    ("102531.txt", "7573804771efeb95974c09323adaec8e374f5b6e"),
    ("102766.txt", "93b402bb47cda4ff0025895b9a8deb59e5b9ed31"),
    ("10290.txt", "e1fb88fd8ac7b1e29c07a0f7ffd5380888e04671"),
    ("103012.txt", "85e49670217b584c796e707aaa4dea38e2ae48e0"),
    ("103235.txt", "33d4be572bb5256938a9af0537c215b8e20fb1cb"),
    ("103448.txt", "386a1a57fc90b19919b6d18b1562cde31ebbc4b7"),
    ("103661.txt", "0aab4638bf0bf07df43f372b0a37a7948ae8bcda"),
    ("103875.txt", "05acb51cbc8568bef5ffbc74e2926a5ed5f1b63a"),
)
SOURCE_GROUPS = {"pilot_champagne":(0,4), "test_champagne_a":(4,12),
                 "test_champagne_b":(12,20)}
ALL = 0x3FE
MAX_NODES = 2_000_000

# Four constraints for each possible (cell, digit) assignment, 324 total.
_ASSIGNMENTS = {}
_COLUMNS = {k:set() for k in range(324)}
for i in range(81):
    r,c=divmod(i,9)
    b=3*(r//3)+c//3
    for digit in range(1,10):
        key=9*i+digit-1
        cols=(i, 81+9*r+digit-1, 162+9*c+digit-1, 243+9*b+digit-1)
        _ASSIGNMENTS[key]=cols
        for col in cols:_COLUMNS[col].add(key)


def git_blob_sha(data:bytes):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def read_source(name, digest):
    if (name,digest) not in SOURCE_FILES:
        raise ValueError("unregistered external puzzle requested")
    url=f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{ROOT}/{name}"
    with urllib.request.urlopen(url,timeout=60) as response:
        raw=response.read(82)
    if len(raw)!=81 or git_blob_sha(raw)!=digest:
        raise ValueError(f"original Champagne Git blob/hash/length mismatch: {name}")
    try:
        s=raw.decode("ascii")
    except UnicodeDecodeError as err:
        raise ValueError("nonascii registered Champagne source") from err
    if len(s)!=81 or any(ch not in "0123456789" for ch in s):
        raise ValueError("source puzzle must contain exactly 81 0..9 digits")
    return [int(ch) for ch in s],raw


def solve_unique(grid, limit=2, max_nodes=MAX_NODES):
    """Exact 324-column Sudoku Algorithm X, enumerate up to two solutions."""
    if len(grid)!=81 or limit!=2:
        raise ValueError("Sudoku exact two-solution cutoff only")
    initial=list(grid)
    if any(type(v) is not int or v<0 or v>9 for v in initial):
        raise ValueError("source Sudoku contains invalid digits")
    active={col:val.copy() for col,val in _COLUMNS.items()}
    chosen=[]
    nodes=0
    solutions=[]

    def select(row):
        popped=[]
        for col in _ASSIGNMENTS[row]:
            for other_row in active[col]:
                for other_col in _ASSIGNMENTS[other_row]:
                    if other_col!=col:
                        active[other_col].remove(other_row)
            popped.append(active.pop(col))
        return popped

    def deselect(row,popped):
        for col in reversed(_ASSIGNMENTS[row]):
            active[col]=popped.pop()
            for other_row in active[col]:
                for other_col in _ASSIGNMENTS[other_row]:
                    if other_col!=col:
                        active[other_col].add(other_row)

    for i,value in enumerate(initial):
        if value:
            row=9*i+value-1
            # Conflicting givens cannot coexist in a valid exact cover.
            if any(col not in active or row not in active[col]
                   for col in _ASSIGNMENTS[row]):
                return {"solutions":0,"target":None,"search_nodes":nodes}
            select(row)
            chosen.append(row)

    def recurse():
        nonlocal nodes
        if len(solutions)>=2:return
        if not active:
            solved=[0]*81
            for row in chosen:
                solved[row//9]=row%9+1
            solutions.append(solved)
            return
        col=min(active,key=lambda x:(len(active[x]),x))
        if not active[col]:return
        for row in sorted(tuple(active[col])):
            nodes+=1
            if nodes>max_nodes:
                raise RuntimeError("fixed exact source solver node cap exceeded")
            chosen.append(row)
            removed=select(row)
            recurse()
            deselect(row,removed)
            chosen.pop()
            if len(solutions)>=2:return

    recurse()
    if solutions:
        target=solutions[0]
        if any(a!=0 and target[i]!=a for i,a in enumerate(initial)):
            raise AssertionError("exact cover solver target contradicts given")
        if not S2.valid_solution(target):
            raise AssertionError("exact cover solver constructed invalid complete Sudoku")
    return {"solutions":len(solutions),"target":solutions[0] if len(solutions)==1 else None,
            "search_nodes":nodes}


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def admitted(source):
    if set(source)!=set(SOURCE_GROUPS):
        raise ValueError("source family coverage modified")
    for group,(start,stop) in SOURCE_GROUPS.items():
        if len(source[group])!=stop-start:
            raise ValueError("registered fixed full/pilot denominator changed")
    rows=[row for group in SOURCE_GROUPS for row in source[group]]
    if len({tuple(row["x"]) for row in rows})!=len(rows):
        raise ValueError("external source duplicates input puzzle")
    if len({tuple(row["target"]) for row in rows})!=len(rows):
        raise ValueError("external source duplicates solved target")
    if any(row["visible_structural_difficulty"]["propagation_solved"] for row in rows):
        raise ValueError("singles-solvable source admitted into external hard cohort")
    return True


def build_rosters():
    groups={name:[] for name in SOURCE_GROUPS}
    for i,(name,digest) in enumerate(SOURCE_FILES):
        x,raw=read_source(name,digest)
        result=solve_unique(x)
        if result["solutions"]!=1 or result["target"] is None:
            raise ValueError(f"source admission failed: {name} has {result['solutions']} solutions")
        vis=S7.visible_structure(x)
        if vis["propagation_solved"] or vis["deduced_missing_cells"]>=x.count(0):
            raise ValueError(f"source admission failed: {name} solved by singles")
        group=next(g for g,(start,stop) in SOURCE_GROUPS.items() if start<=i<stop)
        row={"source_index":i,"source_file":name,"source_git_blob_sha1":digest,
             "source_raw_sha256":hashlib.sha256(raw).hexdigest(),
             "seed":int(name.removesuffix(".txt")),
             "x":x,"target":result["target"],"holes":x.count(0),
             "visible_structural_difficulty":vis,"solver_search_nodes":result["search_nodes"],
             "source_repository_commit":COMMIT}
        groups[group].append(row)
    admitted(groups)
    return groups,{k:fingerprint(v) for k,v in groups.items()}
