"""LSA-02K frozen native equality-transport benchmark.

A word denotes a composition of opaque functions Nat -> Nat applied to x.
Public equations quantify over the suffix, e.g. forall z, a (b z) = c (d z).
The theorem transports a witness P(source) to P(target), for arbitrary
P : Nat -> Prop. Every rewrite is explicitly instantiated at a suffix.
The finite string model proposes scripts; Lean is the proof authority.

Generation selects reachable endpoints by BFS depth but stores no answer,
distance, generator walk or private reference action in public case rows.
Entire equation systems are separated across splits modulo renaming,
equation reordering and orientation. This is a restricted equality grammar,
not general dependent type theory or external Mathlib data.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from itertools import combinations, permutations, product
from pathlib import Path
import argparse
import hashlib
import json
import random

from .data import canonical, digest, sha256_file

SCHEMA = "lsa02k.equality_transport_suite.v1"
ALPHABET = "abcd"
RULE_COUNT = 6
SPLITS = {"train": 64, "calibration": 32, "ordinary_test": 32, "ood_family": 32}
SUITE_SEED = 20261204
BANDS = ((4, 5), (7, 8), (10, 11))
GEN_CAP = 16000


def term(word: str, leaf: str = "x") -> str:
    result = leaf
    for c in reversed(word):
        result = f"{c} {result}" if result == leaf else f"{c} ({result})"
    return result


def at_arg(word: str, leaf: str = "x") -> str:
    t = term(word, leaf)
    return f"({t})" if word else t


@dataclass(frozen=True)
class Move:
    rule: int
    reverse: bool
    position: int
    next_word: str

    def tactic(self, word: str) -> str:
        suffix = word[self.position+2:]
        arrow = "← " if self.reverse else ""
        return f"rw [{arrow}e{self.rule} {at_arg(suffix)}]"


@dataclass(frozen=True)
class EqCase:
    root: str
    lineage: str
    split: str
    family: str
    equations: tuple[tuple[str,str], ...]
    source: str
    target: str

    def validate(self):
        if self.split not in SPLITS or self.family != "indexed_unary_equations":
            raise ValueError("unsupported equality case split or family")
        if len(self.equations) != RULE_COUNT or len(set(tuple(sorted(x)) for x in self.equations)) != RULE_COUNT:
            raise ValueError("equation roster is missing or duplicated")
        for left,right in self.equations:
            if len(left)!=2 or len(right)!=2 or left==right or not set(left+right)<=set(ALPHABET):
                raise ValueError("only distinct length-preserving public pair equations allowed")
        if len(self.source)!=len(self.target) or self.source==self.target or not set(self.source+self.target)<=set(ALPHABET):
            raise ValueError("equality endpoints must be distinct equal-length words")
        if not 6<=len(self.source)<=10:
            raise ValueError("word length outside registered grammar")
        if self.lineage != system_lineage(self.equations):
            raise ValueError("canonical equation-system lineage mismatch")
        expected = digest([self.lineage, self.split, self.equations, self.source, self.target])[:24]
        if self.root != expected:
            raise ValueError("equality theorem root identity mismatch")
        return self

    def binders(self) -> str:
        h = "(a b c d : Nat → Nat) (x : Nat) (P : Nat → Prop)"
        for i,(lhs,rhs) in enumerate(self.equations):
            h += f" (e{i} : ∀ z : Nat, {term(lhs,'z')} = {term(rhs,'z')})"
        return h + f" (w : P {at_arg(self.source)})"

    @property
    def statement(self):
        return f"∀ {self.binders()}, P {at_arg(self.target)}"


def system_lineage(equations) -> str:
    """Conservative whole-system separation under all 24 letter renamings."""
    versions = []
    for p in permutations(ALPHABET):
        trans = str.maketrans(ALPHABET,"".join(p))
        versions.append(tuple(sorted(tuple(sorted((a.translate(trans),b.translate(trans))))
                                     for a,b in equations)))
    return digest(min(versions))


def neighbors(word: str, equations) -> tuple[Move,...]:
    moves = []
    for idx,(a,b) in enumerate(equations):
        for rev,(left,right) in enumerate(((a,b),(b,a))):
            for pos in range(len(word)-1):
                if word[pos:pos+2] == left:
                    moves.append(Move(idx,bool(rev),pos,word[:pos]+right+word[pos+2:]))
    return tuple(moves)


def choose_shell(source: str, rules, band, rng):
    """Private generation only: choose target uniformly from a reached shell."""
    seen={source:0}
    todo=deque([source])
    eligible=[]
    expansions=0
    while todo:
        w=todo.popleft()
        depth=seen[w]
        if depth>=band[0]:
            eligible.append(w)
        if depth>=band[1]:
            continue
        if expansions>=GEN_CAP:
            return None
        expansions+=1
        for action in neighbors(w,rules):
            nxt=action.next_word
            if nxt not in seen:
                seen[nxt]=depth+1
                todo.append(nxt)
    return rng.choice(sorted(eligible)) if eligible else None


def freeze(out: Path, seed: int=SUITE_SEED):
    if out.exists():
        raise ValueError("fresh immutable equality suite required")
    rng=random.Random(seed)
    all_pairs=["".join(p) for p in product(ALPHABET, repeat=2)]
    available=list(combinations(all_pairs,2))
    used=set()
    out.mkdir(parents=True)
    shards={}
    for split,count in SPLITS.items():
        rows=[]
        attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>count*500:
                raise RuntimeError("unable to generate frozen reachable equation-system roster")
            pairs=rng.sample(available,RULE_COUNT)
            # Keep ordering/orientation public and random, independent of target.
            rules=tuple((b,a) if rng.randrange(2) else (a,b) for a,b in pairs)
            lineage=system_lineage(rules)
            if lineage in used:
                continue
            # Balanced requested distance bands, independent system per row.
            band=BANDS[len(rows)%len(BANDS)]
            length=10 if split=="ood_family" else (7+(len(rows)//len(BANDS))%3)
            source="".join(rng.choice(ALPHABET) for _ in range(length))
            target=choose_shell(source,rules,band,rng)
            if target is None:
                continue
            root=digest([lineage,split,rules,source,target])[:24]
            c=EqCase(root,lineage,split,"indexed_unary_equations",rules,source,target).validate()
            used.add(lineage)
            rows.append(asdict(c))
        file=out/(split+".jsonl")
        file.write_text("".join(canonical(x)+"\n" for x in rows),encoding="utf-8")
        shards[split]={"file":file.name,"sha256":sha256_file(file),"roots":len(rows),
                       "lineages":[x["lineage"] for x in rows]}
    manifest={"schema":SCHEMA,"generator_seed":seed,"shards":shards,
              "generation_distance_bands":[list(x) for x in BANDS],
              "generation_expansion_cap":GEN_CAP,
              "public_rows_contain_answer":False,
              "whole_system_alpha_lineages_disjoint":True,
              "generator_sha256":sha256_file(Path(__file__)),
              "scientific_eligible":False}
    (out/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return manifest


def load_split(suite:Path, split:str):
    if split not in ("train","calibration"):
        raise ValueError("preflight refuses sealed ordinary/OOD task contents")
    meta=json.loads((suite/"manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema")!=SCHEMA or meta.get("generator_seed")!=SUITE_SEED:
        raise ValueError("unregistered equality suite")
    if set(meta["shards"])!=set(SPLITS):
        raise ValueError("suite split roster changed")
    seen=set()
    for name,row in meta["shards"].items():
        if len(row["lineages"])!=SPLITS[name] or len(set(row["lineages"]))!=SPLITS[name]:
            raise ValueError("manifest root lineages missing or duplicated")
        if seen.intersection(row["lineages"]):
            raise ValueError("whole equation systems cross split")
        seen.update(row["lineages"])
        # Authenticate sealed shard bytes only; never parse their cases.
        if row["file"]!=name+".jsonl" or sha256_file(suite/row["file"])!=row["sha256"]:
            raise ValueError("frozen equality suite checksum changed")
    data=meta["shards"][split]
    cases=[]
    for line in (suite/data["file"]).read_text(encoding="utf-8").splitlines():
        x=json.loads(line)
        if set(x)!=set(EqCase.__dataclass_fields__):
            raise ValueError("case includes private generation metadata")
        x["equations"]=tuple(tuple(p) for p in x["equations"])
        c=EqCase(**x).validate()
        if c.split!=split:
            raise ValueError("wrong equality split")
        cases.append(c)
    if [x.lineage for x in cases]!=data["lineages"] or len(cases)!=SPLITS[split]:
        raise ValueError("frozen equality lineage census changed")
    return cases,meta


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    print(canonical(freeze(a.out)))

if __name__=="__main__":
    main()
