"""LSA-02K: bounded equational rewrite search with actual Lean certificates.

Words denote nested unary functions on an arbitrary type. Reversible rules
are universally quantified equality hypotheses. The same rewrite problem can
ask for equality or transport of a witness through an indexed Type family.
The public transition system is finite and explicit in its semantics, but
its state graph is implicit and can have exponentially many words.

Search uses a deliberately separate structured simulator, not the Horn-only
Lean REPL. Every claimed successful path is replayed from public inputs,
rendered into a full Lean proof, and independently checked by Lean 4.19.
No native verification is claimed for unsuccessful symbolic search attempts.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import random
import re
import subprocess

from lcgrt import config

SCHEMA = "lsa02k.equality_rewrite_suite.v1"
ALPHABET = "abcdef"
GENERATOR_SEED = 20261204
COUNTS = {"train": 64, "calibration": 32, "ordinary": 32, "ood": 32}
WORK = 128
DEPTH = 16
TEACHER_WORK = 32768
POLICIES = ("uniform", "astar", "bidir", "learned", "sham")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(canonical(obj) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path, rows):
    Path(path).write_text("".join(canonical(x)+"\n" for x in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


@dataclass(frozen=True, order=True)
class Move:
    rule: int
    direction: int
    position: int
    successor: str


@dataclass(frozen=True)
class Case:
    root: str
    lineage: str
    split: str
    family: str
    source: str
    target: str
    rules: tuple[tuple[str, str], ...]

    def validate(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.root):
            raise ValueError("invalid case identifier")
        if self.family not in ("equality", "dependent_transport"):
            raise ValueError("unsupported rewrite proof family")
        if self.split not in COUNTS:
            raise ValueError("unsupported source split")
        if (not 2 <= len(self.source) <= 14 or len(self.target) != len(self.source)
                or self.source == self.target
                or any(c not in ALPHABET for c in self.source+self.target)):
            raise ValueError("invalid public rewrite endpoints")
        if not 1 <= len(self.rules) <= 12:
            raise ValueError("invalid rule count")
        sides = []
        for a,b in self.rules:
            if not 2 <= len(a) <= 3 or len(a) != len(b) or a == b:
                raise ValueError("rules must be nontrivial length-preserving equalities")
            if any(c not in ALPHABET for c in a+b):
                raise ValueError("rule outside public alphabet")
            sides.extend((a,b))
        if len(sides) != len(set(sides)):
            raise ValueError("ambiguous duplicate rewrite sides")
        return self

    def public(self):
        d=asdict(self)
        d["rules"]=[list(x) for x in self.rules]
        return d

    @classmethod
    def parse(cls, obj):
        return cls(
            root=obj["root"],lineage=obj["lineage"],split=obj["split"],
            family=obj["family"],source=obj["source"],target=obj["target"],
            rules=tuple(tuple(x) for x in obj["rules"]),
        ).validate()


def neighbors(case, word):
    result=[]
    for i,(a,b) in enumerate(case.rules):
        for direction,(left,right) in enumerate(((a,b),(b,a))):
            for pos in range(len(word)-len(left)+1):
                if word[pos:pos+len(left)]==left:
                    result.append(Move(i,direction,pos,word[:pos]+right+word[pos+len(left):]))
    if len(result)>64:
        raise ValueError("public candidate menu exceeds the fixed model cap")
    return tuple(sorted(result))


def apply_move(case, word, move):
    if not 0 <= move.rule < len(case.rules) or move.direction not in (0,1):
        raise ValueError("invalid rule or direction")
    a,b=case.rules[move.rule]
    if move.direction:
        a,b=b,a
    p=move.position
    if p<0 or word[p:p+len(a)]!=a:
        raise ValueError("rewrite does not match the indicated public subterm")
    successor=word[:p]+b+word[p+len(a):]
    if successor!=move.successor:
        raise ValueError("claimed successor differs from structural rewrite")
    return successor


def replay(case, path, *, require_target=True):
    word=case.source
    seen=[word]
    for x in path:
        move=x if isinstance(x,Move) else Move(**x)
        word=apply_move(case,word,move)
        seen.append(word)
    if require_target and word!=case.target:
        raise ValueError("rewrite witness does not close the original target")
    return seen


def lineage(rules):
    # Canonicalize the *entire rule system* over alphabet permutations,
    # rule order and equality orientation. Each split must use new systems.
    options=[]
    for p in itertools.permutations(ALPHABET):
        trans=str.maketrans(ALPHABET,"".join(p))
        options.append(tuple(sorted(tuple(sorted((a.translate(trans),b.translate(trans))))
                                    for a,b in rules)))
    return digest(min(options))


def generate(seed=GENERATOR_SEED, counts=None):
    counts=COUNTS if counts is None else counts
    rng=random.Random(seed)
    seen=set()
    cases={k:[] for k in counts}
    witnesses={k:[] for k in counts}
    for split,n in counts.items():
        attempts=0
        while len(cases[split])<n:
            attempts+=1
            if attempts>n*100:
                raise RuntimeError("generator exhausted fixed retry cap")
            # Eight length-two and four length-three rules, all sides distinct.
            pairs=rng.sample(["".join(x) for x in itertools.product(ALPHABET,repeat=2)],16)
            triples=rng.sample(["".join(x) for x in itertools.product(ALPHABET,repeat=3)],8)
            rules=list(zip(pairs[::2],pairs[1::2]))+list(zip(triples[::2],triples[1::2]))
            rng.shuffle(rules)
            rules=tuple(rules)
            length=13 if split=="ood" else 10+len(cases[split])%2
            source="".join(rng.choice(ALPHABET) for _ in range(length))
            family="equality" if len(cases[split])%2==0 else "dependent_transport"
            dummy=Case("tmp","",split,family,source,"a"*length,rules)
            word=source
            visited={source}
            path=[]
            steps=12 if split=="ood" else 10
            for _ in range(steps):
                menu=[x for x in neighbors(dummy,word) if x.successor not in visited]
                if not menu:
                    break
                move=rng.choice(menu)
                path.append(move)
                word=move.successor
                visited.add(word)
            # Filter only generator witness validity/nontrivial endpoints.
            # Never select by performance of any evaluated search policy.
            if len(path)!=steps or hamming(source,word)<4:
                continue
            ident=lineage(rules)
            if ident in seen:
                continue
            seen.add(ident)
            root=f"{split}_{len(cases[split]):03d}_{ident[:12]}"
            case=Case(root,ident,split,family,source,word,rules).validate()
            replay(case,path)
            cases[split].append(case)
            witnesses[split].append({"root":root,"path":[asdict(x) for x in path]})
    return cases,witnesses


def freeze(out,seed=GENERATOR_SEED,counts=None):
    out=Path(out)
    if out.exists():
        raise ValueError("fresh frozen suite required")
    cases,witnesses=generate(seed,counts)
    out.mkdir(parents=True)
    artifacts={}
    for split,rows in cases.items():
        name=f"{split}.jsonl"
        write_rows(out/name,[x.public() for x in rows])
        artifacts[name]=file_sha(out/name)
        if split in ("train","calibration"):
            name=f"{split}_witnesses.jsonl"
            write_rows(out/name,witnesses[split])
            artifacts[name]=file_sha(out/name)
    result={
        "schema":SCHEMA,"seed":seed,"counts":{k:len(v) for k,v in cases.items()},
        "unique_rule_system_lineages":sum(len(v) for v in cases.values()),
        "generator_selects_by_policy_outcomes":False,
        "ordinary_ood_witness_files_written":False,
        "artifacts":artifacts,
    }
    write_json(out/"manifest.json",result)
    return result


def load_split(directory,split,*,witnesses=False):
    directory=Path(directory)
    if split not in ("train","calibration"):
        raise ValueError("ordinary/OOD sealed outcome access prohibited in this pilot")
    m=read_json(directory/"manifest.json")
    if m["schema"]!=SCHEMA:
        raise ValueError("unknown rewrite suite")
    # Hash verification does not deserialize locked shard cases or answers.
    for name,expected in m["artifacts"].items():
        if Path(name).name!=name or file_sha(directory/name)!=expected:
            raise ValueError("frozen suite artifact hash changed")
    cases=[Case.parse(x) for x in read_rows(directory/f"{split}.jsonl")]
    if len(cases)!=m["counts"][split] or len({x.lineage for x in cases})!=len(cases):
        raise ValueError("duplicate/missing rewrite source lineage")
    if any(x.split!=split for x in cases):
        raise ValueError("wrong source split")
    if not witnesses:
        return cases,m
    rows=read_rows(directory/f"{split}_witnesses.jsonl")
    by_root={x["root"]:x["path"] for x in rows}
    if set(by_root)!={x.root for x in cases}:
        raise ValueError("witness source roots differ")
    for x in cases:
        replay(x,by_root[x.root])
    return cases,m,by_root


def hamming(a,b):
    return sum(x!=y for x,y in zip(a,b))


def lower_bound(word,target):
    return (hamming(word,target)+2)//3


def public_packet(case,word,remaining):
    # Keep complete public rule system, goal kind, current and target words.
    prefix="E" if case.family=="equality" else "D"
    text=f"{prefix}|S:{word}|T:{case.target}|B:{remaining}|"+ ";".join(
        f"{i}:{a}={b}" for i,(a,b) in enumerate(case.rules))
    if len(text.encode())>384:
        raise ValueError("public proof packet truncated")
    return text


def action_token(move):
    return f"r{move.rule}:{move.direction}:{move.position}:{move.successor}"


def _path(parent,word):
    path=[]
    while parent[word] is not None:
        previous,move=parent[word]
        path.append(move)
        word=previous
    return list(reversed(path))


def _result(case,policy,path,work,expanded,calls,budget,depth):
    if work>budget:
        raise AssertionError("unaccounted search work")
    closed=path is not None
    if closed:
        replay(case,path)
        if len(path)>depth:
            raise AssertionError("rewrite witness exceeds proof depth cap")
    return {
        "root":case.root,"lineage":case.lineage,"family":case.family,
        "policy":policy,"closed":closed,"work":work,"expanded":expanded,
        "model_calls":calls,"budget":budget,"depth_limit":depth,
        "path":[asdict(x) for x in path] if closed else [],
    }


def search(case,policy,*,budget=WORK,depth=DEPTH,seed=0,ranker=None):
    """One charged unit for each examined legal successor, including repeats.

    A* shares a fixed admissible primary lower bound with both learned arms.
    Models only change tie-breaking within that priority. Public menu
    construction, model calls, and elapsed costs are separately reported.
    """
    if policy=="bidir":
        return bidir(case,budget=budget,depth=depth)
    if policy not in ("astar","uniform","learned","sham"):
        raise ValueError("unknown rewrite search policy")
    if (ranker is not None)!=(policy in ("learned","sham")):
        raise ValueError("learned policies require a frozen ranker")
    rng=random.Random(seed)
    queue=[(lower_bound(case.source,case.target),0.0,0,case.source,0)]
    parent={case.source:None}
    best={case.source:0}
    serial=0
    work=expanded=calls=0
    while queue and work<budget:
        _,_,_,word,g=heapq.heappop(queue)
        if g!=best[word] or g>=depth:
            continue
        expanded+=1
        menu=neighbors(case,word)
        if not menu:
            continue
        if ranker is not None:
            scores=ranker(case,word,menu,depth-g)
            if len(scores)!=len(menu) or any(not __import__("math").isfinite(x) for x in scores):
                raise ValueError("invalid frozen rewrite ranking")
            calls+=1
            ranked=sorted(zip(menu,scores),key=lambda x:(-x[1],x[0]))
        elif policy=="uniform":
            shuffled=list(menu)
            rng.shuffle(shuffled)
            ranked=[(move,rng.random()) for move in shuffled]
        else:
            ranked=[(move,-hamming(move.successor,case.target)) for move in menu]
            ranked.sort(key=lambda x:(-x[1],x[0]))
        for move,score in ranked:
            if work>=budget:
                break
            work+=1
            nxt=move.successor
            newg=g+1
            if nxt==case.target:
                return _result(case,policy,_path(parent,word)+[move],work,expanded,calls,budget,depth)
            if newg>=best.get(nxt,10**9) or newg>=depth:
                continue
            best[nxt]=newg
            parent[nxt]=(word,move)
            serial+=1
            primary=newg+lower_bound(nxt,case.target)
            heapq.heappush(queue,(primary,-float(score),serial,nxt,newg))
    return _result(case,policy,None,work,expanded,calls,budget,depth)


def bidir(case,*,budget=WORK,depth=DEPTH):
    """Layer-synchronous smaller-frontier bidirectional BFS.

    Every inspected successor costs one unit, exactly as in A*. In reverse
    search each rule direction is inverted when reconstructing the path.
    """
    fronts=[deque([case.source]),deque([case.target])]
    parent=[{case.source:None},{case.target:None}]
    distances=[{case.source:0},{case.target:0}]
    work=expanded=0
    while fronts[0] and fronts[1] and work<budget:
        side=0 if len(fronts[0])<=len(fronts[1]) else 1
        layer=fronts[side]
        next_layer=deque()
        while layer and work<budget:
            word=layer.popleft()
            g=distances[side][word]
            if g>=depth:
                continue
            expanded+=1
            for move in neighbors(case,word):
                if work>=budget:
                    break
                work+=1
                nxt=move.successor
                if nxt in distances[side]:
                    continue
                distances[side][nxt]=g+1
                parent[side][nxt]=(word,move)
                next_layer.append(nxt)
                if nxt in distances[1-side] and g+1+distances[1-side][nxt]<=depth:
                    left=_path(parent[0],nxt)
                    back=_path(parent[1],nxt)
                    right=[]
                    current=nxt
                    for prev in reversed(back):
                        # Recover reverse successor from the exact inverse rule.
                        a,b=case.rules[prev.rule]
                        if prev.direction:
                            a,b=b,a
                        pos=prev.position
                        dest=current[:pos]+a+current[pos+len(b):]
                        inverse=Move(prev.rule,1-prev.direction,pos,dest)
                        current=apply_move(case,current,inverse)
                        right.append(inverse)
                    return _result(case,"bidir",left+right,work,expanded,0,budget,depth)
        fronts[side]=next_layer
    return _result(case,"bidir",None,work,expanded,0,budget,depth)


def term(word,base="x"):
    result=base
    for c in reversed(word):
        result=f"(f_{c} {result})"
    return result


def render_proof(case,path,name):
    """Render from structured rule ids/positions only; never arbitrary Lean."""
    case.validate()
    if not re.fullmatch(r"[a-z][a-z0-9_]*",name):
        raise ValueError("unsafe Lean declaration name")
    steps=[x if isinstance(x,Move) else Move(**x) for x in path]
    replay(case,steps)
    binders=["(A : Type)","("+" ".join(f"f_{c}" for c in ALPHABET)+" : A → A)","(x : A)"]
    for i,(a,b) in enumerate(case.rules):
        binders.append(f"(h{i} : ∀ z : A, {term(a,'z')} = {term(b,'z')})")
    src,dst=term(case.source),term(case.target)
    if case.family=="dependent_transport":
        binders.extend(["(P : A → Type)",f"(witness : P {src})"])
        target=f"P {dst}"
    else:
        target=f"{src} = {dst}"
    lines=[f"theorem {name} {' '.join(binders)} : {target} := by",
           f"  have eqproof : {src} = {dst} := by",
           "    exact (calc" if steps else "    rfl"]
    current=case.source
    for i,move in enumerate(steps):
        left,right=case.rules[move.rule]
        if move.direction:
            left,right=right,left
        pos=move.position
        suffix=current[pos+len(left):]
        equality=f"(h{move.rule} {term(suffix)})"
        if move.direction:
            equality=f"(Eq.symm {equality})"
        if pos:
            equality=f"(congrArg (fun z : A => {term(current[:pos],'z')}) {equality})"
        lhs=term(current) if i==0 else "_"
        lines.append(f"      {lhs} = {term(move.successor)} := {equality}")
        current=move.successor
    if steps:
        lines.append("    )")
    lines.append("  exact Eq.mp (congrArg P eqproof) witness" if case.family=="dependent_transport" else "  exact eqproof")
    return "\n".join(lines)


def certify(items,out,timeout=420):
    out=Path(out)
    if out.exists():
        raise ValueError("fresh standalone native certificate destination required")
    out.mkdir(parents=True)
    if not items:
        raise ValueError("native compiler requires at least one concrete proof")
    names=[name for _,_,name in items]
    if len(set(names))!=len(names):
        raise ValueError("duplicate native proof declaration")
    source="namespace RewriteCert\n\n"+"\n\n".join(render_proof(c,p,n) for c,p,n in items)
    source+="\n\nend RewriteCert\n"+"\n".join(f"#print axioms RewriteCert.{n}" for n in names)+"\n"
    file=out/"RewriteCert.lean"
    file.write_text(source,encoding="utf-8")
    version=subprocess.run([str(config.LEAN_EXE),"--version"],capture_output=True,text=True,check=True,env=config.lean_env()).stdout.strip()
    if "version 4.19.0" not in version:
        raise ValueError("expected pinned Lean 4.19.0")
    result=subprocess.run([str(config.LEAN_EXE),str(file)],capture_output=True,text=True,timeout=timeout,env=config.lean_env())
    log=result.stdout+result.stderr
    (out/"compiler.log").write_text(log,encoding="utf-8")
    passed=set(re.findall(r"'RewriteCert\.([a-z0-9_]+)' does not depend on any axioms",log))
    if result.returncode!=0 or passed!=set(names):
        raise RuntimeError("Lean rejected standalone rewrite proof or axiom gate: "+log[-3500:])
    receipt={"lean_version":version,"lean_exe_sha256":file_sha(config.LEAN_EXE),
             "source_sha256":file_sha(file),"compiler_log_sha256":file_sha(out/"compiler.log"),
             "compiled_axiom_free":len(passed),"declarations":names,
             "native_online_tactic_calls":0,"actual_lean_compiler":True}
    write_json(out/"manifest.json",receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest="command",required=True)
    f=cmd.add_parser("freeze")
    f.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    result=freeze(a.out)
    print(canonical(result))
    return result


if __name__=="__main__":
    main()
