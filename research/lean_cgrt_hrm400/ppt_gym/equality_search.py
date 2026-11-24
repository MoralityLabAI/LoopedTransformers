"""Deterministic public equality planning and explicitly charged native Lean search.

The finite string state is an exact proposal grammar for nested unary terms.
It is NOT a proof authority: every live transition must agree with the Lean
server's actual target, and every surviving script must pass Lean kernel and
independent standalone compilation. Model-free planners may use only public
equations, source witness index, and current target index.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
import re
import time

from .data import canonical, digest
from .equality_suite import EqCase, Move, neighbors
from .live_lean import public_state

BUDGET = 16
CAPS = (16, 64, 256, 1024)
EXACT_CAP = 100000
RANDOM_SEEDS = (17,31,47,61,83)
SCHEMA = "lsa02k.native_equality_episode.v1"


def plan(case: EqCase, cap: int=EXACT_CAP) -> dict:
    """Public bidirectional BFS, exact first meeting of completed BFS shells.

    Each expansion enumerates the public moves of one reached word. Record
    both expanded states and candidate edges. The two queues expand complete
    depth shells, choosing the smaller frontier deterministically. If the
    next shell intersects the other visited ball, a shortest path has been
    found. A capped search that finds no intersection returns no proof.
    """
    if not 1<=cap<=EXACT_CAP:
        raise ValueError("invalid public planning expansion cap")
    start, goal=case.target,case.source
    parents=[{start:None},{goal:None}]
    front=[[start],[goal]]
    expansions=edges=0
    meet=None
    t0=time.perf_counter()
    while front[0] and front[1] and meet is None and expansions<cap:
        side=0 if len(front[0])<=len(front[1]) else 1
        new=[]
        for w in front[side]:
            if expansions>=cap or meet is not None:
                break
            expansions+=1
            moves=neighbors(w,case.equations)
            edges+=len(moves)
            for move in moves:
                n=move.next_word
                if n in parents[side]:
                    continue
                parents[side][n]=w
                new.append(n)
                if n in parents[1-side]:
                    meet=n
                    break
        front[side]=new
    words=[]
    if meet is not None:
        left=[meet]
        while parents[0][left[-1]] is not None:
            left.append(parents[0][left[-1]])
        right=[meet]
        while parents[1][right[-1]] is not None:
            right.append(parents[1][right[-1]])
        words=list(reversed(left))+right[1:]
    return {
        "found":meet is not None,
        "expansions":expansions,
        "candidate_edges_enumerated":edges,
        "visited_states":len(parents[0])+len(parents[1]),
        "planning_seconds":time.perf_counter()-t0,
        "expansion_cap":cap,
        "words":words,
        "rewrite_distance":len(words)-1 if words else None,
        "certification_status":"proposed_only",
    }


def path_moves(case: EqCase, words: list[str]) -> list[Move]:
    if not words or words[0]!=case.target or words[-1]!=case.source:
        raise ValueError("proposed path does not connect the actual theorem endpoints")
    result=[]
    for a,b in zip(words,words[1:]):
        move=next((m for m in neighbors(a,case.equations) if m.next_word==b),None)
        if move is None:
            raise ValueError("proposed equality path contains a nonpublic transition")
        result.append(move)
    return result


def _word_target(repl,state,word,case,*,check_context=False):
    visible=public_state(repl,state)
    if len(visible["goals"])!=1:
        raise ValueError("native indexed equality rewrite must retain exactly one goal")
    g=visible["goals"][0]
    ty=g["target"]
    raw=ty.get("raw","")
    # Lean pretty prints function applications with parentheses and possibly
    # line breaks. No type-directed oracle or expression evaluation is used.
    if not re.fullmatch(r"[A-Za-z0-9_()\s]+",raw):
        raise ValueError("unexpected Lean indexed predicate target: "+repr(raw))
    if re.findall(r"[A-Za-z_][A-Za-z_0-9]*",raw)!=["P",*word,"x"]:
        raise ValueError(f"Lean target and public rewrite grammar disagree: {raw!r} vs {word!r}")
    if check_context:
        hypotheses={h["name"]:h["type"] for h in g["hypotheses"]}
        required={"a","b","c","d","x","P","w",*(f"e{i}" for i in range(len(case.equations)))}
        if not required<=set(hypotheses):
            raise ValueError("Lean public local context omitted registered theorem binders")
        witness=hypotheses["w"].get("raw","")
        if re.findall(r"[A-Za-z_][A-Za-z_0-9]*",witness)!=["P",*case.source,"x"]:
            raise ValueError("Lean witness type differs from frozen public source index")
    return digest(visible)


@dataclass
class Frame:
    word: str
    state: dict
    script: tuple[str,...]
    tried: set[Move]=field(default_factory=set)
    sha: str=""


def _init(repl,case):
    repl.reset()
    state=repl.init(case.statement,pp=True)
    sha=_word_target(repl,state,case.target,case,check_context=True)
    return [Frame(case.target,state,(),set(),sha)]


def _act(repl,node,move,case,unit):
    if move is None:
        if node.word!=case.source:
            raise ValueError("exact witness at nonmatching public index")
        tactic="exact w"
        next_word=None
    else:
        if move not in neighbors(node.word,case.equations) or move in node.tried:
            raise ValueError("tactic outside current public rewrite menu")
        tactic=move.tactic(node.word)
        next_word=move.next_word
        node.tried.add(move)
    res=repl.run(node.state["sid"],tactic)
    if res.get("ok") is not True:
        raise ValueError("native Lean rejected a public grammar proposal: "+tactic+" "+str(res.get("error","")))
    child=res["state"]
    if next_word is None:
        if child["goals"]:
            raise ValueError("exact witness failed to close native predicate")
        receipt=repl.kernel(child["sid"])
        if receipt.get("ok") is not True or receipt.get("axioms")!=[]:
            raise ValueError("native equality proof failed axiom-free kernel gate")
        after=digest(public_state(repl,child))
    else:
        after=_word_target(repl,child,next_word,case)
    script=(*node.script,tactic)
    event={
        "unit":unit,"kind":"TACTIC","tactic":tactic,"word_before":node.word,
        "word_after":next_word,"move":asdict(move) if move else None,
        "state_before_sha256":node.sha,"state_after_sha256":after,
        "script_after":list(script),"kernel_closed":next_word is None,
    }
    return Frame(next_word,child,script,set(),after),event


def _rewind(stack,unit,reason):
    discarded=stack.pop()
    node=stack[-1]
    return {"unit":unit,"kind":"RECONSIDER","reason":reason,
            "word_before":discarded.word,"word_after":node.word,
            "state_before_sha256":discarded.sha,"state_after_sha256":node.sha,
            "abandoned_script":list(discarded.script),"script_after":list(node.script)}


def run_live(repl,case,policy,*,seed=0,words=None,budget=BUDGET):
    if policy not in ("random_dfs","hamming_dfs","bidirectional_bfs"):
        raise ValueError("unknown equality preflight policy")
    if not 1<=budget<=BUDGET:
        raise ValueError("registered equality work budget exceeded")
    if (words is not None)!=(policy=="bidirectional_bfs"):
        raise ValueError("only the public planning policy accepts a proposed path")
    stack=_init(repl,case)
    rng=random.Random(int(digest([case.root,seed,"lsa02k"])[:16],16))
    planned=path_moves(case,words) if words is not None else None
    if planned is not None and len(planned)+1>budget:
        raise ValueError("public planned proof does not fit native tactic budget")
    events=[]
    terminal="budget_exhausted"
    for used in range(budget):
        n=stack[-1]
        if n.word==case.source:
            child,event=_act(repl,n,None,case,used+1)
            stack.append(child)
            events.append(event)
            terminal="kernel_closed"
            break
        options=[m for m in neighbors(n.word,case.equations) if m not in n.tried]
        cycle=n.word in {x.word for x in stack[:-1]}
        if not options or cycle:
            if len(stack)==1:
                terminal="root_exhausted"
                break
            events.append(_rewind(stack,used+1,"cycle" if cycle else "exhausted_menu"))
            continue
        if planned is not None:
            action=planned[used]
        elif policy=="random_dfs":
            action=options[rng.randrange(len(options))]
        else:
            action=min(options,key=lambda m:(sum(a!=b for a,b in zip(m.next_word,case.source)),
                                              m.next_word,m.rule,m.reverse,m.position))
        child,event=_act(repl,n,action,case,used+1)
        stack.append(child)
        events.append(event)
    closed=terminal=="kernel_closed"
    return {"schema":SCHEMA,"root":case.root,"lineage":case.lineage,"split":case.split,
            "policy":policy,"seed":seed if policy=="random_dfs" else None,
            "budget":budget,"terminal_status":terminal,"live_kernel":closed,
            "script":list(stack[-1].script),"events":events,
            "tactic_attempts":sum(e["kind"]=="TACTIC" for e in events),
            "backtracks":sum(e["kind"]=="RECONSIDER" for e in events),
            "effort":len(events),"standalone_verified":False}


def replay(repl,case,episode):
    """Exact native state/transition replay with deterministic search history."""
    if (episode.get("schema")!=SCHEMA or episode.get("root")!=case.root
        or episode.get("lineage")!=case.lineage or episode.get("split")!=case.split
        or not 1<=episode.get("budget",0)<=BUDGET):
        raise ValueError("equality episode source identity or budget changed")
    stack=_init(repl,case)
    events=episode["events"]
    closed=False
    for unit,event in enumerate(events,1):
        if closed or unit>episode["budget"]:
            raise ValueError("equality replay violates terminal/work budget")
        n=stack[-1]
        if event["kind"]=="RECONSIDER":
            options=[m for m in neighbors(n.word,case.equations) if m not in n.tried]
            cycle=n.word in {x.word for x in stack[:-1]}
            if len(stack)<2 or (options and not cycle):
                raise ValueError("replay rollback lacks exhaustion or cycle evidence")
            expected=_rewind(stack,unit,"cycle" if cycle else "exhausted_menu")
        elif event["kind"]=="TACTIC":
            m=Move(**event["move"]) if event["move"] is not None else None
            child,expected=_act(repl,n,m,case,unit)
            stack.append(child)
            closed=m is None
        else:
            raise ValueError("unregistered equality action")
        if event!=expected:
            raise ValueError("native equality replay event differs from frozen transcript")
    if (episode["effort"]!=len(events)
        or episode["tactic_attempts"]!=sum(e["kind"]=="TACTIC" for e in events)
        or episode["backtracks"]!=sum(e["kind"]=="RECONSIDER" for e in events)
        or episode["script"]!=list(stack[-1].script)
        or episode["live_kernel"]!=closed):
        raise ValueError("native equality replay terminal or charged effort mismatch")
    return True
