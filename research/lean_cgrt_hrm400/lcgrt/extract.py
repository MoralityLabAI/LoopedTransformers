"""Lean-authoritative proof-data extraction.

For one theorem, Lean (via lgym) supplies: the elaborated local context, every
goal reached by backward search, the outcome of every candidate tactic on every
goal (with sampled error messages), the goals each successful tactic leaves, the
replay of the shortest proof as tactic *text*, and a kernel check of the
resulting term. Python only orchestrates the exploration and solves the AND-OR
graph that Lean reported. A disagreement with the generator's reference
semantics is an `extraction_mismatch`, and the theorem is dropped.
"""
from __future__ import annotations

import collections
import time

from . import fragment as F
from .errors import GymError

INF = F.INF


def flatten_imp(ty):
    """{"imp":[a,{"imp":[b,c]}]} -> ([a,b], c) with atom names."""
    pre = []
    while "imp" in ty:
        a, ty = ty["imp"]
        pre.append(atom_name(a))
    return pre, atom_name(ty)


def atom_name(ty):
    if "a" not in ty:
        raise GymError("extraction_mismatch", f"non-atomic proposition {ty}")
    return ty["a"]


def goal_key(goal) -> str:
    t = goal["target"]
    if "a" in t:
        return t["a"]
    if "and" in t:
        return "and(" + ",".join(goal_key({"target": x}) for x in t["and"]) + ")"
    raise GymError("extraction_mismatch", f"unexpected target {t}")


def parse_context(hyps):
    atoms, facts, rules = [], [], []
    for h in hyps:
        ty = h["type"]
        if ty.get("sort") == "Prop":
            atoms.append(h["name"])
        elif "a" in ty:
            facts.append([h["name"], ty["a"]])
        elif "imp" in ty:
            pre, head = flatten_imp(ty)
            rules.append([h["name"], pre, head])
        else:
            raise GymError("extraction_mismatch", f"unexpected hypothesis {h}")
    return {"atoms": atoms, "facts": facts, "rules": rules}


def solve(table: dict) -> dict:
    """Least fixpoint of min proof size over the Lean-reported transition table."""
    size = {k: INF for k in table}
    changed = True
    while changed:
        changed = False
        for k, cands in table.items():
            for tac, new in cands.items():
                s = 1 + sum(size.get(g, INF) for g in new)
                if s < size[k]:
                    size[k] = s; changed = True
    return size


def optimal_script(table, size, key):
    best = min(table[key], key=lambda tac: (1 + sum(size[g] for g in table[key][tac]), tac))
    out = [best]
    for g in table[key][best]:
        out += optimal_script(table, size, g)
    return out


def explore(repl, thm: F.Theorem, *, check_agreement: bool = True) -> dict:
    t0 = time.perf_counter(); c0 = repl.calls
    repl.reset()
    root = repl.init(F.statement(thm))
    rg = root["goals"]
    ctx = parse_context(repl.contexts[rg[0]["ctx"]])
    check_context(thm, ctx)
    table: dict[str, dict] = {}
    rejected: dict[str, int] = {}
    diagnostics: dict[str, list] = {}
    queue = collections.deque([(goal_key(rg[0]), root["sid"], len(rg))])
    queued = {goal_key(rg[0])}
    while queue:
        key, sid, n_goals = queue.popleft()
        ex = repl.expand(sid, errors=2)
        row, nrej = {}, 0
        for c in ex["candidates"]:
            if not c["ok"]:
                nrej += 1
                if "error" in c:
                    diagnostics.setdefault(key, []).append(c["error"])
                continue
            child = c["state"]["goals"]
            new = child[: len(child) - (n_goals - 1)]
            row[c["tactic"]] = [goal_key(g) for g in new]
            for i, g in enumerate(new):
                k = goal_key(g)
                if k in queued:
                    continue
                queued.add(k)
                if i == 0:
                    queue.append((k, c["state"]["sid"], len(child)))
                else:
                    f = repl.focus(c["state"]["sid"], i)
                    queue.append((k, f["state"]["sid"], len(child)))
        table[key] = row
        rejected[key] = nrej
        if key.startswith("P"):
            ref = F.reference_legal(thm, int(key[1:]))
            if set(row) != ref:
                raise GymError("extraction_mismatch", f"{thm.tid} {key}: lean {sorted(row)} ref {sorted(ref)}")
    size = solve(table)
    rkey = goal_key(rg[0])
    if size[rkey] >= INF:
        raise GymError("extraction_mismatch", f"{thm.tid}: Lean search found no proof")
    ref = F.reference_sizes(thm)
    for k, s in size.items():
        if k.startswith("P") and min(s, INF) != min(ref[int(k[1:])], INF):
            raise GymError("extraction_mismatch", f"{thm.tid} {k}: size lean {s} ref {ref[int(k[1:])]}")
    script = optimal_script(table, size, rkey)
    # Replay the script as tactic text; record the goal list Lean reports at each step.
    path = []
    sid, goals = root["sid"], [goal_key(g) for g in rg]
    for tac in script:
        r = repl.run(sid, tac)
        if not r["ok"]:
            raise GymError("extraction_mismatch", f"{thm.tid}: replay of {tac} failed: {r['error']}")
        path.append({"goals": goals, "tactic": tac})
        sid = r["state"]["sid"]; goals = [goal_key(g) for g in r["state"]["goals"]]
    if goals:
        raise GymError("extraction_mismatch", f"{thm.tid}: replay left goals {goals}")
    kern = repl.kernel(sid)
    if not kern["ok"] or kern["axioms"]:
        raise GymError("certificate_rejected", f"{thm.tid}: kernel {kern}")
    agreement = 0
    if check_agreement and len(path) > 1:
        # Re-run one mid-path state through expand and compare with the table.
        mid = len(path) // 2
        s2 = root["sid"]
        for st in path[:mid]:
            s2 = repl.run(s2, st["tactic"])["state"]["sid"]
        ex = repl.expand(s2, errors=0)
        n_rest = len(path[mid]["goals"]) - 1
        got = {c["tactic"]: [goal_key(g) for g in c["state"]["goals"][: len(c["state"]["goals"]) - n_rest]]
               for c in ex["candidates"] if c["ok"]}
        if got != table[path[mid]["goals"][0]]:
            raise GymError("extraction_mismatch", f"{thm.tid}: mid-path expand disagrees with table")
        agreement = 1
    return {
        "tid": thm.tid, "family": thm.family, "depth": thm.depth, "seed": thm.seed,
        "arity3": thm.arity3, "conj": len(thm.goal) > 1,
        "statement": F.statement(thm), "stmt_sha256": F.stmt_sha(thm),
        "canonical_hash": F.canonical_hash(thm),
        "context": ctx, "root": rkey,
        "table": table, "rejected": rejected, "diagnostics": diagnostics,
        "size": {k: (v if v < INF else None) for k, v in size.items()},
        "proof": {"script": script, "size": len(script), "repl_kernel": {"ok": True, "axioms": []}},
        "path": path, "agreement_checks": agreement,
        "lean": {"version": repl.version, "calls": repl.calls - c0,
                 "seconds": round(time.perf_counter() - t0, 4)},
        "gen": thm.gen,
    }


def check_context(thm: F.Theorem, ctx: dict):
    want_rules = [[f"r{i}", [f"P{p}" for p in pre], f"P{h}"] for i, (pre, h) in enumerate(thm.rules)]
    want_facts = [[f"h{i}", f"P{a}"] for i, a in enumerate(thm.facts)]
    if ctx["rules"] != want_rules or ctx["facts"] != want_facts or len(ctx["atoms"]) != thm.n_atoms:
        raise GymError("extraction_mismatch", f"{thm.tid}: elaborated context differs from statement")
