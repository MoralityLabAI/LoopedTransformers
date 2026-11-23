"""Exact bounded uniform-policy expectation from independently audited Lean transitions.

Uniform selection is over the actual public menu, not all hypotheses. The DP is
exact for the audited static propositional Horn context, including cycles,
multiple goals and dead ends. It is not an empirical Monte Carlo proof count.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
from pathlib import Path

from lcgrt import config
from lcgrt.certify import certify, decl_name
from lcgrt.extract import explore, goal_key, parse_context
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import candidate_menu, fingerprint, public_state
from .live_suite import load_suite

SCHEMA = "lbf10.native_uniform_expectation.v1"


def coefficients(table: dict, budget: int = 8) -> dict:
    """Probability of exactly n successful tactic steps, n <= budget.

    Every recursive child polynomial has zero constant term; increasing
    degree therefore resolves arbitrary cyclic finite goal graphs exactly.
    """
    if budget < 1 or budget > 32:
        raise ValueError("uniform baseline budget must be 1..32")
    for options in table.values():
        if not isinstance(options, dict):
            raise ValueError("expected named native actions")
        for children in options.values():
            if any(g not in table for g in children):
                raise ValueError("native graph contains an unvisited goal")
    p = {g: [Fraction(0) for _ in range(budget + 1)] for g in table}
    for n in range(1, budget + 1):
        for g, options in table.items():
            if not options:
                continue
            for children in options.values():
                # Product coefficient at degree n-1, using only prior degrees.
                poly = [Fraction(1)] + [Fraction(0)] * (n - 1)
                for h in children:
                    poly = [
                        sum((poly[i] * p[h][j-i] for i in range(j+1)), Fraction(0))
                        for j in range(n)
                    ]
                p[g][n] += poly[n - 1] / len(options)
    if any(sum(row) > 1 or any(x < 0 for x in row) for row in p.values()):
        raise ValueError("invalid uniform probability mass")
    return p


def audit_public_menu(repl, thm, graph):
    """Execute each public action on a representative of every reachable goal.

    Every successor must retain the same Horn hypotheses and unchanged pending
    tail. This validates the graph factorization and public-menu denominator.
    """
    repl.reset()
    root = repl.init(graph["statement"])
    table, ctx = graph["table"], graph["context"]
    queue = deque([root])
    seen = set()
    attempts = 0

    def inspect(state):
        goals = public_state(repl, state)["goals"]
        for goal in goals:
            if parse_context(goal["hypotheses"]) != ctx:
                raise ValueError("native uniform context is not static Horn")
        return [goal_key(g) for g in goals]

    initial = inspect(root)
    if initial != [graph["root"]]:
        raise ValueError("native root graph identity changed")
    while queue:
        state = queue.popleft()
        keys = inspect(state)
        if not keys:
            continue
        key = keys[0]
        if key in seen:
            continue
        if key not in table:
            raise ValueError("native public goal omitted from graph")
        seen.add(key)
        menu = candidate_menu(repl, state)
        if len(menu) != len(set(menu)) or set(menu) != set(table[key]):
            raise ValueError("public menu differs from Lean accepted graph at " + key)
        for tactic in menu:
            result = repl.run(state["sid"], tactic)
            attempts += 1
            if result.get("ok") is not True:
                raise ValueError("public uniform action rejected")
            child = result["state"]
            child_keys = inspect(child)
            expected = table[key][tactic] + keys[1:]
            if child_keys != expected:
                raise ValueError("native successor or pending tail changed")
            for i, goal in enumerate(table[key][tactic]):
                if goal in seen:
                    continue
                if i == 0:
                    queue.append(child)
                else:
                    focused = repl.focus(child["sid"], i)["state"]
                    if inspect(focused)[0] != goal:
                        raise ValueError("native focused goal mismatch")
                    queue.append(focused)
    if seen != set(table):
        raise ValueError("native uniform graph not fully audited")
    return {"representative_goals": len(seen), "audited_actions": attempts,
            "public_menu_exact": True, "static_horn_context": True}


def compute(suite: Path, out: Path, roots: int = 32, budget: int = 8):
    if out.exists() or roots != 32 or budget != 8:
        raise ValueError("LBF-10 requires fresh output and registered 32x8 baseline")
    selected, source = load_suite(suite, "calibration", roots)
    if len(selected) != roots:
        raise ValueError("incomplete registered calibration roster")
    out.mkdir(parents=True)
    rows, proofs = [], []
    with LeanRepl(timeout=60, exe=config.REPL_EXE, cwd=config.LEAN_DIR,
                  env=config.lean_env()) as repl:
        backend = fingerprint(config.REPL_EXE, config.LEAN_EXE, repl.version)
        if not backend["actual_lean_process"]:
            raise ValueError("uniform audit requires native Lean")
        for case in selected:
            thm = case.theorem()
            graph = explore(repl, thm)
            audit = audit_public_menu(repl, thm, graph)
            dist = coefficients(graph["table"], budget)[graph["root"]]
            mass = sum(dist, Fraction(0))
            shortest = graph["proof"]["size"]
            if (mass > 0) != (shortest <= budget):
                raise ValueError("uniform probability disagrees with native shortest witness")
            rows.append({
                "root": case.root, "lineage": case.lineage, "family": case.family,
                "native_shortest_steps": shortest, "reference_kernel": graph["proof"]["repl_kernel"],
                "goal": graph["root"], "table": graph["table"],
                "table_sha256": digest(graph["table"]),
                "menu_audit": audit, "exact_step_mass": [str(x) for x in dist],
                "probability": str(mass), "probability_float": float(mass),
            })
            proofs.append((case.tid+"_uniform_reference", thm, graph["proof"]["script"]))
    receipt = certify(proofs, timeout=360, keep=out/"standalone.lean")
    if receipt["timed_out"] or any(receipt["status"].get(decl_name(tid)) != "passed"
                                  for tid, _, _ in proofs):
        raise ValueError("native uniform reference standalone compilation failed")
    (out/"standalone_receipt.json").write_text(canonical(receipt)+"\n", encoding="utf-8")
    (out/"roots.jsonl").write_text("".join(canonical(row)+"\n" for row in rows),encoding="utf-8")
    mean = sum((Fraction(row["probability"]) for row in rows), Fraction(0))/roots
    manifest = {
        "schema": SCHEMA, "source": source, "backend": backend, "roots": roots,
        "check_budget": budget, "split": "calibration", "model_calls": 0,
        "query_label_training_updates": 0, "locked_test_opened": False,
        "native_public_menu_graph_audit_passed": True,
        "standalone_reference_certificates": roots,
        "audited_goals": sum(row["menu_audit"]["representative_goals"] for row in rows),
        "audited_public_actions": sum(row["menu_audit"]["audited_actions"] for row in rows),
        "expected_completion_exact": str(mean), "expected_completion": float(mean),
        "reference_reachable_within_budget": sum(row["native_shortest_steps"]<=budget for row in rows),
        "interpretation": "Exact bounded random-policy expectation under audited static native Horn transitions; not a sampled completion count.",
        "artifacts": {name: sha256_file(out/name) for name in
                      ("roots.jsonl","standalone_receipt.json","standalone.lean")},
        "scientific_eligible": False,
    }
    (out/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    r=compute(args.suite,args.out)
    print("LBF10_UNIFORM",r["expected_completion"],r["reference_reachable_within_budget"],
          r["audited_goals"],r["audited_public_actions"])


if __name__=="__main__":
    main()
