"""Collect a training-only graph of live Lean tactic dispositions."""

import argparse
from collections import Counter, deque
from pathlib import Path

from configs.freeze_ppt_native_progress import DECLARATION, advance, candidate_menu, declaration_type, native_sources, observe
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import REPO
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.native-feedback.v1"
CHECK_DEPTH = 4
EXPANSIONS = 16
EVAL_REGISTRY_SHA = "f1dbaa04f8601ef90a5082f3ef892001bc76d63ba1460f1448b0bf759ed97496"


def source_files(bpd):
    theory = bpd / "mechinterp/bpd/theory"
    experiments = REPO / "experiments"
    return (
        (theory / "GreedyRank.lean", "BPD.GreedyRank", True),
        (theory / "EnergyFiberRank.lean", "BPD.EnergyFiberRank", True),
        (theory / "pd04/PlantedCapacity.lean", "BPD.PD04", True),
        (theory / "pd05/CountSeparation.lean", "BPD.PD05", True),
        (theory / "pd09/PairMixCost.lean", "BPD.PD09", True),
        (experiments / "activation_funnel_af10/CausalCut.lean", None, True),
        (experiments / "activation_funnel_af11/BooleanInfluence.lean", None, True),
        (experiments / "activation_funnel_af12/HornClosure.lean", None, False),
        (experiments / "activation_funnel_af16/RulePriority.lean", None, False),
    )


def catalogue(bpd):
    modules, declarations = [], []
    for path, namespace, needs_import in source_files(bpd):
        modules.append({"module": path.name, "sha256": sha256_file(path), "imported": needs_import})
        found = DECLARATION.findall(path.read_text(encoding="utf-8"))
        for name, header in sorted(found, key=lambda item: digest([SCHEMA, path.name, item[0]])):
            declarations.append({"module": path.name, "theorem": name, "statement": declaration_type(header, namespace)})
    return modules, declarations


def collect_graph(repl, case):
    repl.reset()
    initial = repl.init(case["statement"], pp=True)
    public = observe(repl, initial)
    queue = deque([((), initial, public)])
    nodes = []
    while queue and len(nodes) < EXPANSIONS:
        prefix, state, visible = queue.popleft()
        choices = {}
        for tactic in candidate_menu(visible):
            status, child, after = advance(repl, state, visible, tactic)
            choices[tactic] = {"status": status, "after_sha256": digest(after)}
            if status == "progress" and len(prefix) + 1 < CHECK_DEPTH:
                queue.append(((*prefix, tactic), child, after))
        nodes.append({"prefix": prefix, "public": visible, "choices": choices})
    return {**case, "nodes": nodes, "unexpanded": len(queue)}


def collect(args, tree):
    from configs.run_ppt_mathcan_progress import read_receipt

    if sha256_file(args.held) != EVAL_REGISTRY_SHA:
        raise ValueError("the sealed native comparison cohort changed")
    held = read_receipt(args.held)
    modules, declarations = catalogue(args.bpd)
    if {row["file"] for row in held["modules"]} & {row["module"] for row in modules}:
        raise ValueError("a native evaluation module entered training")
    if {row["statement"] for row in held["cases"] + held["unresolved"]} & {
            row["statement"] for row in declarations}:
        raise ValueError("a native evaluation theorem entered training")
    paths = [path for path, _, needed in source_files(args.bpd) if needed]
    rows = []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected Lean version")
            for case in declarations:
                rows.append(collect_graph(repl, case))
    evidence = Counter(edge["status"] for row in rows for node in row["nodes"] for edge in node["choices"].values())
    print("native train", len(rows), "original statements", sum(len(row["nodes"]) for row in rows),
          "public prefixes", dict(evidence), flush=True)
    write_receipt(args.output, {"schema": SCHEMA, "modules": modules, "rows": rows, "evidence": evidence,
                                "held_sha256": EVAL_REGISTRY_SHA, "lean_sha256": sha256_file(LEAN_EXE),
                                "repl_sha256": sha256_file(REPL_EXE), "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for part in ("bpd", "held", "output"):
        parser.add_argument(part, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        collect(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
