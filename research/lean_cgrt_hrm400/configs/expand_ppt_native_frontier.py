"""Extend previously checked native training prefixes without reading evaluation scores."""

import argparse
from collections import Counter, deque
from pathlib import Path

from configs.collect_ppt_native_feedback import SCHEMA, catalogue, source_files
from configs.freeze_ppt_native_progress import advance, candidate_menu, native_sources, observe
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.native_feedback import PrefixGraph
from ppt_gym.ram_limits import TreeBudget


EXTRA_PREFIXES = 24
PREFIX_DEPTH = 5


def expand(repl, theorem):
    graph = PrefixGraph(theorem)
    nodes = graph.nodes.copy()
    checks = 0
    repl.reset()
    initial = repl.init(theorem["statement"], pp=True)
    visible = observe(repl, initial)
    if visible != nodes[()]["public"]:
        raise ValueError("the training theorem's public root changed")
    states = {(): (initial, visible)}

    def state_at(prefix):
        nonlocal checks
        if prefix not in states:
            parent = prefix[:-1]
            state, before = state_at(parent)
            status, child, after = advance(repl, state, before, prefix[-1])
            checks += 1
            edge = nodes[parent]["choices"][prefix[-1]]
            if status != "progress" or digest(after) != edge["after_sha256"]:
                raise ValueError("an accepted source prefix failed to replay")
            states[prefix] = child, after
        return states[prefix]

    def frontiers(prefix, node):
        return [(child, edge["after_sha256"]) for action, edge in node["choices"].items()
                if edge["status"] == "progress" and (child := (*prefix, action)) not in nodes
                and len(child) <= PREFIX_DEPTH]

    frontier = deque(sorted((item for prefix, node in nodes.items() for item in frontiers(prefix, node)),
                            key=lambda item: (len(item[0]), digest(item[0]))))
    observed = {digest(node["public"]) for node in nodes.values()}
    added = 0
    while frontier and added < EXTRA_PREFIXES:
        prefix, expected = frontier.popleft()
        if expected in observed:
            continue
        state, before = state_at(prefix)
        choices = {}
        for action in candidate_menu(before):
            status, _, after = advance(repl, state, before, action)
            checks += 1
            choices[action] = {"status": status, "after_sha256": digest(after)}
        node = {"prefix": prefix, "public": before, "choices": choices}
        nodes[prefix] = node
        observed.add(expected)
        added += 1
        frontier.extend(sorted(frontiers(prefix, node), key=lambda item: digest(item[0])))
    missing = sum(edge["status"] == "progress" and (*prefix, action) not in nodes
                  for prefix, node in nodes.items() for action, edge in node["choices"].items())
    return {**theorem, "nodes": list(nodes.values()), "unexpanded": missing, "added": added, "new_checks": checks}


def run(args, tree):
    original = read_receipt(args.training)
    modules, _ = catalogue(args.bpd)
    if (original["schema"] != SCHEMA or original["modules"] != modules
            or original["lean_sha256"] != sha256_file(LEAN_EXE) or original["repl_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the native source graph or verifier changed")
    rows = []
    paths = [path for path, _, required in source_files(args.bpd) if required]
    with native_sources(paths) as (directory, env, imports), LeanRepl(imports=imports, env=env, cwd=directory) as repl:
        if repl.version != LEAN_VERSION:
            raise ValueError("unexpected native proof toolchain")
        for theorem in original["rows"]:
            rows.append(expand(repl, theorem))
    counts = Counter(edge["status"] for proof in rows for node in proof["nodes"] for edge in node["choices"].values())
    print("added prefixes", sum(row["added"] for row in rows), "new Lean checks", sum(row["new_checks"] for row in rows),
          "combined dispositions", dict(counts), flush=True)
    write_receipt(args.output, {**original, "parent_sha256": sha256_file(args.training),
                                "rows": rows, "evidence": counts, "memory": tree.stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bpd", "training", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        run(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
