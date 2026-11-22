"""Verify the frozen fresh equality layouts in native Lean."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from configs.check_ppt_bpd_rewrite import EqualityReference
from configs.freeze_ppt_bpd_rewrite_transfer import SCHEMA
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import SearchCase
from ppt_gym.ram_limits import TreeBudget


def load_transfer(path):
    document = json.loads(path.read_bytes())
    groups = document["layouts"]
    if document["schema"] != SCHEMA or len(groups) != document["registration"]["layout_count"]:
        raise ValueError("unexpected fresh rewrite readout")
    roots, cases = set(), {}
    for group in groups:
        cluster = group["layout_id"]
        worlds = group["worlds"]
        if cluster in cases or len(worlds) != 2 or {world["valid"] for world in worlds} != {False, True}:
            raise ValueError("the fresh readout lost its matched layout pair")
        cases[cluster] = []
        for raw in worlds:
            case = SearchCase(**raw["case"])
            if case.split != "confirm" or case.cluster != cluster or case.root in roots:
                raise ValueError("the fresh readout repeated or reassigned a theorem")
            roots.add(case.root)
            cases[cluster].append((case, raw["valid"], group["proof_checks"]))
    return document, cases


def native_reference(path, tree):
    from lcgrt.lean_repl import LeanRepl
    from ppt_gym.hybrid_rl.__main__ import rollout
    from ppt_gym.live_search import replay_search
    from ppt_gym.ram_campaign import configure

    configure(tree.threads, 151)
    document, layouts = load_transfer(path)
    reg = document["registration"]
    options = SimpleNamespace(cpu_seconds=reg["cpu_seconds"], max_macs=reg["max_macs"])
    rows = {}
    with LeanRepl(timeout=15) as repl:
        if repl.version != "4.19.0":
            raise ValueError("the fresh equality layout witness requires Lean 4.19.0")
        for cluster, worlds in layouts.items():
            for case, valid, expected in worlds:
                episode = rollout(repl, tree, case, EqualityReference(), options)
                replay = replay_search(repl, case, episode)
                if (not episode["within_cpu_budget"] or not replay["replay_passed"]
                        or episode["verified_closed"] != valid
                        or (valid and episode["tactic_checks"] != expected)):
                    raise ValueError(f"the fresh layout witness differs for {case.tid}")
                rows[case.root] = {"cluster": cluster, "certified": valid, "checks": episode["tactic_checks"],
                                   "certificate": episode["certificate"], "replay": replay}
    return {"cohort_sha256": sha256_file(path), "native_lean": "4.19.0", "rows": rows, "memory": tree.stats()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        report = native_reference(args.cohort, tree)
        payload = gzip.compress(canonical(report).encode("utf-8"), mtime=0)
        with args.output.open("xb") as output:
            output.write(payload)
    finally:
        tree.close()
    print("Native Lean verified", sum(row["certified"] for row in report["rows"].values()), "fresh equality proofs.")


if __name__ == "__main__":
    main()
