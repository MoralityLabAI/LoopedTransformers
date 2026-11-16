"""AF-24 independent per-episode audit, source hash, and control replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from functools import lru_cache

from experiments.activation_funnel_af23 import branching_gym as gym
from . import symbolic as ctl
from . import evaluate as ex


@lru_cache(maxsize=2048)
def audit_truth(c):
    return gym.independent_bfs(c)


def verify_episode(c, row):
    """Reconstruct legal physical rules and replay symbolic choice and proof."""
    graph = ctl.PublicHorn.from_public_features(gym.public_features(c))
    distance = audit_truth(c)
    if distance != gym.shortest_steps(c):
        raise ValueError("independent BFS mismatch")
    if (row["case_id"], row["split"]) != (c.id, c.split):
        raise ValueError("mismatched physical Horn case")
    if row["optimal_steps"] != distance or row["truth"] != (distance is not None):
        raise ValueError("falsified shortest-proof truth target")
    budget = (distance + 1) if distance is not None else 2
    if row["budget"] != budget or len(row["actions"]) > budget:
        raise ValueError("wrong registered execution budget")
    if row["initial"] != sorted(graph.initial):
        raise ValueError("forged initial physical atoms")
    state = graph.initial
    terminated = False
    for i, rec in enumerate(row["actions"]):
        if terminated or rec["before"] != sorted(state):
            raise ValueError("action after termination or wrong host state")
        action = rec["action"]
        if row["mode"] == "shortest":
            expected = ex.exact_oracle_action(graph, state, c, distance)
        else:
            expected = ctl.choose(
                graph, state, row["mode"], row["seed"], c.id, i
            )["action"]
        if action != expected or rec["work"]["action"] != action:
            raise ValueError("counterfeit symbolic action selection")
        if row["mode"] != "shortest":
            check = ctl.choose(
                graph, state, row["mode"], row["seed"], c.id, i
            )
            for key in ("expanded", "rule_examinations", "closure_passes",
                        "search_exhausted"):
                if check[key] != rec["work"][key]:
                    raise ValueError("symbolic work receipt was modified")
        if action is None:
            if rec["proof"] is not None:
                raise ValueError("STOP cannot contain proof")
            terminated = True
        else:
            apos, _ = gym.physical_maps(c)
            tological = {phys: logical for logical, phys in enumerate(apos)}
            known_logical = frozenset(tological[p] for p in state)
            host_next, proof = gym.sound_fire(c, known_logical, action)
            if proof != rec["proof"]:
                raise ValueError("forged host-incidence proof")
            state = graph.fire(state, action)
            if state != frozenset(apos[x] for x in host_next):
                raise ValueError("host physical incidence diverged")
            if graph.goal in state:
                terminated = True
        if rec["after"] != sorted(state):
            raise ValueError("incorrect post-state")
    if row["final"] != sorted(state) or row["solved"] != (graph.goal in state):
        raise ValueError("false proof terminal state")
    if row["false_accepted_proof"] or (distance is None and row["solved"]):
        raise ValueError("false accepted goal")
    stopped = bool(row["actions"] and row["actions"][-1]["action"] is None)
    if (row["premature_stop"] != bool(
            distance is not None and not row["solved"] and stopped)):
        raise ValueError("incorrect premature STOP receipt")
    if row["first_stop"] != bool(
            row["actions"] and row["actions"][0]["action"] is None):
        raise ValueError("incorrect initial STOP")
    for key, inner in (("node_expansions", "expanded"),
                       ("rule_examinations", "rule_examinations"),
                       ("closure_passes", "closure_passes"),
                       ("symbolic_wall_ns", "wall_ns")):
        if row[key] != sum(s["work"][inner] for s in row["actions"]):
            raise ValueError(f"wrong metered {key}")
    return True


def audit(seed, directory):
    directory = Path(directory)
    rec = json.loads((directory / "evaluation.json").read_text())
    path = directory / "episodes.jsonl"
    if rec["seed"] != seed or rec["experiment"] != "AF-24":
        raise ValueError("invalid AF-24 identity")
    if (rec["episodes_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest() or
        rec["source_protocol_sha256"] != ex.digest((ex.HERE / "PROTOCOL.json").read_bytes()) or
        rec["source_manifest_sha256"] != gym.manifest()["manifest_sha256"] or
        rec["source_branching_grammar_hash"] != ex.digest(
            (ex.HERE.parent / "activation_funnel_af23/branching_gym.py").read_bytes())):
        raise ValueError("incorrect source or episode hash")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    count = len(ex.SPLITS) * 192 * len(ex.ARMS)
    if rec["episode_count"] != count or len(rows) != count:
        raise ValueError("missing registered episodes")
    keys = set()
    for row in rows:
        key = (row["split"], row["mode"], row["case_id"])
        if key in keys:
            raise ValueError("duplicate case/arm trace")
        keys.add(key)
    checks = 0
    for split in ex.SPLITS:
        cases = gym.cases_for(split)
        for arm in ex.ARMS:
            part = []
            for c in cases:
                target = (split, arm, c.id)
                entry = next((r for r in rows if (r["split"], r["mode"],
                                r["case_id"]) == target), None)
                if entry is None:
                    raise ValueError("missing original Horn case")
                verify_episode(c, entry)
                part.append(entry)
                checks += 1
            summary = ex.summarize(part)
            for name in ("cases","provable","unprovable","solved",
                         "premature_stop","correct_initial_stop",
                         "false_accepted_proofs","node_expansions_total",
                         "rule_examinations_total"):
                if summary[name] != rec["scores"][split][arm][name]:
                    raise ValueError(f"tampered summary {name}")
    result = {
        "experiment": "AF-24", "seed": seed, "verified": True,
        "original_af23_run": rec["original_af23_ci"],
        "source_protocol_sha256": rec["source_protocol_sha256"],
        "source_manifest_sha256": rec["source_manifest_sha256"],
        "evaluation_sha256": ex.digest((directory / "evaluation.json").read_bytes()),
        "episode_sha256": rec["episodes_sha256"],
        "episode_count": checks,
        "false_accepted_proofs": 0,
    }
    (directory / "independent_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ex.SEEDS, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(audit(a.seed, a.directory)))


if __name__ == "__main__":
    main()
