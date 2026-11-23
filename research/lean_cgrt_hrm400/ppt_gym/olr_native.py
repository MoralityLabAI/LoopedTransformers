"""OLR-01B: independent native Lean paired-world preflight.

This is a structured-graph oracle and verifier-integration test. It contains
zero learned parameters. A good result cannot establish a neural architecture
advantage. The scorer consumes only public Lean goal/hypothesis JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, replace
from math import inf, isfinite
from pathlib import Path

from lcgrt import config, fragment as F
from lcgrt.certify import certify, decl_name
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import candidate_menu, public_state
from .obligation_ledger import Hypergraph, Rule, mean_premise_cue, propagate

SCHEMA = "ppt_gym.olr_native_pair_preflight.v1"
ARMS = ("ledger", "mean_only", "edge_swap", "force_r0", "force_r1")
NAME = re.compile(r"^P([0-9]{1,4})$")
TACTIC = re.compile(r"^apply ([hr][0-9]{1,4})$")


@dataclass(frozen=True)
class PairWorld:
    pair: str
    world: str
    theorem: F.Theorem
    # Private witness used for auditing outcomes ONLY; never passed to rank().
    witness: str
    lineage: str


@dataclass(frozen=True)
class ParsedGoal:
    graph: Hypergraph
    target: int
    rules: dict[str, Rule]
    facts: dict[str, int]
    unparsed: tuple[str, ...]


def pair_worlds(seed: int, variant: int) -> tuple[PairWorld, PairWorld]:
    """Two paired Horn worlds: same atoms, same facts, opposite viable root.

    Nuisance-chain length varies across pairs, giving distinct structural
    fingerprints in the preflight; rule-name and atom-position cues are shuffled.
    """
    if not 0 <= variant < 8:
        raise ValueError("bounded structural variant 0..7 required")
    rng = random.Random(f"olr-native-pair-v1/{seed}/{variant}")
    indices = list(range(5 + variant))
    rng.shuffle(indices)
    a, b, c, d, target = indices[:5]
    extras = indices[5:]
    facts = [a, b]
    rng.shuffle(facts)
    decoys = []
    previous = c
    for node in extras:
        decoys.append([[previous], node])
        previous = node
    # Both worlds have exactly one feasible binary candidate rule to the goal.
    producers = ([[a, b], target], [[c, d], target])
    pairs = []
    for world, chosen in (("w0", 0), ("w1", 1)):
        order = producers if world == "w0" else producers[::-1]
        thm = F.Theorem(
            tid=f"olr01b_{seed}_{variant}_{world}",
            family="olr01b-native-paired", depth=2, seed=seed,
            n_atoms=len(indices), facts=list(facts),
            rules=[[[*premises], head] for premises, head in (*order, *decoys)],
            goal=[target], gen={"evidence": "preflight_reference_only"},
        )
        pairs.append(PairWorld(
            pair=f"olr01b_{seed}_{variant}", world=world, theorem=thm,
            witness=f"apply r{chosen}", lineage=F.canonical_hash(thm),
        ))
    if pairs[0].lineage != pairs[1].lineage:
        raise AssertionError("paired worlds must share an unlabeled hypergraph")
    return tuple(pairs)


def atom(term: object) -> int | None:
    if isinstance(term, dict) and type(term.get("a")) is str:
        m = NAME.fullmatch(term["a"])
        return int(m.group(1)) if m else None
    return None


def horn_type(term: object) -> tuple[tuple[int, ...], int] | None:
    """Parse only the public fragment of Lean's propJson. Raw types stay unknown."""
    inputs = []
    while isinstance(term, dict) and "imp" in term:
        parts = term["imp"]
        if not isinstance(parts, list) or len(parts) != 2:
            return None
        premise = atom(parts[0])
        if premise is None:
            return None
        inputs.append(premise)
        term = parts[1]
    head = atom(term)
    return (tuple(inputs), head) if head is not None else None


def parse_public(visible: dict) -> ParsedGoal:
    goals = visible.get("goals", [])
    if not isinstance(goals, list) or not goals:
        raise ValueError("OLR-01B parser requires an active native Lean goal")
    # Parse the focused goal; the other goals remain pending in the native
    # packet and are accounted for by Lean across subsequent tactic calls.
    goal = goals[0]
    target = atom(goal.get("target"))
    if target is None:
        raise ValueError("OLR-01B root target is outside simple Horn subset")
    facts, rules, unparsed = {}, {}, []
    nodes = {target}
    for hyp in goal.get("hypotheses", []):
        name = hyp.get("name")
        if not isinstance(name, str):
            continue
        if not TACTIC.fullmatch("apply " + name):
            continue
        parsed = horn_type(hyp.get("type"))
        if parsed is None:
            unparsed.append(name)
            continue
        premises, head = parsed
        nodes.update((*premises, head))
        if premises:
            rules[name] = Rule(head, premises)
        elif name.startswith("h"):
            facts[name] = head
        else:
            unparsed.append(name)
    graph = Hypergraph(
        nodes=max(nodes) + 1,
        facts=frozenset(facts.values()),
        rules=tuple(rules.values()),
    )
    return ParsedGoal(graph, target, rules, facts, tuple(unparsed))


def rule_swap(parsed: ParsedGoal) -> ParsedGoal:
    """Sham: alter *only* paired r0/r1 premise grouping in the readout."""
    if "r0" not in parsed.rules or "r1" not in parsed.rules:
        raise ValueError("paired-edge sham requires both root producers")
    left, right = parsed.rules["r0"], parsed.rules["r1"]
    if left.head != right.head or len(left.premises) != len(right.premises):
        raise ValueError("paired rules changed arity/target")
    changed = dict(parsed.rules)
    changed["r0"] = replace(left, premises=right.premises)
    changed["r1"] = replace(right, premises=left.premises)
    return replace(parsed, graph=replace(parsed.graph, rules=tuple(changed.values())), rules=changed)


def rank(visible: dict, menu: tuple[str, ...], arm: str, *, at_root: bool) -> tuple[str, dict]:
    """Decision uses native public data only. No witness/metadata argument."""
    if not menu:
        raise ValueError("rank called with an empty menu")
    if arm not in ARMS:
        raise ValueError("unrecognized structural arm")
    if at_root and arm.startswith("force_"):
        forced = "apply " + arm.removeprefix("force_")
        if forced not in menu:
            raise ValueError("forced root tactic absent from public grammar")
        return forced, {"forced_witness": True}
    parsed = parse_public(visible)
    if arm == "mean_only" and at_root:
        # Aggregate all incoming premise bits into one pooled head cue, then
        # tie-break the two original candidate IDs without edge information.
        return sorted(menu)[0], {"pooled_cue": mean_premise_cue(parsed.graph, parsed.target)}
    if arm == "edge_swap" and at_root:
        parsed = rule_swap(parsed)
    tr = propagate(parsed.graph, max_passes=16, fact_cost=1.)
    cost = {}
    for choice in menu:
        match = TACTIC.fullmatch(choice)
        name = match.group(1) if match else None
        if name in parsed.facts and parsed.facts[name] == parsed.target:
            value = 1.
        elif name in parsed.rules and parsed.rules[name].head == parsed.target:
            rule = parsed.rules[name]
            value = rule.cost + sum(tr.costs[p] for p in rule.premises)
        else:
            value = inf
        cost[choice] = value
    action = min(menu, key=lambda t: (cost[t], t))
    return action, {"cost": {k: float(v) if isfinite(v) else None
                             for k, v in cost.items()},
                    "converged": tr.converged, "passes": tr.steps,
                    "unparsed": list(parsed.unparsed)}


def run_episode(lean: LeanRepl, world: PairWorld, arm: str, *, checks: int = 3) -> dict:
    if not 1 <= checks <= 8:
        raise ValueError("bounded native tactic budget required")
    lean.reset()
    state = lean.init(F.statement(world.theorem), pp=True)
    expected_menu = ("apply r0", "apply r1")
    original = public_state(lean, state)
    root_menu = candidate_menu(lean, state)
    if root_menu != expected_menu:
        raise ValueError(f"registered Lean root menu changed: {root_menu}")
    root_cue = mean_premise_cue(parse_public(original).graph, parse_public(original).target)
    if root_cue != 0.5:
        raise ValueError("the registered mean-alias cue changed")
    spent, events = 0, []
    status = "budget_exhausted"
    for step in range(checks):
        visible = public_state(lean, state)
        menu = candidate_menu(lean, state)
        if not menu:
            status = "no_candidates"
            break
        # Only root mode changes; continuation uses the same public-only
        # structural action selection for every arm.
        tactic, info = rank(visible, menu, arm if step == 0 else "ledger", at_root=step == 0)
        before = digest(visible)
        reply = lean.run(state["sid"], tactic)
        spent += 1
        event = {"step": spent, "tactic": tactic, "menu": list(menu),
                 "before": before, "rank": info, "ok": reply["ok"]}
        if not reply["ok"]:
            status = "rejected"
            event["after"] = before
            events.append(event)
            break
        state = reply["state"]
        new = public_state(lean, state)
        event["after"] = digest(new)
        event["goals_after"] = len(new["goals"])
        events.append(event)
        if not new["goals"]:
            cert = lean.kernel(state["sid"])
            if cert.get("ok") is True and cert.get("axioms") == []:
                status = "kernel_closed"
            else:
                status = "certificate_rejected"
            event["kernel"] = {"ok": cert.get("ok"), "axioms": cert.get("axioms")}
            break
    return {
        "root": digest(F.statement(world.theorem)), "pair": world.pair,
        "world": world.world, "lineage": world.lineage,
        "arm": arm, "menu": list(root_menu), "root_cue": root_cue,
        "status": status, "checks": spent, "budget": checks,
        "script": [e["tactic"] for e in events if e["ok"]],
        "events": events,
    }


def replay(lean: LeanRepl, world: PairWorld, episode: dict) -> bool:
    """Replay public menus, verified transitions and terminal kernel checks."""
    if episode["root"] != digest(F.statement(world.theorem)):
        raise ValueError("changed native theorem statement")
    lean.reset()
    state = lean.init(F.statement(world.theorem), pp=True)
    for event in episode["events"]:
        public = public_state(lean, state)
        menu = candidate_menu(lean, state)
        if (event["before"] != digest(public) or
                event["menu"] != list(menu) or event["tactic"] not in menu):
            raise ValueError("replayed root/choice/menu differs")
        reply = lean.run(state["sid"], event["tactic"])
        if reply["ok"] != event["ok"]:
            raise ValueError("replayed tactic verdict differs")
        if not reply["ok"]:
            if event["after"] != digest(public):
                raise ValueError("replayed tactic rejection differs")
            break
        state = reply["state"]
        new = public_state(lean, state)
        if event["after"] != digest(new) or event["goals_after"] != len(new["goals"]):
            raise ValueError("replayed successor obligations differ")
        if not new["goals"]:
            cert = lean.kernel(state["sid"])
            if not (cert.get("ok") is True and cert.get("axioms") == []):
                raise ValueError("replayed kernel certificate differs")
            break
    else:
        if episode["status"] == "kernel_closed":
            raise ValueError("missing terminal certificate in replay")
    return True


def verify(seed: int, count: int, out: Path, *, timeout: float = 60.) -> dict:
    if not 1 <= count <= 8 or out.exists():
        raise ValueError("1..8 pairs and a new output directory required")
    pairs = [pair_worlds(seed + i, i) for i in range(count)]
    fingerprints = {pair[0].lineage for pair in pairs}
    if len(fingerprints) != count:
        raise ValueError("distinct pairs share a structural fingerprint")
    episodes = []
    lookup = {}
    with LeanRepl(timeout=timeout) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("requires pinned native Lean 4.19")
        version = lean.version
        for pair in pairs:
            for world in pair:
                lookup[(world.pair, world.world)] = world
                for arm in ARMS:
                    ep = run_episode(lean, world, arm)
                    episodes.append(ep)
        for pair in pairs:
            per_world = [[e for e in episodes if e["pair"] == w.pair and
                          e["world"] == w.world] for w in pair]
            for world, eps in zip(pair, per_world, strict=True):
                assert len(eps) == len(ARMS)
                by_arm = {e["arm"]: e for e in eps}
                if (by_arm["ledger"]["status"] != "kernel_closed" or
                        by_arm["edge_swap"]["status"] == "kernel_closed" or
                        by_arm["force_r0"]["status"] == by_arm["force_r1"]["status"]):
                    raise ValueError("registered native paired witness outcomes changed")
                if (next(e for e in eps if e["arm"] == "ledger")["script"][0]
                        != world.witness):
                    raise ValueError("structural action disagreed with independent witness")
                if any(e["menu"] != list(("apply r0", "apply r1")) for e in eps):
                    raise ValueError("unmatched public action menus")
            if sum(e["status"] == "kernel_closed" for ws in per_world
                   for e in ws if e["arm"] == "mean_only") != 1:
                raise ValueError("mean-aggregation reference must fail one paired world")
        for ep in episodes:
            if not replay(lean, lookup[ep["pair"], ep["world"]], ep):
                raise ValueError("native replay failed")
    winners = [e for e in episodes if e["arm"] == "ledger"]
    declarations = [(f"{lookup[e['pair'],e['world']].theorem.tid}_ledger",
                     lookup[e["pair"],e["world"]].theorem, e["script"]) for e in winners]
    out.mkdir(parents=True, exist_ok=False)
    cert = certify(declarations, timeout=180, keep=out / "standalone.lean")
    if any(cert["status"].get(decl_name(tid)) != "passed" for tid, _, _ in declarations):
        raise ValueError(f"standalone Lean certification failed: {cert['rejected']}")
    receipt = {
        "schema": SCHEMA,
        "classification": "structural_preflight_not_neural_architecture_evidence",
        "seed": seed, "pairs": count, "worlds": 2*count,
        "pair_lineages": sorted(fingerprints), "native_lean_version": version,
        "lean_sha256": sha256_file(config.LEAN_EXE),
        "repl_sha256": sha256_file(config.REPL_EXE),
        "all_root_menus_equal": True,
        "same_mean_premise_cue": 0.5,
        "independent_compiled_scripts": cert["passed"],
        "all_episodes_replayed": len(episodes),
        "scorecard": {arm: {
            "kernel_closures": sum(e["status"] == "kernel_closed" for e in episodes
                                   if e["arm"] == arm),
            "native_tactic_calls": sum(e["checks"] for e in episodes
                                       if e["arm"] == arm),
        } for arm in ARMS},
        "architecture_promotion_eligible": False,
    }
    (out / "episodes.jsonl").write_text(
        "".join(canonical(e) + "\n" for e in episodes), encoding="utf-8")
    (out / "certificate.json").write_text(canonical(cert) + "\n", encoding="utf-8")
    receipt["artifacts"] = {name: sha256_file(out / name)
                            for name in ("episodes.jsonl", "certificate.json", "standalone.lean")}
    (out / "manifest.json").write_text(canonical(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify(args.seed, args.pairs, args.out)
    print(canonical({k: report[k] for k in
                     ("schema", "pairs", "worlds", "scorecard",
                      "independent_compiled_scripts", "all_episodes_replayed")}))


if __name__ == "__main__":
    main()
