"""Seal new equality and Horn roots before scoring the frozen residual policies."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import re

from configs.freeze_ppt_native_progress import LEAN_EXE, REPL_EXE
from configs.freeze_ppt_residual_search import SCHEMA as PARENT_SCHEMA, task_clusters
from configs.freeze_ppt_rollback_diagnostics import Payload
from configs.freeze_ppt_scope_recovery import PREFIX_MODULE, native_scope, prefix_text
from configs.freeze_ppt_successor_curriculum import proof_script
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_native_progress import verify_independent
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt import fragment
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.ram_limits import TreeBudget


REGISTRY = "ppt.residual-transfer-registry.v1"
NATIVE = "ppt.residual-transfer-native.v1"
SEED = 1211
REPAIR_SEED = 1217
COUNTS = {"fresh_bpd_directed": 32, "fresh_bpd_equality": 16, "fresh_horn": 16}
FIT_SHA = "e91be9d89b31025f315fc45f261440ab806b17a0d1246efbb8dbd415ff09a478"
STOP_SHA = "07a72e854cdba9d9a93783077daba12f736aa2c943f16bf2f9df0ec27022817a"
KNOWN = ("development-cohort.json", "replication-cohort.json", "conjunction-pilot-cohort.json",
         "root-adaptation-cohort.json", "ppt-successor-curriculum-1009.json.gz",
         "ppt-premise-budget-curriculum-1021.json.gz", "ppt-priced-fragments-1039.json.gz",
         "ppt-native-feedback-expanded-1009.json.gz")
SIMPLE = re.compile(r"^∀ \(([^()]+) : (Prop|Nat)\) ((?:\([^()]+\) *)+),\s*(.+)$", re.DOTALL)
BINDER = re.compile(r"\(([^():]+) : ([^()]+)\)")


def graph_hash(statement):
    """Alpha-invariant structural fingerprint for registered simple local graphs."""
    match = SIMPLE.fullmatch(" ".join(statement.split()))
    if match is None:
        return None
    names, domain, premises, target = match.groups()
    atoms = {name: pos for pos, name in enumerate(names.split())}
    facts, rules = [], []
    try:
        if domain == "Prop":
            for _, clause in BINDER.findall(premises):
                parts = [atoms[name.strip()] for name in clause.split("→")]
                if len(parts) == 1:
                    facts.extend(parts)
                else:
                    rules.append((parts[:-1], parts[-1]))
            goals = [atoms[name.strip()] for name in target.split("∧")]
        else:
            start, end = (atoms[name.strip()] for name in target.split("="))
            facts, goals = [start], [end]
            for _, clause in BINDER.findall(premises):
                left, right = (atoms[name.strip()] for name in clause.split("="))
                rules.extend((([left], right), ([right], left)))
    except (KeyError, ValueError):
        return None
    graph = fragment.Theorem("", "", 0, 0, len(atoms), facts, rules, goals)
    return digest([domain, fragment.canonical_hash(graph)])


def seen_statements(bank):
    files = [bank, *(bank.parent / name for name in KNOWN)]
    seen, pins = set(), {}
    for path in files:
        content = read_receipt(path) if path.suffix == ".gz" else json.loads(path.read_bytes())
        pins[path.name] = sha256_file(path)
        for key in ("rows", "cases", "source", "held", "training"):
            for row in content.get(key, []):
                if isinstance(row, dict):
                    statement = row.get("statement")
                    if statement is None and "binders" in row and "target" in row:
                        statement = f"∀ {row['binders']}, {row['target']}"
                    if isinstance(statement, str) and (fingerprint := graph_hash(statement)):
                        seen.add(fingerprint)
    return seen, pins


def bpd_shaped(seed, source, *, compatible=False):
    rng = random.Random(seed)
    depth = rng.randint(3, 7)
    equality = source == "fresh_bpd_equality"
    edges = [(i, i + 1) for i in range(depth)]
    n = depth + 1
    # Give each spine successor a competing producer with a fresh premise.
    # Additional links decorate the unreachable side of the proof graph.
    for position in range(1, depth + 1):
        first = n
        length = rng.randint(1, 4)
        edges.extend((i, i + 1) for i in range(n, n + length - 1))
        if equality or rng.random() < .8:
            edges.append((n + length - 1, position))
        n += length
        if rng.random() < .5:
            edges.append((first, n))
            n += 1
    for _ in range(rng.randint(1, 5)):
        src = rng.randrange(depth + 1)
        edges.append((src, n))
        n += 1
    vertices = [f"v{rng.getrandbits(44):011x}" for _ in range(n)]
    if len(set(vertices)) != len(vertices):
        raise ValueError("a prospective root repeated a local atom")
    rule_names = ([f"r{index}" for index in rng.sample(range(1, 9000), len(edges))]
                  if compatible else [f"j{rng.getrandbits(36):09x}" for _ in edges])
    rules = list(zip(edges, rule_names, strict=True))
    if len({name for _, name in rules}) != len(rules):
        raise ValueError("a prospective root repeated a rule name")
    rng.shuffle(rules)
    binder_order = vertices.copy()
    rng.shuffle(binder_order)
    domain, link = ("Nat", "=") if equality else ("Prop", "→")
    facts = []
    if not equality:
        fact = f"h{rng.randrange(1, 9000)}" if compatible else f"base{rng.getrandbits(28):07x}"
        facts.append(f"({fact} : {vertices[0]})")
    bindings = [f"({' '.join(binder_order)} : {domain})", *facts,
                *(f"({name} : {vertices[a]} {link} {vertices[b]})" for (a, b), name in rules)]
    spine = [next(name for edge, name in rules if edge == (i, i + 1)) for i in range(depth)]
    # The native reference follows the live spine; neither its order nor its
    # final fact binder is supplied to the policy.
    if equality:
        proof = tuple(f"rw [{name}]" for name in spine)
        goal = f"{vertices[0]} = {vertices[depth]}"
    else:
        proof = (*[f"apply {name}" for name in reversed(spine)], "apply " + facts[0][1:].split()[0])
        goal = vertices[depth]
    bindings = " ".join(bindings)
    statement = f"∀ {bindings}, {goal}"
    return Payload(source, digest([REGISTRY, source, seed, statement]), "dev", bindings, (), goal, proof, ())


def horn(seed, slot):
    family = ("chain-deep", "tree-deep", "alt-cycle")[slot % 3]
    theorem = fragment.generate(seed, 2 + seed % 3, family, conj_p=.30, mirror_p=.70)
    sizes = fragment.reference_sizes(theorem)
    length = sum(sizes[goal] for goal in theorem.goal) + int(len(theorem.goal) > 1)
    if (not 3 <= length <= 8
            or len(fragment.reference_legal(theorem, theorem.goal[0])) < 2):
        return None
    return Payload("fresh_horn", digest([REGISTRY, seed, fragment.statement(theorem)]), "dev",
                   fragment.binders(theorem), (), fragment.target(theorem), tuple(proof_script(theorem, sizes)), ())


def register(args):
    if sha256_file(args.fit) != FIT_SHA or sha256_file(args.stopping) != STOP_SHA:
        raise ValueError("the prospective transfer has different frozen model inputs")
    seen, exclusions = seen_statements(args.bank)
    regime = REPAIR_SEED if args.repair_of else SEED
    if args.repair_of:
        first = read_receipt(args.repair_of)
        if first["schema"] != REGISTRY or first["seed"] != SEED or first["fit"] != FIT_SHA:
            raise ValueError("the menu repair needs the original sealed cohort")
        seen.update(row["signature"] for row in first["rows"])
        exclusions[args.repair_of.name] = sha256_file(args.repair_of)
    selected = []
    for source, count in COUNTS.items():
        for slot in range(count):
            for attempt in range(256):
                seed = int(digest([REGISTRY, regime, source, slot, attempt])[:16], 16)
                payload = (horn(seed, slot) if source == "fresh_horn" else
                           bpd_shaped(seed, source, compatible=args.repair_of is not None))
                if payload is None:
                    continue
                signature = graph_hash(f"∀ {payload.binders}, {payload.target}")
                if signature is None or signature in seen:
                    continue
                seen.add(signature)
                selected.append({"seed": seed, "signature": signature, "payload": asdict(payload)})
                break
            else:
                raise ValueError("a prospective cell has no new public graph")
    write_receipt(args.registry, {"schema": REGISTRY, "seed": regime, "counts": COUNTS, "excluded_inputs": exclusions,
                                  "repair_of": sha256_file(args.repair_of) if args.repair_of else None,
                                  "training_bank": sha256_file(args.bank), "fit": FIT_SHA, "stopping": STOP_SHA,
                                  "mathcan": sha256_file(args.mathcan), "lean": sha256_file(LEAN_EXE),
                                  "repl": sha256_file(REPL_EXE), "collector": sha256_file(Path(__file__)),
                                  "horn": sha256_file(Path(fragment.__file__)), "rows": selected})


def freeze(args, tree):
    registry = read_receipt(args.registry)
    if (registry["schema"] != REGISTRY or registry["counts"] != COUNTS or registry["fit"] != FIT_SHA
            or registry["collector"] != sha256_file(Path(__file__)) or registry["mathcan"] != sha256_file(args.mathcan)
            or registry["lean"] != sha256_file(LEAN_EXE) or registry["repl"] != sha256_file(REPL_EXE)):
        raise ValueError("the prospective native inputs changed after registration")
    prelude = prefix_text(args.mathcan)
    payloads = [Payload(**row["payload"]) for row in registry["rows"]]
    if Counter(row.source for row in payloads) != COUNTS or len({row.root for row in payloads}) != sum(COUNTS.values()):
        raise ValueError("the sealed population contains missing or duplicated roots")

    def collect(directory, env):
        rows = []
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for payload in payloads:
                row, tasks = task_clusters(repl, payload)
                if not tasks:
                    raise ValueError("a registered root has no distinct residual pair")
                if registry.get("repair_of") and not any(reaches_two(task["target"]) for task in tasks):
                    raise ValueError("a repaired root has no witnessed solution under the registered menu")
                rows.append({**row, "tasks": tasks})
                if len(rows) % 8 == 0:
                    print("witnessed prospective roots", len(rows), flush=True)
        independent = verify_independent(directory, env, (PREFIX_MODULE,),
                                          [(row, row["reference"]) for row in rows])
        return rows, independent

    rows, independent = native_scope(prelude, collect)
    write_receipt(args.output, {"schema": NATIVE, "source_schema": PARENT_SCHEMA, "registry": sha256_file(args.registry),
                                "prelude": digest(prelude), "training_bank": registry["training_bank"],
                                "lean": registry["lean"], "repl": registry["repl"], "rows": rows,
                                "independent": independent, "memory": tree.stats()})


def reaches_two(node):
    return any(edge["status"] == "closed" or any(child["status"] == "closed"
               for child in edge.get("child", {}).get("actions", {}).values()) for edge in node["actions"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "freeze"))
    for name in ("bank", "mathcan", "fit", "stopping", "registry", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--repair-of", type=Path, help="seal a disjoint cohort under the unchanged typed tactic grammar")
    args = parser.parse_args()
    if args.phase == "register":
        register(args)
    else:
        tree = TreeBudget(2, 1)
        try:
            freeze(args, tree)
        finally:
            tree.close()


if __name__ == "__main__":
    main()
