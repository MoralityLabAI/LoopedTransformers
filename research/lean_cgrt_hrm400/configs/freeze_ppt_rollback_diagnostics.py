"""Verify paid Lean diagnostics that survive rollback only as observations."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
from itertools import product
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rollback_diagnostics import DIAGNOSTIC_COST, DiagnosticEpisode, NAMES, PROBES, SCHEMA

from configs.check_ppt_mathcan_native_costs import (
    MODULE, NAMESPACE, PREFIX_MODULE, THEOREM, declaration, invoke_lean,
)
from configs.run_ppt_two_slot import request_state


REVEAL = re.compile(r"proposition ([01]) = 2 is false$")
OPERATIONS = {
    "xor": ("probeLeft + probeRight - 2 * (probeLeft * probeRight)", int.__xor__),
    "and": ("probeLeft * probeRight", int.__and__),
    "or": ("max probeLeft probeRight", int.__or__),
    "xnor": ("1 - (probeLeft + probeRight - 2 * (probeLeft * probeRight))", lambda a, b: 1 - (a ^ b)),
}
SOURCE_COUNTS = {"bpd_lg02": (6, 8), "loopt_horn": (6, 8), "evalmath_lg01": (2, 1)}
MATHCAN_CERTIFICATES = Path(__file__).resolve().parents[1] / "results/ppt_hybrid_rl/ppt-two-slot-mathcan-native-costs.json.gz"


@dataclass(frozen=True)
class Payload:
    source: str
    root: str
    split: str
    binders: str
    names: tuple[str, ...]
    target: str
    proof: tuple[str, ...]
    axioms: tuple[str, ...]


def native_payloads(cohort, audit_path, ledger_path, *, source_counts=None, skip=None):
    audit = json.loads(audit_path.read_bytes())
    trial = audit["trials"]["hybrid_rl"]
    if sha256_file(cohort) != audit["cohort_sha256"] or sha256_file(ledger_path) != trial["session_sha256"]:
        raise ValueError("expected the audited mixed-source training ledger")
    cases = {case.root: case for case in load(cohort)["train"]}
    pools = defaultdict(dict)
    with ram_store.open_store(ledger_path) as db:
        if ram_store.read_manifest(db) != trial["manifest"]:
            raise ValueError("the source ledger changed its manifest")
        for batch in ram_store.ledger(db):
            if batch["phase"] != "train":
                continue
            for episode in batch["episodes"]:
                if not episode["verified_closed"] or not episode["within_cpu_budget"]:
                    continue
                case = cases[episode["root"]]
                cert = episode["certificate"]
                if (not episode["live_kernel"] or not cert["passed"] or cert["axioms"]
                        or digest(cert["source"]) != cert["source_digest"]):
                    raise ValueError("the source proof lost its independent certificate")
                text = cert["source"].splitlines()
                if text[:1] != [f"theorem {case.tid} : {case.statement} := by"] or not text[1].startswith("  intro "):
                    raise ValueError("the original source statement or binder order changed")
                if tuple(line.strip() for line in text[2:-1]) != tuple(episode["accepted_script"]):
                    raise ValueError("the archived native tactics do not match the standalone certificate")
                pools[case.source].setdefault(case.cluster, (case, tuple(text[1].split()[1:]),
                                                             tuple(episode["accepted_script"])))
    counts = SOURCE_COUNTS if source_counts is None else source_counts
    for source, (train, dev) in counts.items():
        pool = [pools[source][key] for key in sorted(pools[source])]
        if skip:
            pool = pool[skip.get(source, 0):]
        if len(pool) < train + dev:
            raise ValueError(f"too few disjoint original proof clusters for {source}")
        for i, (case, names, proof) in enumerate(pool[:train + dev]):
            yield Payload(source, case.root, "train" if i < train else "dev",
                          case.binders, names, case.target, proof, ())


def mathcan_payloads(archive, original):
    replay = MathCanReplay(archive)
    source = original.read_text(encoding="utf-8")
    receipts = json.loads(gzip.decompress(MATHCAN_CERTIFICATES.read_bytes()))
    if (receipts["schema"] != "ppt.mathcan-native-proof-costs.v1" or receipts["lean"] != LEAN_VERSION
            or not receipts["training_only"] or receipts["archive_sha256"] != sha256_file(archive)
            or receipts["source_sha256"] != sha256_file(original)):
        raise ValueError("the MathCan proof certificates do not match the original audited source")
    payloads = []
    entries = ((THEOREM, "train", ("delta", "epsilon")),
               ("full_horizon_recovers", "dev", ("α", "xs", "ys", "T", "hx", "hy", "h")))
    for theorem, split, names in entries:
        key = (MODULE, theorem, ())
        if (key not in replay.starts
                or {row["source_sha256"] for row in replay.states[key].values()} != {sha256_file(original)}):
            raise ValueError("the original MathCan theorem source is not in the audited training archive")
        prelude, header, statement = declaration(source, theorem)
        binders, target = re.fullmatch(r"(.+\))\s*:\s*(.+)", header[len(f"theorem {theorem} "):], re.DOTALL).groups()
        binders = re.sub(r"\bprefixEquivalent\b", f"{NAMESPACE}.prefixEquivalent", binders)
        receipt, = (row for row in receipts["rows"] if row["theorem"] == theorem)
        if receipt["statement"] != statement:
            raise ValueError("the certified MathCan declaration changed")
        proof = tuple(event["action"] for event in receipt["routes"][0]["events"])
        payloads.append(Payload("mathcan_ram08_audited_train", digest(key), split, binders, names,
                                target, proof, tuple(receipt["routes"][0]["axioms"])))
    return payloads, prelude + f"end {NAMESPACE}\n"


def root_statement(payload, bits, expression):
    left, right = (f"let {name} : Nat := {bit}; " for name, bit in zip(NAMES, bits, strict=True))
    return f"∀ {payload.binders}, {left}{right}∃ n : Nat, n = {expression} ∧ ({payload.target})"


def probe(repl, root, index):
    before = public_state(repl, root)
    response = repl.run(root["sid"], PROBES[index])
    match = REVEAL.search(response.get("error", ""))
    if response["ok"] or not match or "state" in response:
        raise ValueError("the paid diagnostic did not yield just one rejected-tactic observation")
    restored = request_state(repl.focus(root["sid"], 0))
    if public_state(repl, restored) != before:
        raise ValueError("the root observation changed after the failed probe")
    return restored, {"action": PROBES[index], "error": response["error"], "observed_value": int(match[1]),
                      "restored_sha256": digest(before)}


def witness_trial(repl, root, value, payload):
    actions = (f"refine ⟨{value}, ?_⟩", "constructor", "decide", *payload.proof)
    state = root
    for spent, action in enumerate(actions, 1):
        response = repl.run(state["sid"], action)
        if not response["ok"]:
            if spent != 3:
                raise ValueError(f"a witnessed payload proof failed at {action}: {response['error']}")
            return {"witness": value, "passed": False, "checks": spent, "error": response["error"]}
        state = response["state"]
    kernel = repl.kernel(state["sid"])
    if state["goals"] or not kernel["ok"] or set(kernel["axioms"]) != set(payload.axioms):
        targets = [goal["target"] for goal in public_state(repl, state)["goals"]]
        raise ValueError(f"uncertified {payload.source} payload {payload.root}: "
                         f"remaining={targets}, kernel={kernel}")
    return {"witness": value, "passed": True, "checks": len(actions), "axioms": sorted(kernel["axioms"]),
            "actions": actions}


def freeze_world(repl, payload, op, bits):
    expression, oracle = OPERATIONS[op]
    statement = root_statement(payload, bits, expression)
    repl.reset()
    root = repl.init(statement)
    view = public_state(repl, root)
    original = root
    probes = []
    for index, bit in enumerate(bits):
        root, observed = probe(repl, root, index)
        if observed["observed_value"] != bit:
            raise ValueError("Lean reported the wrong unfolded diagnostic bit")
        probes.append(observed)
    paths = [witness_trial(repl, root, value, payload) for value in (0, 1)]
    winning = oracle(*bits)
    if [path["passed"] for path in paths] != [winning == value for value in (0, 1)]:
        raise ValueError("the independent operator oracle disagreed with Lean")
    selected = paths[winning]
    direct = witness_trial(repl, original, winning, payload)
    if direct != selected:
        raise ValueError("the proof after rollback differs from the direct proof")
    cluster = digest([payload.root, op])
    tid = f"rollback_probe_{cluster[:12]}_{bits[0]}{bits[1]}"
    intro = " ".join((*payload.names, *NAMES))
    source = (f"theorem {tid} : {statement} := by\n  intro {intro}\n"
              + "".join(f"  {action}\n" for action in selected["actions"])
              + f"#print axioms {tid}\n")
    return {"tid": tid, "source": payload.source, "payload_root": payload.root, "split": payload.split,
            "cluster": cluster, "operation": op, "world": list(bits), "public": view,
            "probe_receipts": probes, "witness_trials": paths, "winning_witness": winning,
            "proof_budget": selected["checks"] + 2, "certificate_source": source}


def verify_layout(repl, payload, op):
    rows = [freeze_world(repl, payload, op, bits) for bits in product((0, 1), repeat=2)]
    if any(row["public"] != rows[0]["public"] or row["proof_budget"] != rows[0]["proof_budget"] for row in rows):
        raise ValueError("the visible starting observation exposed the hidden valuation")
    return rows


def certify_batches(rows, directory, env):
    from configs.replay_ppt_mathcan_witness import audit_axioms

    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        path = directory / "Standalone.lean"
        path.write_text(f"import {PREFIX_MODULE}\n\n" + "\n".join(r["certificate_source"] for r in batch),
                        encoding="utf-8")
        output = invoke_lean(directory, env, path.name)
        for row in batch:
            axiom_report = [
                line.replace(f"'{row['tid']}'", f"'{NAMESPACE}.{row['tid']}'", 1)
                for line in output.splitlines() if line.startswith(f"'{row['tid']}' ")
            ]
            inventory = audit_axioms("\n".join(axiom_report), row["tid"])
            live = row["witness_trials"][row["winning_witness"]]["axioms"]
            if inventory != live:
                raise ValueError(f"{row['tid']} axiom inventory mismatch: live {live}, standalone {inventory}")
            row["standalone_axioms"] = inventory
            row["certificate_source_digest"] = digest(row["certificate_source"])


def policy_bound(layout, budget, available=(0, 1)):
    """Optimize certified proofs minus 1/20 of a proof per diagnostic check."""
    wins = Counter(row["winning_witness"] for row in layout)
    witness = max((0, 1), key=wins.__getitem__)
    best = wins[witness], 0, ("commit", witness)
    if budget:
        for index in available:
            groups = defaultdict(list)
            for row in layout:
                groups[row["probe_receipts"][index]["observed_value"]].append(row)
            rest = tuple(i for i in available if i != index)
            continuations = [policy_bound(group, budget - 1, rest) for group in groups.values()]
            candidate = (sum(proofs for proofs, _, _ in continuations),
                         len(layout) + sum(checks for _, checks, _ in continuations), ("probe", index))
            if candidate[0] - DIAGNOSTIC_COST * candidate[1] > best[0] - DIAGNOSTIC_COST * best[1]:
                best = candidate
    return best


def replay_bound(layout, budget):
    expected_proofs, expected_probes, _ = policy_bound(layout, budget)
    proofs, probes = 0, 0
    for row in layout:
        episode = DiagnosticEpisode(row, budget)
        posterior = layout
        remaining = budget
        observation = episode.observe()
        root = observation.proof_state
        while True:
            _, _, (action, choice) = policy_bound(posterior, remaining, observation.available_probes)
            if action == "commit":
                outcome = episode.commit(choice)
                proofs += outcome.certificate_passed
                probes += outcome.diagnostics
                break
            observation = episode.diagnose(choice)
            if observation.proof_state != root:
                raise ValueError("the diagnostic changed the restored public proof state")
            index, value = observation.latest_diagnostic
            posterior = [case for case in posterior if case["probe_receipts"][index]["observed_value"] == value]
            remaining -= 1
    if (proofs, probes) != (expected_proofs, expected_probes):
        raise ValueError("the compressed episode replay diverged from the information bound")
    return proofs, probes


def information_bounds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["source"], row["cluster"])].append(row)
    totals = defaultdict(Counter)
    for (split, source, _), layout in groups.items():
        if len(layout) != 4:
            raise ValueError("a layout lacks one or more valuations")
        bounds = totals[split, source]
        bounds["worlds"] += len(layout)
        bounds["layouts"] += 1
        for observed, label in (((), "root_only"), ((1,), "last_probe_only"), ((0, 1), "two_probe_history")):
            partitions = defaultdict(Counter)
            for row in layout:
                key = tuple(row["probe_receipts"][i]["observed_value"] for i in observed)
                partitions[key][row["winning_witness"]] += 1
            bounds[label] += sum(max(counts.values()) for counts in partitions.values())
        for budget, label in ((1, "one_diagnostic"), (2, "two_diagnostics")):
            proofs, checks = replay_bound(layout, budget)
            bounds[f"{label}_proofs"] += proofs
            bounds[f"{label}_checks"] += checks
        if bounds["two_probe_history"] != bounds["worlds"]:
            raise ValueError("the full history failed to separate a certified witness")
    return {split: {source: dict(values) for (group, source), values in totals.items() if group == split}
            for split in ("train", "dev")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "audit", "ledger", "mathcan_archive", "mathcan_source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        math, prelude = mathcan_payloads(args.mathcan_archive, args.mathcan_source)
        payloads = [*native_payloads(args.cohort, args.audit, args.ledger), *math]
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            prefix = directory / f"{PREFIX_MODULE}.lean"
            prefix.write_text(prelude, encoding="utf-8")
            env = lean_env()
            invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", prefix.name)
            env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
            with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
                if repl.version != LEAN_VERSION:
                    raise ValueError("the diagnostic replay requires the pinned Lean toolchain")
                rows = []
                for payload in payloads:
                    first = int(payload.root[:8], 16) % 4
                    names = tuple(OPERATIONS)
                    for op in (names[first], names[(first + 1) % len(names)]):
                        rows.extend(verify_layout(repl, payload, op))
            certify_batches(rows, directory, env)
        inputs = {name: sha256_file(getattr(args, name))
                  for name in ("cohort", "audit", "ledger", "mathcan_archive", "mathcan_source")}
        inputs["mathcan_certificates"] = sha256_file(MATHCAN_CERTIFICATES)
        report = {"schema": SCHEMA, "lean": LEAN_VERSION, "server_sha256": sha256_file(REPL_EXE),
                  "verifier_sha256": sha256_file(LEAN_EXE), "source_hashes": inputs,
                  "prelude_sha256": digest(prelude), "information_bounds": information_bounds(rows),
                  "rows": rows, "memory": tree.stats()}
        with args.output.open("xb") as handle:
            handle.write(gzip.compress((canonical(report) + "\n").encode(), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
