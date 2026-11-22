"""Verify two original MathCan TRAIN proof-cost contrasts in the gym's Lean."""

import argparse
import gzip
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.mathcan import MathCanReplay
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget

from configs.replay_ppt_mathcan_witness import ALLOWED_AXIOMS, MODULE, SCRIPT, THEOREM, TRAIN_COMMIT, audit_axioms


NAMESPACE = "MathCan.FiniteHorizonGap"
PREFIX_MODULE = "PPTMathCanPrefix"
TAKE_LEMMAS = "List.take_of_length_le hx, List.take_of_length_le hy"
ROUTES = {THEOREM: (("omega",), SCRIPT),
          "full_horizon_recovers": (
              (f"simpa [{NAMESPACE}.prefixEquivalent, {TAKE_LEMMAS}] using h",),
              (f"dsimp [{NAMESPACE}.prefixEquivalent] at h", f"simpa [{TAKE_LEMMAS}] using h"),
          )}
AXIOM_CONTRACTS = {"standard": ALLOWED_AXIOMS, "without_classical_choice": ALLOWED_AXIOMS - {"Classical.choice"},
                   "axiom_free": frozenset()}


def declaration(original, theorem):
    name = f"theorem {theorem} "
    start = original.index(name)
    end = original.index(" := by", start)
    header = original[start:end]
    binders, target = re.fullmatch(r"(.+\))\s*:\s*(.+)", header[len(name):], flags=re.DOTALL).groups()
    statement = re.sub(r"\bprefixEquivalent\b", f"{NAMESPACE}.prefixEquivalent", f"∀ {binders}, {target}")
    before = original[:start].rstrip()
    # Remove the source declaration's doc comment before closing the namespace.
    if before.endswith("-/"):
        before = before[:before.rindex("/--")].rstrip()
    return before + "\n", header, statement


def invoke_lean(directory, env, *args):
    result = subprocess.run([str(LEAN_EXE), *args], cwd=directory, env=env, timeout=60,
                            text=True, encoding="utf-8", capture_output=True)
    if result.returncode:
        raise ValueError(result.stdout + result.stderr)
    return result.stdout


def is_archived_proof(replay, key, script):
    for i, action in enumerate(script):
        if action not in replay.states[key]:
            return False
        status, key = replay.step(key, action)
        if i + 1 == len(script):
            return status == "archived_closure"
        if status != "accepted_nonterminal":
            return False
    return False


def verify_route(repl, statement, script):
    repl.reset()
    state = repl.init(statement)
    events = []
    for action in script:
        before = public_state(repl, state)["goals"]
        if not 1 <= len(before) <= 2:
            raise ValueError("the registered MathCan proof left the two-slot envelope")
        reply = repl.run(state["sid"], action)
        if not reply["ok"]:
            raise ValueError(f"Lean rejected {action}: {reply['error']}")
        state = reply["state"]
        events.append({"goals": before, "action": action, "after": public_state(repl, state)["goals"]})
    kernel = repl.kernel(state["sid"])
    if not kernel["ok"] or not set(kernel["axioms"]) <= ALLOWED_AXIOMS:
        raise ValueError("the registered MathCan proof failed its kernel check")
    return {"checks": len(events), "events": events, "axioms": sorted(kernel["axioms"])}


def eligible_costs(routes):
    return {name: [route["checks"] if set(route["axioms"]) <= allowed else None for route in routes]
            for name, allowed in AXIOM_CONTRACTS.items()}


def verify(archive, original):
    replay = MathCanReplay(archive)
    source = original.read_text(encoding="utf-8")
    source_hash = sha256_file(original)
    rows = []
    for theorem, scripts in ROUTES.items():
        start = (MODULE, theorem, ())
        if start not in replay.starts or any(row["source_sha256"] != source_hash for row in replay.states[start].values()):
            raise ValueError(f"{theorem} does not match the audited original TRAIN source")
        before, header, statement = declaration(source, theorem)
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            prelude = directory / f"{PREFIX_MODULE}.lean"
            prelude.write_text(f"{before}end {NAMESPACE}\n", encoding="utf-8")
            env = lean_env()
            invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", prelude.name)
            env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
            with LeanRepl(imports=(PREFIX_MODULE,), cwd=directory, env=env) as repl:
                if repl.version != LEAN_VERSION:
                    raise ValueError("the MathCan replay requires the pinned gym toolchain")
                rejected = repl.run(repl.init(statement)["sid"], "rfl")
                if rejected["ok"]:
                    raise ValueError("the registered control tactic unexpectedly succeeded")
                routes = []
                for script in scripts:
                    route = verify_route(repl, statement, script)
                    independent = directory / "Standalone.lean"
                    proof = "\n".join("  " + tactic for tactic in script)
                    independent.write_text(f"{before}{header} := by\n{proof}\n\n"
                                           f"#print axioms {theorem}\nend {NAMESPACE}\n", encoding="utf-8")
                    if audit_axioms(invoke_lean(directory, env, independent.name), theorem) != route["axioms"]:
                        raise ValueError("live and standalone Lean disagree on the MathCan axiom inventory")
                    route.update(standalone_sha256=sha256_file(independent), archived_proof=is_archived_proof(replay, start, script))
                    routes.append(route)
        if (routes[0]["events"][0]["goals"] != routes[1]["events"][0]["goals"]
                or routes[0]["checks"] >= routes[1]["checks"]):
            raise ValueError("expected faster and slower proofs from the same MathCan starting state")
        rows.append({"root": digest(start), "theorem": theorem, "statement": statement, "routes": routes,
                     "eligible_checks": eligible_costs(routes),
                     "rejected_control": {"action": "rfl", "error": rejected["error"]}})
    return {"schema": "ppt.mathcan-native-proof-costs.v1", "training_only": True, "source_commit": TRAIN_COMMIT,
            "module": MODULE, "source_sha256": source_hash, "archive_sha256": sha256_file(archive), "lean": LEAN_VERSION,
            "repl_sha256": sha256_file(REPL_EXE), "lean_sha256": sha256_file(LEAN_EXE), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "original", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        raw = (canonical(verify(args.archive, args.original)) + "\n").encode("utf-8")
        with args.output.open("xb") as output:
            output.write(gzip.compress(raw, mtime=0) if args.output.suffix == ".gz" else raw)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
