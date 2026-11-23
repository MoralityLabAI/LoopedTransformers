"""Witness Lean proof choices after a tentative alias update is kept or rolled back."""

import argparse
from collections import defaultdict
from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from configs.check_ppt_mathcan_native_costs import NAMESPACE, PREFIX_MODULE, declaration, invoke_lean
from configs.freeze_ppt_rollback_diagnostics import OPERATIONS, Payload, certify_batches
from configs.freeze_ppt_rollback_length_transfer import diagnostic, native_commit
from configs.run_ppt_mathcan_progress import read_receipt
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.rollback_diagnostics import NAMES, SCHEMA as ARCHIVE_SCHEMA


SCHEMA = "ppt.scope-recovery-native.v1"
ALIASES = ("baseLeft", "baseRight", "tentative", "keepScope")
ROOTS = {"train": 15, "dev": 18}


def payloads(archive):
    originals = {}
    for row in archive["rows"]:
        certificate = row["certificate_source"]
        if digest(certificate) != row["certificate_source_digest"]:
            raise ValueError("a rollback payload certificate changed")
        header, script = certificate.split(" := by\n", 1)
        prefix = f"theorem {row['tid']} : ∀ "
        if not header.startswith(prefix):
            raise ValueError("the archived proof has a different declaration wrapper")
        binders, suffix = header[len(prefix):].split(f", let {NAMES[0]} : Nat :=", 1)
        expression, _ = OPERATIONS[row["operation"]]
        start = f"∃ n : Nat, n = {expression} ∧ ("
        target = suffix.split(start, 1)[1].removesuffix(")")
        intro = script.splitlines()[0].split()
        if intro[0] != "intro" or tuple(intro[-len(NAMES):]) != NAMES:
            raise ValueError("the archived payload binder order changed")
        witness = row["witness_trials"][row["winning_witness"]]
        payload = Payload(row["source"], row["payload_root"], row["split"], binders,
                          tuple(intro[1:-len(NAMES)]), target, tuple(witness["actions"][3:]), tuple(witness["axioms"]))
        previous = originals.setdefault(payload.root, payload)
        if previous != payload:
            raise ValueError("the same original proof has inconsistent archived payloads")
    if any(sum(payload.split == split for payload in originals.values()) != count for split, count in ROOTS.items()):
        raise ValueError("the original payload split is incomplete")
    return sorted(originals.values(), key=lambda payload: (payload.split, payload.source, payload.root))


def statement_for(payload, bits, edited):
    left, right, tentative, scope = ALIASES
    terms = [left, right]
    terms[edited] = f"(if {scope} = 1 then {tentative} else {terms[edited]})"
    formula = f"({' + '.join(terms)}) % 2"
    binders = "".join(f"let {name} : Nat := {bit}; " for name, bit in zip(ALIASES, bits, strict=True))
    return f"∀ {payload.binders}, {binders}∃ n : Nat, ({payload.target}) ∧ n = {formula}"


def verify(repl, payload, edited):
    cluster = digest([SCHEMA, payload.root, edited])
    shared = None
    rows = []
    for bits in product((0, 1), repeat=len(ALIASES)):
        statement = statement_for(payload, bits, edited)
        repl.reset()
        start = repl.init(statement)
        public = public_state(repl, start)
        if shared is None:
            shared = public
        elif public != shared:
            raise ValueError("the initial Lean observation exposed a hidden bit")
        state = start
        observations = []
        for index, alias in enumerate(ALIASES):
            state, observation = diagnostic(repl, state, alias)
            if observation["value"] != bits[index]:
                raise ValueError("a native scope diagnostic disagrees with its independent bit")
            observations.append(observation)
        paths = [native_commit(repl, state, payload.proof, value, payload.axioms) for value in (0, 1)]
        winning = bits[1 - edited] ^ (bits[2] if bits[3] else bits[edited])
        if [path["passed"] for path in paths] != [winning == value for value in (0, 1)]:
            raise ValueError("the independent scope oracle disagrees with native Lean")
        if paths[1 - winning]["checks"] <= 3:
            raise ValueError("a failed commit would leave time for an unregistered retry")
        tid = f"scope_{cluster[:12]}_{''.join(map(str, bits))}"
        proof = paths[winning]
        certificate = (f"theorem {tid} : {statement} := by\n"
                       f"  intro {' '.join((*payload.names, *ALIASES))}\n"
                       + "".join(f"  {action}\n" for action in proof["actions"])
                       + f"#print axioms {tid}\n")
        rows.append({"tid": tid, "cluster": cluster, "payload_root": payload.root, "source": payload.source,
                     "split": payload.split, "edited": edited, "world": bits, "public": public,
                     "observations": observations, "witness_trials": paths, "winning_witness": winning,
                     "certificate_source": certificate})
    return rows


def prefix_text(mathcan):
    prelude, _, _ = declaration(mathcan.read_text(encoding="utf-8"), "full_horizon_recovers")
    return prelude + f"end {NAMESPACE}\n"


def native_scope(prelude, operation):
    with TemporaryDirectory() as scratch:
        directory = Path(scratch)
        prefix = directory / f"{PREFIX_MODULE}.lean"
        prefix.write_text(prelude, encoding="utf-8")
        env = lean_env()
        invoke_lean(directory, env, "-o", f"{PREFIX_MODULE}.olean", prefix.name)
        env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
        return operation(directory, env)


def freeze(args, tree):
    archive = read_receipt(args.archive)
    if (archive["schema"] != ARCHIVE_SCHEMA or archive["lean"] != LEAN_VERSION
            or archive["verifier_sha256"] != sha256_file(LEAN_EXE) or archive["server_sha256"] != sha256_file(REPL_EXE)):
        raise ValueError("the scope fixture requires the original pinned native rollback archive and toolchain")
    prelude = prefix_text(args.mathcan)
    if digest(prelude) != archive["prelude_sha256"]:
        raise ValueError("the original MathCan namespace changed")
    sources = payloads(archive)

    def collect(directory, env):
        rows = []
        counters = defaultdict(int)
        with LeanRepl(imports=(PREFIX_MODULE,), env=env, cwd=directory) as repl:
            for payload in sources:
                key = payload.split, payload.source
                edited = counters[key] % 2
                counters[key] += 1
                rows.extend(verify(repl, payload, edited))
                print("witnessed scope root", payload.split, payload.source, len(rows), flush=True)
        certify_batches(rows, directory, env)
        return rows

    rows = native_scope(prelude, collect)
    write_receipt(args.bank, {"schema": SCHEMA, "archive": sha256_file(args.archive), "prelude": digest(prelude),
                             "lean": sha256_file(LEAN_EXE), "repl": sha256_file(REPL_EXE),
                             "rows": rows, "memory": tree.stats()})
    print("compiled", len(rows), "distinct native certificates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "mathcan", "bank"):
        parser.add_argument(name, type=Path)
    tree = TreeBudget(2, 1)
    try:
        freeze(parser.parse_args(), tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
