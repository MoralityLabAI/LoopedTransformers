"""Register original theorem statements and a bounded live Lean tactic vocabulary."""

import argparse
from collections import Counter, deque
from contextlib import contextmanager
from itertools import islice
import os
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from configs.check_ppt_mathcan_native_costs import invoke_lean
from configs.run_ppt_rollback_diagnostics import write_receipt
from lcgrt.config import LEAN_EXE, LEAN_VERSION, REPL_EXE, lean_env
from lcgrt.lean_repl import LeanRepl
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import REPO
from ppt_gym.live_lean import public_state
from ppt_gym.ram_limits import TreeBudget


SCHEMA = "ppt.native-progress-transfer.v1"
AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
MAX_DEPTH = 4
MAX_EXPANSIONS = 26
DECLARATION = re.compile(r"^theorem ([A-Za-z_][A-Za-z_0-9]*)\s*((?:(?!^theorem ).)*?)\s*:=", re.MULTILINE | re.DOTALL)
LOCAL_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9']*")
DESTRUCTIBLE = re.compile(r"(^Bool$|^List\b|[∧∨]|^False$)")
FIXED_MENU = ("assumption", "constructor", "contradiction", "decide", "left", "omega", "rfl", "right", "simp")


def inputs(bpd, evalmath, mathcan):
    return (
        ("BPD", bpd / "mechinterp/bpd/theory/OptimalBasis.lean", "BPD.OptimalBasis", 14),
        ("BPD", bpd / "mechinterp/bpd/theory/FiniteFiltration.lean", "BPD.FiniteFiltration", 5),
        ("BPD", bpd / "mechinterp/bpd/theory/pd07/ReadGateSeparation.lean", "BPD.PD07", 4),
        ("EvalMath", evalmath / "fixtures/bpd_lean_curriculum_v2/train_positives.lean", None, 22),
        ("MathCan", mathcan, "MathCan.FiniteHorizonGap", 6),
        ("LoopedTransformers", REPO / "experiments/activation_funnel_af13/AsyncClosure.lean", None, 5),
        ("LoopedTransformers", REPO / "experiments/activation_funnel_af14/BoundedProgress.lean", None, 3),
        ("LoopedTransformers", REPO / "experiments/activation_funnel_af15/RuleSoundness.lean", None, 3),
        ("LoopedTransformers", REPO / "experiments/activation_funnel_af17/BellmanGap.lean", None, 3),
    )


def declaration_type(header, namespace):
    depth = 0
    for index, character in enumerate(header):
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth -= 1
        elif character == ":" and depth == 0:
            binders, target = header[:index].strip(), header[index + 1:].strip()
            statement = f"∀ {binders}, {target}" if binders else target
            return f"open {namespace} in {statement}" if namespace else statement
    raise ValueError("the original theorem has no outer type separator")


def candidate_menu(visible):
    if not visible["goals"]:
        return ()
    goal = visible["goals"][0]
    target = goal["pp"].rsplit("⊢", 1)[-1]
    moves = list(FIXED_MENU)
    if "→" in target or target.strip().startswith("∀"):
        moves.append("intro ppt_intro")
    for hyp in goal["hypotheses"]:
        name = hyp["name"]
        if not LOCAL_NAME.fullmatch(name):
            continue
        moves.append(f"apply {name}")
        if DESTRUCTIBLE.search(hyp["type"].get("raw", "")):
            moves.append(f"cases {name}")
    return tuple(dict.fromkeys(moves))


def observe(repl, state):
    public = public_state(repl, state)
    if any(not isinstance(goal.get("pp"), str) for goal in public["goals"]):
        raise ValueError("the Lean proof state lacks a public pretty-printed goal")
    return public


def advance(repl, state, visible, tactic):
    result = repl.run(state["sid"], tactic)
    if not result["ok"]:
        return "rejected", state, visible
    state = result["state"]
    after = observe(repl, state)
    if not after["goals"]:
        check = repl.kernel(state["sid"])
        if not check["ok"] or not set(check["axioms"]) <= AXIOMS:
            raise ValueError("a selected tactic closed with an invalid proof or axiom")
        return "closed", state, after
    return ("no_change" if after == visible else "progress"), state, after


def shortest(repl, root, visible):
    queue = deque([(root, visible, ())])
    expanded = 0
    while queue and expanded < MAX_EXPANSIONS:
        state, current, prefix = queue.popleft()
        expanded += 1
        for action in candidate_menu(current):
            status, child, after = advance(repl, state, current, action)
            if status == "closed":
                return (*prefix, action), expanded
            if status == "progress" and len(prefix) + 1 < MAX_DEPTH:
                queue.append((child, after, (*prefix, action)))
    return (), expanded


@contextmanager
def native_sources(paths):
    with TemporaryDirectory(prefix="ppt-native-progress-") as scratch:
        directory = Path(scratch)
        env = lean_env()
        # Bound native compiler workers as well as the Windows process affinity.
        env["LEAN_NUM_THREADS"] = "1"
        env["LEAN_PATH"] = os.pathsep.join(filter(None, (scratch, env.get("LEAN_PATH"))))
        imports = []
        for index, path in enumerate(paths):
            module = f"PPTNativeSource{index}"
            source = directory / (module + ".lean")
            source.write_bytes(path.read_bytes())
            invoke_lean(directory, env, "-o", module + ".olean", source.name)
            imports.append(module)
        yield directory, env, tuple(imports)


def inventory(args):
    prior = REPO / "research/lean_cgrt_hrm400/results/ppt_hybrid_rl/root-adaptation-cohort.json"
    excluded = {row["tid"] for row in json.loads(prior.read_bytes())["cases"] if row["source"] == "evalmath_lg01"}
    modules, cases = [], []
    for index, (source, path, namespace, limit) in enumerate(inputs(args.bpd, args.evalmath, args.mathcan)):
        modules.append({"source": source, "file": path.name, "sha256": sha256_file(path)})
        declarations = DECLARATION.findall(path.read_text(encoding="utf-8"))
        if source == "MathCan":
            declarations = [(name, header) for name, header in declarations if name.startswith(("mask_orbit_", "mask_period_"))]
        if source == "EvalMath":
            declarations = [(name, header) for name, header in declarations if name not in excluded]
        declarations.sort(key=lambda item: digest([SCHEMA, source, item[0]]))
        for name, header in islice(declarations, limit):
            cases.append({"module": index, "source": source, "theorem": name, "statement": declaration_type(header, namespace)})
    return modules, cases


def freeze(args, tree):
    modules, declarations = inventory(args)
    paths = [path for _, path, _, _ in inputs(args.bpd, args.evalmath, args.mathcan)]
    rows, rejected = [], []
    with native_sources(paths) as (directory, env, imports):
        with LeanRepl(imports=imports, env=env, cwd=directory) as repl:
            if repl.version != LEAN_VERSION:
                raise ValueError("unexpected Lean toolchain")
            for case in declarations:
                repl.reset()
                root = repl.init(case["statement"], pp=True)
                if "sid" not in root:
                    raise ValueError(f"the registered statement did not elaborate: {case['theorem']}: {root}")
                visible = observe(repl, root)
                script, count = shortest(repl, root, visible)
                if script:
                    rows.append({**case, "reference": script, "checks": len(script), "expanded": count,
                                 "visible_sha256": digest(visible),
                                 "intro_names": [hyp["name"] for hyp in visible["goals"][0]["hypotheses"]]})
                else:
                    rejected.append({**case, "expanded": count})
            print("registered", dict(Counter(row["source"] for row in rows)),
                  "proof checks", dict(Counter(row["checks"] for row in rows)), flush=True)
    if not any(row["checks"] >= 2 for row in rows):
        raise ValueError("the native bank has no reachable multi-tactic search")
    write_receipt(args.output, {"schema": SCHEMA, "modules": modules, "cases": rows, "unresolved": rejected,
                                "lean": LEAN_VERSION, "lean_sha256": sha256_file(LEAN_EXE),
                                "repl_sha256": sha256_file(REPL_EXE), "memory": tree.stats()})


def add_sources(parser):
    for name in ("bpd", "evalmath", "mathcan"):
        parser.add_argument(name, type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sources(parser)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 1)
    try:
        freeze(args, tree)
    finally:
        tree.close()


if __name__ == "__main__":
    main()
