"""OLR-01C: paired computational-guard Lean theorem collector.

The true/false status of each candidate is obtained from a *real* Lean tactic
and kernel response. Source-only structural metadata may verify results but
must never enter model features. This is a small CPU research gym.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest, sha256_file
from .live_lean import public_state, candidate_menu

SCHEMA = "ppt.olr-guard-native.v1"
RAW_SUM = re.compile(r"^\s*\(?\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*\)?\s*$")
VALID_TACTIC = re.compile(r"^apply r[01]$")


@dataclass(frozen=True)
class GuardPair:
    split: str
    pair_id: str
    a: int
    b: int
    true_sum: int
    false_sum: int
    # Witness is generator-private and may only be used for audits.
    delta: int

    def world(self, bit: int):
        if bit not in (0, 1):
            raise ValueError("paired world bit must be 0 or 1")
        targets = (self.true_sum, self.false_sum) if bit == 0 else (self.false_sum, self.true_sum)
        return GuardWorld(self.split, self.pair_id, bit, self.a, self.b, targets)


@dataclass(frozen=True)
class GuardWorld:
    split: str
    pair_id: str
    world: int
    a: int
    b: int
    targets: tuple[int, int]

    def binders(self):
        x, y = self.targets
        return f"(P : Prop) (r0 : ({self.a} + {self.b} = {x}) → P) (r1 : ({self.a} + {self.b} = {y}) → P)"

    def statement(self):
        return "∀ " + self.binders() + ", P"

    def reference_action(self):
        # Private registration audit; never supplied to a policy.
        return "apply r0" if self.world == 0 else "apply r1"


def generate_pairs(split: str, seed: int, count: int, *, low: int, high: int,
                   exclude: frozenset[tuple[int, int, int]] = frozenset()):
    """Create unique arithmetic lineages in a declared operand range."""
    if not (0 <= low <= high <= 40 and 1 <= count <= 1024):
        raise ValueError("invalid bounded arithmetic sample")
    if split not in ("train", "dev", "ood"):
        raise ValueError("split must be train, dev or ood")
    rng = random.Random(f"olr-guard-v1/{seed}/{split}/{low}/{high}")
    used, out = set(), []
    for i in range(count):
        for attempt in range(500):
            a, b = rng.randint(low, high), rng.randint(low, high)
            delta = rng.choice((-3, -2, -1, 1, 2, 3))
            if a+b+delta < 0 or (a, b, delta) in used or (a, b, delta) in exclude:
                continue
            used.add((a,b,delta))
            break
        else:
            raise ValueError("insufficient distinct operand/offset pairs")
        pair_id = digest(["olr-guard-v1", split, seed, a, b, delta])
        out.append(GuardPair(split, pair_id, a, b, a+b, a+b+delta, delta))
    return tuple(out)


def read_raw_guards(visible: dict, menu: tuple[str, ...]):
    """Extract predecision candidate text *only* from public Lean JSON.

    All equality truth labels must come from native Lean, not Python arithmetic.
    """
    goals = visible.get("goals")
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError("expected exactly one public root Lean goal")
    hyps = goals[0].get("hypotheses")
    if not isinstance(hyps, list):
        raise ValueError("missing public Lean hypotheses")
    types = {h.get("name"): h.get("type") for h in hyps if isinstance(h, dict)}
    guards = []
    for action in menu:
        if not VALID_TACTIC.fullmatch(action):
            raise ValueError("registered candidate outside r0/r1 menu")
        ty = types.get(action.split()[1])
        if not isinstance(ty, dict) or set(ty) != {"imp"}:
            raise ValueError("candidate does not have a one-premise implication")
        parts = ty["imp"]
        if (not isinstance(parts, list) or len(parts) != 2 or
                not isinstance(parts[0], dict) or not isinstance(parts[1], dict) or
                not isinstance(parts[0].get("raw"), str) or parts[1] != goals[0]["target"]):
            raise ValueError("computational guard missing from native public view")
        guards.append(parts[0]["raw"])
    return tuple(guards)


def numeric_features(raw: str) -> tuple[float, float, float]:
    """Extract individual public numerals without evaluating the predicate."""
    match = RAW_SUM.fullmatch(raw)
    if match is None:
        raise ValueError(f"unrecognized public equality guard: {raw[:120]!r}")
    a, b, c = (int(x) for x in match.groups())
    if max(a,b,c) > 100:
        raise ValueError("numerals exceed the preregistered range")
    return (a / 40., b / 40., c / 80.)


def proof_attempt(lean: LeanRepl, world: GuardWorld, tactic: str):
    """Spend one root rule application and at most one rfl guard check."""
    if not VALID_TACTIC.fullmatch(tactic):
        raise ValueError("invalid native guard tactic")
    lean.reset()
    state = lean.init(world.statement(), pp=True)
    visible = public_state(lean, state)
    menu = candidate_menu(lean, state)
    if menu != ("apply r0", "apply r1"):
        raise ValueError(f"unmatched public root menu: {menu}")
    guards = read_raw_guards(visible, menu)
    vectors = tuple(numeric_features(g) for g in guards)
    root = lean.run(state["sid"], tactic)
    if root.get("ok") is not True:
        raise ValueError("registered root tactic rejected by Lean")
    after = public_state(lean, root["state"])
    if len(after["goals"]) != 1:
        raise ValueError("root must create exactly one arithmetic subgoal")
    try_guard = lean.run(root["state"]["sid"], "rfl")
    success = False
    kernel = None
    if try_guard.get("ok") is True:
        new = public_state(lean, try_guard["state"])
        if new["goals"]:
            raise ValueError("rfl unexpectedly left another goal")
        kernel = lean.kernel(try_guard["state"]["sid"])
        success = kernel.get("ok") is True and kernel.get("axioms") == []
        if not success:
            raise ValueError("Lean closed without clean kernel certificate")
    return {
        "action": tactic,
        "public_state_sha256": digest(visible),
        "menu": list(menu), "raw_guards": list(guards),
        "numeric_features": [list(v) for v in vectors],
        "root_status": "accepted_nonterminal",
        "guard_status": "kernel_closed" if success else "rfl_rejected",
        "guard_goal_sha256": digest(after),
        "kernel_ok": success,
        "native_tactic_calls": 2,
        "script": [tactic, "rfl"] if success else [tactic],
    }


def collect_pair(lean: LeanRepl, pair: GuardPair):
    """Both candidate outcomes are checked *before* source examples are fitted."""
    items = []
    for bit in (0, 1):
        world = pair.world(bit)
        outcomes = [proof_attempt(lean, world, action)
                    for action in ("apply r0", "apply r1")]
        labels = [int(x["kernel_ok"]) for x in outcomes]
        if (labels.count(1) != 1 or
                outcomes[0]["numeric_features"] != outcomes[1]["numeric_features"] or
                outcomes[0]["raw_guards"] != outcomes[1]["raw_guards"]):
            raise ValueError("native verdict, inputs, or source label mismatch")
        if ("apply r" + str(labels.index(1))) != world.reference_action():
            raise ValueError("native proof disagrees with source registration")
        items.append({
            "root": digest(world.statement()), "pair_id": world.pair_id,
            "split": pair.split, "world": bit,
            "statement": world.statement(),
            "source": outcomes[0]["public_state_sha256"],
            "menu": outcomes[0]["menu"],
            "raw_guards": outcomes[0]["raw_guards"],
            "features": outcomes[0]["numeric_features"],
            "labels": labels, "attempts": outcomes,
            "reference_proof": outcomes[labels.index(1)]["script"],
            "tactic_budget": 2,
        })
    if items[0]["root"] == items[1]["root"]:
        raise ValueError("pair worlds became identical")
    return tuple(items)


def certified_standalone(rows: list[dict], out: Path):
    """Independent Lean compiler for the narrow apply-r0/r1 + rfl grammar."""
    lines = ["namespace OLRGuardNative"]
    names = []
    for index, row in enumerate(rows):
        script = row["reference_proof"]
        if (len(script) != 2 or not VALID_TACTIC.fullmatch(script[0])
                or script[1] != "rfl"):
            raise ValueError("unsafe native proof script")
        # All parameters are frozen integers and generated from our closed grammar.
        name = f"g_{index}"
        stmt = row["statement"]
        if not re.fullmatch(r"∀ \(P : Prop\) \(r0 : \(\d+ \+ \d+ = \d+\) → P\) \(r1 : \(\d+ \+ \d+ = \d+\) → P\), P",stmt):
            raise ValueError("unsafe source declaration")
        # Standalone theorem statement includes a closed binder list.
        binders = stmt[2:-3]  # remove leading '∀ ' and trailing ', P'
        lines.extend([f"theorem {name} {binders} : P := by",
                      f"  {script[0]}", "  rfl"])
        names.append(name)
    lines.append("end OLRGuardNative")
    lines.extend(f"#print axioms OLRGuardNative.{name}" for name in names)
    source = "\n".join(lines) + "\n"
    out.write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "OLRGuard.lean"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run([str(config.LEAN_EXE), str(path)], env=config.lean_env(),
                                capture_output=True, text=True, encoding="utf-8",
                                timeout=180, check=False)
    text = result.stdout + result.stderr
    clean = sum(f"'OLRGuardNative.{name}' does not depend on any axioms" in text
                for name in names)
    if result.returncode != 0 or clean != len(names):
        raise ValueError(f"standalone guard proofs rejected: rc={result.returncode}, clean={clean}/{len(names)}; {text[-1000:]}")
    return {"compiled":len(names),"clean_axioms":clean,"source_sha256":sha256_file(out)}


def freeze(seed: int, train: int, dev: int, out: Path, *, ood: int = 0):
    """Freeze a complete source/held-out verifier bank with separate lineages."""
    if out.exists():
        raise ValueError("output directory must be fresh")
    train_pairs = generate_pairs("train", seed, train, low=0, high=8)
    seen_train = frozenset((p.a, p.b, p.delta) for p in train_pairs)
    dev_pairs = generate_pairs("dev", seed+7, dev, low=0, high=8, exclude=seen_train)
    ood_pairs = generate_pairs("ood", seed+13, ood, low=9, high=16) if ood else ()
    # Scope together all raw arithmetic shapes, regardless of name/order.
    all_pairs = (*train_pairs,*dev_pairs,*ood_pairs)
    seen = [(p.a,p.b,p.delta) for p in all_pairs]
    if len(set(seen)) != len(seen):
        raise ValueError("source and evaluation arithmetic lineages overlap")
    rows: dict[str,list] = {"train":[],"dev":[],"ood":[]}
    with LeanRepl(timeout=60) as lean:
        if not str(lean.version).startswith("4.19.0"):
            raise ValueError("pinned native Lean 4.19.0 required")
        version = lean.version
        for pair in all_pairs:
            rows[pair.split].extend(collect_pair(lean,pair))
    out.mkdir(parents=True)
    for split, shard in rows.items():
        (out/f"{split}.jsonl").write_text(
            "".join(canonical(r)+"\n" for r in shard),encoding="utf-8")
    compiled = certified_standalone([row for s in rows.values() for row in s],
                                    out/"standalone.lean")
    manifest = {
        "schema":SCHEMA, "seed":seed, "lean_version":version,
        "lean_exe_sha256":sha256_file(config.LEAN_EXE),
        "repl_exe_sha256":sha256_file(config.REPL_EXE),
        "source_not_heldout":True, "model_training_uses_train_split_only":True,
        "independent_certificates":compiled,
        "splits":{s:{"pairs":len(shard)//2, "worlds":len(shard),
                       "sha256":sha256_file(out/f"{s}.jsonl")} for s,shard in rows.items()},
        "learning_status":"untrained_native_bank",
        "scientific_scope":"arithmetic_guard_smoke_not_external_mathlib",
    }
    (out/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=20261123)
    p.add_argument("--train",type=int,default=12)
    p.add_argument("--dev",type=int,default=4)
    p.add_argument("--ood",type=int,default=0)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args(argv)
    print(canonical(freeze(x.seed,x.train,x.dev,x.out,ood=x.ood)))


if __name__=="__main__":
    main()
