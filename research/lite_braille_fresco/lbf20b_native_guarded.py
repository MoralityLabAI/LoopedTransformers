"""LBF-20B: development-only native conditional ground rewrite search.

The graded automaton ignores conditional premises optimistically. Lean alone
determines success of the fixed guard tactic portfolio. A strong factored
guard-probe / deferred-proof control prevents artificial eager-verifier wins.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from lbf20_ground_witness import WeightedWitness, expression, replace, steps, subterms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lean_cgrt_hrm400"))
from lcgrt.lean_repl import LeanRepl
from lcgrt import config

SEED = 2026112503
HORIZON = 20
CALL_CAP = 24
WALL_CAP = 4.0
FEATURES = 128
SOLVER = "first | (assumption; done) | (rfl; done) | (simp; done) | (omega; done)"
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
ARMS = ("eager_edit", "eager_edit_learned", "eager_graded", "eager_graded_learned",
        "lazy_graded", "lazy_failfirst", "probe_all")
ARITIES = {**{"c_" + c: 0 for c in "abcdef"}, "f_f": 1, "f_p": 2, "f_q": 2}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


@dataclass(frozen=True)
class Rule:
    lhs: tuple
    rhs: tuple
    guard: str


@dataclass(frozen=True)
class Case:
    root: str
    split: str
    family: str
    source: tuple
    target: tuple
    rules: tuple[Rule, ...]

    def statement(self):
        binders = ["(T : Type)", "(c_a c_b c_c c_d c_e c_f : T)",
                   "(f_f : T → T)", "(f_p f_q : T → T → T)",
                   "(n m : Nat)", "(xs : List Nat)", "(P Q : Prop)",
                   "(hp : P)", "(hn : n ≤ m)"]
        binders += [f"(h{i} : ({r.guard}) → {expression(r.lhs)} = {expression(r.rhs)})"
                    for i, r in enumerate(self.rules)]
        return "∀ " + " ".join(binders) + ", " + expression(self.source) + " = " + expression(self.target)

    def intros(self):
        return ("intro T c_a c_b c_c c_d c_e c_f f_f f_p f_q n m xs P Q hp hn "
                + " ".join(f"h{i}" for i in range(len(self.rules))))


def make_guard(rng, positive):
    k = rng.randint(1, 9)
    if positive:
        pool = ["P", "n + 0 = n", f"n + {k} ≤ m + {k}",
                f"n < m + {k}", "List.reverse (List.reverse xs) = xs",
                "xs ++ [] = xs", "List.length (xs ++ [0]) = List.length xs + 1",
                f"List.length xs ≤ List.length xs + {k}", "List.map id xs = xs"]
    else:
        pool = ["Q", "False", f"m + {k} ≤ n", f"n + {k} ≤ n",
                f"List.length xs + {k} = List.length xs", "xs ++ [0] = xs",
                f"m + {k} < n", "n + 1 = n"]
    return rng.choice(pool)


def join_pairs(rng, nodes):
    nodes = list(nodes)
    while len(nodes) > 1:
        i = rng.randrange(len(nodes) - 1)
        nodes[i:i + 2] = [("p", nodes[i], nodes[i + 1])]
    return nodes[0]


def generate(split, count):
    rng = random.Random(SEED + (0 if split == "train" else 1000))
    cases = []
    for i in range(count):
        a, b, c, d, e, f = [(x,) for x in rng.sample(list("abcdef"), 6)]
        family = ("leaf", "growth", "composite")[i % 3]
        pairs = rng.randint(2, 3 if split == "train" else 4)
        leaves = [("p", a, d) if rng.random() < .65 else ("p", d, a) for _ in range(pairs)]
        source = join_pairs(rng, leaves)

        def target_of(t):
            if len(t) == 1:
                return c if t == a else f if t == d else t
            return (t[0], *(target_of(x) for x in t[1:]))

        target = target_of(source)
        good = [(a, b), (b, c), (d, e), (e, f)]
        if family == "growth":
            good = [(a, ("f", b)), (("f", b), c), (d, ("f", e)), (("f", e), f)]
        if family == "composite":
            good += [(("p", a, d), ("q", b, e)), (("q", b, e), ("p", c, f))]
        bad = [(a, c), (d, f), (a, f), (d, c), (("p", a, d), ("p", c, f)),
               (("p", d, a), ("p", f, c)), (a, e)]
        rules = [Rule(l, r, make_guard(rng, True)) for l, r in good]
        rules += [Rule(l, r, make_guard(rng, False)) for l, r in bad]
        rng.shuffle(rules)
        cases.append(Case(f"{split}_{i:02d}", split, family, source, target, tuple(rules)))
    return cases


def features(guard):
    # Only the public guard string. No split, root, rule index or outcome.
    text = " " + guard + " "
    x = [0.0] * FEATURES
    for width in (2, 3):
        for i in range(len(text) - width + 1):
            h = int.from_bytes(hashlib.blake2b(text[i:i + width].encode(), digest_size=4).digest(), "big")
            x[h % FEATURES] += 1
    norm = max(1.0, math.sqrt(sum(v * v for v in x)))
    return [v / norm for v in x] + [1.0]


def probability(w, guard):
    z = sum(a * b for a, b in zip(w, features(guard)))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def fit_guard_ranker(rows):
    assert rows and all(r["split"] == "train" for r in rows)
    data = [(features(r["guard"]), float(r["native_accepted"])) for r in rows]
    w = [0.0] * (FEATURES + 1)
    for _ in range(160):
        grad = [.01 * v for v in w]
        for x, y in data:
            p = 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, x))))))
            for j, v in enumerate(x):
                grad[j] += (p - y) * v / len(data)
        w = [a - .6 * b for a, b in zip(w, grad)]
    return w


def normalize_eq(raw):
    # Parse Lean's actual pretty-printed ground equality; no text-match guessing.
    tokens = re.findall(r"c_[a-f]|f_[fpq]|[()=]|\S+", raw)
    # Split punctuation glued to tokens by a generic match, if present.
    tokens = re.findall(r"c_[a-f]|f_[fpq]|[()=]|[^\s()=]+", raw)
    pos = 0

    def term():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("truncated native term")
        token = tokens[pos]
        pos += 1
        if token == "(":
            value = term()
            if tokens[pos] != ")":
                raise ValueError("missing close parenthesis")
            pos += 1
            return value
        if token not in ARITIES:
            raise ValueError(f"unexpected native symbol {token!r} in {raw!r}")
        return (token[2:], *(term() for _ in range(ARITIES[token])))

    left = term()
    if tokens[pos] != "=":
        raise ValueError("native target is not an equality")
    pos += 1
    right = term()
    if pos != len(tokens):
        raise ValueError(f"unparsed native target: {raw!r}")
    return left, right


def check_state(state, expected, target):
    assert not state["closed"] and len(state["goals"]) == 1, state
    raw = state["goals"][0]["target"]["raw"]
    assert normalize_eq(raw) == (expected, target), (raw, expected, target)
    return raw


def guard_tactic(rule, i):
    return f"have guard_{i} : ({rule.guard}) := by {SOLVER}"


def proof_step(word, rule, i, pos, cached=False):
    ctx = expression(replace(word, pos, ("HOLE",)), "z")
    guard = f"guard_{i}" if cached else f"(by {SOLVER})"
    return f"congrArg (fun z : T => {ctx}) (h{i} {guard})"


def enumerate_steps(case, word, active=None):
    for i, rule in enumerate(case.rules):
        if active is not None and i not in active:
            continue
        for _, pos, nxt in steps(word, [(rule.lhs, rule.rhs)]):
            yield i, pos, nxt


def path_edges(case, path, active):
    assert path[0] == case.source and path[-1] == case.target
    edges = []
    for a, b in zip(path, path[1:]):
        i, pos, _ = next((i, p, u) for i, p, u in enumerate_steps(case, a, active) if u == b)
        edges.append((a, i, pos, b))
    return edges


def edit(a, b):
    if a == b:
        return 0
    if a[0] != b[0] or len(a) != len(b):
        return max(len(tuple(subterms(a))), len(tuple(subterms(b))))
    return sum(edit(x, y) for x, y in zip(a[1:], b[1:]))


class Episode:
    def __init__(self, repl, case, arm):
        self.repl, self.case, self.arm = repl, case, arm
        self.started = time.perf_counter()
        self.events = []
        self.calls = self.rejects = 0
        self.native_seconds = self.kernel_seconds = 0.0
        self.kernel = None
        self.script = []
        self.status = "exhausted"
        t0 = time.perf_counter()
        repl.reset()
        self.root = repl.init(case.statement())
        check_state(self.root, case.source, case.target)
        self.init_seconds = time.perf_counter() - t0

    def alive(self):
        return self.calls < CALL_CAP and time.perf_counter() - self.started < WALL_CAP

    def attempt(self, sid, tactic, info):
        if not self.alive():
            return None
        t0 = time.perf_counter()
        reply = self.repl.run(sid, tactic)
        elapsed = time.perf_counter() - t0
        self.calls += 1
        self.native_seconds += elapsed
        assert type(reply.get("ok")) is bool, reply
        self.rejects += int(not reply["ok"])
        event = {**info, "tactic": tactic, "accepted": reply["ok"], "seconds": elapsed}
        if not reply["ok"]:
            event["error"] = str(reply.get("error", ""))[:240]
        self.events.append(event)
        return reply

    def certify(self, state, script):
        assert state["closed"] and not state["goals"]
        t0 = time.perf_counter()
        receipt = self.repl.kernel(state["sid"])
        self.kernel_seconds += time.perf_counter() - t0
        assert receipt.get("ok") is True, receipt
        axioms = receipt["axioms"]
        assert set(axioms) <= ALLOWED_AXIOMS, receipt
        self.kernel = {"axioms": axioms, "term_sha256": digest(receipt.get("term"))}
        self.script = list(script)
        self.status = "closed" if time.perf_counter() - self.started <= WALL_CAP else "late_closed"

    def finish(self):
        elapsed = time.perf_counter() - self.started
        if self.status == "exhausted":
            self.status = "time_cap" if elapsed >= WALL_CAP else "call_cap" if self.calls >= CALL_CAP else "exhausted"
        assert self.calls <= CALL_CAP
        return {"root": self.case.root, "split": self.case.split, "family": self.case.family,
                "arm": self.arm, "status": self.status, "closed": self.status == "closed",
                "calls": self.calls, "rejects": self.rejects, "native_seconds": self.native_seconds,
                "init_seconds": self.init_seconds, "kernel_seconds": self.kernel_seconds,
                "wall_seconds": elapsed, "kernel": self.kernel, "script": self.script, "events": self.events}


def eager(ep, w):
    case = ep.case
    graded = "graded" in ep.arm
    learned = "learned" in ep.arm
    # Reverse the optimistic unconditional rules: evaluate each discovered
    # source against a single fixed weighted automaton rooted at the target.
    oracle = WeightedWitness([case.target], [(r.rhs, r.lhs) for r in case.rules], HORIZON) if graded else None
    distances = {}

    def distance(t):
        if t not in distances:
            distances[t] = oracle.distance(case.target, t) if graded else edit(t, case.target)
        return distances[t]

    agenda = []
    serial = itertools.count()
    best_depth = {case.source: 0}

    def add(word, sid, script, g):
        if g >= HORIZON:
            return
        for i, pos, nxt in enumerate_steps(case, word):
            if best_depth.get(nxt, HORIZON + 1) <= g + 1:
                continue
            h = distance(nxt)
            p = probability(w, case.rules[i].guard) if learned else 1.0
            priority = g + 1 + h + 3 * (1 - p)
            heapq.heappush(agenda, (priority, next(serial), word, sid, script, g, i, pos, nxt, h, p))

    add(case.source, ep.root["sid"], (), 0)
    while agenda and ep.alive():
        _, _, word, sid, script, g, i, pos, nxt, h, p = heapq.heappop(agenda)
        if best_depth[word] != g or best_depth.get(nxt, HORIZON + 1) <= g + 1:
            continue
        proof = proof_step(word, case.rules[i], i, pos)
        tac = "exact " + proof if nxt == case.target else f"refine Eq.trans (b := {expression(nxt)}) ({proof}) ?_"
        reply = ep.attempt(sid, tac, {"kind": "transition", "rule": i, "source": word, "predicted": nxt,
                                      "heuristic": h, "guard_score": p})
        if reply is None:
            break
        if not reply["ok"]:
            continue
        state = reply["state"]
        if nxt == case.target:
            ep.certify(state, (*script, tac))
            return
        ep.events[-1]["native_target"] = check_state(state, nxt, case.target)
        best_depth[nxt] = g + 1
        add(nxt, state["sid"], (*script, tac), g + 1)


def factored(ep, w):
    case = ep.case
    available = set(range(len(case.rules)))
    known = set()
    sid = ep.root["sid"]
    script = []

    def probe(i):
        nonlocal sid
        tac = guard_tactic(case.rules[i], i)
        reply = ep.attempt(sid, tac, {"kind": "guard", "rule": i, "guard": case.rules[i].guard,
                                      "guard_score": probability(w, case.rules[i].guard)})
        if reply is None:
            return False
        if not reply["ok"]:
            available.remove(i)  # fixed portfolio failure, not logical refutation
            return True
        state = reply["state"]
        check_state(state, case.source, case.target)
        sid = state["sid"]
        script.append(tac)
        known.add(i)
        return True

    if ep.arm == "probe_all":
        for i in sorted(available):
            if not probe(i):
                return
    while ep.alive():
        pairs = [(case.rules[i].lhs, case.rules[i].rhs) for i in sorted(available)]
        oracle = WeightedWitness([case.source], pairs, HORIZON)
        path = oracle.witness(case.source, case.target)
        if path is None:
            return
        edges = path_edges(case, path, available)
        unknown = list(dict.fromkeys(i for _, i, _, _ in edges if i not in known))
        if unknown:
            i = min(unknown, key=lambda i: probability(w, case.rules[i].guard)) if ep.arm == "lazy_failfirst" else unknown[0]
            if not probe(i):
                return
            continue
        proofs = [proof_step(word, case.rules[i], i, pos, cached=True) for word, i, pos, _ in edges]
        if proofs:
            term = proofs[-1]
            for pr in reversed(proofs[:-1]):
                term = f"Eq.trans ({pr}) ({term})"
            tac = "exact " + term
        else:
            tac = "rfl"
        reply = ep.attempt(sid, tac, {"kind": "deferred_proof", "rewrite_steps": len(edges),
                                      "proved_guards": sorted(known)})
        if reply is None:
            return
        assert reply["ok"], ("deferred proof rejected", reply)
        ep.certify(reply["state"], (*script, tac))
        return


def native_train(repl, cases):
    rows = []
    for case in cases:
        assert case.split == "train"
        repl.reset()
        root = repl.init(case.statement())
        check_state(root, case.source, case.target)
        for i, r in enumerate(case.rules):
            tac = guard_tactic(r, i)
            t0 = time.perf_counter()
            reply = repl.run(root["sid"], tac)
            assert type(reply.get("ok")) is bool, reply
            if reply["ok"]:
                check_state(reply["state"], case.source, case.target)
            rows.append({"root": case.root, "split": case.split, "family": case.family, "rule": i,
                         "guard": r.guard, "tactic": tac, "native_accepted": reply["ok"],
                         "seconds": time.perf_counter() - t0})
    return rows


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    train = generate("train", 12)
    dev = generate("dev", 18)
    with LeanRepl(imports=("Lean",), timeout=20) as repl:
        assert str(repl.version).startswith("4.19.0"), repl.version
        t0 = time.perf_counter()
        train_rows = native_train(repl, train)
        w = fit_guard_ranker(train_rows)
        fit_seconds = time.perf_counter() - t0
        weights_bytes = (canonical({"weights": w}) + "\n").encode()
        (out / "guard_ranker.json").write_bytes(weights_bytes)
        (out / "train_guard_receipts.jsonl").write_text(
            "".join(canonical(r) + "\n" for r in train_rows), encoding="utf-8")
        train_acc = sum((probability(w, r["guard"]) >= .5) == r["native_accepted"] for r in train_rows) / len(train_rows)
        assert any(r["native_accepted"] for r in train_rows) and any(not r["native_accepted"] for r in train_rows)
        rows = []
        proof_lines = ["import Lean", ""]
        for case in dev:
            for arm in ARMS:
                ep = Episode(repl, case, arm)
                (eager if arm.startswith("eager") else factored)(ep, w)
                r = ep.finish()
                rows.append(r)
                if r["kernel"] is not None:
                    name = f"lbf20b_{case.root}_{arm}"
                    proof_lines += [f"theorem {name} : {case.statement()} := by", "  " + case.intros(),
                                    *("  " + t for t in r["script"]), f"#print axioms {name}", ""]
                print(case.root, arm, r["status"], r["calls"], round(r["wall_seconds"], 4), flush=True)
    (out / "Witnesses.lean").write_text("\n".join(proof_lines), encoding="utf-8")
    (out / "dev_episodes.jsonl").write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
    aggregate = {}
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm]
        aggregate[arm] = {
            "roots": len(a), "closed": sum(r["closed"] for r in a),
            "mean_calls": statistics.mean(r["calls"] for r in a),
            "mean_rejects": statistics.mean(r["rejects"] for r in a),
            "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in a),
            "mean_native_seconds": statistics.mean(r["native_seconds"] for r in a),
            "closure_at_calls": {str(k): sum(r["closed"] and r["calls"] <= k for r in a) for k in (8, 16, 24)},
            "status_counts": dict(collections.Counter(r["status"] for r in a)),
            "axioms": sorted({x for r in a if r["kernel"] for x in r["kernel"]["axioms"]}),
        }
    summary = {"schema": "lbf20b.native-guarded.v1", "scope": "development_only",
               "train_roots": len(train), "dev_roots": len(dev), "generator_seed": SEED,
               "call_cap": CALL_CAP, "soft_wall_cap_seconds": WALL_CAP, "horizon": HORIZON,
               "train_guard_receipts": len(train_rows),
               "train_native_accepts": sum(r["native_accepted"] for r in train_rows),
               "guard_ranker": {"parameters": len(w), "optimizer": "full batch logistic gradient descent",
                                "steps": 160, "lr": .6, "l2": .01, "fit_seconds_including_native_labels": fit_seconds,
                                "train_accuracy": train_acc, "checkpoint_sha256": hashlib.sha256(weights_bytes).hexdigest(),
                                "training_receipts_sha256": digest(train_rows)},
               "episodes": len(rows), "native_tactic_calls": sum(r["calls"] for r in rows),
               "native_kernel_certificates": sum(r["kernel"] is not None for r in rows),
               "arms": aggregate}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    run(p.parse_args().out)
