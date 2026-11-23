"""Live Lean-CGRT PPT-G03 proof search, receipts, and support/query routing.

This module calls the actual lgym JSON-lines server. Only Lean decides whether
a tactic is accepted and whether its final proof is kernel-checkable. The
generator never exposes its reference proof or transition table to the model.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import torch

from lcgrt.lean_repl import LeanRepl
from .accretion import state_hash
from .core import FastHead, PlasticProofTransformer, RetrievalControl
from .data import canonical, digest
from .live_suite import LiveCase

MAX_MENU = 64
ALLOWED_AXIOMS = set()  # Fresh Horn theorems need no extra axioms.
TACTIC_NAME = re.compile(r"^[hr][0-9]{1,4}$")


class Repl(Protocol):
    version: str
    contexts: dict
    calls: int

    def reset(self): ...
    def init(self, statement: str, pp: bool = False): ...
    def run(self, sid: int, tactic: str): ...
    def kernel(self, sid: int): ...


def public_state(repl: Repl, state: dict) -> dict:
    """Remove unstable server IDs and preserve Lean's reported goal contexts."""
    goals = state.get("goals")
    if not isinstance(goals, list):
        raise ValueError("Lean reply has no goal list")
    result = []
    for g in goals:
        if not isinstance(g, dict) or not isinstance(g.get("target"), dict):
            raise ValueError("malformed Lean goal")
        context = repl.contexts.get(g.get("ctx"))
        if not isinstance(context, list):
            context = g.get("hyps")
        if not isinstance(context, list):
            raise ValueError("missing Lean local context")
        row = {"target": g["target"], "hypotheses": context}
        if isinstance(g.get("pp"), str):
            row["pp"] = g["pp"]
        result.append(row)
    return {"goals": result}


def candidate_menu(repl: Repl, state: dict):
    """Only syntactic proposals from public local hypotheses; no label peeking."""
    goals = public_state(repl, state)["goals"]
    if not goals:
        return ()
    goal = goals[0]
    candidates = []
    if "and" in goal["target"]:
        candidates.append("constructor")
    for hyp in goal["hypotheses"]:
        name, ty = hyp.get("name"), hyp.get("type")
        if not (isinstance(name, str) and TACTIC_NAME.fullmatch(name)
                and isinstance(ty, dict)):
            continue
        # Public type-directed retrieval: a hypothesis is proposed only when
        # its implication conclusion syntactically matches the active goal.
        # This checks no proof or premise and consults no sealed labels.
        while "imp" in ty:
            pair = ty["imp"]
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[1], dict):
                raise ValueError("malformed Lean implication in public context")
            ty = pair[1]
        if ty == goal["target"]:
            candidates.append("apply " + name)
    return tuple(dict.fromkeys(candidates))[:MAX_MENU]


def state_input(repl: Repl, state: dict, failures: tuple[str, ...], remaining: int):
    visible = public_state(repl, state)
    return (canonical({"state": visible, "failed_choices": list(failures),
                      "checks_remaining": remaining}), digest(visible))


def _automatic(candidates: tuple[str, ...], state: dict) -> int:
    # Host-order independent of verdict. Prioritize constructor for conjunction;
    # then direct facts, then rules; the exact Lean checker adjudicates.
    def priority(item):
        return (0 if item == "constructor" else
                1 if re.match(r"apply h[0-9]+$", item) else 2, item)
    return min(range(len(candidates)), key=lambda i: priority(candidates[i]))


@torch.no_grad()
def choose(model: PlasticProofTransformer, goal: str, tactics: tuple[str, ...],
           *, policy: str, remaining: int, depth: int,
           fast: FastHead | None = None, memory: RetrievalControl | None = None,
           sham: FastHead | None = None, scale: float = 2.0,
           uniform_seed: int = 0):
    if policy == "auto8":
        return _automatic(tactics, {}), None, False
    if policy == "uniform":
        if type(uniform_seed) is not int or not 0 <= uniform_seed <= 2**32 - 1:
            raise ValueError("random-menu seed must be an unsigned 32-bit integer")
        if not tactics:
            raise ValueError("random-menu policy requires candidate actions")
        # Root- and public-packet-specific entropy only. The reference proof,
        # labels, model logits and native verdicts are absent from this draw.
        payload = canonical([uniform_seed, goal, list(tactics), remaining])
        number = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
        return random.Random(number).randrange(len(tactics)), None, False
    if policy not in {"frozen", "dram", "retrieval", "sham"}:
        raise ValueError("unknown proof action policy")
    out = model(goal, tactics, depth=depth, checks_remaining=min(8, remaining))[-1]
    logits = out["logits"]
    if policy == "dram" and fast is not None:
        logits = logits + scale * fast.predict(out["features"])
    elif policy == "retrieval" and memory is not None:
        logits = logits + scale * memory.predict(out["features"])
    elif policy == "sham" and sham is not None:
        logits = logits + scale * sham.predict(out["features"])
    return int(logits.argmax()), out["features"], True


def run_episode(
    repl: Repl, case: LiveCase, model: PlasticProofTransformer, *,
    policy: str, checks: int, depth: int,
    fast: FastHead | None = None, memory: RetrievalControl | None = None,
    sham: FastHead | None = None, scale: float = 2.0, collect_support: bool = False,
    uniform_seed: int = 0,
):
    """Run one proof from a new Lean init. Every attempted tactic costs one check.

    The same checkpoint and eight-check budget apply to all policies. Support
    evidence may update CPU fast weights after a definitive Lean result.
    """
    if checks not in range(1, 33) or depth not in range(1, 17):
        raise ValueError("bounded tactic checks 1..32 and recurrence 1..16 required")
    if policy not in {"auto8", "uniform", "frozen", "auto4_then_ppt", "dram", "retrieval", "sham"}:
        raise ValueError("unsupported policy")
    repl.reset()
    root = repl.init(case.statement, pp=True)
    sid = root["sid"]
    current = root
    failed_at_state: dict[str, set[str]] = defaultdict(set)
    events = []
    accepted_script = []
    deferred_support: list[tuple[dict, torch.Tensor]] = []
    model_calls = kernel_calls = 0
    result = "budget_exhausted"
    error = None
    started = time.perf_counter()
    for attempt in range(checks):
        visible = public_state(repl, current)
        if not visible["goals"]:
            raise ValueError("closed state reached without terminal receipt")
        state_key = digest(visible)
        options = tuple(t for t in candidate_menu(repl, current)
                        if t not in failed_at_state[state_key])
        if not options:
            result = "no_candidates"
            break
        goal, before_sha = state_input(repl, current,
                                       tuple(sorted(failed_at_state[state_key])),
                                       checks - attempt)
        decision_policy = ("auto8" if policy == "auto4_then_ppt" and attempt < 4
                           else "frozen" if policy == "auto4_then_ppt" else policy)
        selected, features, consulted = choose(
            model, goal, options, policy=decision_policy, remaining=checks - attempt,
            depth=depth, fast=fast, memory=memory, sham=sham, scale=scale,
            uniform_seed=uniform_seed,
        )
        model_calls += int(consulted)
        tactic = options[selected]
        step = {"attempt": attempt + 1, "state_before_sha256": before_sha,
                "public_packet_sha256": digest(goal), "tactic": tactic,
                "menu_sha256": digest(options), "selected_index": selected,
                "input": json.loads(goal),
                "candidate_menu": list(options),
                "policy": decision_policy}
        # The live server is the only source of action verdicts.
        reply = repl.run(sid, tactic)
        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
            raise ValueError("malformed Lean tactic verdict")
        if reply["ok"] is False:
            step.update(status="rejected", error=str(reply.get("error", ""))[:512],
                        state_after_sha256=before_sha, kernel=None)
            failed_at_state[state_key].add(tactic)
            label = 0
        else:
            after = reply.get("state")
            if not isinstance(after, dict) or not isinstance(after.get("sid"), int):
                raise ValueError("Lean accepted tactic without successor state")
            sid = after["sid"]
            current = after
            accepted_script.append(tactic)
            next_state = public_state(repl, current)
            step["state_after_sha256"] = digest(next_state)
            step["state_after"] = next_state
            if next_state["goals"]:
                step.update(status="accepted_nonterminal", kernel=None)
                label = None
            else:
                kernel_calls += 1
                cert = repl.kernel(sid)
                verified = (isinstance(cert, dict) and cert.get("ok") is True
                            and isinstance(cert.get("axioms"), list)
                            and all(isinstance(a, str) and a in ALLOWED_AXIOMS
                                    for a in cert["axioms"]))
                step.update(status="kernel_closed" if verified else "certificate_rejected",
                            kernel={"ok": verified,
                                    "axioms": cert.get("axioms") if isinstance(cert, dict) else None,
                                    "term_sha256": digest(cert.get("term")) if verified else None,
                                    "error": str(cert.get("error", ""))[:512]
                                             if isinstance(cert, dict) else "malformed"})
                label = 1 if verified else None
                result = step["status"]
        if collect_support and features is not None:
            f = features[selected]
            if step["status"] == "accepted_nonterminal":
                # A locally valid step may still lead to a dead proof branch.
                # Hold it until a certified terminal result is observed.
                deferred_support.append((step, f))
            elif label is not None:
                # Rejects are definitive negative observations; terminal
                # kernel closures are positive observations for the full path.
                recipients = (deferred_support + [(step, f)]
                              if label == 1 else [(step, f)])
                for earlier, feature in recipients:
                    if fast is not None:
                        fast.update(feature, label)
                    if memory is not None:
                        memory.update(feature, label)
                    if sham is not None:
                        sham.update(feature, 1 - label)
                    earlier["support_label"] = label
                    earlier["support_label_source"] = (
                        "terminal_kernel_backed_up" if earlier is not step
                        else "immediate_verifier")
                if label == 1:
                    deferred_support.clear()
        events.append(step)
        if result in ("kernel_closed", "certificate_rejected"):
            break
    return {
        "root": case.root, "lineage": case.lineage, "split": case.split,
        "family": case.family, "statement_sha256": digest(case.statement),
        "policy": policy, "status": result, "attempts": len(events),
        "model_calls": model_calls, "kernel_calls": kernel_calls,
        "accepted_script": accepted_script, "events": events,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "error": error,
        "live_kernel": bool(result == "kernel_closed"),
        "standalone_certificate": None,
    }


def replay_episode(repl: Repl, case: LiveCase, episode: dict):
    """Re-execute the recorded tactic sequence, independent of the model.

    The transcript is authoritative only if a native live Lean server is used.
    Tests may supply an explicitly simulated Repl for protocol validation.
    """
    repl.reset()
    current = repl.init(case.statement, pp=True)
    sid = current["sid"]
    failures: dict[str, set[str]] = defaultdict(set)
    for index, step in enumerate(episode["events"]):
        visible = public_state(repl, current)
        key = digest(visible)
        failures_here = tuple(sorted(failures[key]))
        before, before_hash = state_input(repl, current, failures_here,
                                         episode.get("max_checks", 8) - index)
        if (before_hash != step["state_before_sha256"]
                or step["public_packet_sha256"] != digest(before)
                or step.get("input") != __import__("json").loads(before)):
            raise ValueError("live replay proof state or public packet changed")
        options = tuple(t for t in candidate_menu(repl, current)
                        if t not in failures[key])
        if (digest(options) != step["menu_sha256"] or
                step.get("candidate_menu") != list(options) or
                step["selected_index"] >= len(options) or
                options[step["selected_index"]] != step["tactic"]):
            raise ValueError("live replay tactic menu or selection differs")
        result = repl.run(sid, step["tactic"])
        if result["ok"] is False:
            if step["status"] != "rejected":
                raise ValueError("Lean replay changed from success to failure")
            failures[key].add(step["tactic"])
            if step["state_after_sha256"] != before_hash:
                raise ValueError("replayed rejected transition differs")
            continue
        sid, current = result["state"]["sid"], result["state"]
        after = public_state(repl, current)
        if digest(after) != step["state_after_sha256"] or step.get("state_after") != after:
            raise ValueError("live replay successor goal state differs")
        if after["goals"]:
            if step["status"] != "accepted_nonterminal":
                raise ValueError("live replay changed proof progress")
        else:
            cert = repl.kernel(sid)
            verified = (cert.get("ok") is True and cert.get("axioms") == [])
            if step["status"] != ("kernel_closed" if verified else "certificate_rejected"):
                raise ValueError("live kernel certificate differs in replay")
            if verified and digest(cert.get("term")) != step["kernel"]["term_sha256"]:
                raise ValueError("live replay certificate proof term differs")
            return True
    return True


def fingerprint(repl_bin: Path | None, lean_bin: Path | None, version: str):
    from .data import sha256_file
    return {"lean_version": version,
            "repl_bin_sha256": sha256_file(repl_bin) if repl_bin else None,
            "lean_bin_sha256": sha256_file(lean_bin) if lean_bin else None,
            "source": "lcgrt.LeanRepl / lgym",
            "actual_lean_process": repl_bin is not None and lean_bin is not None}
