"""LSA-02I bounded native Lean proof search with an explicit RECONSIDER operation.

The only live signals are public Lean goals, syntactically proposed tactics,
Lean's verdict on each attempted tactic, and visited-state history. The
model-free reference solver and generator metadata never enter the online
chooser. Every tactic costs one effort unit and every branch rollback costs
one effort unit. A successful transcript's *final script* excludes abandoned
branches and is separately compiled as an axiom-free Lean declaration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import torch

from .data import canonical, digest
from lcgrt.extract import goal_key
from .live_lean import candidate_menu, choose, public_state, state_input
from .uniform_baseline import _root_seed


SCHEMA = "lsa02i.native_backtrack_episode.v1"
BUDGET = 16
POLICIES = ("frozen_greedy", "frozen_dfs", "auto_dfs", "uniform_dfs")


@dataclass
class _Frame:
    state: dict
    script: tuple[str, ...]
    tried: set[str] = field(default_factory=set)
    public_sha: str = ""


def _stable_state(repl, state):
    return digest(public_state(repl, state))


def _stack_init(repl, case):
    repl.reset()
    root = repl.init(case.statement, pp=True)
    if not root["goals"] or not isinstance(root.get("sid"), int):
        raise ValueError("native proof root has missing goals or state id")
    return [_Frame(root, (), set(), _stable_state(repl, root))]


def _candidates(repl, frames):
    node = frames[-1]
    menu = tuple(t for t in candidate_menu(repl, node.state) if t not in node.tried)
    ancestor = node.public_sha in {f.public_sha for f in frames[:-1]}
    return menu, ancestor


@torch.no_grad()
def run_episode(repl, case, model, *, policy: str, effort_budget: int = BUDGET,
                depth: int = 2, random_seed: int = 0) -> dict:
    if policy not in POLICIES or type(effort_budget) is not int or not 1 <= effort_budget <= 32:
        raise ValueError("registered search policy and bounded effort budget required")
    if policy.startswith("frozen_") and model is None:
        raise ValueError("frozen proof search requires an actual trained checkpoint")
    if not 1 <= depth <= 16 or not 0 <= random_seed < 2**32:
        raise ValueError("invalid recursion depth or unsigned random seed")
    stack = _stack_init(repl, case)
    allow_rollback = policy != "frozen_greedy"
    model_calls = tactic_checks = reconsiderations = kernel_calls = 0
    events = []
    terminal = "budget_exhausted"
    kernel_verified = False
    source_seed = _root_seed(case.root, random_seed) if policy == "uniform_dfs" else None
    for used in range(effort_budget):
        node = stack[-1]
        current = public_state(repl, node.state)
        if not current["goals"]:
            raise ValueError("nonterminal proof loop entered a previously closed state")
        options, cycle = _candidates(repl, stack)
        before = node.public_sha
        remaining = effort_budget - used
        packet, visible_sha = state_input(
            repl, node.state, tuple(sorted(node.tried)), min(8, remaining),
        )
        if visible_sha != before:
            raise ValueError("public state fingerprint changed within a proof frame")
        # A repeated public state on the current proof path represents a
        # cycle; retrying it is unproductive without changing an ancestor.
        reason = "cycle" if cycle else "exhausted_menu" if not options else None
        if reason is not None:
            if not allow_rollback or len(stack) == 1:
                terminal = "cycle_stuck" if cycle else "no_candidates"
                break
            discarded = stack.pop()
            parent = stack[-1]
            reconsiderations += 1
            event = {
                "unit": used + 1, "kind": "RECONSIDER",
                "reason": reason,
                "state_before_sha256": before,
                "state_after_sha256": parent.public_sha,
                "abandoned_prefix": list(discarded.script),
                "script_after": list(parent.script),
                "spent_work_units": used + 1,
                "stack_depth_after": len(stack),
            }
            events.append(event)
            continue

        policy_mode = (
            "frozen" if policy.startswith("frozen_") else
            "auto8" if policy == "auto_dfs" else "uniform"
        )
        if policy_mode == "uniform":
            # The live public packet, available menu, remaining budget and
            # root-specific fixed seed determine a uniform decision.
            index, _, used_model = choose(
                None, packet, options, policy="uniform",
                remaining=min(8, remaining), depth=depth,
                uniform_seed=source_seed,
            )
        else:
            index, _, used_model = choose(
                model, packet, options, policy=policy_mode,
                remaining=min(8, remaining), depth=depth,
            )
        model_calls += int(used_model)
        if not 0 <= index < len(options):
            raise ValueError("the policy chose outside the public candidate menu")
        tactic = options[index]
        node.tried.add(tactic)
        answer = repl.run(node.state["sid"], tactic)
        tactic_checks += 1
        if not isinstance(answer, dict) or type(answer.get("ok")) is not bool:
            raise ValueError("Lean returned a malformed tactic verdict")
        entry = {
            "unit": used + 1, "kind": "TACTIC",
            "state_before_sha256": before,
            "active_goal": goal_key(current["goals"][0]),
            "public_packet_sha256": digest(packet),
            "available_menu": list(options),
            "menu_sha256": digest(options),
            "tactic": tactic, "selected_index": index,
            "lean_accepted": answer["ok"],
            "spent_work_units": used + 1,
        }
        if answer["ok"] is False:
            entry.update(
                state_after_sha256=before, script_after=list(node.script),
                stack_depth_after=len(stack), status="rejected",
                error=str(answer.get("error", ""))[:256],
            )
            events.append(entry)
            continue
        child = answer.get("state")
        if not isinstance(child, dict) or not isinstance(child.get("sid"), int):
            raise ValueError("Lean accepted a tactic without a state identifier")
        child_sha = _stable_state(repl, child)
        accepted = (*node.script, tactic)
        entry.update(
            state_after_sha256=child_sha, script_after=list(accepted),
            stack_depth_after=len(stack) + 1,
        )
        if not child["goals"]:
            kernel_calls += 1
            kernel = repl.kernel(child["sid"])
            kernel_verified = bool(
                kernel.get("ok") is True and kernel.get("axioms") == []
            )
            entry["status"] = "kernel_closed" if kernel_verified else "kernel_rejected"
            if kernel_verified:
                terminal = "kernel_closed"
            else:
                terminal = "kernel_rejected"
            stack.append(_Frame(child, accepted, set(), child_sha))
            events.append(entry)
            break
        entry["status"] = "accepted_nonterminal"
        stack.append(_Frame(child, accepted, set(), child_sha))
        events.append(entry)
    if tactic_checks + reconsiderations != len(events):
        raise AssertionError("every search operation must cost exactly one unit")
    if tactic_checks + reconsiderations > effort_budget:
        raise AssertionError("search exceeded the registered effort allowance")
    last = stack[-1]
    return {
        "schema": SCHEMA,
        "root": case.root, "lineage": case.lineage, "family": case.family,
        "split": case.split, "policy": policy, "effort_budget": effort_budget,
        "depth": depth, "uniform_seed": random_seed if policy == "uniform_dfs" else None,
        "terminal_status": terminal, "live_kernel": kernel_verified,
        "accepted_script": list(last.script),
        "tactic_checks": tactic_checks,
        "backtracks": reconsiderations,
        "effort_spent": tactic_checks + reconsiderations,
        "model_calls": model_calls,
        "kernel_calls": kernel_calls,
        "events": events,
        "standalone_verified": False,
        "scientific_eligible": False,
    }


def replay_episode(repl, case, episode: dict) -> bool:
    """Independently reproduce every tactic and rollback without the model.

    Authenticates public menus, Lean acceptance, path history, work units,
    stack rewinds, terminal kernel closure and the final accepted script.
    """
    if episode.get("schema") != SCHEMA or episode.get("root") != case.root:
        raise ValueError("replay root or episode version differs")
    if episode.get("lineage") != case.lineage or episode.get("split") != case.split:
        raise ValueError("replay source lineage differs")
    policy = episode["policy"]
    if policy not in POLICIES:
        raise ValueError("unknown replay policy")
    stack = _stack_init(repl, case)
    tactics = backtracks = model_calls = 0
    closed = False
    for idx, event in enumerate(episode["events"]):
        if event.get("unit") != idx + 1 or idx + 1 > episode["effort_budget"]:
            raise ValueError("replay effort-unit chronology invalid")
        node = stack[-1]
        current_sha = node.public_sha
        options, cycle = _candidates(repl, stack)
        if event.get("state_before_sha256") != current_sha:
            raise ValueError("replay public-state predecessor fingerprint differs")
        kind = event.get("kind")
        if kind == "RECONSIDER":
            if policy == "frozen_greedy" or len(stack) < 2:
                raise ValueError("replay contains unauthorized backtracking")
            reason = "cycle" if cycle else "exhausted_menu" if not options else None
            if reason is None or event.get("reason") != reason:
                raise ValueError("replay backtrack lacks a public exhaustion/cycle witness")
            discarded = stack.pop()
            after = stack[-1]
            if (event.get("state_after_sha256") != after.public_sha
                or event.get("abandoned_prefix") != list(discarded.script)
                or event.get("script_after") != list(after.script)
                or event.get("stack_depth_after") != len(stack)):
                raise ValueError("backtrack replay restored a different script or stack")
            backtracks += 1
            continue
        if kind != "TACTIC" or cycle or not options:
            raise ValueError("replay tactic is not legal at this public proof state")
        tactic = event.get("tactic")
        if event.get("active_goal") != goal_key(public_state(repl, node.state)["goals"][0]):
            raise ValueError("replay public active proof target changed")
        if tactic not in options or event.get("available_menu") != list(options):
            raise ValueError("replay candidate menu changed")
        if (event.get("menu_sha256") != digest(options)
            or event.get("selected_index") != options.index(tactic)):
            raise ValueError("replay tactic selection is inconsistent with the menu")
        packet, visible_sha = state_input(
            repl, node.state, tuple(sorted(node.tried)),
            min(8, episode["effort_budget"] - idx),
        )
        if (event.get("public_packet_sha256") != digest(packet)
            or visible_sha != current_sha):
            raise ValueError("replay model-visible public input changed")
        node.tried.add(tactic)
        answer = repl.run(node.state["sid"], tactic)
        if type(answer.get("ok")) is not bool or event.get("lean_accepted") != answer["ok"]:
            raise ValueError("replay Lean tactic verdict differs")
        tactics += 1
        if answer["ok"] is False:
            if (event.get("state_after_sha256") != current_sha
                or event.get("script_after") != list(node.script)
                or event.get("status") != "rejected"
                or event.get("stack_depth_after") != len(stack)):
                raise ValueError("rejected Lean tactic replay disagrees")
            continue
        child = answer["state"]
        child_sha = _stable_state(repl, child)
        script = (*node.script, tactic)
        if (event.get("state_after_sha256") != child_sha
            or event.get("script_after") != list(script)
            or event.get("stack_depth_after") != len(stack) + 1):
            raise ValueError("native proof successor changed during replay")
        stack.append(_Frame(child, script, set(), child_sha))
        if not child["goals"]:
            result = repl.kernel(child["sid"])
            closed = result.get("ok") is True and result.get("axioms") == []
            if event.get("status") != ("kernel_closed" if closed else "kernel_rejected"):
                raise ValueError("terminal kernel verdict changed during replay")
            if idx + 1 != len(episode["events"]):
                raise ValueError("replay transcript continues after terminal proof")
            break
        if event.get("status") != "accepted_nonterminal":
            raise ValueError("nonterminal Lean acceptance replay differs")
    if len(episode["events"]) != tactics + backtracks:
        raise ValueError("unaccounted proof-search operation in transcript")
    if (episode.get("tactic_checks") != tactics
        or episode.get("backtracks") != backtracks
        or episode.get("effort_spent") != tactics + backtracks
        or episode.get("accepted_script") != list(stack[-1].script)
        or episode.get("live_kernel") != closed):
        raise ValueError("replay total, terminal proof script or closure mismatch")
    return True
