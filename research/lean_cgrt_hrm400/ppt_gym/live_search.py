"""PPT-G03B: verifier-backed depth-first proof search with bounded rollback.

A rollback restores a previously returned immutable lgym state ID. It costs a
host search decision but no Lean tactic execution call. The accounting records
both separately. All tactic actions remain restricted to public Lean premises.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import torch

from .data import canonical, digest
from .live_lean import candidate_menu, choose, public_state, state_input
from .bundled_rewrite import BPD_BUNDLE_FAMILY, REWRITE_FAMILIES, bundle_continuation
from .rewrite import remember_rewrite_target, unvisited_rewrites


def _support_probe(visible, menu):
    if not menu:
        return None
    hypotheses = visible["goals"][0]["hypotheses"]
    return next(("apply " + h["name"] for h in hypotheses
                 if h["name"].startswith("h") and h["name"][1:].isdigit()
                 and "apply " + h["name"] not in menu), None)


def _recovery_observation(repl, state, menu, frames, spent, budget, *, rewrite=False):
    if spent != 1 or not menu or len(frames) != 1:
        return None
    frame = frames[0]
    options = list(candidate_menu(repl, frame["parent"], rewrite=rewrite))
    if len(options) != 2 or frame["tactic"] not in options:
        return None
    current, _ = state_input(repl, state, (), budget - spent)
    root, _ = state_input(repl, frame["parent"], (), budget)
    return {
        "current": json.loads(current), "parent": json.loads(root),
        "parent_menu": options, "explored": options.index(frame["tactic"]),
    }


def _rollback_record(repl, frame, key, spent, reason):
    if frame["parent_key"] != digest(public_state(repl, frame["parent"])):
        raise ValueError("stored parent proof-state digest changed")
    return {
        "kind": "rollback", "reason": reason, "checks_used": spent,
        "state_before_sha256": key, "state_after_sha256": frame["parent_key"],
        "abandoned_tactic": frame["tactic"], "restore_script_length": frame["script_length"],
    }


@torch.no_grad()
def search_episode(repl, case, model, *, policy="auto8", depth=2,
                   check_budget=8, rollback_budget=8, ranker=None, within_budget=None,
                   probe=False, recovery_ranker=None, rewrite_exploration=False):
    if ranker is None and policy not in {"auto8", "frozen"}:
        raise ValueError("first backtracking study supports auto8 and frozen ranker")
    if not (1 <= check_budget <= 32 and 0 <= rollback_budget <= check_budget
            and 1 <= depth <= 16):
        raise ValueError("invalid bounded search budget")
    repl.reset()
    state = repl.init(case.statement, pp=True)
    sid = state["sid"]
    failures: dict[str, set[str]] = defaultdict(set)
    frames: list[dict] = []
    events = []
    script: list[str] = []
    spent = rolled = model_calls = kernel_calls = 0
    status = "budget_exhausted"
    started = time.perf_counter()
    considered_recovery = False
    rewrite = case.family in REWRITE_FAMILIES
    bundled = case.family == BPD_BUNDLE_FAMILY
    if rewrite_exploration and (not rewrite or bundled or ranker is None or probe or recovery_ranker is not None):
        raise ValueError("rewrite exploration requires ranked original BPD equality search")
    visited = set()

    while spent < check_budget:
        if within_budget is not None and not within_budget():
            status = "compute_budget_exhausted"
            break
        visible = public_state(repl, state)
        if not visible["goals"]:
            raise ValueError("closed state without kernel audit")
        if rewrite_exploration:
            remember_rewrite_target(visible, visited)
        key = digest(visible)
        menu = tuple(t for t in candidate_menu(repl, state, rewrite=rewrite, bundles=bundled) if t not in failures[key])
        if recovery_ranker is not None and not considered_recovery and rolled < rollback_budget:
            recovery = _recovery_observation(repl, state, menu, frames, spent, check_budget, rewrite=rewrite)
            if recovery is not None:
                decision, used_model = recovery_ranker(canonical(recovery))
                if decision is None or (within_budget is not None and not within_budget()):
                    status = "compute_budget_exhausted"
                    break
                if decision not in (0, 1):
                    raise ValueError("expected keep or abandon at recovery checkpoint")
                considered_recovery = True
                model_calls += int(used_model)
                events.append({"kind": "recovery", "input": recovery, "selected_index": decision,
                               "checks_used": spent, "state_before_sha256": key})
                if decision == 1:
                    frame = frames.pop()
                    events.append(_rollback_record(repl, frame, key, spent, "early_recovery"))
                    failures[frame["parent_key"]].add(frame["tactic"])
                    sid, state = frame["sid"], frame["parent"]
                    script = script[:frame["script_length"]]
                    rolled += 1
                    continue
        if not menu:
            if not frames or rolled >= rollback_budget:
                status = "no_candidates" if not frames else "rollback_limit"
                break
            frame = frames.pop()
            step = _rollback_record(repl, frame, key, spent, "branch_exhausted")
            failures[frame["parent_key"]].add(frame["tactic"])
            sid, state = frame["sid"], frame["parent"]
            script = script[:frame["script_length"]]
            rolled += 1
            events.append(step)
            continue

        text, state_digest = state_input(repl, state,
                                         tuple(sorted(failures[key])),
                                         check_budget - spent)
        exploratory = _support_probe(visible, menu) if probe and spent == 0 else None
        choices = unvisited_rewrites(visible["goals"][0], menu, visited) if rewrite_exploration else None
        if exploratory:
            menu = (*menu[:63], exploratory)
            index, used_model = len(menu) - 1, False
        elif ranker is None:
            index, _, used_model = choose(model, text, menu, policy=policy,
                                           remaining=check_budget - spent, depth=depth)
        else:
            if choices is None:
                index, used_model = ranker(text, menu, check_budget - spent)
            else:
                index, used_model = ranker(text, menu, check_budget - spent, rewrite_choices=choices)
            if index is None or (within_budget is not None and not within_budget()):
                status = "compute_budget_exhausted"
                break
            if not 0 <= index < len(menu):
                raise ValueError("ranker selected a tactic outside the public menu")
            if choices is not None and index not in choices:
                raise ValueError("rewrite exploration selected a previously visited target")
        model_calls += int(used_model)
        tactic = menu[index]
        before_sid, before_state, prefix_len = sid, state, len(script)
        reply = repl.run(sid, tactic)
        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
            raise ValueError("malformed native Lean tactic reply")
        spent += 1
        event = {
            "kind": "tactic", "attempt": spent, "policy": policy,
            "public_packet_sha256": digest(text),
            "input": json.loads(text),
            "menu_sha256": digest(menu),
            "candidate_menu": list(menu),
            "selected_index": index, "tactic": tactic,
            "state_before_sha256": state_digest,
        }
        if choices is not None:
            event["rewrite_choices"] = choices
        if reply["ok"] is False:
            event.update(status="rejected", error=str(reply.get("error", ""))[:512],
                         state_after_sha256=state_digest)
            failures[key].add(tactic)
            events.append(event)
            continue

        after = reply.get("state")
        if not isinstance(after, dict) or type(after.get("sid")) is not int:
            raise ValueError("Lean accepted tactic with missing successor state")
        sid, state = after["sid"], after
        script.append(tactic)
        new = public_state(repl, state)
        event["state_after_sha256"] = digest(new)
        event["state_after"] = new
        if new["goals"]:
            frames.append({
                "sid": before_sid, "parent": before_state, "parent_key": key,
                "tactic": tactic, "script_length": prefix_len,
            })
            event["status"] = "accepted_nonterminal"
        else:
            kernel_calls += 1
            cert = repl.kernel(sid)
            good = (isinstance(cert, dict) and cert.get("ok") is True
                    and cert.get("axioms") == [])
            event["status"] = "kernel_closed" if good else "certificate_rejected"
            event["kernel"] = {
                "ok": good, "axioms": cert.get("axioms") if isinstance(cert, dict) else None,
                "term_sha256": digest(cert.get("term")) if good else None,
                "error": str(cert.get("error", ""))[:512] if isinstance(cert, dict) else "invalid",
            }
            status = event["status"]
        events.append(event)
        if status in ("kernel_closed", "certificate_rejected"):
            break

    if spent > check_budget or rolled > rollback_budget:
        raise ValueError("internal budget accounting error")
    return {
        "root": case.root, "lineage": case.lineage, "split": case.split,
        "family": case.family, "statement_sha256": digest(case.statement),
        "policy": policy, "status": status, "check_budget": check_budget,
        "rollback_budget": rollback_budget, "depth": depth,
        "tactic_checks": spent, "rollbacks": rolled, "model_calls": model_calls,
        "kernel_calls": kernel_calls, "accepted_script": script, "events": events,
        "live_kernel": status == "kernel_closed",
        "standalone_certificate": None, "verified_closed": False,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "replay_schema": "ppt_gym.backtracking_episode.v1",
        **({"support_probe": True} if probe else {}),
        **({"early_recovery_enabled": True} if recovery_ranker is not None else {}),
        **({"rewrite_exploration": True} if rewrite_exploration else {}),
    }


def replay_search(repl, case, episode):
    """Independently replay every action and rollback. Learner weights not needed."""
    if episode.get("replay_schema") != "ppt_gym.backtracking_episode.v1":
        raise ValueError("unknown search replay schema")
    if episode.get("check_budget", 0) < 1 or not 0 <= episode.get("rollback_budget", -1) <= episode["check_budget"]:
        raise ValueError("replay has invalid search budgets")
    repl.reset()
    state = repl.init(case.statement, pp=True)
    sid = state["sid"]
    stack = []
    failures: dict[str, set[str]] = defaultdict(set)
    script = []
    spent = rolled = 0
    pending_rollback = considered_recovery = False
    rewrite = case.family in REWRITE_FAMILIES
    bundled = case.family == BPD_BUNDLE_FAMILY
    fixed_continuation = episode.get("bundle_continuation", False)
    if fixed_continuation and not bundled:
        raise ValueError("recorded bundle continuation requires a bundled equality case")
    exploration = episode.get("rewrite_exploration", False)
    if exploration and (not rewrite or bundled or episode.get("support_probe") or episode.get("early_recovery_enabled")):
        raise ValueError("invalid recorded rewrite exploration")
    visited = set()
    for event in episode["events"]:
        visible = public_state(repl, state)
        if exploration:
            remember_rewrite_target(visible, visited)
        key = digest(visible)
        menu = tuple(t for t in candidate_menu(repl, state, rewrite=rewrite, bundles=bundled) if t not in failures[key])
        if pending_rollback and (event["kind"] != "rollback" or event.get("reason") != "early_recovery"):
            raise ValueError("recovery decision did not restore its recorded checkpoint")
        if event["kind"] == "recovery":
            observation = _recovery_observation(repl, state, menu, stack, spent, episode["check_budget"], rewrite=rewrite)
            if (not episode.get("early_recovery_enabled") or considered_recovery or observation is None
                    or rolled >= episode["rollback_budget"] or event["input"] != observation
                    or event["checks_used"] != spent or event["state_before_sha256"] != key
                    or event["selected_index"] not in (0, 1)):
                raise ValueError("invalid early recovery decision")
            considered_recovery = True
            pending_rollback = event["selected_index"] == 1
            continue
        if event["kind"] == "rollback":
            reason = "early_recovery" if pending_rollback else "branch_exhausted"
            if (event.get("reason") != reason or (menu and not pending_rollback)
                    or not stack or rolled >= episode["rollback_budget"]):
                raise ValueError("recorded rollback was not legal in live state")
            frame = stack.pop()
            if (event["state_before_sha256"] != key or
                    event["state_after_sha256"] != frame["parent_key"] or
                    event["abandoned_tactic"] != frame["tactic"] or
                    event["restore_script_length"] != frame["script_length"] or
                    event["checks_used"] != spent):
                raise ValueError("rollback proof-state witness differs")
            failures[frame["parent_key"]].add(frame["tactic"])
            sid, state = frame["sid"], frame["parent"]
            script = script[:frame["script_length"]]
            rolled += 1
            pending_rollback = False
            continue
        if event["kind"] != "tactic":
            raise ValueError("unknown search event kind")
        if episode.get("support_probe") and spent == 0:
            exploratory = _support_probe(visible, menu)
            if exploratory:
                menu = (*menu[:63], exploratory)
                if event["tactic"] != exploratory:
                    raise ValueError("recorded support probe differs from public context")
        before, state_digest = state_input(repl, state,
                                           tuple(sorted(failures[key])),
                                           episode["check_budget"] - spent)
        choices = unvisited_rewrites(visible["goals"][0], menu, visited) if exploration else None
        if (event["attempt"] != spent + 1 or
                event["public_packet_sha256"] != digest(before) or
                event["input"] != json.loads(before) or
                event["state_before_sha256"] != state_digest or
                event["menu_sha256"] != digest(menu) or
                event["candidate_menu"] != list(menu) or
                event.get("rewrite_choices") != choices or
                (choices is not None and event["selected_index"] not in choices) or
                (fixed_continuation and spent > 0
                 and event["selected_index"] != bundle_continuation(visible["goals"][0], menu)) or
                not 0 <= event["selected_index"] < len(menu) or
                menu[event["selected_index"]] != event["tactic"]):
            raise ValueError("replayed decision differs from recorded public state")
        before_sid, before_state, old_len = sid, state, len(script)
        result = repl.run(sid, event["tactic"])
        spent += 1
        if result["ok"] is False:
            if event["status"] != "rejected" or event["state_after_sha256"] != state_digest:
                raise ValueError("replayed tactic rejection differs")
            failures[key].add(event["tactic"])
            continue
        sid, state = result["state"]["sid"], result["state"]
        new = public_state(repl, state)
        script.append(event["tactic"])
        if event["state_after_sha256"] != digest(new) or event.get("state_after") != new:
            raise ValueError("replayed successor goals differ")
        if new["goals"]:
            if event["status"] != "accepted_nonterminal":
                raise ValueError("replayed nonterminal status differs")
            stack.append({"sid": before_sid, "parent": before_state,
                          "parent_key": key, "tactic": event["tactic"],
                          "script_length": old_len})
        else:
            cert = repl.kernel(sid)
            verified = cert.get("ok") is True and cert.get("axioms") == []
            if event["status"] != ("kernel_closed" if verified else "certificate_rejected"):
                raise ValueError("replayed kernel status differs")
            if verified and event["kernel"]["term_sha256"] != digest(cert.get("term")):
                raise ValueError("replayed kernel term differs")

    if (pending_rollback or spent != episode["tactic_checks"] or rolled != episode["rollbacks"]
            or script != episode["accepted_script"] or spent > episode["check_budget"]):
        raise ValueError("replayed action cost or accepted proof script differs")
    return {"replay_passed": True, "tactic_checks": spent, "rollbacks": rolled}
