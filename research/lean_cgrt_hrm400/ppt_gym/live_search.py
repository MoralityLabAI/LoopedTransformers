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

from .data import digest
from .live_lean import candidate_menu, choose, public_state, state_input


@torch.no_grad()
def search_episode(repl, case, model, *, policy="auto8", depth=2,
                   check_budget=8, rollback_budget=8):
    if policy not in {"auto8", "frozen"}:
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

    while spent < check_budget:
        visible = public_state(repl, state)
        if not visible["goals"]:
            raise ValueError("closed state without kernel audit")
        key = digest(visible)
        menu = tuple(t for t in candidate_menu(repl, state) if t not in failures[key])
        if not menu:
            if not frames or rolled >= rollback_budget:
                status = "no_candidates" if not frames else "rollback_limit"
                break
            frame = frames.pop()
            if frame["parent_key"] != digest(public_state(repl, frame["parent"])):
                raise ValueError("stored parent proof-state digest changed")
            step = {
                "kind": "rollback", "reason": "branch_exhausted",
                "checks_used": spent,
                "state_before_sha256": key,
                "state_after_sha256": frame["parent_key"],
                "abandoned_tactic": frame["tactic"],
                "restore_script_length": frame["script_length"],
            }
            failures[frame["parent_key"]].add(frame["tactic"])
            sid, state = frame["sid"], frame["parent"]
            script = script[:frame["script_length"]]
            rolled += 1
            events.append(step)
            continue

        text, state_digest = state_input(repl, state,
                                         tuple(sorted(failures[key])),
                                         check_budget - spent)
        index, _, used_model = choose(model, text, menu, policy=policy,
                                       remaining=check_budget - spent, depth=depth)
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
    for event in episode["events"]:
        visible = public_state(repl, state)
        key = digest(visible)
        menu = tuple(t for t in candidate_menu(repl, state) if t not in failures[key])
        if event["kind"] == "rollback":
            if menu or not stack or rolled >= episode["rollback_budget"]:
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
            continue
        if event["kind"] != "tactic":
            raise ValueError("unknown search event kind")
        before, state_digest = state_input(repl, state,
                                           tuple(sorted(failures[key])),
                                           episode["check_budget"] - spent)
        if (event["attempt"] != spent + 1 or
                event["public_packet_sha256"] != digest(before) or
                event["input"] != json.loads(before) or
                event["state_before_sha256"] != state_digest or
                event["menu_sha256"] != digest(menu) or
                event["candidate_menu"] != list(menu) or
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

    if (spent != episode["tactic_checks"] or rolled != episode["rollbacks"]
            or script != episode["accepted_script"] or spent > episode["check_budget"]):
        raise ValueError("replayed action cost or accepted proof script differs")
    return {"replay_passed": True, "tactic_checks": spent, "rollbacks": rolled}
