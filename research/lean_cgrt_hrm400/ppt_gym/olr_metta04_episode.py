"""OLR-METTA-04: real Lean staged proof controller and independent replay.

The controller makes separate decisions at root, pending-goal, failed-probe,
guard-accepted and terminal stages. Every proof-state change comes from the
native Lean server. Hyperon is an advisory eligibility check; no MeTTa
result is counted as a Lean theorem. The fixed cap is six tactic calls.
"""
from __future__ import annotations

from typing import Callable

from lcgrt.lean_repl import LeanRepl
from .data import digest
from .live_lean import candidate_menu,public_state
from .olr_dependent import DependentWorld, ROOT_MENU
from .olr_dependent_repair import offer,repair,manual
from .olr_metta04_policy import ACTIONS,LEGAL

SCHEMA="ppt.olr-metta04-sequential-native.v1"
BUDGET=6
WHITELIST={*ROOT_MENU,"rfl","apply h0","apply h1","exact h1 _ h0"}


def run_episode(lean:LeanRepl, world:DependentWorld, *,
                claims:list[list[float]], macro_allowed:bool,
                policy:Callable[[str,int,int|None],tuple[str,int]])->dict:
    """The callable returns (legal action, recurrent inference visits)."""
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    original=public_state(lean,root)
    if candidate_menu(lean,root)!=ROOT_MENU:
        raise ValueError("native dependent root legal menu changed")
    root_sha=digest(original)
    steps=[]
    spent=focus_count=restores=metta_checks=model_passes=decisions=0
    kernel_closed=False
    false_stop=False
    chosen_branch=None
    verified_branch=None
    tail=None
    phase="root"
    stop_count=0

    def public_sha():
        return digest(public_state(lean,state))

    def observe():
        return public_state(lean,state)["goals"]

    def take_tactic(action:str)->bool:
        nonlocal state,spent
        if action not in WHITELIST or spent>=BUDGET:
            raise ValueError("native dependent tool cap/tactic whitelist breached")
        before=public_sha()
        output=lean.run(state["sid"],action)
        if type(output.get("ok")) is not bool:
            raise ValueError("malformed native Lean response")
        spent+=1
        if output["ok"]:
            if not isinstance(output.get("state"),dict):
                raise ValueError("native accepted tactic without state")
            state=output["state"]
        steps.append({"kind":"tactic","action":action,"ok":output["ok"],
                      "before":before,"after":public_sha(),"calls":spent})
        return output["ok"]

    def host_focus():
        nonlocal state,focus_count
        before=public_sha()
        answer=lean.focus(state["sid"],1)
        if answer.get("ok") is not True:
            raise ValueError("native focus index 1 failed")
        state=answer["state"]
        focus_count+=1
        steps.append({"kind":"focus","index":1,"before":before,"after":public_sha()})

    def host_restore():
        nonlocal state,restores
        if restores>=1:
            raise ValueError("bounded state restore exceeded")
        before=public_sha()
        state=root
        restores+=1
        steps.append({"kind":"restore","before":before,"after":root_sha})

    def finish_kernel(continuation:str):
        nonlocal kernel_closed,phase,tail,verified_branch
        if observe():
            raise ValueError("attempted terminal proof with open obligations")
        before=public_sha()
        receipt=lean.kernel(state["sid"])
        if receipt.get("ok") is not True or receipt.get("axioms")!=[]:
            raise ValueError("dependent full kernel closure rejected")
        steps.append({"kind":"kernel","before":before,"after":public_sha(),
                      "ok":True,"axioms":[]})
        kernel_closed=True
        tail=continuation
        verified_branch=chosen_branch
        phase="closed"

    for _ in range(12):
        if phase=="done":
            break
        remaining=BUDGET-spent
        before=public_sha()
        action,visits=policy(phase,remaining,chosen_branch)
        if action not in ACTIONS or ACTIONS.index(action) not in LEGAL[phase]:
            raise ValueError(f"policy attempted illegal action {action} at {phase}")
        if type(visits) is not int or not 0<=visits<=12:
            raise ValueError("invalid model inference visit accounting")
        model_passes+=visits
        decisions+=1
        steps.append({"kind":"decision","phase":phase,"action":action,
                      "remaining":remaining,"branch":chosen_branch,
                      "before":before,"after":before,"visits":visits})
        if phase in ("root","retry"):
            if remaining==0:
                phase="done";break
            chosen_branch=0 if action=="root_r0" else 1
            if not take_tactic(f"apply r{chosen_branch}"):
                raise ValueError("registered native root application rejected")
            if len(observe())!=3:
                raise ValueError("dependent rule failed to create three subgoals")
            phase="open"
        elif phase=="open":
            if action=="stop":
                false_stop=True;phase="done"
            elif action=="probe_guard":
                if remaining<1:
                    phase="done"
                else:
                    ok=take_tactic("rfl")
                    phase="guard_ok" if ok else "failed"
            elif action=="work_first":
                if remaining<4:
                    phase="done"
                else:
                    host_focus()
                    if not take_tactic("apply h0"):
                        phase="failed"
                    elif not take_tactic("rfl"):
                        phase="failed"
                    else:
                        if not take_tactic("apply h1") or not take_tactic("apply h0"):
                            raise ValueError("indexed witness failed after native guard success")
                        finish_kernel("manual")
        elif phase=="failed":
            if action=="stop":
                false_stop=True;phase="done"
            else:
                host_restore()
                phase="retry"
        elif phase=="guard_ok":
            if action=="stop":
                false_stop=True;phase="done"
            elif action=="metta_repair":
                start=public_sha()
                outcome=repair(lean,state,budget=remaining,eligible=macro_allowed)
                metta_checks+=1
                steps.append({"kind":"metta_offer","before":start,"after":start,
                    "budget":remaining,"eligible":macro_allowed,
                    "accepted":outcome["admission"]["offered"],
                    "program_sha256":outcome["admission"]["program_sha256"]})
                for event in outcome["events"]:
                    e={"kind":event["op"],"before":event["before"],"after":event["after"]}
                    if event["op"]=="tactic":
                        spent+=1
                        e.update({"action":event["tactic"],"ok":event["ok"],"calls":spent})
                    else:
                        e.update({"ok":event["ok"],"axioms":event["axioms"]})
                    steps.append(e)
                if outcome["committed"]:
                    state=outcome["state"]
                    if spent>BUDGET:
                        raise ValueError("MeTTa repair exceeded the native call budget")
                    kernel_closed=True;tail="fast";verified_branch=chosen_branch
                    phase="closed"
                else:
                    # An unoffered or vetoed speculative macro cannot commit.
                    phase="done"
            elif action=="manual":
                outcome=manual(lean,state,budget=remaining)
                for event in outcome["events"]:
                    e={"kind":event["op"],"before":event["before"],"after":event["after"]}
                    if event["op"]=="tactic":
                        spent+=1
                        e.update({"action":event["tactic"],"ok":event["ok"],"calls":spent})
                    else:
                        e.update({"ok":event["ok"],"axioms":event["axioms"]})
                    steps.append(e)
                if outcome["committed"]:
                    state=outcome["state"];tail="manual"
                    verified_branch=chosen_branch
                    kernel_closed=True;phase="closed"
                else:
                    phase="done"
        elif phase=="closed":
            if action!="stop":
                raise ValueError("a certified theorem must terminate with STOP")
            stop_count+=1
            phase="done"
        else:
            raise ValueError("unknown recurrent decision phase")
        if spent>BUDGET:
            raise ValueError("source-native policy exceeded six actions")
    else:
        raise ValueError("native staged controller exceeded decision bound")
    return {
        "schema":SCHEMA,"root_id":world.root_id,
        "statement_sha256":digest(world.statement()),
        "initial_public_sha256":root_sha,
        "status":"kernel_closed" if kernel_closed else ("premature_stop" if false_stop else "unfinished"),
        "kernel_closed":kernel_closed,"false_stop":false_stop,
        "verified_branch":verified_branch,"tail":tail,
        "tactic_calls":spent,"focus_ops":focus_count,"restores":restores,
        "metta_queries":metta_checks,"model_decisions":decisions,
        "model_passes":model_passes,"terminal_stops":stop_count,
        "events":steps,"utility":round(float(kernel_closed)-.07*spent,8),
    }


def replay_episode(lean:LeanRepl,world:DependentWorld,ep:dict,
                   *,macro_allowed:bool)->bool:
    """Re-execute every native action, focus, saved-state restore and kernel."""
    if ep.get("schema")!=SCHEMA or ep.get("root_id")!=world.root_id:
        raise ValueError("native replay root or schema changed")
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    root_sha=digest(public_state(lean,root))
    if root_sha!=ep["initial_public_sha256"]:
        raise ValueError("replay public root differs")
    spent=focuses=restores=queries=visits=decisions=stops=0
    closed=False
    for event in ep["events"]:
        old=digest(public_state(lean,state))
        if old!=event["before"]:
            raise ValueError("native replay predecessor proof state differs")
        kind=event["kind"]
        if kind=="decision":
            if event["action"] not in ACTIONS or event["phase"] not in LEGAL:
                raise ValueError("native replay has invalid policy control event")
            if ACTIONS.index(event["action"]) not in LEGAL[event["phase"]]:
                raise ValueError("recorded policy action was invalid for stage")
            if event["remaining"]!=BUDGET-spent:
                raise ValueError("recorded policy observed wrong call budget")
            decisions+=1
            visits+=event["visits"]
        elif kind=="tactic":
            if event["action"] not in WHITELIST:
                raise ValueError("native replay tool action unwhitelisted")
            reply=lean.run(state["sid"],event["action"])
            spent+=1
            if reply["ok"]!=event["ok"] or spent!=event["calls"]:
                raise ValueError("independent Lean tactic verdict or call count changed")
            if reply["ok"]:
                state=reply["state"]
        elif kind=="focus":
            if event["index"]!=1:
                raise ValueError("unregistered pending-goal focus")
            response=lean.focus(state["sid"],1)
            if response.get("ok") is not True:
                raise ValueError("native focus rejected")
            state=response["state"]
            focuses+=1
        elif kind=="restore":
            if restores>=1:
                raise ValueError("too many native saved-state restores")
            state=root;restores+=1
        elif kind=="metta_offer":
            if event["eligible"]!=macro_allowed or event["budget"]!=BUDGET-spent:
                raise ValueError("native MeTTa offer's public budget/eligibility changed")
            check=offer(public_state(lean,state),
                        remaining_calls=BUDGET-spent,eligible=macro_allowed)
            if (check["offered"]!=event["accepted"]
                    or check["program_sha256"]!=event["program_sha256"]):
                raise ValueError("native MeTTa eligibility/provenance reply changed")
            queries+=1
        elif kind=="kernel":
            cert=lean.kernel(state["sid"])
            if cert.get("ok") is not True or cert.get("axioms")!=[]:
                raise ValueError("independent Lean kernel no longer certifies proof")
            closed=True
        else:
            raise ValueError("unknown policy/native replay event")
        if digest(public_state(lean,state))!=event["after"]:
            raise ValueError("native replay successor proof state differs")
    if (spent!=ep["tactic_calls"] or focuses!=ep["focus_ops"]
            or restores!=ep["restores"] or queries!=ep["metta_queries"]
            or visits!=ep["model_passes"] or decisions!=ep["model_decisions"]
            or closed!=ep["kernel_closed"] or spent>BUDGET):
        raise ValueError("native replay resource or certificate receipt differs")
    if round(float(closed)-.07*spent,8)!=ep["utility"]:
        raise ValueError("native replay utility differs")
    return True
