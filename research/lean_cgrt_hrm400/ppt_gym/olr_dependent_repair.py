"""OLR-METTA-04: advisory dependent-witness repair with native Hyperon.

All source checks are public, syntactic and deliberately conservative.
Lean 4.19 remains the only proof authority. The two-call macro is offered
only if the observable eligibility/budget/lineage facts authorize it.
"""
from __future__ import annotations

import re
from typing import Any

from .data import digest
from .live_lean import public_state
from .metta_obligations import _native_metta, REPAIR_QUERY, REPAIR_OFFER

FAST = ("apply h0", "exact h1 _ h0")
FULL = ("apply h0", "apply h1", "apply h0")


def typed_pending(public:dict)->bool:
    """Check that exactly P n and Q n are pending with accessible h0, h1.

    This is a shape check, not an arithmetic truth oracle.
    """
    goals=public.get("goals")
    if not isinstance(goals,list) or len(goals)!=2:
        return False
    for role,goal in zip(("P","Q"),goals,strict=True):
        if not isinstance(goal,dict):
            return False
        term=goal.get("target")
        if not isinstance(term,dict) or type(term.get("raw")) is not str:
            return False
        # Exact parser varies in cosmetic parentheses; require a named
        # predicate application but leave index conversion to Lean.
        raw=term["raw"].strip()
        if not (raw.startswith(role+" ") or raw.startswith("("+role+" ")):
            return False
        hyps=goal.get("hypotheses")
        if not isinstance(hyps,list):
            return False
        if not {"h0","h1"}.issubset({
                h.get("name") for h in hyps if isinstance(h,dict)}):
            return False
    return True


def offer(public:dict, *, remaining_calls:int, eligible:bool,
          lineage_unique:bool=True,current_revision:bool=True) -> dict:
    """Real Hyperon policy query with exact same Python reference predicates."""
    if not 0<=remaining_calls<=6:
        raise ValueError("invalid pending-proof call budget")
    if type(eligible) is not bool:
        raise ValueError("public eligibility must be Boolean")
    typed=typed_pending(public)
    affordable=remaining_calls>=len(FAST)
    values=(
        "Typed" if typed else "Untyped",
        "WithinBudget" if affordable else "OverBudget",
        "UniqueLineage" if lineage_unique else "RepeatedLineage",
        "CurrentRevision" if current_revision else "StaleRevision",
        "DependenciesMet" if typed else "MissingDependencies",
        "ConflictFree" if eligible else "Conflict",
    )
    fact="(Repair dep "+" ".join((*values,str(len(FAST))))+")"
    program=fact+"\n"+REPAIR_QUERY+"\n"
    results,sha=_native_metta(program,offer_pattern=REPAIR_OFFER)
    expected=(("dep",str(len(FAST))),) if (typed and affordable and eligible
                                          and lineage_unique and current_revision) else ()
    if results!=expected:
        raise ValueError("native MeTTa dependent repair diverged from typed reference")
    return {"offered":bool(results),"program_sha256":sha,
            "public_state_sha256":digest(public),
            "tactic_budget":remaining_calls,
            "source_eligible":eligible,"typed_obligations":typed,
            "lineage_unique":lineage_unique,"current_revision":current_revision}


def repair(repl:Any,state:dict,*,budget:int,eligible:bool) -> dict:
    """Atomic two-tactic Lean repair; returned state commits only on kernel pass."""
    if type(state.get("sid")) is not int:
        raise ValueError("repair requires a saved native Lean proof state")
    original=public_state(repl,state)
    decision=offer(original,remaining_calls=budget,eligible=eligible)
    if not decision["offered"]:
        return {"status":"no_offer","committed":False,"state":state,
                "calls":0,"events":[],"admission":decision}
    new=state
    events=[]
    for tactic in FAST:
        before=digest(public_state(repl,new))
        result=repl.run(new["sid"],tactic)
        if type(result.get("ok")) is not bool:
            raise ValueError("malformed native Lean tactic")
        if result["ok"]:
            after=result.get("state")
            if not isinstance(after,dict):
                raise ValueError("native tactic accepted without state")
            new=after
        events.append({"op":"tactic","tactic":tactic,"ok":result["ok"],
                       "before":before,"after":digest(public_state(repl,new))})
        if not result["ok"]:
            return {"status":"tactic_rejected","committed":False,"state":state,
                    "calls":len(events),"events":events,"admission":decision}
    if public_state(repl,new)["goals"]:
        return {"status":"pending","committed":False,"state":state,
                "calls":len(events),"events":events,"admission":decision}
    check=repl.kernel(new["sid"])
    if check.get("ok") is not True or check.get("axioms")!=[]:
        return {"status":"kernel_rejected","committed":False,"state":state,
                "calls":len(events),"events":events,"admission":decision}
    events.append({"op":"kernel","before":digest(public_state(repl,new)),
                   "after":digest(public_state(repl,new)),"ok":True,"axioms":[]})
    return {"status":"kernel_closed","committed":True,"state":new,
            "calls":len(FAST),"events":events,"admission":decision}


def manual(repl:Any,state:dict,*,budget:int) -> dict:
    """Strong direct reference, same native allowed script without MeTTa."""
    if budget<len(FULL):
        return {"status":"over_budget","committed":False,"state":state,
                "calls":0,"events":[]}
    new=state
    events=[]
    for tactic in FULL:
        before=digest(public_state(repl,new))
        result=repl.run(new["sid"],tactic)
        if type(result.get("ok")) is not bool:
            raise ValueError("malformed native manual tactic")
        if result["ok"]:
            new=result["state"]
        events.append({"op":"tactic","tactic":tactic,"ok":result["ok"],
                       "before":before,"after":digest(public_state(repl,new))})
        if not result["ok"]:
            return {"status":"tactic_rejected","committed":False,"state":state,
                    "calls":len(events),"events":events}
    if public_state(repl,new)["goals"]:
        raise ValueError("manual dependent source incomplete")
    cert=repl.kernel(new["sid"])
    if cert.get("ok") is not True or cert.get("axioms")!=[]:
        raise ValueError("native dependent manual certificate rejected")
    events.append({"op":"kernel","before":digest(public_state(repl,new)),
                   "after":digest(public_state(repl,new)),"ok":True,"axioms":[]})
    return {"status":"kernel_closed","committed":True,"state":new,
            "calls":len(FULL),"events":events}
