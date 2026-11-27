"""MeTTa/OLR bridge: verified source curation and advisory bounded STOP repair.

The MeTTa query is a typed/lineage/budget filter, following DataGoblin's
merged metta_research_bench/looped/good.metta contract. Native Hyperon must
run for any 'native_metta' evidence. A MeTTa offer is NEVER proof authority:
only Lean's kernel and an independently compiled source certify proof closure.

Native courier transfer DG-METTA-12 is reference-only until its Storyworld
kernel is pinned. This module does not import its unverified courier runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import re
from typing import Any, Sequence

from .data import canonical, digest
from .live_lean import public_state

HYPERON_VERSION = "0.2.10"
ME_TTA_SOURCE = "MoralityLabAI/DataGoblin/metta_research_bench/looped/good.metta"
# Program shape carried over from the native MeTTa curator, with validated
# local atoms. The source program is public and policy-only.
CURATION_QUERY = (
    "!(match &self "
    "(Rollout $id Closed Eligible WithinBudget UniqueLineage "
    "CurrentRevision DependenciesMet ConflictFree "
    "$covered $mentions $quality $negCost) "
    "(Offer $id $covered $quality $negCost))"
)
REPAIR_QUERY = (
    "!(match &self "
    "(Repair $id Typed WithinBudget UniqueLineage CurrentRevision "
    "DependenciesMet ConflictFree $steps) "
    "(Offer $id $steps))"
)
IDENT = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
TACTIC = re.compile(r"^apply h[0-9]{1,4}$")
CURATION_OFFER = re.compile(r"^\(Offer ([a-z][a-z0-9_]{0,31}) (\d+) (\d+) (-?\d+)\)$")
REPAIR_OFFER = re.compile(r"^\(Offer ([a-z][a-z0-9_]{0,31}) (\d+)\)$")


def _identifier(text: str) -> str:
    if type(text) is not str or IDENT.fullmatch(text) is None:
        raise ValueError("MeTTa atom must be a small, safe identifier")
    return text


def _native_metta(program: str, *, offer_pattern: re.Pattern) -> tuple[tuple, str]:
    """Run the pinned native interpreter; reject parse failure and shape drift."""
    try:
        from hyperon import Environment, MeTTa
    except ImportError as e:
        raise RuntimeError("Native Hyperon 0.2.10 required for MeTTa evidence") from e
    if version("hyperon") != HYPERON_VERSION:
        raise RuntimeError("MeTTa source evaluator must use pinned Hyperon 0.2.10")
    interpreter = MeTTa(
        env_builder=Environment.custom_env(create_config=False, is_test=True)
    )
    replies = interpreter.run(program)
    if not isinstance(replies, list) or not replies:
        raise ValueError("MeTTa returned no query result list")
    result = replies[-1]
    offers=[]
    for atom in result:
        match=offer_pattern.fullmatch(str(atom))
        if match is None:
            raise ValueError(f"unexpected MeTTa offer: {str(atom)[:120]!r}")
        offers.append(match.groups())
    if len({offer[0] for offer in offers}) != len(offers):
        raise ValueError("duplicate MeTTa offer")
    return tuple(offers), hashlib.sha256(program.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerifiedRollout:
    """Source curation metadata, created only after independent Lean checks."""

    identifier: str
    lineage: str
    split: str
    kernel_closed: bool
    standalone_certified: bool
    axiom_free: bool
    eligible: bool
    within_budget: bool
    current_revision: bool
    dependencies_met: bool
    conflict_free: bool
    covered: int
    mentions: int
    quality: int
    negative_cost: int

    def __post_init__(self):
        _identifier(self.identifier)
        _identifier(self.lineage)
        if self.split not in ("train", "development", "test"):
            raise ValueError("invalid source split")
        if any(type(v) is not int or abs(v) > 1000 for v in (
                self.covered, self.mentions, self.quality, self.negative_cost)):
            raise ValueError("out-of-range rollout quality fields")


def curate_verified(rollouts: Sequence[VerifiedRollout]) -> dict:
    """MeTTa-filter train-only *native-certified* experiences.

    Python independently checks the selected set to guard against interpreter
    drift. A filter result does not authorize access to held-out labels.
    """
    if not rollouts or len(rollouts)>128:
        raise ValueError("bounded nonempty train rollout pool required")
    if any(row.split!="train" for row in rollouts):
        raise ValueError("evaluation or development rollout cannot enter training curation")
    if len({row.identifier for row in rollouts})!=len(rollouts):
        raise ValueError("duplicate rollout identifier")
    counts={}
    for row in rollouts:
        counts[row.lineage]=counts.get(row.lineage,0)+1
    lines=[]
    expected=[]
    for row in rollouts:
        closed=row.kernel_closed and row.standalone_certified and row.axiom_free
        unique=counts[row.lineage]==1
        statuses=(
            "Closed" if closed else "Unverified",
            "Eligible" if row.eligible else "Ineligible",
            "WithinBudget" if row.within_budget else "OverBudget",
            "UniqueLineage" if unique else "RepeatedLineage",
            "CurrentRevision" if row.current_revision else "StaleRevision",
            "DependenciesMet" if row.dependencies_met else "MissingDependencies",
            "ConflictFree" if row.conflict_free else "Conflict",
        )
        lines.append(
            "(Rollout "
            + " ".join((row.identifier,*statuses,str(row.covered),
                        str(row.mentions),str(row.quality),str(row.negative_cost)))
            + ")"
        )
        if all((closed,row.eligible,row.within_budget,unique,
                row.current_revision,row.dependencies_met,row.conflict_free)):
            expected.append((row.identifier,str(row.covered),str(row.quality),
                             str(row.negative_cost)))
    program="\n".join((*lines,CURATION_QUERY))+"\n"
    observed,script_sha=_native_metta(program,offer_pattern=CURATION_OFFER)
    if set(observed)!=set(expected):
        raise ValueError("native MeTTa curation diverged from independent policy")
    score=lambda v:(-int(v[1]),-int(v[2]),-int(v[3]),v[0])
    ranked=tuple(item[0] for item in sorted(observed,key=score))
    return {"native_metta":True,"evidence_class":"native_hyperon_curated_lean_source",
            "program_sha256":script_sha,"policy_source":ME_TTA_SOURCE,
            "runtime":f"hyperon=={HYPERON_VERSION}",
            "offer_ids":ranked,"input_rows":len(rollouts),
            "native_offer_count":len(observed),
            "training_eligible":True}


@dataclass(frozen=True)
class RepairPlan:
    """Advisory candidate; policy may only inspect public state and metadata."""

    identifier: str
    actions: tuple[str, ...]
    lineage: str
    revision: int
    eligible: bool = True
    conflict_free: bool = True

    def __post_init__(self):
        _identifier(self.identifier)
        _identifier(self.lineage)
        if not 0 <= self.revision <= 1000:
            raise ValueError("invalid repair revision")
        if not 1 <= len(self.actions) <= 4:
            raise ValueError("repair scripts must be bounded")
        if any(TACTIC.fullmatch(a) is None for a in self.actions):
            raise ValueError("untyped or outside-menu repair action")


def _typed_plan(public: dict, actions: tuple[str,...]) -> bool:
    """Require one matching public hypothesis for each currently pending goal.

    This is a *syntactic precheck*. It cannot claim a tactic will execute.
    No private Lean result or generator reference action is exposed.
    """
    goals=public.get("goals",[])
    if not isinstance(goals,list) or len(goals)!=len(actions):
        return False
    for goal,action in zip(goals,actions,strict=True):
        if not isinstance(goal,dict) or not isinstance(goal.get("target"),dict):
            return False
        hypotheses=goal.get("hypotheses")
        if not isinstance(hypotheses,list):
            return False
        name=action.removeprefix("apply ")
        if not any(h.get("name")==name and h.get("type")==goal["target"]
                   for h in hypotheses if isinstance(h,dict)):
            return False
    return True


def offer_repairs(public: dict, plans: Sequence[RepairPlan], budget: int) -> dict:
    """Match source-visible obligations in native MeTTa, return advisory IDs."""
    if not isinstance(public,dict) or not 0 <= budget <= 8:
        raise ValueError("invalid public goal packet or budget")
    if not plans or len(plans)>32 or len({p.identifier for p in plans})!=len(plans):
        raise ValueError("invalid or duplicate repair offers")
    lineage_count={}
    latest={}
    for p in plans:
        lineage_count[p.lineage]=lineage_count.get(p.lineage,0)+1
        latest[p.lineage]=max(p.revision,latest.get(p.lineage,-1))
    lines=[]
    expected=[]
    for p in plans:
        typed=_typed_plan(public,p.actions)
        affordable=len(p.actions)<=budget
        # Exact one-plan lineage scope is a source policy. It prevents a
        # collection of copied/revised proof candidates from earning
        # independent evidence credit.
        unique=lineage_count[p.lineage]==1
        current=p.revision==latest[p.lineage]
        predicates=(
            "Typed" if typed else "Untyped",
            "WithinBudget" if affordable else "OverBudget",
            "UniqueLineage" if unique else "RepeatedLineage",
            "CurrentRevision" if current else "StaleRevision",
            "DependenciesMet" if typed else "MissingDependencies",
            "ConflictFree" if p.conflict_free and p.eligible else "Conflict",
        )
        lines.append("(Repair "+" ".join((
            p.identifier,*predicates,str(len(p.actions))))+")")
        if typed and affordable and unique and current and p.conflict_free and p.eligible:
            expected.append((p.identifier,str(len(p.actions))))
    program="\n".join((*lines,REPAIR_QUERY))+"\n"
    offered,sha=_native_metta(program,offer_pattern=REPAIR_OFFER)
    if set(offered)!=set(expected):
        raise ValueError("native MeTTa repair query diverged from typed public precheck")
    ordered=tuple(x[0] for x in sorted(offered,key=lambda x:(int(x[1]),x[0])))
    return {"runtime":f"hyperon=={HYPERON_VERSION}",
            "native_metta":True,"program_sha256":sha,
            "public_state_sha256":digest(public),
            "offered_plan_ids":ordered,"budget":budget}


def run_atomic_stop_repair(repl: Any, state: dict, *, plans: Sequence[RepairPlan],
                           budget: int, stopped: bool) -> dict:
    """Check a proposed STOP against MeTTa and Lean, committing atomically.

    A rejected/partial macro returns the unchanged source state. An accepted
    macro requires a fully closed Lean proof and an empty kernel axiom set.
    Each native tactic attempt costs a verifier call, whether accepted or not.
    """
    if stopped is not True:
        raise ValueError("STOP-triggered macro requires an explicit STOP proposal")
    if not isinstance(state,dict) or type(state.get("sid")) is not int:
        raise ValueError("expected real Lean state with immutable server ID")
    source=public_state(repl,state)
    if not source["goals"]:
        raise ValueError("already closed; a repair would be redundant")
    offered=offer_repairs(source,plans,budget)
    choices={p.identifier:p for p in plans}
    start_sha=digest(source)
    # Prefer the shortest, then stable name. A policy may propose multiple
    # plans but only the first is attempted in this bounded implementation.
    if not offered["offered_plan_ids"]:
        return {"status":"no_typed_offer","committed":False,"state":state,
                "source_state_sha256":start_sha,"verifier_calls":0,
                "events":[],"me_tta":offered}
    chosen=choices[offered["offered_plan_ids"][0]]
    new=state
    events=[]
    for action in chosen.actions:
        if len(events)>=budget:
            raise ValueError("preflight allowed an over-budget script")
        before=digest(public_state(repl,new))
        reply=repl.run(new["sid"],action)
        if type(reply.get("ok")) is not bool:
            raise ValueError("malformed native Lean tactic reply")
        if reply["ok"]:
            after=reply.get("state")
            if not isinstance(after,dict):
                raise ValueError("Lean success without a proof-state snapshot")
            new=after
        event={"tactic":action,"ok":reply["ok"],"before":before,
               "after":digest(public_state(repl,new))}
        events.append(event)
        if not reply["ok"]:
            return {"status":"native_tactic_rejected","committed":False,
                    "state":state,"source_state_sha256":start_sha,
                    "verifier_calls":len(events),"events":events,"me_tta":offered}
    if public_state(repl,new)["goals"]:
        return {"status":"pending_goals_after_macro","committed":False,
                "state":state,"source_state_sha256":start_sha,
                "verifier_calls":len(events),"events":events,"me_tta":offered}
    receipt=repl.kernel(new["sid"])
    if receipt.get("ok") is not True or receipt.get("axioms")!=[]:
        return {"status":"kernel_rejected","committed":False,
                "state":state,"source_state_sha256":start_sha,
                "verifier_calls":len(events),"events":events,"me_tta":offered}
    return {"status":"kernel_closed","committed":True,"state":new,
            "chosen_plan":chosen.identifier,"source_state_sha256":start_sha,
            "end_state_sha256":digest(public_state(repl,new)),
            "verifier_calls":len(events),"events":events,"me_tta":offered,
            "kernel":{"ok":True,"axioms":[]}}
