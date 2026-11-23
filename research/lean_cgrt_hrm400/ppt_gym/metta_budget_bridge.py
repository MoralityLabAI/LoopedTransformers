"""OLR-METTA-03: independent-source evidence and paid native Lean diagnosis.

The policy sees only synthetic witness claims admitted by Hyperon, *not*
arithmetic guard numerals or oracle truth. Lean sees the complete theorem and
is the only authority for actual tactic success. This intentional restricted
observation is documented in OLR_METTA_03_REGISTRATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from importlib.metadata import version

from lcgrt.lean_repl import LeanRepl
from .data import canonical, digest
from .live_lean import candidate_menu, public_state
from .metta_obligations import _native_metta
from .olr_delayed import DelayedWorld, ROOT_MENU

SCHEMA = "ppt.olr-metta-budget.v1"
ACTIONS = ("skip_r0", "skip_r1", "probe_r0")
COST = .07
EVIDENCE_QUERY = (
    "!(match &self (Claim $id Current Unique Eligible $vote $strength) "
    "(Offer $id $vote $strength))"
)
OFFER = re.compile(r"^\(Offer (c[0-5]) (v[01]) ([1-3])\)$")


@dataclass(frozen=True)
class EvidenceClaim:
    identifier: str
    source: str
    revision: int
    vote: int
    strength: int
    eligible: bool = True

    def __post_init__(self):
        if (not re.fullmatch(r"c[0-5]", self.identifier)
                or not re.fullmatch(r"s[0-3]", self.source)
                or not 1 <= self.revision <= 3 or self.vote not in (0, 1)
                or not 1 <= self.strength <= 3 or type(self.eligible) is not bool):
            raise ValueError("invalid typed public witness claim")


@dataclass(frozen=True)
class WitnessWorld:
    split: str
    pair_id: str
    world: int
    a: int
    b: int
    delta: int
    mode: str
    claims: tuple[EvidenceClaim, ...]

    def __post_init__(self):
        if self.split not in ("train", "dev", "ood") or self.world not in (0, 1):
            raise ValueError("bad split or paired world")
        if len(self.claims) != 6 or sorted(x.identifier for x in self.claims) != [
                f"c{i}" for i in range(6)]:
            raise ValueError("expected six distinct ordered public claims")

    def lean_world(self) -> DelayedWorld:
        target=self.a+self.b
        options=(target,target+self.delta) if self.world==0 else (target+self.delta,target)
        return DelayedWorld(self.pair_id,self.world,self.a,self.b,options)

    def lineage(self) -> tuple[int, int, int]:
        return self.a,self.b,self.delta


def _reported_vote(rng: random.Random, winner: int, reliability: float) -> int:
    return winner if rng.random() < reliability else 1-winner


def generate(split: str, *, seed: int, count: int,
             low: int, high: int,
             exclude: frozenset[tuple[int,int,int]] = frozenset()) -> tuple[WitnessWorld,...]:
    """Produce paired roots; source lineages are disjoint from earlier OLR bands."""
    if split not in ("train","dev","ood") or not 1 <= count <= 96:
        raise ValueError("bad bounded generator request")
    if not 17 <= low <= high <= 44:
        raise ValueError("outside registered operand bands")
    rng=random.Random(f"olr-metta-03/{split}/{seed}/{low}/{high}")
    used=set(exclude)
    output=[]
    for index in range(count):
        for retry in range(300):
            a,b=rng.randint(low,high),rng.randint(low,high)
            delta=rng.choice((1,2,3))
            if (a,b,delta) not in used:
                break
        else:
            raise ValueError("lineage generator exhausted")
        used.add((a,b,delta))
        pair=digest([SCHEMA,split,seed,index,a,b,delta])
        mode=("strong","weak","conflict","stale")[index%4]
        for world in (0,1):
            local=random.Random(f"{pair}/{world}/claims")
            if mode=="strong":
                reliability=(.97,.94,.91)
                weights=(3,3,2)
            elif mode=="weak":
                reliability=(.60,.54,.50)
                weights=(1,1,1)
            elif mode=="conflict":
                reliability=(.85,.18,.50)
                weights=(3,3,1)
            else:
                reliability=(.93,.83,.74)
                weights=(3,2,1)
            votes=tuple(_reported_vote(local,world,p) for p in reliability)
            claims=(
                EvidenceClaim("c0","s0",1,1-votes[0],3),  # stale
                EvidenceClaim("c1","s0",2,votes[0],weights[0]),
                EvidenceClaim("c2","s1",1,votes[1],weights[1]),
                EvidenceClaim("c3","s2",1,1-world,3),  # duplicated aliases
                EvidenceClaim("c4","s2",1,1-world,3),
                EvidenceClaim("c5","s3",1,votes[2],weights[2]),
            )
            output.append(WitnessWorld(split,pair,world,a,b,delta,mode,claims))
    return tuple(output)


def curate_claims(claims: tuple[EvidenceClaim,...]) -> dict:
    """MeTTa checks source uniqueness and revision, Python checks its results.

    The truthfulness of the synthetic testimony is NOT validated by MeTTa.
    """
    if len(claims)!=6:
        raise ValueError("expected fixed six-claim public packet")
    if len({c.identifier for c in claims})!=6:
        raise ValueError("duplicate claim identifier")
    latest={s:max(c.revision for c in claims if c.source==s)
            for s in {c.source for c in claims}}
    current_count={
        s:sum(c.source==s and c.revision==latest[s] for c in claims)
        for s in latest
    }
    statuses=[]
    native_facts=[]
    expected=[]
    for claim in claims:
        is_current=claim.revision==latest[claim.source]
        unique=is_current and current_count[claim.source]==1
        status=(claim.identifier,is_current,unique,claim.eligible)
        statuses.append(status)
        vals=(claim.identifier,
              "Current" if is_current else "Stale",
              "Unique" if unique else "Repeated",
              "Eligible" if claim.eligible else "Ineligible",
              "v"+str(claim.vote),str(claim.strength))
        native_facts.append("(Claim "+" ".join(vals)+")")
        if is_current and unique and claim.eligible:
            expected.append((claim.identifier,"v"+str(claim.vote),str(claim.strength)))
    program="\n".join((*native_facts,EVIDENCE_QUERY))+"\n"
    observed,sha=_native_metta(program,offer_pattern=OFFER)
    if set(observed)!=set(expected):
        raise ValueError("native MeTTa evidence selection diverged from independent reference")
    admitted={x[0] for x in observed}
    features=[]
    for claim,(_,is_current,unique,eligible) in zip(claims,statuses,strict=True):
        keep=int(claim.identifier in admitted)
        features.append([
            float(2*claim.vote-1) * keep,
            (claim.strength / 3.) * keep,
            float(keep),float(is_current),float(unique),float(eligible),
        ])
    if len(features)!=6 or len(admitted)<1:
        raise ValueError("native public evidence packet empty")
    return {"features":features,"admitted_ids":sorted(admitted),
            "program_sha256":sha,"runtime":"hyperon==0.2.10",
            "untrusted_claims":[c.__dict__ for c in claims],
            "selected_lineages":sorted({c.source for c in claims if c.identifier in admitted})}


def masked_packet(world: WitnessWorld) -> dict:
    """Only this packet is used by the model; arithmetic truth is excluded."""
    cured=curate_claims(world.claims)
    return {"root_id":digest([world.pair_id,world.world,world.mode,SCHEMA]),
            "pair_id":world.pair_id,"world":world.world,"split":world.split,
            "mode":world.mode,"model_features":cured["features"],
            "admitted_ids":cured["admitted_ids"],"evidence_program_sha256":cured["program_sha256"],
            "claims":cured["untrusted_claims"],
            # The full statement is execution-side custody; never handed to the
            # neural forward method or to the majority heuristic.
            "theorem_statement":world.lean_world().statement()}


def uncurated_features(packet: dict) -> list[list[float]]:
    """A deliberately sloppy control retains all six claims as source votes."""
    return [[float(2*int(c["vote"])-1),float(c["strength"])/3.,1.,
             row[3],row[4],row[5]]
            for row,c in zip(packet["model_features"],packet["claims"],strict=True)]


def native_episode(lean: LeanRepl, world: DelayedWorld, action: str) -> dict:
    """Execute optional two-call diagnostic, then a four-call proof attempt.

    The speculative probe state is always rolled back to the saved root,
    including when the probe succeeds. A failed skip is terminal. Each native
    action and its public successor hash is recorded for independent replay.
    """
    if action not in ACTIONS:
        raise ValueError("invalid OLR-METTA-03 policy action")
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    visible=public_state(lean,state)
    if candidate_menu(lean,state)!=ROOT_MENU:
        raise ValueError("public Lean root tactic menu changed")
    initial=digest(visible)
    events=[]
    spent=0
    result="not_closed"
    proof_action=None

    def attempt(tactic: str) -> bool:
        nonlocal state,spent
        if spent>=6 or tactic not in ("apply r0","apply r1","rfl","apply h0","apply h1"):
            raise ValueError("native tool budget or tactic allowlist changed")
        before=digest(public_state(lean,state))
        response=lean.run(state["sid"],tactic)
        if type(response.get("ok")) is not bool:
            raise ValueError("malformed Lean response")
        spent+=1
        if response["ok"]:
            state=response["state"]
        after=digest(public_state(lean,state))
        events.append({"op":"tactic","tactic":tactic,"ok":response["ok"],
                       "before":before,"after":after,"call":spent})
        return response["ok"]

    if action=="probe_r0":
        if not attempt("apply r0"):
            raise ValueError("registered speculative root rejected")
        observed=attempt("rfl")
        # Even a successful speculative branch is uncommitted. The model only
        # gets this verifier bit after the two calls have been charged.
        events.append({"op":"restore","before":digest(public_state(lean,state)),
                       "after":initial,"probe_result":bool(observed)})
        state=root
        chosen=0 if observed else 1
    else:
        chosen=0 if action=="skip_r0" else 1
    proof_action=f"apply r{chosen}"
    if not attempt(proof_action):
        raise ValueError("registered final root refused")
    if not attempt("rfl"):
        result="guard_rejected"
    else:
        if not attempt("apply h0") or not attempt("apply h1"):
            raise ValueError("registered pending obligation did not close")
        if public_state(lean,state)["goals"]:
            raise ValueError("nonterminal after all pending obligations")
        kernel=lean.kernel(state["sid"])
        if kernel.get("ok") is not True or kernel.get("axioms")!=[]:
            raise ValueError("native terminal kernel rejected")
        events.append({"op":"kernel","before":digest(public_state(lean,state)),
                       "after":digest(public_state(lean,state)),"ok":True,"axioms":[]})
        result="kernel_closed"
    return {"schema":"ppt.olr-metta-budget-native-episode.v1",
            "root":digest(world.statement()),"action":action,
            "original_public_sha256":initial,"events":events,
            "terminal":result,"kernel_closed":result=="kernel_closed",
            "tactic_calls":spent,"verified_branch":proof_action if result=="kernel_closed" else None,
            "utility":round(float(result=="kernel_closed")-COST*spent,7)}


def replay_episode(lean: LeanRepl, world: DelayedWorld, receipt: dict) -> bool:
    """Re-execute the recorded native actions and prove the receipts agree."""
    if (receipt.get("schema")!="ppt.olr-metta-budget-native-episode.v1"
            or receipt.get("root")!=digest(world.statement())):
        raise ValueError("replay source identity changed")
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    initial=digest(public_state(lean,state))
    if initial!=receipt["original_public_sha256"]:
        raise ValueError("replay initial goal differs")
    calls=0
    terminal=False
    for e in receipt["events"]:
        before=digest(public_state(lean,state))
        if e["before"]!=before:
            raise ValueError("replay old state differs")
        if e["op"]=="tactic":
            output=lean.run(state["sid"],e["tactic"])
            calls+=1
            if output["ok"]!=e["ok"] or e["call"]!=calls:
                raise ValueError("native tactic verdict or accounting differs")
            if output["ok"]:
                state=output["state"]
        elif e["op"]=="restore":
            state=root
            if type(e["probe_result"]) is not bool:
                raise ValueError("malformed saved verifier bit")
        elif e["op"]=="kernel":
            k=lean.kernel(state["sid"])
            terminal=k.get("ok") is True and k.get("axioms")==[]
            if not terminal or e["axioms"]!=[]:
                raise ValueError("kernel closure did not replay")
        else:
            raise ValueError("unregistered replay action")
        if digest(public_state(lean,state))!=e["after"]:
            raise ValueError("replayed resulting Lean goal differs")
    if calls!=receipt["tactic_calls"] or terminal!=receipt["kernel_closed"]:
        raise ValueError("replay call count or proof status differs")
    if round(float(terminal)-COST*calls,7)!=receipt["utility"]:
        raise ValueError("replay cost reward differs")
    return True
