"""OLR-01D: real Lean multi-goal guard probe, rollback and exact replay.

The model chooses only the initial root rule. Lean supplies all subsequent
tactic/verifier outcomes. The two proof schedules are deterministic reference
controllers, not learned recurrence or online adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

from lcgrt import config
from lcgrt.lean_repl import LeanRepl
from .data import digest, sha256_file
from .live_lean import candidate_menu, public_state
from .olr_guards import numeric_features

ARMS = ("probe_early", "probe_late")
ROOT_MENU = ("apply r0", "apply r1")
NATIVE_ACTION = re.compile(r"^(apply r[01]|apply h[01]|rfl)$")
STMT = re.compile(
    r"^∀ \(P Q R : Prop\) \(h0 : P\) \(h1 : Q\) "
    r"\(r0 : \((\d+) \+ (\d+) = (\d+)\) → P → Q → R\) "
    r"\(r1 : \((\d+) \+ (\d+) = (\d+)\) → P → Q → R\), R$"
)


@dataclass(frozen=True)
class DelayedWorld:
    pair: str
    world: int
    a: int
    b: int
    targets: tuple[int, int]

    def statement(self) -> str:
        x, y = self.targets
        return ("∀ (P Q R : Prop) (h0 : P) (h1 : Q) "
                f"(r0 : ({self.a} + {self.b} = {x}) → P → Q → R) "
                f"(r1 : ({self.a} + {self.b} = {y}) → P → Q → R), R")

    @classmethod
    def from_row(cls, row: dict) -> "DelayedWorld":
        m = STMT.fullmatch(row["statement"])
        if not m:
            raise ValueError("OLR-01D statement outside registered safe grammar")
        a, b, x, c, d, y = (int(v) for v in m.groups())
        if a != c or b != d:
            raise ValueError("two candidate guards changed shared operands")
        if digest(row["statement"]) != row["root"]:
            raise ValueError("registered native theorem changed")
        return cls(row["pair"], row["world"], a, b, (x, y))


def raw_root_features(visible: dict, menu: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """Read raw arithmetic operands from Lean. No equality calculation or label."""
    goals = visible.get("goals")
    if not isinstance(goals, list) or len(goals) != 1 or goals[0].get("target") != {"a":"R"}:
        raise ValueError("not a single public root goal R")
    if menu != ROOT_MENU:
        raise ValueError("public root menu changed")
    hyps = goals[0].get("hypotheses")
    if not isinstance(hyps,list):
        raise ValueError("missing Lean root hypotheses")
    types = {h.get("name"):h.get("type") for h in hyps if isinstance(h,dict)}
    if types.get("h0") != {"a":"P"} or types.get("h1") != {"a":"Q"}:
        raise ValueError("pending proof hypotheses changed")
    vectors=[]
    for rule in ("r0","r1"):
        node=types.get(rule)
        for premise in range(3):
            if not isinstance(node,dict) or set(node) != {"imp"}:
                raise ValueError("root candidate lost an implication")
            two=node["imp"]
            if not isinstance(two,list) or len(two)!=2:
                raise ValueError("root implication malformed")
            first,node=two
            if premise==0:
                if not isinstance(first,dict) or not isinstance(first.get("raw"),str):
                    raise ValueError("computational guard not reported as raw")
                vectors.append(numeric_features(first["raw"]))
            elif first != {"a":"P"} if premise==1 else first != {"a":"Q"}:
                raise ValueError("pending premise identity changed")
        if node != {"a":"R"}:
            raise ValueError("candidate consequent changed")
    return tuple(vectors)


def observe_root(lean: LeanRepl, world: DelayedWorld) -> dict:
    lean.reset()
    state=lean.init(world.statement(),pp=True)
    visible=public_state(lean,state)
    menu=candidate_menu(lean,state)
    features=raw_root_features(visible,menu)
    return {
        "source_sha256":digest(visible),
        "root":digest(world.statement()),
        "menu":list(menu),
        "features":[list(x) for x in features],
    }


def run_path(lean: LeanRepl, world: DelayedWorld, initial: int, schedule: str,
             budget: int) -> dict:
    """Execute one policy branch with a saved native root state for rollback."""
    if schedule not in ARMS or initial not in (0,1) or budget not in (4,6):
        raise ValueError("registered action/schedule/check budget mismatch")
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    original=public_state(lean,root)
    root_menu=candidate_menu(lean,root)
    features=raw_root_features(original,root_menu)
    events=[]
    script=[]
    calls=focus_count=rollback_count=0
    closed=False

    def tactic(action: str) -> bool:
        nonlocal calls,state
        if not NATIVE_ACTION.fullmatch(action) or calls>=budget:
            raise ValueError("illegal native tactic or over budget")
        before=digest(public_state(lean,state))
        result=lean.run(state["sid"],action)
        if type(result.get("ok")) is not bool:
            raise ValueError("malformed Lean tactic outcome")
        calls+=1
        if result["ok"]:
            state=result["state"]
            after=digest(public_state(lean,state))
            script.append(action)
        else:
            after=before
        events.append({"kind":"tactic","action":action,"ok":result["ok"],
                       "before":before,"after":after,"calls":calls})
        return result["ok"]

    def focus(index: int) -> None:
        nonlocal focus_count,state
        if index != 1:
            raise ValueError("only registered focus index 1 is allowed")
        before=digest(public_state(lean,state))
        output=lean.focus(state["sid"],index)
        if output.get("ok") is not True:
            raise ValueError("native pending-goal focus rejected")
        state=output["state"]
        after=digest(public_state(lean,state))
        focus_count+=1
        script.append("rotate_left 1")  # exact Lean tactic for the host permutation
        events.append({"kind":"focus","index":index,"before":before,"after":after})

    def rollback() -> None:
        nonlocal state,rollback_count,script
        if rollback_count>=1:
            raise ValueError("rollback cap exceeded")
        before=digest(public_state(lean,state))
        state=root
        script=[]
        rollback_count+=1
        events.append({"kind":"rollback","before":before,"after":digest(original)})

    def active() -> list:
        return public_state(lean,state)["goals"]

    def attempt(choice: int) -> str:
        nonlocal closed
        if not tactic(f"apply r{choice}"):
            raise ValueError("registered native opening rejected")
        if len(active())!=3:
            raise ValueError("root must expose guard + two pending goals")
        if schedule=="probe_late":
            focus(1)
            if not tactic("apply h0") or len(active())!=2:
                raise ValueError("failed to discharge the first pending goal")
            focus(1)
            if not tactic("apply h1") or len(active())!=1:
                raise ValueError("failed to discharge the second pending goal")
        if not tactic("rfl"):
            return "guard_rejected"
        if schedule=="probe_early":
            if len(active())!=2:
                raise ValueError("guard success must retain two pending goals")
            if not tactic("apply h0") or len(active())!=1:
                raise ValueError("first post-guard goal did not close")
            if not tactic("apply h1") or active():
                raise ValueError("second post-guard goal did not close")
        if active():
            raise ValueError("native proof retained goals after terminal action")
        cert=lean.kernel(state["sid"])
        if cert.get("ok") is not True or cert.get("axioms")!=[]:
            raise ValueError("native Lean kernel rejected a closed multi-goal proof")
        events.append({"kind":"kernel","ok":True,"axioms":[],"before":digest(public_state(lean,state))})
        closed=True
        return "kernel_closed"

    status=attempt(initial)
    if status=="guard_rejected":
        rollback()
        if budget-calls >= 4:
            status=attempt(1-initial)
            if status!="kernel_closed":
                raise ValueError("the complementary registered branch did not close")
        else:
            status="insufficient_budget_after_probe"
    if not closed and status=="kernel_closed":
        raise ValueError("bad terminal state")
    if closed and calls>budget:
        raise ValueError("native cost cap breached")
    return {
        "root":digest(world.statement()), "pair":world.pair, "world":world.world,
        "initial_action":f"apply r{initial}", "schedule":schedule, "budget":budget,
        "root_public_sha256":digest(original), "root_menu":list(root_menu),
        "features":[list(v) for v in features],
        "kernel_closed":closed, "status":status, "tactic_calls":calls,
        "focus_actions":focus_count, "rollbacks":rollback_count,
        "script":list(script) if closed else [], "events":events,
        "schema":"ppt.olr-delayed-episode.v1",
    }


def replay_path(lean: LeanRepl, world: DelayedWorld, episode: dict) -> dict:
    """Independent event-driven native replay, including host focus and restore."""
    if (episode.get("schema")!="ppt.olr-delayed-episode.v1"
            or episode["root"]!=digest(world.statement())):
        raise ValueError("replay root or schema differs")
    lean.reset()
    root=lean.init(world.statement(),pp=True)
    state=root
    original=public_state(lean,root)
    if digest(original)!=episode["root_public_sha256"]:
        raise ValueError("replayed root observation differs")
    if tuple(episode["root_menu"])!=candidate_menu(lean,root):
        raise ValueError("replayed root tactic grammar changed")
    if episode["features"] != [list(v) for v in raw_root_features(original,ROOT_MENU)]:
        raise ValueError("replayed predecision features differ")
    calls=focus_count=rollbacks=0
    script=[]
    certified=False
    for event in episode["events"]:
        before=digest(public_state(lean,state))
        if before!=event["before"]:
            raise ValueError("changed native proof state before event")
        if event["kind"]=="tactic":
            if not NATIVE_ACTION.fullmatch(event["action"]):
                raise ValueError("non-whitelisted replay tactic")
            result=lean.run(state["sid"],event["action"])
            calls+=1
            if result["ok"]!=event["ok"] or calls!=event["calls"]:
                raise ValueError("native verdict or step number changed")
            if result["ok"]:
                state=result["state"]
                script.append(event["action"])
        elif event["kind"]=="focus":
            if event["index"]!=1:
                raise ValueError("unregistered focus event")
            result=lean.focus(state["sid"],1)
            if result.get("ok") is not True:
                raise ValueError("native focus rejected on replay")
            state=result["state"]
            script.append("rotate_left 1")
            focus_count+=1
        elif event["kind"]=="rollback":
            state=root
            script=[]
            rollbacks+=1
            if rollbacks>1:
                raise ValueError("more than one rollback in replay")
        elif event["kind"]=="kernel":
            if public_state(lean,state)["goals"]:
                raise ValueError("open goal in recorded terminal proof")
            result=lean.kernel(state["sid"])
            if not (result.get("ok") is True and result.get("axioms")==[]):
                raise ValueError("native kernel differs on replay")
            certified=True
        else:
            raise ValueError("unknown native ledger event")
        if digest(public_state(lean,state))!=event.get("after",event["before"]):
            raise ValueError("replayed successor proof state differs")
    if (calls!=episode["tactic_calls"] or focus_count!=episode["focus_actions"]
            or rollbacks!=episode["rollbacks"] or calls>episode["budget"]
            or certified!=episode["kernel_closed"]):
        raise ValueError("replayed resource / certificate mismatch")
    if certified and script!=episode["script"]:
        raise ValueError("replayed standalone proof script changed")
    if not certified and episode["script"]:
        raise ValueError("uncertified episode includes proof script")
    return {"replay_passed":True,"tactic_calls":calls,"focus_actions":focus_count,
            "rollbacks":rollbacks, "kernel_closed":certified}


def compile_scripts(episodes: list[tuple[DelayedWorld,dict]], out: Path) -> dict:
    """Compile every distinct successful live tactic/focus sequence with Lean.

    Only a closed and fully matched theorem string and an exactly allowlisted
    sequence may enter the standalone source.
    """
    seen={}
    for world,ep in episodes:
        if not ep["kernel_closed"]:
            continue
        key=(world.statement(),tuple(ep["script"]))
        seen[key]=ep
    source=["namespace OLRDelayed"]
    for i,((statement,script),ep) in enumerate(sorted(seen.items())):
        m=STMT.fullmatch(statement)
        if not m or any(t not in ("apply r0","apply r1","apply h0","apply h1",
                                  "rotate_left 1","rfl") for t in script):
            raise ValueError("unsafe delayed-proof source/script")
        binders=statement[2:-3]
        source.append(f"theorem d_{i} {binders} : R := by")
        source.extend("  "+t for t in script)
    source.append("end OLRDelayed")
    source.extend(f"#print axioms OLRDelayed.d_{i}" for i in range(len(seen)))
    text="\n".join(source)+"\n"
    out.write_text(text,encoding="utf-8")
    with tempfile.TemporaryDirectory() as root:
        path=Path(root)/"OLRDelayed.lean"
        path.write_text(text,encoding="utf-8")
        result=subprocess.run([str(config.LEAN_EXE),str(path)],
            capture_output=True,text=True,encoding="utf-8",
            env=config.lean_env(),timeout=180,check=False)
    msgs=result.stdout+result.stderr
    clean=sum(f"'OLRDelayed.d_{i}' does not depend on any axioms" in msgs
              for i in range(len(seen)))
    if result.returncode!=0 or clean!=len(seen):
        raise ValueError(f"standalone delayed proofs failed: code={result.returncode}, "
                         f"axiom_clean={clean}/{len(seen)}, {msgs[-1500:]}")
    return {"standalone_compiled":len(seen),"axiom_clean":clean,
            "source_sha256":sha256_file(out)}
