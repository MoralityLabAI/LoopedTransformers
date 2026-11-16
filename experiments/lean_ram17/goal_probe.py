"""RAM-17 zero-action Lean goal probe with audited intentional shutdown.

The pinned RAM-16 probe server can report the accepted root state and then
continue compiling costly *later* original source declarations. Its close()
sends a stop request and, if Lean remains busy after 0.5s, kills the process.
RAM-17 treats only that explicitly requested shutdown as a benign exit for a
previously accepted root goal. Every TRAIN positive still needs two complete
independent source compiles plus a separate third Lean proof replay.
"""
from __future__ import annotations
import json
import signal
from pathlib import Path
from probe_session import Session, ProbeTimeout


def observe_original_goal(lean: str, original: str, theorem: str,
                          out: Path, guard, timeout: float = 15.0) -> dict:
    folder = Path(out)
    compiler = Session(lean, original, theorem, folder, guard, timeout=timeout)
    try:
        before = compiler.initial
        if not before.get("accepted") or len(before.get("goals", [])) != 1:
            raise AssertionError("original Lean theorem has no single accepted genuine root goal")
        if any(goal.count("⊢") != 1 for goal in before["goals"]):
            raise AssertionError("invalid original Lean-printed root goal")
    finally:
        compiler.close()
    code = compiler.proc.returncode
    receipt_path = folder / f"request-{compiler.number + 1}.json"
    deliberate = False
    if receipt_path.exists():
        try:
            request = json.loads(receipt_path.read_text(encoding="utf8"))
            deliberate = request.get("stop") is True and request.get("actions") == []
        except (ValueError, OSError):
            deliberate = False
    forced = code in (-signal.SIGKILL, -signal.SIGTERM)
    if code != 0 and not (forced and deliberate):
        raise ProbeTimeout(f"source goal probe exited {code} without a valid deliberate shutdown")
    receipt = dict(schema="RAM17-original-goal-shutdown-v1", theorem=theorem,
                   initial_goal_accepted=True, goal_count=len(before["goals"]),
                   requested_stop=deliberate, exitcode=code,
                   deliberately_forced_shutdown=bool(forced and deliberate),
                   complete_proof_source_certified=False)
    (folder / "goal_probe_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return before
