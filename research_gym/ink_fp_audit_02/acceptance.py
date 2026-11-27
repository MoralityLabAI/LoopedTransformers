"""Frozen, replayable generation acceptance.

A decision is a pure function of (criteria, predecessor case outcomes, candidate case
outcomes). Both outcome sets are written into the ledger, so any reader can recompute
every decision and the accepted-checkpoint chain without trusting the recorded verdict.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .ledger import Ledger, sha256

CRITERIA_VERSION = "ink-fp-audit-02.acceptance.v1"


@dataclass(frozen=True)
class Criteria:
    version: str = CRITERIA_VERSION
    primary: str = "paired verified-correct count on the generation-acceptance partition"
    min_primary_gain: int = 1            # candidate must solve >= 1 more paired case
    max_family_loss_frac: float = 0.10   # per family, losses <= max(1, ceil(frac * n_family))
    max_repetition_increase: float = 0.02
    max_completion_drop: float = 0.02

    def digest(self) -> str:
        return sha256(asdict(self))


def summarize(cases: list[dict]) -> dict:
    """cases: [{"task_id", "family", "correct", "completed", "repetition"}] (bool fields)."""
    fam: dict[str, dict] = {}
    for c in cases:
        f = fam.setdefault(c["family"], {"n": 0, "correct": 0, "completed": 0, "repetition": 0})
        f["n"] += 1
        f["correct"] += bool(c["correct"])
        f["completed"] += bool(c["completed"])
        f["repetition"] += bool(c["repetition"])
    n = len(cases)
    return {"n": n, "correct": sum(f["correct"] for f in fam.values()),
            "completion_rate": sum(f["completed"] for f in fam.values()) / n if n else 0.0,
            "repetition_rate": sum(f["repetition"] for f in fam.values()) / n if n else 0.0,
            "families": dict(sorted(fam.items()))}


def decide(criteria: Criteria, predecessor_cases: list[dict], candidate_cases: list[dict]) -> dict:
    pred_ids = sorted(c["task_id"] for c in predecessor_cases)
    cand_ids = sorted(c["task_id"] for c in candidate_cases)
    if pred_ids != cand_ids or len(set(pred_ids)) != len(pred_ids):
        raise ValueError("acceptance requires identical, unique paired task sets")
    p, c = summarize(predecessor_cases), summarize(candidate_cases)
    gain = c["correct"] - p["correct"]
    family_losses, family_limits = {}, {}
    for name, pf in p["families"].items():
        cf = c["families"][name]
        family_losses[name] = pf["correct"] - cf["correct"]
        family_limits[name] = max(1, math.ceil(criteria.max_family_loss_frac * pf["n"]))
    checks = {
        "primary_gain_ok": gain >= criteria.min_primary_gain,
        "family_regression_ok": all(family_losses[f] <= family_limits[f] for f in family_losses),
        "repetition_ok": c["repetition_rate"] <= p["repetition_rate"] + criteria.max_repetition_increase + 1e-12,
        "completion_ok": c["completion_rate"] >= p["completion_rate"] - criteria.max_completion_drop - 1e-12,
    }
    return {"accepted": all(checks.values()), "checks": checks, "primary_gain": gain,
            "family_losses": family_losses, "family_limits": family_limits,
            "predecessor_summary": p, "candidate_summary": c, "criteria_sha256": criteria.digest()}


def record_candidate(ledger: Ledger, criteria: Criteria, *, generation: int, arm: str,
                     predecessor_checkpoint: str, candidate_checkpoint: str,
                     predecessor_cases: list[dict], candidate_cases: list[dict],
                     hyperparameters: dict, scales_examined: list) -> dict:
    """Verify the predecessor is the current accepted head for this arm, decide, persist."""
    head = accepted_head(ledger, arm)
    if head is not None and head != predecessor_checkpoint:
        raise ValueError(f"predecessor {predecessor_checkpoint} is not accepted head {head}")
    decision = decide(criteria, predecessor_cases, candidate_cases)
    body = {"generation": generation, "arm": arm, "criteria": asdict(criteria),
            "predecessor_checkpoint": predecessor_checkpoint,
            "candidate_checkpoint": candidate_checkpoint,
            "predecessor_cases": predecessor_cases, "candidate_cases": candidate_cases,
            "hyperparameters": hyperparameters, "scales_examined": scales_examined,
            "decision": decision,
            "next_head": candidate_checkpoint if decision["accepted"] else predecessor_checkpoint}
    ledger.append("acceptance", body)
    return body


def register_base(ledger: Ledger, arm: str, checkpoint: str) -> None:
    ledger.append("base", {"arm": arm, "checkpoint": checkpoint})


def accepted_head(ledger: Ledger, arm: str) -> str | None:
    head = None
    for rec in ledger.iter_verified():
        b = rec["body"]
        if b.get("arm") != arm:
            continue
        if rec["kind"] == "base":
            head = b["checkpoint"]
        elif rec["kind"] == "acceptance":
            head = b["next_head"]
    return head


def replay(ledger: Ledger) -> dict:
    """Recompute every decision and the head chain; report any mismatch with the record."""
    heads: dict[str, str] = {}
    rows, mismatches = [], []
    for rec in ledger.iter_verified():
        b = rec["body"]
        if rec["kind"] == "base":
            heads[b["arm"]] = b["checkpoint"]
            continue
        if rec["kind"] != "acceptance":
            continue
        crit = Criteria(**b["criteria"])
        d = decide(crit, b["predecessor_cases"], b["candidate_cases"])
        expected_head = heads.get(b["arm"])
        problems = []
        if expected_head is not None and b["predecessor_checkpoint"] != expected_head:
            problems.append("predecessor_not_accepted_head")
        if d["accepted"] != b["decision"]["accepted"]:
            problems.append("recorded_decision_differs")
        nxt = b["candidate_checkpoint"] if d["accepted"] else b["predecessor_checkpoint"]
        if nxt != b["next_head"]:
            problems.append("recorded_next_head_differs")
        heads[b["arm"]] = nxt
        rows.append({"seq": rec["seq"], "arm": b["arm"], "generation": b["generation"],
                     "accepted": d["accepted"], "problems": problems})
        if problems:
            mismatches.append(rows[-1])
    return {"rows": rows, "mismatches": mismatches, "heads": heads}


# ---- upstream record audit (published RSI project) ------------------------------------
UPSTREAM_MODES = ("R3", "R4", "RK2")


def upstream_counts(gate: dict, n: int = 64) -> dict:
    out = {}
    for m in UPSTREAM_MODES:
        v = gate[m]
        out[m] = int(v["correct"]) if isinstance(v, dict) else round(v * n / 100)
    if sum(out.values()) != gate["total_correct"]:
        raise ValueError("inconsistent upstream gate counts")
    return out


def upstream_rule(prev: dict, cand: dict, n: int = 64) -> bool:
    """Published upstream gate: <=2 losses per mode, repetition cap, total gain or completion tie-break."""
    p, c = upstream_counts(prev, n), upstream_counts(cand, n)
    mode_ok = max(p[m] - c[m] for m in UPSTREAM_MODES) <= 2
    rep_ok = cand["max_rep"] <= max(3.12, prev["max_rep"] + 1.6)
    comp_gain = round(cand["mean_comp"] * 3 * n / 100) - round(prev["mean_comp"] * 3 * n / 100)
    gain = cand["total_correct"] - prev["total_correct"]
    return mode_ok and rep_ok and (gain > 0 or (gain == 0 and comp_gain >= 4))


def audit_upstream(record: dict) -> list[dict]:
    """Rows where the reported acceptance disagrees with the published rule, following the
    reported predecessor chain (the published trajectory, not a counterfactual one)."""
    n = record["gate_size_per_mode"]
    prev = record["init_gate"]
    out = []
    for g in record["generations"]:
        winners = [c for c in g["candidates"] if c["scale"] == g["winner_scale"]]
        if len(winners) != 1:
            raise ValueError("winner missing or ambiguous")
        cand = winners[0]
        by_rule = upstream_rule(prev, cand, n)
        out.append({"gen": g["gen"], "predecessor_total": prev["total_correct"],
                    "candidate_total": cand["total_correct"], "reported_accepted": g["accepted"],
                    "accepted_by_rule": by_rule, "discrepancy": by_rule != g["accepted"]})
        if g["accepted"]:
            prev = cand
    return out
