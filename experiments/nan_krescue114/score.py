"""Reduce the preregistered K-specific donor-margin contrast."""
from math import isclose, isfinite
from statistics import fmean

WORLDS = ("A", "B")
SIGNS = (1, -1)
ARMS = ("active_t1", "active_t2", "sham_t1", "sham_t2")
CONDITIONS = ("edited", "k_rescue")
COMPLETE = "closed"
REFUSALS = frozenset(("native_gate_failed", "native_root_unqualified", "geometry_unqualified",
                      "mixed_duplicate_unqualified", "head_unqualified"))


def finite(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("a donor-margin value is missing or nonfinite")
    return value


def margin_footprint(rows, recipient, donor):
    if recipient == donor:
        raise ValueError("the donor and recipient labels must differ")
    if len(rows) != 32:
        raise ValueError("a root needs every world, sign, endpoint and K condition")
    observed = iter(rows)
    cells = []
    for own, peer in ((recipient, donor), (donor, recipient)):
        for sign in SIGNS:
            margins = {condition: {} for condition in CONDITIONS}
            for arm in ARMS:
                edited, rescued = next(observed), next(observed)
                if edited["rescued"] is not False or rescued["rescued"] is not True:
                    raise ValueError("missing or reordered intact and restored head records")
                if any(len(r["phases"]) != 2 for r in (edited, rescued)):
                    raise ValueError("a pass is missing from the observed vocabulary head")
                if edited["phases"][0] != rescued["phases"][0]:
                    raise ValueError("restoring K changed the pass-1 head")
                for condition, r in zip(CONDITIONS, (edited, rescued), strict=True):
                    values = r["phases"][1]["labels"]
                    margins[condition][arm] = finite(values[peer]) - finite(values[own])

            def signed_half(condition, arm):
                m = margins[condition]
                return sign * (m[arm + "_t2"] - m[arm + "_t1"]) / 2

            intact = signed_half("edited", "active") - signed_half("edited", "sham")
            restored = signed_half("k_rescue", "active") - signed_half("k_rescue", "sham")
            sham = signed_half("edited", "sham") - signed_half("k_rescue", "sham")
            cells.append({
                "intact_advantage": intact, "restored_advantage": restored,
                "removed_advantage": intact - restored, "sham_only_removal": sham,
            })
    return {k: finite(fmean(cell[k] for cell in cells)) for k in cells[0]}


def bind_parent_orientation(scored, parent):
    intact = finite(parent["second_pass"])
    restored = finite(parent["delayed_with_clean_k"]) + finite(parent["first_pass"])
    removed = finite(parent["k_rescue_removed"])
    for key, value in zip(("intact_advantage", "restored_advantage", "removed_advantage"),
                          (intact, restored, removed), strict=True):
        # The earlier audited host uses recipient-minus-peer. This protocol
        # uses donor-minus-recipient and independently reduces the raw logits.
        if not isclose(scored[key], -value, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("donor-oriented reduction disagrees with the original host audit")


def gate(rows, expected_roots, criteria):
    expected = set(expected_roots)
    ids = [row["root"] for row in rows]
    count = len(expected)
    if count != criteria["registered_roots"] or len(ids) != count or set(ids) != expected:
        raise ValueError("a decision requires the entire frozen root panel exactly once")
    for row in rows:
        state = row["status"]
        if state == COMPLETE:
            if not isclose(finite(row["removed_advantage"]),
                           finite(row["intact_advantage"]) - finite(row["restored_advantage"]),
                           rel_tol=0.0, abs_tol=1e-10):
                raise ValueError("intact minus restored does not equal the reported removal")
            finite(row["sham_only_removal"])
        elif state not in REFUSALS or set(row) != {"root", "status"}:
            raise ValueError("an unmeasured or unexpected root cannot contain an outcome")

    measured = [r for r in rows if r["status"] == COMPLETE]
    denominator = count
    mean_intact, mean_removed = (
        sum(r[k] for r in measured) / denominator for k in ("intact_advantage", "removed_advantage")
    )
    positive = sum(r["removed_advantage"] > 0 for r in measured)
    sham_max = max((abs(r["sham_only_removal"]) for r in measured), default=None)
    reasons = []
    if len(measured) < criteria["minimum_complete_roots"]:
        reasons.append("too_few_physically_closed_roots")
    if positive < criteria["minimum_positive_roots"]:
        reasons.append("too_few_positive_root_effects")
    if mean_intact < criteria["minimum_mean_intact_advantage_logits"]:
        reasons.append("intact_donor_advantage_below_minimum")
    if mean_removed < criteria["minimum_mean_removed_logits"]:
        reasons.append("removed_advantage_below_minimum")
    if sham_max is None or mean_removed <= sham_max:
        reasons.append("removal_does_not_exceed_observed_sham_maximum")
    return {
        "decision": criteria["otherwise"] if reasons else criteria["positive_planning_verdict"],
        "reasons": reasons, "registered_roots": count, "measured_roots": len(measured),
        "unmeasured_roots": count - len(measured), "positive_measured_roots": positive,
        "gate_mean_intact_logits": mean_intact, "gate_mean_removed_logits": mean_removed,
        "measured_only_mean_removed_logits": fmean(r["removed_advantage"] for r in measured) if measured else None,
        "max_observed_abs_sham_only_removal_logits": sham_max,
        "sham_tolerance_bound": None,
    }
