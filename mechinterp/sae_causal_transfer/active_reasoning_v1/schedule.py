"""LS-SAE12 registered verified-deduction schedule, site rules and family gate.

Pure standard library. Steps are 1-based ACT indices 1..16. A blank cell's
verified deduction step T(c) is the first step from which its argmax digit
equals the exact-solver target at every later step through ACT16; T(c)=17 when
the ACT16 prediction is wrong.
"""
from __future__ import annotations
import math
import random

N_STEPS = 16
T_STAR = 4
MIN_RECOVERY_STEP = 6
NEVER = 17
FAMILY_SIZE = 12
FAMILIES = ("active_a", "active_b")
EVENT_MIN = 9
BOOT_DRAWS = 2000
BOOT_SEEDS = {"active_a": 2026111813, "active_b": 2026111814}


def blanks(puzzle):
    return [i for i, v in enumerate(puzzle) if v == 0]


def verified_schedule(preds_by_step, puzzle, target):
    """preds_by_step[s-1] is the 81-digit argmax prediction after ACT s."""
    if len(preds_by_step) != N_STEPS or any(len(p) != 81 for p in preds_by_step):
        raise ValueError("verified schedule needs 16 complete 81-cell predictions")
    out = {}
    for c in blanks(puzzle):
        t = NEVER
        for s in range(N_STEPS, 0, -1):
            if preds_by_step[s - 1][c] == target[c]:
                t = s
            else:
                break
        out[c] = t
    return out


def recovering_errors(preds_by_step, puzzle, target, schedule=None):
    """Cells wrong after ACT T_STAR whose verified deduction is at step >= MIN_RECOVERY_STEP."""
    sched = schedule or verified_schedule(preds_by_step, puzzle, target)
    at = preds_by_step[T_STAR - 1]
    return sorted(c for c in blanks(puzzle)
                  if at[c] != target[c] and MIN_RECOVERY_STEP <= sched[c] <= N_STEPS)


def select_site(preds_by_step, puzzle, target):
    """Observable-error rule: smallest verified recovery step, then smallest cell index."""
    sched = verified_schedule(preds_by_step, puzzle, target)
    rec = recovering_errors(preds_by_step, puzzle, target, sched)
    if not rec:
        return None, rec, sched
    site = min(rec, key=lambda c: (sched[c], c))
    return site, rec, sched


def control_site(puzzle):
    """Model-blind rule inherited from LS-SAE11: median sorted blank position."""
    b = blanks(puzzle)
    return b[len(b) // 2]


def deduction_change(clean_sched, edit_sched, site):
    if set(clean_sched) != set(edit_sched):
        raise ValueError("schedules cover different blank cells")
    vdd = sum(abs(edit_sched[c] - clean_sched[c]) for c in clean_sched)
    return {
        "vdd": vdd,
        "blocked_recoveries": sum(clean_sched[c] < NEVER and edit_sched[c] == NEVER for c in clean_sched),
        "new_recoveries": sum(clean_sched[c] == NEVER and edit_sched[c] < NEVER for c in clean_sched),
        "accelerated": sum(edit_sched[c] < clean_sched[c] for c in clean_sched),
        "delayed": sum(edit_sched[c] > clean_sched[c] for c in clean_sched),
        "final_correct": sum(edit_sched[c] < NEVER for c in edit_sched),
        "site_T": edit_sched[site],
        "site_shift": edit_sched[site] - clean_sched[site],
        "site_blocked": clean_sched[site] < NEVER and edit_sched[site] == NEVER,
    }


def assign_families(eligible_positions):
    """First 12 eligible pool positions -> active_a, next 12 -> active_b."""
    ordered = sorted(eligible_positions)
    out = {"active_a": ordered[:FAMILY_SIZE], "active_b": ordered[FAMILY_SIZE:2 * FAMILY_SIZE]}
    out["complete"] = {f: len(out[f]) == FAMILY_SIZE for f in FAMILIES}
    return out


ISO_DRAWS = 8
ISO_SEED_BASE = 2026111800000
POWER_SEED_BASE = 2026111900000
SITE_RULES = ("observable_error", "model_blind_median")


def direction_seed(base, pool_position, site_rule, k=0):
    return base + 1000 * pool_position + 100 * SITE_RULES.index(site_rule) + k


def gaussian_unit(seed, dim=512):
    """Stdlib-reproducible isotropic unit vector (random.Random.gauss)."""
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def family_gate(diffs, family):
    """>= 9 of 12 strictly positive, and registered bootstrap lower bound > 0."""
    if family not in BOOT_SEEDS:
        raise ValueError("unregistered LS-SAE12 family")
    n = len(diffs)
    if n != FAMILY_SIZE:
        return {"n_states": n, "complete": False, "registered_success": None}
    rng = random.Random(BOOT_SEEDS[family])
    draws = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT_DRAWS))
    events = sum(d > 0 for d in diffs)
    return {"n_states": n, "complete": True, "diffs": list(diffs), "events_positive": events,
            "mean": sum(diffs) / n, "bootstrap95": [draws[50], draws[1949]],
            "bootstrap_draws": BOOT_DRAWS, "bootstrap_seed": BOOT_SEEDS[family],
            "registered_success": bool(events >= EVENT_MIN and draws[50] > 0)}
