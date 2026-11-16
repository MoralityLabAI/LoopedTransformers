"""AF-26 frozen neural STOP-gate × legal-rule-ranking interventions.

The binary STOP gate is determined *within its native scorer*; its score is
never added to or compared on the other scorer's numerical scale. The rule
ranking is selected independently among host-enabled physical rule slots.
All arms share exactly one set of original Direct/VI-2 scores per state.
No Torch, oracle labels or trained weights are used in this module.
"""
from __future__ import annotations

ARMS = ("dd", "dv", "vd", "vv", "nd", "nv")
RULES = 14
STOP = 14
GATE = {"dd":"direct", "dv":"direct", "vd":"vi2", "vv":"vi2",
        "nd":"never", "nv":"never"}
RANK = {"dd":"direct", "dv":"vi2", "vd":"direct", "vv":"vi2",
        "nd":"direct", "nv":"vi2"}


def _best(scores):
    # Original AF-23/AF-25 score_from_nodes chooses the first *physical
    # index* on ties, including STOP at final index 14.
    return max(range(RULES+1), key=lambda i: scores[i])


def choose(scores, legal, arm, source_selected=None):
    if arm not in ARMS:
        raise ValueError(f"unregistered AF-26 mode {arm!r}")
    if not isinstance(scores, dict) or not {"direct","vi2"} <= set(scores):
        raise ValueError("both original score vectors are required")
    if (len(scores["direct"]) != RULES+1 or
            len(scores["vi2"]) != RULES+1):
        raise ValueError("original neural score vector must have 15 slots")
    legal = tuple(legal)
    if len(set(legal)) != len(legal) or any(
            not isinstance(a,int) or not 0 <= a < RULES for a in legal):
        raise ValueError("invalid host-legal physical action mask")
    if tuple(sorted(legal)) != legal:
        raise ValueError("physical host action mask must be sorted")
    source = {m: _best(scores[m]) for m in ("direct","vi2")}
    if any(a not in legal and a != STOP for a in source.values()):
        raise ValueError("frozen original policy selected illegal rule")
    if source_selected is not None:
        if any(source_selected.get(m) != source[m] for m in source):
            raise ValueError("original full-vector action selection mismatch")
    gate = GATE[arm]
    rank = RANK[arm]
    best_rule = {
        m: max(legal,key=lambda a:scores[m][a]) if legal else None
        for m in ("direct","vi2")
    }
    if not legal:
        selected = STOP
        gate_stop = True
    else:
        gate_stop = gate != "never" and source[gate] == STOP
        selected = STOP if gate_stop else best_rule[rank]
    if selected != STOP and selected not in legal:
        raise ValueError("attempt to fire illegal physical rule")
    # Exact replays of the two original AF-25 policies must be possible.
    if arm == "dd" and selected != source["direct"]:
        raise ValueError("DD must exactly reproduce source Direct")
    if arm == "vv" and selected != source["vi2"]:
        raise ValueError("VV must exactly reproduce source VI-2")
    return {
        "arm":arm,"chosen":selected,
        "gate_source":gate,"rule_source":rank,
        "gate_selected_stop":bool(gate_stop),
        "source_selected":source,
        "best_rule":best_rule,
        "legal_count":len(legal),
    }


def factor_contrast(count):
    """Descriptive fixed 2x2 differences; no claim of additive mediation."""
    for key in ("dd","dv","vd","vv"):
        if key not in count:
            raise ValueError("missing preregistered factorial arm")
    dd,dv,vd,vv=(count[k] for k in ("dd","dv","vd","vv"))
    return {
        "direct_minus_vi2":dd-vv,
        "stop_swap_recovery":dv-vv,
        "rule_swap_recovery":vd-vv,
        "interaction":dd-dv-vd+vv,
        "dd":dd,"dv":dv,"vd":vd,"vv":vv,
    }


def outcome(hard_counts, original_gap=109, threshold=55):
    diffs = factor_contrast(hard_counts)
    if diffs["direct_minus_vi2"] != original_gap or threshold != 55:
        raise ValueError("AF-26 registered original hard gap or threshold changed")
    stop = diffs["stop_swap_recovery"] >= threshold
    rule = diffs["rule_swap_recovery"] >= threshold
    return ("BOTH" if stop and rule else
            "STOP_ONLY" if stop else
            "RULE_ONLY" if rule else "NEITHER")
