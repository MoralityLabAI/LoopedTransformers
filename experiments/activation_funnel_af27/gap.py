"""AF-27 source-only score separation and Bellman legal-action gap receipts.

This module sees only original model outputs, public host states and rule
legality. Root-optimal action truth comes from a separate post-episode auditor.
Native Direct and VI2 margins remain in their respective scales. The
two-step value decomposition identifies which interior STOP branch binds.
"""
from __future__ import annotations
import math

STOP=14
LEGAL_RULES=14


def legal_margin(vector, legal):
    legal=tuple(legal)
    if len(vector)!=15 or not legal:
        raise ValueError("expected complete 15-slot source logits and legal actions")
    if tuple(sorted(set(legal)))!=legal or any(not 0<=a<LEGAL_RULES for a in legal):
        raise ValueError("invalid ordered host rule slots")
    ordered=sorted(legal,key=lambda a:(-vector[a],a))
    top=ordered[0]
    second=ordered[1] if len(ordered)>1 else None
    values=[vector[top]]+([vector[second]] if second is not None else [])
    if not all(math.isfinite(x) for x in values):
        raise ValueError("non-finite native action margin")
    return {
        "top_rule":top,
        "runner_up_rule":second,
        "top_score":float(vector[top]),
        "runner_up_score":float(vector[second]) if second is not None else None,
        "margin":float(vector[top]-vector[second]) if second is not None else None,
        "near_tie_1e5":bool(second is not None and
                           vector[top]-vector[second]<1e-5),
        "stop_minus_top_rule":float(vector[STOP]-vector[top]),
        "source_prefers_stop":bool(vector[STOP]>vector[top]),
    }


def gap_receipt(scores,legal):
    if set(scores)<{"direct","vi2"}:
        raise ValueError("missing independent native scorer outputs")
    d=legal_margin(scores["direct"],legal)
    v=legal_margin(scores["vi2"],legal)
    return {
        "direct":d,"vi2":v,
        "same_top_legal_rule":d["top_rule"]==v["top_rule"],
        "direct_continues":not d["source_prefers_stop"],
        "vi2_continues":not v["source_prefers_stop"],
    }


def value_path_details(trace, legal):
    """Decompose the frozen original VI2 cost for each legal first action.

    On every successor, identify whether the learned STOP boundary is the
    minimizer and report the exact numerical leaf-continuation alternative.
    The tool does not receive or compute true reachability or proof length.
    """
    nodes={tuple(n["state"]):n for n in trace["nodes"]}
    root=tuple(trace["nodes"][0]["state"])
    if not trace.get("root_raw"):
        raise ValueError("missing original model root action logits")
    # In original AF23 state_graph traversal the first entry is the root.
    source=nodes[root]
    parts={}
    for a in legal:
        target=source["edges"].get(str(a))
        if target is None:
            raise ValueError("missing public successor edge for legal rule")
        follow=nodes.get(tuple(target))
        if follow is None:
            raise ValueError("model-visited graph omits legal successor")
        legal_next=list(follow["edges"].values())
        cont=[1.+nodes[tuple(s)]["value0"] for s in legal_next]
        if 14 in tuple(target):
            value=0.
            stop_binds=False
            alternative=None
        else:
            stop_cost=float(follow["stop_cost"])
            alternative=min(cont) if cont else None
            value=min([stop_cost]+cont)
            stop_binds=bool(cont and stop_cost <= alternative)
        if not math.isfinite(value):
            raise ValueError("nonfinite successor learned cost")
        parts[str(a)]={
            "successor_value0":float(follow["value0"]),
            "successor_stop_cost":float(follow["stop_cost"]),
            "min_continuation_cost":alternative,
            "interior_stop_binds":stop_binds,
            "two_step_rule_score":float(-1.-value),
        }
    return parts


def evaluate_root_gap(c, scores, legal, trace, optimal_actions):
    """Posthoc root optimality is *explicitly* analysis-only."""
    r=gap_receipt(scores,legal)
    opt=tuple(optimal_actions)
    if not opt or any(a!=STOP and a not in legal for a in opt):
        raise ValueError("host-audited optimal set is invalid")
    r["exact_host_optimal_rules"]=list(opt)
    r["direct_top_rule_optimal"]=r["direct"]["top_rule"] in opt
    r["vi2_top_rule_optimal"]=r["vi2"]["top_rule"] in opt
    r["direct_vs_vi2_regret_disagreement"]=(
        r["direct_top_rule_optimal"] and not r["vi2_top_rule_optimal"]
    )
    r["vi2_vs_direct_regret_disagreement"]=(
        r["vi2_top_rule_optimal"] and not r["direct_top_rule_optimal"]
    )
    r["value_paths"]=value_path_details(trace,legal)
    for a in legal:
        if abs(r["value_paths"][str(a)]["two_step_rule_score"]-
               scores["vi2"][a]) > 1e-5:
            raise ValueError("frozen VI2 min-plus rule score failed replay")
    return r
