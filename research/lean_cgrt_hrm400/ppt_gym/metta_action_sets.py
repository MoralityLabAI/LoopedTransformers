"""DG-METTA-12-compatible action-set supervision for verified OLR transitions.

A verified state may have several equally good next actions. Normalize over
the *full legal action menu* and credit the total probability mass of all
min-cost, verifier-certified successors. Controls select either the canonical
first action or a stable hashed singleton of the same valid set.

This is a training objective, not an executable MeTTa proof certifier.
Source: MoralityLabAI/DataGoblin draft PR #57 at 88c910993b3a86438c6de85d7efa039dcdb68cac.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def normalized_action_set(
    legal_actions: Sequence[str], optimal_actions: Sequence[str]
) -> tuple[int, ...]:
    """Reject malformed or non-legal action labels and return sorted indices."""
    if (not legal_actions or len(legal_actions)>128
            or len(set(legal_actions))!=len(legal_actions)):
        raise ValueError("legal action menu must be bounded and unique")
    if not optimal_actions or len(set(optimal_actions))!=len(optimal_actions):
        raise ValueError("nonempty unique optimal action set required")
    indexes={act:i for i,act in enumerate(legal_actions)}
    if not set(optimal_actions).issubset(indexes):
        raise ValueError("optimal action outside publicly legal menu")
    return tuple(sorted(indexes[a] for a in optimal_actions))


def choose_hashed_singleton(
    *, root_id: str, state_sha256: str, model_seed: int,
    legal_actions: Sequence[str], optimal_actions: Sequence[str]
) -> str:
    """A stable tie-breaking comparator independent of PYTHONHASHSEED."""
    if not root_id or not state_sha256 or type(model_seed) is not int:
        raise ValueError("hash control requires root, state and model seed")
    normalized_action_set(legal_actions,optimal_actions)
    ordered=tuple(sorted(optimal_actions))
    # Matches DG-METTA-12's public root/seed/state/tied-action SHA256 recipe.
    key="|".join((root_id,str(model_seed),state_sha256,*ordered)).encode("utf-8")
    j=int.from_bytes(hashlib.sha256(key).digest()[:8],"big")%len(ordered)
    return ordered[j]


def objective(
    logits: torch.Tensor, legal_actions: Sequence[str],
    optimal_actions: Sequence[str], *, mode: str,
    root_id: str, state_sha256: str, model_seed: int,
) -> torch.Tensor:
    """Return a differentiable single-state loss for the declared control.

    'set' = -log sum_{a in A*} softmax_legal(logits)[a]
    'canonical' and 'hashed' are matched singleton baselines.
    """
    if logits.ndim!=1 or len(logits)!=len(legal_actions) or not logits.is_floating_point():
        raise ValueError("one float score per legal action required")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite action logits")
    allowed=normalized_action_set(legal_actions,optimal_actions)
    if mode=="set":
        return torch.logsumexp(logits,dim=0)-torch.logsumexp(logits[list(allowed)],dim=0)
    if mode=="canonical":
        idx=legal_actions.index(min(optimal_actions))
    elif mode=="hashed":
        selected=choose_hashed_singleton(root_id=root_id,state_sha256=state_sha256,
            model_seed=model_seed,legal_actions=legal_actions,optimal_actions=optimal_actions)
        idx=legal_actions.index(selected)
    else:
        raise ValueError("unknown action-set control")
    return F.cross_entropy(logits.reshape(1,-1),
                           torch.tensor([idx],device=logits.device,dtype=torch.long))


def teacher_actions_from_verified_graph(
    legal_actions: Sequence[str], transitions: dict[str,dict],
    *, remaining_calls: int, goal_is_kernel_closed: bool,
) -> tuple[str, ...]:
    """Build a local source-only action set from an independently checked DAG.

    Transitions must contain exact native/standalone proof verification and
    nonnegative *total remaining call cost* (including this action).
    The verifier's source receipt is intentionally separate from any model
    input. A STOP is eligible only when Lean has already certified the goal.
    """
    if not 0<=remaining_calls<=32:
        raise ValueError("invalid source action budget")
    if not legal_actions or len(set(legal_actions))!=len(legal_actions):
        raise ValueError("invalid observed legal action menu")
    if set(transitions)!=set(legal_actions):
        raise ValueError("missing independent receipt for candidate")
    eligible=[]
    for action in legal_actions:
        rec=transitions[action]
        if not isinstance(rec,dict):
            raise ValueError("malformed native transition receipt")
        if action=="STOP":
            if goal_is_kernel_closed and rec.get("status")=="kernel_closed":
                eligible.append((0,"STOP"))
            continue
        # Only full kernel and independently checked standalone proofs may be
        # used as positive action targets. Nonterminal alone is insufficient.
        cost=rec.get("total_tactic_calls")
        if (type(cost) is int and 1<=cost<=remaining_calls
                and rec.get("status")=="kernel_closed"
                and rec.get("native_kernel") is True
                and rec.get("standalone_verified") is True
                and rec.get("axioms")==[]):
            eligible.append((cost,action))
    if not eligible:
        return ()
    best=min(cost for cost,_ in eligible)
    return tuple(sorted(action for cost,action in eligible if cost==best))


def batch_objective(logits: torch.Tensor, rows: Sequence[dict], *,
                    mode: str, model_seed: int) -> torch.Tensor:
    """One target state per presentation, regardless of tie count."""
    if logits.ndim!=2 or len(rows)!=len(logits) or not len(rows):
        raise ValueError("expected B x action logits and aligned receipts")
    terms=[]
    for scores,row in zip(logits,rows,strict=True):
        if set(row)!=set(("root_id","state_sha256","legal","optimal")):
            raise ValueError("teacher row missing provenance or source-only action set")
        terms.append(objective(scores,row["legal"],row["optimal"],
            mode=mode,root_id=row["root_id"],state_sha256=row["state_sha256"],
            model_seed=model_seed))
    return torch.stack(terms).mean()
