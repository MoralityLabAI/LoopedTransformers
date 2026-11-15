"""Shared numerical primitives for ARG-03-N / LOOPSCOPE-04.

Zero checkpoint downloads on import. All manifold operations are float32/64.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row_basis(head: torch.Tensor, relative_tol: float = 1e-8) -> torch.Tensor:
    """An orthonormal [D,r] basis of the head rowspace, constructed on CPU."""
    w = head.detach().double().cpu()
    if w.ndim != 2:
        raise ValueError("head must have [readout,hidden] shape")
    _, singular, vh = torch.linalg.svd(w, full_matrices=False)
    rank = int((singular > singular[0] * relative_tol).sum().item())
    return vh[:rank].T.contiguous().float()


def project_visible(state: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    b = basis.to(device=state.device, dtype=state.dtype)
    return (state @ b) @ b.T


def trm_manifold_swap(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    head: torch.Tensor,
    *,
    board_start: int = 16,
    atol: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Replace only board-cell head-orthogonal content, preserving logits and L2 radius.

    Works with [batch,97,512] states at a POST-RMSNorm ACT boundary.
    Prefix and z_L must be handled by caller; this function keeps prefix untouched.
    Rejection is preferable to claiming a matched fiber after low-precision drift.
    """
    if recipient.shape != donor.shape or recipient.ndim != 3:
        raise ValueError("recipient and donor must be equal-shape [B,T,D]")
    if not (0 <= board_start < recipient.shape[1]):
        raise ValueError("invalid board_start")
    if recipient.dtype != torch.float32 or donor.dtype != torch.float32:
        raise ValueError("use float32 TRM states for a strict readout equality audit")
    basis = row_basis(head).to(recipient.device)
    r = recipient[:, board_start:, :]
    d = donor[:, board_start:, :]
    rv = project_visible(r, basis)
    du = d - project_visible(d, basis)
    radius2 = (r.square().sum(-1, keepdim=True) - rv.square().sum(-1, keepdim=True)).clamp_min(0.0)
    if torch.any(du.norm(dim=-1, keepdim=True) < 1e-7):
        raise ValueError("donor has zero head-orthogonal carrier")
    replacement = rv + du / du.norm(dim=-1, keepdim=True) * radius2.sqrt()
    out = recipient.clone()
    out[:, board_start:, :] = replacement
    w = head.float().to(recipient.device)
    logit_err = float((out[:, board_start:, :] @ w.T - r @ w.T).abs().max().item())
    radius_err = float((out[:, board_start:, :].norm(dim=-1) - r.norm(dim=-1)).abs().max().item())
    if logit_err > atol or radius_err > atol:
        raise AssertionError(f"manifold check failed: logits {logit_err}, radius {radius_err}")
    return out, {"max_full_head_logit_error": logit_err, "max_cell_l2_error": radius_err,
                 "head_rank": int(basis.shape[1]), "orthogonal_dim": int(head.shape[1] - basis.shape[1])}


def invisible_context_positions(length: int, *, width: int = 8, readout_tail: int = 3) -> list[int]:
    """Positions preceding answer-site readouts, avoiding the final readout_tail tokens."""
    if length <= readout_tail or width < 1:
        raise ValueError("no eligible context positions")
    hi = length - readout_tail
    return list(range(max(0, hi - width), hi))


def nanbeige_context_swap(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    *,
    width: int = 8,
    readout_tail: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Intervene on context positions absent from the full-vocab readout.

    This is a *sequence-level observational fiber*, NOT an assertion of a
    single-token full-vocabulary LM-head kernel. Donor must be aligned to
    recipient token positions; callers generate same-prompt donors by default.
    """
    if recipient.ndim != 3 or donor.shape != recipient.shape:
        raise ValueError("aligned [candidate_rows, sequence, hidden] required")
    positions = invisible_context_positions(recipient.shape[1], width=width, readout_tail=readout_tail)
    patched = recipient.clone()
    patched[:, positions, :] = donor[:, positions, :]
    if not torch.equal(recipient[:, -readout_tail:, :], patched[:, -readout_tail:, :]):
        raise AssertionError("readout sites changed")
    return patched, {"positions": positions, "readout_tail": readout_tail,
                     "patch_l2": float((patched.float()-recipient.float()).norm().item())}


def arg02_program_scores(logprobs: torch.Tensor, op_ids: dict[str, int]) -> list[float]:
    """64 joint log probabilities using ARG-02's 16-prefix-row layout.

    Rows = (a,b), each ending with those two operator tokens. The final
    three positions predict a, b (given a), and c (given a,b), respectively.
    """
    if logprobs.ndim != 3 or logprobs.shape[0] != 16 or logprobs.shape[1] != 3:
        raise ValueError("expected ARG-02 full-logprob tensor [16,3,vocab]")
    ops = "INSR"
    tok = [int(op_ids[c]) for c in ops]
    lp = logprobs.detach().float()
    scores = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                scores.append(float(lp[0,0,tok[i]] + lp[4*i,1,tok[j]] + lp[4*i+j,2,tok[k]]))
    return scores


def write_json(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def check_split_overlap(train: list[str], test: list[str]) -> dict:
    tr, te = set(train), set(test)
    if len(tr) != len(train) or len(te) != len(test):
        raise ValueError("duplicate samples within a split")
    overlap = tr & te
    if overlap:
        raise ValueError(f"split contamination: {len(overlap)} identical identifiers")
    return {"train_count": len(train), "test_count": len(test), "overlap": 0}
