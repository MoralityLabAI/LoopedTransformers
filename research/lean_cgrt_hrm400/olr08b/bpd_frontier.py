"""Optional BPD decomposition of learned proof-state representations & recurrent frontier updates.

Aligned with MoralityLabAI/BPD (TiedComponentLinear, PassMaskSchedule, pass-tied CI,
RMS-matched Haar rotation nulls) and DataGoblin's map-kind distinction
(``svd_seed_proxy`` vs ``learned_bpd``).

Decomposes shared recurrent linear sites (graph up/down/cell and frontier recurrent
update) into rank-C factor pairs (V_m, U_m) with exact weight residual
Delta_W = W - (V_m @ U_m).T, computes pass-indexed component activations and
causal-importance schedules, and evaluates whether component-level features predict:
  1. Branch utility (on certified path / closure),
  2. Tactic applicability (Lean ok=True),
  3. Recovery following failed exploration (>= 2 consecutive failures -> progress).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

BPD_FEATURE_DIM = 16


@dataclass(frozen=True)
class BPDSiteFactor:
    """Shared factor pair (V [d_in, C], U [C, d_out]) for one tied recurrent site."""

    site_name: str
    map_kind: str  # "svd_seed_proxy" or "learned_bpd"
    rank_c: int
    v: Tensor  # [d_in, C] float32
    u: Tensor  # [C, d_out] float32
    singular_values: Tensor  # [C] float32
    recon_cosine: float
    explained_energy: float
    weight_sha256: str


def _weight_sha(w: Tensor) -> str:
    return hashlib.sha256(w.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()


def decompose_linear_svd(site_name: str, weight: Tensor, rank_c: int = 8) -> BPDSiteFactor:
    """SVD seed-proxy decomposition W_hat = (V @ U).T (strictly tagged ``svd_seed_proxy``)."""
    w64 = weight.detach().double().cpu()
    d_out, d_in = w64.shape
    k = min(rank_c, d_out, d_in)
    u_svd, s_svd, vh_svd = torch.linalg.svd(w64, full_matrices=False)
    v_mat = torch.zeros((d_in, rank_c), dtype=torch.float32)
    u_mat = torch.zeros((rank_c, d_out), dtype=torch.float32)
    s_vec = torch.zeros((rank_c,), dtype=torch.float32)
    sqrt_s = s_svd[:k].clamp_min(0.0).sqrt()
    v_mat[:, :k] = (vh_svd[:k].T * sqrt_s.unsqueeze(0)).float()
    u_mat[:k, :] = (sqrt_s.unsqueeze(1) * u_svd[:, :k].T).float()
    s_vec[:k] = s_svd[:k].float()

    w_hat = (v_mat @ u_mat).transpose(0, 1)
    w_f = weight.detach().cpu().float()
    cos = float(F.cosine_similarity(w_f.reshape(1, -1), w_hat.reshape(1, -1)).item())
    total_e = float(s_svd.pow(2).sum().item())
    ret_e = float(s_svd[:k].pow(2).sum().item())
    return BPDSiteFactor(
        site_name=site_name,
        map_kind="svd_seed_proxy",
        rank_c=rank_c,
        v=v_mat,
        u=u_mat,
        singular_values=s_vec,
        recon_cosine=cos,
        explained_energy=ret_e / max(1e-12, total_e),
        weight_sha256=_weight_sha(weight),
    )


def fit_learned_bpd_site(
    site_name: str,
    weight: Tensor,
    pass_inputs: list[Tensor],
    *,
    rank_c: int = 8,
    steps: int = 60,
    lr: float = 0.02,
    l1_sparse: float = 0.01,
    switch_weight: float = 0.005,
    seed: int = 17,
) -> BPDSiteFactor:
    """Fit a sparse, pass-consistent BPD factor pair (V, U) on recorded multi-pass inputs.

    Strictly tagged ``learned_bpd`` only after behavioral trajectory optimization over
    multi-pass inputs (distinguished from ``svd_seed_proxy`` per DataGoblin MMS contract).
    """
    init = decompose_linear_svd(site_name, weight, rank_c=rank_c)
    if not pass_inputs:
        return init
    torch.manual_seed(seed)
    v = nn.Parameter(init.v.clone())
    u = nn.Parameter(init.u.clone())
    w_target = weight.detach().cpu().float()
    xs = [x.detach().cpu().float().reshape(-1, w_target.shape[1]) for x in pass_inputs if x.numel() > 0]
    if not xs:
        return init
    # Subsample rows deterministically for fast CPU fit
    xs = [x[:128] for x in xs]
    opt = torch.optim.Adam([v, u], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        w_hat = (v @ u).transpose(0, 1)
        loss_faith = F.mse_loss(w_hat, w_target)
        loss_beh = torch.zeros(())
        pass_means = []
        for x in xs:
            y_t = F.linear(x, w_target)
            acts = x @ v
            y_h = acts @ u
            loss_beh = loss_beh + F.mse_loss(y_h, y_t)
            pass_means.append(acts.abs().mean(dim=0))
        loss_beh = loss_beh / len(xs)
        # Pass-pooled level minimality (max over passes) + temporal switch penalty
        stacked = torch.stack(pass_means, dim=0)  # [T, C]
        loss_min = stacked.amax(dim=0).mean()
        loss_sw = (stacked[1:] - stacked[:-1]).abs().mean() if len(pass_means) > 1 else torch.zeros(())
        loss = loss_faith + loss_beh + l1_sparse * loss_min + switch_weight * loss_sw
        loss.backward()
        opt.step()

    v_f = v.detach()
    u_f = u.detach()
    w_hat = (v_f @ u_f).transpose(0, 1)
    cos = float(F.cosine_similarity(w_target.reshape(1, -1), w_hat.reshape(1, -1)).item())
    err_norm = float((w_target - w_hat).pow(2).sum().item())
    tot_norm = float(w_target.pow(2).sum().item())
    s_est = (v_f.norm(dim=0) * u_f.norm(dim=1)).detach()
    return BPDSiteFactor(
        site_name=site_name,
        map_kind="learned_bpd",
        rank_c=rank_c,
        v=v_f,
        u=u_f,
        singular_values=s_est,
        recon_cosine=cos,
        explained_energy=max(0.0, 1.0 - err_norm / max(1e-12, tot_norm)),
        weight_sha256=_weight_sha(weight),
    )


def haar_rotated_factor(factor: BPDSiteFactor, seed: int = 101) -> BPDSiteFactor:
    """RMS-matched random orthogonal Haar rotation null for V (BPD nulls.py contract)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d_in = factor.v.shape[0]
    g = torch.randn((d_in, d_in), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(g)
    sign = torch.sign(torch.diag(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    v_rot = q @ factor.v
    return BPDSiteFactor(
        site_name=factor.site_name + "_haar_null",
        map_kind="haar_rotation_null",
        rank_c=factor.rank_c,
        v=v_rot,
        u=factor.u.clone(),
        singular_values=factor.singular_values.clone(),
        recon_cosine=0.0,
        explained_energy=0.0,
        weight_sha256=factor.weight_sha256,
    )


class BPDFrontierDecomposer:
    """Decomposes tied recurrent proof-state and frontier-update sites into BPD components."""

    def __init__(
        self,
        sites: dict[str, BPDSiteFactor],
        *,
        build_seconds: float = 0.0,
    ) -> None:
        self.sites = dict(sites)
        self.build_seconds = float(build_seconds)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        rank_c: int = 8,
        learned: bool = False,
        pass_inputs_by_site: dict[str, list[Tensor]] | None = None,
        seed: int = 17,
    ) -> "BPDFrontierDecomposer":
        t0 = time.perf_counter()
        target_sites: dict[str, nn.Linear] = {}
        for name in ("up", "down", "frontier_cell"):
            mod = getattr(model, name, None)
            if isinstance(mod, nn.Linear):
                target_sites[name] = mod
        cell = getattr(model, "cell", None)
        if isinstance(cell, nn.Sequential) and len(cell) >= 3 and isinstance(cell[2], nn.Linear):
            target_sites["cell_out"] = cell[2]

        factors: dict[str, BPDSiteFactor] = {}
        for name, lin in sorted(target_sites.items()):
            if learned and pass_inputs_by_site and pass_inputs_by_site.get(name):
                factors[name] = fit_learned_bpd_site(
                    name, lin.weight, pass_inputs_by_site[name], rank_c=rank_c, seed=seed
                )
            else:
                factors[name] = decompose_linear_svd(name, lin.weight, rank_c=rank_c)
        return cls(factors, build_seconds=time.perf_counter() - t0)

    def make_haar_null(self, seed: int = 101) -> "BPDFrontierDecomposer":
        null_sites = {k: haar_rotated_factor(f, seed=seed + i * 31) for i, (k, f) in enumerate(sorted(self.sites.items()))}
        return BPDFrontierDecomposer(null_sites, build_seconds=0.0)

    @torch.no_grad()
    def extract_action_features(
        self,
        pass_action_states: list[Tensor],
        frontier_state: Tensor,
    ) -> Tensor:
        """Compute [N_actions, BPD_FEATURE_DIM] component-level features across passes.

        Features (16 dims):
          - 0..7: top-8 mean-pooled component activations of ``up`` site across passes
          - 8..11: top-4 frontier-update component activations of ``frontier_cell``
          - 12: pass-to-pass temporal switch rate (|a_{tau+1} - a_tau|)
          - 13: sustaining vs transient ratio (min-over-pass / max-over-pass energy)
          - 14: active component duty fraction (> median threshold)
          - 15: component residual energy ratio
        """
        n_act = pass_action_states[0].shape[0]
        out = torch.zeros((n_act, BPD_FEATURE_DIM), dtype=torch.float32)
        up_f = self.sites.get("up")
        if up_f is not None and pass_action_states:
            # Stack [T, N, C]
            proj = torch.stack([h.detach().float() @ up_f.v for h in pass_action_states], dim=0)
            abs_p = proj.abs()
            mean_p = proj.mean(dim=0)  # [N, C]
            k = min(8, mean_p.shape[1])
            out[:, :k] = mean_p[:, :k]
            if proj.shape[0] > 1:
                sw = (proj[1:] - proj[:-1]).abs().mean(dim=(0, 2))
            else:
                sw = torch.zeros(n_act)
            out[:, 12] = sw
            p_max = abs_p.amax(dim=0).mean(dim=-1).clamp_min(1e-6)
            p_min = abs_p.amin(dim=0).mean(dim=-1)
            out[:, 13] = p_min / p_max
            thresh = abs_p.mean(dim=-1, keepdim=True)
            out[:, 14] = (abs_p > thresh).float().mean(dim=(0, 2))
            # Residual fraction on last pass
            last_h = pass_action_states[-1].detach().float()
            recon = (last_h @ up_f.v) @ up_f.u
            full_norm = last_h.norm(dim=-1).clamp_min(1e-6)
            out[:, 15] = (recon.norm(dim=-1) / full_norm).clamp(0.0, 4.0)

        fc_f = self.sites.get("frontier_cell")
        if fc_f is not None:
            fs = frontier_state.detach().float().reshape(1, -1)
            if fs.shape[1] < fc_f.v.shape[0]:
                pad = torch.zeros((1, fc_f.v.shape[0] - fs.shape[1]), dtype=torch.float32)
                fs_in = torch.cat([fs, pad], dim=1)
            else:
                fs_in = fs[:, : fc_f.v.shape[0]]
            fc_act = (fs_in @ fc_f.v).squeeze(0)  # [C]
            k2 = min(4, fc_act.numel())
            out[:, 8 : 8 + k2] = fc_act[:k2].unsqueeze(0).expand(n_act, k2)
        return out

    def store_in_cache(self, cache: Any) -> int:
        stored = 0
        for name, f in self.sites.items():
            ok = cache.put_bpd_map(
                name,
                {
                    "map_kind": f.map_kind,
                    "rank_c": f.rank_c,
                    "v": f.v.cpu().numpy(),
                    "u": f.u.cpu().numpy(),
                    "singular_values": f.singular_values.cpu().numpy(),
                    "recon_cosine": f.recon_cosine,
                    "explained_energy": f.explained_energy,
                    "weight_sha256": f.weight_sha256,
                },
            )
            stored += int(ok)
        return stored

    def summary(self) -> dict[str, Any]:
        return {
            "build_seconds": round(self.build_seconds, 6),
            "sites": {
                k: {
                    "map_kind": f.map_kind,
                    "rank_c": f.rank_c,
                    "recon_cosine": round(f.recon_cosine, 5),
                    "explained_energy": round(f.explained_energy, 5),
                    "weight_sha256": f.weight_sha256,
                }
                for k, f in sorted(self.sites.items())
            },
        }


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U exact rank formula
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Handle ties with average rank
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        for u_idx in np.where(cnt > 1)[0]:
            mask = inv == u_idx
            ranks[mask] = ranks[mask].mean()
    sum_pos = ranks[y_true == 1].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return float((sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _fit_ridge_eval(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    files_te: list[str],
    *,
    lam: float = 1.0,
    seed: int = 20270129,
) -> dict[str, Any]:
    """Fit ridge linear probe on TRAIN features and score on held-out features with file-cluster CI."""
    if len(x_tr) < 4 or len(x_te) < 4 or len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"n_train": int(len(y_tr)), "n_eval": int(len(y_te)), "pos_eval": int(y_te.sum()), "auroc": None, "brier": None}
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = x_tr.std(axis=0, keepdims=True) + 1e-6
    z_tr = (x_tr - mu) / sd
    z_te = (x_te - mu) / sd
    z_tr_b = np.concatenate([z_tr, np.ones((len(z_tr), 1), dtype=np.float64)], axis=1)
    z_te_b = np.concatenate([z_te, np.ones((len(z_te), 1), dtype=np.float64)], axis=1)
    d = z_tr_b.shape[1]
    reg = np.eye(d, dtype=np.float64) * lam
    reg[-1, -1] = 0.0
    w = np.linalg.solve(z_tr_b.T @ z_tr_b + reg, z_tr_b.T @ y_tr.astype(np.float64))
    pred = np.clip(z_te_b @ w, 0.0, 1.0)
    auc = _auroc(y_te, pred)
    brier = float(np.mean((pred - y_te) ** 2))

    # File-cluster bootstrap CI for AUROC
    by_f: dict[str, list[int]] = {}
    for idx, f in enumerate(files_te):
        by_f.setdefault(f, []).append(idx)
    fs = sorted(by_f)
    rng = random.Random(seed)
    boot_aucs = []
    if len(fs) >= 2:
        for _ in range(500):
            idxs = [i for f in rng.choices(fs, k=len(fs)) for i in by_f[f]]
            yb = y_te[idxs]
            if len(np.unique(yb)) == 2:
                boot_aucs.append(_auroc(yb, pred[idxs]))
    boot_aucs.sort()
    ci = [round(boot_aucs[int(0.025 * len(boot_aucs))], 4), round(boot_aucs[int(0.975 * len(boot_aucs)) - 1], 4)] if len(boot_aucs) >= 20 else None
    return {
        "n_train": int(len(y_tr)),
        "n_eval": int(len(y_te)),
        "pos_eval": int(y_te.sum()),
        "auroc": round(auc, 4) if math.isfinite(auc) else None,
        "auroc_ci95_file": ci,
        "brier": round(brier, 4),
    }


def evaluate_bpd_predictiveness(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether BPD component features predict utility, applicability, and recovery.

    Compares ``bpd_features`` against ``haar_null_features`` and ``prior_only_features``
    across three registered diagnostic targets:
      1. ``branch_utility`` (on_certified_path == 1)
      2. ``tactic_applicability`` (ok == 1)
      3. ``recovery_after_failure`` (ok == 1 | closed == 1, conditioned on fail_streak >= 2)
    """
    if not train_records or not eval_records:
        return {"status": "insufficient_records"}

    def unpack(recs: list[dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray([r[key] for r in recs], dtype=np.float64)

    out: dict[str, Any] = {}
    targets = [
        ("branch_utility", lambda r: True, lambda r: int(r["on_path"])),
        ("tactic_applicability", lambda r: True, lambda r: int(r["ok"])),
        ("recovery_after_failure", lambda r: int(r.get("fail_streak", 0)) >= 2, lambda r: int(r["ok"] or r.get("closed", False))),
    ]
    for tname, flt, yfn in targets:
        tr = [r for r in train_records if flt(r)]
        ev = [r for r in eval_records if flt(r)]
        if not tr or not ev:
            out[tname] = {"status": "insufficient_subset", "n_train": len(tr), "n_eval": len(ev)}
            continue
        y_tr = np.asarray([yfn(r) for r in tr], dtype=np.int64)
        y_ev = np.asarray([yfn(r) for r in ev], dtype=np.int64)
        f_ev = [str(r["file"]) for r in ev]
        out[tname] = {
            "bpd_components": _fit_ridge_eval(unpack(tr, "bpd"), y_tr, unpack(ev, "bpd"), y_ev, f_ev),
            "haar_null_components": _fit_ridge_eval(unpack(tr, "haar_null"), y_tr, unpack(ev, "haar_null"), y_ev, f_ev),
            "prior_plus_receipts": _fit_ridge_eval(unpack(tr, "cond"), y_tr, unpack(ev, "cond"), y_ev, f_ev),
            "bpd_plus_prior_receipts": _fit_ridge_eval(
                np.concatenate([unpack(tr, "bpd"), unpack(tr, "cond")], axis=1),
                y_tr,
                np.concatenate([unpack(ev, "bpd"), unpack(ev, "cond")], axis=1),
                y_ev,
                f_ev,
            ),
        }
    return out
