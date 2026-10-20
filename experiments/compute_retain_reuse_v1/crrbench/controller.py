"""HRM-style adaptive controller over a frozen recurrent core.

After every loop k the controller sees the loop's readout features and chooses

    CONTINUE  : keep computing; the output register tracks the live prediction.
    PRESERVE  : latch the current prediction into the output register (or keep an
                existing latch) and keep computing.  Later predictions cannot
                overwrite a latched answer unless the controller chooses CONTINUE.
    HALT      : stop and return the register (the latch if set, else the live prediction).

Like HRM's adaptive-computation-time head, it is a small Q-network (one Q-value per
action) trained with Q-learning; here it is also conditioned on a per-loop compute price
lambda, so one network traces the whole accuracy/FLOPs frontier.  The base model is frozen
and does not depend on the controller's actions, so the MDP is simulated exactly from
pre-computed per-loop trajectories.

Reward: +1 if the returned answer is correct, -lambda per loop executed.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONT, PRES, HALT = 0, 1, 2
N_ACT = 3


class TrajTensors:
    """Per-loop features for a split, as torch tensors on `device`."""

    def __init__(self, split, device="cuda"):
        lg = torch.from_numpy(split["logits"]).float().to(device)       # [n,K,C]
        self.n, self.K, self.C = lg.shape
        self.z = torch.from_numpy(split["z"]).float().to(device)        # [n,K,d]
        self.y = torch.from_numpy(split["y"]).to(device)
        self.N = torch.from_numpy(split["N"]).to(device)
        self.probs = F.softmax(lg, -1)
        self.pred = lg.argmax(-1)                                       # [n,K]
        self.correct = self.pred == self.y[:, None]
        top2 = self.probs.topk(2, -1).values
        self.maxp = top2[..., 0]
        self.margin = top2[..., 0] - top2[..., 1]
        self.ent = -(self.probs * torch.log(self.probs.clamp_min(1e-9))).sum(-1) / math.log(self.C)
        prev = torch.cat([self.pred[:, :1] - 1, self.pred[:, :-1]], 1)
        self.agree_prev = (self.pred == prev).float()
        run = torch.zeros_like(self.agree_prev)                         # length of the current identical-argmax run
        for k in range(self.K):
            run[:, k] = 1 if k == 0 else torch.where(self.agree_prev[:, k] > 0, run[:, k - 1] + 1, torch.ones_like(run[:, k]))
        self.run = run / self.K
        self.base_dim = self.z.shape[-1] + 6                            # z + 6 scalar features

    def base(self, idx, k):
        """Features for rows idx at 0-indexed loop k."""
        sc = torch.stack([self.maxp[idx, k], self.margin[idx, k], self.ent[idx, k], self.agree_prev[idx, k],
                          self.run[idx, k], torch.full_like(self.maxp[idx, k], (k + 1) / self.K)], -1)
        return torch.cat([self.z[idx, k], sc], -1)


LOCK_DIM = 4


def obs(T, idx, k, lock, lockpred, lock_k, lam_feat):
    """Network input: base features, latch features, price feature."""
    b = T.base(idx, k)
    lp = T.probs[idx, k].gather(1, lockpred[:, None])[:, 0]
    lf = torch.stack([lock.float(), (T.pred[idx, k] == lockpred).float() * lock.float(), lp * lock.float(),
                      (k - lock_k).clamp_min(0).float() / T.K * lock.float()], -1)
    return torch.cat([b, lf, lam_feat[:, None]], -1)


def lam_to_feat(lam):
    return torch.log10(lam.clamp_min(1e-4)) / 2 + 1.0  # roughly in [-0.5, 1]


class QNet(nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, N_ACT))
        self.d_in, self.hidden = d_in, hidden

    def forward(self, x):
        return self.net(x)


def mask_q(q, k, K):
    """At the last loop only HALT is legal."""
    if k == K - 1:
        q = q.clone()
        q[:, CONT] = -1e9
        q[:, PRES] = -1e9
    return q


def step_state(a, T, idx, k, lock, lockpred, lock_k):
    """Apply action `a` (for alive rows) to the register state."""
    live = T.pred[idx, k]
    new_lock = torch.where(a == PRES, torch.ones_like(lock), torch.where(a == CONT, torch.zeros_like(lock), lock))
    set_new = (a == PRES) & ~lock
    lockpred = torch.where(set_new, live, lockpred)
    lock_k = torch.where(set_new, torch.full_like(lock_k, k), lock_k)
    return new_lock, lockpred, lock_k


def train_controller(Ttr, cfg, device="cuda", seed=0, log=None):
    g = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    d_in = Ttr.base_dim + LOCK_DIM + 1
    q = QNet(d_in, cfg["ctrl_hidden"]).to(device)
    qt = QNet(d_in, cfg["ctrl_hidden"]).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["ctrl_lr"])
    K, B = Ttr.K, cfg["ctrl_rollout"]
    n_it = cfg["ctrl_iters"]
    for it in range(n_it):
        eps = max(0.05, 0.5 * (1 - it / (0.6 * n_it)))
        idx_all = torch.randint(0, Ttr.n, (B,), device=device, generator=g)
        lam = 10 ** (torch.rand(B, device=device, generator=g) * 3.1 - 3.3)         # 5e-4 .. 0.63 log-uniform
        lam = torch.where(torch.rand(B, device=device, generator=g) < 0.08, torch.zeros_like(lam), lam)
        lf = lam_to_feat(lam)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        lock = torch.zeros(B, dtype=torch.bool, device=device)
        lockpred = torch.zeros(B, dtype=torch.long, device=device)
        lock_k = torch.zeros(B, dtype=torch.long, device=device)
        S, A, R, D, S2, NK, H_T = [], [], [], [], [], [], []
        prev_obs = prev_a = prev_alive = None
        with torch.no_grad():
            for k in range(K):
                o = obs(Ttr, idx_all, k, lock, lockpred, lock_k, lf)
                qv = mask_q(q(o), k, K)
                a = qv.argmax(-1)
                rnd = torch.randint(0, N_ACT, (B,), device=device, generator=g)
                a = torch.where(torch.rand(B, device=device, generator=g) < eps, rnd, a)
                if k == K - 1:
                    a = torch.full_like(a, HALT)
                # halting reward is observable at every visited state -> supervise Q_halt on all of them
                ans = torch.where(lock, lockpred, Ttr.pred[idx_all, k])
                halt_r = (ans == Ttr.y[idx_all]).float()
                # transition
                nl, nlp, nlk = step_state(a, Ttr, idx_all, k, lock, lockpred, lock_k)
                S.append(o[alive]); A.append(a[alive]); H_T.append(halt_r[alive])
                R.append(torch.where(a == HALT, halt_r, -lam)[alive])
                D.append((a == HALT)[alive])
                if k < K - 1:
                    o2 = obs(Ttr, idx_all, k + 1, nl, nlp, nlk, lf)
                    S2.append(o2[alive])
                else:
                    S2.append(o[alive])  # unused (done)
                NK.append(torch.full((int(alive.sum()),), k + 1, device=device))
                alive = alive & (a != HALT)
                lock, lockpred, lock_k = nl, nlp, nlk
        S, A, R, D, S2, NK, H_T = (torch.cat(v) for v in (S, A, R, D, S2, NK, H_T))
        n_tr = S.shape[0]
        for _ in range(cfg["ctrl_epochs"]):
            perm = torch.randperm(n_tr, device=device, generator=g)[: cfg["ctrl_mb"]]
            with torch.no_grad():
                q2 = qt(S2[perm])
                last = NK[perm] >= K - 1  # next state is the last loop -> only HALT legal
                q2 = torch.where(last[:, None] & (torch.arange(N_ACT, device=device)[None] != HALT), torch.full_like(q2, -1e9), q2)
                tgt = R[perm] + (~D[perm]).float() * q2.max(-1).values
            qa = q(S[perm])
            loss = F.smooth_l1_loss(qa.gather(1, A[perm][:, None])[:, 0], tgt) \
                + F.mse_loss(qa[:, HALT], H_T[perm])  # Q_halt = P(correct if we stop now), trained on every visited state
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()
        if (it + 1) % cfg["ctrl_target_every"] == 0:
            qt.load_state_dict(q.state_dict())
        if log and (it % max(1, n_it // 8) == 0 or it == n_it - 1):
            log(f"    ctrl it {it:4d} eps {eps:.2f} loss {loss.item():.4f} trans {n_tr}")
    q.eval()
    return q


@torch.no_grad()
def run_policy(q, T, lam, device="cuda", halt_bias=0.0, idx=None):
    """Deterministic greedy rollout at price `lam`.  Returns dict of per-sample numpy arrays."""
    n = T.n if idx is None else len(idx)
    idx_all = torch.arange(T.n, device=device) if idx is None else idx
    K = T.K
    lamv = torch.full((n,), float(lam), device=device)
    lf = lam_to_feat(lamv)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    lock = torch.zeros(n, dtype=torch.bool, device=device)
    lockpred = torch.zeros(n, dtype=torch.long, device=device)
    lock_k = torch.zeros(n, dtype=torch.long, device=device)
    tau = torch.full((n,), K, device=device)
    ans = torch.zeros(n, dtype=torch.long, device=device)
    qhalt = torch.full((n, K), float("nan"), device=device)
    acts = torch.full((n, K), -1, dtype=torch.long, device=device)
    for k in range(K):
        o = obs(T, idx_all, k, lock, lockpred, lock_k, lf)
        q_raw = q(o)
        qv = q_raw.clone()
        qv[:, HALT] += halt_bias
        qv = mask_q(qv, k, K)
        a = qv.argmax(-1)
        a = torch.where(alive, a, torch.full_like(a, HALT))
        cur = torch.where(lock, lockpred, T.pred[idx_all, k])
        qhalt[:, k] = torch.where(alive, q_raw[:, HALT], qhalt[:, k])
        acts[:, k] = torch.where(alive, a, acts[:, k])
        halting = alive & (a == HALT)
        tau = torch.where(halting, torch.full_like(tau, k + 1), tau)
        ans = torch.where(halting, cur, ans)
        nl, nlp, nlk = step_state(a, T, idx_all, k, lock, lockpred, lock_k)
        lock = torch.where(alive, nl, lock); lockpred = torch.where(alive, nlp, lockpred); lock_k = torch.where(alive, nlk, lock_k)
        alive = alive & (a != HALT)
    y = T.y[idx_all]
    return {"tau": tau.cpu().numpy(), "ans": ans.cpu().numpy(), "correct": (ans == y).cpu().numpy(),
            "qhalt": qhalt.cpu().numpy(), "acts": acts.cpu().numpy()}
