"""Memory-lean losses: full-vocab logits are never materialised for the whole batch (chunked + checkpointed)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_terms(hs, ht, labels, W, w_kl, w_ce):
    ls = (hs @ W.t()).float()
    logp_s = F.log_softmax(ls, -1)
    ce = -logp_s.gather(-1, labels[:, None]).squeeze(-1)
    if ht is None:
        kl = torch.zeros_like(ce)
        top1 = (ls.argmax(-1) == labels).float()
    else:
        lt = (ht @ W.t()).float()
        logp_t = F.log_softmax(lt, -1)
        kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
        top1 = (ls.argmax(-1) == lt.argmax(-1)).float()
    loss = w_kl * kl.sum() + w_ce * ce.sum()
    return loss, kl.detach().sum(), ce.detach().sum(), top1.detach().sum()


def distill_ce(h_s, h_t, ids, pred_mask, W, w_kl=1.0, w_ce=0.5, chunk=256, grad=True):
    """h_s/h_t: post-norm hidden [B,T,D]; ids [B,T]; pred_mask [B,T] marks token positions to be PREDICTED
    (position i-1 predicts token i).  Returns (loss, stats) with token-mean normalisation."""
    m = pred_mask[:, 1:]
    hs = h_s[:, :-1][m]
    ht = h_t[:, :-1][m] if h_t is not None else None
    lab = ids[:, 1:][m]
    n = max(lab.numel(), 1)
    tot = hs.new_zeros((), dtype=torch.float32)
    kl_s = ce_s = t1_s = hs.new_zeros((), dtype=torch.float32)     # stay on-device: no host syncs in the hot loop
    for i in range(0, lab.numel(), chunk):
        a = hs[i:i + chunk]
        b = ht[i:i + chunk] if ht is not None else None
        if grad and torch.is_grad_enabled():
            loss, kl, ce, t1 = checkpoint(_chunk_terms, a, b, lab[i:i + chunk], W, w_kl, w_ce, use_reentrant=False)
        else:
            loss, kl, ce, t1 = _chunk_terms(a, b, lab[i:i + chunk], W, w_kl, w_ce)
        tot = tot + loss
        kl_s = kl_s + kl
        ce_s = ce_s + ce
        t1_s = t1_s + t1
    return tot / n, {"kl": kl_s / n, "ce": ce_s / n, "top1": t1_s / n, "n": n}


def hidden_loss(student_states, teacher_states, mode: str, n_loops: int | None):
    """Normalised MSE between student and teacher residual-stream states (token 0 excluded: massive-activation
    sink).  mode: final | steps | all.   teacher_states[i] = H[i] (output after i original layers)."""
    pairs = []
    for idx, h, tag in student_states:
        if tag == "embed":
            continue
        if mode == "final" and tag not in ("merged", "site_final"):
            continue
        if mode == "steps" and tag not in ("step", "site_final", "merged"):
            continue
        pairs.append((idx, h))
    if not pairs:
        return None, 0
    tot = 0.0
    for idx, h in pairs:
        tgt = teacher_states[idx][:, 1:].float()
        d = (h[:, 1:].float() - tgt)
        tot = tot + (d.pow(2).sum(-1) / (tgt.pow(2).sum(-1) + 1e-6)).mean()
    return tot / len(pairs), len(pairs)
