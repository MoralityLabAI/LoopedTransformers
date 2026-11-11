"""Mini-LaCo replica: small causal Qwen3-style teacher -> LaCo collapse -> recurrent recovery.

Uses the frozen `looprec` snapshot (Qwen3LM, collapse_blocks, RecurrentSite, loop-specific LoRA, hidden_loss) at a
scale that trains on CPU, so the *whole* compression -> recovery -> SAE pipeline can be replicated over seeds.
The task is an automaton word problem (crrbench machines) with a label at EVERY prefix position.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

import recsae  # noqa: F401
from looprec.laco import (build_student, collapse_blocks, count_trainable, laco_search_single, plan_similarity,
                          trainable_params)
from looprec.losses import hidden_loss
from looprec.qwen3_core import Block, Qwen3LM, QCfg, RecurrentSite, StepCtx, rope_cos_sin
from crrbench.tasks import TASKS


# ------------------------------------------------------------------ task / data
class PrefixTask:
    """Tokens 0..n_sym-1 are input symbols; tokens n_sym..n_sym+C-1 are class (automaton-state) labels."""

    def __init__(self, name: str):
        self.name = name
        self.auto = TASKS[name]()
        assert self.auto.n_ctx == 1, "mini replica supports single-machine automata"
        self.n_sym, self.C = self.auto.n_symbols, self.auto.n_classes
        self.vocab = self.n_sym + self.C
        self.tables = self.auto.tables[0]
        self.init = int(self.auto.init[0])

    def sample(self, rng: np.random.Generator, B: int, N: int):
        sym = rng.integers(0, self.n_sym, size=(B, N))
        s = np.full(B, self.init)
        lab = np.zeros((B, N), np.int64)
        for t in range(N):
            s = self.tables[s, sym[:, t]]
            lab[:, t] = s
        return sym.astype(np.int64), lab


def make_cfg(task: PrefixTask, layers=12, d=128, heads=4, kv=2) -> QCfg:
    return QCfg(vocab_size=task.vocab, hidden_size=d, num_layers=layers, num_heads=heads, num_kv_heads=kv,
                head_dim=d // heads, inter=3 * d, rope_theta=10000.0, eps=1e-6, tie=True)


def new_teacher(task: PrefixTask, layers: int, d: int, seed: int) -> Qwen3LM:
    torch.manual_seed(seed)
    m = Qwen3LM(make_cfg(task, layers, d))
    with torch.no_grad():
        m.embed.weight.normal_(0, 0.5)
    for i, e in enumerate(m.entries):
        e.t_end = i
    return m


def lm_loss(model: Qwen3LM, ids, lab, task: PrefixTask, n_loops=None):
    h, _ = model.forward_hidden(ids, n_loops=n_loops)
    lg = model.logits(h)
    tgt = lab + task.n_sym
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)), lg


def train_teacher(task: PrefixTask, layers=12, d=128, seed=0, steps=3000, batch=128, lo=2, hi=16, lr=2e-3,
                  log=print):
    m = new_teacher(task, layers, d, seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps)))))
    t0 = time.time()
    for step in range(steps):
        N = int(rng.integers(lo, hi + 1))
        x, y = task.sample(rng, batch, N)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        loss, lg = lm_loss(m, x, y, task)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            acc = (lg.argmax(-1)[:, -1] == (y[:, -1] + task.n_sym)).float().mean().item()
            log(f"  [teacher {task.name} L{layers} s{seed}] step {step:5d} loss {loss.item():.4f} last-acc {acc:.3f} "
                f"({time.time()-t0:.0f}s)")
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


@torch.no_grad()
def eval_last_acc(model, task: PrefixTask, rng, lengths, n=500, n_loops=None):
    out = {}
    for N in lengths:
        x, y = task.sample(rng, n, N)
        h, _ = model.forward_hidden(torch.from_numpy(x), n_loops=n_loops)
        pred = model.logits(h)[:, -1].argmax(-1) - task.n_sym
        out[N] = float((pred.numpy() == y[:, -1]).mean())
    return out


# ------------------------------------------------------------------ LaCo + recovery
@torch.no_grad()
def laco_plan(teacher: Qwen3LM, task: PrefixTask, n_remove: int, seed=0, n_calib=64, N=16, protect_first=1,
              protect_last=1):
    rng = np.random.default_rng(10_000 + seed)
    x, _ = task.sample(rng, n_calib, N)
    merges, table = laco_search_single(teacher, torch.from_numpy(x), n_remove, protect_first=protect_first,
                                       protect_last=protect_last)
    return merges, table


def build_laco_student(teacher, merges, kind: str, T=4, rank=16, alpha=32.0, adapter="loop", norm="frozen",
                       init="laco"):
    """kind: 'laco' (merged block, single pass, no recovery) | 'rec' (recurrent site with adapters)."""
    if kind == "laco":
        return build_student(teacher, merges)
    rec = dict(max_loops=T, adapter=adapter, rank=rank, alpha=alpha, norm=norm, gate=True, init=init, extrap="clamp")
    return build_student(teacher, merges, rec=rec)


def recovery_loss(student, teacher, ids, lab, task: PrefixTask, T, w_h=1.0, w_kl=1.0, w_ce=0.5, hmode="steps"):
    with torch.no_grad():
        th, tstates = teacher.forward_hidden(ids, collect=True)
        tH = {i: h for i, h, _ in tstates}
        tl = teacher.logits(th)
    sh, sstates = student.forward_hidden(ids, n_loops=T, collect=True)
    sl = student.logits(sh)
    hl, _ = hidden_loss(sstates, tH, hmode, T)
    lp_s = F.log_softmax(sl.float(), -1)
    lp_t = F.log_softmax(tl.float(), -1)
    kl = (lp_t.exp() * (lp_t - lp_s)).sum(-1).mean()
    ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), (lab + task.n_sym).reshape(-1))
    loss = w_kl * kl + w_ce * ce + (w_h * hl if hl is not None else 0.0)
    return loss, {"kl": float(kl), "ce": float(ce), "h": float(hl) if hl is not None else 0.0}


def train_recovery(student, teacher, task: PrefixTask, T, seed=0, steps=1500, batch=128, lo=2, hi=16, lr=3e-3,
                   w_h=1.0, w_kl=1.0, w_ce=0.5, hmode="steps", log=print, tag=""):
    params = [p for _, p in trainable_params(student)]
    assert params, "no trainable parameters"
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps)))))
    rng = np.random.default_rng(777 + seed)
    t0 = time.time()
    student.train()
    for step in range(steps):
        N = int(rng.integers(lo, hi + 1))
        x, y = task.sample(rng, batch, N)
        loss, st = recovery_loss(student, teacher, torch.from_numpy(x), torch.from_numpy(y), task, T, w_h, w_kl, w_ce,
                                 hmode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"  [recover {tag}] step {step:5d} loss {loss.item():.4f} {st} ({time.time()-t0:.0f}s)")
    student.eval()
    for p in params:
        p.requires_grad_(False)
    return student


# ------------------------------------------------------------------ instrumented forward with hooks
def student_forward(model: Qwen3LM, ids, T=None, hook=None, collect_site=True, upto_loop=None, from_state=None,
                    from_loop=0):
    """Re-implements Qwen3LM.forward_hidden so a `hook(loop_k, h) -> h` can edit the residual stream inside the
    recurrent site (k = 1-indexed loop that just finished).  Returns (hidden_post_norm, site_states) where
    site_states = list of h after each loop (token dim kept).  Teacher / LaCo-only models (no site) just run.

    `from_state` with `from_loop` resumes the site loop at that state (used by continuation tests).
    Returns (hidden_post_norm, site_states, h_in) with h_in the state entering the site (None if no site)."""
    h = model.embed(ids)
    cos, sin = rope_cos_sin(ids.shape[1], model.cfg.head_dim, model.cfg.rope_theta, ids.device, h.dtype)
    T = T if T is not None else model.default_loops
    site_states = []
    seen_site = False
    h_in = None
    for e in model.entries:
        if isinstance(e, RecurrentSite):
            seen_site = True
            if from_state is not None:
                h = from_state
            h_in = h
            for t in range(from_loop, T):
                e.ctx.t = e.step_index(t)
                f = e.block.delta(h, cos, sin)
                if e.gate is not None:
                    f = f * e.gate[e.ctx.t].to(f.dtype)
                h = h + f
                if hook is not None:
                    h = hook(t + 1, h)
                if collect_site:
                    site_states.append(h)
                if upto_loop is not None and t + 1 >= upto_loop:
                    break
        else:
            if from_state is not None and not seen_site:
                continue                              # resumed run: skip layers before the site
            h = e(h, cos, sin)
    return model.norm(h), site_states, h_in


def model_summary(model: Qwen3LM) -> dict:
    return {"entries": len(model.entries), "trainable": count_trainable(model),
            "layer_evals": model.n_layer_evals(), "site": model.site() is not None}


# ------------------------------------------------------------------ persistence
import json as _json
import os as _os

from recsae import DATA as _DATA


def mdir(seed: int, task: str = "dfa") -> str:
    p = _os.path.join(_DATA, "mini", f"{task}_s{seed}")
    _os.makedirs(p, exist_ok=True)
    return p


def save_teacher(model: Qwen3LM, seed: int, task: PrefixTask, layers: int, d: int):
    torch.save({"sd": model.state_dict(), "layers": layers, "d": d, "task": task.name, "seed": seed},
               _os.path.join(mdir(seed, task.name), "teacher.pt"))


def load_teacher(seed: int, task_name: str = "dfa"):
    ck = torch.load(_os.path.join(mdir(seed, task_name), "teacher.pt"), map_location="cpu")
    task = PrefixTask(task_name)
    m = new_teacher(task, ck["layers"], ck["d"], seed)
    m.load_state_dict(ck["sd"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval(), task, ck


def adapter_state_dict(model: Qwen3LM) -> dict:
    """All adapter-type tensors (LoRA A/B, loop norms, gates) regardless of requires_grad (train_recovery freezes them
    again once training ends, so a requires_grad filter would save nothing)."""
    import re
    return {n: p.detach().cpu() for n, p in model.named_parameters() if re.search(r"(\.A|\.B|\.loop_weight|\.gate)$", n)}


def save_student(model: Qwen3LM, seed: int, task_name: str, variant: str, merges, spec: dict):
    torch.save({"train_sd": adapter_state_dict(model), "merges": merges, "spec": spec},
               _os.path.join(mdir(seed, task_name), f"{variant}.pt"))


def load_student(seed: int, task_name: str, variant: str):
    teacher, task, tck = load_teacher(seed, task_name)
    ck = torch.load(_os.path.join(mdir(seed, task_name), f"{variant}.pt"), map_location="cpu")
    sp = ck["spec"]
    st = build_laco_student(teacher, ck["merges"], sp["kind"], T=sp.get("T", 4), rank=sp.get("rank", 16),
                            alpha=sp.get("alpha", 32.0), adapter=sp.get("adapter", "loop"), norm=sp.get("norm", "frozen"),
                            init=sp.get("init", "laco"))
    if ck["train_sd"]:
        res = st.load_state_dict(ck["train_sd"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
    for p in st.parameters():
        p.requires_grad_(False)
    return st.eval(), teacher, task, ck


def teacher_forward(model: Qwen3LM, ids, hook_depth: int | None = None, fn=None):
    """Plain forward through all entries; `fn(h) -> h` is applied to the residual stream right after `hook_depth`
    layers have run (i.e. on H[hook_depth]).  Returns post-norm hidden."""
    h = model.embed(ids)
    cos, sin = rope_cos_sin(ids.shape[1], model.cfg.head_dim, model.cfg.rope_theta, ids.device, h.dtype)
    if hook_depth == 0 and fn is not None:
        h = fn(h)
    for i, e in enumerate(model.entries):
        h = e(h, cos, sin)
        if fn is not None and hook_depth == i + 1:
            h = fn(h)
    return model.norm(h)
