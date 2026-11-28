"""MOLT-15: decision-aligned, amortized repair on the competent clock.

The teacher and fitted BPD factors are frozen. The controller receives
privileged donor cut margins. It does not learn autonomous task reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2 import digest_file, dump
from research_gym.interp.molt_bpd_source import load_verified
from research_gym.interp.molt_clock_carry import CUT, fit_source, run_cut
from research_gym.interp.molt_clock_repair import (
    FRACTIONS, STEPS, TOL, donor_world, margin_jacobian, margins,
)
from research_gym.interp.molt_typed_clock import MOD, bank_digest, make_worlds, trace_of

LEARNED = ("native_ce", "native_l2", "random_ce", "observer_ce")
CONTROLS = ("intact", "native_swap", "native_warm_l2", "native_zero_ce", "native_warm_ce",
            "random_warm_ce", "observer_warm_ce")
TRAIN_SEED = 151100
DEV_SEED = 151200
EVAL_DEPTHS = (4, 8)
CONTROLLER_WIDTH = 64


def bounded(x, radius):
    """Euclidean projection onto a ball, along the final dimension."""
    scale = (radius / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return x * scale


def legal_logits(m):
    return torch.cat((m, torch.zeros_like(m[..., :1])), dim=-1)


def decision_loss(m, winner):
    return F.cross_entropy(legal_logits(m), winner, reduction="none")


def objective(m, target, winner, kind):
    if kind == "ce":
        return decision_loss(m, winner)
    if kind == "l2":
        return (m - target).square().mean(dim=-1)
    raise ValueError("unknown objective")


@torch.no_grad()
def solve(model, h, target, basis, radius, initial, kind):
    """Fixed eight-step, four-proposal GN or damped Fisher-scoring control."""
    radius = radius.reshape(1)
    c = bounded(initial.clone(), radius)
    winner = legal_logits(target).argmax(-1)

    def score(z):
        return float(objective(margins(model, h + basis @ z), target, winner, kind))

    initial_loss = score(c)
    current = initial_loss
    for _ in range(STEPS):
        reached = h + basis @ c
        m = margins(model, reached)
        j = margin_jacobian(model, reached) @ basis
        if kind == "ce":
            p = legal_logits(m).softmax(-1)[:-1]
            t = F.one_hot(winner, MOD).to(h.dtype)[:-1]
            fisher = torch.diag(p) - torch.outer(p, p)
            g = j.T @ (p - t)
            a = j.T @ fisher @ j
        else:
            a = j.T @ j
            g = j.T @ (m - target)
        damping = 1e-4 * max(1., float(a.trace()))
        step = torch.linalg.solve(a + damping * torch.eye(len(c), dtype=h.dtype), -g)
        choices = [bounded(c + alpha * step, radius) for alpha in FRACTIONS]
        values = [score(x) for x in choices]
        i = min(range(len(values)), key=lambda k: (values[k], k))
        if values[i] < current:
            c, current = choices[i], values[i]
    d = basis @ c
    if float(d.norm()) > float(radius) + TOL or not bool(torch.isfinite(d).all()):
        raise RuntimeError("invalid analytical solver output")
    return d, {"initial_objective": initial_loss, "final_objective": current,
               "jacobian_calls": STEPS, "trial_readouts": STEPS * len(FRACTIONS),
               "initial_readouts": 1}


def root_pool(depth, train_count, dev_count):
    """Exclude previous root rosters, including MOLT-14's full cap.

    Exclude the training roots and their two donor variants from development
    roots and donors. The exact tuple is the independent source identity.
    """
    prior_train = make_worlds(121001, 4, 512)
    old = make_worlds(121100 + depth, depth, 128, prior_train)
    prior = make_worlds(131000 + depth, depth, 16, prior_train + old)
    recent = make_worlds(141000 + depth, depth, 32, prior_train + old + prior)
    excluded = set(prior_train + old + prior + recent)
    excluded.update(donor_world(w, k) for w in list(excluded)
                    if len(w) == 2 * depth + 1 for k in ("changed", "same"))

    def draw(seed, count, forbidden):
        chosen = []
        blocked = set(forbidden)
        # Bounded deterministic oversampling, never select on model outcomes.
        candidates = make_worlds(seed + depth, depth, count * 4, list(blocked))
        for w in candidates:
            group = {w, donor_world(w, "changed"), donor_world(w, "same")}
            if len(group) != 3 or group & blocked:
                continue
            chosen.append(w)
            blocked.update(group)
            if len(chosen) == count:
                return chosen, blocked
        raise RuntimeError("insufficient disjoint roots")
    train, blocked = draw(TRAIN_SEED, train_count, excluded)
    dev, _ = draw(DEV_SEED, dev_count, blocked)
    return train, dev


@torch.no_grad()
def prepare(model, worlds, v, u):
    records = []
    for world in worlds:
        tokens = torch.tensor([world])
        clean, _, (a, _), h = run_cut(model, tokens, CUT)
        for kind in ("changed", "same"):
            donor = donor_world(world, kind)
            donor_logits, _, (b, _), donor_h = run_cut(model, torch.tensor([donor]), CUT)
            d = (((b[:, -1] - a[:, -1]) @ v) @ u)[0] * model.decoder.beta
            target = margins(model, donor_h[0])
            donor_gold = torch.tensor(trace_of(donor)[CUT:])
            anchor_gold = torch.tensor(trace_of(world)[CUT:])
            anchor_ok = bool((clean[0, CUT:].argmax(-1) == anchor_gold).all())
            donor_ok = bool((donor_logits[0, CUT:].argmax(-1) == donor_gold).all())
            records.append({"world": world, "donor": donor, "kind": kind, "h": h[0].detach(),
                            "raw": d.detach(), "target": target.detach(), "radius": d.norm().reshape(1),
                            "anchor_ok": anchor_ok, "donor_ok": donor_ok})
    return records


def tensors(records):
    h = torch.stack([r["h"] for r in records])
    raw = torch.stack([r["raw"] for r in records])
    target = torch.stack([r["target"] for r in records])
    radius = torch.stack([r["radius"] for r in records])
    return h, raw, target, radius


def features(model, h, raw, target, radius):
    # Observed donor margins are inputs to all controllers. No gold values,
    # future operands, labels, or model predictions from a patched suffix.
    own = margins(model, h)
    return torch.cat((h, raw, (target - own), radius), dim=-1)


class Controller(nn.Module):
    def __init__(self, feature_count, rank):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_count, CONTROLLER_WIDTH), nn.Tanh(),
                                 nn.Linear(CONTROLLER_WIDTH, rank))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, raw, radius, basis):
        base = bounded(raw @ basis, radius)
        # Start at the projected raw swap, then learn a bounded correction.
        c = bounded(base + radius * self.net(x).tanh(), radius)
        return c @ basis.T


@torch.no_grad()
def charts(model, train, native, seed):
    """Fix all charts before development; observer map is TRAIN-only."""
    gen = torch.Generator().manual_seed(seed)
    random = torch.linalg.qr(torch.randn(native.shape, dtype=native.dtype, generator=gen),
                             mode="reduced").Q
    gram = sum((j.T @ j for j in (margin_jacobian(model, r["h"]) for r in train)),
               torch.zeros(native.shape[0], native.shape[0], dtype=native.dtype))
    _, eig = torch.linalg.eigh(gram)
    obs = eig[:, -native.shape[1]:].contiguous()
    return {"native": native, "random": random, "observer": obs}


def fit(model, train, bases, updates, batch, seed):
    model.requires_grad_(False)
    h, raw, target, radius = tensors(train)
    x = features(model, h, raw, target, radius).detach()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / scale
    y = legal_logits(target).argmax(-1)
    schedule = torch.randint(len(train), (updates, batch),
                             generator=torch.Generator().manual_seed(seed + 1000))
    controllers, receipts = {}, []
    for arm in LEARNED:
        chart, kind = arm.split("_")
        basis = bases[chart]
        torch.manual_seed(seed)
        net = Controller(x.shape[1], basis.shape[1]).double()
        opt = torch.optim.AdamW(net.parameters(), lr=0.003)
        start = time.perf_counter()
        history = []
        for idx in schedule:
            d = net(x[idx], raw[idx], radius[idx], basis)
            m = margins(model, h[idx] + d)
            loss = objective(m, target[idx], y[idx], kind).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite controller loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            history.append(float(loss.detach()))
        controllers[arm] = net.eval()
        receipts.append({"arm": arm, "updates": updates, "batch": batch, "seed": seed,
                         "parameters": sum(p.numel() for p in net.parameters()),
                         "training_seconds": time.perf_counter() - start,
                         "first_loss": history[0], "last_loss": history[-1]})
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("teacher gradient leaked")
    return controllers, mean, scale, receipts


@torch.no_grad()
def evaluate(model, records, bases, controllers, mean, scale):
    h, raw, target, radius = tensors(records)
    x = (features(model, h, raw, target, radius) - mean) / scale
    learned = {a: net(x, raw, radius, bases[a.split("_")[0]])
               for a, net in controllers.items()}
    rows = []
    for i, r in enumerate(records):
        start = time.perf_counter()
        v = {"intact": torch.zeros_like(h[i]), "native_swap": raw[i]}
        work = {}
        for name in CONTROLS[2:]:
            chart, initialization, kind = name.split("_")
            basis = bases[chart]
            initial = raw[i] @ basis if initialization == "warm" else torch.zeros_like(raw[i] @ basis)
            v[name], work[name] = solve(model, h[i], target[i], basis, radius[i], initial, kind)
        for a, ds in learned.items():
            v[a] = ds[i]
        tokens = torch.tensor([r["world"]])
        anchor_gold = torch.tensor(trace_of(r["world"])[CUT:])
        donor_gold = torch.tensor(trace_of(r["donor"])[CUT:])
        winner = legal_logits(target[i]).argmax(-1)
        for arm, d in v.items():
            if float(d.norm()) > float(radius[i]) + TOL:
                raise RuntimeError("controller exceeded physical dose cap")
            logits, _, _, _ = run_cut(model, tokens, CUT, (d / model.decoder.beta)[None])
            local = logits[0, CUT - 1, :MOD]
            expected = margins(model, h[i] + d)
            if float((local[:-1] - local[-1] - expected).abs().max()) > TOL:
                raise RuntimeError("physical repair parity failed")
            actual = logits[0, CUT:].argmax(-1)
            rows.append({"root": bank_digest([r["world"]]), "kind": r["kind"],
                         "depth": len(r["world"]) // 2, "arm": arm,
                         "anchor_correct": r["anchor_ok"], "donor_correct": r["donor_ok"],
                         "dose": float(d.norm()), "dose_cap": float(radius[i]),
                         "donor_suffix_correct": bool((actual == donor_gold).all()),
                         "donor_terminal_correct": bool(actual[-1] == donor_gold[-1]),
                         "anchor_suffix_preserved": bool((actual == anchor_gold).all()),
                         "cut_teacher_winner_match": bool(local.argmax() == winner),
                         "teacher_ce": float(decision_loss(expected, winner)),
                         "margin_l2": float((expected - target[i]).norm()),
                         "solver": work.get(arm)})
    return rows


def summarize(rows):
    out = []
    for arm in (*CONTROLS, *LEARNED):
        for kind in ("changed", "same"):
            group = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            n = len(group)
            out.append({"arm": arm, "kind": kind, "rows": n,
                        "donor_suffix_correct": sum(r["donor_suffix_correct"] for r in group),
                        "donor_terminal_correct": sum(r["donor_terminal_correct"] for r in group),
                        "anchor_suffix_preserved": sum(r["anchor_suffix_preserved"] for r in group),
                        "teacher_winner_match": sum(r["cut_teacher_winner_match"] for r in group),
                        "mean_teacher_ce": sum(r["teacher_ce"] for r in group) / n,
                        "mean_margin_l2": sum(r["margin_l2"] for r in group) / n,
                        "mean_dose": sum(r["dose"] for r in group) / n})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "factors", "bpd-checkout", "bpd-source-receipt", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--train-roots", type=int, default=64)
    p.add_argument("--dev-roots", type=int, default=16)
    p.add_argument("--updates", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=15715)
    args = p.parse_args()
    if (args.out.exists() or not 8 <= args.train_roots <= 128 or not 1 <= args.dev_roots <= 32
            or not 1 <= args.updates <= 512 or not 1 <= args.batch <= 64):
        p.error("fresh output and bounded development arguments required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, v, u, native, ck, fh = fit_source(args)
    load_verified(args.bpd_checkout, args.bpd_source_receipt)
    before = {k: t.detach().clone() for k, t in model.state_dict().items()}
    train_worlds, dev_worlds = [], []
    for depth in EVAL_DEPTHS:
        a, b = root_pool(depth, args.train_roots, args.dev_roots)
        train_worlds += a
        dev_worlds += b
    train = prepare(model, train_worlds, v, u)
    dev = prepare(model, dev_worlds, v, u)
    bases = charts(model, train, native, args.seed)
    controllers, mean, scale, fits = fit(model, train, bases, args.updates, args.batch, args.seed)
    start = time.perf_counter()
    rows = evaluate(model, dev, bases, controllers, mean, scale)
    elapsed = time.perf_counter() - start
    if any(not torch.equal(t, before[k]) for k, t in model.state_dict().items()):
        raise RuntimeError("frozen teacher mutated")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save({"controllers": {a: net.state_dict() for a, net in controllers.items()},
                "bases": bases, "mean": mean, "scale": scale}, args.out / "controllers.pt")
    report = {"schema": "molt15-amortized-repair-v1", "status": "development_only",
              "checkpoint_sha256": ck, "factor_sha256": fh,
              "controller_sha256": digest_file(args.out / "controllers.pt"),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_roots": len(train_worlds), "dev_roots": len(dev_worlds),
              "train_roster_sha256": bank_digest(train_worlds), "dev_roster_sha256": bank_digest(dev_worlds),
              "train_pairs": len(train), "dev_pairs": len(dev), "rows": rows,
              "fits": fits, "summary": summarize(rows),
              "evaluation_seconds": elapsed,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
              "teacher_frozen": True, "privileged_donor_margins": True,
              "teacher_training_performed": False, "confirmation_ready": False,
              "promotion_authorized": False}
    dump(args.out / "receipt.json", report)
    print(json.dumps({k: report[k] for k in ("schema", "train_roots", "dev_roots", "train_pairs",
        "dev_pairs", "fits", "summary", "evaluation_seconds", "peak_rss_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
