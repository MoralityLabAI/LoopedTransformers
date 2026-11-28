"""MOLT-14: dose-capped finite carry repair on a competent native clock."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

from research_gym.interp.aux_gym_v2 import dump
from research_gym.interp.molt_bpd_source import load_verified
from research_gym.interp.molt_clock_carry import CUT, fit_source, run_cut, suffix
from research_gym.interp.molt_typed_clock import MOD, bank_digest, make_worlds, trace_of

ARMS = ("intact", "native_swap", "native_solve", "random_solve", "observer_solve",
        "finite_carry_sham", "donor_residual_ceiling")
STEPS = 8
FRACTIONS = (1., 0.5, 0.25, 0.125)
TOL = 1e-9


def margins(model, h):
    logits = model.decoder.lens(h)
    return logits[..., :MOD - 1] - logits[..., MOD - 1:MOD]


def margin_jacobian(model, h):
    """Actual legal logit contrasts after final LayerNorm, column convention."""
    norm = model.decoder.final_norm
    w = model.decoder.embedding.weight
    contrasts = w[:MOD - 1] - w[MOD - 1:MOD]
    n = h.numel()
    c = h - h.mean()
    scale = (c.square().mean() + norm.eps).sqrt()
    projection = torch.eye(n, dtype=h.dtype, device=h.device) - 1 / n
    derivative = projection / scale - torch.outer(c, c) / (n * scale**3)
    return (contrasts * norm.weight[None, :]) @ derivative


def constrain(c, radius):
    length = float(c.norm())
    return c * (radius / length) if length > radius else c


def solve(model, h, target, basis, radius):
    """Fixed-budget local LM with projected line search, using no gold labels."""
    if basis.ndim != 2 or basis.shape[0] != len(h) or radius < 0:
        raise ValueError("invalid repair chart")
    if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=h.dtype),
                          atol=TOL, rtol=0):
        raise ValueError("nonorthonormal repair chart")
    c = h.new_zeros(basis.shape[1])

    def loss(x):
        return float((margins(model, h + basis @ x) - target).square().sum())

    current = loss(c)
    history = [current]
    for _ in range(STEPS):
        reached = h + basis @ c
        j = margin_jacobian(model, reached) @ basis
        error = margins(model, reached) - target
        # Identical scale-aware ridge rule and four trial proposals in each span.
        ridge = 1e-4 * max(1., float(j.square().sum()))
        step = torch.linalg.solve(j.T @ j + ridge * torch.eye(len(c), dtype=h.dtype), -j.T @ error)
        candidates = [constrain(c + fraction * step, radius) for fraction in FRACTIONS]
        scores = [loss(candidate) for candidate in candidates]
        i = min(range(len(scores)), key=lambda k: (scores[k], k))
        if scores[i] < current:
            c, current = candidates[i], scores[i]
        history.append(current)
    d = basis @ c
    if float(d.norm()) > radius + TOL or not bool(torch.isfinite(d).all()):
        raise RuntimeError("repair dose violation")
    return d, {"margin_objective_history": history, "jacobian_calls": STEPS,
               "trial_readouts": STEPS * len(FRACTIONS), "initial_readouts": 1,
               "radius": radius}


def finite_carry_sham(model, h, d):
    """Reflect a reached state within its exact legal-margin fiber.

    The constraints preserve endpoint mean, norm, legal margin numerators,
    and dot product with the original residual. They imply the same
    LayerNorm denominator, legal logits up to a common shift, and write dose.
    The support is the full ambient residual, not the native factor span.
    """
    t = h + d
    w = model.decoder.embedding.weight
    a = (w[:MOD - 1] - w[MOD - 1:MOD]) * model.decoder.final_norm.weight[None, :]
    constraints = torch.cat((a, torch.ones_like(h)[None], h[None]), dim=0)
    norm = constraints.norm(dim=1)
    rows = constraints[norm > 1e-14] / norm[norm > 1e-14, None]
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    q = vh[s > 1e-11]
    p = q.T @ (q @ t)
    residual = t - p
    if float(residual.norm()) <= 1e-10 or float(d.norm()) <= 1e-12:
        return d.clone(), {"status": "no_distinct_match", "constraint_rank": len(q)}
    other = p - residual
    sham = other - h
    receipt = {
        "status": "matched", "constraint_rank": len(q),
        "dose_error": abs(float(sham.norm()) - float(d.norm())),
        "endpoint_norm_error": abs(float(other.norm()) - float(t.norm())),
        "mean_error": abs(float(other.mean()) - float(t.mean())),
        "legal_margin_error": float((margins(model, other) - margins(model, t)).abs().max()),
        "direction_distance": float((sham - d).norm()),
    }
    if (max(receipt[k] for k in ("dose_error", "endpoint_norm_error", "mean_error",
                                "legal_margin_error")) > TOL
            or receipt["direction_distance"] <= 1e-10):
        raise RuntimeError("finite carry sham constraint violation")
    return sham, receipt


def donor_world(world, kind):
    if kind not in ("changed", "same"):
        raise ValueError("unknown paired edit")
    donor = list(world)
    # Fourth primitive is addition. Changed edits +1. Same-target edits
    # the second addition and compensates at the fourth after multiplication.
    if kind == "changed":
        donor[CUT] = 16 + (donor[CUT] - 16 + 1) % MOD
    else:
        donor[2] = 16 + (donor[2] - 16 + 1) % MOD
        donor[CUT] = 16 + (donor[CUT] - 16 - (world[3] + 1)) % MOD
    anchor_trace, donor_trace = trace_of(world), trace_of(donor)
    if (tuple(donor[CUT + 1:]) != tuple(world[CUT + 1:])
            or (anchor_trace[CUT - 1] == donor_trace[CUT - 1]) != (kind == "same")):
        raise RuntimeError("invalid source edit")
    return tuple(donor)


@torch.no_grad()
def evaluate_case(model, world, donor, v, u, basis, seed):
    tokens, source = torch.tensor([world]), torch.tensor([donor])
    clean, _, (a, _), h = run_cut(model, tokens, CUT)
    donor_logits, donor_carry, (b, _), donor_h = run_cut(model, source, CUT)
    h = h[0]
    target = margins(model, donor_h[0])
    # The physical displacement at the actual residual includes beta.
    raw = (((b[:, -1] - a[:, -1]) @ v) @ u)[0] * model.decoder.beta
    radius = float(raw.norm())
    generator = torch.Generator().manual_seed(seed)
    random_basis = torch.linalg.qr(torch.randn(basis.shape, dtype=h.dtype, generator=generator),
                                   mode="reduced").Q
    j = margin_jacobian(model, h)
    _, s, vh = torch.linalg.svd(j, full_matrices=False)
    if basis.shape[1] > len(vh):
        raise ValueError("native span larger than declared observer chart")
    observer_basis = vh[:basis.shape[1]].T.contiguous()
    native_singular = torch.linalg.svdvals(j @ basis)
    variants = {"intact": torch.zeros_like(h), "native_swap": raw}
    work = {}
    for name, q in (("native_solve", basis), ("random_solve", random_basis),
                    ("observer_solve", observer_basis)):
        start = time.perf_counter()
        variants[name], work[name] = solve(model, h, target, q, radius)
        work[name]["wall_seconds"] = time.perf_counter() - start
    variants["finite_carry_sham"], sham = finite_carry_sham(model, h, variants["native_solve"])
    variants["donor_residual_ceiling"] = donor_h[0] - h
    gold = torch.tensor(trace_of(donor)[CUT:])
    anchor_gold = torch.tensor(trace_of(world)[CUT:])
    outputs, rows = {}, []
    for arm in ARMS:
        d = variants[arm]
        # Evaluate every intervention through the actual dense forward path.
        logits, carry, _, _ = run_cut(model, tokens, CUT, (d / model.decoder.beta)[None])
        p = carry[:, CUT - 1]
        replay = suffix(model, p, tokens[:, CUT + 1:])[0]
        future = logits[:, CUT:]
        if float((replay - future).abs().max()) > TOL:
            raise RuntimeError("finite suffix replay mismatch")
        if arm != "donor_residual_ceiling" and float(d.norm()) > radius + TOL:
            raise RuntimeError("arm exceeded native residual dose cap")
        local = logits[0, CUT - 1, :MOD]
        expected = margins(model, h + d)
        if float((local[:-1] - local[-1] - expected).abs().max()) > TOL:
            raise RuntimeError("physical recipient margin mismatch")
        outputs[arm] = (p, future, logits[:, CUT - 1])
        rows.append({
            "arm": arm, "residual_dose": float(d.norm()),
            "margin_error_l2": float((expected - target).norm()),
            "carry_error_l2": float((p - donor_carry[:, CUT - 1]).norm()),
            "donor_suffix_correct": bool((future[0].argmax(-1) == gold).all()),
            "donor_terminal_correct": bool(future[0, -1].argmax() == gold[-1]),
            "anchor_suffix_preserved": bool((future[0].argmax(-1) == anchor_gold).all()),
            "optimization": work.get(arm),
        })
    p, out, current = outputs["native_solve"]
    ps, outs, currents = outputs["finite_carry_sham"]
    sham.update({"carry_error": float((p - ps).abs().max()),
                 "future_logit_error": float((out - outs).abs().max()),
                 "current_logit_difference": float((current - currents).norm())})
    if max(sham["carry_error"], sham["future_logit_error"]) > TOL:
        raise RuntimeError("finite carry sham changed suffix")
    ceiling = rows[-1]
    if ceiling["margin_error_l2"] > TOL:
        raise RuntimeError("donor residual ceiling parity failed")
    return {
        "anchor_correct": bool((clean[0, CUT:].argmax(-1) == anchor_gold).all()),
        "donor_correct": bool((donor_logits[0, CUT:].argmax(-1) == gold).all()),
        "native_dose_cap": radius, "native_margin_singular_values": native_singular.tolist(),
        "native_margin_rank_abs_1e_8": int((native_singular > 1e-8).sum()),
        "ambient_margin_rank_abs_1e_8": int((s > 1e-8).sum()),
        "sham": sham, "arms": rows,
    }


def summarize(rows):
    result = []
    for kind in ("changed", "same"):
        subset = [r for r in rows if r["kind"] == kind]
        for arm in ARMS:
            group = [next(a for a in r["arms"] if a["arm"] == arm) for r in subset]
            result.append({
                "kind": kind, "arm": arm, "rows": len(group),
                "donor_suffix_correct": sum(a["donor_suffix_correct"] for a in group),
                "donor_terminal_correct": sum(a["donor_terminal_correct"] for a in group),
                "anchor_suffix_preserved": sum(a["anchor_suffix_preserved"] for a in group),
                "mean_margin_error_l2": sum(a["margin_error_l2"] for a in group) / len(group),
                "mean_carry_error_l2": sum(a["carry_error_l2"] for a in group) / len(group),
                "mean_residual_dose": sum(a["residual_dose"] for a in group) / len(group),
            })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "factors", "bpd-checkout", "bpd-source-receipt", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--roots", type=int, default=16)
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.roots <= 32:
        p.error("fresh output and bounded development roster required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, v, u, basis, ck, fh = fit_source(args)
    load_verified(args.bpd_checkout, args.bpd_source_receipt)
    train = make_worlds(121001, 4, 512)
    rows = []
    for depth in (4, 8):
        old = make_worlds(121100 + depth, depth, 128, train)
        prior = make_worlds(131000 + depth, depth, 16, train + old)
        worlds = make_worlds(141000 + depth, depth, args.roots, train + old + prior)
        for index, world in enumerate(worlds):
            for kind in ("changed", "same"):
                donor = donor_world(world, kind)
                result = evaluate_case(model, world, donor, v, u, basis,
                                       141000 + depth * 100 + index * 2 + int(kind == "same"))
                rows.append({"depth": depth, "root": index, "kind": kind,
                             "world_digest": bank_digest([world]), "donor_digest": bank_digest([donor]), **result})
    report = {"schema": "molt14-carry-repair-v1", "status": "development_only",
              "checkpoint_sha256": ck, "factor_sha256": fh, "cut": CUT,
              "steps": STEPS, "trial_fractions": list(FRACTIONS), "native_span_dimension": basis.shape[1],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "world_roots": 2 * args.roots, "paired_edits": len(rows), "rows": rows,
              "summary": summarize(rows),
              "sham_matches": sum(r["sham"]["status"] == "matched" for r in rows),
              "max_sham_future_error": max(r["sham"]["future_logit_error"] for r in rows),
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
              "native_api_swap_parity_tested": False, "training_performed": False,
              "confirmation_ready": False, "promotion_authorized": False}
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "receipt.json", report)
    print(json.dumps({k: report[k] for k in (
        "schema", "world_roots", "paired_edits", "native_span_dimension", "summary",
        "sham_matches", "max_sham_future_error", "peak_rss_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
