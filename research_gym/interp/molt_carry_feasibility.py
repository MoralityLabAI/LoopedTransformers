"""MOLT-16: finite-dose legal-winner feasibility sandwich.

All numerical infeasibility witnesses are conditional on real-arithmetic
bounds and are not outward-rounded certificates. Gold suffix labels are
used only for evaluation, never for selecting the repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump
from research_gym.interp.molt_amortized_repair import (
    Controller, bounded, features, legal_logits, prepare, root_pool, solve, tensors,
)
from research_gym.interp.molt_bpd_source import load_verified
from research_gym.interp.molt_clock_carry import CUT, fit_source, run_cut
from research_gym.interp.molt_clock_repair import donor_world, margins
from research_gym.interp.molt_typed_clock import MOD, bank_digest, make_worlds, trace_of

CHARTS = ("native", "observer", "random")
SWEEPS = 64
TOL = 1e-8


def legal_gap(model, h, winner):
    z = legal_logits(margins(model, h))
    rivals = [j for j in range(MOD) if j != winner]
    return float((z[winner] - z[rivals]).min())


def halfspaces(model, h, basis, radius, winner):
    """Necessary/sufficient linear inequalities over the same support ball.

    Preserve the LayerNorm bias and exact epsilon. The norm envelope
    uses the centered support span and its operator norm.
    """
    if radius < 0 or not math.isfinite(radius) or not 0 <= winner < MOD:
        raise ValueError("invalid physical dose or legal class")
    if basis.ndim != 2 or basis.shape[0] != len(h):
        raise ValueError("invalid support")
    if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=h.dtype),
                          atol=TOL, rtol=0):
        raise ValueError("support must be orthonormal")
    norm = model.decoder.final_norm
    weight = model.decoder.embedding.weight
    rivals = [j for j in range(MOD) if j != winner]
    contrast = weight[winner] - weight[rivals]
    a = contrast * norm.weight[None]
    b = contrast @ norm.bias
    z = h - h.mean()
    q = basis - basis.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(q, full_matrices=False)
    # Keep the complete thin left basis, even for rank-deficient charts.
    # Its span contains range(CQ); dropping tiny singular directions could
    # invalidate the lower envelope. A superset only loosens the bound.
    parallel = u @ (u.T @ z)
    orthogonal = z - parallel
    # Any centered displacement has norm <= R*||C Q|| and lies in span C Q.
    reach = radius * float(s[0])
    lo_norm = math.hypot(float(orthogonal.norm()), max(0., float(parallel.norm()) - reach))
    hi_norm = math.hypot(float(orthogonal.norm()), float(parallel.norm()) + reach)
    lo = math.sqrt(lo_norm**2 / len(h) + norm.eps)
    hi = math.sqrt(hi_norm**2 / len(h) + norm.eps)
    # Slack helps avoid false comparisons near floating-point thresholds.
    # It is not a directed-rounding proof.
    pad = 1e-10 * (1 + hi)
    lo = max(math.sqrt(norm.eps) - pad, lo - pad)
    hi += pad
    mat = a @ q
    offset = a @ z
    # For zero desired winning margin: numerator >= -bias * denominator.
    k = -b
    necessary = torch.where(k >= 0, k * lo, k * hi) - offset
    sufficient = torch.where(k >= 0, k * hi, k * lo) - offset
    return mat, necessary, sufficient, (lo, hi)


def projection(mat, rhs, radius, sweeps=SWEEPS):
    """Fixed dual coordinate iterations for min ||c||²/2, A c >= b.

    A nonnegative dual vector also gives a separating witness over the
    radius-R ball: lambda^T b > R ||A^T lambda||.
    Incomplete convergence never establishes infeasibility.
    """
    a = mat.detach().cpu().numpy()
    b = rhs.detach().cpu().numpy()
    if a.ndim != 2 or b.shape != (len(a),) or not 1 <= sweeps <= 256:
        raise ValueError("invalid halfspace system")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or radius < 0 or not math.isfinite(radius):
        raise ValueError("nonfinite halfspace input")
    c = np.zeros(a.shape[1], dtype=np.float64)
    lam = np.zeros(len(a), dtype=np.float64)
    sq = np.einsum("ij,ij->i", a, a)
    for _ in range(sweeps):
        for i in range(len(a)):
            if sq[i] <= 1e-24:
                # Retain a valid dual vector even for a zero constraint row.
                if b[i] > 0:
                    lam[i] = max(lam[i], 1.)
                continue
            new = max(0., lam[i] + (b[i] - a[i] @ c) / sq[i])
            c += (new - lam[i]) * a[i]
            lam[i] = new
    if not np.isfinite(c).all() or not np.isfinite(lam).all():
        raise RuntimeError("nonfinite dual solver")
    # Normalize for a scale-independent numerical separation receipt.
    if lam.sum() > 0:
        lam /= lam.sum()
    dual = a.T @ lam
    lhs = float(lam @ b)
    support = float(np.linalg.norm(dual))
    separation = lhs - radius * support
    scale = 1 + abs(lhs) + radius * support
    obstruction = separation > TOL * scale
    norm_lower = max(0., lhs / support) if support > 1e-14 else None
    unconstrained_norm = float(np.linalg.norm(c))
    clipped = bounded(torch.from_numpy(c), torch.tensor([radius], dtype=mat.dtype))
    return clipped, {
        "sweeps": sweeps, "dual_lambda": lam.tolist(),
        "dual_lhs": lhs, "dual_support_norm": support,
        "separation": separation, "obstruction_witness": obstruction,
        "required_dose_lower_bound": norm_lower,
        "unconstrained_candidate_norm": unconstrained_norm,
        "clipped_linear_min_slack": float((mat @ clipped - rhs).min()),
    }


def fresh_roots(depth, count):
    old_train = make_worlds(121001, 4, 512)
    old = make_worlds(121100 + depth, depth, 128, old_train)
    prior = make_worlds(131000 + depth, depth, 16, old_train + old)
    recent = make_worlds(141000 + depth, depth, 32, old_train + old + prior)
    train15, dev15 = root_pool(depth, 64, 32)
    forbidden = {w for w in old_train + old + prior + recent + train15 + dev15 if len(w) == 2 * depth + 1}
    forbidden.update(donor_world(w, k) for w in list(forbidden) for k in ("changed", "same"))
    roots = []
    for w in make_worlds(161000 + depth, depth, count * 4, list(forbidden)):
        group = {w, donor_world(w, "changed"), donor_world(w, "same")}
        if len(group) != 3 or group & forbidden:
            continue
        roots.append(w)
        forbidden.update(group)
        if len(roots) == count:
            return roots
    raise RuntimeError("fresh roster shortfall")


def load_controllers(args, model, native, checkpoint, factors):
    report = json.loads((args.controllers / "receipt.json").read_text())
    path = args.controllers / "controllers.pt"
    if (report["schema"] != "molt15-amortized-repair-v1"
            or report["checkpoint_sha256"] != checkpoint or report["factor_sha256"] != factors
            or report["controller_sha256"] != digest_file(path)
            or report["train_roots"] != 128 or report["dev_roots"] != 32):
        raise ValueError("controller receipt binding mismatch")
    data = torch.load(path, weights_only=True, map_location="cpu")
    bases = data["bases"]
    if set(bases) != set(CHARTS) or not torch.allclose(bases["native"], native, atol=TOL, rtol=0):
        raise ValueError("controller chart mismatch")
    for q in bases.values():
        if q.shape != native.shape or not torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=q.dtype),
                                                         atol=TOL, rtol=0):
            raise ValueError("invalid controller support")
    mean, scale = data["mean"], data["scale"]
    if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("invalid TRAIN normalization")
    controllers = {}
    for chart in CHARTS:
        net = Controller(len(mean), bases[chart].shape[1]).double()
        net.load_state_dict(data["controllers"][chart + "_ce"])
        controllers[chart] = net.eval()
    return bases, controllers, mean, scale


@torch.no_grad()
def evaluate(model, records, bases, controllers, mean, scale):
    h, raw, target, radius = tensors(records)
    x = (features(model, h, raw, target, radius) - mean) / scale
    learned = {name: net(x, raw, radius, bases[name]) for name, net in controllers.items()}
    rows = []
    for i, rec in enumerate(records):
        winner = int(legal_logits(target[i]).argmax())
        r = float(radius[i])
        tokens = torch.tensor([rec["world"]])
        anchor = torch.tensor(trace_of(rec["world"])[CUT:])
        donor = torch.tensor(trace_of(rec["donor"])[CUT:])
        for name in CHARTS:
            q = bases[name]
            start = time.perf_counter()
            a, necessary, sufficient, envelope = halfspaces(model, h[i], q, r, winner)
            cn, nr = projection(a, necessary, r)
            cs, sr = projection(a, sufficient, r)
            projected = bounded(raw[i] @ q, radius[i])
            warm, work = solve(model, h[i], target[i], q, radius[i], projected, "ce")
            candidates = {"projected_swap": q @ projected, "learned": learned[name][i],
                          "warm_ce": warm, "necessary": q @ cn, "sufficient": q @ cs}
            gaps = {k: legal_gap(model, h[i] + d, winner) for k, d in candidates.items()}
            selected = max(candidates, key=lambda k: (gaps[k], -float(candidates[k].norm()), k))
            witnessed = gaps[selected] > TOL
            obstruction = nr["obstruction_witness"]
            if witnessed and obstruction:
                raise RuntimeError("inconsistent numerical feasibility sandwich")
            # A successful strict legal winner is an explicit primal witness.
            # Negative/near-zero gap alone never proves impossibility.
            status = "witnessed" if witnessed else "conditional_obstruction" if obstruction else "unresolved"
            candidate_seconds = time.perf_counter() - start
            candidates["best_cut_gap"] = candidates[selected]
            outcomes = {}
            for arm, d in candidates.items():
                if float(d.norm()) > r + TOL:
                    raise RuntimeError("physical dose cap violated")
                logits, _, _, _ = run_cut(model, tokens, CUT, (d / model.decoder.beta)[None])
                m = logits[0, CUT - 1, :MOD]
                local = float((m[winner] - m[[j for j in range(MOD) if j != winner]]).min())
                if abs(local - legal_gap(model, h[i] + d, winner)) > TOL:
                    raise RuntimeError("actual cut gap parity failed")
                future = logits[0, CUT:].argmax(-1)
                outcomes[arm] = {"gap": local, "dose": float(d.norm()),
                                 "cut_winner": bool(m.argmax() == winner),
                                 "donor_suffix_correct": bool((future == donor).all()),
                                 "anchor_suffix_preserved": bool((future == anchor).all())}
            rows.append({"root": bank_digest([rec["world"]]), "kind": rec["kind"],
                         "depth": len(rec["world"]) // 2, "chart": name,
                         "anchor_correct": rec["anchor_ok"], "donor_correct": rec["donor_ok"],
                         "radius": r, "teacher_winner": winner, "denominator_bounds": envelope,
                         "necessary": nr, "sufficient": sr, "status": status,
                         "chosen": selected, "outcomes": outcomes, "warm_solver": work,
                         "candidate_seconds": candidate_seconds})
    return rows


def summarize(rows):
    summary = []
    for chart in CHARTS:
        for kind in ("changed", "same"):
            group = [r for r in rows if r["chart"] == chart and r["kind"] == kind]
            summary.append({
                "chart": chart, "kind": kind, "rows": len(group),
                "witnessed": sum(r["status"] == "witnessed" for r in group),
                "conditional_obstruction": sum(r["status"] == "conditional_obstruction" for r in group),
                "unresolved": sum(r["status"] == "unresolved" for r in group),
                "suffix_correct": {a: sum(r["outcomes"][a]["donor_suffix_correct"] for r in group)
                                   for a in group[0]["outcomes"]},
                "cut_winner": {a: sum(r["outcomes"][a]["cut_winner"] for r in group)
                               for a in group[0]["outcomes"]},
            })
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "factors", "bpd-checkout", "bpd-source-receipt", "controllers", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--roots", type=int, default=16)
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.roots <= 32:
        p.error("fresh output and bounded root count required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, v, u, native, ck, fh = fit_source(args)
    load_verified(args.bpd_checkout, args.bpd_source_receipt)
    bases, nets, mean, scale = load_controllers(args, model, native, ck, fh)
    worlds = [w for d in (4, 8) for w in fresh_roots(d, args.roots)]
    start = time.perf_counter()
    rows = evaluate(model, prepare(model, worlds, v, u), bases, nets, mean, scale)
    report = {
        "schema": "molt16-feasibility-v1", "status": "development_only",
        "checkpoint_sha256": ck, "factor_sha256": fh,
        "controller_sha256": digest_file(args.controllers / "controllers.pt"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "root_roster_sha256": bank_digest(worlds), "world_roots": len(worlds),
        "paired_edits": 2 * len(worlds), "rows": rows, "summary": summarize(rows),
        "sweeps": SWEEPS, "wall_seconds": time.perf_counter() - start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "conditional_real_arithmetic_only": True, "directed_rounding_verified": False,
        "formal_certificate_authorized": False, "training_performed": False,
        "confirmation_ready": False, "promotion_authorized": False,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "receipt.json", report)
    print(json.dumps({k: report[k] for k in ("schema", "world_roots", "paired_edits", "summary",
                                             "wall_seconds", "peak_rss_bytes")}, sort_keys=True))


if __name__ == "__main__":
    main()
