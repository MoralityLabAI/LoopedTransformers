"""MOLT-13: exact clock-carry suffix factorization and observer rank audit.

Uses a competent MOLT-12 checkpoint and native BPD write coordinates when
available. The cut is after one operation's readout and legal-state softmax.
The suffix observer excludes the cut visit by definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import torch

from research_gym.interp.aux_gym_v2 import digest_file, dump
from research_gym.interp.aux_gym_v2_native_bpd import weight_digest
from research_gym.interp.molt_bpd_source import BPD_REVISION, load_verified
from research_gym.interp.molt_typed_clock import (
    MOD, SITE, TypedClockDecoder, bank_digest, competent, make_worlds, trace_of,
)
from research_gym.interp.molt_typed_clock_bpd import ARM, SEED
from lsa.observability import spectrum

CUT = 4
TOL = 1e-8


def center(logits):
    return logits - logits.mean(-1, keepdim=True)


def suffix(model, carry, operations):
    """Continue with exactly the public future operands and predicted carry."""
    if carry.ndim != 2 or carry.shape[1] != MOD or operations.ndim != 2 or len(operations) != len(carry):
        raise ValueError("typed suffix register mismatch")
    logits, states = [], []
    for j in range(operations.shape[1]):
        q, carry = model.stage(carry, operations[:, j])
        logits.append(q)
        states.append(carry)
    if not logits:
        raise ValueError("empty future observer")
    return torch.stack(logits, dim=1), torch.stack(states, dim=1)


def encode_carry(p):
    if p.shape[-1] != MOD or not bool(torch.isfinite(p).all()):
        raise ValueError("invalid carry")
    if float((p.sum(-1) - 1).abs().max()) > 1e-10 or bool((p < -1e-10).any()):
        raise ValueError("carry outside simplex")
    return p[..., :-1].clone()


def decode_carry(c):
    if c.shape[-1] != MOD - 1 or not bool(torch.isfinite(c).all()):
        raise ValueError("invalid carry chart")
    p = torch.cat((c, 1 - c.sum(-1, keepdim=True)), dim=-1)
    if bool((p < -1e-10).any()):
        raise ValueError("decoded carry outside simplex")
    return p


def run_cut(model, tokens, visit, displacement=None):
    """Native unscaled physical write hook; capture the actual post-block state."""
    count, block_count = 0, 0
    captured, recipient = [], []
    def hook(_module, inputs, output):
        nonlocal count
        count += 1
        if count != visit:
            return None
        captured.append((inputs[0].detach().clone(), output.detach().clone()))
        if displacement is None:
            return None
        return torch.cat((output[:, :-1], output[:, -1:] + displacement[:, None]), dim=1)

    def block_hook(_module, _inputs, output):
        nonlocal block_count
        block_count += 1
        if block_count == visit:
            recipient.append(output[:, -1].detach().clone())

    a = model.get_submodule(SITE).register_forward_hook(hook)
    b = model.decoder.block_at(0).register_forward_hook(block_hook)
    try:
        logits, carries = model.rollout(tokens)
    finally:
        a.remove()
        b.remove()
    if count != tokens.shape[1] - 1 or block_count != count or len(captured) != 1 or len(recipient) != 1:
        raise RuntimeError("physical visit accounting mismatch")
    return logits, carries, captured[0], recipient[0]


def carry_from_write(model, recipient, delta):
    # Residual beta is part of the physical intervention's downstream map.
    logits = model.decoder.lens(recipient + model.decoder.beta * delta)
    return logits[..., :MOD].softmax(-1)


def chart(dtype):
    # Orthonormal basis of the zero-sum tangent space of the simplex.
    e = torch.eye(MOD, dtype=dtype)[:, :-1]
    e[-1] = -1
    return torch.linalg.qr(e, mode="reduced").Q


def geometry(model, recipient, operations, native_basis):
    """Independent full forward-mode Jacobians, then the typed chain identity."""
    p = carry_from_write(model, recipient, torch.zeros_like(recipient))
    e = chart(p.dtype)
    zero = torch.zeros_like(recipient)
    zero_chart = p.new_zeros(MOD - 1)

    def observer(carry):
        return center(suffix(model, carry[None], operations)[0][0]).flatten()

    def cut(d):
        return carry_from_write(model, recipient, d)

    def future(d):
        return observer(cut(d))

    def tangent(c):
        return observer(p + e @ c)

    def immediate(d):
        return center(model.decoder.lens(recipient + model.decoder.beta * d))

    jc = torch.func.jacfwd(cut)(zero)
    jg = torch.func.jacfwd(tangent)(zero_chart)
    direct = torch.func.jacfwd(future)(zero)
    factored = jg @ (e.T @ jc)
    first = torch.func.jacfwd(immediate)(zero)
    error = float((direct - factored).abs().max())
    tangency = float(jc.sum(0).abs().max())
    if error > TOL or tangency > TOL:
        raise RuntimeError("carry-cut derivative factorization failed")
    return {
        "future": spectrum(direct, relative=0, absolute=1e-8),
        "future_native_span": spectrum(direct @ native_basis, relative=0, absolute=1e-8),
        "cut_included": spectrum(torch.cat((first, direct)), relative=0, absolute=1e-8),
        "carry_derivative": spectrum(jc, relative=0, absolute=1e-8),
        "factorization_max_abs": error, "carry_tangency_max_abs": tangency,
    }


@torch.no_grad()
def replay_case(model, tokens, donor, v, u, basis, visit):
    clean, states, (reads, _), h = run_cut(model, tokens, visit)
    donor_logits, _, (donor_reads, _), _ = run_cut(model, donor, visit)
    d = ((donor_reads[:, -1] - reads[:, -1]) @ v) @ u
    patched, carry, _, _ = run_cut(model, tokens, visit, d)
    actual = carry[:, visit - 1]
    local = carry_from_write(model, h, d)
    reconstructed = decode_carry(encode_carry(actual))
    future_ops = tokens[:, visit + 1:]
    direct = patched[:, visit:]
    replay, replay_states = suffix(model, reconstructed, future_ops)
    local_error = float((actual - local).abs().max())
    replay_error = float((direct - replay).abs().max())
    replay_carry_error = float((carry[:, visit:] - replay_states).abs().max())
    if max(local_error, replay_error, replay_carry_error) > TOL:
        raise RuntimeError("native carry replay parity failed")
    if not torch.equal(direct.argmax(-1), replay.argmax(-1)):
        raise RuntimeError("suffix decision parity failed")
    gold = torch.tensor(trace_of(tuple(donor[0].tolist()))[visit:])
    anchor_gold = torch.tensor(trace_of(tuple(tokens[0].tolist()))[visit:])
    # Source-valid single public operand edit, identical suffix.
    return {
        "native_write_norm": float(d.norm()),
        "anchor_suffix_correct": bool((clean[0, visit:].argmax(-1) == anchor_gold).all()),
        "donor_suffix_correct": bool((donor_logits[0, visit:].argmax(-1) == gold).all()),
        "native_patch_suffix_correct": bool((direct[0].argmax(-1) == gold).all()),
        "native_patch_terminal_correct": bool(direct[0, -1].argmax() == gold[-1]),
        "native_patch_clean_suffix_preserved": bool((direct[0].argmax(-1) == anchor_gold).all()),
        "local_carry_error": local_error, "suffix_replay_max_abs": replay_error,
        "suffix_carry_replay_max_abs": replay_carry_error,
        "suffix_carry_bytes": (MOD - 1) * actual.element_size(),
        "full_normalized_suffix_bytes": direct.shape[1] * h.shape[-1] * actual.element_size(),
        "dense_suffix_logit_bytes": direct.numel() * direct.element_size(),
        "geometry": geometry(model, h[0], future_ops, basis),
    }


def fit_source(args):
    report = json.loads((args.clock / "receipt.json").read_text())
    selected = [r for r in report["rows"] if r["arm"] == ARM and r["seed"] == SEED]
    if len(selected) != 1 or not competent(selected[0]):
        raise ValueError("predeclared competence gate failed")
    ck = args.clock / f"{ARM}-{SEED}.pt"
    factor = json.loads((args.factors / "receipt.json").read_text())
    fh = digest_file(args.factors / "factors.pt")
    if (factor["status"] != "engineering_fit_completed" or not factor["competence_gate_passed"]
            or factor["bpd_revision"] != BPD_REVISION or factor["site"] != SITE
            or factor["checkpoint_sha256"] != digest_file(ck)
            or selected[0]["checkpoint_sha256"] != digest_file(ck)
            or factor["factor_sha256"] != fh
            or factor["bpd_source_receipt_sha256"] != digest_file(args.bpd_source_receipt)):
        raise ValueError("native factor/checkpoint binding mismatch")
    model = TypedClockDecoder(ARM)
    model.load_state_dict(torch.load(ck, weights_only=True, map_location="cpu"))
    if weight_digest(model.get_submodule(SITE).weight) != factor["teacher_weight_sha256"]:
        raise ValueError("native physical weight mismatch")
    f = torch.load(args.factors / "factors.pt", weights_only=True, map_location="cpu")
    v, u = f["V"].double(), f["U"].double()
    if not all(bool(torch.isfinite(t).all()) for t in (v, u)):
        raise ValueError("nonfinite factors")
    q, s, _ = torch.linalg.svd(u.T, full_matrices=False)
    rank = int((s > s[0] * 1e-8).sum())
    if rank < 1:
        raise ValueError("empty native span")
    return model.double().eval(), v, u, q[:, :rank].contiguous(), digest_file(ck), fh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clock", "factors", "bpd-checkout", "bpd-source-receipt", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--roots", type=int, default=8)
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.roots <= 16:
        p.error("fresh output and bounded root count required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, v, u, basis, ck, fh = fit_source(args)
    # Verify the same immutable native package before consuming its factors.
    load_verified(args.bpd_checkout, args.bpd_source_receipt)
    train = make_worlds(121001, 4, 512)
    results = []
    for depth in (4, 8):
        old = make_worlds(121100 + depth, depth, 128, train)
        worlds = make_worlds(131000 + depth, depth, args.roots, train + old)
        for i, w in enumerate(worlds):
            donor = list(w)
            donor[CUT] = 16 + (donor[CUT] - 16 + 1) % MOD
            row = replay_case(model, torch.tensor([w]), torch.tensor([donor]), v, u, basis, CUT)
            results.append({"depth": depth, "root": i, "world_digest": bank_digest([w]),
                            "donor_digest": bank_digest([donor]), **row})
    report = {"schema": "molt13-clock-carry-v1", "status": "development_only",
              "checkpoint_sha256": ck, "factor_sha256": fh, "site": SITE, "cut": CUT,
              "native_span_dimension": basis.shape[1], "carry_chart_dimension": MOD - 1,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "roots": len(results), "rows": results,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
              "confirmation_ready": False, "promotion_authorized": False}
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "receipt.json", report)
    print(json.dumps({"schema": report["schema"], "roots": len(results),
                      "native_span_dimension": report["native_span_dimension"],
                      "anchor_correct": sum(r["anchor_suffix_correct"] for r in results),
                      "donor_correct": sum(r["donor_suffix_correct"] for r in results),
                      "patch_correct": sum(r["native_patch_suffix_correct"] for r in results),
                      "patch_terminal_correct": sum(r["native_patch_terminal_correct"] for r in results),
                      "rank_tuples": sorted({(r["geometry"]["future"]["d_obs"],
                                               r["geometry"]["future_native_span"]["d_obs"],
                                               r["geometry"]["cut_included"]["d_obs"]) for r in results}),
                      "max_factorization_error": max(r["geometry"]["factorization_max_abs"] for r in results),
                      "max_replay_error": max(r["suffix_replay_max_abs"] for r in results),
                      "peak_rss_bytes": report["peak_rss_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
