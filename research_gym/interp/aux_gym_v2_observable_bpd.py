"""MOLT-09 native BPD observer reuse and continuation engineering panel.

Uses fitted native BPD factors, a frozen LOOP-T teacher and the actual
down-projection write. Local maps are teacher-only. No held split is read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from lsa.observable_memory import MapBinding, Sketch, tensor_digest
from research_gym.interp.aux_gym_v2 import digest_file, dump, majority, metrics, rows, trace_loss
from research_gym.interp.aux_gym_v2_native_bpd import BPD_REVISION, SITE, import_bpd, weight_digest

ARMS = ("ordinary", "ordinary_reinvest", "native_full", "native_stack", "native_terminal")
METHODS = ("stack", "terminal")
EPSILONS = (1e-3, 5e-4)
RELATIVE_GATE = 0.10


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def center(logits):
    return logits - logits.mean(-1, keepdim=True)


def trajectory(model, output, visit):
    """All answer-position centered vocabulary logits, including cut visit."""
    return torch.stack([center(model.lens(h[:, -1])) for h in output.visit_outputs[visit - 1:]], dim=1)


def run_write(model, tokens, visit, delta=None, capture=False):
    """Hook the unscaled physical MLP output; native residual beta remains in the model."""
    if not 1 <= visit <= model.expanded_visits:
        raise ValueError("visit outside execution word")
    calls, captured = 0, []

    def hook(_module, inputs, output):
        nonlocal calls
        calls += 1
        if calls != visit:
            return None
        if capture:
            captured.append(inputs[0].detach())
        if delta is None:
            return None
        if delta.shape != output[:, -1].shape:
            raise ValueError("write shape mismatch")
        return torch.cat((output[:, :-1], output[:, -1:] + delta[:, None].to(output)), dim=1)

    handle = model.get_submodule(SITE).register_forward_hook(hook)
    try:
        out = model(tokens)
    finally:
        handle.remove()
    if calls != model.expanded_visits or (capture and len(captured) != 1):
        raise RuntimeError("unexpected tied execution word")
    return out, captured[0] if capture else None


def native_span(u):
    q, s, _ = torch.linalg.svd(u.double().T, full_matrices=False)
    if not len(s) or float(s[-1]) <= 1e-8 * float(s[0]):
        raise ValueError("native factor output span is rank deficient")
    return q.contiguous()


@torch.no_grad()
def observer(model, tokens, visit, q):
    """Two-step central difference estimate of the full declared answer observer."""
    estimates = []
    for eps in EPSILONS:
        directions = torch.cat((q.T, -q.T), dim=0) * eps
        x = tokens.expand(len(directions), -1)
        output, _ = run_write(model, x, visit, directions)
        f = trajectory(model, output, visit)
        k = q.shape[1]
        estimates.append(((f[:k] - f[k:]) / (2 * eps)).permute(1, 2, 0).contiguous())
    first, second = estimates
    error = float((first - second).norm() / second.norm().clamp_min(1e-12))
    if not math.isfinite(error) or error > 0.005:
        raise RuntimeError(f"finite-difference resolution gate failed: {error}")
    return second, error


def gate(full, projected, baseline):
    """Finite teacher replay gate, deliberately not a uniform error certificate."""
    absolute = float((projected - full).norm())
    relative = absolute / max(float((full - baseline).norm()), 1e-12)
    parity = bool((projected.argmax(-1) == full.argmax(-1)).all())
    return {"absolute_error": absolute, "relative_error": relative,
            "argmax_parity": parity, "accepted": parity and relative <= RELATIVE_GATE}


def read_checked(data, manifest, family, split):
    if split not in {"train", "validation"}:
        raise ValueError("sealed split unavailable")
    name = f"{split}/{family}.jsonl"
    p = data / name
    if digest_file(p) != manifest["files"][name]["sha256"]:
        raise ValueError("dataset hash mismatch")
    return rows(p)


def load(args):
    from research_gym.interp.models import build_model

    teacher = json.loads((args.run / "receipt.json").read_text())
    factor = json.loads((args.factors / "receipt.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    ck = digest_file(args.run / "model.pt")
    fh = digest_file(args.factors / "factors.pt")
    mh = digest_file(args.data / "manifest.json")
    if (teacher["arm"] != "LOOP-T" or teacher["visits"] != 8
            or teacher["checkpoint_sha256"] != ck or teacher["manifest_sha256"] != mh
            or factor["teacher_checkpoint_sha256"] != ck or factor["manifest_sha256"] != mh
            or factor["factor_sha256"] != fh or factor["bpd_revision"] != BPD_REVISION
            or factor["site"] != SITE or factor["family"] != teacher["family"]):
        raise ValueError("teacher/factor/dataset binding mismatch")
    model = build_model("LOOP-T", visits=8)
    model.load_state_dict(torch.load(args.run / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    if weight_digest(model.get_submodule(SITE).weight) != factor["teacher_weight_sha256"]:
        raise ValueError("physical weight hash mismatch")
    factors = torch.load(args.factors / "factors.pt", weights_only=True, map_location="cpu")
    v, u = factors["V"], factors["U"]
    if not 1 <= args.rank < u.shape[0] <= 16:
        raise ValueError("engineering panel requires compressed rank below <=16 native components")
    if v.shape[1] != u.shape[0] or not all(bool(torch.isfinite(t).all()) for t in (v, u)):
        raise ValueError("invalid native factors")
    return model, v, u, ck, fh, read_checked(args.data, manifest, teacher["family"], "train"), \
        read_checked(args.data, manifest, teacher["family"], "validation"), factor


@torch.no_grad()
def build_bank(args, model, v, u, checkpoint, factor_hash, train, bpd):
    # All native components are used. Q is a common orthonormal coordinate
    # chart of their output span, not an additional learned decomposition.
    q = native_span(u)
    teacher64 = copy.deepcopy(model).double().eval()
    wrapped = copy.deepcopy(model)
    bmodel = bpd.BPDModel(wrapped, {SITE: len(u)}, max_passes=8,
                         ablation=bpd.AblationConfig(sphere_rescale=False))
    wrapper = bmodel.wrappers[SITE]
    wrapper.V.copy_(v)
    wrapper.U.copy_(u)
    qhash = tensor_digest(q)
    source = Path(__file__).read_bytes()
    observer_id = identity(["all_answer_centered_logits", args.visit, 8, EPSILONS,
                            hashlib.sha256(source).hexdigest(), factor_hash])
    ordered = sorted(train, key=lambda r: r["example_id"])
    depth = sorted({r["difficulty"] for r in ordered})[-1]
    pool = [r for r in ordered if r["difficulty"] == depth]
    if len(pool) < args.anchors + args.donors:
        raise ValueError("insufficient distinct TRAIN rows at declared depth")
    anchors = pool[:args.anchors]
    donors = pool[args.anchors:args.anchors + args.donors]
    donor_tokens = torch.tensor([r["tokens"] for r in donors])
    _, donor_inputs = run_write(model, donor_tokens, args.visit, capture=True)
    receipt_rows, bank, maps = [], [], {}
    common_basis_bytes = q.numel() * q.element_size()
    for anchor in anchors:
        aid = anchor["example_id"]
        tokens = torch.tensor([anchor["tokens"]])
        clean, inputs = run_write(model, tokens, args.visit, capture=True)
        base = trajectory(model, clean, args.visit)[0]
        deltas = ((donor_inputs[:, -1].double() - inputs[0, -1].double()) @ v.double()) @ u.double()
        coeff = deltas @ q
        if float((deltas - coeff @ q.T).abs().max()) > 1e-8:
            raise RuntimeError("native span reconstruction failed")
        # Independent native swap API parity, all declared readouts, every donor.
        batch = tokens.expand(len(donors), -1)
        donor_acts = (inputs.float() @ v.float()).expand(len(donors), -1, -1).clone()
        donor_acts[:, -1] = donor_inputs[:, -1].float() @ v.float()
        with bmodel.intervene(SITE, list(range(len(u))), passes={args.visit - 1},
                              mode="swap_acts", donor_acts=donor_acts):
            native = wrapped(batch)
        native_logits = trajectory(wrapped, native, args.visit)
        direct, _ = run_write(model, batch, args.visit, deltas.float())
        full = trajectory(model, direct, args.visit)
        parity = float((native_logits - full).abs().max())
        if parity > 3e-4:
            raise RuntimeError(f"native API vs dense additive parity failed: {parity}")
        # Native clean path must also preserve the original residual.
        with bmodel.target_capture():
            wrapped_clean = wrapped(tokens)
        clean_parity = float((trajectory(wrapped, wrapped_clean, args.visit)[0] - base).abs().max())
        if clean_parity > 3e-4:
            raise RuntimeError("wrapped clean path parity failed")
        start = time.perf_counter()
        o, fd_error = observer(teacher64, tokens, args.visit, q)
        stack = Sketch.build(o.flatten(0, 1), args.rank)
        final = Sketch.build(o[-1], args.rank)
        build_seconds = time.perf_counter() - start
        binding = MapBinding(checkpoint, aid, "frozen-teacher", tensor_digest(tokens),
                             tensor_digest(clean.visit_outputs[args.visit - 1]), observer_id,
                             args.visit, tokens.shape[1] - 1, qhash, args.rank)
        maps[aid] = {"stack": stack, "terminal": final}
        hashes = [tensor_digest(c) for c in coeff]
        distinct = len(set(hashes))
        native_bytes = coeff.numel() * coeff.element_size()
        # Teacher response is a common target; its cost may dominate the map.
        response = (full - base[None]).float()
        target_bytes = response.numel() * response.element_size()
        entry = {"anchor": aid, "binding_digest": binding.key().hex(),
                 "donor_count": len(donors), "distinct_displacement_count": distinct,
                 "native_coordinate_bytes": native_bytes,
                 "ambient_write_bytes": deltas.numel() * deltas.element_size(),
                 "common_teacher_target_bytes": target_bytes,
                 "native_api_parity_max_abs": parity, "clean_parity_max_abs": clean_parity,
                 "fd_relative_error": fd_error, "map_build_seconds": build_seconds,
                 "observer_rows": o.numel() // q.shape[1], "methods": {}}
        stored = {}
        for method, sketch in maps[aid].items():
            compressed = coeff @ sketch.basis
            replay_coeff = compressed @ sketch.basis.T
            output, _ = run_write(model, batch, args.visit, (replay_coeff @ q.T).float())
            approx = trajectory(model, output, args.visit)
            checks = [gate(full[i], approx[i], base) for i in range(len(donors))]
            valid = torch.tensor([r["accepted"] for r in checks])
            # Store rank-k coefficients or native-rank fallback, not both,
            # in the candidate representation. The common raw fixture is
            # retained in this development process to compare all arms.
            codes = [compressed[i].clone() if valid[i] else coeff[i].clone()
                     for i in range(len(donors))]
            map_size = len(binding.key()) + len(sketch.encode())
            # Charge a one-byte tag on every candidate record, including fallback.
            candidate_bytes = map_size + sum(c.numel() * c.element_size() + 1 for c in codes)
            use_map = distinct == len(donors) and candidate_bytes < native_bytes
            entry["methods"][method] = {
                "accepted": int(valid.sum()), "fallback": len(donors) - int(valid.sum()),
                "map_bytes": map_size, "candidate_payload_bytes": candidate_bytes,
                "planned_payload_bytes": candidate_bytes if use_map else native_bytes,
                "amortized_map_selected": use_map,
                "relative_error_max": max(r["relative_error"] for r in checks),
                "singular_values": sketch.singular_values.tolist(),
            }
            # Training uses conditional projection with fallback even when
            # the storage planner rejects the map. Keep the distinction.
            stored[method] = (codes, valid)
            for i, donor in enumerate(donors):
                receipt_rows.append({"anchor": aid, "donor": donor["example_id"], "method": method,
                                     "delta_digest": hashes[i], **checks[i]})
        for i, donor in enumerate(donors):
            record = {"anchor": anchor, "donor": donor["example_id"], "native": coeff[i].clone(),
                      "response": response[i].clone()}
            for method, (codes, valid) in stored.items():
                record[method] = (codes[i], bool(valid[i]))
            bank.append(record)
        entry["common_basis_bytes"] = common_basis_bytes
        entry["native_components"] = len(u)
        entry["rank"] = args.rank
        entry["source_depth"] = depth
        entry["scope"] = "distinct TRAIN cross-item swaps; not source-valid causal donors"
        receipt_rows.append({"map_receipt": entry})
    return q, bank, maps, receipt_rows


def training_batch(records):
    anchors = [r["anchor"] for r in records]
    tokens = torch.tensor([r["tokens"] for r in anchors])
    target = torch.tensor([r["target_token"] for r in anchors])
    trace = torch.tensor([[r["symbolic_trace_tokens"][min(j, r["difficulty"])]
                           for j in range(1, 9)] for r in anchors])
    return tokens, target, trace


def decode(record, arm, maps, q):
    if arm == "native_full":
        return record["native"] @ q.T
    method = arm.removeprefix("native_")
    code, compressed = record[method]
    coeff = code @ maps[record["anchor"]["example_id"]][method].basis.T if compressed else code
    return coeff @ q.T


def continuation(args, teacher, train, validation, q, bank, maps):
    rng = torch.Generator().manual_seed(args.seed + 1009)
    draws = torch.randint(len(bank), (2 * args.updates, args.batch), generator=rng).tolist()
    results = []
    for arm in ARMS:
        model = copy.deepcopy(teacher).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
        updates = 2 * args.updates if arm == "ordinary_reinvest" else args.updates
        aux_values, losses = [], []
        start = time.perf_counter()
        for ids in draws[:updates]:
            records = [bank[i] for i in ids]
            tokens, target, trace = training_batch(records)
            optimizer.zero_grad(set_to_none=True)
            clean = model(tokens)
            loss = trace_loss(model, clean, target, trace, "symbolic_trace", 0.5)
            if arm.startswith("native_"):
                delta = torch.stack([decode(r, arm, maps, q) for r in records]).float()
                patched, _ = run_write(model, tokens, args.visit, delta)
                response = trajectory(model, patched, args.visit) - trajectory(model, clean, args.visit)
                desired = torch.stack([r["response"] for r in records])
                aux = F.mse_loss(response, desired)
                loss = loss + args.aux_weight * aux
                aux_values.append(float(aux.detach()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - start
        model.eval()
        with torch.no_grad():
            predictions = []
            for i in range(0, len(validation), args.batch):
                x = torch.tensor([r["tokens"] for r in validation[i:i + args.batch]])
                predictions.extend(model(x).logits[:, -1].argmax(-1).tolist())
        results.append({"arm": arm, "updates": updates, "batch": args.batch,
                        "forward_example_visits": updates * args.batch * 8 * (
                            2 if arm.startswith("native_") else 1),
                        "loss_mean": sum(losses) / len(losses),
                        "aux_loss_mean": sum(aux_values) / len(aux_values) if aux_values else None,
                        "wall_seconds": elapsed, "validation": metrics(validation, predictions, majority(train)),
                        "predictions": predictions, "evaluation_steering_disabled": True})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "factors", "bpd-checkout", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--visit", type=int, default=4)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--donors", type=int, default=16)
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026112609)
    p.add_argument("--aux-weight", type=float, default=1.0)
    args = p.parse_args()
    if (args.out.exists() or not 1 <= args.visit <= 8 or not 1 <= args.anchors <= 8
            or not 2 <= args.donors <= 32 or not 1 <= args.updates <= 64
            or not 1 <= args.batch <= 4 or not 0 <= args.aux_weight <= 10):
        p.error("outside development bounds or output exists")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    teacher, v, u, ck, fh, train, val, factor = load(args)
    bpd = import_bpd(args.bpd_checkout)
    q, bank, maps, receipts = build_bank(args, teacher, v, u, ck, fh, train, bpd)
    result = continuation(args, teacher, train, val, q, bank, maps)
    report = {"schema": "molt09-native-observer-v1", "status": "engineering_development_only",
              "teacher_checkpoint_sha256": ck, "native_factor_sha256": fh,
              "bpd_revision": BPD_REVISION, "native_factor_fit_steps": factor["steps"],
              "teacher_factor_relative_weight_residual": factor["relative_weight_residual"],
              "site": SITE, "visit": args.visit, "native_span_rank": q.shape[1],
              "compact_rank": args.rank, "common_basis_bytes": q.numel() * q.element_size(),
              "teacher_targets_shared_across_arms": True, "bank_records": len(bank),
              "map_receipts": [r["map_receipt"] for r in receipts if "map_receipt" in r],
              "donor_receipts": [r for r in receipts if "map_receipt" not in r],
              "continuation": result, "real_hardware_memory_reduction_verified": False,
              "formal_certificate_authorized": False, "confirmation_ready": False,
              "promotion_authorized": False}
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "receipt.json", report)
    print(json.dumps({"status": report["status"], "bank_records": len(bank),
                      "maps": report["map_receipts"], "arms": [
                          {"arm": r["arm"], "updates": r["updates"],
                           "validation_exact": r["validation"]["exact"],
                           "constant_exact": r["validation"]["constant_exact"],
                           "aux_loss_mean": r["aux_loss_mean"], "wall_seconds": r["wall_seconds"]}
                          for r in result]}, sort_keys=True))


if __name__ == "__main__":
    main()
