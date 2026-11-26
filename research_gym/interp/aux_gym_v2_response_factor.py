"""MOLT-10: exact tied-unembedding response factorization and training parity.

The native MOLT-09 teacher supplies full BPD-span swap coordinates. No
approximate rank reduction, teacher Jacobian reuse in the student, or
sealed evaluation. Payload accounting and actual process memory differ.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2 import dump, majority, metrics, trace_loss
from research_gym.interp.aux_gym_v2_native_bpd import BPD_REVISION
from research_gym.interp.aux_gym_v2_observable_bpd import (
    load, native_span, run_write, trajectory, training_batch,
)

ARMS = ("dense", "decode", "gram", "recompute")
ATOL = 1e-9


def centered_weight(model):
    """The actual tied unembedding, centered in the vocabulary axis."""
    w = model.embedding.weight
    return w - w.mean(0, keepdim=True)


def normalized_trajectory(model, output, visit):
    """Use the native final LayerNorm at every declared readout site."""
    return torch.stack([model.final_norm(h[:, -1]) for h in output.visit_outputs[visit - 1:]], dim=1)


def decode(coefficients, weight):
    if coefficients.shape[-1] != weight.shape[-1] or weight.ndim != 2:
        raise ValueError("response coefficient/weight shape mismatch")
    return coefficients @ weight.T


def gram_loss(student, target, student_weight, teacher_weight, teacher_gram=None):
    """Squared centered-logit MSE with a trainable student unembedding.

    A and B are normalized-state differences, not activations before
    LayerNorm. W_s and W_t may differ. All paths through A and W_s remain
    differentiable. No clamping of the scalar loss; near-zero negative
    roundoff is audited against direct squared error.
    """
    if (student.shape != target.shape or student_weight.shape != teacher_weight.shape
            or student.shape[-1] != student_weight.shape[-1] or student.ndim < 2):
        raise ValueError("response/weight shape mismatch")
    a = student.reshape(-1, student.shape[-1])
    b = target.reshape_as(a)
    gs = student_weight.T @ student_weight
    gt = teacher_weight.T @ teacher_weight if teacher_gram is None else teacher_gram
    cross = student_weight.T @ teacher_weight
    if gt.shape != gs.shape:
        raise ValueError("teacher Gram shape mismatch")
    ss = ((a @ gs) * a).sum()
    st = ((a @ cross) * b).sum()
    tt = ((b @ gt) * b).sum()
    return (ss - 2 * st + tt) / (len(a) * student_weight.shape[0])


def max_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


def payload_report(records, visits, hidden, vocab, native_rank, itemsize=8):
    if min(records, visits, hidden, vocab, native_rank, itemsize) < 1 or native_rank > hidden:
        raise ValueError("invalid payload dimensions")
    response_dense = records * visits * vocab * itemsize
    response_factor = records * visits * hidden * itemsize
    coordinates = records * native_rank * itemsize
    chart = hidden * native_rank * itemsize
    frozen_head = hidden * vocab * itemsize
    gram = hidden * hidden * itemsize
    common = coordinates + chart
    return {
        "records": records, "visits": visits, "hidden": hidden, "vocab": vocab,
        "native_rank": native_rank, "itemsize": itemsize,
        "dense_response_bytes": response_dense, "factor_response_bytes": response_factor,
        "native_coordinate_bytes": coordinates, "native_chart_bytes": chart,
        "frozen_centered_head_bytes": frozen_head, "frozen_teacher_gram_bytes": gram,
        "dense_plus_native_bytes": response_dense + common,
        "factor_plus_native_if_head_resident_bytes": response_factor + common,
        "factor_plus_native_with_head_bytes": response_factor + common + frozen_head,
        "gram_plus_native_with_head_and_gram_bytes": response_factor + common + frozen_head + gram,
        "recompute_target_archive_bytes": 0,
        "recompute_plus_native_bytes_excluding_teacher": common,
        "response_payload_reduction_fraction": 1 - response_factor / response_dense,
    }


@torch.no_grad()
def teacher_pair(model, tokens, visit, delta):
    clean = model(tokens)
    patched, _ = run_write(model, tokens, visit, delta)
    normalized = normalized_trajectory(model, patched, visit) - normalized_trajectory(model, clean, visit)
    direct = trajectory(model, patched, visit) - trajectory(model, clean, visit)
    return normalized, direct


@torch.no_grad()
def build_records(args, teacher, v, u, train):
    """Engineer distinct TRAIN donor/anchor pairs; no donor-causality claim."""
    q = native_span(u)
    ordered = sorted(train, key=lambda r: r["example_id"])
    depth = max(r["difficulty"] for r in ordered)
    pool = [r for r in ordered if r["difficulty"] == depth]
    if len(pool) < args.anchors + args.donors:
        raise ValueError("insufficient disjoint anchor/donor rows")
    anchors = pool[:args.anchors]
    donors = pool[args.anchors:args.anchors + args.donors]
    tokens = torch.tensor([r["tokens"] for r in donors])
    _, donor_inputs = run_write(teacher, tokens, args.visit, capture=True)
    w = centered_weight(teacher).detach().clone()
    bank, receipts = [], []
    for anchor in anchors:
        tokens = torch.tensor([anchor["tokens"]])
        _, a = run_write(teacher, tokens, args.visit, capture=True)
        writes = ((donor_inputs[:, -1] - a[0, -1]) @ v.double()) @ u.double()
        coords = writes @ q
        if max_error(writes, coords @ q.T) > ATOL:
            raise RuntimeError("native span parity failed")
        normalized, dense = teacher_pair(teacher, tokens.expand(len(donors), -1),
                                         args.visit, coords @ q.T)
        parity = max_error(decode(normalized, w), dense)
        if parity > ATOL:
            raise RuntimeError("teacher response factorization failed")
        distinct = len({tuple(c.tolist()) for c in coords})
        if distinct != len(donors):
            raise RuntimeError("distinct donor displacement roster failed")
        for i, donor in enumerate(donors):
            bank.append({"anchor": anchor, "donor": donor["example_id"],
                         "coords": coords[i].clone(), "factor": normalized[i].clone(),
                         "dense": dense[i].clone()})
        receipts.append({"anchor": anchor["example_id"], "depth": depth,
                         "distinct_donors": distinct, "factor_decode_max_abs": parity})
    # All representations coexist solely in this comparative engineering run.
    # The reported payload alternatives do not describe the resident process.
    return q, w, bank, receipts


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        a = p.detach().cpu().contiguous()
        h.update(str((name, a.dtype, tuple(a.shape))).encode())
        h.update(bytes(a.view(torch.uint8).reshape(-1).tolist()))
    return h.hexdigest()


def gradient_error(current, expected):
    if set(current) != set(expected):
        raise RuntimeError("gradient roster mismatch")
    return max((max_error(current[k], expected[k]) for k in current), default=0.0)


def gradients(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def evaluate(model, validation, train, batch):
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(validation), batch):
            x = torch.tensor([r["tokens"] for r in validation[i:i + batch]])
            pred.extend(model(x).logits[:, -1].argmax(-1).tolist())
    return metrics(validation, pred, majority(train)), pred


def continuation(args, teacher, train, validation, q, w, bank):
    rng = torch.Generator().manual_seed(args.seed + 1013)
    schedule = torch.randint(len(bank), (args.updates, args.batch), generator=rng).tolist()
    gt = w.T @ w
    results, baseline_grads, baseline_state = [], {}, None
    baseline_losses, baseline_auxs = None, None
    teacher_id = parameter_hash(teacher)
    for arm in ARMS:
        student = copy.deepcopy(teacher).train()
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005)
        losses, auxs = [], []
        gradient_errors = {}
        start = time.perf_counter()
        for step, ids in enumerate(schedule):
            records = [bank[i] for i in ids]
            tokens, target, traces = training_batch(records)
            delta = torch.stack([r["coords"] for r in records]) @ q.T
            optimizer.zero_grad(set_to_none=True)
            clean = student(tokens)
            patched, _ = run_write(student, tokens, args.visit, delta)
            task = trace_loss(student, clean, target, traces, "symbolic_trace", 0.5)
            if arm == "gram":
                a = normalized_trajectory(student, patched, args.visit) - normalized_trajectory(
                    student, clean, args.visit)
                b = torch.stack([r["factor"] for r in records])
                aux = gram_loss(a, b, centered_weight(student), w, gt)
            else:
                actual = trajectory(student, patched, args.visit) - trajectory(student, clean, args.visit)
                if arm == "dense":
                    desired = torch.stack([r["dense"] for r in records])
                elif arm == "decode":
                    desired = decode(torch.stack([r["factor"] for r in records]), w)
                else:
                    _, desired = teacher_pair(teacher, tokens, args.visit, delta)
                aux = F.mse_loss(actual, desired)
            if not bool(torch.isfinite(aux)):
                raise RuntimeError("nonfinite response loss")
            loss = task + args.aux_weight * aux
            loss.backward()
            if step in {0, args.updates - 1}:
                current = gradients(student)
                if arm == "dense":
                    baseline_grads[step] = current
                    gradient_errors[step] = 0.0
                else:
                    error = gradient_error(current, baseline_grads[step])
                    gradient_errors[step] = error
                    if error > (ATOL if step == 0 else 1e-7):
                        raise RuntimeError(f"gradient parity failed: {arm} {step} {error}")
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            auxs.append(float(aux.detach()))
        elapsed = time.perf_counter() - start
        state = {n: p.detach().clone() for n, p in student.state_dict().items()}
        if arm == "dense":
            baseline_state, baseline_losses, baseline_auxs = state, losses, auxs
            final_error = loss_error = aux_error = 0.0
        else:
            final_error = max(max_error(state[n], baseline_state[n]) for n in state)
            loss_error = max(abs(a - b) for a, b in zip(losses, baseline_losses, strict=True))
            aux_error = max(abs(a - b) for a, b in zip(auxs, baseline_auxs, strict=True))
            if final_error > 1e-7 or loss_error > 1e-7 or aux_error > ATOL:
                raise RuntimeError(f"continuation parity failed: {arm} {final_error} {loss_error}")
        val, predictions = evaluate(student, validation, train, args.batch)
        results.append({"arm": arm, "updates": args.updates, "batch": args.batch,
                        "student_forward_example_visits": 2 * args.updates * args.batch * 8,
                        "teacher_recompute_example_visits": (
                            2 * args.updates * args.batch * 8 if arm == "recompute" else 0),
                        "first_gradient_max_abs_error": gradient_errors[0],
                        "last_gradient_max_abs_error": gradient_errors[args.updates - 1],
                        "max_aux_loss_error": aux_error,
                        "final_parameter_max_abs_error": final_error,
                        "max_training_loss_error": loss_error,
                        "mean_aux_loss": sum(auxs) / len(auxs),
                        "min_aux_loss": min(auxs), "wall_seconds": elapsed,
                        "validation": val, "predictions": predictions,
                        "checkpoint": parameter_hash(student)})
    if parameter_hash(teacher) != teacher_id:
        raise RuntimeError("frozen teacher changed")
    if any(r["predictions"] != results[0]["predictions"] for r in results):
        raise RuntimeError("ordinary validation prediction parity failed")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "factors", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rank", type=int, default=4, help="existing native factor loader constraint only")
    p.add_argument("--visit", type=int, default=4)
    p.add_argument("--anchors", type=int, default=2)
    p.add_argument("--donors", type=int, default=16)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026112710)
    p.add_argument("--aux-weight", type=float, default=1.0)
    args = p.parse_args()
    if (args.out.exists() or not 1 <= args.visit <= 8 or not 1 <= args.anchors <= 8
            or not 2 <= args.donors <= 32 or not 1 <= args.updates <= 32
            or not 1 <= args.batch <= 4 or not 0 <= args.aux_weight <= 10):
        p.error("outside engineering bounds or output exists")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    teacher, v, u, ck, fh, train, val, factor = load(args)
    teacher = teacher.double().eval()
    q, w, bank, receipts = build_records(args, teacher, v, u, train)
    result = continuation(args, teacher, train, val, q, w, bank)
    payload = payload_report(len(bank), 9 - args.visit, w.shape[1], w.shape[0], q.shape[1])
    report = {"schema": "molt10-response-factor-v1", "status": "engineering_development_only",
              "teacher_checkpoint_sha256": ck, "native_factor_sha256": fh,
              "bpd_revision": BPD_REVISION, "native_factor_fit_steps": factor["steps"],
              "visit": args.visit, "dtype": "float64", "payload": payload,
              "anchor_receipts": receipts, "continuation": result,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "all_representations_resident_for_comparison": True,
              "real_hardware_memory_reduction_verified": False,
              "confirmation_ready": False, "promotion_authorized": False}
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "receipt.json", report)
    print(json.dumps({"status": report["status"], "payload": payload, "anchors": receipts,
                      "arms": [{k: r[k] for k in ("arm", "first_gradient_max_abs_error",
                                                  "last_gradient_max_abs_error", "max_aux_loss_error",
                                                  "final_parameter_max_abs_error", "max_training_loss_error",
                                                  "mean_aux_loss", "min_aux_loss", "wall_seconds")}
                               | {"validation_exact": r["validation"]["exact"]} for r in result]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
