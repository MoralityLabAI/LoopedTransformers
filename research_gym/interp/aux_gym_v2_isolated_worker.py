"""MOLT-11 isolated CPU worker for exact response-memory alternatives.

The supervisor owns resource limits. Each invocation handles one preparation,
training or audit phase. No locked split is read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from research_gym.interp.aux_gym_v2 import digest_file, dump, majority, metrics, rows
from research_gym.interp.aux_gym_v2_observable_bpd import load, native_span, run_write
from research_gym.interp.aux_gym_v2_response_factor import centered_weight, decode, gram_loss
from research_gym.interp.models import build_model

ARMS = ("dense", "decode", "gram", "recompute")
SITE = "blocks.0.mlp.2"


def rss_bytes():
    with open("/proc/self/status", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("missing process RSS")


def peak_bytes():
    # Linux ru_maxrss is KiB, including import and load peaks.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def payload_bytes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(payload_bytes(v) for v in obj.values())
    return 0


def answer_states(model, tokens, visit=None, delta=None):
    """Exact native tied block sequence, with readout only at the answer token.

    Restricted to the reset-carry, fully visible, eight-visit LOOP-T teacher
    or a smaller equally tied engineering fixture. All full sequence states
    continue through causal attention. This is not a strict-prefix MoLT kernel.
    """
    n = model.expanded_visits
    if (model.carry_policy != "reset" or len(set(model.module_indices)) != 1
            or model.parameter_visits != frozenset(range(n))
            or model.retained_state_edges != frozenset(range(n - 1))
            or model.activation_checkpointing):
        raise ValueError("answer-only path requires the declared reset LOOP-T schedule")
    if (visit is None) != (delta is None) or (visit is not None and not 1 <= visit <= n):
        raise ValueError("invalid write intervention")
    state = model.embed(tokens)
    mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                                  device=tokens.device), diagonal=1)
    captured = []
    for i in range(n):
        if visit == i + 1:
            calls = [0]

            def hook(_module, _inputs, output):
                calls[0] += 1
                if delta.shape != output[:, -1].shape:
                    raise ValueError("answer write shape mismatch")
                return torch.cat((output[:, :-1], output[:, -1:] + delta[:, None].to(output)), dim=1)

            handle = model.get_submodule(SITE).register_forward_hook(hook)
            try:
                state = model.block_at(i)(state, causal_mask=mask)
            finally:
                handle.remove()
            if calls != [1]:
                raise RuntimeError("unexpected physical write count")
        else:
            state = model.block_at(i)(state, causal_mask=mask)
        captured.append(model.final_norm(state[:, -1]))
    return torch.stack(captured, dim=1)


def task_loss(model, normalized, targets, traces):
    logits = F.linear(normalized, model.embedding.weight)
    final = F.cross_entropy(logits[:, -1].float(), targets)
    aux = [F.cross_entropy(logits[:, i].float(), traces[:, i])
           for i in range(logits.shape[1] - 1)]
    return final + 0.5 * torch.stack(aux).mean() if aux else final


def checked_checkpoint(run):
    receipt = json.loads((run / "receipt.json").read_text())
    if (receipt["arm"] != "LOOP-T" or receipt["visits"] != 8
            or digest_file(run / "model.pt") != receipt["checkpoint_sha256"]):
        raise ValueError("checkpoint binding mismatch")
    model = build_model("LOOP-T", visits=8)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    return model.double(), receipt


def checked_validation(data, teacher_receipt):
    manifest = json.loads((data / "manifest.json").read_text())
    if digest_file(data / "manifest.json") != teacher_receipt["manifest_sha256"]:
        raise ValueError("dataset binding mismatch")
    name = f"validation/{teacher_receipt['family']}.jsonl"
    path = data / name
    if digest_file(path) != manifest["files"][name]["sha256"]:
        raise ValueError("validation shard binding mismatch")
    return rows(path)


@torch.no_grad()
def predictions(model, validation, batch):
    model.eval()
    result = []
    for i in range(0, len(validation), batch):
        tokens = torch.tensor([r["tokens"] for r in validation[i:i + batch]])
        h = answer_states(model, tokens)
        result.extend(F.linear(h[:, -1], model.embedding.weight).argmax(-1).tolist())
    return result


def new_directory(path):
    path.mkdir(parents=True, exist_ok=False)


def write_tensors(path, obj):
    torch.save(obj, path)
    return {"sha256": digest_file(path), "tensor_payload_bytes": payload_bytes(obj),
            "file_bytes": path.stat().st_size}


def read_common(path):
    rec = json.loads((path / "receipt.json").read_text())
    if rec.get("schema") != "molt11-common-v1":
        raise ValueError("unexpected common receipt")
    tensor_path = path / "common.pt"
    if digest_file(tensor_path) != rec["archive"]["sha256"]:
        raise ValueError("common archive hash mismatch")
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if payload_bytes(data) != rec["archive"]["tensor_payload_bytes"]:
        raise ValueError("common archive payload mismatch")
    n = rec["records"]
    if (set(data) != {"tokens", "targets", "traces", "coords", "q"}
            or data["tokens"].shape[0] != n or data["targets"].shape != (n,)
            or data["traces"].shape != (n, 8)
            or data["coords"].shape != (n, rec["native_rank"])
            or data["q"].shape != (rec["hidden"], rec["native_rank"])):
        raise ValueError("common record shape mismatch")
    if any(not bool(torch.isfinite(data[k]).all()) for k in ("coords", "q")):
        raise ValueError("nonfinite native coordinates")
    return data, rec


def bind_common(rec, teacher_receipt, visit):
    if (rec["teacher_checkpoint_sha256"] != teacher_receipt["checkpoint_sha256"]
            or rec["manifest_sha256"] != teacher_receipt["manifest_sha256"]
            or rec["visit"] != visit):
        raise ValueError("common archive/teacher binding mismatch")


@torch.no_grad()
def prepare_common(args):
    model, v, u, ck, fh, train, validation, factor = load(args)
    model = model.double().eval()
    q = native_span(u)
    ordered = sorted(train, key=lambda r: r["example_id"])
    depth = max(r["difficulty"] for r in ordered)
    pool = [r for r in ordered if r["difficulty"] == depth]
    if len(pool) < args.anchors + args.donors:
        raise ValueError("insufficient distinct TRAIN donors")
    anchors, donors = pool[:args.anchors], pool[args.anchors:args.anchors + args.donors]
    _, inputs = run_write(model, torch.tensor([r["tokens"] for r in donors]), args.visit, capture=True)
    tokens, targets, traces, coords, roster = [], [], [], [], []
    for anchor in anchors:
        _, a = run_write(model, torch.tensor([anchor["tokens"]]), args.visit, capture=True)
        d = ((inputs[:, -1] - a[0, -1]) @ v.double()) @ u.double()
        c = d @ q
        if float((d - c @ q.T).abs().max()) > 1e-9:
            raise RuntimeError("native coordinate reconstruction failed")
        if len({tuple(x.tolist()) for x in c}) != len(donors):
            raise RuntimeError("duplicate donor displacements")
        for i, donor in enumerate(donors):
            tokens.append(anchor["tokens"])
            targets.append(anchor["target_token"])
            traces.append([anchor["symbolic_trace_tokens"][min(j, anchor["difficulty"])]
                           for j in range(1, 9)])
            coords.append(c[i])
            roster.append([anchor["example_id"], donor["example_id"]])
    data = {"tokens": torch.tensor(tokens), "targets": torch.tensor(targets),
            "traces": torch.tensor(traces), "coords": torch.stack(coords), "q": q}
    pred = predictions(model, validation, args.batch)
    quality = metrics(validation, pred, majority(train))
    # Fixed development gate; no selecting a teacher by its validation result.
    competent = (quality["exact"] >= 0.25
                 and quality["exact"] - quality["constant_exact"] >= 0.10)
    # Verify answer-only path against native full readout at the first anchor.
    x = data["tokens"][:1]
    native = model(x)
    h = answer_states(model, x)
    parity = float((F.linear(h[:, -1], model.embedding.weight) - native.logits[:, -1]).abs().max())
    if parity > 1e-10:
        raise RuntimeError("native answer-path parity failed")
    new_directory(args.out)
    archive = write_tensors(args.out / "common.pt", data)
    report = {"schema": "molt11-common-v1", "records": len(roster), "roster": roster,
              "teacher_checkpoint_sha256": ck, "factor_sha256": fh,
              "manifest_sha256": factor["manifest_sha256"], "native_rank": q.shape[1],
              "hidden": q.shape[0], "vocab": model.vocab_size, "visit": args.visit,
              "constant_token": majority(train), "teacher_validation": quality,
              "competence_gate_passed": competent,
              "competence_gate": {"min_exact": 0.25, "min_lift_over_constant": 0.10},
              "native_answer_parity_max_abs": parity, "archive": archive,
              "peak_rss_bytes": peak_bytes()}
    dump(args.out / "receipt.json", report)


@torch.no_grad()
def prepare_targets(args):
    if args.arm not in ("dense", "factor"):
        raise ValueError("target preparation arm must be dense or factor")
    data, common = read_common(args.common)
    model, teacher = checked_checkpoint(args.run)
    model.eval()
    bind_common(common, teacher, args.visit)
    w = centered_weight(model).detach()
    n, visits = common["records"], 9 - args.visit
    width = common["vocab"] if args.arm == "dense" else common["hidden"]
    target = torch.empty(n, visits, width, dtype=torch.float64)
    start = time.perf_counter()
    for i in range(0, n, args.batch):
        x = data["tokens"][i:i + args.batch]
        d = data["coords"][i:i + args.batch] @ data["q"].T
        a = answer_states(model, x)
        b = answer_states(model, x, args.visit, d)
        response = b[:, args.visit - 1:] - a[:, args.visit - 1:]
        target[i:i + args.batch] = decode(response, w) if args.arm == "dense" else response
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("nonfinite teacher response archive")
    elapsed = time.perf_counter() - start
    # Never store the other target representation in this process.
    archive = {"targets": target}
    if args.arm == "factor":
        archive["head"] = w.contiguous()
    new_directory(args.out)
    result = {"schema": "molt11-target-v1", "arm": args.arm,
              "common_sha256": common["archive"]["sha256"],
              "teacher_checkpoint_sha256": common["teacher_checkpoint_sha256"],
              "response_bytes": payload_bytes(target), "head_bytes": payload_bytes(archive.get("head")),
              "archive": write_tensors(args.out / "targets.pt", archive),
              "preparation_seconds": elapsed, "peak_rss_bytes": peak_bytes()}
    dump(args.out / "receipt.json", result)


def load_targets(path, common, arm):
    rec = json.loads((path / "receipt.json").read_text())
    expected = "dense" if arm == "dense" else "factor"
    if (rec["schema"] != "molt11-target-v1" or rec["arm"] != expected
            or rec["common_sha256"] != common["archive"]["sha256"]
            or rec["teacher_checkpoint_sha256"] != common["teacher_checkpoint_sha256"]
            or rec["archive"]["sha256"] != digest_file(path / "targets.pt")):
        raise ValueError("target archive binding mismatch")
    archive = torch.load(path / "targets.pt", map_location="cpu", weights_only=True)
    width = common["vocab"] if arm == "dense" else common["hidden"]
    expected_keys = {"targets"} if arm == "dense" else {"targets", "head"}
    if (set(archive) != expected_keys or payload_bytes(archive) != rec["archive"]["tensor_payload_bytes"]
            or archive["targets"].shape != (common["records"], 9 - common["visit"], width)
            or any(t.dtype != torch.float64 or not bool(torch.isfinite(t).all()) for t in archive.values())):
        raise ValueError("target archive shape or dtype mismatch")
    if "head" in archive and archive["head"].shape != (common["vocab"], common["hidden"]):
        raise ValueError("target head shape mismatch")
    return archive, rec


def train_arm(args):
    if args.arm not in ARMS:
        raise ValueError("unknown arm")
    data, common = read_common(args.common)
    student, receipt = checked_checkpoint(args.run)
    bind_common(common, receipt, args.visit)
    validation = checked_validation(args.data, receipt)
    archive, target_rec, teacher, w, gt = {}, None, None, None, None
    if args.arm == "recompute":
        teacher, _ = checked_checkpoint(args.run)
        teacher.eval().requires_grad_(False)
        w = centered_weight(teacher).detach()
    else:
        archive, target_rec = load_targets(args.targets_dir, common, args.arm)
        if args.arm != "dense":
            w = archive["head"]
            if args.arm == "gram":
                gt = w.T @ w
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005)
    rng = torch.Generator().manual_seed(args.seed + 1013)
    schedule = torch.randint(common["records"], (args.updates, args.batch), generator=rng)
    student.train()
    gc.collect()
    start_rss = rss_bytes()
    peak_before = peak_bytes()
    wall, cpu = time.perf_counter(), time.process_time()
    losses, response_losses = [], []
    for ids in schedule:
        x, y, trace = data["tokens"][ids], data["targets"][ids], data["traces"][ids]
        delta = data["coords"][ids] @ data["q"].T
        optimizer.zero_grad(set_to_none=True)
        clean = answer_states(student, x)
        patched = answer_states(student, x, args.visit, delta)
        task = task_loss(student, clean, y, trace)
        a = patched[:, args.visit - 1:] - clean[:, args.visit - 1:]
        ws = centered_weight(student)
        if args.arm == "gram":
            aux = gram_loss(a, archive["targets"][ids], ws, w, gt)
        else:
            if args.arm == "dense":
                desired = archive["targets"][ids]
            elif args.arm == "decode":
                desired = decode(archive["targets"][ids], w)
            else:
                with torch.no_grad():
                    t = answer_states(teacher, x, args.visit, delta) - answer_states(teacher, x)
                    desired = decode(t[:, args.visit - 1:], w)
            aux = F.mse_loss(decode(a, ws), desired)
        loss = task + args.aux_weight * aux
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        response_losses.append(float(aux.detach()))
    elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
    train_peak = peak_bytes()
    pred = predictions(student, validation, args.batch)
    new_directory(args.out)
    torch.save(student.state_dict(), args.out / "student.pt")
    teacher_bytes = sum(t.numel() * t.element_size() for t in teacher.state_dict().values()) if teacher else 0
    report = {"schema": "molt11-arm-v1", "arm": args.arm, "updates": args.updates, "batch": args.batch,
              "seed": args.seed, "aux_weight": args.aux_weight, "common_sha256": common["archive"]["sha256"],
              "target_sha256": target_rec["archive"]["sha256"] if target_rec else None,
              "checkpoint_sha256": digest_file(args.out / "student.pt"), "visit": args.visit,
              "student_example_visits": 16 * args.updates * args.batch,
              "teacher_example_visits": 16 * args.updates * args.batch if teacher else 0,
              "common_tensor_bytes": payload_bytes(data), "target_tensor_bytes": payload_bytes(archive),
              "extra_gram_bytes": payload_bytes(gt), "recompute_teacher_parameter_bytes": teacher_bytes,
              "recompute_centered_head_bytes": payload_bytes(w) if teacher else 0,
              "rss_after_load_bytes": start_rss, "lifetime_peak_before_training_bytes": peak_before,
              "lifetime_peak_after_training_bytes": train_peak, "lifetime_peak_at_receipt_bytes": peak_bytes(),
              "training_wall_seconds": elapsed, "training_cpu_seconds": cpu_elapsed,
              "losses": losses, "response_losses": response_losses, "predictions": pred,
              "validation": metrics(validation, pred, common["constant_token"]),
              "competence_gate_passed": common["competence_gate_passed"],
              "training_effect_claim_authorized": False, "promotion_authorized": False}
    dump(args.out / "receipt.json", report)


def audit(args):
    common, receipt = read_common(args.common)
    del common
    baseline = None
    ref = None
    results = []
    for arm in ARMS:
        directory = args.arms_dir / arm
        report = json.loads((directory / "receipt.json").read_text())
        if (report["schema"] != "molt11-arm-v1" or report["arm"] != arm
                or report["common_sha256"] != receipt["archive"]["sha256"]
                or report["checkpoint_sha256"] != digest_file(directory / "student.pt")):
            raise ValueError("arm receipt binding mismatch")
        state = torch.load(directory / "student.pt", map_location="cpu", weights_only=True)
        if baseline is None:
            baseline, ref = state, report
            parameter_error = loss_error = aux_error = 0.0
        else:
            if (set(state) != set(baseline)
                    or any(state[k].shape != baseline[k].shape for k in state)
                    or any(report[k] != ref[k] for k in ("updates", "batch", "seed", "visit",
                                                         "aux_weight", "predictions"))):
                raise RuntimeError("arm comparison contract mismatch")
            parameter_error = max(float((state[k] - baseline[k]).abs().max()) for k in state)
            loss_error = max(abs(a - b) for a, b in zip(report["losses"], ref["losses"], strict=True))
            aux_error = max(abs(a - b) for a, b in zip(report["response_losses"],
                                                       ref["response_losses"], strict=True))
            if parameter_error > 1e-7 or loss_error > 1e-7 or aux_error > 1e-9:
                raise RuntimeError(f"isolated parity failed: {arm} {parameter_error} {loss_error} {aux_error}")
        results.append({"arm": arm, "parameter_max_abs_error": parameter_error,
                        "task_loss_max_abs_error": loss_error, "response_loss_max_abs_error": aux_error})
        del state
    new_directory(args.out)
    dump(args.out / "receipt.json", {"schema": "molt11-audit-v1", "arms": results,
                                      "competence_gate_passed": receipt["competence_gate_passed"],
                                      "training_effect_claim_authorized": False,
                                      "promotion_authorized": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("common", "targets", "train", "audit"))
    for name in ("data", "run", "factors", "common", "targets-dir", "arms-dir", "out"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--arm", default="dense")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--visit", type=int, default=4)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--donors", type=int, default=32)
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026112711)
    p.add_argument("--aux-weight", type=float, default=1.0)
    args = p.parse_args()
    if (args.out is None or args.out.exists() or not 1 <= args.anchors <= 8
            or not 2 <= args.donors <= 32 or not 1 <= args.visit <= 8
            or not 1 <= args.batch <= 8 or not 1 <= args.updates <= 64
            or not 0 <= args.aux_weight <= 10):
        p.error("outside isolated fixture bounds or output exists")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    {"common": prepare_common, "targets": prepare_targets, "train": train_arm, "audit": audit}[args.phase](args)


if __name__ == "__main__":
    main()
