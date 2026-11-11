"""Measure fixed-depth LoopedChase trajectories with the checkpoint's own head.

This adapter retains every token state and uses the original Block.forward.
Endpoint agreement selects observational pairs; future outcomes never select
pairs. Full-state donor replay is a plumbing control, while partial swaps are
reported as interventions with their complete finite perturbation sizes.

PyTorch is loaded after argument parsing, so --help works without PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_gym.analysis.semantic_echo import aggregate_trajectories, analyze_trajectory

SCHEMA_VERSION = "semantic_echo.looped_chase.v1"
DEFAULT_SEED = 940117
PATCH_KINDS = (
    "full_donor", "cursor_swap", "edge_swap", "per_token_layernorm_shift",
    "random_cursor", "random_edges", "random_full",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for checkpoint evaluation and --smoke. Install a "
            "PyTorch build for your CPU/CUDA environment, then rerun this command. "
            "For CPU: python -m pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu. --help remains available without it."
        ) from error
    return torch


def native_embedding(model, tokens):
    """Exactly the prelude in LoopedChase.forward, including token positions."""
    torch = _torch()
    n = model.n_nodes
    edges = tokens[:, :2 * n].reshape(tokens.shape[0], n, 2)
    embedded = torch.cat([
        model.tok(edges[..., 0]) + model.dst(edges[..., 1]),
        model.tok(tokens[:, 2 * n:]),
    ], dim=1)
    return embedded + model.pos(torch.arange(embedded.shape[1], device=tokens.device))


def native_step(model, state, embedding):
    """Advance the full carried state through the source model's native blocks."""
    state = model.inject(state) + embedding
    for block in model.core:
        state = block(state)
    return state


def native_trace(model, tokens, n_loops):
    """Return fixed input embedding and states [T+1,B,tokens,d], including h_0."""
    torch = _torch()
    embedding = native_embedding(model, tokens)
    state = torch.zeros_like(embedding)
    states = [state]
    for _ in range(n_loops):
        state = native_step(model, state, embedding)
        states.append(state)
    return embedding, torch.stack(states)


def rollout(model, state, embedding, steps):
    torch = _torch()
    states = []
    for _ in range(steps):
        state = native_step(model, state, embedding)
        states.append(state)
    return torch.stack(states)


def assert_state_parity(actual, reference, *, rtol, atol, name):
    """Per-visit, per-item L2 tolerance; every item must pass."""
    torch = _torch()
    if actual.shape != reference.shape:
        raise RuntimeError(f"{name}: shape mismatch {actual.shape} != {reference.shape}")
    delta = (actual - reference).flatten(start_dim=2)
    ref = reference.flatten(start_dim=2)
    error = torch.linalg.vector_norm(delta, dim=-1)
    scale = torch.linalg.vector_norm(ref, dim=-1)
    finite = torch.isfinite(actual).all() & torch.isfinite(reference).all()
    failures = error > atol + rtol * scale
    if not bool(finite) or bool(failures.any()):
        indices = failures.nonzero().cpu().tolist()[:8]
        raise RuntimeError(f"{name}: native parity failed; nonfinite={not bool(finite)}, "
                           f"failed visit/item indices={indices}")
    relative = error / scale.clamp_min(1e-30)
    return {"passed": True, "comparisons": error.numel(),
            "max_abs_error": float(delta.abs().max().item()),
            "max_item_l2_error": float(error.max().item()),
            "max_item_relative_l2_error": float(relative.max().item()),
            "rtol": rtol, "atol": atol}


def verify_native_trace(model, tokens, full_states, *, rtol, atol):
    reference = model(tokens, len(full_states) - 1)
    cursors = full_states[1:, :, -1, :]
    result = assert_state_parity(cursors, reference, rtol=rtol, atol=atol,
                                 name="cursor trajectory vs LoopedChase.forward")
    actual_logits = model.readout(cursors)
    reference_logits = model.readout(reference)
    result["logit_parity"] = assert_state_parity(actual_logits, reference_logits,
                                                 rtol=rtol, atol=atol,
                                                 name="native own-head logits")
    actual_labels = actual_logits.argmax(-1)
    reference_labels = reference_logits.argmax(-1)
    mismatches = int((actual_labels != reference_labels).sum().item())
    if mismatches:
        raise RuntimeError(f"native parity failed: {mismatches} own-head label mismatches")
    result["own_head_label_mismatches"] = mismatches
    return result


def patch_state(base, donor, kind, n_edges, rng):
    """Build a finite intervention; source/base shapes are [B,tokens,d]."""
    torch = _torch()
    if base.shape != donor.shape or base.ndim != 3:
        raise ValueError("base and donor must have identical [B,tokens,d] shapes")
    if n_edges < 1 or n_edges >= base.shape[1]:
        raise ValueError("n_edges must leave at least one cursor token")
    scope = "all_tokens"
    if kind in ("cursor_swap", "random_cursor"):
        token_slice = slice(-1, None)
        scope = "cursor"
    elif kind in ("edge_swap", "random_edges"):
        token_slice = slice(0, n_edges)
        scope = "edge_tokens"
    else:
        token_slice = slice(None)
    target = donor[:, token_slice] - base[:, token_slice]
    target_norm = torch.linalg.vector_norm(target)
    result = base.clone()
    if kind in ("full_donor", "cursor_swap", "edge_swap"):
        result[:, token_slice] = donor[:, token_slice]
    elif kind == "per_token_layernorm_shift":
        noise = torch.as_tensor(rng.standard_normal((*base.shape[:-1], 1)),
                                dtype=base.dtype, device=base.device).expand_as(base)
        direction_norm = torch.linalg.vector_norm(noise)
        result = base + noise * (target_norm / direction_norm.clamp_min(1e-30))
        scope = "all_tokens_constant_shift_per_token"
    elif kind in ("random_cursor", "random_edges", "random_full"):
        noise = torch.as_tensor(rng.standard_normal(tuple(target.shape)),
                                dtype=base.dtype, device=base.device)
        direction_norm = torch.linalg.vector_norm(noise)
        result[:, token_slice] += noise * (target_norm / direction_norm.clamp_min(1e-30))
    else:
        raise ValueError(f"unknown intervention kind: {kind}")
    delta = result - base
    return result, {"kind": kind, "scope": scope,
                    "donor_difference_l2_in_scope": float(target_norm.item()),
                    "applied_l2": float(torch.linalg.vector_norm(delta).item()),
                    "base_l2": float(torch.linalg.vector_norm(base).item()),
                    "zero_size_intervention": bool(target_norm.item() == 0),
                    "changed_token_indices": (delta.abs().sum(-1) > 0).nonzero().cpu().tolist()}


def _ratio(n, d):
    return n / d if d else None


def _stratum(t, u, training_loops):
    if u <= training_loops:
        return "both_within_training_depth"
    if t <= training_loops:
        return "donor_within_receiver_beyond_training_depth"
    return "both_beyond_training_depth"


def select_pairs(analyses, max_pairs, seed):
    """Selection reads only item IDs, times, exact endpoint labels and gates."""
    candidates = []
    for item, analysis in enumerate(analyses):
        for row in analysis["continuation"]["raw_rows"]:
            if row["endpoint_gate"]:
                candidates.append({"item": item, "t": row["t"], "u": row["u"],
                                   "semantic_label": row["semantic_label"]})
    if len(candidates) <= max_pairs:
        return candidates
    indices = np.random.default_rng(seed).choice(len(candidates), max_pairs, replace=False)
    return [candidates[int(i)] for i in sorted(indices)]


def replay_pair(model, states, embedding, pair, expected, *, horizon, seed, rtol, atol):
    """Replay all scopes in one batch; full donor is checked against its trace."""
    torch = _torch()
    item, t, u = pair["item"], pair["t"], pair["u"]
    donor, base = states[t, item:item + 1], states[u, item:item + 1]
    rng = np.random.default_rng(seed)
    patched, metadata = zip(*(patch_state(base, donor, kind, model.n_nodes, rng)
                              for kind in PATCH_KINDS))
    resumed = rollout(model, torch.cat(patched),
                      embedding[item:item + 1].expand(len(PATCH_KINDS), -1, -1), horizon)
    native_donor = states[t + 1:t + 1 + horizon, item:item + 1]
    parity = assert_state_parity(resumed[:, :1], native_donor, rtol=rtol, atol=atol,
                                 name="full-donor replay positive control")
    resumed_logits = model.readout(resumed[:, :, -1])
    native_donor_logits = model.readout(native_donor[:, :, -1])
    logit_parity = assert_state_parity(resumed_logits[:, :1], native_donor_logits,
                                       rtol=rtol, atol=atol,
                                       name="full-donor own-head logit replay")
    predictions = resumed_logits.argmax(-1).cpu().numpy()
    donor_labels = native_donor_logits.argmax(-1)[:, 0].cpu().numpy()
    if not np.array_equal(predictions[:, 0], donor_labels):
        raise RuntimeError("full-donor replay changed the source trajectory's own-head labels")
    base_future = states[u + 1:u + 1 + horizon, item:item + 1]
    baseline_labels = model.readout(base_future[:, :, -1]).argmax(-1)[:, 0].cpu().numpy()
    gauge_index = PATCH_KINDS.index("per_token_layernorm_shift")
    gauge_delta = resumed[:, gauge_index:gauge_index + 1] - base_future
    gauge_scale = torch.linalg.vector_norm(base_future.flatten(start_dim=2), dim=-1).clamp_min(1e-30)
    gauge_relative = torch.linalg.vector_norm(gauge_delta.flatten(start_dim=2), dim=-1) / gauge_scale
    expected = np.asarray(expected, dtype=np.int64)
    outcomes = []
    for index, entry in enumerate(metadata):
        labels = predictions[:, index]
        outcomes.append({**entry, "future_predictions": labels.tolist(),
                         "oracle_correct_count": int((labels == expected).sum()),
                         "oracle_comparisons": horizon,
                         "disagreement_with_receiver_count": int((labels != baseline_labels).sum()),
                         "disagreement_with_donor_count": int((labels != donor_labels).sum())})
    return {**pair, "intervention_seed": seed, "expected_future": expected.tolist(),
            "receiver_future": baseline_labels.tolist(), "donor_future": donor_labels.tolist(),
            "full_donor_replay_parity": parity,
            "full_donor_logit_parity": logit_parity,
            "layernorm_shift_max_relative_state_error": float(gauge_relative.max().item()),
            "interventions": outcomes,
            "interpretation": "full_donor replay is a positive control; partial swaps are finite interventions"}


def _state_dict_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode() + b"\0")
        digest.update(str(array.dtype).encode() + repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_model(checkpoint, *, smoke, seed, device, dtype):
    torch = _torch()
    from research_gym.scripts.loopscope_toy import LoopedChase
    if smoke:
        torch.manual_seed(seed)
        cfg = {"n_nodes": 4, "d": 16, "layers": 1, "heads": 2,
               "fixed_depth": True, "max_train_depth": 8, "loops": 8,
               "seed": seed, "lam": None, "training_status": "UNTRAINED_SMOKE"}
        state = None
    else:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError(f"Could not safely load checkpoint {checkpoint} with weights_only=True: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("cfg"), dict) or not isinstance(payload.get("model"), dict):
            raise ValueError(f"{checkpoint}: expected a final checkpoint with dict keys 'cfg' and 'model'")
        cfg, state = payload["cfg"], payload["model"]
        if cfg.get("fixed_depth") is not True:
            raise ValueError(f"{checkpoint}: this pilot requires fixed_depth=True; variable-depth prompts have a stopping rule and need a separate oracle")
    for key in ("n_nodes", "d", "layers", "heads"):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"checkpoint cfg.{key} must be a positive integer")
    if cfg["d"] % cfg["heads"]:
        raise ValueError("checkpoint width must be divisible by attention heads")
    model = LoopedChase(cfg["n_nodes"], cfg["d"], cfg["layers"], cfg["heads"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
    return model, cfg


def _pooled_counts(analyses):
    names = ("eligible_pairs", "endpoint_gated_pairs", "pair_any_disagreement_count",
             "per_step_mismatch_count", "per_step_comparisons", "oracle_correct_count",
             "oracle_comparisons", "both_oracle_correct_count", "both_oracle_comparisons")
    out = {key: sum(a["continuation"][key] for a in analyses) for key in names}
    out.update(endpoint_gate_coverage=_ratio(out["endpoint_gated_pairs"], out["eligible_pairs"]),
               pair_any_disagreement_rate=_ratio(out["pair_any_disagreement_count"], out["endpoint_gated_pairs"]),
               per_step_mismatch_rate=_ratio(out["per_step_mismatch_count"], out["per_step_comparisons"]),
               oracle_continuation_accuracy=_ratio(out["oracle_correct_count"], out["oracle_comparisons"]))
    return out


def future_diagnostics(rows, horizon):
    """Raw future counts with explicit pair-level and offset-level partitions."""
    gated = [row for row in rows if row["endpoint_gate"]]
    cell_names = ("agree_both_correct", "agree_shared_error",
                  "disagree_one_side_correct", "disagree_both_sides_error")
    pair_cells = dict.fromkeys(cell_names, 0)
    step_cells = dict.fromkeys(cell_names, 0)
    offsets = [{"offset": k + 1, "comparisons": len(gated), "disagreement_count": 0,
                "donor_oracle_correct_count": 0, "receiver_oracle_correct_count": 0}
               for k in range(horizon)]
    donor_all = receiver_all = both_all = 0
    for row in gated:
        ft, fu = np.asarray(row["future_predictions_t"]), np.asarray(row["future_predictions_u"])
        hit_t, hit_u = np.asarray(row["oracle_correct_t"], dtype=bool), np.asarray(row["oracle_correct_u"], dtype=bool)
        if any(len(value) != horizon for value in (ft, fu, hit_t, hit_u)):
            raise ValueError("future row length does not match the fixed horizon")
        equal = ft == fu
        all_t, all_u = bool(hit_t.all()), bool(hit_u.all())
        donor_all += int(all_t)
        receiver_all += int(all_u)
        both_all += int(all_t and all_u)
        if bool(equal.all()):
            pair_cells["agree_both_correct" if all_t and all_u else "agree_shared_error"] += 1
        else:
            pair_cells["disagree_one_side_correct" if all_t or all_u else "disagree_both_sides_error"] += 1
        for k, entry in enumerate(offsets):
            entry["disagreement_count"] += int(not equal[k])
            entry["donor_oracle_correct_count"] += int(hit_t[k])
            entry["receiver_oracle_correct_count"] += int(hit_u[k])
            if equal[k]:
                key = "agree_both_correct" if hit_t[k] and hit_u[k] else "agree_shared_error"
            else:
                key = "disagree_one_side_correct" if hit_t[k] or hit_u[k] else "disagree_both_sides_error"
            step_cells[key] += 1
    for entry in offsets:
        entry.update(disagreement_rate=_ratio(entry["disagreement_count"], entry["comparisons"]),
                     donor_oracle_accuracy=_ratio(entry["donor_oracle_correct_count"], entry["comparisons"]),
                     receiver_oracle_accuracy=_ratio(entry["receiver_oracle_correct_count"], entry["comparisons"]))
    n_pairs = len(gated)
    return {"endpoint_gated_pairs": n_pairs, "per_future_offset": offsets,
            "all_h_both_correct_pair_count": both_all,
            "all_h_both_correct_pair_rate": _ratio(both_all, n_pairs),
            "all_h_donor_correct_pair_count": donor_all,
            "all_h_receiver_correct_pair_count": receiver_all,
            "pair_comparison_cells": pair_cells, "step_comparison_cells": step_cells,
            "pair_cell_definition": "agreement means all H labels agree; a correct side means its complete H-step sequence is correct",
            "step_cell_definition": "each cell concerns one pair at one future offset",
            "orientation": "donor=t is earlier; receiver=u is later"}


def _code_revision():
    try:
        result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def evaluate_one(checkpoint, args, run_dir):
    torch = _torch()
    from research_gym.scripts.loopscope_toy import make_batch
    started = time.monotonic()
    model, cfg = load_model(checkpoint, smoke=args.smoke, seed=args.seed,
                            device=args.device, dtype=args.dtype)
    rtol = 2e-5 if args.dtype == "float32" else 1e-10
    atol = 2e-6 if args.dtype == "float32" else 1e-12
    tokens, _ = make_batch(cfg["n_nodes"], args.items, args.seed, args.max_loops, depth=None)
    tokens = tokens.to(args.device)
    with torch.inference_mode():
        embedding, states = native_trace(model, tokens, args.max_loops)
        parity = verify_native_trace(model, tokens, states, rtol=rtol, atol=atol)
        logits = model.readout(states[1:, :, -1])
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("own-head logits contain nonfinite values")
        predictions = logits.argmax(-1).cpu().numpy()
        token_array = tokens.cpu().numpy()
        permutations = np.empty((args.items, cfg["n_nodes"]), dtype=np.int64)
        edges = token_array[:, :2 * cfg["n_nodes"]].reshape(args.items, cfg["n_nodes"], 2)
        for item in range(args.items):
            permutations[item, edges[item, :, 0]] = edges[item, :, 1]
        starts = token_array[:, -1]
        analyses = [analyze_trajectory(permutations[i], int(starts[i]), predictions[:, i],
                                       horizon=args.horizon, min_visit=args.min_visit,
                                       max_visit=args.max_loops - args.horizon)
                    for i in range(args.items)]
        aggregation = aggregate_trajectories(analyses, unit="item")
        primary = {"name": "conditional_continuation_disagreement",
                   "source_metric": "per_step_mismatch_rate",
                   "aggregation": "unweighted mean over iid generated items with a defined endpoint-gated score",
                   **aggregation["unit_means"]["per_step_mismatch_rate"],
                   "unit": "item", "value_repetition": "independent draws can share the same permutation value"}
        all_rows = [row for analysis in analyses for row in analysis["continuation"]["raw_rows"]]
        pooled_diagnostics = future_diagnostics(all_rows, args.horizon)
        for analysis in analyses:
            analysis["future_diagnostics"] = future_diagnostics(analysis["continuation"]["raw_rows"], args.horizon)
        per_depth = [{"loop": t + 1, "oracle_correct_count": sum(a["per_loop_correct"][t] for a in analyses),
                      "comparisons": args.items,
                      "oracle_accuracy": sum(a["per_loop_correct"][t] for a in analyses) / args.items}
                     for t in range(args.max_loops)]
        selected = select_pairs(analyses, args.max_pairs, args.seed + 1)
        training_loops = int(cfg.get("loops", cfg.get("max_train_depth", 8)))
        pair_results = []
        for index, pair in enumerate(selected):
            expected = analyses[pair["item"]]["oracle_labels"][pair["u"]:pair["u"] + args.horizon]
            result = replay_pair(model, states, embedding, pair, expected,
                                 horizon=args.horizon, seed=args.seed + 2 + index,
                                 rtol=rtol, atol=atol)
            result.update(record_type="intervention_pair", endpoint_gate=True,
                          stratum=_stratum(pair["t"], pair["u"], training_loops))
            pair_results.append(result)
        smoke_control = None
        if args.smoke and not selected:
            pair = {"item": 0, "t": 1, "u": 2,
                    "semantic_label": analyses[0]["oracle_labels"][1]}
            expected = analyses[0]["oracle_labels"][2:2 + args.horizon]
            smoke_control = replay_pair(model, states, embedding, pair, expected,
                                        horizon=args.horizon, seed=args.seed + 2,
                                        rtol=rtol, atol=atol)
            smoke_control.update(record_type="UNTRAINED_SMOKE_UNGATED_CONTROL",
                                 endpoint_gate=False,
                                 interpretation="plumbing only; excluded from scientific pair counts")
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "full_trace.npz"
    np.savez_compressed(trace_path, full_states=states.detach().cpu().numpy(),
                        input_embedding=embedding.detach().cpu().numpy(), tokens=token_array,
                        permutations=permutations, starts=starts, predictions=predictions,
                        oracle=np.asarray([a["oracle_labels"] for a in analyses]).T,
                        own_head_logits=logits.detach().cpu().numpy())
    graph_ids = [_json_hash(p.tolist()) for p in permutations]
    with (run_dir / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for item, analysis in enumerate(analyses):
            for row in analysis["continuation"]["raw_rows"]:
                stream.write(json.dumps({"record_type": "observational_pair", "item": item,
                                         "graph_id": graph_ids[item],
                                         "stratum": _stratum(row["t"], row["u"], training_loops),
                                         **row}, allow_nan=False) + "\n")
        for result in pair_results:
            stream.write(json.dumps({"graph_id": graph_ids[result["item"]], **result}, allow_nan=False) + "\n")
        if smoke_control is not None:
            stream.write(json.dumps(smoke_control, allow_nan=False) + "\n")
    for item, analysis in enumerate(analyses):
        analysis["continuation"].pop("raw_rows")
        analysis["phase"].pop("raw_rows")
        analysis.update(item=item, graph_id=graph_ids[item], permutation=permutations[item].tolist())
    source_paths = [Path(__file__), REPO_ROOT / "research_gym/scripts/loopscope_toy.py",
                    REPO_ROOT / "research_gym/analysis/semantic_echo.py"]
    status = "UNTRAINED_SMOKE" if args.smoke else "MEASURED_CHECKPOINT"
    receipt = {"schema_version": SCHEMA_VERSION, "status": status,
               "scope": "own_head_only", "checkpoint": str(checkpoint) if checkpoint else None,
               "argv": args.invocation_argv, "code_revision": _code_revision(),
               "code_identity": "source_sha256 is authoritative; code_revision is optional context",
               "evidence_stage": "untrained_execution_smoke" if args.smoke else "fixed_checkpoint_measurement",
               "fixed_depth_compatible": cfg["fixed_depth"] is True,
               "oracle_contract": "pi^t(start), fixed graph/start, no requested-depth token",
               "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
               "config": cfg, "config_sha256": _json_hash(cfg),
               "evaluated_state_dict_sha256": _state_dict_hash(model),
               "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in source_paths},
               "utc_created": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "device": args.device, "dtype": args.dtype, "cpu_threads": args.cpu_threads,
               "seed": args.seed, "pair_selection_seed": args.seed + 1,
               "items": args.items, "max_loops": args.max_loops, "horizon": args.horizon,
               "min_visit": args.min_visit, "last_source_visit": args.max_loops - args.horizon,
               "max_intervention_pairs": args.max_pairs,
               "selection": "seeded sample using item/time/oracle equality and endpoint own-head correctness only",
               "external_input": "same fixed graph/start/embedding for every pair and replay",
               "trace_indexing": "full_states[t] is after loop t; full_states[0] is the all-zero initial state",
               "trace_sha256": sha256_file(trace_path), "native_parity": parity,
               "wall_seconds": time.monotonic() - started}
    pooled = _pooled_counts(analyses)
    strata = {}
    for result in pair_results:
        strata[result["stratum"]] = strata.get(result["stratum"], 0) + 1
    summary = {"receipt": receipt, "primary": primary, "item_aggregation": aggregation,
               "pooled_continuation": pooled, "pooled_future_diagnostics": pooled_diagnostics,
               "per_depth_oracle_accuracy": per_depth,
               "items": analyses, "unique_permutation_values": len(set(graph_ids)),
               "selected_intervention_pairs": selected, "selected_pair_strata": strata,
               "intervention_pair_count": len(pair_results),
               "smoke_ungated_control_ran": smoke_control is not None,
               "limits": ["Own-head observations concern the candidate one-hop-per-loop quotient.",
                          "Overlapping visit pairs are dependent; counts are descriptive denominators.",
                          "Full donor replay checks plumbing and time-homogeneous execution.",
                          "Known Q5-Q9 outcomes remain separate; this run fits no predictive cutoff."]}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [f"# Semantic Echo: {status}", "", f"Scope: `{receipt['scope']}`. Device `{args.device}`, dtype `{args.dtype}`.",
             "", "## Primary: mean conditional continuation disagreement", "",
             f"Unweighted item mean: **{primary['mean']}**. Defined for **{primary['units_with_metric']}/{primary['units_total']}** generated items.",
             "Each eligible item receives equal weight. Items are independent generator draws; their permutation values can repeat.",
             f"The run contains {len(set(graph_ids))} unique permutation values across {args.items} item draws.",
             "", f"Native cursor/head parity passed on {parity['comparisons']} visit/item comparisons.",
             f"Retained {args.max_loops + 1} complete token states for each of {args.items} items, including the initial state.",
             "", "## Pooled continuation counts", "", "Pooled pair rates are descriptive companions to the item mean.",
             "", "| Quantity | Value |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in pooled.items()]
    lines += ["", "## Future offsets", "", "Donor is the earlier visit t; receiver is the later visit u.",
              "", "| Offset | Disagreement | Donor oracle accuracy | Receiver oracle accuracy | Pair count |",
              "|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['offset']} | {row['disagreement_rate']} | {row['donor_oracle_accuracy']} | {row['receiver_oracle_accuracy']} | {row['comparisons']} |"
              for row in pooled_diagnostics["per_future_offset"]]
    lines += ["", f"Both complete futures are correct for {pooled_diagnostics['all_h_both_correct_pair_count']}/{pooled_diagnostics['endpoint_gated_pairs']} gated pairs.",
              "", "| Complete-future comparison | Pair count |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in pooled_diagnostics["pair_comparison_cells"].items()]
    lines += ["", "## Accuracy by trace depth", "", "| Loop | Correct items | Items | Accuracy |", "|---:|---:|---:|---:|"]
    lines += [f"| {row['loop']} | {row['oracle_correct_count']} | {row['comparisons']} | {row['oracle_accuracy']} |" for row in per_depth]
    lines += ["", f"Selected {len(pair_results)} endpoint-gated pairs for finite interventions; the cap is {args.max_pairs} per checkpoint.",
              "Selection uses endpoint information only. Every selected pair uses the same fixed future horizon.",
              "", "Full-state donor replay is a positive control. Cursor, edge, LayerNorm-shift and norm-matched random interventions are reported individually in `pairs.jsonl`.",
              "The fixed input embedding remains unchanged during every replay.",
              "", "## Files", "", "- `metrics.json`: configuration, hashes, coverage and per-item metrics.",
              "- `pairs.jsonl`: all structural pairs and selected finite intervention results.",
              "- `full_trace.npz`: all token states, fixed embeddings, tokens, oracle and own-head outputs.",
              "", "Pair counts overlap within graphs. These descriptive counts carry no independence assumption."]
    if args.smoke:
        lines += ["", "This model is randomly initialized. Its output validates execution plumbing and supplies no trained-model finding."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return {"run_directory": run_dir.name, "status": status, "checkpoint_sha256": receipt["checkpoint_sha256"],
            "native_parity": parity["passed"], "eligible_pairs": pooled["eligible_pairs"],
            "endpoint_gated_pairs": pooled["endpoint_gated_pairs"],
            "intervention_pairs": len(pair_results)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", action="append", type=Path, help="Final .pt checkpoint; repeat for an explicit list")
    source.add_argument("--checkpoint-dir", type=Path, help="Read sorted *.final.pt files from this directory only")
    source.add_argument("--smoke", action="store_true", help="Use a tiny UNTRAINED model to validate plumbing")
    parser.add_argument("--outdir", "--out", dest="outdir", required=True, type=Path)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--max-loops", type=int, default=32, help="Total native trace length; source visits stop at max-loops minus horizon")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--min-visit", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=16, help="Global per-checkpoint cap for finite intervention pairs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Allow overwriting this runner's files in an existing output directory")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.invocation_argv = list(sys.argv) if argv is None else [str(Path(__file__).resolve()), *argv]
    if not 1 <= args.items <= 4096:
        parser.error("--items must be in 1..4096")
    if args.horizon < 1 or args.max_loops < args.horizon + 2 or args.max_loops > 512:
        parser.error("--horizon must be positive; --max-loops must be in horizon+2..512")
    if not 1 <= args.min_visit <= args.max_loops - args.horizon:
        parser.error("--min-visit must be within the available source visits")
    if not 1 <= args.max_pairs <= 4096 or args.cpu_threads < 1 or args.seed < 0:
        parser.error("--max-pairs must be 1..4096; --cpu-threads positive; --seed nonnegative")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        parser.error("--device must be cpu, cuda, or cuda:N")
    if args.checkpoint_dir:
        if not args.checkpoint_dir.is_dir():
            parser.error(f"checkpoint directory does not exist: {args.checkpoint_dir}")
        checkpoints = sorted(args.checkpoint_dir.glob("*.final.pt"))
        if not checkpoints:
            parser.error(f"no *.final.pt checkpoints in {args.checkpoint_dir}; supply --checkpoint for another filename")
    else:
        checkpoints = args.checkpoint or [None]
    resolved = [p.resolve() if p is not None else None for p in checkpoints]
    if len(set(resolved)) != len(resolved):
        parser.error("duplicate checkpoint paths are not allowed")
    for path in resolved:
        if path is not None and not path.is_file():
            parser.error(f"checkpoint file does not exist: {path}")
    if args.outdir.exists() and not args.outdir.is_dir():
        parser.error("--outdir exists and is not a directory")
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        parser.error("output directory is nonempty; choose a new --outdir or pass --force")
    try:
        torch = _torch()
        torch.set_num_threads(args.cpu_threads)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch environment has no available CUDA device")
        runs = []
        for index, checkpoint in enumerate(resolved):
            stem = "untrained_smoke" if checkpoint is None else re.sub(r"[^A-Za-z0-9_.-]", "_", checkpoint.stem)
            suffix = "" if checkpoint is None else "_" + sha256_file(checkpoint)[:8]
            run_dir = args.outdir / f"{index:02d}_{stem}{suffix}"
            result = evaluate_one(checkpoint, args, run_dir)
            runs.append(result)
            print(json.dumps(result), flush=True)
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                                               "complete": True, "runs": runs}, indent=2) + "\n")
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Semantic Echo failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
