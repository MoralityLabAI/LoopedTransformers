"""Fit fixed-budget fixed-target probes on frozen LoopedChase states.

All selections use training/validation graphs. A selection receipt and fitted
weights are saved before held-out test scoring. The smoke actor is untrained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np

from research_gym.analysis.bounded_probes import (
    ProbeSpec, array_hash, evaluate_selected, fixed_target, group_means, json_hash,
    make_graph_splits, scores, select_probes, validate_graph_splits,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract_native_features(model, records, visits, *, device, dtype, batch_size=32):
    """Capture [fixed embedding; full carried state], with a zero-state baseline."""
    import torch
    from research_gym.scripts.loopscope_semantic_echo import native_trace, verify_native_trace
    if not records or visits[0] != 0 or max(visits) < 1:
        raise ValueError("records and positive visits plus visit zero are required")
    collected, native, parity = [], [], []
    tolerance = 1e-8 if dtype == "float64" else 2e-5
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset + batch_size]
            tokens = torch.tensor([row["tokens"] for row in batch], dtype=torch.long, device=device)
            embedding, states = native_trace(model, tokens, max(visits))
            parity.append(verify_native_trace(model, tokens, states, rtol=tolerance, atol=tolerance))
            if not bool((states[0] == 0).all()):
                raise RuntimeError("source model no longer has a zero initial carried state")
            fixed = embedding.flatten(start_dim=1)
            selected = states[list(visits)]
            features = torch.cat([fixed.unsqueeze(0).expand(len(visits), -1, -1),
                                  selected.flatten(start_dim=2)], dim=-1)
            collected.append(features.cpu().double().numpy())
            native.append(model.readout(selected[:, :, -1]).cpu().double().numpy())
    return np.concatenate(collected, axis=1), np.concatenate(native, axis=1), parity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-nodes", type=int, default=8)
    parser.add_argument("--group-counts", type=int, nargs=3, default=[12, 4, 4],
                        metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--items-per-group", type=int, default=16)
    parser.add_argument("--targets", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003, 0.01])
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--families", choices=["affine", "mlp"], nargs="+", default=["affine", "mlp"])
    parser.add_argument("--shuffle-control", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=941117)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-features", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path already exists; use a new run directory to preserve receipts")
    if (any(k not in (8, 16) for k in args.targets) or len(set(args.targets)) != len(args.targets)
            or len(set(args.families)) != len(args.families)):
        parser.error("targets must be distinct members of {8,16}; families must be distinct")
    if min(*args.group_counts, args.items_per_group, args.bootstrap, args.threads,
           args.steps, args.batch_size, args.hidden) < 1:
        parser.error("counts and budgets must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main():
    args = parse_args()
    import torch
    from research_gym.scripts import loopscope_semantic_echo as native_adapter
    from research_gym.scripts.loopscope_toy import LoopedChase
    torch.set_num_threads(args.threads)
    for family in args.families:
        for lr in args.learning_rates:
            ProbeSpec(family, args.hidden, args.steps, args.batch_size, lr,
                      args.weight_decay, args.dtype).validate()
    started = time.monotonic()
    model, cfg = native_adapter.load_model(args.checkpoint, smoke=args.smoke, seed=args.seed,
                                           device=args.device, dtype=args.dtype)
    if args.smoke and args.smoke_nodes != cfg["n_nodes"]:
        if not 4 <= args.smoke_nodes <= 32:
            raise ValueError("--smoke-nodes must be in 4..32")
        torch.manual_seed(args.seed)
        cfg = {**cfg, "n_nodes": args.smoke_nodes}
        model = LoopedChase(cfg["n_nodes"], cfg["d"], cfg["layers"], cfg["heads"])
        model = model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    actor_before = array_hash({key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    splits = make_graph_splits(cfg["n_nodes"], args.group_counts, args.items_per_group, args.seed)
    split_audit = validate_graph_splits(splits)
    visits = [0, 1, 2, 4, 8] + ([16] if 16 in args.targets else [])
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "run_config.json", {"arguments": arguments, "actor_config": cfg,
                                             "planned_visits": visits, "probe_training_only": True})
    write_json(args.out / "split_manifest.json", splits)
    features, native_logits, parity = {}, {}, {}
    for split, records in splits.items():
        features[split], native_logits[split], parity[split] = extract_native_features(
            model, records, visits, device=args.device, dtype=args.dtype)
    features_hash = array_hash(features)
    if args.save_features:
        np.savez_compressed(args.out / "features.npz", **features)
    groups = {name: [row["group_id"] for row in rows] for name, rows in splits.items()}
    selected_studies, selection_receipts, probe_arrays = [], [], {}
    for target in args.targets:
        active_indices = [i for i, visit in enumerate(visits) if visit <= target]
        active_visits = [visits[i] for i in active_indices]
        labels = {name: np.array([fixed_target(row["permutation"], row["start"], target)
                                 for row in rows], dtype=np.int64) for name, rows in splits.items()}
        for family in args.families:
            specs = [ProbeSpec(family, args.hidden, args.steps, args.batch_size, lr,
                               args.weight_decay, args.dtype) for lr in args.learning_rates]
            for shuffled in ([False, True] if args.shuffle_control else [False]):
                for mode in ("separate", "shared"):
                    fit = select_probes(features["train"][active_indices], labels["train"],
                                        features["validation"][active_indices], labels["validation"],
                                        groups["validation"], active_visits, cfg["n_nodes"], specs,
                                        mode=mode, seed=args.seed + target, shuffle_labels=shuffled)
                    index = len(selected_studies)
                    selected_studies.append((target, active_indices, labels, fit))
                    selection_receipts.append({"target": target, **fit.receipt()})
                    for probe_index, probe in enumerate(fit.probes):
                        for key, value in probe.weights.items():
                            probe_arrays[f"study{index}/probe{probe_index}/{key}"] = value
                    probe_arrays[f"study{index}/normalizer/mean"] = fit.normalizer.mean
                    probe_arrays[f"study{index}/normalizer/scale"] = fit.normalizer.scale
    # These files are sealed before the first test prediction or test score.
    write_json(args.out / "selection.json", selection_receipts)
    np.savez_compressed(args.out / "selected_weights.npz", **probe_arrays)
    selection_hash = native_adapter.sha256_file(args.out / "selection.json")
    results = []
    for target, indices, labels, fit in selected_studies:
        scored = evaluate_selected(fit, features["test"][indices], labels["test"], groups["test"],
                                   seed=args.seed + target, bootstrap=args.bootstrap)
        results.append({"target": target, **scored})
    native_results = []
    for target in args.targets:
        target_labels = np.array([fixed_target(row["permutation"], row["start"], target)
                                  for row in splits["test"]], dtype=np.int64)
        for index, visit in enumerate(visits):
            if visit > target:
                continue
            loss, correct = scores(native_logits["test"][index], target_labels)
            native_results.append({"target": target, "visit": visit,
                                   "fixed_target_ce_nats": float(group_means(loss, groups["test"])[1].mean()),
                                   "fixed_target_accuracy": float(group_means(correct.astype(float), groups["test"])[1].mean())})
    actor_after = array_hash({key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    if actor_before != actor_after:
        raise RuntimeError("actor weights changed during the frozen probe study")
    if native_adapter.sha256_file(args.out / "selection.json") != selection_hash:
        raise RuntimeError("selection receipt changed during test evaluation")
    dependencies = {}
    for name, obj in (("model", LoopedChase), ("native_adapter", native_adapter)):
        path = Path(inspect.getfile(obj))
        dependencies[name] = {"path": str(path), "sha256": native_adapter.sha256_file(path)}
    source_paths = [Path(__file__), Path(inspect.getfile(select_probes))]
    result = {"experiment": "RMP-MG-01", "part": "bounded_probes", "schema_version": 1,
              "status": "UNTRAINED_SMOKE" if args.smoke else "FROZEN_CHECKPOINT_DEVELOPMENT",
              "actor_parameters_updated": False, "actor_sha256_before": actor_before,
              "actor_sha256_after": actor_after,
              "checkpoint_sha256": native_adapter.sha256_file(args.checkpoint) if args.checkpoint else None,
              "cfg": cfg, "arguments": arguments, "split_audit": split_audit,
              "sampling_distribution": "uniform_selected_cycle_types_then_random_renaming_start_edge_order",
              "features_sha256": features_hash, "selection_file_sha256": selection_hash,
              "selection_saved_before_test_scoring": True,
              "native_parity": parity, "native_fixed_target_scores": native_results,
              "results": results, "torch": torch.__version__, "python": platform.python_version(),
              "numpy": np.__version__, "dependencies": dependencies,
              "source_sha256": {str(path): native_adapter.sha256_file(path) for path in source_paths},
              "wall_seconds": time.monotonic() - started,
              "trained_Q4_Q5_evaluated": False if args.smoke else "requires_checkpoint_provenance",
              "natural_language_monitor_evaluated": False,
              "test_labels_used_in_selection": False}
    write_json(args.out / "results.json", result)
    lines = ["# RMP-MG-01 bounded-probe receipt", "", f"Status: `{result['status']}`.", "",
             "Actor weights remained frozen. Test scores followed the saved selection receipt.",
             "Groups are disjoint cycle types, including all graph renamings.", "",
             "| K | Family | Decoder | Shuffled train labels | Visit | Gain, nats | 95% group bootstrap |",
             "|---|---|---|---|---|---:|---|"]
    for entry in results:
        row = entry["rows"][-1]
        gain = row["gain_from_input_only"]
        ci = gain["bootstrap_percentile_95"]
        ci_text = "insufficient groups" if ci is None else f"[{ci[0]:.5f}, {ci[1]:.5f}]"
        lines.append(f"| {entry['target']} | {entry['family']} | {entry['mode']} | "
                     f"{entry['shuffled_training_labels']} | {row['visit']} | "
                     f"{gain['mean_gain_nats']:.6f} | {ci_text} |")
    lines.extend(["", "Intervals are exploratory and conditional on this actor and selected structural families.",
                  "Untrained smoke scores exercise fitting and scoring; they do not establish learned reasoning.",
                  "Q4/Q5 evaluation requires the original external checkpoints."])
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "studies": len(results),
                      "split_audit": split_audit, "actor_unchanged": True,
                      "selection_sealed": True, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
