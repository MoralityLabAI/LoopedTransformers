"""SUD-02: native LoopedChase semantic noise audit on hash-gated Q5 checkpoints.

A graph's permutation edges stay intact. The neural prompt contains only the
corrupted start y. The original uniform start x is never supplied to the model.

For each graph, 16 native forwards for all possible y suffice. Marginalizing
all clean x and noisy y against exact BSC weights gives complete metrics
without noisy Monte Carlo pair sampling. Q5 weights remain external.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from research_gym.analysis.semantic_noise import (auxiliary_memory_lower_bound,
    evaluate_channel, reversible_echo_control)

ROOT = Path("experiments/recurrence_monitorability_v0")
SCHEMA_VERSION = "semantic_noise.sud02.v1"
SEED = 940211


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for buf in iter(lambda: src.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def _save(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _check_directory(path: Path, force: bool):
    if path.is_file():
        raise ValueError("output path is a file")
    if path.is_dir() and any(path.iterdir()) and not force:
        raise ValueError("output directory is nonempty; choose a new --out or pass --force")
    path.mkdir(parents=True, exist_ok=True)


def _q5_preflight(args) -> list[dict]:
    from research_gym.scripts.loopscope_mg_cohort import inventory

    spec = SimpleNamespace(cohorts=["q5"], receipt_root=args.receipt_root,
                           q4_dir=Path("_unused_q4"), q5_dir=args.q5_dir)
    records = inventory(spec)
    if len(records) != 12:
        raise RuntimeError("expected exactly 12 Q5 training receipts")
    selection = set(args.arms or [row["arm"] for row in records])
    available = {row["arm"] for row in records}
    if selection - available:
        raise ValueError("unknown Q5 arms: " + ", ".join(sorted(selection - available)))
    return [row for row in records if row["arm"] in selection]


def _graphs(n: int, count: int, seed: int):
    rng = random.Random(seed)
    for graph_idx in range(count):
        permutation = list(range(n))
        order = list(range(n))
        rng.shuffle(permutation)
        rng.shuffle(order)
        yield graph_idx, permutation, order


def _tokens_for_all_starts(n: int, permutation: list[int],
                           edge_order: list[int], torch, device: str):
    edges = [node_token for src in edge_order
             for node_token in (src, permutation[src])]
    # No clean x anywhere in the model's prompt. All 2^n noisy starts are run.
    return torch.tensor([edges + [y] for y in range(n)],
                        dtype=torch.long, device=device)


def _aggregate(records: list[dict], depths: list[int]) -> list[dict]:
    grouped = []
    for depth in depths:
        rows = [row for row in records if row["depth"] == depth]
        if not rows:
            raise RuntimeError("missing depth in per-graph results")
        avg = lambda key: sum(row[key] for row in rows) / len(rows)
        result = {
            "depth": depth,
            "graph_units": len(rows),
            "mean_exact_source_information_bits": avg("source_information_bits"),
            "mean_exact_source_bayes_error": avg("source_bayes_error"),
            "mean_source_dimension_iid_episode_reference": avg("iid_episode_dimension_reference"),
            "mean_walsh_degree1_mass": avg("degree1_mass"),
            "mean_native_bit_information_bits": avg("native_bit_information_bits"),
            "mean_native_head_information_loss_bits": avg("native_head_information_loss_bits"),
            "mean_native_clean_cursor_accuracy": avg("native_clean_cursor_accuracy"),
            "mean_native_clean_bit_accuracy": avg("native_clean_bit_accuracy"),
            "mean_native_noisy_target_bit_accuracy": avg("native_noisy_target_bit_accuracy"),
            "mean_oracle_noisy_target_bit_accuracy": avg("oracle_noisy_target_bit_accuracy"),
            "mean_bayes_target_bit_accuracy": avg("bayes_target_bit_accuracy"),
        }
        # Each sampled graph/start/noise episode is independently generated.
        # Conditional on the graph list, the Fano side-memory bound applies.
        source_h = [1.0 - row["source_information_bits"] for row in rows]
        native_h = [1.0 - row["native_bit_information_bits"] for row in rows]
        result["independent_graph_episode_memory_bound"] = {
            "episode_count": len(rows),
            "allowed_block_error": 0.01,
            "source_side_memory_lower_bound_bits":
                auxiliary_memory_lower_bound(source_h, .01),
            "native_head_side_memory_lower_bound_bits":
                auxiliary_memory_lower_bound(native_h, .01),
            "exact_sum_head_information_loss_bits":
                sum(row["native_head_information_loss_bits"] for row in rows),
        }
        grouped.append(result)
    return grouped


def _model_rows(model, cfg, args, torch) -> tuple[list[dict], dict]:
    from research_gym.scripts.loopscope_semantic_echo import (
        native_trace, verify_native_trace,
    )

    n = cfg["n_nodes"]
    if n & (n - 1) or not (2 <= n <= 16):
        raise ValueError("Q5 bridge requires a power-of-two node alphabet of size 2..16")
    if args.bit >= n.bit_length() - 1:
        raise ValueError("selected semantic bit is beyond the native node register")
    depth_max = max(args.depths)
    rows = []
    parity = None
    with torch.inference_mode():
        for idx, permutation, order in _graphs(n, args.graphs, args.seed):
            inputs = _tokens_for_all_starts(n, permutation, order, torch, args.device)
            # Model.forward recomputes the entire state for each noisy start.
            states = model(inputs, depth_max)
            logits = model.readout(states)
            if tuple(logits.shape) != (depth_max, n, n):
                raise RuntimeError("unexpected native readout tensor shape")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite native logits")
            predicted_nodes = logits.argmax(dim=-1).cpu().tolist()
            if idx == 0:
                _, full_states = native_trace(model, inputs, depth_max)
                parity = verify_native_trace(
                    model, inputs, full_states,
                    rtol=2e-5 if args.dtype == "float32" else 1e-10,
                    atol=2e-6 if args.dtype == "float32" else 1e-12)
                if not parity.get("passed"):
                    raise RuntimeError("native LoopedChase and explicit trace disagree")
            for depth in args.depths:
                m = evaluate_channel(permutation, depth, args.p, bit=args.bit,
                                     ratio=args.ratio,
                                     model_nodes_by_y=predicted_nodes[depth - 1])
                bit_head = m["native_head"]
                rows.append({
                    "graph_idx": idx,
                    "depth": depth,
                    "permutation": permutation,
                    "edge_order": order,
                    "source_information_bits": m["source_information_bits"],
                    "source_bayes_error": m["source_bayes_error"],
                    "iid_episode_dimension_reference": m["iid_episode_dimension_reference"],
                    "degree1_mass": m["spectral_mass_by_degree"][1],
                    "spectral_mass_by_degree": m["spectral_mass_by_degree"],
                    "native_bit_information_bits": bit_head["mutual_information_bits"],
                    "native_head_information_loss_bits": bit_head["additional_conditional_entropy_bits"],
                    "native_clean_cursor_accuracy": bit_head["clean_native_cursor_accuracy"],
                    "native_clean_bit_accuracy": bit_head["clean_semantic_bit_accuracy"],
                    "native_noisy_target_bit_accuracy": bit_head["accuracy_vs_clean_target"],
                    "oracle_noisy_target_bit_accuracy": m["raw_corrupted_oracle_bit"]["accuracy_vs_clean_target"],
                    "bayes_target_bit_accuracy": m["bayes_optimal_bit"]["accuracy_vs_clean_target"],
                })
    return rows, parity


def _evaluate(checkpoint: Path | None, row: dict | None, args) -> dict:
    from research_gym.scripts.loopscope_semantic_echo import _torch, load_model

    torch = _torch()
    torch.set_num_threads(args.threads)
    model, cfg = load_model(checkpoint, smoke=checkpoint is None, seed=args.seed,
                            device=args.device, dtype=args.dtype)
    rows, parity = _model_rows(model, cfg, args, torch)
    report = {
        "status": "UNTRAINED_SMOKE" if checkpoint is None else "Q5_TRAINED_DEVELOPMENT",
        "arm": "untrained_smoke" if row is None else row["arm"],
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else _sha(checkpoint),
        "expected_checkpoint_sha256": None if row is None else row["expected_sha256"],
        "training_receipt_sha256": None if row is None else row["receipt_sha256"],
        "model_cfg": cfg,
        "source": {"exact_pairs_per_graph": cfg["n_nodes"] ** 2,
                   "graph_units": args.graphs,
                   "clean_start_prior": "uniform",
                   "noise": "independent BSC across binary start bits",
                   "clean_start_reaches_model": False},
        "native_trace_parity": parity,
        "by_depth": _aggregate(rows, args.depths),
    }
    return report, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--controls", action="store_true", help="Run exact four-node reversible echo only")
    source.add_argument("--smoke", action="store_true", help="Untrained native LoopedChase plumbing audit")
    source.add_argument("--q5-dir", type=Path, help="Directory of original hash-verified Q5 .final.pt weights")
    parser.add_argument("--receipt-root", type=Path, default=ROOT)
    parser.add_argument("--arms", nargs="+", default=None, help="Q5 arm names, e.g. lam0_s4 lam0.01_s4")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly use only hash-valid selected Q5 arms")
    parser.add_argument("--graphs", type=int, default=32, help="Independent graph draws per checkpoint")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--p", type=float, default=.1, help="BSC input-bit flip probability")
    parser.add_argument("--bit", type=int, default=0, help="Bit of clean successor's integer label")
    parser.add_argument("--ratio", type=float, default=1 / 3, help="Reference Cantor encoding ratio")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if not (1 <= args.graphs <= 256) or args.seed < 0 or args.threads < 1:
        parser.error("--graphs must be 1..256; seed nonnegative; threads positive")
    if (not args.depths or min(args.depths) < 1 or max(args.depths) > 64 or
            len(args.depths) != len(set(args.depths)) or args.bit < 0 or
            not 0 <= args.p <= .5 or not 0 < args.ratio < .5):
        parser.error("invalid depths, semantic bit, BSC p or Cantor ratio")
    if args.device != "cpu" and not (args.device == "cuda" or (
            args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("--device must be cpu, cuda or cuda:N")
    if args.controls and (args.arms or args.allow_partial):
        parser.error("--controls cannot select trained checkpoints")
    if args.smoke and (args.arms or args.allow_partial):
        parser.error("--smoke cannot select trained checkpoints")
    if args.q5_dir is None and args.threads != 4 and args.controls:
        parser.error("--controls is independent of native threads")
    try:
        _check_directory(args.out, args.force)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "STARTED",
            "seed": args.seed,
            "graphs": args.graphs,
            "depths": args.depths,
            "noise_p": args.p,
            "semantic_bit": args.bit,
            "cantor_ratio": args.ratio,
            "repeated_loop_trajectory_dimension_claim": False,
            "git_source_files": {
                "math": _sha(Path(__file__).resolve().parents[1] / "analysis" / "semantic_noise.py"),
                "runner": _sha(Path(__file__).resolve()),
            },
        }
        if args.controls:
            control = reversible_echo_control(args.p, args.ratio)
            manifest.update(status="CONTROL_ONLY", results=[control], complete=True)
            _save(args.out / "manifest.json", manifest)
            print(json.dumps({"status": "CONTROL_ONLY", "out": str(args.out),
                              "period": 3, "depths": len(control["depths"])}))
            return 0
        if args.q5_dir is not None:
            rows = _q5_preflight(args)
            _save(args.out / "checkpoint_inventory.json", rows)
            blocked = [row for row in rows if row["status"] != "READY"]
            if blocked and not args.allow_partial:
                manifest.update(status="BLOCKED_CHECKPOINTS", complete=False,
                                blocked=[{"arm": r["arm"], "status": r["status"]}
                                         for r in blocked])
                _save(args.out / "manifest.json", manifest)
                print("Original Q5 trained weights missing or hash-mismatched. "
                      "See checkpoint_inventory.json. No substitute model was run.",
                      file=sys.stderr)
                return 2
            rows = [r for r in rows if r["status"] == "READY"]
            if not rows:
                raise RuntimeError("no hash-valid trained Q5 checkpoints to evaluate")
        else:
            rows = [None]
        outputs = []
        for number, row in enumerate(rows):
            checkpoint = None if row is None else Path(row["checkpoint"])
            report, graph_rows = _evaluate(checkpoint, row, args)
            stem = "untrained_smoke" if row is None else row["arm"]
            folder = args.out / f"{number:02d}_{stem}"
            _save(folder / "report.json", report)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "graphs.jsonl").open("w", encoding="utf-8") as target:
                for record in graph_rows:
                    target.write(json.dumps(record, allow_nan=False) + "\n")
            outputs.append({"arm": report["arm"], "status": report["status"],
                            "checkpoint_sha256": report["checkpoint_sha256"],
                            "report": str(folder / "report.json")})
            manifest.update(status="PARTIAL", complete=False, results=outputs)
            _save(args.out / "manifest.json", manifest)
        manifest.update(
            status="UNTRAINED_SMOKE" if args.smoke else "Q5_TRAINED_DEVELOPMENT",
            complete=True)
        _save(args.out / "manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "complete": True,
                          "models": len(outputs), "out": str(args.out)}))
        return 0
    except (ValueError, RuntimeError, OSError) as e:
        print(f"SUD-02 failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
