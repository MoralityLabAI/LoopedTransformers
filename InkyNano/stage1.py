"""Run the local Stage 1 routing-residency development sweep."""
import argparse
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from stage1_memory import MemoryGuard, MemoryPressure


def expert_byte_vector(value):
    values = tuple(int(v) for v in value.split(","))
    if len(values) not in (1, 24) or min(values) <= 0:
        raise argparse.ArgumentTypeError("provide one positive byte estimate or 24 layer estimates")
    return values * 24 if len(values) == 1 else values


def run(args, guard):
    import numpy as np
    import torch
    from stage1_controller import (BUDGETS, CAPACITIES, CONTROLLERS, controls,
                                   evaluate, fit_or_restore, predict)
    from stage1_data import ARMS, HORIZONS, SYNTHETIC, prepare, read_json, source_receipt, write_json

    started = perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / "run.json"
    provenance = [source_receipt(source, args.synthetic) for source in args.input]
    training = {field: getattr(args, field) for field in ("seed", "width", "epochs", "batch")}
    spec = {
        "data_kind": SYNTHETIC if args.synthetic else "verified_cached_teacher",
        "input_manifests": provenance, "arms": list(ARMS), "horizons": list(HORIZONS),
        "capacities": list(CAPACITIES), "replacement_budgets": list(BUDGETS),
        "training": training, "expert_bytes_by_physical_layer": list(args.expert_bytes),
        "transfer_model": "one estimated H2D copy per admission; continuation cache only",
        "adaptive": {"drift_threshold": args.drift, "mass_cost_per_64_mib": args.transfer_cost},
        "runtime": {"device": args.device, "torch": str(torch.__version__),
                    "torch_cuda_build": torch.version.cuda, "numpy": np.__version__,
                    "python": platform.python_version(), "system": platform.system(),
                    "threads": args.threads},
    }
    completed = []
    if receipt_path.is_file():
        previous = read_json(receipt_path)
        if previous["spec"] != spec:
            raise ValueError("run settings differ; choose a new --out directory")
        completed = previous["completed"]
        if previous["status"] == "complete":
            print(f"already complete: {args.out}")
            return 0
    elif any(args.out.iterdir()):
        raise ValueError("output directory has no matching run receipt")

    def save(status, reason=None):
        write_json(receipt_path, {"status": status, "reason": reason, "spec": spec,
                                  "completed": completed, "process_memory": guard.receipt(),
                                  "attempt_wall_seconds": perf_counter() - started})

    def run_horizon(arm, h, dataset, frequency, folder):
        folder.mkdir(parents=True, exist_ok=True)
        train = np.array([i for i, w in enumerate(dataset.windows) if w.turn.split == "train"])
        dev = np.array([i for i, w in enumerate(dataset.windows) if w.turn.split == "dev"])
        settings = {**training, "data_kind": spec["data_kind"], "arm": arm, "horizon": h,
                    "input_manifests": [source["manifest_sha256"] for source in provenance]}
        predicted, truth, static_oracle = controls(dataset, dev, frequency)
        model_stats = {}
        for name in CONTROLLERS:
            model, model_stats[name] = fit_or_restore(
                folder / f"{name}.pt", dataset, train, name, settings, device, guard,
            )
            predicted[name] = predict(model, dataset, dev, args.batch, device, guard)
            del model
        opportunities = sum(dataset.windows[i].start != dataset.windows[i].turn.prompt for i in dev)
        budgets = BUDGETS if opportunities else (0,)
        results = []
        for capacity in CAPACITIES:
            ceiling = evaluate(dataset, dev, truth, capacity, capacity, args.expert_bytes, guard)
            results.append({"policy": "hindsight_segment_oracle", "capacity": capacity,
                            "replacement_budget": None, **ceiling})
            for budget in budgets:
                policies = {**predicted}
                policies["hindsight_capped_greedy" if budget else "hindsight_fixed_turn"] = (
                    truth if budget else static_oracle
                )
                if budget:
                    policies["pass_mesh_adaptive"] = predicted["pass_mesh"]
                for name, scores in policies.items():
                    adaptive = (args.drift, args.transfer_cost) if name == "pass_mesh_adaptive" else None
                    result = evaluate(dataset, dev, scores, capacity, budget, args.expert_bytes,
                                      guard, adaptive=adaptive)
                    result["oracle_headroom"] = ceiling["root_mean_mass"] - result["root_mean_mass"]
                    for root, stats in result["roots"].items():
                        stats["oracle_headroom"] = (
                            ceiling["roots"][root]["retained_mass"] - stats["retained_mass"]
                        )
                    results.append({"policy": name, "capacity": capacity,
                                    "replacement_budget": budget, **result})
        split_summary = {}
        for split, indices in (("train", train), ("dev", dev)):
            root_windows = Counter(dataset.windows[i].turn.root for i in indices)
            split_summary[split] = {"n_roots": len(root_windows), "windows": len(indices),
                                    "windows_by_root": dict(root_windows)}
        write_json(folder / "receipt.json", {
            "data_kind": spec["data_kind"], "arm": arm, "horizon": h, "splits": split_summary,
            "dev_refresh_boundaries_per_layer": opportunities, "replacement_budgets_run": list(budgets),
            "models": model_stats, "results": results, "process_memory": guard.receipt(),
        })

    try:
        guard.check()
        if device.type == "cuda":
            guard.configure_cuda(torch, device)
            device = torch.device("cuda", guard.device)
        save("running")
        for arm in ARMS:
            if all(f"{arm}/H{h}" in completed for h in HORIZONS):
                continue
            with TemporaryDirectory(prefix="windows-", dir=args.out) as work:
                datasets, frequency = prepare(
                    args.input, arm, Path(work), teacher=not args.synthetic, guard=guard,
                )
                try:
                    for h in HORIZONS:
                        key = f"{arm}/H{h}"
                        if key in completed:
                            continue
                        run_horizon(arm, h, datasets[h], frequency, args.out / key)
                        completed.append(key)
                        save("running")
                        print(f"completed {key}", flush=True)
                finally:
                    for dataset in datasets.values():
                        dataset.close()
        save("complete")
        return 0
    except (KeyboardInterrupt, MemoryPressure, MemoryError, torch.OutOfMemoryError) as error:
        save("stopped", str(error) or type(error).__name__)
        print(f"Routing gym stopped. Completed results are in {args.out}", file=sys.stderr)
        return 2
    except Exception as error:
        save("failed", str(error))
        raise


def main():
    guard = MemoryGuard()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-pilot", help="verify and extract the pinned routing-only archive")
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    synthetic = commands.add_parser("make-synthetic", help="create a labeled native-schema fixture")
    synthetic.add_argument("out", type=Path)
    synthetic.add_argument("--seed", type=int, default=17)
    sweep = commands.add_parser("run", help="run both arms and all horizons sequentially")
    sweep.add_argument("--input", nargs="+", type=Path, required=True)
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--synthetic", action="store_true")
    sweep.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    sweep.add_argument("--width", type=int, default=32)
    sweep.add_argument("--epochs", type=int, default=20)
    sweep.add_argument("--batch", type=int, default=4)
    sweep.add_argument("--threads", type=int, default=1)
    sweep.add_argument("--seed", type=int, default=17)
    sweep.add_argument("--expert-bytes", type=expert_byte_vector, default=str(3 * 2304 * 1152 // 2))
    sweep.add_argument("--drift", type=float, default=0.002)
    sweep.add_argument("--transfer-cost", type=float, default=0.025)
    args = parser.parse_args()
    guard.check()
    if args.command == "extract-pilot":
        from stage1_data import extract_pilot
        print(extract_pilot(args.archive, args.destination))
    elif args.command == "make-synthetic":
        from stage1_fixture import generate
        generate(args.out, args.seed, guard)
        print(guard.receipt())
    else:
        if (min(args.width, args.epochs, args.batch, args.threads) < 1 or args.width > 128
                or args.batch > 32 or args.seed < 0 or not 0 <= args.drift <= 1
                or not math.isfinite(args.transfer_cost) or args.transfer_cost < 0):
            parser.error("invalid training or refresh settings")
        return run(args, guard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
