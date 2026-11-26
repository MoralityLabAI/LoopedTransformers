from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.interp.training import PAUSED, CellConfig, load_pool, run_cell


PAUSED_EXIT_CODE = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume one RMP training cell.")
    parser.add_argument("--arm", choices=("LOOP-T", "FF-U"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--visits", type=int, default=8, help="loop visits; the gate fixes 8")
    parser.add_argument("--init-from", default="", help="warm-start weights from this checkpoint")
    parser.add_argument("--batch-per-family", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--input-injection", action="store_true")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--staircase", action="store_true", help="add each family's staircase-depth pool")
    parser.add_argument("--stop-at-gate", action="store_true", help="stop once C3 exists; resume later for C4")
    parser.add_argument("--permute-targets", action="store_true", help="control model: shuffle each family's targets")
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--result-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--vram-mb", type=float, default=2500.0)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    parser.add_argument("--max-cpu-seconds", type=float, default=None, help="soft per-invocation process CPU budget")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--mms-mib", type=float, default=0)
    parser.add_argument("--mms-mode", choices=("observe", "dense", "bpd", "sham"), default="bpd")
    parser.add_argument("--mms-ratio", type=float, default=.2)
    parser.add_argument("--mms-rank", type=int, default=16)
    parser.add_argument("--mms-topk", type=int, default=4)
    parser.add_argument("--mms-refresh", type=int, default=128)
    parser.add_argument("--mms-max-age", type=int, default=32768)
    parser.add_argument("--mms-visit", type=int, default=-1)
    parser.add_argument("--mms-map", default="")
    parser.add_argument("--mms-checkpoint-every", type=int, default=2000)
    parser.add_argument("--mms-sites", default="mlp", help="comma-separated mlp,mlp_in,attention,qkv")
    parser.add_argument("--mms-schedule", choices=("round_robin","adaptive"), default="round_robin")
    args = parser.parse_args()
    if args.mms_mib and (args.device != "cpu" or args.arm != "LOOP-T"):
        parser.error("MMS v0 requires CPU LOOP-T")
    if args.mms_map and not args.mms_mib:
        parser.error("MMS map requires positive MMS budget")
    from research_gym.interp.mms import file_sha256
    map_sha = file_sha256(args.mms_map) if args.mms_map else ""

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_mb / total_mb), 0)
    receipt = json.loads((args.pool_dir / "pool_receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    used = {name: [name] for name in args.families}
    if args.staircase:
        for name in args.families:
            if f"{name}.staircase" in receipt:
                used[name].append(f"{name}.staircase")
    pools = {}
    for name, keys in used.items():
        parts = [load_pool(args.pool_dir / receipt[key]["path"], receipt[key]["sha256"]) for key in keys]
        pools[name] = parts[0] if len(parts) == 1 else parts[0].merged(parts[1])
    config = CellConfig(
        arm=args.arm,
        seed=args.seed,
        families=tuple(args.families),
        variant=args.variant,
        max_steps=args.max_steps,
        visits=args.visits,
        init_from=args.init_from,
        batch_per_family=args.batch_per_family,
        microbatch=args.microbatch,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        deep_supervision=args.deep_supervision,
        input_injection=args.input_injection,
        permute_targets=args.permute_targets,
        beta=args.beta,
        curriculum=args.curriculum,
        mms_mib=args.mms_mib,
        mms_mode=args.mms_mode,
        mms_ratio=args.mms_ratio,
        mms_rank=args.mms_rank,
        mms_topk=args.mms_topk,
        mms_refresh=args.mms_refresh,
        mms_max_age=args.mms_max_age,
        mms_visit=args.mms_visit,
        mms_map=args.mms_map,
        mms_map_sha=map_sha,
        mms_checkpoint_every=args.mms_checkpoint_every,
        mms_sites=args.mms_sites,
        mms_schedule=args.mms_schedule,
    )
    provenance = {
        "bundle_hash": manifest["bundle_hash"],
        "pools": {
            key: {field: receipt[key][field] for field in ("sha256", "size")}
            for keys in used.values()
            for key in keys
        },
    }
    result = run_cell(
        config,
        pools,
        bundle_dir=args.bundle_dir,
        result_dir=args.result_dir,
        artifact_dir=args.artifact_dir,
        provenance=provenance,
        device=device,
        max_seconds=args.max_seconds,
        max_cpu_seconds=args.max_cpu_seconds,
        stop_at_gate=args.stop_at_gate,
    )
    summary = {key: result[key] for key in ("cell_id", "status", "optimizer_steps", "gate_passed", "events")}
    print(json.dumps(summary, sort_keys=True))
    if result["status"] == PAUSED:
        raise SystemExit(PAUSED_EXIT_CODE)


if __name__ == "__main__":
    main()
