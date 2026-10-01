from __future__ import annotations

import argparse
import json
from multiprocessing import Pool as ProcessPool
from pathlib import Path

from research_gym.interp.bundle import BundleConfig
from research_gym.interp.training import pool_from_rows, pool_rows, save_pool
from lsa.canonical import digest


def _chunk(arguments: tuple[str, int, int, dict, bool]) -> list[list[int]]:
    family, start, stop, raw, staircase = arguments
    return pool_rows(family, start, stop, BundleConfig.from_mapping(raw), staircase=staircase)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the first N training-stream examples per family.")
    parser.add_argument("--config", type=Path, default=Path("configs/rmp_task_bundle_v0.json"))
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--staircase",
        action="store_true",
        help="generate each family's easier-than-evaluated depths instead, as <family>.staircase",
    )
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = BundleConfig.from_mapping(raw)
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / "pool_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    with ProcessPool(args.workers) as workers:
        for family in args.families:
            key = f"{family}.staircase" if args.staircase else family
            jobs = [
                (family, start, min(start + args.chunk, args.size), raw, args.staircase)
                for start in range(0, args.size, args.chunk)
            ]
            rows = [row for chunk in workers.imap(_chunk, jobs) for row in chunk]
            receipt[key] = {
                **save_pool(pool_from_rows(rows), args.out / f"{key}.pt"),
                "config_sha256": digest(config.to_dict()),
            }
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
            )
            print(json.dumps({key: receipt[key]}))


if __name__ == "__main__":
    main()
