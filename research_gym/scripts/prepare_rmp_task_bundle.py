from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.bundle import BundleConfig, materialize_bundle
from research_gym.interp.tasks import SEQUENCE_LENGTH, VOCABULARY_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the RMP v0 trace-annotated task bundle.")
    parser.add_argument("--config", type=Path, default=Path("configs/rmp_task_bundle_v0.json"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets")
    )
    parser.add_argument(
        "--write",
        nargs="*",
        default=None,
        help="splits to write to disk; every split is generated and hashed regardless",
    )
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if int(raw["sequence_length"]) != SEQUENCE_LENGTH or int(raw["vocabulary_size"]) != VOCABULARY_SIZE:
        raise ValueError("config shape does not match the task module")
    manifest = materialize_bundle(BundleConfig.from_mapping(raw), args.out, write_splits=args.write)
    summary = {
        name: {
            split: {key: shard[key] for key in ("base_rows", "sibling_rows", "shortfall", "written")}
            for split, shard in shards.items()
        }
        for name, shards in manifest["families"].items()
    }
    print(json.dumps({"bundle_hash": manifest["bundle_hash"], "families": summary}, indent=2))


if __name__ == "__main__":
    main()
