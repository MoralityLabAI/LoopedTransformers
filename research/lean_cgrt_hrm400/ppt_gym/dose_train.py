"""LSA-02F fixed 384-byte pretty dosage: one-worker matched CPU fitting.

The four dosages share an exactly identical typed candidate-premise grammar.
Raw legacy JSON acts as a separate input-format anchor on the same fresh data.
Each cell is duplicated on the same physical CPU worker and audited bitwise.
"""
from __future__ import annotations

import argparse
from .data import canonical
from .pretty_train import run as run_matched


SCHEMA = "lsa02f.dosage_same_worker_train.v1"
SEEDS = (137, 139)
VIEWS = ("raw", "pp0", "pp48", "pp96", "pp160")
PRIMARY = ("pp96", "pp0")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--out", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-seconds-per-fit", type=float, default=300)
    args = parser.parse_args(argv)
    receipt = run_matched(
        args.teacher, args.out,
        seeds=tuple(args.seeds), steps=args.steps,
        max_seconds_per_fit=args.max_seconds_per_fit,
        views=VIEWS, schema=SCHEMA,
    )
    print(canonical({
        "schema": receipt["schema"], "native_train_roots": receipt["native_training_roots"],
        "train_frames": receipt["native_training_frames"],
        "same_worker": receipt["same_worker"],
        "all_duplicates_bitwise": receipt["all_duplicate_fits_reproduced"],
        "cells": [
            {"seed": cell["seed"], "view": cell["view"],
             "checkpoint_sha256": cell["checkpoint_sha256"],
             "avg_bytes": cell["mean_public_bytes"]}
            for cell in receipt["cells"]
        ],
    }))
    return receipt


if __name__ == "__main__":
    main()
