"""LSA-02I: exact same-worker CPU Plan B seed training, for search-only arms.

Uses the existing source-verified native shortest-path teacher and the
original pp96 public representation. This does not introduce a new training
objective. The scientific intervention is the *inference-time*, explicitly
costed reconsideration operation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import canonical
from .pretty_train import run as matched_run


SCHEMA = "lsa02i.same_worker_frozen_policy_fit.v1"
SEEDS = (173, 179)
VIEWS = ("pp96",)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--max-seconds-per-fit", type=float, default=300)
    a = p.parse_args(argv)
    report = matched_run(
        a.teacher, a.out, views=VIEWS, seeds=SEEDS,
        steps=a.steps, max_seconds_per_fit=a.max_seconds_per_fit,
        schema=SCHEMA,
    )
    print(canonical({
        "schema": SCHEMA,
        "source_train_roots": report["native_training_roots"],
        "source_frames": report["native_training_frames"],
        "seed_roster": report["seed_roster"],
        "view_roster": report["view_roster"],
        "exact_duplicates": report["all_duplicate_fits_reproduced"],
        "matched_seed_start": report["same_seed_initial_weights"],
        "cells": [
            {"seed": row["seed"], "sha256": row["checkpoint_sha256"],
             "optimizer_steps": row["optimizer_steps"]}
            for row in report["cells"]
        ],
    }))
    return report


if __name__ == "__main__":
    main()
