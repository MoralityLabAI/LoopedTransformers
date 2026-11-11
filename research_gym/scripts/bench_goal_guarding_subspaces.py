"""Run GG-02, the locked causal rank-2/4/8 belief-subspace successor to GG-01.

The base model is explicitly supervised in GG-01. GG-02 trains only one
interchange projector at a time, on calibration counterfactual pairs.
Holdout locked results have no role in rank selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from research_gym.interp.goal_guarding_subspace import run_gg02


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
        default=Path("output/goal_guarding_causal/gg02.json"))
    parser.add_argument("--checkpoint-dir",type=Path,
        default=Path("output/goal_guarding_causal/gg02_checkpoints"))
    args=parser.parse_args()
    receipt=run_gg02(output=args.out,checkpoints=args.checkpoint_dir)
    print(json.dumps({
        "protocol":receipt["protocol"],
        "report":str(args.out),
        "split_sizes":receipt["split_sizes"],
        "two_seed_causal_gate":receipt["two_seed_causal_gate"],
        "seeds":[{
            "seed":row["seed"],
            "GG01_six_cue_accuracy":row["base_training"]["heldout_6_accuracy"],
            "calibration_eligible":row["calibration_eligible"],
            "validation_eligible":row["validation_eligible"],
            "locked_eligible":row["locked_eligible"],
            "selected_rank":row["selected_rank_from_validation"],
            "selected_locked":row["locked_rank_metrics"][
                str(row["selected_rank_from_validation"])],
            "locked_sham":row["locked_sham"],
            "GG01_rank_one":row["locked_gg01_linear_probe"],
            "rank_passed":row["selected_rank_causal_gate"],
        } for row in receipt["results"]],
    },sort_keys=True,indent=2))


if __name__=="__main__":
    main()
