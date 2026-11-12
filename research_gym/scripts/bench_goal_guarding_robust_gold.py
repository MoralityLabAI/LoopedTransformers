"""Run frozen GG-06 multi-panel gold robustness and exact finite TV audit.

This executes native SGD on cloned model weights for each truth/false
report and independently evaluates four- and six-visit gold panels.
It reports empirical-support-only TV certificates and baseline-CE
hard-case weighting. No reporting controller is trained or evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_robust_gold import run_gg06


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("output/gg06/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg06/checkpoints"))
    args=parser.parse_args()
    receipt=run_gg06(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":receipt["protocol"],
        "source":receipt["source_main_commit"],
        "full_receipt":str(args.out),
        "two_seed_strong_nominal_stability":receipt[
            "two_seed_strong_nominal_stability"],
        "two_seed_strong_TV_robustness":receipt[
            "two_seed_strong_TV_robustness"],
        "results":[{
            "seed":r["seed"],
            "four_cue_gold":r["four_cue_gold"],
            "six_cue_gold":r["six_cue_gold"],
            "cross_depth_reversals":r["cross_depth_reversals"],
            "actual_physical_sgd_steps":r["actual_physical_sgd_steps"],
        } for r in receipt["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
