"""Run registered GG-04B with depth-aligned protected gold and real SGD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from research_gym.interp.goal_guarding_actual_updates_b import run_gg04b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/goal_guarding/gg04b.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/goal_guarding/gg04b_artifacts"))
    args=parser.parse_args()
    result=run_gg04b(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "full_report":str(args.out),
        "two_seed_controller_gate":result["two_seed_controller_gate"],
        "results":[{
            "seed":r["seed"],
            "four_cue_calibration_damage_median":
                r["calibration_median_positive_damage"],
            "four_cue_heldout_damage_median":
                r["heldout_median_raw_damage"],
            "cross_depth_heldout":r["cross_depth_heldout"],
            "real_sgd_steps_per_seed":r["real_sgd_steps_per_seed"],
            "truth_probability_high_minus_low":
                r["controller_eval"]["truth_probability_high_minus_low"],
            "expected_rewards":r["controller_eval"][
                "exact_q_expected_reward"],
            "registered_gate":r["controller_eval"][
                "controller_registered_gate_pass"],
        } for r in result["results"]],
    },indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
