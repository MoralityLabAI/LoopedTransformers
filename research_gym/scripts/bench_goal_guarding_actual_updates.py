"""GG-04: physically apply native SGD updates, then fit a disclosure controller.

The controller's feedback table is constructed from actual, reset-to-baseline
SGD updates on the frozen native recurrent task model. The disclosed update
probability predicts later randomized update assignment, which is invisible
to the controller at report time. No optimizer-awareness claim beyond this
synthetic meta-controller setting is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_actual_updates import run_gg04


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/goal_guarding/gg04.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/goal_guarding/gg04_artifacts"))
    args=parser.parse_args()
    result=run_gg04(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "full_report":str(args.out),
        "two_seed_controller_gate":result["two_seed_controller_gate"],
        "results":[{
            "seed":x["seed"],
            "gold_before":x["pre_update_sealed_gold_ce"],
            "real_sgd_steps":x["real_one_step_sgd_runs"],
            "median_calibration_damage":x["calibration_median_damage"],
            "median_heldout_damage":x["heldout_damage_median"],
            "first_order_prediction_mae":x["heldout_first_order_prediction_mae"],
            "cue_effect":x["controller_eval"][
                "truth_probability_high_minus_low"],
            "expected_rewards":x["controller_eval"][
                "exact_q_expected_reward"],
            "controller_gate":x["controller_eval"][
                "controller_registered_gate_pass"],
        } for x in result["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
