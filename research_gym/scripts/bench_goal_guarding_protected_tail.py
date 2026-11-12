"""Execute GG-07 protected-tail gold scoring robustness on native SGD.

The registered protected objective combines exact-oracle natural, boundary,
rare and late-reversal gold cases across depths 4, 6 and 8. All gold
sampling and report-gradient steps are fixed, and every post-SGD per-example
outcome is retained in a source-bound experimental artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_protected_tail import run_gg07


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/gg07/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg07/checkpoints"))
    args=parser.parse_args()
    result=run_gg07(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "source_main":result["source_main_commit"],
        "full_receipt":str(args.out),
        "two_seed_strong_protected_tail_gate":result[
            "two_seed_strong_protected_tail_gate"],
        "any_tail_induced_sign_change_event":result[
            "any_tail_induced_sign_change_event"],
        "results":[{
            "seed":row["seed"],
            "gold_accuracy":row["baseline_gold_accuracy"],
            "all_gold_cells_capable":row[
                "all_12_gold_cells_accuracy_ge_90pct"],
            "primary_protected_tail":row["protected_gold"][
                "primary_protected_summary"],
            "tail_induced_nominal_sign_change_queries":row[
                "protected_gold"]["natural_to_protected_any_sign_flip_queries"],
            "actual_sgd_steps":row["actual_native_sgd_steps"],
            "strong_protected_tail_gate":row[
                "registered_strong_protected_tail_sign"],
        } for row in result["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
