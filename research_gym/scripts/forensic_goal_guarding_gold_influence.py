"""GG-05P: post-hoc exhaustive historical gold scoring influence audit.

This consumes the original GG-05 physical-update ledgers and exact native
GG-01 model checkpoint artifacts; the registered GG-05 result is unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from research_gym.interp.goal_guarding_gold_influence_posthoc import run_posthoc_gg05


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,
                        default=Path("output/gg05/checkpoints"))
    parser.add_argument("--out",type=Path,
                        default=Path("output/gg05/posthoc_gold_influence.json"))
    args=parser.parse_args()
    result=run_posthoc_gg05(source_dir=args.source_dir,output=args.out)
    print(json.dumps({
        "protocol":result["protocol"],
        "frozen_primary_unmodified":result["registered_primary_GG05_unchanged"],
        "two_sample_deletions_enumerated":result["exact_two_deletion_pairs"],
        "results":[{
            "seed":entry["seed"],
            "misclassified_examples":entry["baseline_misclassified_examples"],
            "top_two_indices":entry["top_two_baseline_loss_indices"],
            "top_two_baseline_loss_share":entry[
                "top_two_fraction_of_total_baseline_loss"],
            "reversals_before":entry["original_negative_gap_queries"],
            "reversals_after_top_two_deletion":entry[
                "negative_gap_queries_after_removing_hardest_two"],
            "number_pairs_eliminating_all_reversals":entry[
                "full_two_deletion_census"][
                "number_pairs_eliminating_all_negative_gaps"],
        } for entry in result["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
