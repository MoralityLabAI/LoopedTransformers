"""Reproduce frozen GG-03 finite-support and gradient control experiment.

All trained model weights and selected GG-02 bases are regenerated from
their registered seeds and optimization schedules. This program uses a new
locked source/donor manifest, so GG-02's already-observed locked panel is
never reused as fresh evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_specificity import run_gg03


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/goal_guarding_causal/gg03.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/goal_guarding_causal/gg03_checkpoints"))
    args=parser.parse_args()
    result=run_gg03(output=args.out,artifacts=args.artifact_dir)
    summary={
        "protocol":result["protocol"],
        "new_reporting_pair_identifier_sha256":result[
            "new_reporting_pair_identifier_sha256"],
        "same_statistic_pair_identifier_sha256":result[
            "same_statistic_pair_identifier_sha256"],
        "two_seed_specificity_gate":result["two_seed_specificity_gate"],
        "source_main_commit":result["source_main_commit"],
        "results":[{
            "seed":r["seed"],
            "rank":r["selected_rank"],
            "new_reporting_eligible":r["new_reporting_eligible"],
            "invariant_eligible":r["invariant_eligible"],
            "learned":r["new_reporting"]["learned"],
            "pca":r["new_reporting"]["pca"],
            "output_gradient":r["new_reporting"]["output_gradient"],
            "linear_probe":r["new_reporting"]["linear_probe"],
            "order_swap_learned":r["invariant_order_swap"]["learned"],
            "unique_causal_specificity_gate":r[
                "unique_causal_specificity_gate"],
        } for r in result["results"]],
        "full_report":str(args.out),
    }
    print(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
