"""Execute registered GG-05 native SGD gradient/curvature decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from research_gym.interp.goal_guarding_gradient_conflict import run_gg05


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("output/gg05/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg05/artifacts"))
    args=parser.parse_args()
    receipt=run_gg05(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":receipt["protocol"],
        "full_receipt":str(args.out),
        "source":receipt["source_main_commit"],
        "query_count":receipt["query_count"],
        "fresh_panels":{k:v for k,v in receipt["gold_panel_definitions"].items()
                       if v["is_new_scoring_panel"]},
        "first_order_conflict_two_seed_gate":receipt[
            "first_order_conflict_two_seed_gate"],
        "any_curvature_sensitive_seed":receipt["any_curvature_sensitive_seed"],
        "results":[{
            "seed":x["seed"],
            "gold_gradient_cosines":x["gold_gradient_cosines"],
            "fresh_gold_summary":x["fresh_gold_summary"],
            "historical_gold_summary":x["historical_gold_summary"],
            "sgd_steps":x["total_physical_sgd_steps"],
        } for x in receipt["results"]],
    },indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
