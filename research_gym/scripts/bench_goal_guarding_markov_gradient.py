"""Reproduce GG-09 Markov SGD endpoints, dissect gradients and TV bounds.

Replays two frozen native Markov-trained model seeds and their matched
static control. Historical GG-09 gold panels are post-hoc mechanism
reconstructions. Fresh heldout HMM scoring panels use frozen new seeds,
with the same physical weight updates as historical scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_markov_gradient import run_gg10


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/gg10/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg10/checkpoints"))
    parser.add_argument("--original-gg09-source-dir",type=Path,
                        default=None,
                        help="Directory with original SHA-pinned GG-09 model checkpoints")
    args=parser.parse_args()
    result=run_gg10(output=args.out,artifacts=args.artifact_dir,
                    original_gg09_source_dir=args.original_gg09_source_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "full_receipt":str(args.out),
        "two_seed_nominal_first_order_gate":result[
            "two_seed_nominal_first_order_gate"],
        "any_residual_sensitive_markov_seed":result[
            "any_residual_sensitive_markov_seed"],
        "two_seed_unresolved_replication_event":result[
            "two_seed_unresolved_replication_event"],
        "results":[{
            "seed":row["seed"],
            "markov":row["arms"]["markov_transfer"]["summary"],
            "static":row["arms"]["static_control"]["summary"],
            "physical_update_steps":{
                k:v["physical_post_report_sgd_steps"]
                for k,v in row["arms"].items()
            },
        } for row in result["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
