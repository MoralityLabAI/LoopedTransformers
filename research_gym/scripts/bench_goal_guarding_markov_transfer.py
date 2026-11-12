"""GG-09: cross-grammar static Bayes to switching-state HMM transfer.

Reproduces GG-08's native mixed-depth pretrained weights, trains paired
static-versus-HMM continuations under matched visit budgets, and runs
real truth/flip SGD updates on exact new-grammar heldout input identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_gym.interp.goal_guarding_markov_transfer import run_gg09


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("output/gg09/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg09/checkpoints"))
    args=parser.parse_args()
    result=run_gg09(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "source":result["registered_source_main_sha"],
        "report":str(args.out),
        "exact_HMM_vs_static_grammar_label_differences":
            result["input_grammar_mismatch_census"],
        "two_seed_markov_competency_transfer":
            result["two_seed_markov_competency_transfer"],
        "two_seed_markov_protected_harm_transfer":
            result["two_seed_markov_protected_harm_transfer"],
        "results":[{
            "seed":item["seed"],
            "markov_min_accuracy_gain_over_static_control":
                item["markov_min_accuracy_gain_over_static_control"],
            "markov_robust_sign_gain_over_static_control":
                item["markov_robust_sign_gain_over_static_control"],
            "competency_gate":item["registered_markov_competency_transfer_gate"],
            "protected_harm_gate":
                item["registered_markov_protected_harm_transfer_gate"],
            "arms":[{
                "arm":name,
                "minimum_markov_accuracy":row["minimum_markov_accuracy"],
                "capability_eligible":row["markov_capability_eligible"],
                "markov_accuracy":row["markov_accuracy_cells"],
                "static_accuracy_on_same_inputs":
                    row["static_accuracy_on_same_inputs"],
                "robust_signed":row["primary_robust_signed_queries"],
                "robust_census":row["protected_gold"]["primary_protected_summary"],
                "nominal_natural_to_protected_sign_shift_queries":
                    row["protected_gold"][
                        "natural_to_protected_any_sign_flip_queries"],
                "native_physical_sgd_steps":row["native_physical_sgd_steps"],
            } for name,row in item["arms"].items()],
        } for item in result["results"]],
    },indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
