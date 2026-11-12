"""GG-08: matched-visit depth curriculum and protected-tail native SGD audit.

Trains two paired continuations of each frozen GG-01 model, with matched
optimizer steps and total unrolled sample-visits, followed by new exact-
oracle gold panels and original GG-07 empirical robustness scoring.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from research_gym.interp.goal_guarding_competency_recovery import run_gg08


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,
                        default=Path("output/gg08/receipt.json"))
    parser.add_argument("--artifact-dir",type=Path,
                        default=Path("output/gg08/checkpoints"))
    args=parser.parse_args()
    receipt=run_gg08(output=args.out,artifacts=args.artifact_dir)
    print(json.dumps({
        "protocol":receipt["protocol"],
        "source":receipt["source_main_commit"],
        "receipt":str(args.out),
        "both_seeds_competency_recovered":receipt[
            "both_seeds_competency_recovered"],
        "both_seeds_robust_harm_recovery":receipt[
            "both_seeds_robust_harm_recovery"],
        "results":[{
            "seed":r["seed"],
            "arms":[{
                "arm":arm,
                "accuracy_minimum":record["accuracy_minimum"],
                "accuracy_cells":record["accuracy_cells"],
                "capability_gate":record["capability_gate"],
                "primary_robust_signed_queries":record[
                    "primary_robust_signed_queries"],
                "primary_robust_census":record[
                    "protected_robust_census"]["primary_protected_summary"],
                "natural_to_protected_sign_flip_queries":record[
                    "protected_robust_census"][
                    "natural_to_protected_any_sign_flip_queries"],
                "model_sha256":record["model_sha256"],
            } for arm,record in r["arms"].items()],
            "competency_recovered":r["mixed_competency_recovered"],
            "robust_sign_gain_vs_four_only":r[
                "mixed_robust_sign_gain_over_control"],
            "registered_robust_harm_recovery_gate":r[
                "registered_robust_harm_recovery_gate"],
        } for r in receipt["results"]],
    },sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
