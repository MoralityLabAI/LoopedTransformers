"""Run GG-01 supervised evaluation-belief looped-transformer causal pilot.

CPU-only. No pretrained weights or external model downloads. Use --smoke
for an implementation check; only the registered full 400-step, 2-seed
run is eligible for a GG-01 empirical result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from research_gym.interp.goal_guarding_causal import run_pilot, save_receipt


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out",type=Path,
                        default=Path("output/goal_guarding_causal/gg01.json"))
    parser.add_argument("--checkpoint-dir",type=Path,
                        default=Path("output/goal_guarding_causal/checkpoints"))
    args=parser.parse_args()
    steps=20 if args.smoke else 400
    seeds=(11,) if args.smoke else (11,17)
    result=run_pilot(steps=steps,seeds=seeds,
                     save_dir=args.checkpoint_dir,
                     include_shuffled_control=not args.smoke)
    result["registered_full_run"]=not args.smoke
    result["run_label"]="GG01_SMOKE_ONLY" if args.smoke else "GG01_SUPERVISED_CPU_PILOT"
    result["artifacts"]="local CPU experiment only; no Q4/Q5 checkpoint reuse"
    save_receipt(result,args.out)
    printable={
        "run_label":result["run_label"],
        "receipt_path":str(args.out),
        "all_seeds_eligible":result["all_seeds_eligible"],
        "all_seeds_rank_one_supported":result["all_seeds_rank_one_supported"],
        "seeds":[
          {"seed":row["training"]["seed"],
           "heldout_4_accuracy":row["training"]["heldout_4_accuracy"],
           "heldout_6_accuracy":row["training"]["heldout_6_accuracy"],
           "probe_validation_r2":row["probe"]["validation_r2"],
           "causal_hit_rate":row["causal_intervention"].get("rank_one",{}).get("hit"),
           "eligible_pairs":row["causal_intervention"].get("eligible"),
          } for row in result["results"]
        ]
    }
    print(json.dumps(printable,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
