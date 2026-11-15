"""NRF-08: frozen-state suffix-depth experiment on NRF-07 ToyLoop models.

Strict replay halts if the regenerated checkpoint .pt SHA-256 is different.
Use --exploratory to generate separately labeled numerical reconstructions
after the registered byte-identity gate fails. The new study is exploratory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def complete_median(arms: dict, lam: float = 1.0) -> float:
    if lam < 0 or not all(f"random_{i}" in arms for i in range(8)):
        raise ValueError("eight random controls and lambda>=0 required")
    scores = sorted(arms[f"random_{i}"]["heldout_mse"] +
                    lam * arms[f"random_{i}"]["local_fiber_leakage"] ** 2
                    for i in range(8))
    return (scores[3] + scores[4])/2

def frozen_suffix_evaluation(snapshot: dict, suffix_lengths=(8, 16)) -> dict:
    import torch
    from research_gym.analysis.nrf07_independent_seed_train import gaussian_inputs
    from research_gym.analysis.nrf04_semantic_fiber import (
        blocks, half_states, supervised_basis, random_basis, coordinate_basis,
        ridge_head, heldout_mse, leakage_at,
    )
    T, seed = snapshot["rounds"], snapshot["seed"]
    state = snapshot["state_dict"]
    weights = blocks(tuple(state[key] for key in (
        "blocks.0.in_proj.weight", "blocks.0.in_proj.bias",
        "blocks.0.out_proj.weight", "blocks.0.out_proj.bias")))
    def dataset(fold, n):
        x, y = gaussian_inputs(seed, fold, n)
        return half_states(weights, x, T), y
    with torch.no_grad():
        train_h, train_y = dataset(1, 160)
        test_h, test_y = dataset(5, 96)
        shadow_h, _ = dataset(6, 12)
        mean_h, mean_y = train_h.mean(0), train_y.mean(0)
        bases = {
            "supervised": supervised_basis(train_h, train_y, mean_h, mean_y),
            "label_shifted_train_null": supervised_basis(
                train_h, train_y, mean_h, mean_y, shifted=True),
            "first_eight_coordinates": coordinate_basis(),
        }
        for i in range(8):
            bases[f"random_{i}"] = random_basis(400000 + 31*i + 17*seed)
        errors = {}
        for name, basis in bases.items():
            head = ridge_head(train_h, train_y, basis, mean_h, mean_y)
            errors[name] = heldout_mse(
                test_h, test_y, basis, mean_h, mean_y, head)
        rows = []
        for suffix in suffix_lengths:
            arms = {}
            for name, basis in bases.items():
                leakage = sum(leakage_at(weights, h, suffix, basis)
                              for h in shadow_h)/len(shadow_h)
                mse = errors[name]
                arms[name] = {
                    "heldout_mse": mse, "local_fiber_leakage": leakage,
                    "lambda_one_objective": mse + leakage*leakage,
                }
            sup = arms["supervised"]["lambda_one_objective"]
            rand = complete_median(arms)
            rows.append({
                "suffix_visits": suffix, "arms": arms,
                "supervised_objective": sup,
                "random_median_objective": rand,
                "supervised_wins": sup < rand,
            })
    return {
        "id": f"R{T}_S{seed}_E{snapshot['exposure']}",
        "trained_rounds": T, "seed": seed,
        "exposure": snapshot["exposure"],
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "suffix_rows": rows,
    }

def verify_original_suffix(replay: dict, manifest: dict, tolerance=1e-12):
    T, seed = replay["trained_rounds"], replay["seed"]
    key = f"R{T}_S{seed}_E4096"
    source = next((r for r in manifest["rows"] if r["id"] == key), None)
    if source is None or replay["exposure"] != 4096:
        raise ValueError("NRF-07 reference cell missing")
    base = next(r for r in replay["suffix_rows"]
                if r["suffix_visits"] == T//2)
    errors = {
        "supervised_objective": abs(base["supervised_objective"] -
                                    source["supervised_objective"]),
        "random_median_objective": abs(base["random_median_objective"] -
                                       source["random_median_objective"]),
        "supervised_MSE": abs(base["arms"]["supervised"]["heldout_mse"] -
                              source["supervised_MSE"]),
        "supervised_leakage": abs(base["arms"]["supervised"]["local_fiber_leakage"] -
                                  source["supervised_leakage"]),
    }
    return {
        "expected_checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_hash_match": replay["checkpoint_sha256"] == source["checkpoint_sha256"],
        "numerical_absolute_errors": errors,
        "baseline_numerically_matches": max(errors.values()) <= tolerance,
    }

def run_one(seed: int, trained_rounds: int, outdir: Path, manifest: dict,
            *, exploratory: bool = False):
    import torch
    from research_gym.analysis.nrf07_independent_seed_train import train_one
    torch.set_num_threads(1)
    snapshots = train_one(seed, trained_rounds, outdir/"checkpoints")
    final = snapshots[-1]
    if final["exposure"] != 4096:
        raise AssertionError("missing final checkpoint")
    original = next(r for r in manifest["rows"]
                    if r["id"] == f"R{trained_rounds}_S{seed}_E4096")
    matching = final["checkpoint_sha256"] == original["checkpoint_sha256"]
    if not matching and not exploratory:
        blocker = {
            "study_id": "NRF-08-frozen-state-suffix-depth-intervention-v1",
            "status": "BLOCKED_STRICT_HASH_GATE",
            "expected_checkpoint_sha256": original["checkpoint_sha256"],
            "replayed_checkpoint_sha256": final["checkpoint_sha256"],
        }
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir/f"nrf08_strict_blocked_R{trained_rounds}_S{seed}.json").write_text(
            json.dumps(blocker,indent=2,sort_keys=True)+"\n")
        raise RuntimeError("NRF-08 strict SHA mismatch; --exploratory labels a separate diagnostic")
    replay = frozen_suffix_evaluation(final)
    verification = verify_original_suffix(replay, manifest)
    if not verification["baseline_numerically_matches"]:
        raise AssertionError("original NRF-07 numerical baseline mismatch")
    replay["verification"] = verification
    replay["registration_status"] = (
        "STRICT_HASH_REPLAY_MATCHED" if matching
        else "EXPLORATORY_AFTER_STRICT_GATE_BLOCKER")
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/f"nrf08_R{trained_rounds}_S{seed}.json").write_text(
        json.dumps(replay,indent=2,sort_keys=True)+"\n")
    return replay

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=(109,113,127),required=True)
    parser.add_argument("--trained-rounds",type=int,choices=(16,32),required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--exploratory",action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root/
        "experiments/numerical_range_fixed_point_v1/nrf07_independent_seed_summary.json").read_text())
    result = run_one(args.seed,args.trained_rounds,args.outdir,manifest,
                     exploratory=args.exploratory)
    print(result["id"],result["registration_status"],
          [(r["suffix_visits"],r["supervised_wins"]) for r in result["suffix_rows"]])

if __name__ == "__main__":
    main()
