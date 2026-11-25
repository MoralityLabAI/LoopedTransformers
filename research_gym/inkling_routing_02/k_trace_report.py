"""Recompute the paired expert-swap contrasts from native development archives."""

import argparse
from pathlib import Path

import numpy as np

from ..inkling_routing_01.checkpoint import file_sha256
from ..inkling_routing_01.local import write_json
from .k_only import ARMS, SMOKE, TREATMENTS, UNCLAMPED, total_variation
from .native import REPLAYS
from .receipts import read_rows, verified_run


def compare_replay(discovery, swap, root, replay, layer, arm):
    filename = f"k_trace_{root}_replay{replay}.npz"
    lanes = ("clean", *ARMS, *UNCLAMPED)
    pair_index = int(arm == "vector")
    clamped, unclamped = (lanes.index(name) for name in (TREATMENTS[pair_index], UNCLAMPED[pair_index]))
    pair = [clamped, unclamped]
    prefix = f"router_hf{layer:02}"
    with (np.load(discovery / filename, allow_pickle=False) as baseline,
          np.load(swap / filename, allow_pickle=False) as counterfactual):
        for key in ("input_ids", "original", "bank3", "delivered", "shared_beta3", f"{prefix}_indices"):
            if not np.array_equal(baseline[key], counterfactual[key]):
                raise ValueError(f"{root} replay {replay} drifted before the swap at {key}")
        native = baseline[f"{prefix}_indices"]
        patched = counterfactual[f"{prefix}_swapped"]
        expected = native.copy()
        membership_changed = set(native[clamped]) != set(native[unclamped])
        if membership_changed:
            expected[pair] = native[pair[::-1]]
        if not np.array_equal(expected, patched):
            raise ValueError(f"{root} replay {replay} delivered unexpected router indices")
        original, final = baseline["logp"], counterfactual["logp"]
        zero = lanes.index("zero_k")
        return {
            "replay": replay, "expert_membership_changed": membership_changed,
            "baseline_vector_pair_tv": total_variation(original[clamped], original[unclamped]),
            "restored_clamped_to_native_unclamped_tv": total_variation(final[clamped], original[unclamped]),
            "forced_unclamped_to_native_clamped_tv": total_variation(final[unclamped], original[clamped]),
            "restored_clamped_vs_original_clamped_tv": total_variation(final[clamped], original[clamped]),
            "forced_unclamped_vs_original_unclamped_tv": total_variation(final[unclamped], original[unclamped]),
            "baseline_clamped_k_ablation_vs_vector_tv": total_variation(original[zero], original[lanes.index("vector_k")]),
            "after_swap_clamped_k_ablation_vs_vector_tv": total_variation(final[zero], final[lanes.index("vector_k")]),
            "max_unmodified_lane_tv": max(total_variation(a, b) for lane, (a, b) in enumerate(zip(original, final))
                                          if lane not in pair),
            "native_top_tokens": original.argmax(axis=-1).tolist(),
            "counterfactual_top_tokens": final.argmax(axis=-1).tolist(),
        }


def analyze(discovery, swap):
    manifests = [verified_run(path)[0] for path in (discovery, swap)]
    if any(m["schema"] != "ink-route-02.k-clamp-downstream-trace.v1" or m["root_prompts"] != SMOKE
           or m["development_only"] is not True or m["sealed_evaluations_accessed"] is not False
           for m in manifests):
        raise ValueError("expected both verified native development traces")
    baseline, intervention = manifests
    if baseline["offload_vocabulary"] != intervention["offload_vocabulary"]:
        raise ValueError("paired trace placements differ")
    layer, arm = intervention["swap_experts_at_hf_layer"], intervention["swap_arm"]
    if layer is None or arm != "vector":
        raise ValueError("expected a downstream vector-lane expert swap")
    stages = read_rows(discovery / "stages.jsonl")
    roots = {}
    for root in SMOKE:
        comparisons = [compare_replay(discovery, swap, root, replay, layer, arm) for replay in range(REPLAYS)]
        switches = {}
        for arm_name in ("ablation", "vector"):
            switches[arm_name] = []
            for replay in range(REPLAYS):
                switches[arm_name].append([
                    row["stage"] for row in stages if row["root_id"] == root and row["replay"] == replay
                    and row.get(f"{arm_name}_expert_set_changed")
                ])
        roots[root] = {"replays": comparisons, "native_router_switches": switches}
    return {
        "development_only": True, "sealed_evaluations_accessed": False, "swap_hf_layer": layer,
        "discovery_manifest_sha256": file_sha256(discovery / "sha256.json"),
        "swap_manifest_sha256": file_sha256(swap / "sha256.json"), "roots": roots,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--swap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, analyze(args.discovery, args.swap))
