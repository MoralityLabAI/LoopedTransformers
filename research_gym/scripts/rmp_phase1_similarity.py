"""Phase 1, E3: representation similarity across visits, seeds and arms, and block Jacobian spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.analysis import load_analysis_model, with_injection
from research_gym.interp.bundle import read_shard
from research_gym.interp.probes import answer_states, rms_normalize
from research_gym.interp.similarity import (
    block_spectral_norms,
    cka_matrix,
    debiased_cka,
    monotone_path,
    procrustes_similarity,
)


FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")
EXAMPLES, JACOBIAN_EXAMPLES = 4096, 16


def specs(matched: dict | None) -> list[dict]:
    items = [{"name": f"LOOP-T s{seed}", "cell": f"LOOP-T-core4-ds-s{seed}", "label": "C3"} for seed in (1, 2, 3)]
    items.append({"name": "FF-U s1 (below gate)", "cell": "FF-U-core4-ds-inj-s1", "label": "latest"})
    # Added 2026-10-12: the first untied seed to pass G0 (plan amendment).
    items.append({"name": "FF-U s3", "cell": "FF-U-core4-ds-inj-s3", "label": "C3"})
    if matched is not None:
        items.append({"name": f"LOOP-T s{matched['seed']} matched", "cell": matched["cell_id"], "label": matched["label"]})
    items += [{"name": f"LOOP-T s{seed} C0", "cell": f"LOOP-T-core4-s{seed}", "label": "C0"} for seed in (1, 2, 3)]
    items.append({"name": "FF-U s1 C0", "cell": "FF-U-core4-s1", "label": "C0", "injection": True})
    items.append({"name": "FF-U s3 C0", "cell": "FF-U-core4-s3", "label": "C0", "injection": True})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--matched", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/matched_checkpoint.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase1/e3_similarity.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    matched = json.loads(args.matched.read_text(encoding="utf-8"))["chosen"] if args.matched.exists() else None

    tokens = {}
    for family in FAMILIES:
        rows = [row for row in read_shard(args.monitor_bundle, "monitor_train", family) if row.sibling is None]
        order = np.random.default_rng(0).permutation(len(rows))[:EXAMPLES]
        tokens[family] = torch.tensor([rows[index].tokens for index in order])

    states: dict = {}
    report: dict = {"examples": EXAMPLES, "matched": matched, "models": {}, "pairs": {}}
    for spec in specs(matched):
        model = load_analysis_model(args.runs, args.artifact_dir, spec["cell"], spec["label"], device=device)
        if spec.get("injection"):
            model = with_injection(model, True)
        entry = {"cell_id": spec["cell"], "label": spec["label"], "step": model.step, "families": {}}
        jacobian_tokens = torch.cat([tokens[family][:JACOBIAN_EXAMPLES // 4] for family in FAMILIES])
        full = block_spectral_norms(model, jacobian_tokens)
        update = block_spectral_norms(model, jacobian_tokens, update=True)
        entry["jacobian_top_singular_value"] = {"block_median": np.median(full, axis=0).tolist(),
                                                "update_median": np.median(update, axis=0).tolist()}
        for family in FAMILIES:
            raw, _ = answer_states(model, tokens[family])
            normalized = rms_normalize(raw).to(device)
            states[(spec["name"], family)] = normalized.cpu()
            centered = normalized - normalized[:, 1:].mean(dim=1, keepdim=True)
            permuted = normalized[torch.randperm(normalized.shape[0], generator=torch.Generator().manual_seed(0)).to(device)]
            entry["families"][family] = {
                "cka": cka_matrix(normalized, normalized).round(4).tolist(),
                "cka_visit_mean_centered": cka_matrix(centered[:, 1:], centered[:, 1:]).round(4).tolist(),
                "cka_permutation_null": [debiased_cka(normalized[:, site], permuted[:, site]) for site in range(normalized.shape[1])],
                "procrustes_next": [procrustes_similarity(normalized[:, site], normalized[:, site + 1])
                                    for site in range(normalized.shape[1] - 1)],
                "raw_norm_mean": raw.norm(dim=-1).mean(dim=0).tolist(),
            }
        report["models"][spec["name"]] = entry
        print(spec["name"], "jacobian update median", [round(v, 3) for v in entry["jacobian_top_singular_value"]["update_median"]], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    names = list(report["models"])
    tied = [name for name in names if name.startswith("LOOP-T") and "C0" not in name]
    comparisons = [(a, b) for i, a in enumerate(tied) for b in tied[i + 1 :]]
    for untied in ("FF-U s1 (below gate)", "FF-U s3"):
        if untied in names:
            comparisons += [(name, untied) for name in tied]
    comparisons.append(("FF-U s1 (below gate)", "FF-U s3"))
    comparisons += [("LOOP-T s1 C0", "FF-U s1 C0"), ("LOOP-T s3 C0", "FF-U s3 C0"), ("FF-U s1 C0", "FF-U s3 C0")]
    for left, right in comparisons:
        if left not in names or right not in names:
            continue
        block = {}
        for family in FAMILIES:
            matrix = cka_matrix(states[(left, family)].to(device), states[(right, family)].to(device))
            path, mean = monotone_path(matrix)
            block[family] = {"cka": matrix.round(4).tolist(), "monotone_path": path, "path_mean": mean,
                             "diagonal_mean": float(np.mean(np.diag(matrix)))}
        report["pairs"][f"{left} | {right}"] = block
        print(left, "|", right, {family: round(value["path_mean"], 3) for family, value in block.items()}, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8", newline="\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
