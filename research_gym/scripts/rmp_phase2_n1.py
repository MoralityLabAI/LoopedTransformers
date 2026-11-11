"""Phase 2, N1: closed-loop sparse replacement with its controls, and the frozen outcome label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.bundle import read_shard
from research_gym.interp.dictionaries import load_dictionary, model_states, rms
from research_gym.interp.phase2 import MODELS, core_examples, load_model, selection_half, subset
from research_gym.interp.replacement import (
    answer_scores,
    at_visits,
    fit_pca,
    matched_noise,
    pca_components_matching,
    pca_truncation,
    reconstruct,
    transcoder_step,
)


GRAPH_FAMILIES = ("pointer_chase", "gated_pointer_chase", "reachability")
DEEP = (5, 6, 7, 8)


def validation_data(bundle: Path) -> dict:
    rows = [(family, row) for family in GRAPH_FAMILIES for row in read_shard(bundle, "validation", family)
            if row.sibling is None]
    return {
        "tokens": torch.tensor([row.tokens for _, row in rows], dtype=torch.long),
        "targets": torch.tensor([row.target_token for _, row in rows], dtype=torch.long),
        "depths": np.array([row.difficulty for _, row in rows]),
        "families": np.array([family for family, _ in rows]),
    }


def em_summary(correct: np.ndarray, data: dict, clean: dict | None = None) -> dict:
    deep = np.isin(data["depths"], DEEP)
    summary = {"deep": float(correct[deep].mean()), "all": float(correct.mean()),
               "deep_by_family": {family: float(correct[deep & (data["families"] == family)].mean())
                                  for family in GRAPH_FAMILIES}}
    if clean is not None:
        summary["deep_ratio"] = summary["deep"] / max(clean["deep"], 1e-12)
    return summary


@torch.no_grad()
def normalized_states(model, tokens: torch.Tensor, *, batch: int = 100) -> torch.Tensor:
    device = next(model.model.parameters()).device
    parts = []
    for start in range(0, tokens.shape[0], batch):
        states = model_states(model, tokens[start : start + batch].to(device))[:, 1:]
        states = states.reshape(-1, states.shape[-1])
        parts.append(states / rms(states))
    return torch.cat(parts)


def label(entries: list[dict]) -> dict:
    """The program's frozen N1 rules applied over every D-shared dictionary of one model."""

    faithful = [e["name"] for e in entries
                if e["closed_loop"]["deep_ratio"] >= 0.90 and e["closed_loop"]["deep_ratio"] > e["truncation"]["deep_ratio"]]
    eligible = [e for e in entries if e["open_loop_min_ce_recovered"] >= 0.95]
    if faithful:
        return {"label": "faithful_state", "dictionaries": faithful}
    if eligible and all(e["closed_loop"]["deep_ratio"] <= 0.5 for e in eligible):
        noise_ok = all(e["noise"]["deep_ratio"] >= 0.9 for e in eligible)
        return {"label": "open_loop_only" if noise_ok else "fragile_dynamics",
                "dictionaries": [e["name"] for e in eligible]}
    return {"label": "unresolved", "eligible": [e["name"] for e in eligible],
            "note": "no dictionary reached the open-loop bar" if not eligible else "mixed closed-loop results"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/n1.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    model = load_model(args.model, args.runs, args.artifact_dir, device)
    entries_by_name = json.loads(args.metrics.read_text(encoding="utf-8"))[args.model]
    data = validation_data(args.bundle_dir)
    clean_correct = answer_scores(model, data["tokens"], data["targets"])[1]
    clean = em_summary(clean_correct, data)

    train = core_examples(args.monitor_bundle, "monitor_train")
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    select = subset(calibration, selection_half(calibration["groups"]))
    pca_states = normalized_states(model, train["tokens"][torch.randperm(train["tokens"].shape[0], generator=torch.Generator().manual_seed(1))[:400]])
    mean, components, _ = fit_pca(pca_states.to(device))
    calibration_states = normalized_states(model, select["tokens"][:300]).to(device)
    # Truncation error on the calibration states, the same states the dictionary's error is measured on.
    centered = calibration_states - mean
    captured = (centered @ components).pow(2).mean(dim=0)
    total_error = float(centered.pow(2).sum(dim=-1).mean())

    shared = sorted((name, entry) for name, entry in entries_by_name.items()
                    if entry["config"]["kind"] == "shared" and entry["config"]["seed"] == 0)
    results = []
    for name, entry in shared:
        dictionary, _ = load_dictionary(Path(entry["path"]), device=device)
        with torch.no_grad():
            mse = float((calibration_states - dictionary(calibration_states)[0]).pow(2).sum(dim=-1).mean())
        count = pca_components_matching(captured, mse, total_error)
        closed = answer_scores(model, data["tokens"], data["targets"], at_visits(reconstruct(dictionary)))[1]
        noisy = answer_scores(model, data["tokens"], data["targets"], at_visits(matched_noise(dictionary, seed=0)))[1]
        truncated = answer_scores(model, data["tokens"], data["targets"],
                                  at_visits(pca_truncation(mean, components[:, :count])))[1]
        open_loop = {
            f"resid{visit + 1}": em_summary(
                answer_scores(model, data["tokens"], data["targets"], at_visits(reconstruct(dictionary), [visit]))[1],
                data, clean)["deep_ratio"]
            for visit in range(model.config.visits)
        }
        result = {
            "name": name, "width": entry["config"]["width"], "k": entry["config"]["k"],
            "open_loop_min_ce_recovered": entry["splice"]["min_ce_recovered"],
            "calibration_mse": mse, "pca_components": count,
            "closed_loop": em_summary(closed, data, clean), "noise": em_summary(noisy, data, clean),
            "truncation": em_summary(truncated, data, clean), "open_loop_deep_ratio": open_loop,
        }
        results.append(result)
        print(args.model, name, "closed", round(result["closed_loop"]["deep_ratio"], 3), "noise",
              round(result["noise"]["deep_ratio"], 3), "pca", round(result["truncation"]["deep_ratio"], 3),
              f"(q={count})", flush=True)

    # Replacement model: the block is never used; each visit adds the transcoder's predicted update.
    replacement = []
    trans = {name: entry for name, entry in entries_by_name.items() if entry["config"]["kind"] == "trans"
             and entry["config"]["seed"] == 0}
    for key in sorted({(e["config"]["width"], e["config"]["k"]) for e in trans.values()}):
        members = sorted((e["config"]["site"], e) for e in trans.values() if (e["config"]["width"], e["config"]["k"]) == key)
        sites = [site for site, _ in members]
        if sites == [-1]:
            dictionaries = [load_dictionary(Path(members[0][1]["path"]), device=device)[0]]
        elif sites == list(range(model.config.visits)):
            dictionaries = [load_dictionary(Path(e["path"]), device=device)[0] for _, e in members]
        else:
            continue
        correct = answer_scores(model, data["tokens"], data["targets"], at_visits(transcoder_step(dictionaries)))[1]
        replacement.append({"width": key[0], "k": key[1], "per_layer": sites != [-1], **em_summary(correct, data, clean)})
        print(args.model, "replacement", key, round(replacement[-1]["deep_ratio"], 3), flush=True)

    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    report[args.model] = {"clean": clean, "dictionaries": results, "replacement_model": replacement,
                          "label": label(results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(args.model, "N1 label:", report[args.model]["label"], flush=True)


if __name__ == "__main__":
    main()
