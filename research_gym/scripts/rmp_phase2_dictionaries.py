"""Phase 2, E4: train a grid of dictionaries for one model and score each on single-visit splices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.dictionaries import (
    DictionaryConfig,
    batch_source,
    fraction_of_variance_unexplained,
    gaussian_source,
    model_states,
    normalized_visit_states,
    save_dictionary,
    site_vectors,
    train_dictionary,
)
from research_gym.interp.phase2 import (
    MODELS,
    adjacent_width,
    core_examples,
    load_model,
    mean_states,
    select_config,
    selection_half,
    slug,
    subset,
)
from research_gym.interp.replacement import answer_scores, at_visits, ce_recovered, reconstruct, transcoder_step


def splice_report(model, dictionary, config: DictionaryConfig, data, means: torch.Tensor) -> dict:
    """Cross-entropy recovered when the dictionary replaces the state (or the update) at one visit."""

    visits = model.config.visits
    tokens, targets = data["tokens"], data["targets"]
    clean = float(answer_scores(model, tokens, targets)[0].mean())
    report = {"ce_clean": clean, "by_visit": {}}
    if config.kind == "trans":
        # Visit v adds the update to resid[v]; the ablation adds the mean update at that visit instead.
        chosen = range(visits) if config.site < 0 else [config.site]
        mean_update = lambda context: context.before + (means[context.visit + 1] - means[context.visit])  # noqa: E731
        replace = transcoder_step([dictionary])
        labels = {visit: f"update{visit}" for visit in chosen}
    elif config.kind in ("shared", "gauss", "visit"):
        # Hooking visit v replaces resid[v + 1].
        chosen = [config.site - 1] if config.kind == "visit" else range(visits)
        mean_update = lambda context: means[context.visit + 1].unsqueeze(0).expand_as(context.after)  # noqa: E731
        replace = reconstruct(dictionary)
        labels = {visit: f"resid{visit + 1}" for visit in chosen}
    else:
        return report
    for visit in chosen:
        ablated = float(answer_scores(model, tokens, targets, at_visits(mean_update, [visit]))[0].mean())
        spliced = float(answer_scores(model, tokens, targets, at_visits(replace, [visit]))[0].mean())
        report["by_visit"][labels[visit]] = {
            "ce_mean_ablated": ablated, "ce_spliced": spliced, "ce_recovered": ce_recovered(clean, ablated, spliced),
        }
    report["min_ce_recovered"] = min(value["ce_recovered"] for value in report["by_visit"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--kind", choices=("shared", "trans", "tok", "visit", "gauss", "centered"), required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--selected-from", default="", help="use this model's selected size instead of --widths/--ks")
    parser.add_argument("--selected-kind", default="shared", choices=("shared", "trans"))
    parser.add_argument("--adjacent", action="store_true", help="with --selected-from, add the adjacent width")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sites", type=int, nargs="+", default=[-1], help="visit (visit) or layer (trans); -1 pooled")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="where dictionaries are saved")
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--eval-examples", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    model = load_model(args.model, args.runs, args.artifact_dir, device)
    train = core_examples(args.monitor_bundle, "monitor_train")
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    select = subset(calibration, selection_half(calibration["groups"]))
    order = np.random.default_rng(0).permutation(len(select["groups"]))[: args.eval_examples]
    evaluation = subset(select, np.isin(np.arange(len(select["groups"])), order))
    means = mean_states(model, train["tokens"])
    eval_states = model_states(model, evaluation["tokens"][:256].to(device))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics.exists() else {}
    entry = metrics.setdefault(args.model, {})
    widths, ks = args.widths, args.ks
    if args.selected_from:
        chosen, _ = select_config(metrics[args.selected_from], kind=args.selected_kind)
        base = metrics[args.selected_from][chosen]["config"]
        widths = [base["width"]] + ([adjacent_width(base["width"])] if args.adjacent else [])
        ks = [base["k"]]
        print("selected size from", args.selected_from, chosen, "->", widths, ks, flush=True)

    # Controls 2 and 10 need moments of the normalized training states (400 examples, all positions).
    centering = moments = None
    if args.kind in ("gauss", "centered"):
        sample = train["tokens"][torch.randperm(train["tokens"].shape[0], generator=torch.Generator().manual_seed(1))[:400]]
        states = normalized_visit_states(model, sample)
        if args.kind == "centered":
            centering = states.mean(dim=(0, 2))
        else:
            flat = states.reshape(-1, states.shape[-1])
            moments = (flat.mean(dim=0), torch.cov(flat.T))
        del states

    for seed in args.seeds:
        for site in args.sites:
            for width in widths:
                for k in ks:
                    config = DictionaryConfig(kind=args.kind, width=width, k=k, seed=seed, site=site, steps=args.steps)
                    path = args.work_dir / slug(args.model) / f"{config.name}.pt"
                    if config.name in entry and path.exists() and not args.force:
                        print("exists, skipped:", args.model, config.name, flush=True)
                        continue
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    if moments is not None:
                        count = config.batch_examples * train["tokens"].shape[1] * model.config.visits
                        source = gaussian_source(*moments, count, seed=seed)
                    else:
                        source = batch_source(model, train["tokens"], config,
                                              generator=torch.Generator().manual_seed(seed), centering=centering)
                    dictionary, summary = train_dictionary(source, config, dim=model.config.hidden_size, device=device)
                    sha = save_dictionary(dictionary, summary, path)
                    x, y = site_vectors(eval_states, config, layers=model.config.visits, centering=centering)
                    result = {**summary, "path": str(path), "sha256": sha,
                              "calibration_fvu": fraction_of_variance_unexplained(dictionary, x, y),
                              "splice": splice_report(model, dictionary, config, evaluation, means)}
                    if centering is not None:
                        result["centering"] = centering.cpu().tolist()
                    if config.name in entry:
                        # Retrained after the artifact directory was lost; training is meant to be deterministic.
                        previous = entry[config.name].get("previous_sha256", entry[config.name]["sha256"])
                        result["previous_sha256"], result["reproduced"] = previous, sha == previous
                    if device.type == "cuda":
                        result["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
                    entry[config.name] = result
                    args.metrics.parent.mkdir(parents=True, exist_ok=True)
                    args.metrics.write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
                    print(args.model, config.name, "fvu", round(result["calibration_fvu"], 4), "min CE recovered",
                          round(result["splice"].get("min_ce_recovered", float("nan")), 4), flush=True)


if __name__ == "__main__":
    main()
