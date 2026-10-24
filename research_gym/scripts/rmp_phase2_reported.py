"""Phase 2, E4: the reported controls 3, 9 and 10, and the shared-versus-visit dictionary gap.

All at each model's selected `D-shared` size (docs/rmp_phase2_plan.md). Each count is also given for the
features the association pipeline found (`association.json`), since those are the ones a claim would rest on.

- Control 3, dictionary seeds: a seed-0 feature is matched when its best decoder cosine in seeds 1 and 2 is
  0.9 or more.
- Control 9, given versus computed: a feature is given when its best decoder cosine to `D-tok` (the
  dictionary of the states entering visit 0) is 0.9 or more.
- Control 10, visit identity: a feature is a visit-identity feature when visit index alone explains 90% or
  more of its activation variance over calibration states (all positions, visits 1 to 8); reported for the
  selected dictionary and for the one trained on visit-mean-centered states.

Cosines are signed: TopK features are non-negative, so a feature and its negation are different features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.interp.dictionaries import load_dictionary, normalized_visit_states
from research_gym.interp.phase2 import core_examples, load_model, select_config, selection_half, subset

MAIN = ["LOOP-T s1", "LOOP-T s2", "LOOP-T s3", "FF-U s3"]
MATCH = 0.9


def best_cosines(decoder: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """For each row of ``decoder``, its largest cosine with any row of ``others`` (rows are unit norm)."""

    return (decoder @ others.T).max(dim=1).values


@torch.no_grad()
def visit_identity(dictionary, states: torch.Tensor, centering: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per feature: share of activation variance explained by visit index, and whether it ever fired.

    ``states``: normalized (N, visits, positions, H)."""

    visits = states.shape[1]
    sums = sums_sq = None
    counts = 0
    for start in range(0, states.shape[0], 25):
        x = states[start : start + 25]
        if centering is not None:
            x = x - centering[None, :, None, :]
        codes = dictionary.encode(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], visits, x.shape[2], -1)
        part, part_sq = codes.sum(dim=(0, 2)), codes.pow(2).sum(dim=(0, 2))  # (visits, width)
        sums = part if sums is None else sums + part
        sums_sq = part_sq if sums_sq is None else sums_sq + part_sq
        counts += x.shape[0] * x.shape[2]
    total = counts * visits
    mean = sums.sum(dim=0) / total
    total_var = sums_sq.sum(dim=0) / total - mean.pow(2)
    between = ((sums / counts - mean).pow(2)).mean(dim=0)
    alive = sums.sum(dim=0) > 0
    share = torch.where(total_var > 1e-12, between / total_var.clamp_min(1e-12), torch.zeros_like(total_var))
    return share, alive


def found_features(association: dict, name: str) -> list[int]:
    entry = association.get(name)
    return sorted({test["feature"] for test in entry["tests"] if test["found"]}) if entry else []


def fraction(mask: torch.Tensor, index: list[int]) -> dict:
    chosen = mask[index] if index else mask[:0]
    return {"count": int(chosen.sum()), "of": len(index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MAIN)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--association", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/association.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/reported_controls.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    association = json.loads(args.association.read_text(encoding="utf-8")) if args.association.exists() else {}
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    select = subset(calibration, selection_half(calibration["groups"]))
    report = {}
    for name in args.models:
        if name not in metrics:
            print("no dictionaries yet, skipped:", name, flush=True)
            continue
        entries = metrics[name]
        chosen, _ = select_config(entries)
        size = chosen.removeprefix("shared_").removesuffix("_s0")
        load = lambda key: load_dictionary(Path(entries[key]["path"]), device=device)[0]  # noqa: E731
        shared = load(chosen)
        found = found_features(association, name)
        entry: dict = {"dictionary": chosen, "association_found_features": found}
        model = load_model(name, args.runs, args.artifact_dir, device)
        states = normalized_visit_states(model, select["tokens"][:300])
        share, alive = visit_identity(shared, states)
        living = torch.flatnonzero(alive).tolist()  # features that never fire are left at their random start
        entry["alive_features"] = len(living)

        seeds = [f"shared_{size}_s{seed}" for seed in (1, 2)]
        if all(key in entries for key in seeds):
            matched = torch.stack([best_cosines(shared.decoder, load(key).decoder) for key in seeds]).min(dim=0).values >= MATCH
            entry["control3_seeds"] = {"matched_alive": fraction(matched, living),
                                       "matched_found": fraction(matched, found)}

        if f"tok_{size}_s0" in entries:
            given = best_cosines(shared.decoder, load(f"tok_{size}_s0").decoder) >= MATCH
            entry["control9_given"] = {"given_alive": fraction(given, living),
                                       "given_found": fraction(given, found),
                                       "computed_found": len(found) - int(given[found].sum()) if found else 0}

        visit = (share >= 0.9) & alive
        entry["control10_visit_identity"] = {"visit_identity_alive": fraction(visit, living),
                                             "visit_identity_found": fraction(visit, found)}
        centered_key = f"centered_{size}_s0"
        if centered_key in entries:
            centering = torch.tensor(entries[centered_key]["centering"], device=device)
            c_share, c_alive = visit_identity(load(centered_key), states, centering)
            entry["control10_visit_identity"]["centered"] = {
                "calibration_fvu": entries[centered_key]["calibration_fvu"],
                "uncentered_calibration_fvu": entries[chosen]["calibration_fvu"],
                "visit_identity_alive": fraction((c_share >= 0.9) & c_alive, torch.flatnonzero(c_alive).tolist()),
            }

        shared_by_visit = entries[chosen]["splice"]["by_visit"]
        gap = {}
        for site in range(1, model.config.visits + 1):
            key = f"visit_t{site}_{size}_s0"
            if key in entries:
                gap[f"resid{site}"] = {"shared": shared_by_visit[f"resid{site}"]["ce_recovered"],
                                       "visit": entries[key]["splice"]["by_visit"][f"resid{site}"]["ce_recovered"]}
        if gap:
            entry["shared_versus_visit_ce_recovered"] = gap
        report[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "association_found_features"}), flush=True)
        del states

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
