"""Read-only CPU diagnosis of existing LSA-02B native training artifacts.

Consumes the exact immutable outputs of a completed GitHub Actions campaign.
No new training, native proof claims, model promotion, or test-set reads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from .cli import checkpoint_load
from .data import canonical, sha256_file
from .native_train import _load_teacher


SCHEMA = "lsa02b.cpu_architecture_diagnostics.v1"


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_artifacts(directory: Path, references: dict) -> None:
    for filename, expected in references.items():
        if filename != Path(filename).name or sha256_file(directory / filename) != expected:
            raise ValueError("result artifact SHA-256 differs from manifest: " + str(filename))


def choice_geometry(rows: list[dict]) -> dict:
    """Count meaningful teaching forks, treating each native theorem as one unit."""
    roots: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        roots[row["lineage"]].append(row)
    if not roots:
        raise ValueError("teacher has no independent lineages")
    counts = {
        "independent_lineages": len(roots),
        "steps": len(rows),
        "menu_singletons": 0,
        "menu_forks": 0,
        "discriminating_forks": 0,
        "ambiguous_optimal_forks": 0,
    }
    for line in rows:
        n = len(line["candidates"])
        positives = sum(line["labels"])
        if n == 1:
            counts["menu_singletons"] += 1
        else:
            counts["menu_forks"] += 1
            if 0 < positives < n:
                counts["discriminating_forks"] += 1
            if positives > 1:
                counts["ambiguous_optimal_forks"] += 1
    counts["mean_discriminating_forks_per_root"] = sum(
        sum(len(r["candidates"]) > 1 and
            0 < sum(r["labels"]) < len(r["labels"]) for r in frames)
        for frames in roots.values()
    ) / len(roots)
    return counts


def shared_decisions(left: dict, right: dict) -> tuple[int, int]:
    """Match only identical source+menu+budget packets across both arm traces."""
    def indexed(episode: dict) -> dict[tuple[str, str], list[str]]:
        result: dict[tuple[str, str], list[str]] = defaultdict(list)
        for event in episode["events"]:
            key = (event["public_packet_sha256"], event["menu_sha256"])
            result[key].append(event["tactic"])
        return result

    a, b = indexed(left), indexed(right)
    shared = different = 0
    for key in a.keys() & b.keys():
        for aa, bb in zip(a[key], b[key]):
            shared += 1
            different += int(aa != bb)
    return shared, different


@torch.no_grad()
def training_scores(model, frames: list[dict], depth: int) -> dict:
    """Source-only action-ranker diagnostic; same frames under frozen AB/BA."""
    by_root: dict[str, list[int]] = defaultdict(list)
    logits_shift = []
    changed = 0
    total = len(frames)
    for row in frames:
        logits = {}
        for order in ("ab", "ba"):
            logits[order] = model(
                row["goal"], row["candidates"], depth=depth,
                checks_remaining=row["checks_remaining"],
                middle_order=order,
            )[-1]["logits"].float().cpu()
        indexes = {order: int(value.argmax()) for order, value in logits.items()}
        labels = row["labels"]
        for order in ("ab", "ba"):
            by_root[(row["lineage"], order)].append(int(labels[indexes[order]] == 1))
        changed += int(indexes["ab"] != indexes["ba"])
        shift = float((logits["ab"] - logits["ba"]).norm())
        if not math.isfinite(shift):
            raise FloatingPointError("nonfinite output displacement")
        logits_shift.append(shift)
    families = sorted({key[0] for key in by_root})
    return {
        "source_frames": total,
        "independent_train_roots": len(families),
        "train_optimal_accuracy_ab_root_mean": sum(
            sum(by_root[(root, "ab")]) / len(by_root[(root, "ab")]) for root in families
        ) / len(families),
        "train_optimal_accuracy_ba_root_mean": sum(
            sum(by_root[(root, "ba")]) / len(by_root[(root, "ba")]) for root in families
        ) / len(families),
        "train_frozen_order_choice_changes": changed,
        "mean_train_logit_shift": sum(logits_shift) / total,
        "max_train_logit_shift": max(logits_shift),
    }


def _probe_stats(directory: Path, teacher_sha: str, checkpoint_sha: str) -> dict:
    root = directory / "lsa02b-cpu-probe"
    replay_file = directory / "lsa02b-cpu-replay" / "replay.json"
    manifest = _manifest(root / "manifest.json")
    if manifest.get("schema") != "lsa02b.native_block_order.v1":
        raise ValueError("incorrect native probe schema")
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("native probe not bound to supplied checkpoint")
    if (
        not manifest.get("training_provenance", {}).get("training_source_verified")
        or not manifest["training_provenance"]["source_lineages_disjoint"]
        or manifest["training_provenance"].get("native_teacher_manifest_sha256") != teacher_sha
    ):
        raise ValueError("native probe lacks matching verified teacher source")
    _verify_artifacts(root, manifest["artifacts"])
    replay = _manifest(replay_file)
    if not replay.get("replay_passed") or replay.get("paired_roots") != manifest.get("paired_roots"):
        raise ValueError("native probe lacks independent matching action replay")
    if manifest.get("scientific_eligible") or manifest.get("architecture_promotion_eligible"):
        raise ValueError("development-only probe incorrectly claims promotion")
    episodes = [json.loads(s) for s in (root / "paired_episodes.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if s.strip()]
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    counts = {
        "native_paired_roots": manifest["paired_roots"],
        "reference_budget_reachable": manifest["reference_reachable_within_budget"],
        "ab_kernel_and_standalone": manifest["scores"]["ab"]["standalone_verified_closures"],
        "ba_kernel_and_standalone": manifest["scores"]["ba"]["standalone_verified_closures"],
        "auto8_kernel_and_standalone": manifest["scores"]["auto8"]["standalone_verified_closures"],
        "ab_tactic_checks": manifest["scores"]["ab"]["tactic_checks"],
        "ba_tactic_checks": manifest["scores"]["ba"]["tactic_checks"],
        "first_action_disagreements": manifest["first_tactic_disagreements"],
        "ab_menu_fork_steps": 0,
        "ab_menu_singleton_steps": 0,
        "ba_menu_fork_steps": 0,
        "ba_menu_singleton_steps": 0,
        "shared_public_packet_decisions": 0,
        "different_actions_on_shared_packets": 0,
        "all_episodes": len(episodes),
    }
    for ep in episodes:
        if ep["root"] in groups and ep["order"] in groups[ep["root"]]:
            raise ValueError("duplicate native root arm")
        groups[ep["root"]][ep["order"]] = ep
        if ep["order"] in ("ab", "ba"):
            counts[f"{ep['order']}_menu_fork_steps"] += sum(
                len(e["candidate_menu"]) > 1 for e in ep["events"]
            )
            counts[f"{ep['order']}_menu_singleton_steps"] += sum(
                len(e["candidate_menu"]) == 1 for e in ep["events"]
            )
    if len(groups) != manifest["paired_roots"] or any(
        set(by_arm) != {"ab", "ba", "auto8"} for by_arm in groups.values()
    ):
        raise ValueError("incomplete same-theorem model and control arms")
    for by_arm in groups.values():
        shared, different = shared_decisions(by_arm["ab"], by_arm["ba"])
        counts["shared_public_packet_decisions"] += shared
        counts["different_actions_on_shared_packets"] += different
    counts["native_probe_manifest_sha256"] = sha256_file(root / "manifest.json")
    counts["native_replay_manifest_sha256"] = sha256_file(replay_file)
    return counts


def analyze(
    teacher: Path,
    fits: Path,
    probes: Path,
    out: Path,
    *,
    seeds: tuple[int, ...] = (101, 103),
) -> dict:
    if out.exists():
        raise ValueError("output destination must be fresh")
    frames, metadata = _load_teacher(teacher)
    if not metadata["backend"]["actual_lean_process"]:
        raise ValueError("native source must come from pinned Lean")
    teacher_sha = sha256_file(teacher / "manifest.json")
    geometry = choice_geometry(frames)
    grid = []
    for seed in seeds:
        init_by_order: dict[str, str] = {}
        for order in ("ab", "ba"):
            key = f"{seed}-{order}"
            model, checkpoint = checkpoint_load(fits / key / "checkpoint.pt")
            manifest = _manifest(fits / key / "manifest.json")
            if (
                model.cfg.plan != "b" or checkpoint.get("seed") != seed
                or checkpoint.get("train_order") != order
                or checkpoint.get("native_teacher_sha256") != teacher_sha
                or manifest.get("native_teacher_manifest_sha256") != teacher_sha
                or manifest.get("source_manifest_sha256") != metadata["source"]["suite_manifest_sha256"]
                or manifest.get("checkpoint_sha256") != sha256_file(fits / key / "checkpoint.pt")
                or manifest.get("optimizer_steps") != 240
                or manifest.get("train_order") != order
                or manifest.get("seed") != seed
            ):
                raise ValueError("checkpoint and registered seed/order/source identity differ")
            init_by_order[order] = manifest["initial_weights_sha256"]
            model.eval()
            score = training_scores(model, frames, int(manifest["depth"]))
            native = _probe_stats(
                probes / key, teacher_sha, sha256_file(fits / key / "checkpoint.pt"),
            )
            grid.append({
                "seed": seed, "trained_order": order,
                "parameter_count": model.parameter_count(),
                "optimizer_steps": manifest["optimizer_steps"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "initial_weights_sha256": init_by_order[order],
                "last_training_loss": manifest["last_training_loss"],
                "mean_final_32_training_loss": manifest["mean_final_32_loss"],
                "train": score,
                "native_calibration": native,
            })
        if init_by_order["ab"] != init_by_order["ba"]:
            raise ValueError("matched seed AB/BA did not share initial weights")

    trained = {
        "ab": [v for v in grid if v["trained_order"] == "ab"],
        "ba": [v for v in grid if v["trained_order"] == "ba"],
    }
    report = {
        "schema": SCHEMA,
        "source_run_id": 45167489504,
        "source_teacher_sha256": teacher_sha,
        "source_lineages": metadata["independent_lineages"],
        "source_frames": len(frames),
        "source_choices": geometry,
        "calibration_independent_roots": grid[0]["native_calibration"]["native_paired_roots"],
        "matched_seed_count": len(seeds),
        "cell_count": len(grid),
        "trained_order_native_mean_close_rate": {
            order: sum(
                v["native_calibration"][f"{order}_kernel_and_standalone"]
                / v["native_calibration"]["native_paired_roots"] for v in cells
            ) / len(cells)
            for order, cells in trained.items()
        },
        "checkpoint_records": grid,
        "formal_competence_gate": "BLOCKED_INCOMPETENT_CHECKPOINT",
        "claim": "Development-only source and action-fork diagnosis; no heldout exposure or new training",
        "no_new_native_lean_calls": True,
        "scientific_eligible": False,
    }
    out.mkdir(parents=True)
    (out / "summary.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    r = analyze(a.teacher, a.fits, a.probes, a.out)
    print(canonical({
        "source_choices": r["source_choices"],
        "trained_order_native_mean_close_rate": r["trained_order_native_mean_close_rate"],
        "checkpoint_train_diagnostics": [
            {"seed": x["seed"], "trained_order": x["trained_order"],
             "train": x["train"], "native": x["native_calibration"]}
            for x in r["checkpoint_records"]
        ],
        "competence_gate": r["formal_competence_gate"],
    }))
    return r


if __name__ == "__main__":
    main()
