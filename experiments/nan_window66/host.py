"""Independently score only immutable, complete signed window worlds."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean

from experiments.nan_ranksite13.resources import below_normal
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_route58.custody import bound_source, settings
from experiments.nan_route58.host import audit as audit_route

from .capture import FOCUS
from .lane import EXPERIMENT, verify
from .telemetry import LINEAR, TERMS, score_profile


def load_bound(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Immutable result changed: {path.name}")
    return data


def reduce_complete(roots):
    complete = [tuple(worlds.values()) for worlds in roots.values()
                if set(worlds) == {"A", "B"} and all(value is not None for value in worlds.values())]
    summaries = {}

    def macro(get):
        return fmean(fmean(get(world) for world in pair) for pair in complete)

    if complete:
        for layer in FOCUS:
            key = str(layer)
            summaries[key] = {
                "restricted_head_effect": macro(lambda world: world[key]["restricted_head_effect"]),
                "rounded_head_allocation": {term: macro(lambda world: world[key]["rounded_head_allocation"][term])
                                            for term in TERMS},
                "continuous_attribution": {term: macro(lambda world: world[key]["continuous_attribution"][term])
                                           for term in LINEAR},
            }
    return len(complete), summaries


def audit(folder, work, manifest, cfg, round_cfg, dose, *, fixture=False):
    terminal_path = work / "terminal.json"
    starting_sha = sha(terminal_path)
    terminal = json.loads(load_bound(terminal_path, starting_sha))
    mode = "fixture" if fixture else "pretrained_cuda"
    if (terminal["experiment"], terminal["mode"], terminal["window_layers"]) != (EXPERIMENT, mode, list(FOCUS)):
        raise ValueError("Expected the registered WINDOW66 terminal and layer window")
    preflight = json.loads(load_bound(work / "preflight.json", terminal["preflight_sha256"]))
    if preflight["window"] != {"layers": list(FOCUS), "same_shape_native_bytes_equal": True,
                               "checked_residual_additions": len(FOCUS)*2*2*2}:
        raise ValueError("The signed result lacks a checked native window preflight")
    if not fixture and (preflight.get("synthetic") or not terminal.get("published_commit")):
        raise ValueError("A device result needs native preflight and a published runtime")

    endpoint = audit_route(folder, work, manifest, cfg, round_cfg, dose)
    gold = {(r["root"], r["world"]): r for r in bound_source(folder, "gold") if r["arm"] == "focused"}
    labels = bound_source(folder, "tokens")["bare_label_token_ids"]
    native = json.loads(load_bound(work / "native.json", terminal["native_sha256"]))
    native_tokens = {row["id"]: row["token_id"] for row in native}
    journal = load_bound(work / "closed.jsonl", terminal["closed_journal_sha256"]).splitlines()
    roots = defaultdict(dict)
    for i, (line, ((root, world), item)) in enumerate(zip(journal, sorted(gold.items()), strict=True)):
        path = work / f"episode_{i:03d}.json"
        entry = json.loads(line)
        if set(entry) != {"file", "sha256"} or entry["file"] != path.name:
            raise ValueError("The delayed window is assigned to a different journal slot")
        row = json.loads(load_bound(path, entry["sha256"]))
        if row["source_id"] != item["id"]:
            raise ValueError("The delayed window is assigned to a different source world")
        foil = gold[root, "B" if world == "A" else "A"]
        score = score_profile(row, native_tokens[item["id"]], labels[foil["answer"]], round_cfg, dose)
        roots[root][world] = None if score is None else score["delayed_window_effects"]
    count, summaries = reduce_complete(roots)
    if (endpoint["worker_terminal_sha256"] != starting_sha or sha(terminal_path) != starting_sha
            or endpoint["roots_with_both_worlds_scored"] != count
            or endpoint["worlds_scored"] != sum(v is not None for pair in roots.values() for v in pair.values())):
        raise ValueError("Native endpoint and window scorer disagree about terminal or coverage")
    return endpoint | {"experiment": EXPERIMENT, "mode": mode, "window_layers": list(FOCUS),
                       "window_root_macro_mean": summaries}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    below_normal()
    manifest = verify()
    cfg, _, rounding, dose = settings()
    report = audit(args.source_dir, args.work, manifest, cfg, rounding, dose)
    write_new(args.output, report)


if __name__ == "__main__":
    main()
