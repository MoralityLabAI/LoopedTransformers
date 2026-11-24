"""Freeze the larger calibration roster and the packed phase traversal schedule."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gated46.source import inputs
from experiments.nan_nativerepair91.source import PIN, load as lowdose
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source
from experiments.nan_suffix80.prepare import load as suffix, prior_prompts
from experiments.nan_tangent87.source import NODES, graph_signature, load as tangent
from experiments.nan_writer77.source import exposed

from .verify import MANIFEST, REPO, seal


CONFIG = {"experiment": "NAN-PASSREUSE-96", "source_seed": 2026112396, "source_roots": 64,
          "arms": ["focused", "copy", "focused_erased"], "checkpoint_revision": PIN}
BUDGET = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
          "maximum_private_bytes": 256 * 2**20, "maximum_seconds": 150}


def schedule(roots, phase_components, input_tokens):
    cases = [[f"component_{i}_{sign}" for sign in ("positive", "negative")] for i in phase_components]
    cases += [["twin_donor", "span_sham"], ["ambient_sham"]]
    controls = [name for pair in cases for name in pair]
    if len(controls) != 17 or len(controls) != len(set(controls)):
        raise ValueError("The registered DELAY39 arm roster changed")
    # Initial native mixed and independent self captures use two passes apiece.
    # The duplicate baseline uses two passes in both legacy and shared modes.
    fixed = 6
    worlds = 2 * roots
    calibration_passes = 2 * len(CONFIG["arms"]) * roots
    per_world = {"legacy_companion": fixed + 2*len(controls), "shared_companion": fixed + len(controls),
                 "shared_packed": fixed + len(cases)}
    totals = {k: calibration_passes+worlds*n for k, n in per_world.items()}
    minimum = next(n for n in range(1, roots+1) if 1-Fraction(19, 20)**n >= Fraction(19, 20))
    return {"reference": "dense qualified ceiling at the registered batch and prompt shape",
            "phase1_components": phase_components, "packed_edit_pairs": cases,
            "independent_self_preserved": True, "singletons_retaining_clean_companion": len([p for p in cases if len(p)==1]),
            "edited_pass2_batches_per_world": {"legacy_companion": len(controls), "shared_companion": len(controls),
                                                "shared_packed": len(cases)},
            "focused_worlds": worlds, "full_B2_baseline_batches": calibration_passes//2,
            "physical_pass_traversals_per_focused_world": per_world, "total_B2_pass_traversals": totals,
            "distinct_matched_sham_roots_required_for_95_95": minimum,
            "available_roots_if_all_qualify": roots,
            "whole_panel_maximum_confidence_if_independent": float(1-Fraction(19, 20)**roots),
            "max_raw_phase1_canvas_bytes": 2 * input_tokens * 3072 * 2}


def prepare(snapshot, out):
    freeze = seal()
    if json.loads(MANIFEST.read_bytes()) != freeze or snapshot.name != PIN:
        raise ValueError("Registered source or checkpoint changed")
    with lease(BUDGET) as (check, samples):
        verify_source()
        prior = prior_prompts() | exposed()
        for reader, path in ((suffix, "nan_suffix80"), (tangent, "nan_tangent87"), (lowdose, "nan_nativerepair91")):
            data, _ = reader(REPO / "runs" / path / "source01", ("public",))
            prior.update(p["prompt"] for p in data["public"])
        old_graphs = {sig for prompt in prior if (sig := graph_signature(prompt)) is not None}
        public, gold = source_cases(CONFIG)
        for item in public + gold:
            for field in ("id", "root"):
                item[field] = item[field].replace("delay39:", "passreuse96:", 1)
            item["split"] = "fresh_calibration"
        roots = CONFIG["source_roots"]
        observed = independently_verify(public, gold, roots, prior)
        graphs = [graph_signature(p["prompt"]) for p in public if p["arm"] == "focused"]
        if len(graphs) != 2*roots or None in graphs or set(graphs) & old_graphs or len(set(graphs)) != observed:
            raise ValueError("The fresh calibration graphs overlap previously exposed sources")
        if Counter(row["answer"] for row in gold if row["arm"] == "focused") != dict.fromkeys(NODES, 8):
            raise ValueError("The larger calibration roster is not answer balanced")
        tok, pins = tokenizer(snapshot)
        tokens = inputs(tok, public, snapshot, check)
        batches = {}
        for row in public:
            batches.setdefault((row["root"], row["arm"]), {})[row["world"]] = tokens["row_bindings"][row["id"]]
        for (_, arm), pair in batches.items():
            a, b = pair["A"], pair["B"]
            if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
                raise ValueError("Fixed B2 token parity or erased twin identity changed")
        limit = max(row["tokens"] for row in tokens["row_bindings"].values())
        old = json.loads((REPO / "experiments/nan_delay39/registration.json").read_bytes())
        work_plan = schedule(roots, old["phase1_components"], limit)
        assert len(batches) == work_plan["full_B2_baseline_batches"] == 192
    assert seal() == freeze
    out.mkdir(parents=True, exist_ok=False)
    fields = {"public": public, "gold": gold, "tokens": tokens}
    for k, v in fields.items():
        write_new(out / f"{k}.json", v)
    receipt = {"experiment": CONFIG["experiment"], "status": "source_tokenizer_verified",
               "source_manifest_sha256": sha(MANIFEST), "payload_sha256": {k: digest(v) for k, v in fields.items()},
               "tokenizer_pins": pins, "source_seed": CONFIG["source_seed"], "registered_roots": roots,
               "verified_focused_worlds": observed, "registered_prompts": len(public), "max_prompt_tokens": limit,
               "max_preflight_position": limit + 2, "prior_exposed_prompts": len(prior),
               "prior_complete_labelled_graphs": len(old_graphs), "capacity_plan": work_plan,
               "pretrained_forwards": 0, "sealed_test_accessed": False,
               "max_sampled_private_bytes": max(s["private_bytes"] for s in samples)}
    write_new(out / "SOURCE_CHECK.json", receipt)
    print(receipt, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Require source assertions")
    prepare(args.snapshot, args.work)


if __name__ == "__main__":
    main()
