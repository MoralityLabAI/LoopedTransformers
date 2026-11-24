"""Freeze distinct counterfactual B2 roots for the native boundary rescue pilot."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.nan_delay39.source import ARMS, digest, independently_verify, source_cases
from experiments.nan_gated46.source import inputs
from experiments.nan_packedcal97.source import read_source as packed_source
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source
from experiments.nan_suffix80.prepare import load as suffix_source, prior_prompts
from experiments.nan_tangent87.source import BUDGET, NODES, graph_signature, load as tangent_source
from experiments.nan_writer77.source import exposed
from experiments.nan_readpath37.run import write_new

from .core import REPO
from .verify import HERE, MANIFEST as NATIVE_MANIFEST, registration as native_registration


MANIFEST = HERE / "ROSTER_MANIFEST.json"
RECEIPT = HERE / "ROSTER_VERIFICATION.json"
EXPERIMENT = "NAN-BOUNDRESCUE-109"
CONFIG = {
    "source_seed": 20261123109, "source_roots": 16, "arms": list(ARMS),
    "checkpoint_revision": "451ed48c3273ecef7ea8faaa43c31ce529763bb1",
}


def registration():
    if native_registration() != json.loads(NATIVE_MANIFEST.read_bytes()):
        raise ValueError("The native commissioning source changed")
    paths = [Path(__file__), NATIVE_MANIFEST] + [REPO / name for name in (
        "experiments/nan_tangent87/SOURCE_CHECK.json", "experiments/nan_tangent87/source.py",
        "experiments/nan_delay39/source.py", "experiments/nan_packedcal97/source.py",
        "experiments/nan_passreuse96/SOURCE_CHECK.json", "experiments/nan_suffix80/prepare.py",
        "experiments/nan_suffix80/SOURCE_CHECK.json", "experiments/nan_comp14/sources.py",
        "experiments/nan_gated46/source.py", "experiments/nan_writer77/source.py",
        "experiments/nan_readout34/tokenizer_check.py")]
    return {"experiment": EXPERIMENT, "config": CONFIG, "budget": BUDGET,
            "files_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in paths}}


def prepare(snapshot, output):
    if snapshot.name != CONFIG["checkpoint_revision"] or registration() != json.loads(MANIFEST.read_bytes()):
        raise ValueError("Require the frozen roster and pinned tokenizer snapshot")
    with lease(BUDGET) as (check, samples):
        verify_source()
        historical = prior_prompts() | exposed()
        for loader, path in ((suffix_source, "nan_suffix80"), (tangent_source, "nan_tangent87"),
                             (packed_source, "nan_passreuse96")):
            old, _ = loader(REPO / "runs" / path / "source01", ("public",))
            historical.update(row["prompt"] for row in old["public"])
        signatures = {s for p in historical if (s := graph_signature(p)) is not None}
        public, gold = source_cases(CONFIG)
        for row in public + gold:
            for key in ("id", "root"):
                row[key] = row[key].replace("delay39:", "boundrescue109:", 1)
            row["split"] = "fresh_development"
        count = independently_verify(public, gold, CONFIG["source_roots"], historical)
        graphs = [graph_signature(row["prompt"]) for row in public if row["arm"] == "focused"]
        if None in graphs or signatures.intersection(graphs) or len(set(graphs)) != count:
            raise ValueError("A new focused graph reuses a known labeled graph")
        counts = Counter(row["answer"] for row in gold if row["arm"] == "focused")
        if dict(counts) != dict.fromkeys(NODES, 2):
            raise ValueError("The focused answer balance changed")
        tok, pins = tokenizer(snapshot)
        tokens = inputs(tok, public, snapshot, check)
        bindings = tokens["row_bindings"]
        groups = {(p["root"], p["arm"], p["world"]): bindings[p["id"]] for p in public}
        roots = sorted({row["root"] for row in public})
        for root in roots:
            for arm in ARMS:
                a, b = (groups[root, arm, w] for w in ("A", "B"))
                if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
                    raise ValueError("The paired tokenizer lengths or erased rows differ")
        if len(groups) != 2*len(roots)*len(ARMS):
            raise ValueError("The full B2 roster did not close")
    fields = {"public": public, "tokens": tokens, "gold": gold}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in fields.items():
        write_new(output / f"{name}.json", value)
    write_new(output / "verification.json", {
        "experiment": EXPERIMENT, "status": "fresh_source_and_tokenizer_verified",
        "manifest_sha256": sha(MANIFEST), "payload_sha256": {k: digest(v) for k, v in fields.items()},
        "tokenizer_pins": pins, "roots": len(roots), "prompts": len(public), "verified_focused": count,
        "b2_batches": len(groups)//2, "max_position": max(p["tokens"] for p in groups.values())+2,
        "prior_exposed_prompts": len(historical), "prior_complete_labeled_graphs": len(signatures),
        "pretrained_forwards": 0, "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(x["private_bytes"] for x in samples),
        "min_sampled_available_RAM_bytes": min(x["available_physical_bytes"] for x in samples),
    })


def worker_source(path, snapshot, check):
    proof = json.loads(RECEIPT.read_bytes())
    if (proof["status"] != "fresh_source_and_tokenizer_verified" or proof["manifest_sha256"] != sha(MANIFEST)
            or json.loads(MANIFEST.read_bytes()) != registration() or snapshot.name != CONFIG["checkpoint_revision"]):
        raise ValueError("The fresh source receipt differs")
    data = {name: json.loads((path / f"{name}.json").read_bytes()) for name in ("public", "tokens")}
    if any(digest(v) != proof["payload_sha256"][k] for k, v in data.items()):
        raise ValueError("A public input changed")
    tok, pins = tokenizer(snapshot)
    if pins != proof["tokenizer_pins"] or data["tokens"] != inputs(tok, data["public"], snapshot, check):
        raise ValueError("The worker tokenizer must reproduce the fixed inputs")
    return data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Source verification requires assertions")
    if args.freeze:
        write_new(MANIFEST, registration())
    elif args.snapshot and args.output:
        prepare(args.snapshot, args.output)
    else:
        cli.error("Require the pinned snapshot and a fresh roster directory")


if __name__ == "__main__":
    main()
