"""Bind a disjoint 16-root NF4 source roster for the repaired writer pilot."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.nan_delay39.source import digest, independently_verify, source_cases
from experiments.nan_gated46.source import inputs
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha, tokenizer, verify_source
from experiments.nan_suffix80.prepare import load as load_suffix, prior_prompts
from experiments.nan_tangent87.source import BUDGET, NODES, graph_signature, load as load_tangent, seal as parent_seal
from experiments.nan_writer77.source import exposed


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "NAN-NATIVEREPAIR-91"
PIN = "451ed48c3273ecef7ea8faaa43c31ce529763bb1"
SOURCE = HERE / "SOURCE_MANIFEST.json"
CHECK = HERE / "SOURCE_CHECK.json"
CONFIG = {"experiment": EXPERIMENT, "source_seed": 2026112391, "source_roots": 16, "backend": "nf4",
          "arms": ["focused", "copy", "focused_erased"], "checkpoint_revision": PIN,
          "minimum_clean_both_world_roots": 12, "minimum_literal_copy_both_world_roots": 16,
          "minimum_meaningful_delayed_margin_logit": .125, "formal_comparisons": 0}


def source_manifest():
    parent = REPO / "experiments/nan_tangent87/SOURCE_MANIFEST.json"
    inherited = parent_seal()
    if json.loads(parent.read_bytes()) != inherited:
        raise ValueError("Prior fresh-source freeze changed")
    paths = [HERE / p for p in (".gitattributes", "__init__.py", "PROTOCOL.md", "source.py")]
    paths += [parent, REPO / "experiments/nan_tangent87/SOURCE_CHECK.json"]
    return {"experiment": EXPERIMENT, "config": CONFIG, "files_sha256": inherited["files_sha256"] |
            {p.relative_to(REPO).as_posix(): sha(p) for p in paths}}


def prepare(snapshot, out):
    frozen = source_manifest()
    if json.loads(SOURCE.read_bytes()) != frozen or snapshot.name != PIN:
        raise ValueError("Source or tokenizer pin changed")
    with lease(BUDGET) as (check, samples):
        verify_source()
        prior = prior_prompts() | exposed()
        suffix, _ = load_suffix(REPO / "runs/nan_suffix80/source01", ("public",))
        tangent, _ = load_tangent(REPO / "runs/nan_tangent87/source01", ("public",))
        prior.update(row["prompt"] for row in suffix["public"] + tangent["public"])
        seen = {signature for prompt in prior if (signature := graph_signature(prompt)) is not None}
        public, gold = source_cases(CONFIG)
        for row in public + gold:
            for key in ("id", "root"):
                row[key] = row[key].replace("delay39:", "nativerepair91:", 1)
            row["split"] = "fresh_development"
        count = independently_verify(public, gold, CONFIG["source_roots"], prior)
        signatures = [graph_signature(row["prompt"]) for row in public if row["arm"] == "focused"]
        if None in signatures or seen.intersection(signatures) or len(set(signatures)) != count:
            raise ValueError("Focused graphs are incomplete or overlap prior sources")
        if Counter(p["answer"] for p in gold if p["arm"] == "focused") != dict.fromkeys(NODES, 2):
            raise ValueError("The source answer roster is imbalanced")
        tok, pins = tokenizer(snapshot)
        tokens = inputs(tok, public, snapshot, check)
        pairs = {}
        for row in public:
            pairs.setdefault((row["root"], row["arm"]), {})[row["world"]] = tokens["row_bindings"][row["id"]]
        for (_, arm), pair in pairs.items():
            a, b = pair["A"], pair["B"]
            if a["tokens"] != b["tokens"] or (arm == "focused_erased" and a != b):
                raise ValueError("Paired prompt length or erased parity changed")
    assert source_manifest() == frozen
    out.mkdir(parents=True, exist_ok=False)
    fields = {"public": public, "gold": gold, "tokens": tokens}
    for name, value in fields.items():
        write_new(out / f"{name}.json", value)
    proof = {"experiment": EXPERIMENT, "status": "source_tokenizer_verified", "source_manifest_sha256": sha(SOURCE),
             "payload_sha256": {k: digest(v) for k, v in fields.items()}, "tokenizer_pins": pins,
             "verified_focused": count, "roots": CONFIG["source_roots"], "prompts": len(public),
             "b2_batches": len(pairs), "max_position": max(r["tokens"] for r in tokens["row_bindings"].values()) + 2,
             "prior_exposed_prompts": len(prior), "prior_complete_labelled_graphs": len(seen),
             "pretrained_forwards": 0, "sealed_test_accessed": False,
             "max_sampled_private_bytes": max(s["private_bytes"] for s in samples)}
    write_new(out / "SOURCE_CHECK.json", proof)
    print(proof, flush=True)


def load(folder, keys):
    proof = json.loads(CHECK.read_bytes())
    if (proof["source_manifest_sha256"] != sha(SOURCE) or json.loads(SOURCE.read_bytes()) != source_manifest()
            or proof["experiment"] != EXPERIMENT or proof["status"] != "source_tokenizer_verified"):
        raise ValueError("Frozen source receipt changed")
    data = {key: json.loads((folder / f"{key}.json").read_bytes()) for key in keys}
    if any(digest(value) != proof["payload_sha256"][key] for key, value in data.items()):
        raise ValueError("Immutable prompt or oracle payload changed")
    return data, proof


def worker_inputs(folder, snapshot, check):
    payload, proof = load(folder, ("public", "tokens"))
    tok, pins = tokenizer(snapshot)
    if pins != proof["tokenizer_pins"] or inputs(tok, payload["public"], snapshot, check) != payload["tokens"]:
        raise ValueError("Worker tokenizer did not reproduce the sealed public bindings")
    return payload


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--snapshot", type=Path)
    cli.add_argument("--work", type=Path)
    args = cli.parse_args()
    if not __debug__:
        cli.error("Require source assertions")
    if args.freeze:
        write_new(SOURCE, source_manifest())
    elif args.snapshot and args.work:
        prepare(args.snapshot, args.work)
    else:
        cli.error("Require the pinned snapshot and a fresh work directory")


if __name__ == "__main__":
    main()
