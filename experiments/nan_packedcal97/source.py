"""Bind the registered 64-root source without giving gold to the worker."""

import json
from pathlib import Path

from experiments.nan_delay39 import pilot
from experiments.nan_delay39.source import digest
from experiments.nan_gated46.source import inputs
from experiments.nan_passreuse96.verify import MANIFEST as PARENT, seal
from experiments.nan_readout34.tokenizer_check import sha, tokenizer


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "NAN-PACKEDCAL-97"
SOURCE_CHECK = REPO / "experiments/nan_passreuse96/SOURCE_CHECK.json"
CPU_CHECK = REPO / "experiments/nan_passreuse96/CPU_VERIFICATION.json"


def frozen_source():
    pins = json.loads(SOURCE_CHECK.read_bytes())
    if (json.loads(PARENT.read_bytes()) != seal() or pins["source_manifest_sha256"] != sha(PARENT)
            or pins["registered_roots"] != 64 or pins["verified_focused_worlds"] != 128
            or pins["status"] != "source_tokenizer_verified" or pins["pretrained_forwards"] != 0
            or pins["capacity_plan"]["full_B2_baseline_batches"] != 192):
        raise ValueError("The fixed PASSREUSE96 source roster changed")
    return pins


def config():
    parent = json.loads((pilot.HERE / "registration.json").read_bytes())
    return parent | {"experiment": EXPERIMENT, "source_roots": 64, "minimum_clean_both_world_roots": 48,
                     "minimum_literal_copy_both_world_roots": 64, "minimum_matched_sham_roots": 59,
                     "parity_audit_root_stride": 8}


def read_source(path, names):
    pins = frozen_source()
    records = {name: json.loads((path / f"{name}.json").read_bytes()) for name in names}
    if any(digest(value) != pins["payload_sha256"][name] for name, value in records.items()):
        raise ValueError("Frozen public or host-only source payload changed")
    return records, pins


def worker_sources(path, snapshot, guard):
    data, pins = read_source(path, ("public", "tokens"))
    tok, checked = tokenizer(snapshot)
    if checked != pins["tokenizer_pins"] or inputs(tok, data["public"], snapshot, guard) != data["tokens"]:
        raise ValueError("Worker tokenizer does not reproduce the fixed source")
    labels = data["tokens"]["bare_label_token_ids"]
    tokens = data["tokens"]["row_bindings"]
    roster = {(p["root"], p["arm"], p["world"]): (p["id"], tokens[p["id"]]["ids"]) for p in data["public"]}
    if len(roster) != 384:
        raise ValueError("The fixed native source roster does not close")
    return roster, [labels[k] for k in "ABCDEFGHIJKLMNOP"], tok.tok
