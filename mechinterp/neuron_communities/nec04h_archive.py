"""Read SHA-pinned NEC-04F archives and run zero-forward diagnostics."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04f_oracles import ROOTS, SOURCE, VALIDATION
from mechinterp.neuron_communities.nec04h_oracles import arrival_stratum, verify_labels


PREFLIGHT = "nec04f-original-six-root-pretraining-and-independent-aux-oracle-seal"
PINS = json.loads(Path(__file__).with_name("nec04h_artifacts.json").read_text(encoding="utf-8"))


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def open_verified(inputs, name):
    path = inputs / f"{name}.zip"
    if file_sha(path) != PINS["artifacts"][name]["sha256"]:
        raise ValueError(f"original artifact hash changed: {name}")
    return ZipFile(path)


def read_root(inputs, root):
    if root not in ROOTS:
        raise PermissionError("only original roots 12–17 are authorized")
    with open_verified(inputs, PREFLIGHT) as arc:
        ledger_bytes = arc.read("pretraining/nec04f_pretraining_oracles.json")
        ledger = json.loads(ledger_bytes)
        ref = ledger["roots"][root - ROOTS[0]]
        rows_bytes = arc.read(f"pretraining/nec04f_root_{root}.jsonl")
    rows = [json.loads(line) for line in rows_bytes.splitlines()]
    world, _ = gym.calibration_world(root)
    if (ledger["root_indices"] != list(ROOTS) or ledger["lock_sha256"] != PINS["training_lock_sha256"]
            or ledger["preflight_source_git_sha"] != PINS["training_commit"]
            or ref["index"] != root or ref["world_sha256"] != world["world_sha256"]
            or ref["program_sha256"] != world["program_sha256"]
            or ref["rows_sha256"] != sha256(rows_bytes).hexdigest() or len(rows) != SOURCE + VALIDATION):
        raise ValueError("original source split, program or SHA changed")
    source_contexts, owned = gym.contexts_for_world(root)
    panels = [*source_contexts, *owned["development"]]
    if len(source_contexts) != 80 or len(panels) != 104 or len(set(panels)) != 104:
        raise ValueError("unexpected context ownership")
    labels = []
    for i, row in enumerate(rows):
        group, mask = divmod(i, 32)
        split = "source_train" if i < SOURCE else "calibration_world_validation"
        bits = row["history_bits12"]
        focal_mask = sum(bits[pos] << k for k, pos in enumerate(world["focal_positions"]))
        context = tuple(bits[pos] for pos in world["context_positions"])
        if (row["split"] != split or row["root_index"] != root or row["context_group"] != group
                or row["focal_mask"] != mask or focal_mask != mask or context != tuple(panels[group])):
            raise ValueError("archived input escaped its source or owned split")
        labels.append(verify_labels(world, row))

    with open_verified(inputs, f"nec04f-native-root-{root}") as arc:
        result = json.loads(arc.read("nec04f_native_root_result.json"))
        arm = result["arms"]["aux_prefix"]
        filename = f"nec04f.root{root}.aux_prefix.checkpoints.json"
        manifest_bytes = arc.read(filename)
        manifest = json.loads(manifest_bytes)
        checkpoint, = (c for c in manifest["checkpoints"] if c["step"] == 900)
        observation, = (o for o in arm["observations"] if o["step"] == 900)
        checkpoint_bytes = arc.read(checkpoint["file"])
        original_audit = json.loads(arc.read("nec04f_independent_native_audit.json"))
    if (result["root_index"] != root or result["world_sha256"] != world["world_sha256"]
            or result["original_program_sha256"] != world["program_sha256"]
            or result["source_lock_sha256"] != PINS["training_lock_sha256"]
            or result["training_git_commit_sha"] != PINS["training_commit"]
            or result["pretraining_oracle_ledger_sha256"] != sha256(ledger_bytes).hexdigest()
            or result["original_source_validation_sha256"] != ref["rows_sha256"]
            or sha256(manifest_bytes).hexdigest() != arm["checkpoint_seal_manifest_sha256"]
            or manifest != arm["checkpoint_seal_manifest"]
            or checkpoint["sha256"] != sha256(checkpoint_bytes).hexdigest()
            or observation["original_checkpoint_sha256"] != checkpoint["sha256"]
            or observation["original_model_sha256"] != checkpoint["model_sha256"]
            or not original_audit["all_six_native_full_and_four_auxiliary_logits_independently_replayed"]):
        raise ValueError("the original aux-prefix checkpoint or archived logits lost provenance")
    return world, ref, rows, labels, checkpoint_bytes, observation


def archived_root(root, world, ref, rows, labels, observation):
    native = observation["all_final_native_logits"]
    auxiliary = observation["all_intermediate_auxiliary_logits"]
    if len(native) != len(rows) or len(auxiliary) != len(rows):
        raise ValueError("archived logits incomplete")
    counts = {str(sel): Counter() for sel in (0, 1)}
    for i in range(SOURCE, len(rows)):
        selector, phase = labels[i]
        true = rows[i]["composed_label"]
        base = int(native[i][1] > native[i][0])
        prefix = auxiliary[i][-1]
        diagnostic = (int(prefix[1] > prefix[0]) ^ phase) if selector else base
        counts[str(selector)].update(n=1, native=int(base == true), oracle=int(diagnostic == true))
    return {"root": root, "checkpoint_sha256": observation["original_checkpoint_sha256"],
            "rows_sha256": ref["rows_sha256"],
            "arrival": arrival_stratum(world), "oracle_true_phase_and_selector": counts,
            "validation_labels_independently_verified": VALIDATION, "model_forwards": 0}


def run_archived(inputs):
    rows = []
    for root in ROOTS:
        world, ref, data, labels, _, observation = read_root(inputs, root)
        rows.append(archived_root(root, world, ref, data, labels, observation))
    return {"status": "REDESIGN_OR_PILOT_ONLY", "historical_artifacts": PINS, "roots": rows}


def write_json(path, result):
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, run_archived(args.inputs))
