"""NEC-04F: SHA-owned calibration roots12..17 and matched XOR auxiliary teachers.

Original NEC04A source corpus, first 80 complete context blocks per world.
24 distinct owned validation-context blocks from the SAME calibration worlds.
Balanced-shuffle controls exist on SOURCE rows only. No developmental WORLD
roots, confirmations, calibration reserved contexts or old NEC03P data.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mechinterp.neuron_communities import nec04a_boolean_gym as gym
from mechinterp.neuron_communities.nec04a_independent_oracle import oracle

SCHEMA = "NEC-04F/original-six-calibration-root-auxiliary-teacher-ledger/v1"
ROOTS = tuple(range(12, 18))
ARMS = ("aux_shuffled", "aux_prefix", "vanilla")
STAGES = (2, 3, 4, 5)
STEPS = (240, 900)
SOURCE = 80 * 32
VALIDATION = 24 * 32
ORIGINAL_SHA = "82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0"
REGISTRY_SHA = "88bf58d32fdf15a1270986c238afb3def1a7e6371d3eee64c4edd264c9cf041e"
SHUFFLE_DOMAIN = b"NEC04F/balanced-shuffled-source-aux-label/v1|"
SEED_DOMAIN = b"NEC04F/paired-native-initial-seed/v1|"
CODE_FILES = (
    "mechinterp/neuron_communities/NEC04F_TRAIN_LOCK.json",
    "mechinterp/neuron_communities/NEC04F_PROTOCOL.md",
    "mechinterp/neuron_communities/nec04f_oracles.py",
    "mechinterp/neuron_communities/nec04f_independent_oracles.py",
    "mechinterp/neuron_communities/nec04f_native.py",
    "mechinterp/neuron_communities/nec04f_audit.py",
    "mechinterp/neuron_communities/nec04f_aggregate.py",
    "mechinterp/neuron_communities/test_nec04f.py",
    ".github/workflows/nec04f-xor-auxiliary.yml",
    "mechinterp/neuron_communities/nec04a_boolean_gym.py",
    "mechinterp/neuron_communities/nec04a_independent_oracle.py",
    "research/pef07/pef07.py",
    "lsa/mini_transformer_probe.py",
)


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def lock():
    p = Path(__file__).with_name("NEC04F_TRAIN_LOCK.json")
    cfg = json.loads(p.read_text())
    frozen = {
        "schema": "NEC-04F/fixed-original-calibration-roots12to17-matched-XOR-intermediate-auxiliary/v1",
        "issue": 305,
        "root_indices": list(ROOTS),
        "original_nec04a_run": 44983509977,
        "original_source_sha256": ORIGINAL_SHA,
        "original_registry_sha256": REGISTRY_SHA,
        "source_context_groups": 80,
        "source_rows_per_root": SOURCE,
        "owned_calibration_validation_context_groups": 24,
        "owned_validation_rows_per_root": VALIDATION,
        "reserved_context_groups_per_root_unopened": 24,
        "training_arms": list(ARMS),
        "primary_comparison": ["aux_prefix", "aux_shuffled"],
        "focal_stage_counts": list(STAGES),
        "continuous_optimizer_steps": 900,
        "checkpoint_steps": list(STEPS),
        "learning_rate": .003,
        "weight_decay": 0.,
        "batch_size": 64,
        "gradient_clip_norm": 1.,
        "auxiliary_weight": .5,
        "vanilla_auxiliary_weight": 0.,
        "primary_minimum_mean_delta": .10,
        "primary_minimum_positive_roots": 4,
        "primary_minimum_aux_prefix_parity_accuracy": .60,
        "alternate_branch_minimum_mean_delta": -.05,
        "same_initial_model_and_aux_weights_in_all_arms": True,
        "same_minibatch_source_sequence_all_arms": True,
        "aux_shuffled_balanced_per_source_context_and_stage": True,
        "new_development_world_roots_materialized": 0,
        "new_confirmation_world_roots_materialized": 0,
        "original_reserved_contexts_scored": 0,
        "old_nec03p_causal_outcomes_scored": 0,
        "native_source_SAE_trained": False,
        "sham_interventions_scored": False,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(name) != value for name, value in frozen.items()):
        raise AssertionError("NEC04F precommitted worlds/arms/optimizer/gates changed")
    return cfg, file_sha(p)


def code_identity():
    root = Path(__file__).resolve().parents[2]
    return {name: file_sha(root / name) for name in CODE_FILES}


def seed(index):
    if type(index) is not int or index not in ROOTS:
        raise PermissionError("NEC04F only original previously untrained calibration roots12..17")
    value = sha256(SEED_DOMAIN + gym.seed_for_identity("calibration", index)).digest()
    return 1 + int.from_bytes(value[:8], "big") % (2**31-2)


def xor_prefix(bits, focal_positions):
    ordered = sorted(focal_positions)
    if len(ordered) != 5 or len(set(ordered)) != 5:
        raise ValueError("original five focal positions required")
    value = 0
    result = {}
    for k, position in enumerate(ordered, start=1):
        value ^= bits[position]
        if k in STAGES:
            result[str(k)] = value
    return result


def shuffle_map(index, group, k):
    """Balanced random-looking source control, independent of teacher labels."""
    if index not in ROOTS or group not in range(80) or k not in STAGES:
        raise PermissionError("balanced pseudo teacher only original source panels")
    key = SHUFFLE_DOMAIN + gym.seed_for_identity("calibration", index)
    order = sorted(range(32), key=lambda mask: (
        sha256(key + group.to_bytes(2,"big") + bytes([k,mask])).digest(), mask
    ))
    top = set(order[:16])
    labels = [int(mask in top) for mask in range(32)]
    assert len(labels) == 32 and sum(labels) == 16
    return labels


def input_history(focal, context, context_bits, mask):
    x = [0] * 12
    for k, position in enumerate(focal):
        x[position] = (mask >> k) & 1
    for k, position in enumerate(context):
        x[position] = context_bits[k]
    return x


def materialize(original: Path, out: Path):
    cfg, lock_sha = lock()
    base = original / "nec04a-calibration"
    manifest = json.loads((base / "nec04a_calibration_source_manifest.json").read_text())
    registry = json.loads((base / "nec04a_world_root_commitments.json").read_text())
    world_file = base / "nec04a_calibration_worlds.jsonl"
    corpus_file = base / "nec04a_calibration_source_train.jsonl"
    if (
        manifest["source_corpus_sha256"] != ORIGINAL_SHA
        or file_sha(corpus_file) != ORIGINAL_SHA
        or registry["registration_sha256"] != REGISTRY_SHA
        or file_sha(world_file) != manifest["calibration_world_specs_sha256"]
    ):
        raise AssertionError("original NEC04A parent seed/source corpus SHA changed")
    worlds = [json.loads(x) for x in world_file.read_text().splitlines()]
    corpus = [json.loads(x) for x in corpus_file.read_text().splitlines()]
    if len(worlds) != 24 or len(corpus) != 61440:
        raise AssertionError("NEC04A original calibration-world population incomplete")
    out.mkdir(parents=True, exist_ok=True)
    roots = []
    for index in ROOTS:
        original_world = worlds[index]
        world, truth = gym.calibration_world(index)
        if world != original_world or world["root_index"] != index:
            raise AssertionError("original immutable NEC04A program/world altered")
        if world["canonical_program"][:2] != [
            "MUX", ["VAR", world["context_positions"][0]]
        ]:
            raise AssertionError("original context-selected five-bit parity AST changed")
        source_contexts, split_contexts = gym.contexts_for_world(index)
        owned = split_contexts["development"]
        if (
            len(source_contexts) != 80 or len(owned) != 24
            or len(split_contexts["reserved"]) != 24
            or set(source_contexts) & set(owned)
            or set(source_contexts) & set(split_contexts["reserved"])
            or set(owned) & set(split_contexts["reserved"])
        ):
            raise AssertionError("original source/owned/reserved context ownership changed")
        stages = {str(k): sorted(world["focal_positions"])[k-1]+1 for k in STAGES}
        staged_shuffles = {k: [shuffle_map(index,g,int(k)) for g in range(80)]
                           for k in map(str,STAGES)}
        source_rows = []
        original_rows = corpus[index*SOURCE:(index+1)*SOURCE]
        for n,old in enumerate(original_rows):
            group,mask=divmod(n,32)
            bits=input_history(
                world["focal_positions"], world["context_positions"],
                source_contexts[group], mask
            )
            i = sum(v<<j for j,v in enumerate(bits))
            if (old["world_id"] != world["world_id"] or old["cohort"] != "calibration"
                or old["split"] != "source_train"
                or old["source_context_group"] != group
                or old["focal_mask"] != mask
                or old["history_bits12"] != bits
                or old["label"] != ((truth >> i)&1)
                or old["world_sha256"] != world["world_sha256"]):
                raise AssertionError("original NEC04A source/circuit labels or whole groups altered")
            source_rows.append({
                "root_index": index, "world_id": world["world_id"],
                "split": "source_train", "context_group": group, "focal_mask": mask,
                "history_bits12": bits, "composed_label": old["label"],
                "mux_selector": bits[world["context_positions"][0]],
                "xor_prefix": xor_prefix(bits,world["focal_positions"]),
                "shuffled_aux": {k:staged_shuffles[k][group][mask] for k in stages},
            })
        validation = []
        for local_group,ctx in enumerate(owned):
            for mask in range(32):
                bits = input_history(
                    world["focal_positions"], world["context_positions"],ctx,mask
                )
                validation.append({
                    "root_index":index,"world_id":world["world_id"],
                    "split":"calibration_world_validation",
                    "context_group":80+local_group,"focal_mask":mask,
                    "history_bits12":bits,
                    "composed_label":int(oracle(world["canonical_program"],bits)),
                    "mux_selector":bits[world["context_positions"][0]],
                    "xor_prefix":xor_prefix(bits,world["focal_positions"]),
                })
        combined=source_rows+validation
        if (
            len(source_rows)!=SOURCE or len(validation)!=VALIDATION
            or len({tuple(r["history_bits12"]) for r in combined})!=SOURCE+VALIDATION
            or any(sum(staged_shuffles[str(k)][g])!=16
                   for k in STAGES for g in range(80))
        ):
            raise AssertionError("original world-source and auxiliary-control data incomplete")
        path=out/f"nec04f_root_{index}.jsonl"
        blob=b"".join(json_bytes(row) for row in combined)
        path.write_bytes(blob)
        roots.append({
            "index":index, "world_id":world["world_id"],
            "difficulty_tier":world["difficulty_tier"],
            "world_sha256":world["world_sha256"],
            "program_sha256":world["program_sha256"],
            "truth_4096_sha256":world["truth_table_4096_sha256"],
            "stage_visits":stages,
            "matched_seed":seed(index),
            "source_cases":SOURCE,
            "owned_validation_cases":VALIDATION,
            "source_parity_cases":sum(r["mux_selector"] for r in source_rows),
            "validation_parity_cases":sum(r["mux_selector"] for r in validation),
            "rows_file":path.name,"rows_sha256":sha256(blob).hexdigest(),
        })
    ledger={
        "schema":SCHEMA,
        "lock_sha256":lock_sha,
        "original_source_corpus_sha256":ORIGINAL_SHA,
        "original_registry_sha256":REGISTRY_SHA,
        "preflight_source_git_sha":os.environ.get("GITHUB_SHA","UNPINNED_LOCAL"),
        "source_code_sha256_before_training":code_identity(),
        "root_indices":list(ROOTS),
        "training_arms":list(ARMS),
        "stage_k":list(STAGES),
        "checkpoint_steps":list(STEPS),
        "original_source_rows_per_root":SOURCE,
        "original_owned_validation_rows_per_root":VALIDATION,
        "optimizer_updates_before_preflight":0,
        "source_SAE_updates":0,
        "reserved_contexts_scored":0,
        "new_development_world_roots_materialized":0,
        "new_confirmation_world_roots_materialized":0,
        "old_NEC03P_worlds_scored":0,
        "FSI_01_gate":"REDESIGN_OR_PILOT_ONLY",
        "roots":roots,
    }
    (out/"nec04f_pretraining_oracles.json").write_bytes(json_bytes(ledger))
    print("NEC04F_SOURCE_ORIGINAL_SIX_ROOT_AUX_TEACHERS_FROZEN",len(roots),
          SOURCE*len(roots),VALIDATION*len(roots),flush=True)
    return ledger


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    materialize(a.original,a.out)


if __name__=="__main__":
    main()
