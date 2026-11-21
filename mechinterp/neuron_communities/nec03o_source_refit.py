"""NEC-03O A: source-only TopK16 refit on immutable reset-AdamW native weights.

Source worlds dev5-000..003 only. This source instrument is tied to the
NEC-03N-B checkpoint bytes, never to old NEC-03M model/SAE weights.
The new 5-bit development-evaluation and reserved contexts stay sealed.
"""
from __future__ import annotations

import argparse
from hashlib import sha1
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import (
    source_roster, fit_five_topk, source_direction_coverage,
    SOURCE_FIT_CONTEXTS, SOURCE_RANK_CONTEXTS, SOURCE_CLEAN_FLOOR,
    SAE_RECON_FLOOR, SAE_STEPS, sha_file,
)
from mechinterp.neuron_communities.nec03nb_checkpoint_reset import source_confusion
from mechinterp.neuron_communities.topk_sae import dev_ranks
from research.pef07.pef07 import Looper, model_digest

SCHEMA = "NEC-03O/reset1800-native-fresh-topk16-source-refit/v1"
LOCK = "NEC03O_SOURCE_LOCK.json"
ROOTS = (0, 1, 2, 3)
ORIGINAL_RUN = 44894980345
RESET_RUN = 44898361742
FIT_ROWS = 40 * 32
RANK_ROWS = 40 * 32
SOURCE_ROWS = 80 * 32
SOURCE_GEOMETRY_GROUPS = 8


def read(path: Path):
    return json.loads(path.read_text())


def verify_lock():
    root = Path(__file__).resolve().parents[2]
    path = root / "mechinterp/neuron_communities" / LOCK
    cfg = read(path)
    expected = {
        "schema": "NEC-03O/frozen-reset1800-source-SAE-fit-lock/v1",
        "original_model_run_id": ORIGINAL_RUN,
        "reset1800_model_run_id": RESET_RUN,
        "five_input_registry_sha256": registration()["registry_sha256"],
        "four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "source_world_indices": list(ROOTS),
        "source_groups": 80,
        "source_fit_groups": 40,
        "source_rank_groups": 40,
        "source_geometry_groups": SOURCE_GEOMETRY_GROUPS,
        "source_focal_masks": 32,
        "sae_features": 128,
        "sae_active_k": 16,
        "sae_updates": SAE_STEPS,
        "native_update_count": 1800,
        "new_native_training_updates": 0,
        "development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0,
        "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise AssertionError("NEC-03O source-only registration changed")
    if cfg.get("max_cosine_gap") != .25 or cfg.get("minimum_angle_coverage") != .8 or cfg.get("minimum_physical_coverage") != .8 or cfg.get("minimum_median_retained_dose") != .25 or cfg.get("maximum_normalized_sae_mse") != .03 or cfg.get("minimum_native_source_accuracy") != .9:
        raise AssertionError("NEC-03O source feasibility gates changed")
    for name, expected_blob in cfg["git_blob_sha1"].items():
        blob = (root / name).read_bytes()
        actual = sha1(b"blob " + str(len(blob)).encode() + b"\x00" + blob).hexdigest()
        if actual != expected_blob:
            raise AssertionError("NEC-03O frozen implementation changed: " + name)
    cfg["lock_sha256"] = sha_file(path)
    return cfg


def restore_model(path: Path, world: dict, arm: str, report: dict):
    if sha_file(path) != report["new_checkpoint_sha256"]:
        raise AssertionError("new native checkpoint file SHA256 differs from root receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["schema"] != "NEC-03N-B/frozen900-checkpoint-reset-AdamW-900-source/v1"
        or payload["world_sha256"] != world["world_sha256"]
        or payload["arm"] != arm
        or payload["total_model_updates"] != 1800
        or payload["new_reset_AdamW_steps"] != 900
        or payload["model_weights_sha256"] != report["new_native_model_sha256"]
        or payload["prior_frozen900_model_sha256"] != report["old_native_model_sha256"]):
        raise AssertionError("wrong reset1800 native ancestry")
    model = Looper().eval()
    model.load_state_dict(payload["model"], strict=True)
    if model_digest(model) != report["new_native_model_sha256"]:
        raise AssertionError("loaded 1800 native model tensor digest changed")
    return model


def run(index: int, original: Path, reset: Path, output: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_lock()
    if index not in ROOTS:
        raise PermissionError("NEC-03O phase A is restricted to four already opened source worlds")
    world = worlds()[index]
    catalog, manifest = context_catalog(world)
    prior, updated = read(original / "source_pilot_receipt.json"), read(reset / "checkpoint_reset_source_report.json")
    original_audit = read(original / "independent_source_audit.json")
    reset_audit = read(reset / "independent_warmstart_audit.json")
    if (read(original / "world_spec.json") != world
        or read(original / "source_split_manifest.json") != manifest
        or prior["world_sha256"] != world["world_sha256"]
        or updated["world_sha256"] != world["world_sha256"]
        or updated["world_index"] != index
        or prior["source_manifest_sha256"] != manifest["source_sha256"]
        or updated["original_NEC03M_run"] != ORIGINAL_RUN
        or not updated["old_optimizer_state_reset_at900"]
        or updated["exact_continuous_AdamW_1800_claim"] is not False
        or original_audit["world_sha256"] != world["world_sha256"]
        or original_audit["raw_receipt_sha256"] != sha_file(original / "source_pilot_receipt.json")
        or not original_audit["source_direction_active_candidates_independently_reenumerated"]
        or not reset_audit["two_native_models_audited"]
        or reset_audit["world_sha256"] != world["world_sha256"]
        or not reset_audit["native_optimizer_moments_reset_at_900"]
        or updated["new_sae_models_fit"] != 0):
        raise AssertionError("original or reset-native source ancestry not independently certified")
    for receipt in (prior, updated):
        if any(receipt.get(key, 0) != 0 for key in (
            "development_evaluation_context_groups_scored",
            "new_five_input_development_evaluation_context_groups_scored",
            "reserved_context_groups_scored", "new_embargoed_worlds_scored",
            "five_input_embargoed_worlds_scored",
            "original_confirmation_worlds_scored",
            "original_four_input_confirmation_worlds_scored")):
            raise AssertionError("upstream experiment opened development or confirmation contexts")
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ("composed", "additive"):
        old_native = prior["arms"][arm]["native"]
        new_native = updated["arms"][arm]
        if (new_native["old_native_model_sha256"] != old_native["final_model_weights_sha256"]
            or new_native["old_frozen900_checkpoint_sha256"] != old_native["checkpoint_sha256"]
            or sha_file(original / old_native["checkpoint_file"]) != old_native["checkpoint_sha256"]):
            raise AssertionError("1800 native model is not an exact-900 checkpoint descendant")
        model = restore_model(reset / new_native["new_checkpoint_file"], world, arm, new_native)
        scores = source_confusion(model, catalog["train"], world, arm=arm)
        if (scores["correct"] != new_native["new_source_score"]["correct"]
            or scores["source_prediction_labels"] != new_native["new_source_score"]["source_prediction_labels"]):
            raise AssertionError("native source truth labels and predictions changed")
        chosen, graph, roster = source_roster(model, catalog, world)
        frozen_roster = {
            "arm": arm, "world_sha256": world["world_sha256"],
            "source_sha256": manifest["source_sha256"],
            "new_native_model_sha256": model_digest(model),
            "native_checkpoint_sha256": new_native["new_checkpoint_sha256"],
            "roster": roster,
        }
        (output / f"{arm}.source_roster.json").write_bytes(encoded(frozen_roster))
        fit_receipt, angle = None, None
        angle_failure = None
        if chosen is not None:
            sae, mean, scale, fit_receipt = fit_five_topk(
                model, catalog["train"][:FIT_ROWS], seed=world["native_model_seed"] + (100000 if arm == "additive" else 0))
            ranks, rank_sha = dev_ranks(model, sae, mean, scale, catalog["train"][FIT_ROWS:FIT_ROWS + RANK_ROWS])
            path = output / f"{arm}.fresh1800_topk16.pt"
            torch.save({
                "schema": SCHEMA,
                "arm": arm,
                "world_sha256": world["world_sha256"],
                "model_sha256": model_digest(model),
                "native_checkpoint_sha256": new_native["new_checkpoint_sha256"],
                "source_manifest_sha256": manifest["source_sha256"],
                "source_fit_context_groups": 40,
                "source_rank_context_groups": 40,
                "state": sae.state_dict(), "mean": mean, "scale": scale,
                "rank": ranks, "rank_sha256": rank_sha,
            }, path)
            fit_receipt["source_dictionary_file"] = path.name
            fit_receipt["source_dictionary_sha256"] = sha_file(path)
            fit_receipt["source_rank_sha256"] = rank_sha
            fit_receipt["source_rank_context_groups"] = 40
            try:
                angle = source_direction_coverage(model, catalog, world, chosen, graph, sae, mean, scale, ranks)
            except (ValueError, AssertionError, RuntimeError) as err:
                # Fixed intent-to-treat denominator. Invalid or missing
                # active-code nulls count as unavailable source controls.
                angle_failure = f"{type(err).__name__}: {err}"
        eligible = (
            scores["accuracy"] >= SOURCE_CLEAN_FLOOR
            and chosen is not None
            and fit_receipt is not None
            and fit_receipt["reconstruction_mse_normalized"] <= SAE_RECON_FLOOR
            and angle is not None and angle["source_null_feasible"])
        old_sae = prior["arms"][arm]["SAE_source"]
        arms[arm] = {
            "native_reset1800": {
                "new_checkpoint_file": new_native["new_checkpoint_file"],
                "new_checkpoint_sha256": new_native["new_checkpoint_sha256"],
                "new_model_sha256": model_digest(model),
                "source_score": scores,
                "old_model_sha256": old_native["final_model_weights_sha256"],
                "old_checkpoint_sha256": old_native["checkpoint_sha256"],
            },
            "source_site_kind": ("fallback" if roster["fallback_used"] else "graph") if chosen else "none",
            "source_candidate": chosen,
            "source_roster_sha256": sha_file(output / f"{arm}.source_roster.json"),
            "SAE_source": fit_receipt,
            "angle_source": angle,
            "angle_failure": angle_failure,
            "old_900_sae_source_mse": None if old_sae is None else old_sae["reconstruction_mse_normalized"],
            "new_1800_sae_source_mse": None if fit_receipt is None else fit_receipt["reconstruction_mse_normalized"],
            "native_source_90pct": scores["accuracy"] >= SOURCE_CLEAN_FLOOR,
            "source_model_eligible": bool(eligible),
        }
        if model_digest(model) != new_native["new_native_model_sha256"]:
            raise AssertionError("new SAE fit mutated native checkpoint")
    report = {
        "schema": SCHEMA, "world_id": world["world_id"], "world_index": index,
        "world_sha256": world["world_sha256"],
        "five_input_registry_sha256": registration()["registry_sha256"],
        "four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "source_manifest_sha256": manifest["source_sha256"],
        "original_source_receipt_sha256": sha_file(original / "source_pilot_receipt.json"),
        "native_reset_receipt_sha256": sha_file(reset / "checkpoint_reset_source_report.json"),
        "source_lock_sha256": lock["lock_sha256"],
        "source_fit_groups": 40, "source_rank_groups": 40, "source_geometry_groups": 8,
        "new_native_training_updates": 0,
        "development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0,
        "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "all_arms_in_intent_to_treat_denominator": True,
        "arms": arms,
        "both_arms_source_eligible": all(a["source_model_eligible"] for a in arms.values()),
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    (output / "nec03o_source_refit_receipt.json").write_bytes(encoded(report))
    print("NEC03O_SOURCE_RETRAINED_TOPK16_REFIT_WRITTEN", world["world_id"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-index", type=int, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--reset-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.world_index, a.original_root, a.reset_root, a.out)
