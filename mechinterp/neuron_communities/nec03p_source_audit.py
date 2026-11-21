"""NEC-03P independent source-recruitment checkpoint/SAE audit.

No optimizer training here. Independently reload both native milestones,
recalculate all original source truth labels/predictions, check shifted
source minibatch generator, SAE reconstruction/rank, and enumerate every
physically projected disjoint control from raw 32D source contributions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03p_source_recruit import (
    SCHEMA, MODEL_SCHEMA, ROOTS, SOURCE_FIT_ROWS, SOURCE_RANK_ROWS,
    read, verify_lock,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import source_roster, sha_file
from mechinterp.neuron_communities.nec03nb_checkpoint_reset import (
    source_confusion, batch_digest,
)
from mechinterp.neuron_communities.nec03o_source_audit import audit_geometry
from mechinterp.neuron_communities.nec03k_source_audit import close
from mechinterp.neuron_communities.source_sae import collect_source, seal_array
from mechinterp.neuron_communities.topk_sae import TopKSAE, dev_ranks
from research.pef07.pef07 import Looper, model_digest

AUDIT_SCHEMA = "NEC-03P/independent-development-source-recruitment-audit/v1"


def read_model(path: Path, sha: str, world: dict, arm: str, expected_model: str, *, old: bool):
    if sha_file(path) != sha:
        raise AssertionError("prospective source-native checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    seed = world["native_model_seed"] + (100000 if arm == "additive" else 0)
    if (state["world_sha256"] != world["world_sha256"] or state["arm"] != arm
        or state["model_weights_sha256"] != expected_model):
        raise AssertionError("wrong prospective source world or checkpoint identity")
    if old:
        if state["steps"] != 900 or state["seed"] != seed:
            raise AssertionError("first native source milestone is not registered 900 updates")
    else:
        if (state["schema"] != MODEL_SCHEMA or state["model_seed"] != seed
            or state["native_updates_before_reset"] != 900
            or state["new_AdamW_updates_after_reset"] != 900
            or state["total_native_updates"] != 1800
            or not state["AdamW_moments_reset"]
            or not state["source_training_only"]):
            raise AssertionError("prospective reset1800 native training stage changed")
    model = Looper().eval()
    model.load_state_dict(state["model"], strict=True)
    if model_digest(model) != expected_model:
        raise AssertionError("native checkpoint parameters differ from reported digest")
    return model, state


def audit(index: int, folder: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_lock()
    if index not in ROOTS:
        raise PermissionError("audit cannot open undeclared source or confirmation root")
    world = worlds()[index]
    catalog, manifest = context_catalog(world)
    report = read(folder / "nec03p_source_receipt.json")
    if (read(folder / "world_spec.json") != world
        or read(folder / "source_split_manifest.json") != manifest
        or report["schema"] != SCHEMA
        or report["world_index"] != index
        or report["world_sha256"] != world["world_sha256"]
        or report["source_manifest_sha256"] != manifest["source_sha256"]
        or report["five_input_registry_sha256"] != registration()["registry_sha256"]
        or report["four_input_confirmation_sha256"] != old_registry()["confirmation_full_spec_sha256"]
        or report["source_lock_sha256"] != lock["lock_sha256"]
        or not report["optimizer_reset_at900"]
        or report["exact_continuous_adamw_1800_claim"] is not False
        or (report["source_training_groups"], report["source_fit_groups"],
            report["source_rank_groups"], report["source_geometry_groups"],
            report["source_focal_masks"], report["sae_steps"]) != (80,40,40,8,32,480)
        or not report["all_arms_in_intent_to_treat_denominator"]
        or report["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"
        or any(report[key] for key in (
            "new_development_evaluation_groups_scored", "reserved_context_groups_scored",
            "embargoed_worlds_scored", "four_input_confirmation_worlds_scored"))):
        raise AssertionError("prospective source-only cohort or registered protocol altered")
    arms = {}
    for arm in ("composed", "additive"):
        row = report["arms"][arm]
        n = row["native"]
        old_receipt = n["old900_native"]
        old_model, old_ckpt = read_model(
            folder / old_receipt["checkpoint_file"], old_receipt["checkpoint_sha256"],
            world, arm, old_receipt["final_model_weights_sha256"], old=True)
        if (n["seed"] != old_receipt["training_seed"] or n["seed"] !=
            world["native_model_seed"] + (100000 if arm == "additive" else 0)
            or n["source_histories"] != 2560
            or n["native_updates_before_reset"] != 900
            or n["new_optimizer_updates_after_reset"] != 900
            or n["total_native_updates"] != 1800
            or not n["fresh_AdamW_moments_at900"]
            or n["source_batch_generator_offset"] != 900
            or n["source_index_generator_seed"] != n["seed"] * 7919 + 59):
            raise AssertionError("native source-training stages/seed/AdamW reset changed")
        new_model, new_ckpt = read_model(
            folder / n["reset1800_checkpoint_file"], n["reset1800_checkpoint_sha256"],
            world, arm, n["reset1800_model_sha256"], old=False)
        if (new_ckpt["original900_model_sha256"] != old_receipt["final_model_weights_sha256"]
            or new_ckpt["original900_checkpoint_sha256"] != old_receipt["checkpoint_sha256"]):
            raise AssertionError("second optimizer stage has incorrect 900 ancestry")
        # Recover first and last new batch-index digests without retraining.
        gen = torch.Generator().manual_seed(n["seed"] * 7919 + 59)
        for _ in range(900):
            torch.randint(2560, (128,), generator=gen)
        first, last = None, None
        for _ in range(900):
            idx = torch.randint(2560, (128,), generator=gen)
            if first is None:
                first = batch_digest(idx)
            last = batch_digest(idx)
        if (first != n["first_reset_batch_indices_sha256"]
            or last != n["last_reset_batch_indices_sha256"]):
            raise AssertionError("prospective second optimizer used another source draw stream")
        score900 = source_confusion(old_model, catalog["train"], world, arm=arm)
        score1800 = source_confusion(new_model, catalog["train"], world, arm=arm)
        if (score900 != n["old900_source_score"]
            or score1800 != n["reset1800_source_score"]
            or score900["correct"] != old_receipt["source_correct"]
            or n["old_native_90pct"] != (score900["accuracy"] >= .9)
            or n["new_native_90pct"] != (score1800["accuracy"] >= .9)):
            raise AssertionError("prospective native source predictions altered")
        chosen, graph, roster = source_roster(new_model, catalog, world)
        roster_path = folder / f"{arm}.source_roster.json"
        recorded = read(roster_path)
        if (sha_file(roster_path) != row["source_roster_sha256"]
            or recorded["world_sha256"] != world["world_sha256"]
            or recorded["source_manifest_sha256"] != manifest["source_sha256"]
            or recorded["arm"] != arm
            or recorded["native_model_sha256"] != n["reset1800_model_sha256"]
            or recorded["source_roster"] != roster or row["source_candidate"] != chosen):
            raise AssertionError("source-only graph/roster/candidate changed")
        ae, angle = row["SAE_source"], row["angle_source"]
        audit_arm = {
            "old900_correct": score900["correct"],
            "reset1800_correct": score1800["correct"],
            "new_model_sha256": n["reset1800_model_sha256"],
            "source_candidate_exists": chosen is not None,
            "source_null_feasible": False,
        }
        if chosen is None:
            if ae is not None or angle is not None or row["source_model_eligible"]:
                raise AssertionError("missing source site incorrectly marked eligible")
            audit_arm["status"] = "no_source_candidate"
        else:
            if (ae is None or ae["source_contexts_fit"] != 40
                or ae["source_rank_context_groups"] != 40 or ae["sae_training_steps"] != 480):
                raise AssertionError("fitted source AE doesn't follow fixed 40/40 policy")
            path = folder / ae["source_dictionary_file"]
            if sha_file(path) != ae["source_dictionary_sha256"]:
                raise AssertionError("fresh prospective source SAE checkpoint byte hash mismatch")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (saved["schema"] != SCHEMA or saved["arm"] != arm
                or saved["world_sha256"] != world["world_sha256"]
                or saved["source_manifest_sha256"] != manifest["source_sha256"]
                or saved["model_sha256"] != n["reset1800_model_sha256"]
                or saved["model_checkpoint_sha256"] != n["reset1800_checkpoint_sha256"]
                or saved["source_fit_context_groups"] != 40
                or saved["source_rank_context_groups"] != 40):
                raise AssertionError("new TopK16 SAE fitted against the wrong frozen native weights")
            sae = TopKSAE().eval()
            sae.load_state_dict(saved["state"], strict=True)
            mean, scale = saved["mean"], saved["scale"]
            if (seal_array(mean) != ae["normalization_mean_sha256"]
                or seal_array(scale) != ae["normalization_scale_sha256"]):
                raise AssertionError("source mean/scale SHA256 altered")
            sampled = collect_source(new_model, catalog["train"][:SOURCE_FIT_ROWS]).reshape(-1,64)
            normed = (sampled-mean)/scale
            with torch.no_grad():
                recon, code = sae(normed)
                mse = float((recon-normed).square().mean())
                active = float((code>1e-7).float().sum(1).mean())
            close(mse, ae["reconstruction_mse_normalized"], atol=2e-6)
            close(active, ae["observed_mean_active_codes"], atol=1e-6)
            ranks, digest = dev_ranks(new_model, sae, mean, scale,
                                     catalog["train"][SOURCE_FIT_ROWS:SOURCE_FIT_ROWS+SOURCE_RANK_ROWS])
            if (ranks != saved["rank"] or digest != saved["rank_sha256"]
                or digest != ae["source_rank_sha256"]):
                raise AssertionError("new world source-only SAE feature ranks not reproducible")
            audit_arm["sae_mse"] = mse
            audit_arm["sae_exact_active_fraction"] = ae["fraction_exactly_16_active"]
            audit_arm["sae_sha256"] = ae["source_dictionary_sha256"]
            if angle is not None:
                audit_arm.update(audit_geometry(angle, chosen, ranks, catalog))
                audit_arm["status"] = "source_null_replayed"
            else:
                if row["angle_failure"] is None:
                    raise AssertionError("unexplained source-null omission")
                audit_arm["status"] = "directional_null_unavailable"
        eligible = (score1800["accuracy"] >= .9 and chosen is not None
                    and ae is not None and ae["reconstruction_mse_normalized"] <= .03
                    and audit_arm["source_null_feasible"])
        if eligible != row["source_model_eligible"]:
            raise AssertionError("prospective ITT source gate misreported")
        audit_arm["source_eligible"] = eligible
        arms[arm] = audit_arm
    root_ok = all(v["source_eligible"] for v in arms.values())
    if root_ok != report["both_arms_source_eligible"]:
        raise AssertionError("paired world source eligibility misreported")
    result = {
        "schema": AUDIT_SCHEMA, "world_id": world["world_id"], "world_index": index,
        "world_sha256": world["world_sha256"],
        "source_receipt_sha256": sha_file(folder / "nec03p_source_receipt.json"),
        "both_stage_checkpoint_bytes_and_training_source_labels_audited": True,
        "both_arm_source_SAEs_and_geometry_independently_checked": True,
        "source_root_eligible": root_ok,
        "new_development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0, "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY", "arms": arms,
    }
    (folder / "independent_nec03p_source_audit.json").write_bytes(encoded(result))
    print("NEC03P_INDEPENDENT_SOURCE_RECRUIT_AUDIT_PASS", world["world_id"])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-index", type=int, required=True)
    p.add_argument("--folder", type=Path, required=True)
    a = p.parse_args()
    audit(a.world_index, a.folder)
