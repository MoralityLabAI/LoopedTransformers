"""NEC-03P: predeclared dev5-008..095 source-only recruitment.

Exactly 900 source-only native updates, checkpoint, FRESH AdamW moments,
and 900 additional updates with the registered generator offset.
Then fit new reset1800-specific 40/40 TopK16 source SAE and 8-group
direction-matched active sham. No development evaluation or reserved data.
"""
from __future__ import annotations

import argparse
from hashlib import sha1
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, labels, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import (
    native_teacher, source_roster, fit_five_topk, source_direction_coverage,
    sha_file,
)
from mechinterp.neuron_communities.nec03nb_checkpoint_reset import (
    source_confusion, batch_digest,
)
from mechinterp.neuron_communities.topk_sae import dev_ranks
from research.pef07.pef07 import model_digest, as_tensor

SCHEMA = "NEC-03P/development-source-recruitment-native-reset-sae/v1"
MODEL_SCHEMA = "NEC-03P/fresh-source900-reset900-AdamW-native/v1"
ROOTS = tuple(range(8, 96))
B_RUN = 44900628360
SOURCE_FIT_ROWS = 40 * 32
SOURCE_RANK_ROWS = 40 * 32
NATIVE_STEPS = 900
RESET_STEPS = 900
SAE_STEPS = 480


def read(path: Path):
    return json.loads(path.read_text())


def verify_lock():
    root = Path(__file__).resolve().parents[2]
    lock_path = root / "mechinterp/neuron_communities/NEC03P_SOURCE_LOCK.json"
    cfg = read(lock_path)
    exact = {
        "schema": "NEC-03P/frozen-development-source-recruitment/v1",
        "preceding_source_B_verified_run_id": B_RUN,
        "source_B_feasible_model_arms": 8,
        "source_B_complete_feasible_worlds": 3,
        "five_input_registry_sha256": registration()["registry_sha256"],
        "four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "development_source_recruitment_world_indices": list(ROOTS),
        "native_first_stage_steps": NATIVE_STEPS,
        "native_reset_stage_steps": RESET_STEPS,
        "native_adamw_lr": .003, "native_adamw_weight_decay": 0.,
        "native_batch_size": 128, "native_grad_clip": 1.,
        "source_training_groups": 80, "source_focal_masks": 32,
        "source_fit_groups": 40, "source_rank_groups": 40, "source_geometry_groups": 8,
        "sae_updates": SAE_STEPS, "sae_features": 128, "sae_active_k": 16,
        "native_competence_floor": .9, "sae_normalized_mse_ceiling": .03,
        "max_direction_cosine_gap": .25, "min_directional_coverage": .8,
        "min_physical_positive_coverage": .8, "min_median_retained_dose": .25,
        "new_development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0,
        "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k) != v for k, v in exact.items()):
        raise AssertionError("NEC-03P source recruitment protocol modified after registration")
    for path, pinned in cfg["git_blob_sha1"].items():
        b = (root / path).read_bytes()
        actual = sha1(b"blob " + str(len(b)).encode() + b"\x00" + b).hexdigest()
        if actual != pinned:
            raise AssertionError("NEC-03P frozen implementation changed: " + path)
    cfg["lock_sha256"] = sha_file(lock_path)
    return cfg


def train_two_stage_native(world, source, arm: str, out: Path):
    """Return frozen native model, with both source milestones retained."""
    if arm not in ("composed", "additive") or world["world_index"] not in ROOTS:
        raise PermissionError("unregistered source-recruitment root or arm")
    seed = world["native_model_seed"] + (100000 if arm == "additive" else 0)
    model, old = native_teacher(world["native_model_seed"], world, source, out, arm)
    if (old["source_training_steps"] != 900
        or old["training_source_histories"] != 2560
        or old["training_seed"] != seed
        or old["final_model_weights_sha256"] != model_digest(model)
        or sha_file(out / old["checkpoint_file"]) != old["checkpoint_sha256"]):
        raise AssertionError("900-step native source checkpoint incomplete")
    # Start the second optimizer stage from the exact frozen first-stage bytes.
    frozen_payload = torch.load(out / old["checkpoint_file"], map_location="cpu", weights_only=True)
    model.load_state_dict(frozen_payload["model"], strict=True)
    model.eval()
    if model_digest(model) != old["final_model_weights_sha256"]:
        raise AssertionError("frozen original900 checkpoint did not reload exactly")
    old_score = source_confusion(model, source, world, arm=arm)
    if old_score["correct"] != old["source_correct"]:
        raise AssertionError("original prospective source900 predictions differ")
    model.train()
    x = as_tensor(source)
    y = labels(x, world, control=(arm == "additive"))
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    if opt.state:
        raise AssertionError("new source optimizer already had AdamW moments")
    rng = torch.Generator().manual_seed(seed * 7919 + 59)
    for _ in range(NATIVE_STEPS):
        torch.randint(len(x), (128,), generator=rng)
    first_index = last_index = None
    first_loss = last_loss = None
    for _ in range(RESET_STEPS):
        idx = torch.randint(len(x), (128,), generator=rng)
        if first_index is None:
            first_index = batch_digest(idx)
        last_index = batch_digest(idx)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.logits(x[idx]), y[idx])
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite stage2 source native loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
    model.eval()
    new_sha = model_digest(model)
    ck = out / f"{arm}.prospective_reset1800.pt"
    torch.save({
        "schema": MODEL_SCHEMA, "world_id": world["world_id"],
        "world_sha256": world["world_sha256"], "arm": arm,
        "model_seed": seed, "native_updates_before_reset": 900,
        "new_AdamW_updates_after_reset": 900, "total_native_updates": 1800,
        "AdamW_moments_reset": True, "original900_model_sha256": old["final_model_weights_sha256"],
        "original900_checkpoint_sha256": old["checkpoint_sha256"],
        "model_weights_sha256": new_sha,
        "source_training_only": True, "model": model.state_dict(),
    }, ck)
    score = source_confusion(model, source, world, arm=arm)
    return model, {
        "arm": arm, "seed": seed, "source_histories": 2560,
        "old900_native": old,
        "old900_source_score": old_score,
        "reset1800_checkpoint_file": ck.name,
        "reset1800_checkpoint_sha256": sha_file(ck),
        "reset1800_model_sha256": new_sha,
        "native_updates_before_reset": 900,
        "new_optimizer_updates_after_reset": 900,
        "total_native_updates": 1800,
        "fresh_AdamW_moments_at900": True,
        "source_index_generator_seed": seed * 7919 + 59,
        "source_batch_generator_offset": 900,
        "first_reset_batch_indices_sha256": first_index,
        "last_reset_batch_indices_sha256": last_index,
        "first_reset_batch_loss": first_loss,
        "last_reset_batch_loss": last_loss,
        "reset1800_source_score": score,
        "old_native_90pct": old_score["accuracy"] >= .9,
        "new_native_90pct": score["accuracy"] >= .9,
    }


def run(index: int, out: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_lock()
    if index not in ROOTS:
        raise PermissionError("source recruitment accepts only dev5-008..095")
    world = worlds()[index]
    if world["cohort"] != "development" or world["world_id"] != f"dev5-{index:03d}":
        raise PermissionError("source root is embargoed or mismapped")
    catalog, manifest = context_catalog(world)
    if (len(catalog["train"]) != 2560
        or len(catalog["development"]) != 768
        or len(catalog["reserved"]) != 768
        or manifest["source_fit_context_count"] != 40
        or manifest["source_rank_context_count"] != 40):
        raise AssertionError("prospective source split is not the original 80/24/24")
    out.mkdir(parents=True, exist_ok=True)
    (out / "world_spec.json").write_bytes(encoded(world))
    (out / "source_split_manifest.json").write_bytes(encoded(manifest))
    records = {}
    for arm in ("composed", "additive"):
        model, two_stage = train_two_stage_native(world, catalog["train"], arm, out)
        source_candidate, graph, roster = source_roster(model, catalog, world)
        roster_receipt = {
            "world_sha256": world["world_sha256"],
            "source_manifest_sha256": manifest["source_sha256"],
            "arm": arm, "native_model_sha256": model_digest(model),
            "source_roster": roster,
        }
        (out / f"{arm}.source_roster.json").write_bytes(encoded(roster_receipt))
        fit, angle, angle_failure = None, None, None
        if source_candidate is not None:
            sae, mean, scale, fit = fit_five_topk(
                model, catalog["train"][:SOURCE_FIT_ROWS],
                seed=two_stage["seed"])
            ranks, rank_digest = dev_ranks(
                model, sae, mean, scale,
                catalog["train"][SOURCE_FIT_ROWS:SOURCE_FIT_ROWS + SOURCE_RANK_ROWS])
            p = out / f"{arm}.prospective_topk16.pt"
            torch.save({
                "schema": SCHEMA, "world_sha256": world["world_sha256"], "arm": arm,
                "model_sha256": two_stage["reset1800_model_sha256"],
                "model_checkpoint_sha256": two_stage["reset1800_checkpoint_sha256"],
                "source_manifest_sha256": manifest["source_sha256"],
                "source_fit_context_groups": 40, "source_rank_context_groups": 40,
                "state": sae.state_dict(), "mean": mean, "scale": scale,
                "rank": ranks, "rank_sha256": rank_digest,
            }, p)
            fit["source_dictionary_file"] = p.name
            fit["source_dictionary_sha256"] = sha_file(p)
            fit["source_rank_sha256"] = rank_digest
            fit["source_rank_context_groups"] = 40
            try:
                angle = source_direction_coverage(
                    model, catalog, world, source_candidate,
                    graph, sae, mean, scale, ranks)
            except (ValueError, AssertionError, RuntimeError) as exc:
                angle_failure = f"{type(exc).__name__}: {exc}"
        if model_digest(model) != two_stage["reset1800_model_sha256"]:
            raise AssertionError("source-only SAE fit mutated native weights")
        ready = (
            two_stage["new_native_90pct"]
            and source_candidate is not None
            and fit is not None
            and fit["reconstruction_mse_normalized"] <= .03
            and angle is not None and angle["source_null_feasible"])
        records[arm] = {
            "native": two_stage,
            "source_candidate": source_candidate,
            "source_roster_sha256": sha_file(out / f"{arm}.source_roster.json"),
            "source_site_kind": ("fallback" if roster["fallback_used"] else "graph") if source_candidate else "none",
            "SAE_source": fit, "angle_source": angle, "angle_failure": angle_failure,
            "source_model_eligible": bool(ready),
        }
    result = {
        "schema": SCHEMA, "world_id": world["world_id"],
        "world_index": index, "world_sha256": world["world_sha256"],
        "source_manifest_sha256": manifest["source_sha256"],
        "five_input_registry_sha256": registration()["registry_sha256"],
        "four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "source_lock_sha256": lock["lock_sha256"],
        "source_focal_masks": 32, "source_training_groups": 80,
        "source_fit_groups": 40, "source_rank_groups": 40,
        "source_geometry_groups": 8, "sae_steps": 480,
        "optimizer_reset_at900": True,
        "exact_continuous_adamw_1800_claim": False,
        "all_arms_in_intent_to_treat_denominator": True,
        "new_development_evaluation_groups_scored": 0,
        "reserved_context_groups_scored": 0, "embargoed_worlds_scored": 0,
        "four_input_confirmation_worlds_scored": 0,
        "arms": records,
        "both_arms_source_eligible": all(a["source_model_eligible"] for a in records.values()),
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    (out / "nec03p_source_receipt.json").write_bytes(encoded(result))
    print("NEC03P_SOURCE_RECRUIT_NATIVE_AND_SAE_COMPLETE", world["world_id"])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-index", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.world_index, a.out)
