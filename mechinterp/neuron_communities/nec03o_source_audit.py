"""NEC-03O independent source-only model/SAE/directional-null replay.

Recomputes native source predictions, 40/40 SAE reconstruction/ranking,
and exhaustively re-enumerates every logged active-code geometry option.
Failed/missing controls remain denominator failures. No evaluation split.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import median

import torch

from mechinterp.neuron_communities.nec03o_source_refit import (
    ROOTS, SCHEMA, FIT_ROWS, RANK_ROWS, read, restore_model, verify_lock,
)
from mechinterp.neuron_communities.nec03l_five_input_worlds import worlds, registration, context_catalog, encoded
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03m_source_pilot import source_roster, sha_file
from mechinterp.neuron_communities.nec03nb_checkpoint_reset import source_confusion
from mechinterp.neuron_communities.nec03k_source_audit import cosine, vector_sum, norm, close
from mechinterp.neuron_communities.source_sae import collect_source, seal_array
from mechinterp.neuron_communities.topk_sae import TopKSAE, dev_ranks
from research.pef07.pef07 import model_digest

AUDIT_SCHEMA = "NEC-03O/independent-reset1800-source-SAE-null-audit/v1"


def audit_geometry(angle: dict, candidate: dict, ranks: list, catalog: dict):
    if (angle["source_contexts"] != 8
        or angle["focal_histories_per_source_group"] != 32
        or angle["heldout_development_reserved_or_embargoed_inputs_seen"] != 0
        or not angle["source_training_label_free_control_selection"]
        or len(angle["source_contexts_and_site_receipts"]) != 8):
        raise AssertionError("source-only geometry cohort changed")
    visit_counts = {}
    for item in candidate["nodes"]:
        visit = int(item.split(":")[0])
        visit_counts[visit] = visit_counts.get(visit, 0) + 1
    total = matched = positive = 0
    retained = []
    for group_idx, entry in enumerate(angle["source_contexts_and_site_receipts"]):
        if (entry["source_group"] != group_idx
            or entry["all_32_focal_input_masks"] != list(range(32))
            or tuple(entry["source_context_bits"]) != tuple(catalog["train"][group_idx * 32])
            or set(map(int, entry["sites"])) != set(visit_counts)):
            raise AssertionError("geometry opened non-source or partial focal groups")
        for key, site in entry["sites"].items():
            visit = int(key)
            n = visit_counts[visit]
            if len(site["routes"]) != 32:
                raise AssertionError("missing source geometry rows")
            dose = site["direct_dose"]
            rank_at = {f: j for j, f in enumerate(ranks[visit])}
            for i, route in enumerate(site["routes"]):
                active = route["all_active"]
                source = route["source_top_active"]
                chosen = route["selected_disjoint_control"]
                contributions = {x["id"]: x["vector"] for x in route["active_output_contributions32"]}
                if (len(active) != 16 or len(set(active)) != 16
                    or set(active) != set(contributions)
                    or len(source) != n or len(chosen) != n
                    or set(source) & set(chosen)
                    or source != sorted(active, key=lambda f: rank_at[f])[:n]):
                    raise AssertionError("invalid TopK16 source/disjoint control membership")
                possible = list(combinations(sorted(set(active)-set(source)), n))
                if len(possible) != route["candidate_count"]:
                    raise AssertionError("source null search failed exhaustive enumeration")
                g = route["hub_gradient32"]
                source32 = vector_sum(source, contributions)
                control32 = vector_sum(chosen, contributions)
                source_cos = cosine(source32, g)
                control_cos = cosine(control32, g)
                gap = abs(source_cos-control_cos)
                best = min(abs(cosine(vector_sum(subset, contributions), g)-source_cos)
                           for subset in possible)
                if gap > best + 5e-5:
                    raise AssertionError("saved source null is not closest disjoint active direction")
                close(route["absolute_cosine_gap"], gap, atol=4e-5)
                a, b = norm(source32), norm(control32)
                shared = min(a, b)
                for name, value in (("full_left_dose", a), ("full_right_dose", b),
                                    ("shared_achievable_dose", shared),
                                    ("realized_left_dose", shared),
                                    ("realized_right_dose", shared)):
                    close(dose[name][i], value, atol=1e-4)
                if dose["left_gain"][i] > 1.000001 or dose["right_gain"][i] > 1.000001:
                    raise AssertionError("output equalization amplified a feature direction")
                ratio = shared/a if a > 1e-6 else 0.
                close(dose["retained_fraction_of_full_left_dose"][i], ratio)
                valid = (gap <= .25 and norm(g) > 1e-7 and a > 1e-6 and b > 1e-6)
                if valid != route["within_angle_gate"]:
                    raise AssertionError("direction gate differs from original exhaustive candidates")
                matched += int(valid)
                positive += int(shared > 1e-6)
                total += 1
                retained.append(ratio)
    if not total:
        raise AssertionError("no source-only angular routes")
    fraction, posfrac, med = matched/total, positive/total, median(retained)
    close(angle["angle_match_fraction"], fraction, atol=1e-6)
    close(angle["physical_dose"]["fraction_with_positive_shared_dose"], posfrac, atol=1e-6)
    close(angle["physical_dose"]["median_fraction_full_left_dose_preserved"], med, atol=1e-6)
    feasible = (fraction >= .8 and posfrac >= .8 and med >= .25
                and angle["physical_dose"]["candidate_physically_comparable_and_signal_sufficient"])
    if angle["source_null_feasible"] != feasible:
        raise AssertionError("source feasibility misclassified")
    return {"verified_routes": total, "directional_coverage": fraction,
            "positive_dose_fraction": posfrac, "median_retained_dose": med,
            "source_null_feasible": feasible}


def audit_one(index: int, original: Path, reset: Path, refit: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock = verify_lock()
    if index not in ROOTS:
        raise PermissionError("source-only audit cannot open other development roots")
    world = worlds()[index]
    catalog, manifest = context_catalog(world)
    receipt = read(refit / "nec03o_source_refit_receipt.json")
    old = read(original / "source_pilot_receipt.json")
    reset_report = read(reset / "checkpoint_reset_source_report.json")
    if (receipt["schema"] != SCHEMA
        or receipt["world_index"] != index
        or receipt["world_sha256"] != world["world_sha256"]
        or receipt["five_input_registry_sha256"] != registration()["registry_sha256"]
        or receipt["four_input_confirmation_sha256"] != old_registry()["confirmation_full_spec_sha256"]
        or receipt["source_manifest_sha256"] != manifest["source_sha256"]
        or receipt["source_lock_sha256"] != lock["lock_sha256"]
        or receipt["original_source_receipt_sha256"] != sha_file(original / "source_pilot_receipt.json")
        or receipt["native_reset_receipt_sha256"] != sha_file(reset / "checkpoint_reset_source_report.json")
        or (receipt["source_fit_groups"], receipt["source_rank_groups"], receipt["source_geometry_groups"]) != (40, 40, 8)
        or not receipt["all_arms_in_intent_to_treat_denominator"]
        or any(receipt[k] for k in (
            "new_native_training_updates", "development_evaluation_groups_scored",
            "reserved_context_groups_scored", "embargoed_worlds_scored",
            "four_input_confirmation_worlds_scored"))
        or receipt["scientific_gate"] != "REDESIGN_OR_PILOT_ONLY"):
        raise AssertionError("NEC-03O source lock/provenance/embargo was broken")
    out_arms = {}
    for arm in ("composed", "additive"):
        row = receipt["arms"][arm]
        upstream = reset_report["arms"][arm]
        model = restore_model(reset / upstream["new_checkpoint_file"], world, arm, upstream)
        newsha = model_digest(model)
        frozen = row["native_reset1800"]
        if (newsha != frozen["new_model_sha256"]
            or frozen["new_checkpoint_sha256"] != upstream["new_checkpoint_sha256"]
            or frozen["old_model_sha256"] != old["arms"][arm]["native"]["final_model_weights_sha256"]
            or frozen["old_checkpoint_sha256"] != old["arms"][arm]["native"]["checkpoint_sha256"]):
            raise AssertionError("new model or exact old ancestry SHA256 changed")
        score = source_confusion(model, catalog["train"], world, arm=arm)
        if score != frozen["source_score"] or score["correct"] != upstream["new_source_score"]["correct"]:
            raise AssertionError("source input outcomes changed between original and SAE fit")
        candidate, graph, roster = source_roster(model, catalog, world)
        stored_roster_path = refit / f"{arm}.source_roster.json"
        stored_roster = read(stored_roster_path)
        if sha_file(stored_roster_path) != row["source_roster_sha256"]:
            raise AssertionError("source roster changed")
        if (stored_roster["new_native_model_sha256"] != newsha
            or stored_roster["world_sha256"] != world["world_sha256"]
            or stored_roster["source_sha256"] != manifest["source_sha256"]
            or stored_roster["native_checkpoint_sha256"] != upstream["new_checkpoint_sha256"]
            or stored_roster["roster"] != roster or row["source_candidate"] != candidate):
            raise AssertionError("40 complete source groups changed the graph candidate")
        old_sae = old["arms"][arm]["SAE_source"]
        oldmse = None if old_sae is None else old_sae["reconstruction_mse_normalized"]
        if row["old_900_sae_source_mse"] != oldmse:
            raise AssertionError("original 900-step SAE baseline changed")
        fitted = row["SAE_source"]
        geom = row["angle_source"]
        result = {"old_model_sha256": frozen["old_model_sha256"],
                  "new_model_sha256": newsha,
                  "old_source_sae_normalized_mse": oldmse,
                  "source_correct": score["correct"], "source_total": score["total"]}
        if candidate is None:
            if fitted is not None or geom is not None or row["source_model_eligible"]:
                raise AssertionError("invented a new SAE or source null for missing candidate")
            result["status"] = "source_candidate_missing"
            result["source_null_feasible"] = False
        else:
            if fitted is None or fitted["source_contexts_fit"] != 40 or fitted["source_rank_context_groups"] != 40 or fitted["sae_training_steps"] != 480:
                raise AssertionError("wrong fresh SAE fitting protocol")
            dictionary = refit / fitted["source_dictionary_file"]
            if sha_file(dictionary) != fitted["source_dictionary_sha256"]:
                raise AssertionError("new checkpoint-specific SAE byte hash mismatch")
            state = torch.load(dictionary, map_location="cpu", weights_only=True)
            if (state["schema"] != SCHEMA or state["arm"] != arm
                or state["model_sha256"] != newsha
                or state["native_checkpoint_sha256"] != upstream["new_checkpoint_sha256"]
                or state["world_sha256"] != world["world_sha256"]
                or state["source_manifest_sha256"] != manifest["source_sha256"]
                or state["source_fit_context_groups"] != 40
                or state["source_rank_context_groups"] != 40):
                raise AssertionError("SAE was fitted to another native checkpoint or source split")
            sae = TopKSAE().eval()
            sae.load_state_dict(state["state"], strict=True)
            mean, scale = state["mean"], state["scale"]
            if (seal_array(mean) != fitted["normalization_mean_sha256"]
                or seal_array(scale) != fitted["normalization_scale_sha256"]):
                raise AssertionError("fresh source-only SAE normalization altered")
            acts = collect_source(model, catalog["train"][:FIT_ROWS]).reshape(-1, 64)
            normed = (acts - mean)/scale
            with torch.no_grad():
                reconstruction, codes = sae(normed)
                mse = float((reconstruction - normed).square().mean())
                active = float((codes > 1e-7).float().sum(1).mean())
            close(mse, fitted["reconstruction_mse_normalized"], atol=2e-6)
            close(active, fitted["observed_mean_active_codes"], atol=1e-6)
            ranks, digest = dev_ranks(model, sae, mean, scale, catalog["train"][FIT_ROWS:FIT_ROWS + RANK_ROWS])
            if ranks != state["rank"] or digest != state["rank_sha256"] or digest != fitted["source_rank_sha256"]:
                raise AssertionError("new source-only ranking drifted")
            if geom is None:
                if row["angle_failure"] is None:
                    raise AssertionError("source angle missing without a recorded failure")
                result["status"] = "source_null_missing"
                result["source_null_feasible"] = False
            else:
                result.update(audit_geometry(geom, candidate, ranks, catalog))
                result["status"] = "source_geometry_replayed"
            result["new_source_sae_normalized_mse"] = mse
            if row["new_1800_sae_source_mse"] is None:
                raise AssertionError("new source SAE comparison missing")
            close(mse, row["new_1800_sae_source_mse"], atol=2e-6)
            result["source_sae_checkpoint_sha256"] = fitted["source_dictionary_sha256"]
            result["source_rank_sha256"] = digest
        eligible = (score["accuracy"] >= .9
                    and candidate is not None
                    and fitted is not None
                    and fitted["reconstruction_mse_normalized"] <= .03
                    and result["source_null_feasible"])
        if row["native_source_90pct"] != (score["accuracy"] >= .9) or row["source_model_eligible"] != eligible:
            raise AssertionError("source admission or ITT flag incorrect")
        result["source_model_eligible"] = eligible
        out_arms[arm] = result
    if receipt["both_arms_source_eligible"] != all(a["source_model_eligible"] for a in out_arms.values()):
        raise AssertionError("complete-world source admission differs from arm audit")
    output = {
        "schema": AUDIT_SCHEMA,
        "world_id": world["world_id"], "world_index": index,
        "world_sha256": world["world_sha256"],
        "source_refit_receipt_sha256": sha_file(refit / "nec03o_source_refit_receipt.json"),
        "original_900_native_and_reset1800_model_provenance_verified": True,
        "fresh_1800_specific_source_SAE_and_rank_audited": True,
        "exhaustive_disjoint_active_sham_routes_reenumerated_for_available_controls": True,
        "source_root_eligible": all(x["source_model_eligible"] for x in out_arms.values()),
        "new_development_or_reserved_contexts_scored": 0,
        "embargoed_or_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
        "arms": out_arms,
    }
    (refit / "independent_nec03o_source_audit.json").write_bytes(encoded(output))
    print("NEC03O_NEW_SAE_SOURCE_AUDIT_PASS", world["world_id"])
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-index", required=True, type=int)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--reset-root", type=Path, required=True)
    p.add_argument("--refit-root", type=Path, required=True)
    a = p.parse_args()
    audit_one(a.world_index, a.original_root, a.reset_root, a.refit_root)
