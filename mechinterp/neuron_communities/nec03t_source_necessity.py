"""NEC-03T: frozen source-only necessity ablation from full-affine SAE background.

Exactly the original 8 dev5 source worlds, composed and additive teachers,
original frozen reset1800 native / TopK16 SAE source checkpoints, and
original source-training context groups 24..31 (8 x 32 masks).
Zero native/SAE optimizer steps; no development/confirmation readouts.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
from itertools import combinations
import json
from pathlib import Path

import torch

from mechinterp.neuron_communities.nec03l_five_input_worlds import (
    worlds, registration, context_catalog, labels, encoded,
)
from mechinterp.neuron_communities.nec03g_worlds import registry as old_registry
from mechinterp.neuron_communities.nec03r_source_recoverability import load_source_arm
from mechinterp.neuron_communities.nec03j_direction_transport import traced_forward
from mechinterp.neuron_communities.source_sae import clean_activations
from mechinterp.neuron_communities.nec03m_source_pilot import sha_file
from research.pef07.pef07 import as_tensor, model_digest

SCHEMA = "NEC-03T/source-only-full-affine-necessity-ablation/v1"
LOCK_FILE = "NEC03T_SOURCE_LOCK.json"
ROOTS = tuple(range(8))
PANEL = tuple(range(24, 32))
SOURCE_ELIGIBLE = (0, 1, 2, 4, 6, 7)
ITT_ZERO = (3, 5)
CASES = 256
FOCAL = 32
ANGLE_MAX = .25
EFFECT_FLOOR = .02
BUDGETS = ("orig", "wide")
HASH_SALT = b"NEC03T-source-full-affine-necessity-random-v1"
MODES = (
    "clean", "hubs", "native_nonhub", "full_affine",
    "contrib16_only", "ablate_orig_source", "ablate_orig_sham",
    "ablate_wide_source", "ablate_wide_sham",
    "ablate_hash_source", "ablate_hash_sham",
)


def read(path: Path):
    return json.loads(path.read_text())


def verify_lock():
    path = Path(__file__).with_name(LOCK_FILE)
    cfg = read(path)
    exact = {
        "schema": "NEC-03T/frozen-source-full-affine-necessity/v1",
        "original_world_root_indices": list(ROOTS),
        "source_group_indices": list(PANEL),
        "complete_focal_masks_per_group": FOCAL,
        "source_histories_per_arm": CASES,
        "paired_source_eligible": list(SOURCE_ELIGIBLE),
        "paired_ITT_zero": list(ITT_ZERO),
        "original_native_A_run": 44898361742,
        "original_SAE_A_run": 44899750508,
        "original_native_SAE_B_run": 44900628360,
        "five_input_registry_sha256": registration()["registry_sha256"],
        "old_four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "source_active_codes": 16,
        "budget_names": list(BUDGETS),
        "all_intervention_modes": list(MODES),
        "angular_gap_limit": ANGLE_MAX,
        "random_control_hash_salt": HASH_SALT.decode(),
        "primary_root_effect": "((composed sham-minus-source necessity)-(additive sham-minus-source necessity))/256; root 3/5 zero; mean 8",
        "primary_comparisons": 1,
        "minimum_meaningful_effect": EFFECT_FLOOR,
        "angle_sham_policy": "exhaustive equal-n disjoint active-code minimum absolute 32D cosine gap under full-affine gradient",
        "ablation_dose_policy": "single min(source,sham) 32D output L2 per input/visit, attenuation only, no amplification",
        "invalid_case_policy": "both donors restore identical full-affine background on all source-selected visits",
        "native_and_SAE_optimizer_updates": 0,
        "new_development_eval_groups_scored": 0,
        "reserved_groups_scored": 0,
        "newly_recruited_worlds_scored": 0,
        "embargoed_worlds_scored": 0,
        "old_four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k) != value for k, value in exact.items()):
        raise AssertionError("NEC03T frozen source necessity preregistration changed")
    root = Path(__file__).resolve().parents[2]
    for name, pin in cfg["git_blob_sha1"].items():
        data = (root / name).read_bytes()
        actual = sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
        if actual != pin:
            raise AssertionError("NEC03T source code byte identity altered: " + name)
    return cfg, sha_file(path)


def budget_width(n: int, budget: str):
    if n not in range(1, 9) or budget not in BUDGETS:
        raise ValueError("unregistered source ablation budget")
    return n if budget == "orig" else min(2 * n, 8)


def fixed_random_control(options: list[tuple[int, ...]], *, world_sha: str,
                         arm: str, group: int, mask: int, visit: int):
    if not options:
        raise AssertionError("no disjoint hash-active control at registered budget")
    key = f"{world_sha}|{arm}|{group}|{mask}|{visit}|orig".encode()
    target = int.from_bytes(sha256(HASH_SALT + b"|" + key).digest(), "big")
    return list(options[target % len(options)])


def cosine32(vec, grad):
    return (vec * grad).sum(dim=-1) / (
        vec.norm(dim=-1) * grad.norm(dim=-1)).clamp_min(1e-10)


def shared_removal(src, sham, down):
    """One physical min-L2 dose; returns two attenuated physical 64D removals."""
    if src.shape != sham.shape or src.ndim != 2 or src.shape[1] != 64:
        raise ValueError("invalid ablation donor dimensions")
    norms = {"source": (src @ down.T).norm(dim=1),
             "sham": (sham @ down.T).norm(dim=1)}
    target = torch.minimum(norms["source"], norms["sham"])
    gains = {
        key: torch.where(norm > 1e-10, target / norm.clamp_min(1e-10),
                         torch.zeros_like(target)).clamp(0, 1)
        for key, norm in norms.items()
    }
    realized = {"source": src * gains["source"][:, None],
                "sham": sham * gains["sham"][:, None]}
    physical = {key: (x @ down.T).norm(dim=1) for key, x in realized.items()}
    matched = torch.ones(len(src), dtype=torch.bool)
    for key in realized:
        matched &= ((physical[key] - target).abs() /
                    target.clamp_min(1e-6)) <= .01
        if bool((gains[key] > 1 + 1e-6).any()):
            raise AssertionError("ablation component amplified")
    return realized, {
        "raw_dose": {key: x.tolist() for key, x in norms.items()},
        "shared_dose": target.tolist(),
        "gains": {key: x.tolist() for key, x in gains.items()},
        "physical_dose_equal_within_1pct": matched.tolist(),
        "nonzero_shared_dose": (target > 1e-6).tolist(),
    }


def group_measure(prepared: dict, group: int, histories):
    if group not in PANEL or len(histories) != FOCAL:
        raise PermissionError("only full original source-training groups 24..31")
    model = prepared["model"]
    world = prepared["world"]
    arm = prepared["arm"]
    x = as_tensor(histories)
    clean = traced_forward(model, x, mode="clean")
    clean_orientation = clean["logits"].argmax(dim=1)
    sites = prepared["source_sites"]
    hubs = prepared["source_hubs"]
    gates, empty = {}, {}
    for visit in sites:
        gate = torch.zeros(64)
        if visit in hubs:
            gate[sorted(hubs[visit])] = 1
        gates[visit] = gate
        empty[visit] = {"hubs": torch.zeros(FOCAL, 64)}
    hubs_trace = traced_forward(
        model, x, mode="hubs", orientation=clean_orientation,
        donors=empty, gates=gates)
    variants = {name: {} for name in MODES if name not in ("clean", "hubs")}
    geometry = {}
    valid = {budget: torch.full((FOCAL,), bool(sites), dtype=torch.bool)
             for budget in BUDGETS}
    random_valid = torch.full((FOCAL,), bool(sites), dtype=torch.bool)
    down = model.block.mlp_down.weight.detach()
    clean_states = clean_activations(model, x) if sites else None
    full = {}
    code_by_visit = {}
    decoder_by_visit = {}
    for visit, nodes in sorted(sites.items()):
        act = clean_states[:, visit, :]
        nonhub = 1 - gates[visit]
        sae = prepared["sae"]
        with torch.no_grad():
            z = sae.codes((act - prepared["mean"]) / prepared["scale"])
            decoded = sae.decoder(z) * prepared["scale"] + prepared["mean"]
            physical_cols = sae.decoder.weight.detach().T * prepared["scale"][None, :]
            contrib16 = z @ physical_cols
            physical_offset = prepared["mean"] + (
                sae.decoder.bias.detach() * prepared["scale"])
            if not torch.allclose(decoded, contrib16 + physical_offset,
                                  atol=2e-5, rtol=1e-5):
                raise AssertionError("full-affine source SAE decomposition changed")
        full[visit] = decoded * nonhub
        code_by_visit[visit] = z
        decoder_by_visit[visit] = physical_cols
        variants["native_nonhub"][visit] = act * nonhub
        variants["full_affine"][visit] = full[visit]
        variants["contrib16_only"][visit] = contrib16 * nonhub
    # The angle gradient must be taken on the full SAE-affine background,
    # not on the original hubs-only lesion and never on outcome labels.
    if sites:
        full_donors = {visit: {"native": v} for visit, v in full.items()}
        background = traced_forward(
            model, x, mode="native", orientation=clean_orientation,
            donors=full_donors, gates=gates, derivative=True)
    else:
        background = clean
    for visit, nodes in sorted(sites.items()):
        k_orig = len(nodes)
        z = code_by_visit[visit]
        columns = decoder_by_visit[visit]
        nonhub = 1 - gates[visit]
        grad = background["grad_down"][visit]
        physical_cols = (columns * nonhub[None, :]) @ down.T
        importance = {feature: j for j, feature in
                      enumerate(prepared["ranks"][visit])}
        geometry[str(visit)] = {
            "native_source_units": sorted(nodes),
            "original_graph_hubs": sorted(hubs.get(visit, [])),
            "full_affine_physical_dose": (full[visit] @ down.T).norm(dim=1).tolist(),
            "full_affine_relative_32d_error": (
                ((full[visit] - variants["native_nonhub"][visit]) @ down.T).norm(dim=1) /
                (variants["native_nonhub"][visit] @ down.T).norm(dim=1).clamp_min(1e-6)
            ).tolist(),
            "budget": {},
        }
        for budget in BUDGETS:
            k = budget_width(k_orig, budget)
            src = torch.zeros(FOCAL, 64)
            sham = torch.zeros(FOCAL, 64)
            rnd = torch.zeros(FOCAL, 64)
            routes = {
                "k": k, "active_ids": [], "source_ids": [],
                "angular_sham_ids": [], "hash_sham_ids": [],
                "candidate_count": [], "angle_gap": [],
                "exact_positive16_codes": [],
            }
            local_valid = torch.ones(FOCAL, dtype=torch.bool)
            local_hash = torch.ones(FOCAL, dtype=torch.bool)
            for mask in range(FOCAL):
                active = torch.where(z[mask] > 1e-7)[0].tolist()
                routes["active_ids"].append(active)
                enough = len(active) == 16
                routes["exact_positive16_codes"].append(enough)
                if not enough:
                    local_valid[mask] = False
                    local_hash[mask] = False
                    routes["source_ids"].append([])
                    routes["angular_sham_ids"].append([])
                    routes["hash_sham_ids"].append([])
                    routes["candidate_count"].append(0)
                    routes["angle_gap"].append(None)
                    continue
                picked = sorted(active, key=lambda f: importance[f])[:k]
                alternatives = sorted(set(active) - set(picked))
                options = list(combinations(alternatives, k))
                idx = torch.tensor(options, dtype=torch.long)
                raw_source = (z[mask, picked] @ columns[picked]) * nonhub
                # Exhaustive disjoint active-control comparator in physical
                # 32D output, using exactly the same scalar cosine objective.
                vector32 = (
                    z[mask, idx].unsqueeze(-1) * physical_cols[idx]).sum(dim=1)
                projected = raw_source @ down.T
                cos_src = cosine32(projected, grad[mask])
                gaps = (cosine32(vector32, grad[mask]) - cos_src).abs()
                chosen = int(torch.argmin(gaps))
                angular_ids = list(options[chosen])
                random_ids = fixed_random_control(
                    options, world_sha=world["world_sha256"],
                    arm=arm, group=group, mask=mask, visit=visit) if budget == "orig" else []
                # For wide budget the secondary comparison has only an
                # exhaustive angular sham; hashed-null is original-n only.
                src[mask] = raw_source
                sham[mask] = (z[mask, angular_ids] @ columns[angular_ids]) * nonhub
                if budget == "orig":
                    rnd[mask] = (z[mask, random_ids] @ columns[random_ids]) * nonhub
                routes["source_ids"].append(picked)
                routes["angular_sham_ids"].append(angular_ids)
                routes["hash_sham_ids"].append(random_ids)
                routes["candidate_count"].append(len(options))
                routes["angle_gap"].append(float(gaps[chosen]))
                if float(gaps[chosen]) > ANGLE_MAX or float(grad[mask].norm()) <= 1e-7:
                    local_valid[mask] = False
            matched, dose = shared_removal(src, sham, down)
            local_valid &= torch.tensor([
                dose["nonzero_shared_dose"][i] and
                dose["physical_dose_equal_within_1pct"][i]
                for i in range(FOCAL)], dtype=torch.bool)
            valid[budget] &= local_valid
            if budget == "orig":
                random_dose_pair, random_dose = shared_removal(src, rnd, down)
                local_hash &= torch.tensor([
                    random_dose["nonzero_shared_dose"][i] and
                    random_dose["physical_dose_equal_within_1pct"][i]
                    for i in range(FOCAL)], dtype=torch.bool)
                random_valid &= local_hash
                variants["ablate_hash_source"][visit] = (
                    full[visit] - random_dose_pair["source"])
                variants["ablate_hash_sham"][visit] = (
                    full[visit] - random_dose_pair["sham"])
                routes["hash_removal_dose"] = random_dose
            variants[f"ablate_{budget}_source"][visit] = (
                full[visit] - matched["source"])
            variants[f"ablate_{budget}_sham"][visit] = (
                full[visit] - matched["sham"])
            routes["angular_removal_dose"] = dose
            geometry[str(visit)]["budget"][budget] = routes
    # Failure at any source-selected recurrent visit retains identical
    # unablated full-affine background for both paired arms on that input.
    for visit in sites:
        for budget in BUDGETS:
            mask = valid[budget][:, None]
            for side in ("source", "sham"):
                v = f"ablate_{budget}_{side}"
                variants[v][visit] = (full[visit] +
                    (variants[v][visit] - full[visit]) * mask)
        for side in ("source", "sham"):
            v = f"ablate_hash_{side}"
            variants[v][visit] = full[visit] + (
                variants[v][visit] - full[visit]) * random_valid[:, None]
    traces = {"clean": clean, "hubs": hubs_trace}
    for mode, site_vecs in variants.items():
        donors = {visit: {"native": vec} for visit, vec in site_vecs.items()}
        with torch.no_grad():
            traces[mode] = traced_forward(
                model, x, mode="native", orientation=clean_orientation,
                donors=donors, gates=gates)
    for budget in BUDGETS:
        invalid = ~valid[budget]
        if not torch.allclose(
            traces[f"ablate_{budget}_source"]["logits"][invalid],
            traces[f"ablate_{budget}_sham"]["logits"][invalid],
            atol=1e-6):
            raise AssertionError("invalid angular null generated nonzero effect")
    if not torch.allclose(
        traces["ablate_hash_source"]["logits"][~random_valid],
        traces["ablate_hash_sham"]["logits"][~random_valid], atol=1e-6):
        raise AssertionError("invalid hash null generated nonzero effect")
    return {
        "schema": SCHEMA,
        "source_group": group,
        "input_histories": histories,
        "focal_mask_order": list(range(FOCAL)),
        "clean_chosen_orientation": clean_orientation.tolist(),
        "labels_used_for_routing": False,
        "site_geometry": geometry,
        "valid_angular_all_visits": {b: valid[b].tolist() for b in BUDGETS},
        "valid_hash_all_visits": random_valid.tolist(),
        "logits": {k: v["logits"].tolist() for k, v in traces.items()},
        "mean_memory_distance_from_full_affine_by_visit": {
            k: (v["memory"] - background["memory"]).norm(dim=-1).mean(dim=0).tolist()
            for k, v in traces.items()},
    }


def run(index: int, source: Path, output: Path, *, reset: Path | None = None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg, lock_sha = verify_lock()
    if index not in ROOTS:
        raise PermissionError("NEC03T opens only eight original source world roots")
    world = worlds()[index]
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ("composed", "additive"):
        frozen = load_source_arm(index, arm, source, reset=reset)
        frozen["arm"] = arm
        if frozen["world"]["world_sha256"] != world["world_sha256"]:
            raise AssertionError("source teacher checkpoint mismapped")
        rows = []
        totals = {k: 0 for k in MODES}
        matched_case_count = {b: 0 for b in BUDGETS}
        hash_count = 0
        for group in PANEL:
            source_rows = frozen["catalog"]["train"][group*FOCAL:(group+1)*FOCAL]
            datum = group_measure(frozen, group, source_rows)
            truth = labels(as_tensor(source_rows), world,
                           control=(arm=="additive")).tolist()
            predictions = {}
            for mode in MODES:
                logits = torch.tensor(datum["logits"][mode])
                pred = logits.argmax(dim=1).tolist()
                predictions[mode] = pred
                totals[mode] += sum(int(a==b) for a,b in zip(pred, truth))
            for budget in BUDGETS:
                matched_case_count[budget] += sum(
                    datum["valid_angular_all_visits"][budget])
            hash_count += sum(datum["valid_hash_all_visits"])
            datum["arm"] = arm
            datum["true_teacher_labels"] = truth
            datum["class_predictions"] = predictions
            rows.append(datum)
        content = b"".join(encoded(row) for row in rows)
        filename = f"{arm}.nec03t_source_necessity.jsonl"
        (output / filename).write_bytes(content)
        necessity = {
            b: totals[f"ablate_{b}_sham"] -
               totals[f"ablate_{b}_source"]
            for b in BUDGETS}
        arms[arm] = {
            "arm": arm,
            "source_model_sha256": frozen["model_sha256"],
            "source_sae_checkpoint_sha256": frozen["sae_sha256"],
            "source_sae_rank_sha256": frozen["rank_sha256"],
            "source_receipt_sha256": frozen["source_report_sha256"],
            "original_source_candidate_present": bool(frozen["source_candidate"]),
            "original_arm_source_eligible": frozen["original_arm_source_eligible"],
            "source_histories": CASES,
            "source_groups_scored": list(PANEL),
            "correct": totals,
            "valid_angular_case_count": matched_case_count,
            "valid_hash_case_count": hash_count,
            "source_ranked_necessity_correct_count": necessity,
            "raw_file": filename,
            "raw_sha256": sha256(content).hexdigest(),
        }
        if model_digest(frozen["model"]) != frozen["model_sha256"]:
            raise AssertionError("native model was changed by read-only ablation")
    difference = (
        arms["composed"]["source_ranked_necessity_correct_count"]["orig"] -
        arms["additive"]["source_ranked_necessity_correct_count"]["orig"]) / CASES
    admitted = index in SOURCE_ELIGIBLE
    record = {
        "schema": SCHEMA,
        "world_id": world["world_id"], "world_index": index,
        "world_sha256": world["world_sha256"],
        "source_manifest_sha256": context_catalog(world)[1]["source_sha256"],
        "five_input_registry_sha256": registration()["registry_sha256"],
        "old_four_input_confirmation_sha256": old_registry()["confirmation_full_spec_sha256"],
        "original_paired_source_eligible": admitted,
        "original_ITT_source_failure": index in ITT_ZERO,
        "raw_primary_source_necessity_DID": difference,
        "primary_ITT_root_effect": difference if admitted else 0.0,
        "source_lock_sha256": lock_sha,
        "source_groups_scored": list(PANEL),
        "source_training_cases_per_arm": CASES,
        "arms": arms,
        "new_native_or_SAE_training_updates": 0,
        "new_development_eval_groups_scored": 0,
        "reserved_groups_scored": 0,
        "newly_recruited_worlds_scored": 0,
        "embargoed_worlds_scored": 0,
        "old_four_input_confirmation_worlds_scored": 0,
        "scientific_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    (output / "nec03t_source_necessity_report.json").write_bytes(encoded(record))
    print("NEC03T_FULL_AFFINE_NECESSITY_SOURCE_ROOT_COMPLETE", world["world_id"])
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world-index", type=int, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--reset-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.world_index, a.source_root, a.out, reset=a.reset_root)
