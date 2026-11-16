"""LS-SAE09 registered native hidden-state transport after LS-SAE08 FAIL.

Reads the original LS-SAE08 signed 512D directions and all81 logit receipts.
A numerical mismatch to the original 2026-11-16 neural output aborts the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as SOURCEFIT
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as FITPACK
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as S6
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as A8
from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAMILY

SCHEMA = "LS-SAE09/native-ACT3-6-12-hidden-transport-v1"
PROTOCOL_COMMIT = "94477c21425509c896112eda1dde1af802ce0659"
FULL_SHA = "69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
PILOT_SHA = "3b26bec178fdea4a2120001b72f3c035a91e728642adf635b9b42af985a05824"
TOL = 2e-4  # Same-run readout/logit identity; unchanged from registration.
ARCHIVE_TOL = 1e-3  # Dated amendment A1: cross-run CPU drift, argmax held fixed.
HORIZONS = ("ACT3", "ACT6", "ACT12")


def load_baseline(path: Path, pilot: bool):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != (PILOT_SHA if pilot else FULL_SHA):
        raise ValueError("LS-SAE08 original raw JSON SHA256 mismatch")
    record = json.loads(data)
    A8.audit(record)  # Original independent source and 81x9 score audit, unchanged.
    if record["mode"] != ("pilot" if pilot else "full"):
        raise ValueError("LS-SAE08 pilot/full evidence swapped")
    return record, digest


def readout_basis(model):
    # Nine digit logits (lm_head rows 2..10); remove per-cell additive gauge.
    weight = model.lm_head.weight.detach().double().cpu().numpy()[2:11]
    wc = weight - weight.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(wc, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-7))
    if not 1 <= rank <= 8:
        raise ArithmeticError("centered digit readout has invalid rank")
    return wc, vt[:rank].T.copy(), singular[:rank].tolist()


def classification(h_gain, blank_flips, board_norm, readout_fraction):
    return {
        "contracted": bool(h_gain < 0.1),
        "persistent_output_silent": bool(h_gain >= 1.0 and blank_flips == 0),
        "readout_small": bool(board_norm > 1e-8 and readout_fraction < 0.1),
    }


def pressure_and_flips(clean, changed, source):
    """Maximum signed move toward *any* clean competing digit boundary."""
    blanks = np.flatnonzero(np.asarray(source["x"]) == 0)
    original = np.argmax(clean, axis=1)
    perturbed = np.argmax(changed, axis=1)
    margin = np.sort(clean[blanks], axis=1)
    minimum_clean_margin = float(np.min(margin[:, -1] - margin[:, -2]))
    delta = changed - clean
    pressure = -np.inf
    ties = 0
    for cell in blanks:
        win = int(original[cell])
        for rival in range(9):
            if rival == win:
                continue
            gap = float(clean[cell, win] - clean[cell, rival])
            if gap <= 1e-12:
                ties += 1
                continue
            ratio = float((delta[cell, rival] - delta[cell, win]) / gap)
            pressure = max(pressure, ratio)
    if not np.isfinite(pressure):
        pressure_value = None
    else:
        pressure_value = float(pressure)
    target = np.asarray(source["target"], dtype=np.int64)
    correct = int(np.sum(perturbed[blanks] + 1 == target[blanks]))
    return {
        "blank_flip_count": int(np.count_nonzero(perturbed[blanks] != original[blanks])),
        "blank_correct": correct,
        "zero_clean_margin_pairs": int(ties),
        "boundary_pressure_max": pressure_value,
        "minimum_clean_blank_margin": minimum_clean_margin,
    }


def case_transport(model, anchor, source, old, wc, basis):
    """Run the same batched native continuations as LS-SAE08, retaining hidden norms."""
    position = FAMILY.median_blank(source["x"])
    token = model.puzzle_emb_len + position
    if position != old["visible_goal_position"] or source["seed"] != old["seed"]:
        raise ValueError("LS-SAE08 frozen source or intervention site altered")
    seq = [("clean", None, None, 0)]
    vectors = {}
    for arm in SCORE.ARMS:
        geometry = old["direction_metadata"]["geometry"][arm]
        status = old["source_direction_status"][arm]
        if status != (geometry is not None):
            raise ValueError("LS-SAE08 geometry ineligible status mismatch")
        if not status:
            vectors[arm] = None
            continue
        unit = np.asarray(geometry["direction_512"], dtype=np.float64)
        if unit.shape != (512,) or not np.all(np.isfinite(unit)):
            raise ValueError("original LS-SAE08 physical vector damaged")
        if not np.isclose(np.linalg.norm(unit), 1.0, atol=2e-5):
            raise ValueError("original LS-SAE08 physical direction changed norm")
        vectors[arm] = unit
        for radius in SCORE.RADII:
            for sign in ("plus", "minus"):
                seq.append((arm, f"{radius:.2f}", sign, 1 if sign == "plus" else -1))
    n = len(seq)
    blanks = np.flatnonzero(np.asarray(source["x"]) == 0)
    givens = np.flatnonzero(np.asarray(source["x"]) != 0)
    output = {}
    source_norm = float(torch.linalg.vector_norm(anchor["zh"][0, token].float()))
    if not np.isclose(source_norm, old["original_source_zH_token_norm"], atol=TOL, rtol=1e-5):
        raise ValueError("reloaded ACT1 z_H source differs from LS-SAE08")
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        zh = anchor["zh"].detach().repeat(n, 1, 1)
        zl = anchor["zl"].detach().repeat(n, 1, 1)
        x_emb = anchor["x_emb"].detach().repeat(n, 1, 1)
        for index, (arm, r, sign, direction_sign) in enumerate(seq[1:], 1):
            unit = torch.as_tensor(vectors[arm], dtype=zh.dtype, device=zh.device)
            zh[index, token, :] += float(r) * direction_sign * unit
        for step in range(1, 12):
            zh, zl = model.act_step(zh, zl, x_emb)
            if step not in (2, 5, 11):
                continue
            horizon = "ACT" + str(step + 1)
            old_logits = old["native_81x9_logits"][horizon]
            l = model.digit_logits(zh).detach().double().cpu().numpy()
            if l.shape != (n, 81, 9) or not np.isfinite(l).all():
                raise ArithmeticError("native full81 logits are invalid")
            baseline_clean = np.asarray(old_logits["clean"], dtype=np.float64)
            clean_drift = float(np.max(np.abs(l[0] - baseline_clean)))
            if clean_drift > ARCHIVE_TOL or not np.array_equal(
                l[0].argmax(axis=1), baseline_clean.argmax(axis=1)
            ):
                raise ValueError("clean replay drift or all81 predicted digit changed")
            delh = (zh[1:].float() - zh[0:1].float()).detach().double().cpu().numpy()
            dell = (zl[1:].float() - zl[0:1].float()).detach().double().cpu().numpy()
            rows = []
            for index, (arm, r, sign, _) in enumerate(seq[1:]):
                historical = np.asarray(old_logits[arm][r][sign], dtype=np.float64)
                actual = l[index + 1]
                drift = float(np.max(np.abs(actual - historical)))
                if drift > ARCHIVE_TOL or not np.array_equal(
                    actual.argmax(axis=1), historical.argmax(axis=1)
                ):
                    raise ValueError(
                        f"signed native replay drift {horizon}/{arm}/{r}/{sign}: {drift}"
                    )
                dh = delh[index]
                dl = dell[index]
                board = dh[model.puzzle_emb_len:, :]
                h_global = float(np.linalg.norm(dh))
                h_site = float(np.linalg.norm(dh[token]))
                h_other = float(np.linalg.norm(np.delete(dh, token, axis=0)))
                board_norm = float(np.linalg.norm(board))
                projection = board @ basis
                readout_norm = float(np.linalg.norm(projection))
                null_norm = float(np.linalg.norm(board - projection @ basis.T))
                fraction = float(readout_norm / board_norm) if board_norm > 1e-12 else 0.0
                centered_logits = actual - l[0]
                centered_logits -= centered_logits.mean(axis=1, keepdims=True)
                logit_error = float(np.linalg.norm(centered_logits - board @ wc.T))
                if logit_error > TOL:
                    raise ArithmeticError("centered native logit delta/readout identity violated")
                score = pressure_and_flips(l[0], actual, source)
                dose = float(r)
                record = {
                    "arm": arm, "radius": r, "sign": sign,
                    "h_global_l2": h_global, "h_gain": h_global / dose,
                    "l_global_l2": float(np.linalg.norm(dl)),
                    "l_gain": float(np.linalg.norm(dl)) / dose,
                    "h_source_site_l2": h_site, "h_non_site_l2": h_other,
                    "h_blank_l2": float(np.linalg.norm(board[blanks])),
                    "h_given_l2": float(np.linalg.norm(board[givens])),
                    "h_board_l2": board_norm,
                    "h_readout_l2": readout_norm, "h_null_l2": null_norm,
                    "readout_fraction": fraction,
                    "centered_logit_fro": float(np.linalg.norm(centered_logits)),
                    "centered_readout_reconstruction_error": logit_error,
                    "original_logits_max_absolute_drift": drift,
                    **score,
                }
                record["classification"] = classification(
                    record["h_gain"], score["blank_flip_count"], board_norm, fraction
                )
                rows.append(record)
            output[horizon] = {
                "original_clean_logit_max_absolute_drift": clean_drift,
                "clean_blank_correct": int(np.sum(
                    np.argmax(l[0], axis=1)[blanks] + 1
                    == np.asarray(source["target"])[blanks])),
                "variants": rows,
            }
    if set(output) != set(HORIZONS):
        raise AssertionError("ACT3/6/12 horizon coverage changed")
    return {
        "seed": source["seed"], "visible_goal_position": position,
        "original_source_zH_norm": source_norm,
        "visible_structural_difficulty": source["visible_structural_difficulty"],
        "direction_status": old["source_direction_status"],
        "horizons": output,
    }


def run(pilot, source_dir, baseline_path, out):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    original, baseline_sha = load_baseline(Path(baseline_path), pilot)
    source_dir = Path(source_dir)
    fit, manifest, manifest_sha = FITPACK.load_pack(
        source_dir / "frozen_calibration.npz",
        source_dir / "frozen_calibration_manifest.json",
    )
    if (manifest_sha != S6.MANIFEST_SHA or
        manifest["npz_file_sha256"] != S6.NPZ_SHA or
        fit.fingerprint != S6.FIT_SHA or
        manifest["physical_dictionary_sha256"] != S6.PHYSICAL_SHA):
        raise ValueError("original preheldout source fit SHA mismatch")
    model, checkpoint_sha = TRM.load_trm_checkpoint(
        "gaoxin492", "cpu", dtype=torch.float32
    )
    if checkpoint_sha != S6.MODEL_SHA:
        raise ValueError("original checkpoint source SHA mismatch")
    model.eval().requires_grad_(False)
    wc, basis, singular_values = readout_basis(model)
    groups = ("pilot_52", "pilot_54") if pilot else ("test_52", "test_54")
    families = {}
    for family in groups:
        roster = FAMILY.roster(family)
        old = original["families"][family]["per_puzzle"]
        if roster != original["families"][family]["source_roster"]:
            raise ValueError("LS-SAE08 original source roster changed")
        _, anchors = SOURCEFIT.stage_one(model, roster, retain_zl=True)
        cases = []
        for index, (source, prior, anchor) in enumerate(zip(roster, old, anchors)):
            if prior["puzzle_index"] != index:
                raise ValueError("LS-SAE08 puzzle order changed")
            case = case_transport(model, anchor, source, prior, wc, basis)
            cases.append(case)
            key = [x for x in case["horizons"]["ACT6"]["variants"]
                   if x["arm"] == "state_power" and x["radius"] == "8.00"]
            print("LS_SAE09_CASE", family, index, "ACT6_STATE_POWER",
                  [(x["sign"], round(x["h_gain"], 6),
                    round(x["readout_fraction"], 6),
                    x["blank_flip_count"]) for x in key], flush=True)
        families[family] = {"per_puzzle": cases}
    payload = {
        "schema": SCHEMA, "preregistration_commit": PROTOCOL_COMMIT,
        "source_ls_sae08_json_sha256": baseline_sha,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_calibration_npz_sha256": S6.NPZ_SHA,
        "source_calibration_manifest_sha256": manifest_sha,
        "source_fit_array_sha256": fit.fingerprint,
        "physical_dictionary_sha256": S6.PHYSICAL_SHA,
        "mode": "pilot" if pilot else "full",
        "horizons": list(HORIZONS),
        "radii": [f"{v:.2f}" for v in SCORE.RADII],
        "arms": list(SCORE.ARMS),
        "readout_centered_rank": int(basis.shape[1]),
        "readout_nonzero_singular_values": singular_values,
        "max_original_logit_replay_drift": ARCHIVE_TOL,
        "same_run_readout_identity_tolerance": TOL,
        "families": families,
        "scope": "Registered descriptive state-transport replay; summary-state norms cannot independently reconstruct 97x512 native tensors.",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run(args.pilot, args.source_dir, args.baseline, args.out)
    print(json.dumps({"mode": result["mode"], "out": str(args.out),
                      "cohorts": {k: len(v["per_puzzle"])
                                  for k, v in result["families"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
