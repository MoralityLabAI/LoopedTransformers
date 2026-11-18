"""LS-SAE13 exploratory CPU diagnostic: matched ACT4 edits and batch-shape invariance.

Run:
 python -m mechinterp.sae_causal_transfer.ls_sae13.run_pilot \
   --source mechinterp/sae_causal_transfer/results/ls_sae12/source_48.json \
   --out runs/ls_sae13/pilot-linux-cpu.json

Uses LS-SAE12 source admission and pretrained checkpoint unchanged.
No hypothesis gate and no LS-SAE12 registered workflow mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.sae_causal_transfer.active_reasoning_v1 import run_native as RN
from mechinterp.sae_causal_transfer.active_reasoning_v1 import schedule as S

SCHEMA = "LS-SAE13/numerical-stability-pilot-v1"
PROTOCOL_COMMIT = "e68b2302e74849d372f74537a8a69ea4441bb780"
POSITIONS = (23, 24, 28, 40)
SITES = {23: 40, 24: 40, 28: 41, 40: 43}
BATCHES = (1, 2, 21, 61)
RETAIN_LOGITS_STEPS = (5, 8, 16)
EDIT_AMPLITUDE = np.float32("0.0008838834764831844")
EDIT_SHA256 = {
    23: "58835aafeed6f44b92d93c78a1f174261737a33d5fb03a872cb9ebad8be20ca0",
    24: "4d08b318a329dca836c90161d29a1372d575859438602ac5b748a5ae99d445d3",
    28: "a0af587f02075b5332410d39f8253c1966eeb9029f65b96d6c55d2c5c9af4704",
    40: "ba7b20e199dc843b8a552c6175a53facaac01433beb83712f4770a663f4a4380",
}


def as_f32(a):
    return np.asarray(a, dtype="<f4")


def sha_logits(a):
    return hashlib.sha256(as_f32(a).tobytes(order="C")).hexdigest()


def locked_edit(position: int, site: int) -> np.ndarray:
    """Rademacher signs from SHA256, exactly pinned IEEE float32 components."""
    if position not in SITES or site != SITES[position]:
        raise ValueError("unregistered position or median-blank site")
    v = np.empty(512, dtype="<f4")
    for i in range(512):
        bit = hashlib.sha256(
            f"LS-SAE13/rademacher-v1/{position}/{site}/{i}".encode("ascii")
        ).digest()[0] & 1
        v[i] = EDIT_AMPLITUDE if bit else -EDIT_AMPLITUDE
    if len(v.tobytes()) != 2048:
        raise ArithmeticError("unexpected float32 vector size")
    if hashlib.sha256(v.tobytes()).hexdigest() != EDIT_SHA256[position]:
        raise ArithmeticError("locked float32 edit digest mismatch")
    if abs(float(np.linalg.norm(v.astype(np.float64))) - 0.02) > 1e-7:
        raise ArithmeticError("edit norm mismatch")
    return v


def layouts(batch: int) -> list[list[str]]:
    if batch == 1:
        return [["clean"], ["plus"], ["minus"]]
    if batch == 2:
        return [["clean", "plus"], ["clean", "minus"]]
    if batch in (21, 61):
        return [["clean", "plus", "minus"] + ["clean"] * (batch - 3)]
    raise ValueError("unregistered batch size")


def margins(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (81, 9):
        raise ValueError("need 81x9 logits")
    ordered = np.sort(a, axis=-1)
    return ordered[:, -1] - ordered[:, -2]


def logits_pair(clean: np.ndarray, other: np.ndarray, x: list[int]) -> dict:
    a, b = np.asarray(clean, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if a.shape != (81, 9) or b.shape != a.shape or not np.isfinite(b).all():
        raise ArithmeticError("invalid logits")
    delta = b - a
    centered = delta - delta.mean(axis=1, keepdims=True)
    cells = np.array(S.blanks(x), dtype=np.int64)
    margin = margins(a)[cells]
    eps = np.max(np.abs(delta[cells]), axis=-1)
    certified = margin > 2.0 * eps
    flips = np.argmax(a[cells], axis=-1) != np.argmax(b[cells], axis=-1)
    if np.any(certified & flips):
        raise ArithmeticError("sufficient argmax certificate contradicted")
    return {
        "max_abs_logit_delta": float(np.max(np.abs(delta))),
        "centered_logit_rms": float(np.sqrt(np.mean(centered * centered))),
        "clean_min_blank_margin": float(np.min(margin)),
        "certified_blank_cells": int(np.sum(certified)),
        "uncertified_blank_cells": int(np.sum(~certified)),
        "actual_blank_argmax_flips": int(np.sum(flips)),
        "min_margin_over_twice_epsilon": float(np.min(margin / np.maximum(2.0 * eps, 1e-30))),
    }


def forward_layout(model, anchor: dict, site: int, edit: np.ndarray, layout: list[str]):
    """From the exact same ACT4 anchor, continue all native rows in one batch."""
    b = len(layout)
    if b not in BATCHES or (b > 1 and layout[0] != "clean") or len(edit) != 512:
        raise ValueError("bad layout")
    if len(layout) != b or any(tag not in ("clean", "plus", "minus") for tag in layout):
        raise ValueError("bad row labels")
    token = model.puzzle_emb_len + site
    saved = {"clean": [], "plus": [], "minus": []}
    pads = []
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        zh = anchor["zh"].repeat(b, 1, 1)
        zl = anchor["zl"].repeat(b, 1, 1)
        x_emb = anchor["x_emb"].repeat(b, 1, 1)
        plus = torch.from_numpy(edit.copy())
        for i, tag in enumerate(layout):
            if tag == "plus":
                zh[i, token] += plus
            elif tag == "minus":
                zh[i, token] -= plus
        for step in range(5, 17):
            zh, zl = model.act_step(zh, zl, x_emb)
            lg = model.digit_logits(zh).detach().cpu().numpy()
            if lg.shape != (b, 81, 9) or not np.isfinite(lg).all():
                raise ArithmeticError(f"nonfinite ACT{step} logits or bad shape")
            for i, tag in enumerate(layout[:3]):
                saved[tag].append({
                    "logits": as_f32(lg[i]).copy(),
                    "zh": as_f32(zh[i].detach().cpu().numpy()).copy(),
                    "zl": as_f32(zl[i].detach().cpu().numpy()).copy(),
                })
            if b >= 21:
                max_delta = float(np.max(np.abs(lg[3:].astype(np.float64) - lg[0].astype(np.float64))))
                pad_different = sum(not np.array_equal(lg[i], lg[0]) for i in range(3, b))
                pads.append({"act": step, "max_abs_clean_padding_logit_difference": max_delta,
                             "padding_rows_different": pad_different, "padding_count": b - 3})
    return saved, pads


def run_batches(model, anchor: dict, x: list[int], target: list[int],
                site: int, edit: np.ndarray, screen_preds, screen_logits):
    results = {}
    blanks = S.blanks(x)
    for batch in BATCHES:
        rows = {"clean": [], "plus": [], "minus": []}
        pad_checks = []
        for layout in layouts(batch):
            got, pads = forward_layout(model, anchor, site, edit, layout)
            for tag in rows:
                rows[tag].extend(got[tag])
            pad_checks.extend(pads)

        # B2 has two separately computed clean baselines, one in each layout.
        # Other batch sizes have one. All other labels must have exactly 12 steps.
        expected_clean_steps = 24 if batch == 2 else 12
        if len(rows["clean"]) != expected_clean_steps or any(len(rows[t]) != 12 for t in ("plus", "minus")):
            raise ArithmeticError("batch row receipt shape changed")

        duplicate_clean = None
        if batch == 2:
            duplicate_clean = [{
                "act": 5 + j,
                "bit_identical": sha_logits(rows["clean"][j]["logits"])
                                 == sha_logits(rows["clean"][12 + j]["logits"]),
                "max_abs_logit_difference": float(np.max(np.abs(
                    rows["clean"][j]["logits"].astype(np.float64) -
                    rows["clean"][12 + j]["logits"].astype(np.float64)))),
            } for j in range(12)]
        clean = rows["clean"][:12]

        by_tag = {}
        for tag in ("clean", "plus", "minus"):
            traj = rows[tag][:12]
            preds = list(screen_preds[:4]) + [RN.argmax_digits(row["logits"]) for row in traj]
            sched = S.verified_schedule(preds, x, target)
            by_tag[tag] = {
                "predicted_digits_by_act": preds,
                "schedule": {str(k): v for k, v in sched.items()},
                "ACT16_blank_correct": sum(preds[-1][c] == target[c] for c in blanks),
                "ACT16_exact": all(preds[-1][c] == target[c] for c in blanks),
            }
        clean_sched = {int(k): v for k, v in by_tag["clean"]["schedule"].items()}
        negative_clean = rows["clean"][12:24] if batch == 2 else clean
        negative_clean_preds = list(screen_preds[:4]) + [
            RN.argmax_digits(row["logits"]) for row in negative_clean
        ]
        negative_clean_sched = S.verified_schedule(negative_clean_preds, x, target)
        for tag in ("plus", "minus"):
            edited_sched = {int(k): v for k, v in by_tag[tag]["schedule"].items()}
            own_clean = negative_clean_sched if tag == "minus" else clean_sched
            by_tag[tag]["deduction_change"] = S.deduction_change(own_clean, edited_sched, site)

        per_step = []
        first_clean_different_act = None
        for j in range(12):
            act = j + 5
            cr = clean[j]
            screen_lg = as_f32(screen_logits[act - 1])
            c_comparison = logits_pair(screen_lg, cr["logits"], x)
            full81_flips = int(np.sum(np.argmax(screen_lg, axis=-1) != np.argmax(cr["logits"], axis=-1)))
            c_comparison["full81_argmax_flips"] = full81_flips
            if full81_flips and first_clean_different_act is None:
                first_clean_different_act = act
            signs = {}
            for tag in ("plus", "minus"):
                er = rows[tag][j]
                own_cr = negative_clean[j] if tag == "minus" else cr
                measurements = logits_pair(own_cr["logits"], er["logits"], x)
                measurements.update({
                    "h_difference_norm": float(np.linalg.norm(
                        (er["zh"].astype(np.float64) - own_cr["zh"].astype(np.float64)).ravel())),
                    "l_difference_norm": float(np.linalg.norm(
                        (er["zl"].astype(np.float64) - own_cr["zl"].astype(np.float64)).ravel())),
                    "h_gain_over_0p02": float(np.linalg.norm(
                        (er["zh"].astype(np.float64) - own_cr["zh"].astype(np.float64)).ravel()) / 0.02),
                    "logit_sha256": sha_logits(er["logits"]),
                })
                signs[tag] = measurements
            saved_logits = ({tag: rows[tag][j]["logits"].tolist()
                             for tag in ("clean", "plus", "minus")} if act in RETAIN_LOGITS_STEPS else None)
            if saved_logits is not None and batch == 2:
                saved_logits["clean_minus_reference"] = negative_clean[j]["logits"].tolist()
            per_step.append({
                "act": act,
                "clean_sha256": sha_logits(cr["logits"]),
                "clean_against_screen": c_comparison,
                "edits": signs,
                "retained_float32_logits": saved_logits,
            })
        if batch == 1 and first_clean_different_act is not None:
            raise ArithmeticError("batch-1 continuation changed its clean screening digits")

        results[str(batch)] = {
            "batch_size": batch,
            "first_clean_digit_divergence_from_batch1": first_clean_different_act,
            "baseline_repeat_comparison": duplicate_clean,
            "padding_checks": pad_checks,
            "per_step": per_step,
            "trajectories": by_tag,
        }
    return results


def run(source: Path, out: Path):
    source_data, source_sha = RN.load_source(source)
    by_pos = {r["pool_position"]: r for r in source_data["pool"]}
    model, checkpoint_sha = RN.load_model()
    board_results = []
    for position in POSITIONS:
        src = by_pos[position]
        x, target = src["x"], src["target"]
        site = S.control_site(x)
        if site != SITES[position]:
            raise ArithmeticError(f"source/median site mismatch for position {position}")
        edit = locked_edit(position, site)
        screen_preds, screen_logits, anchor = RN.clean_trajectory(model, x)
        r = {
            "pool_position": position, "source_file": src["source_file"],
            "site": site, "site_rule": "median_blank", "blank_cells": S.blanks(x),
            "edit_sha256": EDIT_SHA256[position],
            "edit_little_endian_float32_hex": edit.tobytes().hex(),
            "edit_l2_norm": float(np.linalg.norm(edit.astype(np.float64))),
            "screen_ACT16_blank_correct": sum(screen_preds[-1][c] == target[c] for c in S.blanks(x)),
            "probe": run_batches(model, anchor, x, target, site, edit, screen_preds, screen_logits),
        }
        board_results.append(r)
        print("LS_SAE13_PILOT", position, src["source_file"], "site", site,
              "first_clean_divergence", {b: r["probe"][str(b)]["first_clean_digit_divergence_from_batch1"]
                                        for b in BATCHES},
              "vdd_plus", {b: r["probe"][str(b)]["trajectories"]["plus"]["deduction_change"]["vdd"]
                           for b in BATCHES}, flush=True)

    result = {
        "schema": SCHEMA, "scope": "exploratory_engineering_no_scientific_gate",
        "protocol_registration_commit": PROTOCOL_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_receipt_fingerprint": source_data["pool_fingerprint"],
        "platform": RN.platform_fingerprint(),
        "positions": list(POSITIONS), "batches": list(BATCHES),
        "intervention_time": 4, "edit_type": "locked_rademacher",
        "saved_logits_acts": list(RETAIN_LOGITS_STEPS), "boards": board_results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print("LS_SAE13_RESULT", out, "SHA256",
          hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.out)


if __name__ == "__main__":
    main()
