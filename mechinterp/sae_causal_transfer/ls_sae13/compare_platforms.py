"""Compare LS-SAE13 exploratory receipts across CPU or CUDA platforms.

Only matches identical source, checkpoint, puzzle/site, float32 edit SHA and
batch layouts. The ACT5 centered-logit effect is the main diagnostic.
No p-values or confirmatory gates are calculated.

python -m mechinterp.sae_causal_transfer.ls_sae13.compare_platforms \
  --run linux=runs/ls_sae13/pilot-linux-cpu.json \
  --run windows=runs/ls_sae13/pilot-windows-cpu.json \
  --out runs/ls_sae13/compare-linux-windows.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from mechinterp.sae_causal_transfer.ls_sae13 import run_pilot as P


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("effect arrays have incompatible sizes")
    a = a - a.mean()
    b = b - b.mean()
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if scale <= 1e-25 else float(np.dot(a, b) / scale)


def retained(probe, act, tag, batch):
    record = next(s for s in probe["per_step"] if s["act"] == act)
    raw = record["retained_float32_logits"]
    if raw is None:
        raise ValueError("missing retained ACT logits")
    if tag == "clean_minus" and batch == 2:
        # The B2 negative edit was run with a second clean baseline.
        return P.as_f32(raw["clean_minus_reference"])
    return P.as_f32(raw["clean"] if tag == "clean_minus" else raw[tag])


def effects(probe, act, tag, batch):
    clean = retained(probe, act, "clean_minus" if tag == "minus" else "clean", batch)
    edited = retained(probe, act, tag, batch)
    d = edited.astype(np.float64) - clean.astype(np.float64)
    centered = d - d.mean(axis=-1, keepdims=True)
    return clean, edited, centered


def compare_pair(name_a, a, name_b, b):
    if a["schema"] != P.SCHEMA or b["schema"] != P.SCHEMA:
        raise ValueError("wrong receipt schema")
    for key in ("checkpoint_sha256", "source_sha256", "positions", "batches", "saved_logits_acts"):
        if a[key] != b[key]:
            raise ValueError(f"receipts mismatch: {key}")
    by_pos_a = {r["pool_position"]: r for r in a["boards"]}
    by_pos_b = {r["pool_position"]: r for r in b["boards"]}
    if set(by_pos_a) != set(by_pos_b):
        raise ValueError("different board sets")
    rows = []
    for pos in a["positions"]:
        ra, rb = by_pos_a[pos], by_pos_b[pos]
        if (ra["site"] != rb["site"] or ra["edit_sha256"] != rb["edit_sha256"]
                or ra["blank_cells"] != rb["blank_cells"]):
            raise ValueError(f"different site, blanks or edit for puzzle {pos}")
        if ra["edit_little_endian_float32_hex"] != rb["edit_little_endian_float32_hex"]:
            raise ValueError("edit SHA collision or corrupted receipt")
        blanks = np.array(ra["blank_cells"], dtype=np.int64)
        for batch in a["batches"]:
            pa, pb = ra["probe"][str(batch)], rb["probe"][str(batch)]
            for act in a["saved_logits_acts"]:
                for tag in ("plus", "minus"):
                    ca, ea, d_a = effects(pa, act, tag, batch)
                    cb, eb, d_b = effects(pb, act, tag, batch)
                    shift_a = P.margins(ea) - P.margins(ca)
                    shift_b = P.margins(eb) - P.margins(cb)
                    eligible = (np.maximum(np.abs(shift_a[blanks]), np.abs(shift_b[blanks])) >= 1e-5)
                    row = {
                        "position": pos, "batch": batch, "act": act, "sign": tag,
                        "centered_effect_pearson": pearson(d_a, d_b),
                        "blank_centered_effect_pearson": pearson(d_a[blanks], d_b[blanks]),
                        "centered_effect_l2_a": float(np.linalg.norm(d_a)),
                        "centered_effect_l2_b": float(np.linalg.norm(d_b)),
                        "centered_effect_max_abs_difference": float(np.max(np.abs(d_a - d_b))),
                        "margin_shift_pearson": pearson(shift_a[blanks], shift_b[blanks]),
                        "margin_shift_cells_above_1e_minus_5": int(np.sum(eligible)),
                        "margin_shift_sign_agreement": (
                            float(np.mean(np.sign(shift_a[blanks][eligible]) == np.sign(shift_b[blanks][eligible])))
                            if np.any(eligible) else None),
                        "clean_argmax_agreement_cells": int(np.sum(np.argmax(ca[blanks], axis=-1) == np.argmax(cb[blanks], axis=-1))),
                        "edited_argmax_agreement_cells": int(np.sum(np.argmax(ea[blanks], axis=-1) == np.argmax(eb[blanks], axis=-1))),
                        "vdd_a": pa["trajectories"][tag]["deduction_change"]["vdd"],
                        "vdd_b": pb["trajectories"][tag]["deduction_change"]["vdd"],
                        "clean_first_divergence_a": pa["first_clean_digit_divergence_from_batch1"],
                        "clean_first_divergence_b": pb["first_clean_digit_divergence_from_batch1"],
                    }
                    rows.append(row)
    act5 = [r for r in rows if r["act"] == 5 and r["batch"] == 61]
    return {
        "schema": "LS-SAE13/cross-platform-comparison-v1",
        "inputs": {name_a: a["platform"], name_b: b["platform"]},
        "primary_diagnostic": "ACT5 centered 81x9 logit shift after the exact same ACT4 edit",
        "matched_acts": list(a["saved_logits_acts"]),
        "matched_board_count": len(by_pos_a), "rows": rows,
        "act5_batch61": act5,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="LABEL=path/to/receipt.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if len(args.run) != 2:
        p.error("pass exactly two LABEL=path inputs")
    inputs = []
    for item in args.run:
        label, path = item.split("=", 1)
        inputs.append((label, json.loads(Path(path).read_text())))
    x = compare_pair(*inputs[0], *inputs[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(x, sort_keys=True, indent=1, allow_nan=False) + "\n")
    print("LS_SAE13_COMPARE", args.out, "rows", len(x["rows"]))
    for r in x["act5_batch61"]:
        print("ACT5_BATCH61", r["position"], r["sign"],
              "centered_r", r["centered_effect_pearson"], "margin_r",
              r["margin_shift_pearson"], "VDD", r["vdd_a"], r["vdd_b"])


if __name__ == "__main__":
    main()
