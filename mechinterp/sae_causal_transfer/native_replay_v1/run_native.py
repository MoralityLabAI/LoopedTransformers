"""LS-SAE10 retrospective two-sentinel native numeric reproducibility diagnostic.

Four frozen forward modes, archived LS-SAE08 81x9-logit comparisons and raw
numeric receipts. Deliberately does NOT stop on cross-run drift: that is the
quantity being measured, not a posthoc tolerance-adjusted scientific gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as STAGE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as S6
from mechinterp.sae_causal_transfer.native_hard_task_v1 import run_native as NATIVE8
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as AUDIT8
from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAMILIES

SCHEMA = "LS-SAE10/retrospective-original-numeric-replay-forensics-v1"
PROTOCOL_COMMIT = "2c9a5db2ca39f82f94649df66aea6907a80d47ea"
ORIGINAL_FULL_SHA = "69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
SENTINELS = (("test_52", 0), ("test_54", 3))
MODES = ("canonical_a", "canonical_b", "instrumented", "single_source")
HORIZONS = ("ACT3", "ACT6", "ACT12")


def sha_tensor(value):
    arr = value.detach().float().contiguous().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def source_stats(pair, single):
    out = {}
    for key in ("zh", "zl", "x_emb"):
        a = pair[key].detach().double().cpu().numpy()
        b = single[key].detach().double().cpu().numpy()
        if a.shape != (1, 97, 512) or b.shape != a.shape:
            raise ValueError("ACT1 original H/L/X embedding shape mismatch")
        out[key] = {
            "paired_sha256": sha_tensor(pair[key]),
            "single_sha256": sha_tensor(single[key]),
            "max_abs_paired_minus_single": float(np.max(np.abs(a - b))),
            "fro_paired_minus_single": float(np.linalg.norm(a - b)),
        }
    return out


def frozen_dirs(old):
    vectors = {}
    for arm in SCORE.ARMS:
        admitted = old["source_direction_status"][arm]
        geometry = old["direction_metadata"]["geometry"][arm]
        if admitted != (geometry is not None):
            raise ValueError("original LS-SAE08 direction eligibility changed")
        if geometry is None:
            vectors[arm] = None
            continue
        x = np.asarray(geometry["direction_512"], dtype=np.float64)
        if x.shape != (512,) or not np.isfinite(x).all():
            raise ValueError("missing original 512D finite direction")
        if abs(float(np.linalg.norm(x)) - 1) > 2e-5:
            raise ValueError("original LS-SAE08 source direction not unit length")
        vectors[arm] = x
    return vectors


def ordered_edits(directions):
    ordering = [("clean", None, None, 0)]
    for arm in SCORE.ARMS:
        if directions[arm] is None:
            continue
        for radius in SCORE.RADII:
            for sign in ("plus", "minus"):
                ordering.append((arm, float(radius), sign, 1 if sign == "plus" else -1))
    return ordering


def instrumented_forward(model, anchor, position, directions):
    """Shadow historical batched ACT forward, without modifying hidden states."""
    ordering = ordered_edits(directions)
    token = model.puzzle_emb_len + position
    n = len(ordering)
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        zh = anchor["zh"].detach().repeat(n, 1, 1)
        zl = anchor["zl"].detach().repeat(n, 1, 1)
        x_emb = anchor["x_emb"].detach().repeat(n, 1, 1)
        for i, (arm, radius, sign, s) in enumerate(ordering[1:], 1):
            zh[i, token] += radius * s * torch.tensor(
                directions[arm], dtype=zh.dtype, device=zh.device
            )
        output = {}
        states = {}
        for step in range(1, 12):
            zh, zl = model.act_step(zh, zl, x_emb)
            if step not in (2, 5, 11):
                continue
            horizon = f"ACT{step + 1}"
            z = model.digit_logits(zh).detach().double().cpu().numpy()
            if z.shape != (n, 81, 9) or not np.isfinite(z).all():
                raise ArithmeticError("instrumented 81x9 output invalid")
            record = {"clean": z[0].tolist()}
            for i, (arm, radius, sign, _) in enumerate(ordering[1:], 1):
                record.setdefault(arm, {}).setdefault(f"{radius:.2f}", {})[sign] = z[i].tolist()
            output[horizon] = record
            states[horizon] = {
                "clean_zh_sha256": sha_tensor(zh[0:1]),
                "clean_zl_sha256": sha_tensor(zl[0:1]),
                "clean_zh_l2": float(torch.linalg.vector_norm(zh[0].double())),
                "clean_zl_l2": float(torch.linalg.vector_norm(zl[0].double())),
            }
    if set(output) != set(HORIZONS):
        raise AssertionError("native ACT horizons changed")
    return output, states


def each_variant(raw, arms):
    for horizon in HORIZONS:
        block = raw[horizon]
        yield horizon, "clean", block["clean"]
        for arm in SCORE.ARMS:
            if not arms[arm]:
                if arm in block:
                    raise ValueError("original ineligible arm unexpectedly evaluated")
                continue
            for radius in SCORE.RADII:
                dose = f"{radius:.2f}"
                for sign in ("plus", "minus"):
                    yield horizon, f"{arm}/{dose}/{sign}", block[arm][dose][sign]


def variant_lookup(raw, horizon, name):
    if name == "clean":
        return raw[horizon]["clean"]
    arm, dose, sign = name.split("/")
    return raw[horizon][arm][dose][sign]


def drift_stats(left, right, blank_positions):
    """Paired fixed logits; stored output is independently auditable."""
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    if l.shape != (81, 9) or r.shape != l.shape or not (
        np.isfinite(l).all() and np.isfinite(r).all()
    ):
        raise ArithmeticError("nonfinite or incomplete 81x9 paired logits")
    delta = r - l
    center = delta - delta.mean(axis=1, keepdims=True)
    a, b = l.argmax(axis=1), r.argmax(axis=1)
    return {
        "max_absolute_drift": float(np.abs(delta).max()),
        "centered_drift_rms": float(np.sqrt(np.mean(center ** 2))),
        "all81_argmax_changes": int(np.count_nonzero(a != b)),
        "blank_argmax_changes": int(np.count_nonzero(a[blank_positions] != b[blank_positions])),
    }


def comparison(ref, actual, arms, blanks):
    return {
        horizon: {
            name: drift_stats(orig, variant_lookup(actual, horizon, name), blanks)
            for h, name, orig in each_variant(ref, arms) if h == horizon
        } for horizon in HORIZONS
    }


def run(baseline_path: Path, out: Path):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    original_bytes = baseline_path.read_bytes()
    full_sha = hashlib.sha256(original_bytes).hexdigest()
    if full_sha != ORIGINAL_FULL_SHA:
        raise ValueError("original LS-SAE08 full.json byte SHA differs from registration")
    original = json.loads(original_bytes)
    AUDIT8.audit(original)
    if original["mode"] != "full":
        raise ValueError("LS-SAE08 engineering pilot substituted for full source")
    model, sha = TRM.load_trm_checkpoint("gaoxin492", "cpu", dtype=torch.float32)
    if sha != S6.MODEL_SHA:
        raise ValueError("original pretrained Sudoku checkpoint changed")
    model.eval().requires_grad_(False)
    cases = []
    for family, index in SENTINELS:
        roster = FAMILIES.roster(family)
        if roster != original["families"][family]["source_roster"]:
            raise ValueError("historical first-admissible source roster changed")
        row = roster[index]
        old = original["families"][family]["per_puzzle"][index]
        pos = FAMILIES.median_blank(row["x"])
        if old["seed"] != row["seed"] or old["visible_goal_position"] != pos:
            raise ValueError("source position or seed deviated from LS-SAE08")
        # Paired batch size 2 exactly as in the historical run.
        _, paired_anchors = STAGE.stage_one(model, roster, batch_size=2, retain_zl=True)
        _, single_anchors = STAGE.stage_one(model, [row], batch_size=1, retain_zl=True)
        anchor, lone = paired_anchors[index], single_anchors[0]
        physical = frozen_dirs(old)
        records = {}
        records["canonical_a"] = NATIVE8.native_all_outputs(model, anchor, pos, physical)
        records["canonical_b"] = NATIVE8.native_all_outputs(model, anchor, pos, physical)
        records["instrumented"], hidden = instrumented_forward(model, anchor, pos, physical)
        records["single_source"] = NATIVE8.native_all_outputs(model, lone, pos, physical)
        blanks = np.flatnonzero(np.asarray(row["x"]) == 0)
        status = old["source_direction_status"]
        for name, raw in records.items():
            # Fail for missing/altered grid; output drift is intentionally recorded.
            if [k[:2] for k in each_variant(raw, status)] != [
                k[:2] for k in each_variant(old["native_81x9_logits"], status)
            ]:
                raise ValueError("original arm, horizon, sign and physical dose grid changed")
        tests = {
            name: {
                "vs_historical": comparison(old["native_81x9_logits"], raw, status, blanks),
                "vs_canonical_a": comparison(records["canonical_a"], raw, status, blanks),
            } for name, raw in records.items()
        }
        maxima = {
            name: max(d["max_absolute_drift"]
                      for by_horizon in mode["vs_historical"].values()
                      for d in by_horizon.values())
            for name, mode in tests.items()
        }
        case = {
            "family":family, "index":index, "seed":row["seed"],
            "visible_goal_position":pos,
            "blank_count":int(len(blanks)),
            "direction_status":status,
            "source_anchor_comparison":source_stats(anchor, lone),
            "modes":records, "instrumented_clean_state":hidden,
            "comparisons":tests,
            "max_original_drift_by_mode":maxima,
        }
        cases.append(case)
        print("LS_SAE10_CASE",family,index,"seed",row["seed"],
              "max_original_drift",json.dumps(maxima,sort_keys=True),
              "fresh_repeat_max", max(x["max_absolute_drift"]
                                    for per_h in tests["canonical_b"]["vs_canonical_a"].values()
                                    for x in per_h.values()), flush=True)
    result = {
        "schema":SCHEMA, "preregistration_commit":PROTOCOL_COMMIT,
        "mode":"retrospective_two_sentinels",
        "source_ls_sae08_sha256":full_sha,
        "source_checkpoint_sha256":sha,
        "sentinels":[{"family":f,"index":i} for f,i in SENTINELS],
        "modes":list(MODES), "horizons":list(HORIZONS),
        "physical_radii":[f"{x:.2f}" for x in SCORE.RADII],
        "direction_arm_order":list(SCORE.ARMS),
        "cpu_threads":4,
        "cases":cases,
        "limits":"Descriptive retrospective drift diagnostic. No LS-SAE09 scientific replay acceptance or recomputed primary gate.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=run(args.baseline,args.out)
    print(json.dumps({"out":str(args.out),
                      "maxima":{x["family"]:x["max_original_drift_by_mode"] for x in r["cases"]}},
                     indent=2))


if __name__=="__main__":
    main()
