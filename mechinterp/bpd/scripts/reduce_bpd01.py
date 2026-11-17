"""Independent NumPy reducer for the BPD-01 receipts.

Reads the executed JSON receipts (never the models) and writes paired per-board
statistics and Markdown tables:

- Experiment A: per-board KL for the seven registered identity controls at passes
  20, 41 and 62, the zero/pass-mean x sphere factorial on the same frozen masks
  with board-paired contrasts, and the closure-mechanism decisions.
- Experiment B: static / clean-informed / online top-k against the identity floor,
  paired online-minus-oracle differences with bootstrap CIs and sign-flip p-values,
  the reserved-64 margin statistic (descriptive at n = 16), argmax flips and mask
  disagreement, separately for CUDA BF16 and CUDA FP32.

A readout is interpretable for mechanism classification only when every identity
arm in that regime has per-board KL <= 1e-5 on all boards at that readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

READOUTS = (20, 41, 62)
CLOSURE_TOL = 1e-5
MARGIN = 0.10
SEED = 20261117
N_BOOT = 20000
N_FLIP = 100000

CONTROL_KEYS = (
    ("1_same_dtype_target_repeat", "control_0_same_dtype_target_rerun"),
    ("2_target_fp32_linear", "control_1a_target_f32_linear_vs_bf16_target"),
    ("3_full_fp32_original_weights", "control_1b_full_f32_model_vs_bf16_target"),
    ("4_fp32_bf16_quantized_weights", "control_1c_loaded_bf16_weights_full_f32"),
    ("5_svd_all_512_residual_off", "control_2a_svd_start_all_512_on"),
    ("6_svd_all_512_residual_on", "control_2a_svd_start_all_512_plus_delta"),
    ("7_trained_all_residual_on", "control_2b_trained_all_512_plus_delta_on"),
)


def paired(d: np.ndarray, *, seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = d.size
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    obs = float(d.mean())
    cnt = 0
    left = N_FLIP
    while left:
        b = min(4096, left)
        s = rng.integers(0, 2, size=(b, n)) * 2 - 1
        cnt += int((np.abs((s * d[None, :]).mean(axis=1)) >= abs(obs)).sum())
        left -= b
    return {
        "mean": obs,
        "median": float(np.median(d)),
        "bootstrap_ci95": [float(lo), float(hi)],
        "signflip_two_sided_p": (1 + cnt) / (N_FLIP + 1),
        "n_negative": int((d < 0).sum()),
        "n_positive": int((d > 0).sum()),
        "per_board": [float(x) for x in d],
    }


def margin_stat(S: np.ndarray, C: np.ndarray, O: np.ndarray, *, seed: int = SEED) -> dict[str, Any]:
    """Reserved-64 statistic M = (O - C) - 0.10 S, 97.5% bootstrap upper bound."""
    M = (O - C) - MARGIN * S
    rng = np.random.default_rng(seed + 1)
    boots = M[rng.integers(0, M.size, size=(N_BOOT, M.size))].mean(axis=1)
    return {
        "mean_M": float(M.mean()),
        "upper_97_5": float(np.quantile(boots, 0.975)),
        "below_zero": bool(np.quantile(boots, 0.975) < 0),
    }


def fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def reduce_controls(ctrl: dict[str, Any], mech: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    md: list[str] = []
    for ck_json, ck in (("gaoxin492_ref", "gaoxin492"), ("yagizdevre_tgt", "yagizdevre")):
        d = ctrl[ck_json]
        rows: dict[str, Any] = {}
        md.append(f"\n### {ck}: registered identity controls, KL(native BF16 target || arm)\n")
        for tau in (20, 62):
            md.append(f"\nPass {tau}, per board (boards 0-15), mean, max, boards > 1e-3:\n")
            md.append("| Control | " + " | ".join(str(b) for b in range(16)) + " | mean | max | n>1e-3 |")
            md.append("|---|" + "---:|" * 19)
            for label, key in CONTROL_KEYS:
                pb = np.array(d[key][f"pass_{tau}"]["per_board"], dtype=np.float64)
                rows.setdefault(label, {})[f"pass_{tau}"] = pb.tolist()
                md.append(
                    f"| {label} | " + " | ".join(fmt(v) for v in pb)
                    + f" | {fmt(pb.mean())} | {fmt(pb.max())} | {int((pb > 1e-3).sum())} |"
                )
        for label, key in CONTROL_KEYS:
            rows[label]["pass_41"] = d[key]["pass_41"]["per_board"]
        fac = d["svd_top60_ablation_2x2_factorial"]
        arms = {k: {t: np.array(v[f"pass_{t}"]["per_board"]) for t in READOUTS} for k, v in fac.items()}
        z0 = "mode_zero__sphere_rescale_false"
        z1 = "mode_zero__sphere_rescale_true"
        m0 = "mode_pass_mean__sphere_rescale_false"
        m1 = "mode_pass_mean__sphere_rescale_true"
        contrasts = {}
        for t in READOUTS:
            contrasts[f"pass_{t}"] = {
                "sphere_effect_in_zero_mode": paired(arms[z1][t] - arms[z0][t]),
                "mean_fill_effect_without_sphere": paired(arms[m0][t] - arms[z0][t]),
                "sphere_effect_in_mean_mode": paired(arms[m1][t] - arms[m0][t]),
                "interaction": paired((arms[m1][t] - arms[m0][t]) - (arms[z1][t] - arms[z0][t])),
            }
        md.append(f"\n{ck}: ablation factorial on the frozen clean-informed top-307 SVD masks (mean KL)\n")
        md.append("| Arm | pass 20 | pass 41 | pass 62 |")
        md.append("|---|---:|---:|---:|")
        for k in (z0, z1, m0, m1):
            md.append(f"| {k} | " + " | ".join(fmt(arms[k][t].mean()) for t in READOUTS) + " |")
        md.append("\nPaired contrasts at pass 20 (mean, bootstrap 95% CI, sign-flip p):\n")
        md.append("| Contrast | mean | 95% CI | p | boards + / - |")
        md.append("|---|---:|---|---:|---|")
        for name, st in contrasts["pass_20"].items():
            md.append(
                f"| {name} | {fmt(st['mean'])} | [{fmt(st['bootstrap_ci95'][0])}, {fmt(st['bootstrap_ci95'][1])}]"
                f" | {st['signflip_two_sided_p']:.4f} | {st['n_positive']} / {st['n_negative']} |"
            )
        mdec = mech["checkpoints"][ck]["decisions"]
        out[ck] = {
            "identity_controls_per_board": rows,
            "svd_relative_residual_fro": d["svd_start_relative_residual_fro"],
            "ablation_factorial_contrasts": contrasts,
            "mechanism_decisions": mdec,
        }
    return out, md


def reduce_online(rec: dict[str, Any], regime: str, identity_closed: dict[str, dict[str, bool]]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    md: list[str] = [f"\n### Online SVD, {regime}\n"]
    for ck, r in rec["metrics"].items():
        f64 = r["float64_kl_and_argmax_flips"]
        res: dict[str, Any] = {"identity_floor_closed": {}, "readouts": {}}
        md.append(f"\n{ck} (k = {r['k_keep']} of {r['components_per_module']}, zero ablation, residual off)\n")
        md.append("| Readout | repeat max | identity floor mean / max | static S | oracle C | online O | O-C mean [95% CI] | p | online better / worse | M upper 97.5% | interpretable |")
        md.append("|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|")
        for t in READOUTS:
            S = np.array(f64["static_rank_k"][f"pass_{t}"]["kl64"]["per_board"])
            C = np.array(f64["clean_oracle_dynamic_topk"][f"pass_{t}"]["kl64"]["per_board"])
            O = np.array(f64["online_dynamic_topk"][f"pass_{t}"]["kl64"]["per_board"])
            F = np.array(f64["svd_all_on_identity_floor"][f"pass_{t}"]["kl64"]["per_board"])
            R = np.array(f64["same_dtype_target_rerun"][f"pass_{t}"]["kl64"]["per_board"])
            floor_closed = bool(F.max() <= CLOSURE_TOL and R.max() <= CLOSURE_TOL)
            interp = floor_closed and identity_closed[ck][f"pass_{t}"]
            D = paired(O - C)
            ms = margin_stat(S, C, O)
            flips = {
                a: f64[a][f"pass_{t}"]["argmax_flips_per_board"]
                for a in ("svd_all_on_identity_floor", "static_rank_k", "clean_oracle_dynamic_topk",
                          "online_dynamic_topk", "online_vs_oracle_rollout")
            }
            above_floor = {
                a: int((arr > np.maximum(F, CLOSURE_TOL)).sum()) for a, arr in (("static", S), ("oracle", C), ("online", O))
            }
            res["readouts"][f"pass_{t}"] = {
                "static_kl": S.tolist(), "oracle_kl": C.tolist(), "online_kl": O.tolist(),
                "identity_floor_kl": F.tolist(), "target_repeat_kl": R.tolist(),
                "means": {"static": float(S.mean()), "oracle": float(C.mean()), "online": float(O.mean()),
                          "floor": float(F.mean())},
                "online_minus_oracle": D,
                "reserved64_margin_statistic_descriptive": ms,
                "argmax_flips_per_board": flips,
                "boards_above_identity_floor": above_floor,
                "identity_floor_closed": floor_closed,
                "interpretable_for_mechanism_classification": interp,
            }
            md.append(
                f"| {t} | {fmt(R.max())} | {fmt(F.mean())} / {fmt(F.max())} | {fmt(S.mean())} | {fmt(C.mean())}"
                f" | {fmt(O.mean())} | {fmt(D['mean'])} [{fmt(D['bootstrap_ci95'][0])}, {fmt(D['bootstrap_ci95'][1])}]"
                f" | {D['signflip_two_sided_p']:.4f} | {D['n_negative']} / {D['n_positive']} | {fmt(ms['upper_97_5'])}"
                f" | {'yes' if interp else 'no'} |"
            )
        dis = r["oracle_mask_disagreement"]
        res["mask_disagreement_overall_mean"] = {m: v["overall_mean"] for m, v in dis.items()}
        res["mask_disagreement_by_pass_mean"] = {m: v["by_pass_mean"] for m, v in dis.items()}
        res["component_density"] = r["component_density"]
        md.append("\nOnline vs clean-informed mask disagreement (mean Hamming fraction over boards, tokens, components):\n")
        md.append("| Module | overall | pass 0 | pass 20 | pass 41 | pass 62 |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for m, v in dis.items():
            bp = v["by_pass_mean"]
            md.append(f"| {m} | {v['overall_mean']:.4f} | {bp[0]:.4f} | {bp[20]:.4f} | {bp[41]:.4f} | {bp[62]:.4f} |")
        md.append("\nArgmax cell flips vs clean target at pass 20 / 62 (sum over 16 boards x 81 cells):\n")
        md.append("| Arm | pass 20 | pass 62 |")
        md.append("|---|---:|---:|")
        for a in ("svd_all_on_identity_floor", "static_rank_k", "clean_oracle_dynamic_topk", "online_dynamic_topk"):
            md.append(
                f"| {a} | {sum(res['readouts']['pass_20']['argmax_flips_per_board'][a])}"
                f" | {sum(res['readouts']['pass_62']['argmax_flips_per_board'][a])} |"
            )
        out[ck] = res
    return out, md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("mechinterp/bpd/results/v2")
    ap.add_argument("--controls", type=Path, default=base / "gate_b4_v2_closure_controls_corrected.json")
    ap.add_argument("--mechanism", type=Path, default=base / "gate_b4_v2_closure_mechanism.json")
    ap.add_argument("--online-bf16", type=Path, default=base / "gate_b4_v2_online_svd.json")
    ap.add_argument("--online-fp32", type=Path, default=base / "gate_b4_v2_online_svd_cuda_fp32.json")
    ap.add_argument("--out-json", type=Path, default=base / "bpd01_summary.json")
    ap.add_argument("--out-md", type=Path, default=base / "bpd01_tables.md")
    args = ap.parse_args()

    ctrl = json.loads(args.controls.read_text(encoding="utf-8"))
    mech = json.loads(args.mechanism.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"closure_tolerance_kl": CLOSURE_TOL, "bootstrap_resamples": N_BOOT,
                               "signflip_draws": N_FLIP, "seed": SEED}
    md: list[str] = ["# BPD-01 reduced tables", "",
                     "Generated by `mechinterp/bpd/scripts/reduce_bpd01.py` from the committed JSON receipts.",
                     "KL is token-mean KL(reference || arm) over 81 cells and 9 digits."]
    a_out, a_md = reduce_controls(ctrl, mech)
    summary["experiment_A"] = a_out
    md += ["", "## Experiment A: numerical closure (CUDA BF16, 16 boards, 63 invocations)"] + a_md

    closures = {
        ck: {
            "native_bf16": {t: v["closed"] for t, v in a_out[ck]["mechanism_decisions"]["identity_closure_by_regime"]["native_bf16"].items()},
            "fp32": {t: v["closed"] for t, v in a_out[ck]["mechanism_decisions"]["identity_closure_by_regime"]["fp32_original_fp32_weights"].items()},
        }
        for ck in a_out
    }
    summary["experiment_B"] = {}
    md += ["", "## Experiment B: online vs clean-informed SVD top-k"]
    for label, path, key in (("cuda_bf16_registered", args.online_bf16, "native_bf16"),
                             ("cuda_fp32_original_weights", args.online_fp32, "fp32")):
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        b_out, b_md = reduce_online(rec, label, {ck: closures[ck][key] for ck in closures})
        summary["experiment_B"][label] = b_out
        md += b_md

    args.out_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
