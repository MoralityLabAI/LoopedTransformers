"""B4-FACTOR-03: source-locked residual-floor and spectral-gauge audit.

See docs/B4_FACTOR03_SPECTRAL_REBASE_REGISTRATION_20261118.md. No fitting,
hyperparameter tuning, or evaluation-board access occurs in the geometry stage.
The only full evaluation mode is the registered two-checkpoint CUDA FP32 run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mechinterp.bpd.adapters.trm import (
    TRMBPDAdapter, default_trm_c_per_module, load_pretrained_trm,
    load_wb_stall_and_train_sudoku,
)
from mechinterp.bpd.causal_ci import causal_ci_rollout
from mechinterp.bpd.components import BPDModel, PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.scripts.audit_b4_causal_ci import (
    FACTOR_KEY, ZERO, load_ci,
)
from mechinterp.bpd.scripts.audit_b4_online_svd_reserved64 import BOARD_INDICES as RESERVED64
from mechinterp.bpd.scripts.audit_b4_v2_controls_and_ablation_split import EXPECTED_WEIGHTS_SHA256
from mechinterp.bpd.scripts.b4_numerics import argmax_flips_per_board, kl64_per_board
from mechinterp.bpd.scripts.refit_b4_ci_frozen_factors import validation_indices

REGISTRATION = "mechinterp/bpd/docs/B4_FACTOR03_SPECTRAL_REBASE_REGISTRATION_20261118.md"
REGISTRATION_COMMIT = "80cf55b6a056dcf0925d2aad4466c7d8897de6f2"
EXPERIMENT = "B4-FACTOR-03"
SALT = "B4-FACTOR-03-FRESH48-20261118"
FRESH48 = (
    71, 93, 37, 91, 99, 106, 178, 29, 153, 32, 155, 80,
    104, 98, 175, 96, 182, 160, 30, 66, 81, 158, 173, 169,
    111, 56, 127, 139, 35, 65, 42, 141, 159, 19, 64, 79,
    63, 129, 24, 59, 20, 70, 149, 51, 114, 43, 95, 119,
)
FRESH48_SHA256 = "16a2f3762ebf0e87bff955f34597e4e2cdcc79af87cef166d004a44635b3b954"
MODEL_SHA256 = {
    "gaoxin492": "932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3",
    "yagizdevre": "d67723394a7fe439f1fa498b58196399ee89cd82a794c7da9ce0a8cc3df7fa39",
}
CI_SHA256 = {
    "gaoxin492": "e5675fe056d1f8d50a1b4f3a6a6b52377d69ed8d01bce2c942faa7a201c15f99",
    "yagizdevre": "254e4fca5c1a3707d8af8f003d9bf8cf441a41f539ed630a4628f202c0d321a1",
}
N_PASSES = 63
READOUTS = (20, 41, 62)
C = 512
K = 307
N_BOOT = 20_000
BOOT_SEED = 2026111803
TOL = 1e-5
ARMS = ("F0", "F1", "F2", "F3", "F4", "S0", "S2", "T1", "R1", "T3")


def validate_fresh48(n_available: int) -> tuple[int, ...]:
    """Refuse changed dataset, previous-test reuse, or index/digest drift."""
    if n_available != 183:
        raise AssertionError(f"WB-Stall source changed: {n_available} != 183")
    reserved = set(RESERVED64)
    val = set(validation_indices())
    if len(reserved) != 64 or len(val) != 32:
        raise AssertionError("Historical holdout partition changed")
    pool = [i for i in range(16, 183) if i not in reserved and i not in val]
    pool.sort(key=lambda i: hashlib.sha256(f"{SALT}|{i:03d}".encode("ascii")).digest())
    indices = tuple(pool[:48])
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("ascii")).hexdigest()
    if (len(pool) != 71 or len(indices) != 48 or indices != FRESH48
            or len(set(indices)) != 48 or digest != FRESH48_SHA256):
        raise AssertionError("Fresh-48 registered index selection or hash changed")
    if set(indices) & (reserved | val | set(range(16))):
        raise AssertionError("Fresh-48 overlaps a previous training/validation/eval set")
    return indices


@torch.no_grad()
def rebase_site(w: TiedComponentLinear) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Spectral factorization of frozen (V U).T via a small float64 QR/SVD core.

    If V=Qv Rv and U.T=Qu Ru, then V U=Qv (Rv Ru.T) Qu.T.
    The core SVD yields the full thin SVD of the represented operator without
    SVD on potentially wide activation matrices. No target weights are used.
    """
    v = w.V.detach().double().cpu()
    u = w.U.detach().double().cpu()
    qv, rv = torch.linalg.qr(v, mode="reduced")
    qu, ru = torch.linalg.qr(u.T, mode="reduced")
    left, singular, right_t = torch.linalg.svd(rv @ ru.T, full_matrices=False)
    n = min(w.C, singular.numel())
    root = singular[:n].clamp_min(0).sqrt()
    rebased_v = torch.zeros(v.shape, dtype=torch.float32)
    rebased_u = torch.zeros(u.shape, dtype=torch.float32)
    rebased_v[:, :n] = ((qv @ left[:, :n]) * root.unsqueeze(0)).float()
    rebased_u[:n, :] = (root.unsqueeze(1) * (right_t[:n] @ qu.T)).float()
    original = v @ u
    reproduced = rebased_v.double() @ rebased_u.double()
    relative = float((original - reproduced).norm() / original.norm().clamp_min(1e-30))
    return rebased_v, rebased_u, relative


@torch.no_grad()
def rebase_model(original: BPDModel, dest: BPDModel) -> dict[str, float]:
    """Replace destination (V,U) by spectral coordinates, retaining identical target W."""
    if set(original.wrappers) != set(dest.wrappers):
        raise ValueError("Module sets differ")
    relative: dict[str, float] = {}
    for name in original.module_order:
        v, u, err = rebase_site(original.wrappers[name])
        if err > TOL:
            raise RuntimeError(f"{name}: relative operator error {err} exceeds {TOL}")
        dest.wrappers[name].V.copy_(v.to(dest.wrappers[name].V.device))
        dest.wrappers[name].U.copy_(u.to(dest.wrappers[name].U.device))
        relative[name] = err
    return relative


def _sq(x: torch.Tensor) -> float:
    return float(x.double().square().sum(dim=-1).mean())


@torch.no_grad()
def _site_geometry(w: TiedComponentLinear, x: torch.Tensor, k: int) -> dict[str, float]:
    """Training-board-only residual, omission and coherence diagnostics."""
    x = x.float()
    v, u = w.V.detach().float(), w.U.detach().float()
    acts = x @ v
    score = acts.abs() * u.norm(dim=-1)
    mask = torch.zeros_like(acts).scatter_(-1, score.topk(k, dim=-1).indices, 1.0)
    omitted = ((1.0 - mask) * acts) @ u
    residual = F.linear(x, w.weight_delta().detach())
    target = F.linear(x, w.W_target.float())
    den = max(_sq(target), 1e-30)
    nr, no = _sq(residual), _sq(omitted)
    cross = 2.0 * float((residual.double() * omitted.double()).sum(dim=-1).mean())
    unorm = F.normalize(u, dim=-1, eps=1e-12)
    gram = (unorm @ unorm.T).abs()
    n = int(gram.shape[0])
    offdiag = float((gram.sum() - gram.diag().sum()) / max(1, n * (n - 1)))
    w_hat = (w.V.detach().double().cpu() @ w.U.detach().double().cpu()).T
    wt = w.W_target.detach().double().cpu()
    return {
        "reconstruction_residual_relative_fro": float((wt - w_hat).norm() / wt.norm().clamp_min(1e-30)),
        "residual_rms_over_target": (nr / den) ** 0.5,
        "omitted_rms_over_target": (no / den) ** 0.5,
        "combined_error_rms_over_target": (max(0.0, nr + no + cross) / den) ** 0.5,
        "residual_omission_cross_fraction": cross / max(nr + no, 1e-30),
        "mean_abs_offdiag_write_gram": offdiag,
    }


@torch.no_grad()
def training_geometry(
    trained: BPDModel, rebased: BPDModel, adapter: TRMBPDAdapter,
    train_inputs: torch.Tensor, k: int, n_passes: int,
) -> dict[str, Any]:
    """Read only eight TRAIN boards at five fixed invocations for geometry."""
    selected = [t for t in (0, 10, 20, 40, 60) if t < n_passes]
    with trained.target_capture(pass_offset=0, update_means=False, detach_inputs=True):
        adapter.rollout(train_inputs[:8], n_passes, 0, None)
    recorded = trained.collect_recorded_inputs(expected_passes=n_passes)
    for w in trained.wrappers.values():
        w.recorded_inputs.clear()
    report: dict[str, Any] = {"invocations": selected, "n_training_boards": min(8, len(train_inputs)), "sites": {}}
    for m in trained.module_order:
        x = recorded[m][:, selected].reshape(-1, trained.wrappers[m].d_in)
        report["sites"][m] = {
            "trained": _site_geometry(trained.wrappers[m], x, k),
            "rebased": _site_geometry(rebased.wrappers[m], x, k),
        }
    return report


@torch.no_grad()
def _run(bpd: BPDModel, ad: TRMBPDAdapter, inputs: torch.Tensor,
         n_passes: int, c: int, k: int | None, delta: float = 0.0,
         static: bool = False) -> dict[int, torch.Tensor]:
    dev = inputs.device
    mask = torch.ones((1, n_passes, 1, c), device=dev)
    if static:
        assert k is not None
        mask[..., k:] = 0.0
    dmask = torch.full((1, n_passes, 1), delta, device=dev)
    schedule = {
        m: PassMaskSchedule(comp_mask=mask, delta_mask=dmask,
                            online_topk=(k if k is not None and not static else None))
        for m in bpd.module_order
    }
    with bpd.masked_components(schedule, pass_offset=0, record_inputs=False):
        ro, _, _, _ = ad.rollout(inputs, n_passes, 0, None)
    return ro


@torch.no_grad()
def score_chunk(
    svd: BPDModel, ads: TRMBPDAdapter,
    trained: BPDModel, adt: TRMBPDAdapter,
    rebased: BPDModel, adr: TRMBPDAdapter,
    ci: Any, inputs: torch.Tensor,
    *, n_passes: int = N_PASSES, c: int = C, k: int = K,
    readouts: tuple[int, ...] = READOUTS,
) -> dict[str, Any]:
    """All arms on the same inputs, same source-locked FP32 recurrence."""
    clean, _, _, _ = ads.rollout(inputs, n_passes, 0, None)
    ro = {
        "F0": ads.rollout(inputs, n_passes, 0, None)[0],
        "F1": _run(svd, ads, inputs, n_passes, c, None),
        "F2": _run(trained, adt, inputs, n_passes, c, None, delta=1.0),
        "F3": _run(trained, adt, inputs, n_passes, c, None),
        "F4": _run(rebased, adr, inputs, n_passes, c, None),
        "S0": _run(svd, ads, inputs, n_passes, c, k, static=True),
        "S2": _run(svd, ads, inputs, n_passes, c, k),
        "T1": _run(trained, adt, inputs, n_passes, c, k),
        "R1": _run(rebased, adr, inputs, n_passes, c, k),
        "T3": causal_ci_rollout(trained, adt, ci, inputs, n_passes,
                               observation="preflight_target", selection="topk", k=k)["readouts"],
    }
    arms: dict[str, Any] = {}
    for name, readout in ro.items():
        arms[name] = {
            f"pass_{t}": {
                "kl64": kl64_per_board(readout[t], clean[t]),
                "argmax_flips": argmax_flips_per_board(readout[t], clean[t]),
            }
            for t in readouts
        }
    parity = {f"pass_{t}": kl64_per_board(ro["F4"][t], ro["F3"][t]) for t in readouts}
    return {"arms": arms, "F4_vs_F3_kl64": parity}


def _paired(a: np.ndarray, b: np.ndarray, *, seed: int, n_boot: int) -> dict[str, Any]:
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    ab = np.abs(d)
    return {
        "n": int(len(d)),
        "mean": float(d.mean()), "median": float(np.median(d)),
        "ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
        "n_negative": int((d < 0).sum()), "n_positive": int((d > 0).sum()),
        "n_zero": int((d == 0).sum()),
        "largest_abs_effect_share": float(ab.max() / max(ab.sum(), 1e-30)),
        "effective_effect_n": float(ab.sum() ** 2 / max((ab * ab).sum(), 1e-30)),
        "per_board_difference": d.tolist(),
        "n_bootstrap": n_boot, "seed": seed,
    }


def decide(results: Mapping[str, Any], *, readouts: tuple[int, ...] = READOUTS,
           n_boot: int = N_BOOT) -> dict[str, Any]:
    """Registered H1 and descriptive contrasts, conditional on precision/parity floors."""
    out: dict[str, Any] = {"by_checkpoint": {}, "primary_decision": "pending"}
    for ck, entry in results.items():
        by_pass: dict[str, Any] = {}
        for t in readouts:
            key = f"pass_{t}"
            raw = {a: np.asarray(entry["arms"][a][key]["kl64"], dtype=np.float64) for a in ARMS}
            parity = np.asarray(entry["F4_vs_F3_kl64"][key], dtype=np.float64)
            finite = all(np.isfinite(x).all() for x in list(raw.values()) + [parity])
            operator_ok = all(float(x) <= TOL for x in entry["operator_relative_error"].values())
            floor_max = max(float(np.abs(raw[a]).max()) for a in ("F0", "F1", "F2"))
            parity_max = float(np.abs(parity).max())
            valid = bool(finite and operator_ok and floor_max <= TOL and parity_max <= TOL)
            if not finite:
                by_pass[key] = {"valid": False, "invalid_reason": "nonfinite measurements"}
                continue
            comparison = {
                "H1_R1_minus_T1": _paired(raw["R1"], raw["T1"], seed=BOOT_SEED, n_boot=n_boot),
                "H2_R1_minus_S2_minus_10pct_S0": _paired(
                    raw["R1"] - 0.10 * raw["S0"], raw["S2"], seed=BOOT_SEED, n_boot=n_boot),
                "H3_F3_minus_F1": _paired(raw["F3"], raw["F1"], seed=BOOT_SEED, n_boot=n_boot),
                "H4_R1_minus_T3": _paired(raw["R1"], raw["T3"], seed=BOOT_SEED, n_boot=n_boot),
                "H4_T3_minus_S2": _paired(raw["T3"], raw["S2"], seed=BOOT_SEED, n_boot=n_boot),
            }
            by_pass[key] = {
                "valid": valid, "max_identity_kl": floor_max, "max_rebase_parity_kl": parity_max,
                "max_operator_relative_error": max(entry["operator_relative_error"].values()),
                "mean_kl": {arm: float(x.mean()) for arm, x in raw.items()},
                "comparisons": comparison,
            }
        out["by_checkpoint"][ck] = by_pass
    if len(results) == 2:
        p20 = [out["by_checkpoint"][ck]["pass_20"] for ck in sorted(results)]
        if all(x.get("valid", False) for x in p20):
            h1 = [x["comparisons"]["H1_R1_minus_T1"]["ci95"] for x in p20]
            if all(v[1] < 0 for v in h1):
                out["primary_decision"] = "H1: spectral rebase improves top-307 causal magnitude selection on both"
            elif all(v[0] > 0 for v in h1):
                out["primary_decision"] = "H1: spectral rebase worsens top-307 causal magnitude selection on both"
            else:
                out["primary_decision"] = "H1: inconclusive / checkpoint-discordant"
            h2 = [x["comparisons"]["H2_R1_minus_S2_minus_10pct_S0"]["ci95"] for x in p20]
            out["secondary_H2_noninferior"] = all(v[1] < 0 for v in h2)
        else:
            out["primary_decision"] = "BLOCKED: registered identity or rebase parity floor failed"
            out["secondary_H2_noninferior"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--weights", type=Path, default=Path("mechinterp/bpd/results/v2/bpd_b4_v2_stage4_weights.pt"))
    ap.add_argument("--ci-dir", type=Path, default=Path("mechinterp/bpd/results/v2/ci_closed_loop"))
    ap.add_argument("--out", type=Path, default=Path("mechinterp/bpd/results/v2/gate_b4_factor03_rebase.json"))
    args = ap.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        ap.error("Registered full evaluation requires CUDA FP32. Mini-path tests run in pytest.")
    if args.chunk != 8:
        ap.error("Registered fresh-48 shard size is exactly 8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    factor_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if factor_sha != EXPECTED_WEIGHTS_SHA256:
        ap.error(f"Stage-4 weights SHA mismatch: {factor_sha}")
    factors = torch.load(args.weights, map_location="cpu", weights_only=True)
    wb = load_wb_stall_and_train_sudoku(None, n_train=128, device="cpu")
    indices = validate_fresh48(int(wb["eval_inputs"].shape[0]))
    train = wb["train_inputs"][:128]
    result: dict[str, Any] = {
        "experiment": EXPERIMENT, "registration": REGISTRATION,
        "registration_commit": REGISTRATION_COMMIT,
        "fresh48_indices": list(indices), "fresh48_sha256": FRESH48_SHA256,
        "checkpoint_sha256": MODEL_SHA256, "factor_sha256": factor_sha,
        "ci_sha256": CI_SHA256, "n_passes": N_PASSES, "readouts": READOUTS,
        "component_count": C, "kept_components": K,
        "regime": "CUDA FP32 with BF16-rounded target weights; TF32 off",
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "checkpoints": {}, "decision": None,
    }
    for ck in ("gaoxin492", "yagizdevre"):
        ci_path = args.ci_dir / f"{ck}_seed0_closed_loop_ci.pt"
        if hashlib.sha256(ci_path.read_bytes()).hexdigest() != CI_SHA256[ck]:
            raise ValueError(f"{ck}: frozen closed-loop CI hash mismatch")
        ci = load_ci(ci_path, CI_SHA256[ck], "cuda")
        base = load_pretrained_trm(ck, dtype=torch.bfloat16, verify_sha256=True).float().to("cuda").eval()
        c_map = default_trm_c_per_module(base, c=C)
        svd = BPDModel(copy.deepcopy(base), c_map, max_passes=N_PASSES, ablation=ZERO, init_svd=True).to("cuda")
        trained = BPDModel(copy.deepcopy(base), c_map, max_passes=N_PASSES, ablation=ZERO, init_svd=False).to("cuda")
        rebased = BPDModel(copy.deepcopy(base), c_map, max_passes=N_PASSES, ablation=ZERO, init_svd=False).to("cuda")
        with torch.no_grad():
            for m, w in trained.wrappers.items():
                w.V.copy_(factors[FACTOR_KEY[ck]][m]["V"].to(device="cuda", dtype=torch.float32))
                w.U.copy_(factors[FACTOR_KEY[ck]][m]["U"].to(device="cuda", dtype=torch.float32))
        # Check gauge fidelity before accessing any fresh evaluation-board tensor.
        errors = rebase_model(trained, rebased)
        ads = TRMBPDAdapter(svd.target_model, readout_at="outer_step")
        adt = TRMBPDAdapter(trained.target_model, readout_at="outer_step")
        adr = TRMBPDAdapter(rebased.target_model, readout_at="outer_step")
        geometry = training_geometry(trained, rebased, adt, train[:8].to("cuda"), K, N_PASSES)
        entry: dict[str, Any] = {
            "operator_relative_error": errors, "geometry_training_only": geometry,
            "evaluation_index_hash_prefix16": [
                hashlib.sha256(bytes(map(int, row.tolist()))).hexdigest()[:16]
                for row in wb["eval_inputs"][list(indices)]
            ],
            "arms": {}, "F4_vs_F3_kl64": {},
        }
        chunks = []
        for off in range(0, 48, args.chunk):
            batch = wb["eval_inputs"][[*indices[off:off + args.chunk]]].to("cuda")
            chunks.append(score_chunk(svd, ads, trained, adt, rebased, adr, ci, batch))
            print(f"{ck}: fresh-48 scored {off + args.chunk}/48", flush=True)
        for name in ARMS:
            entry["arms"][name] = {
                f"pass_{t}": {
                    "kl64": sum((x["arms"][name][f"pass_{t}"]["kl64"] for x in chunks), []),
                    "argmax_flips": sum((x["arms"][name][f"pass_{t}"]["argmax_flips"] for x in chunks), []),
                }
                for t in READOUTS
            }
        entry["F4_vs_F3_kl64"] = {
            f"pass_{t}": sum((x["F4_vs_F3_kl64"][f"pass_{t}"] for x in chunks), [])
            for t in READOUTS
        }
        result["checkpoints"][ck] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        del base, svd, trained, rebased, ads, adt, adr, ci
        torch.cuda.empty_cache()
    result["decision"] = decide(result["checkpoints"])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
