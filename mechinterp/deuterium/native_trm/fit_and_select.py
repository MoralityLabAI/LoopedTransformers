"""DEUTERIUM-NATIVE-01 Stage B: fit stock VPD to a native weight-tied TRM.

This program only uses original frozen training seed 31041 plus the newly
registered calibration seed 731055 AFTER verifying a separately uploaded
source lock. The sealed heldout seed 731057 is never generated in this module.
Actual four-pass recurrent BLOCK weights, not output-head weights, are edited.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mechinterp.loopscope04.gradient_flow import (
    TinyRecursiveModel, fixed_data, tensor_int64_sha
)
from mechinterp.loopscope05.fit_and_freeze import load_frozen, tensor_fingerprint
from mechinterp.deuterium.native_trm.preflight import (
    MODEL_SHA, META_SHA, sha256, four_pass_manual
)
from mechinterp.deuterium.native_trm.stageb_lock import verify_lock, UPSTREAM_SHA

SCHEMA = "DEUTERIUM-NATIVE-01/first-calibration-and-selection-v1"
TEACHERS = (103, 211, 307)
TARGET = "block.chan_mix.2"
WEIGHT_NAME = TARGET + ".weight"
GOALS = (0, 10, 20, 30, 40, 50, 60, 70)
COMPONENTS = 16
SOURCE_TRAIN_SEED = 31041
SOURCE_TRAIN_N = 512
CALIBRATION_SEED = 731055
CALIBRATION_N = 256
RESERVED_TEST_SEED = 731057
TRAIN_BATCH = 8
SCORE_BATCH = 32
SMOKE_SEED = 61073
OLD_SEED_COUNTS = {
    31041: 512, 31042: 128, 31043: 256, 31044: 256,
    31048: 768, 31049: 192, 31050: 256, 31052: 256,
    31053: 64, 31054: 256, 31055: 256, 31056: 64,
    31057: 256, 31058: 256, 31060: 64, 31061: 64,
    31062: 64, 31063: 64, 31064: 64,
}
# The 31059 reservation is explicitly excluded from any generator call.
# Additional 31065..70 data in previous studies require an explicit source
# lineage preflight before production; source disjointness is conservative.
MAX_REPLAY_ERROR = 1e-4
MAX_REL_DOSE_ERROR = 1e-4


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def config(seed: int, smoke=False):
    from param_decomp.configs import Config
    steps = 1 if smoke else 64
    return Config.model_validate({
        "wandb_project": None, "wandb_run_name": None,
        "seed": seed, "autocast_bf16": False,
        "n_mask_samples": 1,
        "ci_config": {"mode": "layerwise", "fn_type": "mlp", "hidden_dims": [16]},
        "sigmoid_type": "leaky_hard",
        "module_info": [{"module_pattern": TARGET, "C": COMPONENTS}],
        "identity_module_info": None,
        "use_delta_component": True,
        "loss_metric_configs": [
            {"classname": "FaithfulnessLoss", "coeff": 1.0},
            {"classname": "ImportanceMinimalityLoss", "coeff": 0.001,
             "pnorm": 1.0, "beta": 0},
            {"classname": "PGDReconLoss", "coeff": 1.0, "init": "random",
             "step_size": 0.25, "n_steps": 2,
             "mask_scope": "unique_per_datapoint"},
        ],
        "lr_schedule": {"start_val": 0.001, "fn_type": "constant", "warmup_pct": 0.0},
        "steps": steps, "batch_size": TRAIN_BATCH, "eval_batch_size": TRAIN_BATCH,
        "faithfulness_warmup_steps": 0 if smoke else 32,
        "faithfulness_warmup_lr": 0.001,
        "faithfulness_warmup_weight_decay": 0.1,
        "train_log_freq": 1 if smoke else 16,
        "eval_freq": steps, "n_eval_steps": 1,
        "slow_eval_freq": steps * 2,
        "slow_eval_on_first_step": False,
        "save_freq": steps,
        "ci_alive_threshold": 0.1,
        "eval_metric_configs": [],
        "pretrained_model_class": "mechinterp.loopscope04.gradient_flow.TinyRecursiveModel",
        "pretrained_model_path": "local:original-LGA02-frozen-44k-TRM",
        "task_config": {"task_name": "resid_mlp", "feature_probability": 0.01,
                        "data_generation_type": "at_least_zero_active"},
    })


def train_vpd(model, train_x, seed, out_dir, smoke=False):
    from param_decomp.models.batch_and_loss_fns import (
        recon_loss_kl, run_batch_first_element
    )
    from param_decomp.run_param_decomp import optimize
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config(seed, smoke)
    write_json(out_dir / "config.json", cfg.model_dump(mode="json"))
    model.eval().requires_grad_(False)
    train = DataLoader(
        TensorDataset(train_x), batch_size=TRAIN_BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(911000 + seed)
    )
    internal_eval = DataLoader(TensorDataset(train_x), batch_size=TRAIN_BATCH, shuffle=False)
    t0 = time.perf_counter()
    optimize(target_model=model, config=cfg, device="cpu",
             train_loader=train, eval_loader=internal_eval,
             run_batch=run_batch_first_element,
             reconstruction_loss=recon_loss_kl, out_dir=out_dir)
    path = out_dir / ("model_1.pth" if smoke else "model_64.pth")
    if not path.is_file():
        raise RuntimeError("stock param_decomp did not save its expected terminal checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    stem = "_components.block-chan_mix-2"
    u, v = state.get(stem + ".U"), state.get(stem + ".V")
    if u is None or v is None:
        raise RuntimeError("BLOCKED_RECURRENT_TARGET_ADAPTER: absent native recurrent target factors")
    if tuple(u.shape) != (COMPONENTS, 48) or tuple(v.shape) != (128, COMPONENTS):
        raise RuntimeError("BLOCKED_RECURRENT_TARGET_ADAPTER: learned factor dimensions changed")
    if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):
        raise RuntimeError("nonfinite VPD recurrent components")
    return state, u.detach(), v.detach(), {
        "upstream_commit": UPSTREAM_SHA,
        "implementation": "param_decomp.run_param_decomp.optimize",
        "terminal_step": cfg.steps, "actual_optimizer_main_updates": cfg.steps,
        "faithfulness_warmup_updates": cfg.faithfulness_warmup_steps,
        "source_training_n": len(train_x),
        "runtime_seconds": time.perf_counter() - t0,
        "checkpoint_sha256": sha256(path),
        "factor_shapes": [list(u.shape), list(v.shape)],
        "original_target_tensor_sha256": tensor_fingerprint(
            {WEIGHT_NAME: model.block.chan_mix[2].weight.detach().clone()}),
    }


def normalize_rank_one(matrix, dose):
    norm = float(matrix.norm())
    if not math.isfinite(norm) or norm <= 1e-10 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("invalid rank-one steering direction")
    return (matrix / norm * dose).detach().contiguous()


def derived_seed(*values):
    raw = ("DEUTERIUM-NATIVE-01|" + "|".join(map(str, values))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def banks(u, v, weight, seed):
    dose = 0.01 * float(weight.norm())
    learned = [normalize_rank_one(torch.outer(u[i], v[:, i]), dose)
               for i in range(COMPONENTS)]
    rng = torch.Generator().manual_seed(derived_seed("random_bank", seed))
    random_bank = [
        normalize_rank_one(torch.outer(
            torch.randn(48, generator=rng),
            torch.randn(128, generator=rng)), dose)
        for _ in range(COMPONENTS)
    ]
    return {"vpd": learned, "random": random_bank}, dose


def task_gradient(model, x, y, dose):
    """Actual four-pass native gradient through all four tied uses of W."""
    param = model.block.chan_mix[2].weight
    prev = param.requires_grad
    param.requires_grad_(True)
    grad = torch.zeros_like(param)
    try:
        for i in range(0, len(x), SCORE_BATCH):
            xx, yy = x[i:i+SCORE_BATCH], y[i:i+SCORE_BATCH]
            logits = model(xx)[:, GOALS]
            loss = F.cross_entropy(logits.reshape(-1,10),
                                   yy[:, GOALS].reshape(-1))
            (g,) = torch.autograd.grad(loss, param)
            grad.add_(g.detach(), alpha=len(xx) / len(x))
    finally:
        param.requires_grad_(prev)
    left, _, right = torch.linalg.svd(grad, full_matrices=False)
    return normalize_rank_one(-torch.outer(left[:,0], right[0,:]), dose)


@contextlib.contextmanager
def patched_recurrent_weight(model, delta):
    param = model.block.chan_mix[2].weight
    original = param.detach().clone()
    actual = None
    try:
        with torch.no_grad():
            param.copy_(original + delta)
            actual = param.detach().clone() - original
        yield actual
    finally:
        with torch.no_grad():
            param.copy_(original)
        if not torch.equal(param.detach(), original):
            raise RuntimeError("recurrent weight restoration failed")


def score(model, x, y):
    """Per-independent-source native goal/offgoal CE and accuracy."""
    result = {name: [] for name in ("goal_nll", "offgoal_nll", "full_nll",
                                   "goal_accuracy", "offgoal_accuracy", "full_accuracy")}
    keep = torch.ones(81, dtype=torch.bool)
    keep[list(GOALS)] = False
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), SCORE_BATCH):
            xx, yy = x[i:i+SCORE_BATCH], y[i:i+SCORE_BATCH]
            logits = model(xx)
            if tuple(logits.shape) != (len(xx), 81, 10):
                raise RuntimeError("expected original 81-token ten-class native logits")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite native logits")
            nll = F.cross_entropy(logits.reshape(-1,10), yy.reshape(-1),
                                  reduction="none").reshape(len(xx), 81)
            correct = logits.argmax(-1).eq(yy).float()
            for name, val in (
                ("goal_nll", nll[:, GOALS].mean(-1)),
                ("offgoal_nll", nll[:, keep].mean(-1)),
                ("full_nll", nll.mean(-1)),
                ("goal_accuracy", correct[:, GOALS].mean(-1)),
                ("offgoal_accuracy", correct[:, keep].mean(-1)),
                ("full_accuracy", correct.mean(-1))
            ):
                result[name].extend(float(x) for x in val)
    return result


def score_edit(model, x, y, delta, dose, validate=True):
    with torch.no_grad():
        clean_weight = model.block.chan_mix[2].weight.detach().clone()
    with patched_recurrent_weight(model, delta) as actual:
        actual_relative_error = abs(float(actual.norm()) - dose) / max(dose, 1e-20)
        if validate and actual_relative_error > MAX_REL_DOSE_ERROR:
            raise RuntimeError("actual FP32 recurrent edit dose did not match")
        output = score(model, x, y)
        if validate:
            with torch.no_grad():
                native = model(x[:2])
                replay = four_pass_manual(model, x[:2])
                max_error = float((native-replay).abs().max())
            if max_error > MAX_REPLAY_ERROR:
                raise RuntimeError("patched native four-pass forward parity failed")
        else:
            max_error = None
    with torch.no_grad():
        restored = model(x[:2])
        clean = four_pass_manual(model, x[:2])
        restoration_error = float((restored - clean).abs().max())
    if restoration_error > MAX_REPLAY_ERROR or not torch.equal(
            model.block.chan_mix[2].weight.detach(), clean_weight):
        raise RuntimeError("full original recurrent weight was not restored")
    return output, {
        "actual_applied_frobenius": float(actual.norm()),
        "actual_relative_dose_error": actual_relative_error,
        "max_native_manual_replay_error": max_error,
        "weight_restore_max_logit_error": restoration_error,
    }


def cohort():
    x,y = fixed_data(CALIBRATION_N, CALIBRATION_SEED)
    train_x,_ = fixed_data(SOURCE_TRAIN_N, SOURCE_TRAIN_SEED)
    seen = {tuple(row) for row in train_x.tolist()}
    for sd, n in OLD_SEED_COUNTS.items():
        prior,_ = fixed_data(n,sd)
        seen.update(map(tuple,prior.tolist()))
    rows = list(map(tuple,x.tolist()))
    if len(rows) != len(set(rows)) or any(row in seen for row in rows):
        raise RuntimeError("new calibration cohort overlaps trained or exposed old source")
    return x,y, {
        "seed": CALIBRATION_SEED, "n": len(x),
        "input_sha256": tensor_int64_sha(x),
        "target_sha256": tensor_int64_sha(y),
        "no_exact_sequence_overlap": True,
        "old_source_roster": OLD_SEED_COUNTS,
    }


def run_calibration(seed, teacher_bundle, teacher_meta, source_lock, upstream, out):
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA, "teacher_seed": seed, "status": "SOURCE_VERIFICATION",
        "new_calibration_exposed": False, "heldout_seed": RESERVED_TEST_SEED,
        "heldout_generated": False, "heldout_labels_read": False,
        "scoring_completed": False, "selected_before_heldout": False,
        "teacher_bundle_sha256": MODEL_SHA, "teacher_meta_sha256": META_SHA,
        "torch": torch.__version__, "python": platform.python_version(),
        "git_branch_sha": os.environ.get("GITHUB_SHA"),
    }
    write_json(root/"progress.json", record)
    verify_lock(source_lock, teacher_bundle, teacher_meta, upstream)
    meta,bundle = load_frozen(teacher_bundle,teacher_meta)
    if seed not in TEACHERS or meta["teacher_init_seeds"] != list(TEACHERS):
        raise RuntimeError("unexpected original root selection")
    model = TinyRecursiveModel(48,4)
    model.load_state_dict(bundle["models"][str(seed)], strict=True)
    model.eval().requires_grad_(False)
    weight = model.block.chan_mix[2].weight.detach().clone()
    # Original training source only: available to VPD optimizer and gradient oracle.
    train_x, train_y = fixed_data(SOURCE_TRAIN_N, SOURCE_TRAIN_SEED)
    record["training"] = {
        "input_sha256": tensor_int64_sha(train_x),
        "target_sha256": tensor_int64_sha(train_y),
        "n": len(train_x),
        "original_teacher_state_sha256": meta["teacher_receipts"][str(seed)]["model_state_sha256"],
    }
    with torch.no_grad():
        record["training"]["clean_accuracy"] = float(
            (model(train_x).argmax(-1) == train_y).float().mean())
    write_json(root/"progress.json", record)
    state,u,v,optim_info = train_vpd(model, train_x, seed, root/"vpd")
    if not torch.equal(model.block.chan_mix[2].weight,weight):
        raise RuntimeError("VPD optimizer altered original teacher weights")
    learned_matrix = u.T @ v.T
    residual = weight - learned_matrix
    with torch.no_grad():
        original_output = model(train_x[:16])
        model.block.chan_mix[2].weight.copy_(learned_matrix + residual)
        restored_output = model(train_x[:16])
        model.block.chan_mix[2].weight.copy_(weight)
        model.block.chan_mix[2].weight.copy_(learned_matrix)
        comp_only_output = model(train_x[:16])
        model.block.chan_mix[2].weight.copy_(weight)
    restore_error = float((original_output-restored_output).abs().max())
    if restore_error > MAX_REPLAY_ERROR:
        raise RuntimeError("VPD component plus residual native four-pass replay failed")
    components_only_kl = float(F.kl_div(
        F.log_softmax(comp_only_output,-1),
        F.softmax(original_output,-1),reduction="none").sum(-1).mean())
    optim_info.update({
        "full_residual_replay_max_logit_error": restore_error,
        "components_only_per_token_kl": components_only_kl,
        "components_only_argmax_agreement": float(
            (comp_only_output.argmax(-1)==original_output.argmax(-1)).float().mean()),
        "components_only_reconstruction_relative_residual": float(
            residual.norm()/weight.norm().clamp_min(1e-12))
    })
    record["vpd"] = optim_info
    write_json(root/"progress.json",record)
    bank,dose = banks(u,v,weight,seed)
    gradient = task_gradient(model, train_x, train_y, dose)
    # Freeze VPD learned and random candidate banks before calibration generation.
    torch.save({"vpd":torch.stack(bank["vpd"]),
                "random":torch.stack(bank["random"]),
                "task_gradient": gradient},root/"candidate_bank.pt")
    record["candidate_bank_sha256"] = sha256(root/"candidate_bank.pt")
    record["dose"] = dose
    record["stage"] = "ALL_DIRECTIONS_TRAIN_ONLY_FROZEN"
    write_json(root/"progress.json",record)
    # The seed is considered irrevocably exposed from this marker, including on failure.
    write_json(root/"calibration_exposure.json", {
        "seed": CALIBRATION_SEED, "status": "GENERATION_STARTED",
        "teacher_seed": seed, "heldout_seed_reserved": RESERVED_TEST_SEED,
        "heldout_generated": False, "candidate_bank_sha256": record["candidate_bank_sha256"],
    })
    record["new_calibration_exposed"] = True
    write_json(root/"progress.json",record)
    x,y,roster = cohort()
    record["calibration"] = roster
    # Save literal old/public calibration x for an independent pure-stdlib auditor.
    write_json(root/"calibration_sources.json", {
        "input": x.tolist(), "target": y.tolist(),
        "seed": CALIBRATION_SEED, **roster
    })
    with torch.no_grad():
        clean_scores = score(model,x,y)
    record["calibration_clean"] = {
        "mean_full_accuracy": float(np.mean(clean_scores["full_accuracy"])),
        "mean_goal_nll": float(np.mean(clean_scores["goal_nll"])),
        "mean_offgoal_accuracy":float(np.mean(clean_scores["offgoal_accuracy"])),
    }
    searches = {}
    selected = {"task_gradient": gradient}
    for name in ("vpd","random"):
        candidates = []
        for i, direction in enumerate(bank[name]):
            for sign in (1,-1):   # registered tie break: plus before minus
                metrics, applied = score_edit(model,x,y,sign*direction,dose)
                candidates.append({
                    "candidate_index":i, "sign":sign,
                    "candidate_matrix_sha256": hashlib.sha256(
                        (sign*direction).numpy().astype("<f4").tobytes()).hexdigest(),
                    "mean_goal_nll":float(np.mean(metrics["goal_nll"])),
                    "per_sequence_goal_nll": metrics["goal_nll"],
                    "mean_offgoal_accuracy":float(np.mean(metrics["offgoal_accuracy"])),
                    "applied":applied,
                })
        winner = min(range(len(candidates)),
                     key=lambda index:(candidates[index]["mean_goal_nll"],index))
        best=candidates[winner]
        selected[name] = best["sign"]*bank[name][best["candidate_index"]]
        searches[name] = {"budget":len(candidates),"sign_order":[1,-1],
                          "selection":"minimum_mean_goal_nll_on_calibration",
                          "best_index":winner,
                          "best_candidate_index":best["candidate_index"],
                          "best_sign":best["sign"],
                          "best_calibration_goal_nll":best["mean_goal_nll"],
                          "all_candidates": candidates}
        record["stage"]="CALIBRATION_"+name.upper()+"_SELECTED"
        write_json(root/"progress.json",record)
    selected["sham"] = torch.zeros_like(weight)
    torch.save(selected,root/"selected_edits.pt")
    record["selected_edits_sha256"]=sha256(root/"selected_edits.pt")
    record["searches"] = {name: {key:value for key,value in search.items()
                               if key!="all_candidates"}
                          for name,search in searches.items()}
    record["task_gradient"] = {
        "source_label_supervised": True, "dose":float(gradient.norm()),
        "matrix_sha256":hashlib.sha256(
            gradient.numpy().astype("<f4").tobytes()).hexdigest(),
    }
    record["runtime_seconds"] = time.perf_counter()-t0
    record["selected_before_heldout"] = True
    record["status"] = "CALIBRATION_SELECTION_FROZEN"
    record["new_calibration_exposed"] = True
    write_json(root/"calibration_searches.json",{
        "schema": SCHEMA,
        "teacher_seed":seed,
        "calibration_source_sha256":roster["input_sha256"],
        "selected_edits_sha256":record["selected_edits_sha256"],
        "searches":searches,
    })
    write_json(root/"seed_result.json",record)
    write_json(root/"progress.json",record)
    return record


def adapter_smoke(out):
    """Untrained, protected-source-excluded smoke; exercises stock optimizer."""
    torch.set_num_threads(2)
    torch.manual_seed(SMOKE_SEED)
    m=TinyRecursiveModel(48,4).eval().requires_grad_(False)
    x,_ = fixed_data(8,SMOKE_SEED)
    state,u,v,info = train_vpd(m,x,SMOKE_SEED,Path(out)/"vpd",smoke=True)
    if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):
        raise RuntimeError("nonfinite tied recurrent VPD smoke")
    result={"status":"ADAPTER_SMOKE_PASS","schema":SCHEMA,
            "trained_model_used":False,
            "source_seed":SMOKE_SEED,"calibration_731055_generated":False,
            "heldout_731057_generated":False,
            "optimizer":info}
    write_json(Path(out)/"smoke.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("smoke","calibrate"),required=True)
    p.add_argument("--teacher-seed",type=int)
    p.add_argument("--teacher-bundle",type=Path)
    p.add_argument("--teacher-meta",type=Path)
    p.add_argument("--source-lock",type=Path)
    p.add_argument("--upstream",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.phase=="smoke":
            result=adapter_smoke(a.out)
        else:
            if any(x is None for x in (
                    a.teacher_seed,a.teacher_bundle,a.teacher_meta,
                    a.source_lock,a.upstream)):
                p.error("first calibration requires frozen source lock and original teachers")
            result=run_calibration(a.teacher_seed,a.teacher_bundle,a.teacher_meta,
                                   a.source_lock,a.upstream,a.out)
    except Exception as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        write_json(Path(a.out)/"failure.json",{
            "schema":SCHEMA,"status":"ENGINEERING_OR_PREREQUISITE_FAILURE",
            "calibration_exposed":(Path(a.out)/"calibration_exposure.json").exists(),
            "heldout_generated":False,"heldout_labels_read":False,
            "exception":str(exc),"type":type(exc).__name__,
            "traceback":traceback.format_exc(),
        })
        raise
    print("DEUTERIUM_NATIVE_STAGEB",result["status"],flush=True)


if __name__=="__main__":
    main()
