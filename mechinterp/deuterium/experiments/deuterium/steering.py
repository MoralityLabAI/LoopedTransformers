"""Recovered-checkpoint prerequisites and a separately named recurrent toy study.

The pilot never constructs heldout examples. The score phase requires a committed
protocol and source. Historical forward code is not inferred from tensor names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def environment() -> dict:
    names = ["torch", "numpy", "pydantic", "transformers", "datasets", "wandb",
             "wandb-workspaces", "jaxtyping", "einops", "matplotlib", "scipy"]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    try:
        commit = git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": cpu, "cpu_count": os.cpu_count(), "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(), "packages": packages,
            "git_commit": commit, "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}


def unique_suffix(root: Path, suffix: str) -> Path:
    found = [p for p in root.rglob("*") if p.is_file() and p.as_posix().endswith(suffix)]
    if len(found) != 1:
        raise ValueError(f"Expected one {suffix}; found {len(found)}")
    return found[0]


def recovered_preflight(root: Path, protocol: dict, out: Path) -> dict:
    start = time.perf_counter()
    cfg = protocol["recovered_lane"]
    base_path = unique_suffix(root, cfg["model_suffix"])
    learned_path = unique_suffix(root, cfg["decomposition_suffix"])
    for path, expected in [(base_path, cfg["model_sha256"]), (learned_path, cfg["decomposition_sha256"])]:
        if digest(path) != expected:
            raise ValueError(f"Recovered checkpoint hash mismatch: {path.relative_to(root)}")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    learned = torch.load(learned_path, map_location="cpu", weights_only=True)
    state = base["state_dict"]
    matches = {name: bool(torch.equal(tensor, learned.get("target_model." + name)))
               for name, tensor in state.items()}
    modules = []
    for key in sorted(learned):
        if not key.startswith("_components.") or not key.endswith(".U"):
            continue
        stem = key[:-2]
        u, v = learned[key], learned[stem + ".V"]
        name = stem.removeprefix("_components.").replace("-", ".")
        weight = state[name + ".weight"]
        reconstruction = u.T @ v.T
        modules.append({"module": name, "components": int(u.shape[0]),
                        "weight_shape": list(weight.shape),
                        "component_sum_relative_norm": float(reconstruction.norm() / weight.norm()),
                        "residual_relative_norm": float((weight-reconstruction).norm() / weight.norm()),
                        "all_finite": bool(torch.isfinite(u).all() and torch.isfinite(v).all()),
                        "note": "Matrix audit only; no task forward pass was evaluated."})
    missing = [cfg["missing_forward_class"], cfg["missing_dataset_class"],
               "independent original forward logits for replay validation",
               "historical x-transformers/package versions"]
    result = {"schema_version": 1, "protocol_id": protocol["protocol_id"],
              "lane": "recovered_powered_int3", "status": "BLOCKED_ARCHITECTURE_PROVENANCE",
              "comparison_completed": False, "task_performance_evaluated": False,
              "heldout_evaluated": False, "missing_prerequisites": missing,
              "base_checkpoint": str(base_path.relative_to(root)),
              "base_checkpoint_sha256": digest(base_path),
              "decomposition_checkpoint": str(learned_path.relative_to(root)),
              "decomposition_checkpoint_sha256": digest(learned_path),
              "base_parameters": sum(t.numel() for t in state.values()),
              "base_config": base["config"], "frozen_tensor_count": len(matches),
              "all_frozen_tensors_equal": all(matches.values()),
              "mismatched_frozen_tensors": [n for n, ok in matches.items() if not ok],
              "modules": modules, "runtime_seconds": time.perf_counter()-start,
              "checkpoint_hashes_verified": True,
              "source_search": {"outer_archive_members": 406,
                                "nested_export_member_counts": [51, 1015],
                                "nested_python_source_counts": [0, 48],
                                "original_class_definitions_recovered": False,
                                "public_hrm_source_commit_inspected": "ea756068ea9277e4d1c489fec3e06d98401a3a79"},
              "environment": environment(),
              "limitation": "Exact target tensors establish weight lineage. Missing forward/tokenizer provenance prevents clean-task and intervention/replay gates. Configured reasoning_steps=1 cannot demonstrate multi-loop steering."}
    if not result["all_frozen_tensors_equal"]:
        result["status"] = "BLOCKED_CHECKPOINT_LINEAGE"
    write_json(out / "recovered_preflight.json", result)
    return result


def toy_data(cfg: dict, seed: int, split: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    d = cfg["data"]
    n = d[split + "_rows"]
    generator = torch.Generator().manual_seed(seed + d["split_offsets"][split])
    x = torch.rand(n, cfg["input_dim"], generator=generator) * 2 - 1
    quality = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]
    risk = x[:, 3] + 0.5 * x[:, 4]
    y = torch.where(risk > 0.35, 1, torch.where(quality > 0.1, 0, 2)).long()
    h = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    return x, y, h


def rank_one_normalize(matrix: torch.Tensor, dose: float) -> torch.Tensor:
    norm = matrix.norm()
    if not torch.isfinite(matrix).all() or float(norm) <= 1e-12:
        raise ValueError("Nonfinite or zero intervention direction")
    return matrix / norm * dose


def gradient_direction(hidden: torch.Tensor, labels: torch.Tensor,
                       weight: torch.Tensor, bias: torch.Tensor, dose: float) -> torch.Tensor:
    w = weight.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(F.linear(hidden[labels == 1], w, bias), labels[labels == 1])
    grad, = torch.autograd.grad(loss, w)
    u, s, vh = torch.linalg.svd(grad, full_matrices=False)
    return rank_one_normalize(-s[0] * torch.outer(u[:, 0], vh[0]), dose)


def select_direction(candidates: list[torch.Tensor], hidden: torch.Tensor,
                     labels: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     signs: list[int]) -> tuple[torch.Tensor, dict]:
    target = labels == 1
    scored = []
    for index, candidate in enumerate(candidates):
        for sign in signs:
            logits = F.linear(hidden[target], weight + sign * candidate, bias)
            scored.append((float(F.cross_entropy(logits, labels[target])), index, sign))
    loss, index, sign = min(scored)
    return sign * candidates[index], {"candidate_index": index, "sign": sign,
                                      "calibration_target_nll": loss,
                                      "candidate_scores": [{"nll": l, "index": i, "sign": s}
                                                           for l, i, s in scored]}


@contextmanager
def edited_weight(model, delta: torch.Tensor):
    original = model.head.weight.detach().clone()
    try:
        with torch.no_grad():
            model.head.weight.copy_(original + delta)
        yield
    finally:
        with torch.no_grad():
            model.head.weight.copy_(original)


def score_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    pred = logits.argmax(-1)
    target = labels == 1
    collateral = ~target
    loss = F.cross_entropy(logits, labels, reduction="none")
    return {"accuracy": float((pred == labels).float().mean()),
            "target_accuracy": float((pred[target] == labels[target]).float().mean()),
            "target_nll": float(loss[target].mean()),
            "collateral_accuracy": float((pred[collateral] == labels[collateral]).float().mean()),
            "false_commit_rate_on_veto": float((pred[target] == 0).float().mean()),
            "counts": torch.bincount(labels, minlength=3).tolist()}


def paired_interval(values: np.ndarray, seed: int, repetitions: int) -> dict:
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for i in range(repetitions):
        boot[i] = values[rng.integers(0, len(values), len(values))].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci95": [float(lo), float(hi)],
            "n": len(values), "method": "paired-row percentile bootstrap"}


def registration_check(commit: str | None, protocol_path: Path) -> dict:
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Scoring requires --registration-commit with a full committed SHA")
    git("cat-file", "-e", commit + "^{commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Scoring refused: tracked worktree has uncommitted changes")
    paths = ["experiments/deuterium/steering.py", "experiments/deuterium/protocol.json",
             "experiments/deuterium/requirements-steering.txt", "param_decomp"]
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "experiments/deuterium", "param_decomp")
    untracked += "\n" + git("ls-files", "--others", "--ignored", "--exclude-standard", "--", "experiments/deuterium", "param_decomp")
    if any(line.endswith(".py") for line in untracked.splitlines()):
        raise ValueError("Scoring refused: untracked Python source could shadow frozen imports")
    for path in paths[:-1]:
        git("cat-file", "-e", f"{commit}:{path}")
    if git("diff", "--name-only", commit, "HEAD", "--", *paths):
        raise ValueError("Scoring refused: experiment/source differ from registration commit")
    relative = protocol_path.resolve().relative_to(REPO).as_posix()
    frozen = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{relative}"])
    if hashlib.sha256(frozen).hexdigest() != digest(protocol_path):
        raise ValueError("Scoring refused: protocol differs from committed bytes")
    return {"registration_commit": commit, "protocol_sha256": digest(protocol_path),
            "source_commit": git("rev-parse", "HEAD")}


def build_vpd_config(cfg: dict, seed: int):
    from param_decomp.configs import Config
    d = cfg["decomposition"]
    return Config.model_validate({
        "wandb_project": None, "wandb_run_name": None, "seed": seed,
        "n_mask_samples": 1, "autocast_bf16": False,
        "ci_config": {"mode": "layerwise", "fn_type": "mlp", "hidden_dims": [16]},
        "sigmoid_type": "leaky_hard", "module_info": [{"module_pattern": "head", "C": d["components"]}],
        "identity_module_info": None, "use_delta_component": d["use_delta_component"],
        "loss_metric_configs": [
            {"classname": "FaithfulnessLoss", "coeff": d["faithfulness_coefficient"]},
            {"classname": "ImportanceMinimalityLoss", "coeff": d["importance_minimality_coefficient"], "pnorm": 1.0, "beta": 0},
            {"classname": "PGDReconLoss", "coeff": 1.0, "init": "random", "step_size": d["pgd_step_size"],
             "n_steps": d["pgd_steps"], "mask_scope": "unique_per_datapoint"}],
        "batch_size": d["batch_size"], "eval_batch_size": d["batch_size"], "steps": d["steps"],
        "lr_schedule": {"start_val": d["learning_rate"], "fn_type": "constant", "warmup_pct": 0.0},
        "faithfulness_warmup_steps": d["warmup_steps"], "faithfulness_warmup_lr": d["warmup_learning_rate"],
        "faithfulness_warmup_weight_decay": 0.1, "train_log_freq": 50,
        "eval_freq": d["steps"], "n_eval_steps": 1, "slow_eval_freq": d["steps"] * 2,
        "slow_eval_on_first_step": False, "save_freq": d["steps"], "ci_alive_threshold": 0.1,
        "eval_metric_configs": [],
        "pretrained_model_class": cfg["model_class"], "pretrained_model_path": "local:deuterium_new_toy_gate",
        "task_config": {"task_name": "resid_mlp", "feature_probability": 0.01, "data_generation_type": "at_least_zero_active"}})


def run_seed(cfg: dict, seed: int, phase: str, out: Path) -> dict:
    from param_decomp.experiments.metta_amalgam.model import RecursiveGateTRM
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_first_element
    from param_decomp.run_param_decomp import optimize
    from torch.utils.data import DataLoader, TensorDataset

    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y, train_hash = toy_data(cfg, seed, "train")
    xc, yc, calibration_hash = toy_data(cfg, seed, "calibration")
    model = RecursiveGateTRM(cfg["input_dim"], cfg["hidden_dim"], 3, cfg["recursive_steps"])
    train = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
    rng = torch.Generator().manual_seed(seed + 40000)
    trace = []
    for step in range(train["steps"]):
        idx = torch.randint(len(x), (train["batch_size"],), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == train["steps"] - 1:
            trace.append({"step": step + 1, "train_batch_nll": float(loss.detach())})
    model.eval().requires_grad_(False)
    with torch.no_grad():
        h, hc = model.hidden(x), model.hidden(xc)
        clean_cal = model(xc)
    clean_metrics = score_logits(clean_cal, yc)
    result = {"seed": seed, "phase": phase, "status": "PREPARING", "heldout_evaluated": False,
              "comparison_completed": False, "train_sha256": train_hash,
              "calibration_sha256": calibration_hash, "model_parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter()-start, "calibration_clean": clean_metrics,
              "failed_gates": [], "training_trace": trace}
    if clean_metrics["accuracy"] < cfg["gates"]["min_calibration_accuracy"]:
        result["failed_gates"].append("calibration_clean_accuracy")
    if min(clean_metrics["counts"]) < cfg["gates"]["min_examples_per_class_per_scored_split"]:
        result["failed_gates"].append("calibration_class_counts")
    torch.save(model.state_dict(), out / "base_model.pt")
    result["stage"] = "clean_calibration_completed"
    write_json(out / "progress.json", result)
    if result["failed_gates"]:
        result.update(status="BLOCKED_CLEAN_TASK", runtime_seconds=time.perf_counter()-start)
        write_json(out / "seed_result.json", result)
        return result

    dstart = time.perf_counter()
    config = build_vpd_config(cfg, seed)
    write_json(out / "vpd_config.json", config.model_dump(mode="json"))
    loader = DataLoader(TensorDataset(x), batch_size=cfg["decomposition"]["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed + 50000))
    # Decomposition's internal eval also uses train rows, never test or calibration.
    eval_loader = DataLoader(TensorDataset(x), batch_size=cfg["decomposition"]["batch_size"], shuffle=False)
    ddir = out / "vpd"
    ddir.mkdir()
    optimize(target_model=model, config=config, device="cpu", train_loader=loader,
             eval_loader=eval_loader, run_batch=run_batch_first_element,
             reconstruction_loss=recon_loss_kl, out_dir=ddir)
    checkpoint_path = ddir / f"model_{cfg['decomposition']['steps']}.pth"
    learned = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    u, v = learned["_components.head.U"], learned["_components.head.V"]
    weight, bias = model.head.weight.detach(), model.head.bias.detach()
    component_weight = u.T @ v.T
    component_bias = learned.get("_components.head.bias", bias)
    residual = weight - component_weight
    with torch.no_grad():
        restored = F.linear(hc, component_weight + residual, component_bias)
        component_logits = F.linear(hc, component_weight, component_bias)
    restore_error = float((restored-clean_cal).abs().max())
    result["decomposition"] = {"checkpoint_sha256": digest(checkpoint_path),
        "runtime_seconds": time.perf_counter()-dstart, "residual_relative_norm": float(residual.norm()/weight.norm()),
        "components_only_kl": float(F.kl_div(F.log_softmax(component_logits, -1), F.softmax(clean_cal, -1), reduction="batchmean")),
        "components_only_argmax_agreement": float((component_logits.argmax(-1) == clean_cal.argmax(-1)).float().mean()),
        "residual_restored_max_logit_error": restore_error,
        "method": "stock optimize: PGDReconLoss + ImportanceMinimalityLoss + FaithfulnessLoss"}
    dose = cfg["intervention"]["frobenius_dose_fraction_of_base_weight"] * float(weight.norm())
    learned_bank = [rank_one_normalize(torch.outer(u[i], v[:, i]), dose) for i in range(len(u))]
    gen = torch.Generator().manual_seed(seed + 60000)
    random_bank = [rank_one_normalize(torch.outer(torch.randn(weight.shape[0], generator=gen),
                                                  torch.randn(weight.shape[1], generator=gen)), dose)
                   for _ in range(cfg["intervention"]["random_candidates"])]
    learned_delta, learned_selection = select_direction(learned_bank, hc, yc, weight, bias, cfg["intervention"]["signs"])
    random_delta, random_selection = select_direction(random_bank, hc, yc, weight, bias, cfg["intervention"]["signs"])
    gradient_delta = gradient_direction(h, y, weight, bias, dose)
    deltas = {"sham": torch.zeros_like(weight), "vpd": learned_delta, "random": random_delta, "task_gradient": gradient_delta}
    result["selection"] = {"vpd": learned_selection, "random": random_selection}
    result["edits"] = {}
    replay_tolerance = cfg["gates"]["max_replay_absolute_logit_error"]
    if restore_error > replay_tolerance:
        result["failed_gates"].append("residual_restored_replay")
    for name, delta in deltas.items():
        with torch.no_grad(), edited_weight(model, delta):
            native = model(xc)
        cached = F.linear(hc, weight + delta, bias)
        error = float((native-cached).abs().max())
        rank = int(torch.linalg.matrix_rank(delta))
        norm = float(delta.norm())
        applied = (weight + delta) - weight
        applied_norm = float(applied.norm())
        rel = abs(norm-dose)/dose if name != "sham" else 0.0
        applied_rel = abs(applied_norm-dose)/dose if name != "sham" else 0.0
        result["edits"][name] = {"rank": rank, "frobenius_norm": norm,
            "applied_frobenius_norm": applied_norm, "applied_dose_relative_error": applied_rel,
            "applied_minus_intended_max_absolute": float((applied-delta).abs().max()),
            "expected_dose": 0.0 if name == "sham" else dose,
            "dose_relative_error": rel, "native_cached_max_error": error,
            "calibration": score_logits(native, yc)}
        if error > replay_tolerance or (name != "sham" and rank != 1):
            result["failed_gates"].append(name + "_intervention_replay_or_rank")
        if max(rel, applied_rel) > cfg["gates"]["max_dose_relative_error"]:
            result["failed_gates"].append(name + "_dose")
    with torch.no_grad():
        restored_native = model(xc)
    if float((restored_native-clean_cal).abs().max()) > replay_tolerance:
        result["failed_gates"].append("native_weight_restoration")
    torch.save(deltas, out / "selected_edits.pt")
    result["selected_edits_sha256"] = digest(out / "selected_edits.pt")
    result["stage"] = "directions_selected_and_replay_checked"
    write_json(out / "progress.json", result)
    if result["failed_gates"]:
        result.update(status="BLOCKED_REPLAY", runtime_seconds=time.perf_counter()-start)
        write_json(out / "seed_result.json", result)
        return result
    # No heldout generator is called in pilot mode, including after a failed gate.
    if phase == "pilot":
        result.update(status="PILOT_CALIBRATION_ONLY", runtime_seconds=time.perf_counter()-start)
        write_json(out / "seed_result.json", result)
        return result

    # Persist exposure before even constructing test features or labels. Exceptions
    # and timeouts after this marker can never be reported as an untouched holdout.
    write_json(out / "heldout_exposure.json", {"seed": seed, "status": "GENERATION_STARTED",
                                              "instruction": "Treat this seed's heldout as exposed after any failure."})
    xt, yt, test_hash = toy_data(cfg, seed, "heldout")
    with torch.no_grad():
        ht, clean = model.hidden(xt), model(xt)
    clean_test = score_logits(clean, yt)
    result.update(heldout_evaluated=True, heldout_exposed=True, heldout_sha256=test_hash, heldout_clean=clean_test)
    write_json(out / "progress.json", result)
    if clean_test["accuracy"] < cfg["gates"]["min_heldout_clean_accuracy"]:
        result["failed_gates"].append("heldout_clean_accuracy")
    if min(clean_test["counts"]) < cfg["gates"]["min_examples_per_class_per_scored_split"]:
        result["failed_gates"].append("heldout_class_counts")
    if result["failed_gates"]:
        result.update(status="BLOCKED_HELDOUT_CLEAN_TASK", runtime_seconds=time.perf_counter()-start)
        write_json(out / "seed_result.json", result)
        return result
    logits_by_arm = {"clean": clean}
    for name, delta in deltas.items():
        with torch.no_grad(), edited_weight(model, delta):
            logits_by_arm[name] = model(xt)
        if float((logits_by_arm[name] - F.linear(ht, weight+delta, bias)).abs().max()) > replay_tolerance:
            result["failed_gates"].append(name + "_heldout_replay")
    metrics = {name: score_logits(logits, yt) for name, logits in logits_by_arm.items()}
    losses = {name: F.cross_entropy(logits, yt, reduction="none").numpy() for name, logits in logits_by_arm.items()}
    target = (yt == 1).numpy()
    effects = {}
    for arm in ["random", "task_gradient", "clean"]:
        effects[arm + "_minus_vpd_target_nll"] = paired_interval(
            (losses[arm]-losses["vpd"])[target], seed + cfg["endpoints"]["bootstrap_seed"], cfg["endpoints"]["bootstrap_resamples"])
    rows_path = out / "heldout_rows.jsonl"
    with rows_path.open("w") as f:
        for i in range(len(yt)):
            row = {"seed": seed, "example": i, "label": int(yt[i]),
                   "input_sha256": hashlib.sha256(xt[i].numpy().tobytes()).hexdigest(),
                   "arms": {name: {"logits": logits[i].tolist(), "prediction": int(logits[i].argmax()),
                                   "nll": float(losses[name][i])} for name, logits in logits_by_arm.items()}}
            f.write(json.dumps(row, allow_nan=False) + "\n")
    result.update(status="COMPLETED" if not result["failed_gates"] else "FAILED_HELDOUT_REPLAY",
                  comparison_completed=not result["failed_gates"], metrics=metrics, effects=effects,
                  collateral_accuracy_change=metrics["vpd"]["collateral_accuracy"]-metrics["clean"]["collateral_accuracy"],
                  heldout_rows_sha256=digest(rows_path), runtime_seconds=time.perf_counter()-start)
    write_json(out / "seed_result.json", result)
    return result


def aggregate(cfg: dict, results: list[dict], phase: str) -> dict:
    if len(cfg["seeds"]) != 3 or len(set(cfg["seeds"])) != 3:
        raise ValueError("This protocol's frozen Student t analysis requires three unique seeds")
    complete = len(results) == len(cfg["seeds"]) and all(r["comparison_completed"] for r in results)
    exposures = [r.get("heldout_exposed", r.get("heldout_evaluated", False)) for r in results]
    evaluations = [r.get("heldout_evaluated", False) for r in results]
    evaluated = True if any(evaluations) else (None if any(x is None for x in evaluations) else False)
    initial_status = "BLOCKED_PREREQUISITE"
    if any(r.get("status") == "ENGINEERING_FAILURE" for r in results):
        initial_status = "ENGINEERING_FAILURE"
    elif phase == "pilot":
        initial_status = "PILOT_ONLY" if all(r.get("status") == "PILOT_CALIBRATION_ONLY" for r in results) else "PILOT_PREREQUISITE_FAILURE"
    result = {"comparison_completed": complete, "heldout_evaluated": evaluated, "heldout_exposed": any(exposures),
              "status": initial_status,
              "seed_results": results}
    if not complete:
        return result
    values = np.array([r["effects"]["random_minus_vpd_target_nll"]["estimate"] for r in results])
    mean = float(values.mean())
    # Frozen n=3 design: t_(.975,2), not a row-level interval over pseudo-replicates.
    half = 4.302652729911275 * float(values.std(ddof=1)) / np.sqrt(3)
    collateral_ok = all(r["collateral_accuracy_change"] >= -cfg["gates"]["max_collateral_accuracy_drop"] for r in results)
    positive = bool(np.all(values > 0) and mean-half > 0 and collateral_ok)
    result.update(status="POSITIVE" if positive else "NO_REGISTERED_ADVANTAGE", primary_mean=mean,
                  primary_seed_effects=values.tolist(), primary_ci95=[mean-half, mean+half],
                  interval_method="Student t over three independent model/decomposition seeds, df=2",
                  collateral_gate_passed=collateral_ok)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p = sub.add_parser("toy")
    p.add_argument("--phase", choices=["pilot", "score"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--registration-commit")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cfg = protocol["new_toy_lane"]
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    if args.command == "preflight":
        result = recovered_preflight(args.records, protocol, args.out)
        result["protocol_sha256"] = digest(args.protocol)
        write_json(args.out / "recovered_preflight.json", result)
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("Output directory must be empty; retain earlier attempts and choose a new path")
        receipt = registration_check(args.registration_commit, args.protocol) if args.phase == "score" else {
            "registration_commit": None, "protocol_sha256": digest(args.protocol), "pilot_only": True}
        seeds = [cfg["pilot_seed"]] if args.phase == "pilot" else cfg["seeds"]
        results = []
        for seed in seeds:
            started = time.perf_counter()
            def budget_expired(signum, frame):
                raise TimeoutError("Registered per-seed wall-time budget on CPU exhausted")
            old = signal.signal(signal.SIGALRM, budget_expired)
            signal.alarm(cfg["execution"]["per_seed_budget_seconds"])
            try:
                seed_result = run_seed(cfg, seed, args.phase, args.out / f"seed_{seed}")
            except Exception as exc:
                progress = args.out / f"seed_{seed}" / "progress.json"
                seed_result = json.loads(progress.read_text()) if progress.exists() else {}
                exposed = (args.out / f"seed_{seed}" / "heldout_exposure.json").exists()
                seed_result.update(seed=seed, phase=args.phase, status="ENGINEERING_FAILURE",
                                   heldout_exposed=exposed,
                                   heldout_evaluated=(True if seed_result.get("heldout_evaluated") is True else (None if exposed else False)),
                                   comparison_completed=False,
                                   exception=type(exc).__name__ + ": " + str(exc),
                                   traceback=traceback.format_exc(), runtime_seconds=time.perf_counter()-started)
                write_json(args.out / f"seed_{seed}" / "seed_result.json", seed_result)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            results.append(seed_result)
        result = aggregate(cfg, results, args.phase)
        result.update(protocol_id=protocol["protocol_id"], lane=cfg["label"],
                      phase=args.phase, registration=receipt, environment=environment())
        write_json(args.out / "steering_summary.json", result)
    print(json.dumps({k: result[k] for k in ["status", "comparison_completed", "heldout_evaluated"]}))
    if args.command == "toy" and any(r["status"] == "ENGINEERING_FAILURE" for r in result["seed_results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
