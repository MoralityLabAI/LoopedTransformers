"""Frozen CPU runner for the public LRM-REV-02 engineering pilot.

Importing this module creates no models or source episodes. The executable
requires an exact clean source commit and the explicit engineering flag.
Its evidence verdict remains ENGINEERING_PILOT_ONLY for every outcome.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import torch
from torch import nn

from lrm_rev02_fixture import (
    CONTROL_COUNT, INPUT_DIM, J, N, control_slots, encode_prefix, encode_stream,
    initial_posterior, make_pilot_episode, model_seed, training_slots,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REGISTRATION_PATH = HERE / "REGISTRATION_LRM_REV02.json"
VERDICT = "ENGINEERING_PILOT_ONLY"
PREFLIGHT_STATUS = "PENDING_MATCHED_SHAM_AND_CONFIRMATION_DESIGN"
PROBABILITY_TOLERANCE = 1e-10


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def gzip_jsonl(path: Path):
    """Close partial archives on every exit and fix the gzip timestamp."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed:
            with io.TextIOWrapper(packed, encoding="utf-8", newline="\n") as text:
                yield text


def load_registration(path: Path = REGISTRATION_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "experiment_id": "LRM-REV-02",
        "roots": [0, 1, 2, 3],
        "evidence_law": {"n": 12, "j": 6, "priors": [[9, 20], [7, 20], [1, 5]],
                         "match_probabilities": [[19, 20], [13, 20], [3, 20]]},
        "model": {"class": "GRU", "input_size": 17, "hidden_size": 32, "num_layers": 1,
                  "output_size": 3, "parameter_count": 4995,
                  "device": "cpu", "dtype": "float32"},
        "training": {"steps": 800, "batch_size": 64,
                     "episodes_per_root": 2048, "optimizer": "Adam",
                     "learning_rate": 0.003,
                     "loss_weights": {"initial": 0.25, "corrected": 0.25,
                                      "reminded": 0.5},
                     "corrected_slot_policy_probability": 0.5,
                     "checkpoint_selection": "fixed_final_step", "development_selection": False},
        "development": {"episodes_per_root": 512, "controls": 4,
                        "reveal_stages": list(range(7)), "primary_stage": 6,
                        "inference_batch_size": 1},
        "budget": {"wall_seconds": 900, "rss_bytes": 2147483648,
                   "torch_threads": 1, "gpu_allocation": False, "automatic_retries": False},
        "endpoint": {"loss": "squared_L2_to_full_history_Bayes",
                     "units": "unhalved_three_class_Brier", "scientific_min_effect": 0.01,
                     "planned_primary_comparisons": 1},
        "diagnostics": {"initial_excess_max": 0.01,
                        "candidate_excess_max": 0.02, "replay_abs_max": 1e-6,
                        "checkpoint_replay_probability_abs_max": 1e-6,
                        "checkpoint_replay_effect_abs_max": 1e-6},
        "verdict": VERDICT, "preflight_status": PREFLIGHT_STATUS,
        "confirmation_population_allocated": False,
    }

    def require(actual, wanted, location):
        if isinstance(wanted, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"registration object required: {location}")
            for key, value in wanted.items():
                if key not in actual:
                    raise ValueError(f"missing registration field: {location}.{key}")
                require(actual[key], value, f"{location}.{key}")
        elif isinstance(wanted, list):
            if not isinstance(actual, list) or len(actual) != len(wanted):
                raise ValueError(f"registration array differs: {location}")
            for index, value in enumerate(wanted):
                require(actual[index], value, f"{location}[{index}]")
        elif type(actual) is not type(wanted) or actual != wanted:
            raise ValueError(f"registration differs from frozen runner: {location}")

    require(config, expected, "registration")
    files = config.get("source_files")
    if (not isinstance(files, list) or not files
            or any(type(item) is not str or not item for item in files)
            or len(files) != len(set(files))):
        raise ValueError("a unique nonempty frozen source_files list is required")
    required = {str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                str(REGISTRATION_PATH.relative_to(REPO_ROOT)),
                str((HERE / "lrm_rev02_fixture.py").relative_to(REPO_ROOT))}
    if not required <= set(files):
        raise ValueError("runner, fixture and registration must be source-hashed")
    return config


def verify_source(source_commit: str, source_files: list[str],
                  repo: Path = REPO_ROOT) -> dict:
    """Require the declared commit and compare every frozen byte with Git."""
    if (type(source_commit) is not str or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("a complete lowercase source commit SHA is required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip()
    if head != source_commit:
        raise ValueError("working HEAD differs from the declared source commit")
    tree = subprocess.check_output(["git", "rev-parse", f"{source_commit}^{{tree}}"],
                                   cwd=repo, text=True).strip()
    hashes = {}
    for relative in source_files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise ValueError("frozen source paths must be normalized repository paths")
        local = repo / path
        if local.is_symlink() or not local.is_file():
            raise ValueError(f"frozen source is missing or symlinked: {relative}")
        expected = subprocess.check_output(["git", "show", f"{source_commit}:{relative}"],
                                           cwd=repo)
        observed = local.read_bytes()
        if observed != expected:
            raise ValueError(f"frozen source differs from commit: {relative}")
        hashes[relative] = hashlib.sha256(observed).hexdigest()
    return {"schema_version": 1, "source_commit": source_commit,
            "source_tree_sha": tree, "source_hashes": hashes}


def configure_cpu(threads: int = 1) -> None:
    if type(threads) is not int or threads != 1:
        raise ValueError("this runner requires exactly one CPU thread")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def sampler_seed(root: int) -> int:
    if type(root) is not int or root not in range(4):
        raise ValueError("registered model root required")
    key = canonical_json(["lrm-rev02/public-cpu-pilot/v1", "training-minibatches", root])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


class RevisionGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, 32, num_layers=1, batch_first=True,
                          dropout=0.0, device="cpu", dtype=torch.float32)
        self.head = nn.Linear(32, 3, device="cpu", dtype=torch.float32)

    def forward(self, inputs, hidden=None):
        states, reached = self.gru(inputs, hidden)
        return self.head(states), reached


def training_loss(logits, initial_targets, full_targets, weights):
    """Only the initial target supervises time 12; full targets start at 18."""
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (24, 3):
        raise ValueError("a batch of twenty-four three-class outputs is required")
    probabilities = logits.softmax(dim=-1)
    initial = ((probabilities[:, 11] - initial_targets) ** 2).sum(-1).mean()
    corrected = ((probabilities[:, 17] - full_targets) ** 2).sum(-1).mean()
    reminded = ((probabilities[:, 23] - full_targets) ** 2).sum(-1).mean()
    total = (weights["initial"] * initial + weights["corrected"] * corrected
             + weights["reminded"] * reminded)
    return total, {"initial": initial, "corrected": corrected, "reminded": reminded}


def posterior_from_logits(logits) -> list[float]:
    if tuple(logits.shape) != (3,) or not torch.isfinite(logits).all().item():
        raise ArithmeticError("nonfinite or malformed forecast logits")
    probability = logits.detach().to(dtype=torch.float64).softmax(-1).tolist()
    if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probability)
            or abs(math.fsum(probability) - 1) > PROBABILITY_TOLERANCE):
        raise ArithmeticError("invalid normalized posterior")
    return probability


def excess(probability: list[float], target: list[float]) -> float:
    return math.fsum((p - q) ** 2 for p, q in zip(probability, target))


def _score_episode(model: RevisionGRU, episode, root: int, episode_index: int,
                   controls: int, context: dict) -> dict:
    """Score one public history at the registered inference batch size one."""
    if controls != CONTROL_COUNT:
        raise ValueError("all four active controls must be retained")
    context.update(phase="prefix_encoding", stage=0)
    model.eval()
    prefix = torch.tensor([encode_prefix(episode)], dtype=torch.float32, device="cpu")
    target_initial = list(map(float, initial_posterior(sum(episode.old))))
    target_full = list(map(float, episode.full_reference()))
    bayes_risk = 1 - math.fsum(q * q for q in target_full)
    with torch.no_grad():
        context.update(phase="prefix_replay", stage=N + J)
        full_logits, reached = model(prefix)
        old_logits, old_state = model(prefix[:, :N])
        corrected_logits, staged_reached = model(prefix[:, N:], old_state)
        initial_probability = posterior_from_logits(old_logits[0, -1])
        base_probability = posterior_from_logits(full_logits[0, -1])
        replay_probability = posterior_from_logits(corrected_logits[0, -1])
        replay_probability_error = max(abs(a - b) for a, b in zip(base_probability, replay_probability))
        replay_logits = float((full_logits[:, -1] - corrected_logits[:, -1]).abs().max())
        replay_hidden = float((reached - staged_reached).abs().max())
        arms = {}
        policies = {"corrected": tuple(episode.correction_slots)}
        policies.update({f"random_{b}": control_slots(episode, b) for b in range(controls)})
        for name, slots in policies.items():
            context.update(phase="reminder", arm=name, stage=0)
            hidden = reached.clone()
            first_excess = excess(base_probability, target_full)
            curve = [{"r": 0, "probability": list(base_probability),
                      "excess": first_excess, "expected_brier": first_excess + bayes_risk,
                      "hidden_displacement_l2": 0.0}]
            reminders = encode_stream(episode, slots)[N + J:]
            for stage, token in enumerate(reminders, start=1):
                context["stage"] = stage
                inputs = torch.tensor([[token]], dtype=torch.float32, device="cpu")
                logits, hidden = model(inputs, hidden)
                probability = posterior_from_logits(logits[0, -1])
                loss = excess(probability, target_full)
                displacement = float(torch.linalg.vector_norm(hidden.double() - reached.double()))
                if not math.isfinite(displacement):
                    raise ArithmeticError("nonfinite hidden-state displacement")
                curve.append({"r": stage, "probability": probability, "excess": loss,
                              "expected_brier": loss + bayes_risk,
                              "hidden_displacement_l2": displacement})
            arms[name] = {"slots": list(sorted(slots)), "curve": curve}
    paired = (math.fsum(arms[f"random_{b}"]["curve"][-1]["excess"] for b in range(controls))
              / controls - arms["corrected"]["curve"][-1]["excess"])
    return {"schema_version": 1, "root": root, "episode_index": episode_index,
            "source_id": episode.source_id, "old": list(episode.old),
            "correction_slots": list(episode.correction_slots),
            "replacements": list(episode.replacements), "true_regime": episode.true_regime,
            "target_initial": target_initial, "target_full": target_full,
            "initial_probability": initial_probability,
            "initial_excess": excess(initial_probability, target_initial),
            "prefix_replay_probability": replay_probability,
            "prefix_replay_probability_abs_max": replay_probability_error,
            "prefix_replay_logits_abs_max": replay_logits,
            "prefix_replay_hidden_abs_max": replay_hidden,
            "arms": arms, "paired_effect": paired}


class EpisodeScoreFailure(RuntimeError):
    def __init__(self, message: str, context: dict):
        super().__init__(message)
        self.context = dict(context)


def score_episode(model: RevisionGRU, episode, root: int, episode_index: int,
                  controls: int = CONTROL_COUNT) -> dict:
    context = {"phase": "score_entry", "stage": 0}
    try:
        return _score_episode(model, episode, root, episode_index, controls, context)
    except Exception as error:
        raise EpisodeScoreFailure(f"{type(error).__name__}: {error}", context) from error


def summarize_rows(rows: list[dict], diagnostics: dict) -> tuple[dict, dict, dict]:
    if not rows:
        return {}, {}, {}
    mean = lambda values: math.fsum(values) / len(rows)
    means = {
        "initial_excess": mean(row["initial_excess"] for row in rows),
        "base_excess": mean(row["arms"]["corrected"]["curve"][0]["excess"] for row in rows),
        "candidate_excess": mean(row["arms"]["corrected"]["curve"][-1]["excess"] for row in rows),
        "control_mean_excess": mean(math.fsum(row["arms"][f"random_{b}"]["curve"][-1]["excess"]
                                               for b in range(CONTROL_COUNT)) / CONTROL_COUNT
                                    for row in rows),
        "paired_effect": mean(row["paired_effect"] for row in rows),
        "bayes_risk": mean(1 - math.fsum(q * q for q in row["target_full"]) for row in rows),
        "candidate_expected_brier": mean(row["arms"]["corrected"]["curve"][-1]["expected_brier"] for row in rows),
        "control_mean_expected_brier": mean(math.fsum(row["arms"][f"random_{b}"]["curve"][-1]["expected_brier"]
                                                      for b in range(CONTROL_COUNT)) / CONTROL_COUNT
                                           for row in rows),
    }
    curves = {}
    for arm in rows[0]["arms"]:
        curves[arm] = [{"r": r, **{key: mean(row["arms"][arm]["curve"][r][key] for row in rows)
                                    for key in ("excess", "expected_brier", "hidden_displacement_l2")}}
                       for r in range(J + 1)]
    probability_error = max(abs(math.fsum(row["initial_probability"]) - 1) for row in rows)
    r0_error = 0.0
    for row in rows:
        probability_error = max(probability_error,
                                abs(math.fsum(row["prefix_replay_probability"]) - 1))
        base = row["arms"]["corrected"]["curve"][0]["probability"]
        for arm in row["arms"].values():
            for point in arm["curve"]:
                probability_error = max(probability_error, abs(math.fsum(point["probability"]) - 1))
            r0_error = max(r0_error, max(abs(a - b) for a, b in zip(base, arm["curve"][0]["probability"])))
    logit_error = max(row["prefix_replay_logits_abs_max"] for row in rows)
    hidden_error = max(row["prefix_replay_hidden_abs_max"] for row in rows)
    forecast_error = max(row["prefix_replay_probability_abs_max"] for row in rows)
    numerical = (probability_error <= PROBABILITY_TOLERANCE
                 and forecast_error <= diagnostics["replay_abs_max"]
                 and logit_error <= diagnostics["replay_abs_max"]
                 and hidden_error <= diagnostics["replay_abs_max"] and r0_error <= 1e-12)
    initial_pass = means["initial_excess"] <= diagnostics["initial_excess_max"]
    candidate_pass = means["candidate_excess"] <= diagnostics["candidate_excess_max"]
    checks = {"probability_abs_sum_error_max": probability_error,
              "prefix_replay_probability_abs_max": forecast_error,
              "prefix_replay_logits_abs_max": logit_error,
              "prefix_replay_hidden_abs_max": hidden_error,
              "r0_abs_difference_max": r0_error, "initial_competence": initial_pass,
              "candidate_competence": candidate_pass, "numeric_pass": numerical,
              "engineering_competence_pass": numerical and initial_pass and candidate_pass}
    return means, curves, checks


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, wall_seconds: float, rss_bytes: int):
        self.started = time.perf_counter()
        self.wall_seconds = wall_seconds
        self.rss_bytes = rss_bytes

    @property
    def elapsed(self):
        return time.perf_counter() - self.started

    def check(self):
        if self.elapsed >= self.wall_seconds:
            raise BudgetStop("registered global wall-clock cap reached")
        if peak_rss_bytes() > self.rss_bytes:
            raise BudgetStop("registered peak resident-memory cap reached")


def weights_tensor_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous()
        digest.update(canonical_json([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def _source_record(episode, root: int, index: int):
    return {"root": root, "episode_index": index, "source_id": episode.source_id,
            "old": list(episode.old), "correction_slots": list(episode.correction_slots),
            "replacements": list(episode.replacements), "true_regime": episode.true_regime}


def run_root(root: int, config: dict, source: dict, out: Path,
             budget: Budget, seen_sources: set[str]) -> dict:
    """Collect one root; preserve complete and interrupted artifacts alike."""
    directory = out / f"root_{root}"
    directory.mkdir()
    started = time.perf_counter()
    summary = {"schema_version": 1, "root": root, "status": "STARTED",
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_seed": model_seed(root), "sampler_seed": sampler_seed(root),
               "train_steps_completed": 0, "n_train_episodes": 0,
               "development_count": 0, "means": {}, "curve_means": {}, "diagnostics": {}}
    model = None
    rows = []
    final_checkpoint = False
    context = {"phase": "initialization", "root": root}
    try:
        budget.check()
        torch.manual_seed(summary["model_seed"])
        model = RevisionGRU()
        summary["model_parameters"] = sum(p.numel() for p in model.parameters())
        if summary["model_parameters"] != config["model"]["parameter_count"]:
            raise ValueError("model parameter count differs from registration")
        inputs, old_targets, full_targets, training_records = [], [], [], []
        train_path = directory / "training_episodes.jsonl.gz"
        with gzip_jsonl(train_path) as handle:
            for index in range(config["training"]["episodes_per_root"]):
                context = {"phase": "training_source", "root": root, "episode_index": index}
                budget.check()
                episode = make_pilot_episode(root, index, "train")
                context.update(_source_record(episode, root, index))
                if episode.source_id in seen_sources:
                    raise ValueError("source identifier collision across model roots or phases")
                seen_sources.add(episode.source_id)
                slots = training_slots(episode)
                initial = list(map(float, initial_posterior(sum(episode.old))))
                full = list(map(float, episode.full_reference()))
                inputs.append(encode_stream(episode, slots))
                old_targets.append(initial)
                full_targets.append(full)
                record = {**_source_record(episode, root, index), "reminder_slots": list(slots),
                          "target_initial": initial, "target_full": full}
                training_records.append(record)
                handle.write(canonical_json(record) + "\n")
                summary["n_train_episodes"] += 1
        tensors = (torch.tensor(inputs, dtype=torch.float32, device="cpu"),
                   torch.tensor(old_targets, dtype=torch.float32, device="cpu"),
                   torch.tensor(full_targets, dtype=torch.float32, device="cpu"))
        del inputs, old_targets, full_targets
        summary["training_archive_sha256"] = sha256_file(train_path)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                     weight_decay=0.0)
        generator = torch.Generator(device="cpu").manual_seed(summary["sampler_seed"])
        with (directory / "training_trace.jsonl").open("w", encoding="utf-8") as trace:
            for step in range(config["training"]["steps"]):
                context = {"phase": "training_step", "root": root, "step": step + 1}
                budget.check()
                selected = torch.randint(len(tensors[0]), (config["training"]["batch_size"],),
                                         generator=generator, device="cpu")
                context["batch_sources"] = [training_records[i] for i in selected.tolist()]
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(tensors[0][selected])
                loss, components = training_loss(logits, tensors[1][selected], tensors[2][selected],
                                                 config["training"]["loss_weights"])
                if not torch.isfinite(loss).item():
                    raise ArithmeticError("nonfinite training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                       for p in model.parameters()):
                    raise ArithmeticError("nonfinite training gradient")
                optimizer.step()
                summary["train_steps_completed"] = step + 1
                trace.write(canonical_json({"step": step + 1, "total_loss": float(loss.detach()),
                                           **{name + "_loss": float(value.detach())
                                              for name, value in components.items()}}) + "\n")
                if (step + 1) % 100 == 0:
                    trace.flush()
                    print(f"root {root}: step {step + 1}/{config['training']['steps']}, loss {float(loss.detach()):.6f}",
                          flush=True)
        del optimizer, tensors, training_records
        context = {"phase": "checkpoint_final", "root": root,
                   "step": summary["train_steps_completed"]}
        checkpoint = directory / "checkpoint_final.pt"
        torch.save({"schema_version": 1, "root": root, "source_commit": source["source_commit"],
                    "source_hashes": source["source_hashes"], "steps": summary["train_steps_completed"],
                    "model_state_dict": model.state_dict()}, checkpoint)
        summary["checkpoint_file"] = str(checkpoint.relative_to(out))
        summary["checkpoint_sha256"] = sha256_file(checkpoint)
        summary["weights_tensor_sha256"] = weights_tensor_sha256(model)
        final_checkpoint = True
        outcome_path = directory / "development_outcomes.jsonl.gz"
        summary["development_outcomes_file"] = str(outcome_path.relative_to(out))
        with gzip_jsonl(outcome_path) as handle:
            for index in range(config["development"]["episodes_per_root"]):
                context = {"phase": "development", "root": root, "episode_index": index}
                budget.check()
                episode = make_pilot_episode(root, index, "development")
                context.update(_source_record(episode, root, index))
                if episode.source_id in seen_sources:
                    raise ValueError("source identifier collision across model roots or phases")
                seen_sources.add(episode.source_id)
                row = score_episode(model, episode, root, index, config["development"]["controls"])
                handle.write(canonical_json(row) + "\n")
                rows.append(row)
                summary["development_count"] += 1
                if (index + 1) % 128 == 0:
                    handle.flush()
                    print(f"root {root}: scored {index + 1}/{config['development']['episodes_per_root']} episodes",
                          flush=True)
        context = {"phase": "development_complete", "root": root,
                   "episode_index": summary["development_count"]}
        budget.check()
        summary["status"] = "COMPLETE"
    except BudgetStop as error:
        summary["status"] = "BUDGET_STOP"
        summary["error"] = str(error)
        summary["failure_context"] = dict(context)
    except Exception as error:
        summary["status"] = "FAILED"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["failure_context"] = dict(context)
        if isinstance(error, EpisodeScoreFailure):
            summary["failure_context"]["score_context"] = error.context
        if context["phase"] == "development" and "source_id" in context:
            write_json(directory / "failed_episode.json", summary["failure_context"])
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if model is not None and not final_checkpoint:
            checkpoint = directory / "checkpoint_partial.pt"
            torch.save({"schema_version": 1, "root": root, "source_commit": source["source_commit"],
                        "source_hashes": source["source_hashes"], "steps": summary["train_steps_completed"],
                        "model_state_dict": model.state_dict()}, checkpoint)
            summary["checkpoint_file"] = str(checkpoint.relative_to(out))
            summary["checkpoint_sha256"] = sha256_file(checkpoint)
            summary["weights_tensor_sha256"] = weights_tensor_sha256(model)
        means, curves, diagnostics = summarize_rows(rows, config["diagnostics"])
        summary.update(means=means, curve_means=curves, diagnostics=diagnostics)
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["peak_rss_bytes"] = peak_rss_bytes()
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "root_summary.json", summary)
    return summary


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "visible_logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": "cpu", "recurrence_dtype": "float32",
            "evaluation_softmax_and_score_dtype": "float64", "inference_batch_size": 1}


def artifact_hashes(out: Path) -> dict:
    return {str(path.relative_to(out)): sha256_file(path)
            for path in sorted(out.rglob("*")) if path.is_file() and path != out / "summary.json"}


def run(config: dict, source: dict, out: Path, budget: Budget) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty; previous outcomes are preserved")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "source_manifest.json", source)
    summary = {"schema_version": 1, "experiment_id": "LRM-REV-02", "verdict": VERDICT,
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "preflight_status": PREFLIGHT_STATUS, "run_status": "RUNNING",
               "source_commit": source["source_commit"], "source_hashes": source["source_hashes"],
               "registration": config, "environment": environment(),
               "planned_roots": config["roots"], "completed_roots": [], "n_units_completed": 0,
               "roots": [], "elapsed_seconds": budget.elapsed, "peak_rss_bytes": peak_rss_bytes(),
               "artifact_sha256": {}}
    write_json(out / "summary.json", summary)
    seen_sources = set()
    for root in config["roots"]:
        result = run_root(root, config, source, out, budget, seen_sources)
        summary["roots"].append(result)
        if result["status"] == "COMPLETE":
            summary["completed_roots"].append(root)
        summary["n_units_completed"] = len(summary["completed_roots"])
        summary["elapsed_seconds"] = budget.elapsed
        summary["peak_rss_bytes"] = peak_rss_bytes()
        summary["artifact_sha256"] = artifact_hashes(out)
        if result["status"] != "COMPLETE":
            summary["run_status"] = result["status"]
            write_json(out / "summary.json", summary)
            break
        write_json(out / "summary.json", summary)
        print(f"root {root}: complete, paired effect {result['means']['paired_effect']:.8f}", flush=True)
    if summary["n_units_completed"] == len(config["roots"]):
        summary["run_status"] = "COMPLETE"
    summary["elapsed_seconds"] = budget.elapsed
    summary["peak_rss_bytes"] = peak_rss_bytes()
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["artifact_sha256"] = artifact_hashes(out)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-engineering", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_engineering:
        parser.error("--run-engineering is required for this public development collection")
    config = load_registration()
    budget = Budget(config["budget"]["wall_seconds"], config["budget"]["rss_bytes"])
    source = verify_source(args.source_commit, config["source_files"])
    configure_cpu(config["budget"]["torch_threads"])
    result = run(config, source, args.out.resolve(), budget)
    print(canonical_json({"run_status": result["run_status"], "verdict": result["verdict"],
                          "completed_roots": result["completed_roots"],
                          "elapsed_seconds": result["elapsed_seconds"],
                          "peak_rss_bytes": result["peak_rss_bytes"]}), flush=True)
    return 0 if result["run_status"] == "COMPLETE" else (2 if result["run_status"] == "BUDGET_STOP" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
