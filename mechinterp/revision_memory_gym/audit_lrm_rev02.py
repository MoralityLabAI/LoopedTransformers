"""Independent arithmetic and provenance audit for LRM-REV-02 receipts.

Bayes targets and score arithmetic use independent standard-library code.
The frozen fixture reconstructs only source rows already in the archives.
A valid interrupted archive remains an interrupted run. The audit does not
replay model weights or attest that archived inputs were the inputs executed.
"""
from __future__ import annotations

import argparse
import builtins
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType


PRIORS = (Fraction(9, 20), Fraction(7, 20), Fraction(1, 5))
MATCH = (Fraction(19, 20), Fraction(13, 20), Fraction(3, 20))
N, J = 12, 6
ROOTS = [0, 1, 2, 3]
TRAIN_COUNT, DEVELOPMENT_COUNT, TRAIN_STEPS = 2048, 512, 800
ARMS = ("corrected", "random_0", "random_1", "random_2", "random_3")
SCORE_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-10
REGISTRATION = "mechinterp/revision_memory_gym/REGISTRATION_LRM_REV02.json"
FROZEN_GYM = "mechinterp/revision_memory_gym/gym.py"
FROZEN_FIXTURE = "mechinterp/revision_memory_gym/lrm_rev02_fixture.py"
PREFLIGHT_STATUS = "PENDING_MATCHED_SHAM_AND_CONFIRMATION_DESIGN"
MEAN_KEYS = ("initial_excess", "base_excess", "candidate_excess",
             "control_mean_excess", "paired_effect", "bayes_risk",
             "candidate_expected_brier", "control_mean_expected_brier")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _no_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _json(text):
    return json.loads(text, parse_constant=_no_constant)


def _read_json(path):
    return _json(path.read_text(encoding="utf-8"))


def _number(value, name, *, nonnegative=False):
    _need(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    _need(not nonnegative or value >= 0, name + " must be nonnegative")
    return float(value)


def _integer(value, name, low, high):
    _need(type(value) is int and low <= value <= high, name + " is outside its integer contract")
    return value


def _close(observed, expected, name):
    value = _number(observed, name)
    _need(abs(value - expected) <= SCORE_TOLERANCE,
          f"{name} differs: recorded {value!r}, recomputed {expected!r}")


def _probability(value, name):
    _need(type(value) is list and len(value) == 3, name + " requires three probabilities")
    out = [_number(x, name) for x in value]
    _need(all(0 <= x <= 1 for x in out), name + " is outside [0,1]")
    _need(abs(math.fsum(out) - 1) <= PROBABILITY_TOLERANCE, name + " is unnormalized")
    return out


def _bits(value, size, name):
    _need(type(value) is list and len(value) == size, name + " has wrong length")
    _need(all(type(x) is int and x in (0, 1) for x in value), name + " requires integer bits")
    return value


def _slots(value, name):
    _need(type(value) is list and len(value) == J, name + " requires six positions")
    _need(all(type(x) is int and 0 <= x < N for x in value), name + " has an invalid position")
    _need(len(set(value)) == J and value == sorted(value), name + " must be distinct and ascending")
    return value


def _relative(root, value):
    _need(type(value) is str and bool(value), "artifact/source path must be nonempty")
    path = Path(value)
    _need(not path.is_absolute() and ".." not in path.parts and str(path) == value,
          "artifact/source path must be normalized and relative")
    resolved = (root / path).resolve()
    _need(resolved.is_relative_to(root.resolve()), "artifact/source path escapes its root")
    _need(not (root / path).is_symlink(), "artifact/source path is symlinked")
    return resolved


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value, length, name):
    _need(type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
          name + " has an invalid digest")
    return value


def _normalize(weights):
    total = sum(weights, Fraction(0))
    return tuple(value / total for value in weights)


@lru_cache(maxsize=13)
def _initial_target(old_count):
    # Integrate informative and fair-bit counts independently. The named old
    # vector has a common combinatorial factor that cancels in Bayes' rule.
    weights = []
    for prior, match in zip(PRIORS, MATCH):
        likelihood = Fraction(0)
        for informative_ones in range(N - J + 1):
            fair_ones = old_count - informative_ones
            if 0 <= fair_ones <= J:
                likelihood += (math.comb(N - J, informative_ones)
                               * match ** informative_ones
                               * (1 - match) ** (N - J - informative_ones)
                               * math.comb(J, fair_ones) / Fraction(2 ** J))
        weights.append(prior * likelihood)
    return _normalize(weights)


@lru_cache(maxsize=13)
def _full_target(accepted_count):
    return _normalize(tuple(prior * match ** accepted_count
                            * (1 - match) ** (N - accepted_count)
                            for prior, match in zip(PRIORS, MATCH)))


def _load_source_identity(repo, commit):
    """Load the registered generator from Git, with its frozen gym dependency.

    Tests replace this loader with a hand-fixture oracle. Loading definitions
    generates no source worlds; callers request only rows present in archives.
    """
    prefix = "_lrm_rev02_audit_" + commit
    gym = ModuleType(prefix + "_gym")
    fixture = ModuleType(prefix + "_fixture")
    modules = (gym, fixture)
    for module in modules:
        sys.modules[module.__name__] = module
    try:
        code = subprocess.check_output(["git", "show", f"{commit}:{FROZEN_GYM}"], cwd=repo)
        exec(compile(code, f"{commit}/{FROZEN_GYM}", "exec"), gym.__dict__)
        original_import = builtins.__import__

        def frozen_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gym" and level == 0:
                return gym
            return original_import(name, globals, locals, fromlist, level)

        fixture.__dict__["__builtins__"] = {**vars(builtins), "__import__": frozen_import}
        code = subprocess.check_output(["git", "show", f"{commit}:{FROZEN_FIXTURE}"], cwd=repo)
        exec(compile(code, f"{commit}/{FROZEN_FIXTURE}", "exec"), fixture.__dict__)
        return {key: fixture.__dict__[key]
                for key in ("make_pilot_episode", "control_slots", "training_slots")}
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)


def _evidence(row, root, seen_sources, identity, phase):
    _need(type(row) is dict and row.get("root") == root and type(row.get("root")) is int,
          "row root differs from its archive")
    index = _integer(row.get("episode_index"), "episode_index", 0, TRAIN_COUNT - 1)
    source = _digest(row.get("source_id"), 64, "source_id")
    _need(source not in seen_sources, "duplicate source_id across roots or phases")
    seen_sources.add(source)
    old = _bits(row.get("old"), N, "old")
    slots = _slots(row.get("correction_slots"), "correction_slots")
    new = _bits(row.get("replacements"), J, "replacements")
    _integer(row.get("true_regime"), "true_regime", 0, 2)
    expected_source = identity["make_pilot_episode"](root, index, phase)
    _need(source == expected_source.source_id
          and tuple(old) == tuple(expected_source.old)
          and tuple(slots) == tuple(expected_source.correction_slots)
          and tuple(new) == tuple(expected_source.replacements)
          and row["true_regime"] == expected_source.true_regime,
          "archived source differs from its frozen root/index/phase generator")
    initial = tuple(map(float, _initial_target(sum(old))))
    count = sum(old) - sum(old[i] for i in slots) + sum(new)
    full = tuple(map(float, _full_target(count)))
    for name, expected in (("target_initial", initial), ("target_full", full)):
        observed = _probability(row.get(name), name)
        for k, (actual, target) in enumerate(zip(observed, expected)):
            _close(actual, target, f"{name}[{k}]")
    return index, slots, initial, full, expected_source


def _row(row, root, seen_sources, identity):
    _need(row.get("schema_version") == 1, "development row schema_version must be 1")
    index, slots, initial, full, expected_source = _evidence(row, root, seen_sources, identity, "development")
    _integer(index, "development episode_index", 0, DEVELOPMENT_COUNT - 1)
    probability = _probability(row.get("initial_probability"), "initial_probability")
    initial_loss = math.fsum((p - q) ** 2 for p, q in zip(probability, initial))
    _close(row.get("initial_excess"), initial_loss, "initial_excess")
    bayes_risk = 1 - math.fsum(q * q for q in full)
    probability_error = abs(math.fsum(probability) - 1)
    replay_logit = _number(row.get("prefix_replay_logits_abs_max"), "replay logits", nonnegative=True)
    replay_hidden = _number(row.get("prefix_replay_hidden_abs_max"), "replay hidden", nonnegative=True)
    replay_probability = _number(row.get("prefix_replay_probability_abs_max"), "replay probability", nonnegative=True)
    replay_forecast = _probability(row.get("prefix_replay_probability"), "prefix_replay_probability")
    probability_error = max(probability_error, abs(math.fsum(replay_forecast) - 1))
    arms = row.get("arms")
    _need(type(arms) is dict and set(arms) == set(ARMS), "all five registered arms are required")
    curves = {}
    base = None
    r0_error = 0.0
    for name in ARMS:
        arm = arms[name]
        _need(type(arm) is dict, name + " must be an object")
        arm_slots = _slots(arm.get("slots"), name + " slots")
        _need(name != "corrected" or arm_slots == slots, "corrected slots differ from S")
        if name != "corrected":
            expected_slots = identity["control_slots"](expected_source, int(name.split("_")[1]))
            _need(tuple(arm_slots) == tuple(expected_slots), name + " slots differ from frozen control policy")
        curve = arm.get("curve")
        _need(type(curve) is list and len(curve) == J + 1, name + " requires stages 0..6")
        audited = []
        for stage, point in enumerate(curve):
            _need(type(point) is dict and type(point.get("r")) is int and point["r"] == stage,
                  name + " has a missing, duplicate or reordered stage")
            forecast = _probability(point.get("probability"), name + " probability")
            probability_error = max(probability_error, abs(math.fsum(forecast) - 1))
            loss = math.fsum((p - q) ** 2 for p, q in zip(forecast, full))
            _close(point.get("excess"), loss, name + " excess")
            _close(point.get("expected_brier"), loss + bayes_risk, name + " expected_brier")
            displacement = _number(point.get("hidden_displacement_l2"), "hidden displacement", nonnegative=True)
            if stage == 0:
                _close(displacement, 0.0, name + " r0 displacement")
                if base is None:
                    base = forecast
                r0_error = max(r0_error, max(abs(x - y) for x, y in zip(base, forecast)))
            audited.append({"r": stage, "excess": loss, "expected_brier": loss + bayes_risk,
                            "hidden_displacement_l2": displacement})
        curves[name] = audited
    _need(r0_error <= SCORE_TOLERANCE, "r0 forecasts differ across cloned arms")
    _close(replay_probability, max(abs(x - y) for x, y in zip(base, replay_forecast)),
           "prefix_replay_probability_abs_max")
    candidate = curves["corrected"][-1]["excess"]
    control = math.fsum(curves[name][-1]["excess"] for name in ARMS[1:]) / 4
    paired = control - candidate
    _close(row.get("paired_effect"), paired, "paired_effect")
    return {"episode_index": index,
            "means": {"initial_excess": initial_loss, "base_excess": curves["corrected"][0]["excess"],
                      "candidate_excess": candidate, "control_mean_excess": control,
                      "paired_effect": paired, "bayes_risk": bayes_risk,
                      "candidate_expected_brier": candidate + bayes_risk,
                      "control_mean_expected_brier": control + bayes_risk},
            "curves": curves,
            "maxima": {"probability_abs_sum_error_max": probability_error,
                       "prefix_replay_logits_abs_max": replay_logit,
                       "prefix_replay_hidden_abs_max": replay_hidden,
                       "prefix_replay_probability_abs_max": replay_probability,
                       "r0_abs_difference_max": r0_error}}


def _lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            _need(bool(line.strip()), f"blank archive line {number}")
            value = _json(line)
            _need(type(value) is dict, f"archive line {number} must be an object")
            yield value


def _failure_context(out, root_summary, artifacts, identity):
    """Check retained failing source identities without expanding a cohort."""
    root = root_summary["root"]
    context = root_summary.get("failure_context")
    _need(type(context) is dict and type(context.get("root")) is int
          and context["root"] == root, "interrupted root requires its failure context")

    def raw_identity(row, phase):
        limit = DEVELOPMENT_COUNT if phase == "development" else TRAIN_COUNT
        index = _integer(row.get("episode_index"), "failed episode_index", 0, limit - 1)
        _need(type(row.get("root")) is int and row["root"] == root, "failed source root differs")
        old = _bits(row.get("old"), N, "failed old")
        slots = _slots(row.get("correction_slots"), "failed correction_slots")
        new = _bits(row.get("replacements"), J, "failed replacements")
        expected = identity["make_pilot_episode"](root, index, phase)
        _need(row.get("source_id") == expected.source_id and tuple(old) == tuple(expected.old)
              and tuple(slots) == tuple(expected.correction_slots) and tuple(new) == tuple(expected.replacements)
              and type(row.get("true_regime")) is int and row["true_regime"] == expected.true_regime,
              "failure source differs from frozen generator")

    phase = context.get("phase")
    _need(phase in ("initialization", "training_source", "training_step", "checkpoint_final",
                    "development", "development_complete"), "unknown failure phase")
    if "source_id" in context:
        _need(phase in ("training_source", "development"), "source identity in unexpected failure phase")
        raw_identity(context, "train" if phase == "training_source" else "development")
    if "batch_sources" in context:
        _need(phase == "training_step" and type(context["batch_sources"]) is list,
              "batch source context must belong to training")
        for row in context["batch_sources"]:
            raw_identity(row, "train")
    failed_file = f"root_{root}/failed_episode.json"
    if failed_file in artifacts:
        _need(root_summary["status"] == "FAILED" and phase == "development" and "source_id" in context,
              "failed episode artifact has the wrong scope")
        _need(_read_json(_relative(out, failed_file)) == context, "failed episode differs from retained failure context")
    if root_summary["status"] == "FAILED" and phase == "development" and "source_id" in context:
        _need(failed_file in artifacts, "failing development episode was not retained")


def _root_summary(out, root_summary, artifacts, config, seen_sources, identity):
    root = _integer(root_summary.get("root"), "root", 0, ROOTS[-1])
    _need(root_summary.get("schema_version") == 1, "root summary schema_version must be 1")
    status = root_summary.get("status")
    _need(status in ("COMPLETE", "BUDGET_STOP", "FAILED"), "root status must preserve its terminal outcome")
    if status != "COMPLETE":
        _need(type(root_summary.get("error")) is str and bool(root_summary["error"]), "partial root requires its error")
        _failure_context(out, root_summary, artifacts, identity)
    root_file = f"root_{root}/root_summary.json"
    _need(root_file in artifacts, "root summary is absent from artifact hashes")
    _need(_read_json(_relative(out, root_file)) == root_summary, "embedded root summary differs from root_summary.json")
    steps = _integer(root_summary.get("train_steps_completed"), "train_steps_completed", 0, TRAIN_STEPS)
    train_count = _integer(root_summary.get("n_train_episodes"), "n_train_episodes", 0, TRAIN_COUNT)
    count = _integer(root_summary.get("development_count"), "development_count", 0, DEVELOPMENT_COUNT)
    for key in ("model_seed", "sampler_seed"):
        _integer(root_summary.get(key), key, 0, (1 << 63) - 1)
    if "model_parameters" in root_summary:
        _need(type(root_summary["model_parameters"]) is int and root_summary["model_parameters"] == 4995,
              "model parameter count differs from registration")
    if "checkpoint_file" in root_summary:
        checkpoint = root_summary["checkpoint_file"]
        _need(checkpoint in artifacts, "checkpoint is absent from artifact hashes")
        _need(checkpoint in (f"root_{root}/checkpoint_final.pt", f"root_{root}/checkpoint_partial.pt"),
              "checkpoint path differs from its root")
        _need(artifacts[checkpoint] == _digest(root_summary.get("checkpoint_sha256"), 64, "checkpoint_sha256"),
              "checkpoint digest differs between root and run summaries")
        _digest(root_summary.get("weights_tensor_sha256"), 64, "weights_tensor_sha256")
    else:
        _need(steps == 0 and count == 0 and "model_parameters" not in root_summary,
              "a created/scored model requires a retained checkpoint")
    train_path = f"root_{root}/training_episodes.jsonl.gz"
    if train_path in artifacts:
        indexes = []
        for row in _lines(_relative(out, train_path)):
            index, _, _, _, episode = _evidence(row, root, seen_sources, identity, "train")
            slots = _slots(row.get("reminder_slots"), "training reminder_slots")
            _need(tuple(slots) == tuple(identity["training_slots"](episode)), "training reminder policy differs from frozen source")
            indexes.append(index)
        _need(indexes == list(range(train_count)), "training archive count/order differs from summary")
        if "training_archive_sha256" in root_summary:
            _need(root_summary["training_archive_sha256"] == artifacts[train_path], "training archive hash differs")
    else:
        _need(train_count == 0, "training archive missing for reported episodes")
    trace_path = f"root_{root}/training_trace.jsonl"
    if trace_path in artifacts:
        trace = list(_lines(_relative(out, trace_path)))
        _need([row.get("step") for row in trace] == list(range(1, steps + 1)), "training trace steps differ from summary")
        for row in trace:
            for key, value in row.items():
                if key.endswith("_loss"):
                    _number(value, "training " + key, nonnegative=True)
    else:
        _need(steps == 0, "training trace missing for completed steps")
    rows = []
    if "development_outcomes_file" in root_summary:
        path = root_summary["development_outcomes_file"]
        _need(path == f"root_{root}/development_outcomes.jsonl.gz" and path in artifacts,
              "development archive path/hash missing or wrong")
        rows = [_row(row, root, seen_sources, identity) for row in _lines(_relative(out, path))]
    _need([row["episode_index"] for row in rows] == list(range(count)), "development archive count/order differs from summary")
    if rows:
        _need(steps == TRAIN_STEPS and train_count == TRAIN_COUNT, "development scoring preceded scheduled training completion")
        _need(root_summary.get("checkpoint_file") == f"root_{root}/checkpoint_final.pt", "development scoring requires final checkpoint")
    if status == "COMPLETE":
        _need(steps == TRAIN_STEPS and train_count == TRAIN_COUNT and count == DEVELOPMENT_COUNT,
              "complete root lacks registered training/development counts")
    if not rows:
        _need(root_summary.get("means") == {} and root_summary.get("curve_means") == {}
              and root_summary.get("diagnostics") == {}, "unscored root must retain empty diagnostic aggregates")
        return {"root": root, "status": status, "development_count": 0}
    means = {key: math.fsum(row["means"][key] for row in rows) / count for key in MEAN_KEYS}
    for key, expected in means.items():
        _close(root_summary.get("means", {}).get(key), expected, "root means." + key)
    curves = root_summary.get("curve_means")
    _need(type(curves) is dict and set(curves) == set(ARMS), "root curve means require all five arms")
    for arm in ARMS:
        _need(type(curves[arm]) is list and len(curves[arm]) == 7, "root curve means require all stages")
        for stage in range(7):
            point = curves[arm][stage]
            _need(type(point.get("r")) is int and point["r"] == stage, "root curve stages differ")
            for key in ("excess", "expected_brier", "hidden_displacement_l2"):
                expected = math.fsum(row["curves"][arm][stage][key] for row in rows) / count
                _close(point.get(key), expected, f"root curve {arm}/{stage}/{key}")
    maxima = {key: max(row["maxima"][key] for row in rows) for key in rows[0]["maxima"]}
    diagnostics = root_summary.get("diagnostics")
    _need(type(diagnostics) is dict, "root diagnostics required")
    for key, expected in maxima.items():
        _close(diagnostics.get(key), expected, "root diagnostics." + key)
    threshold = config["diagnostics"]
    numerical = (maxima["probability_abs_sum_error_max"] <= PROBABILITY_TOLERANCE
                 and maxima["prefix_replay_logits_abs_max"] <= threshold["replay_abs_max"]
                 and maxima["prefix_replay_hidden_abs_max"] <= threshold["replay_abs_max"]
                 and maxima["prefix_replay_probability_abs_max"] <= threshold["replay_abs_max"]
                 and maxima["r0_abs_difference_max"] <= SCORE_TOLERANCE)
    flags = {"initial_competence": means["initial_excess"] <= threshold["initial_excess_max"],
             "candidate_competence": means["candidate_excess"] <= threshold["candidate_excess_max"],
             "numeric_pass": numerical}
    flags["engineering_competence_pass"] = all(flags.values())
    for key, expected in flags.items():
        _need(type(diagnostics.get(key)) is bool and diagnostics[key] == expected, "root diagnostic flag differs: " + key)
    return {"root": root, "status": status, "development_count": count,
            "recomputed_means": means, "recomputed_diagnostics": {**maxima, **flags}}


def _provenance(out, repo, summary, artifacts):
    _need("source_manifest.json" in artifacts, "source manifest is absent from artifact hashes")
    source = _read_json(out / "source_manifest.json")
    _need(source.get("schema_version") == 1, "source manifest schema_version must be 1")
    commit = _digest(source.get("source_commit"), 40, "source_commit")
    tree = _digest(source.get("source_tree_sha"), 40, "source_tree_sha")
    _need(summary.get("source_commit") == commit, "source commit differs across receipts")
    hashes = source.get("source_hashes")
    _need(type(hashes) is dict and hashes and hashes == summary.get("source_hashes"), "source hashes differ or are missing")
    config = summary.get("registration")
    _need(type(config) is dict and config.get("experiment_id") == "LRM-REV-02", "registration is missing")
    _need(config.get("roots") == ROOTS and summary.get("planned_roots") == ROOTS, "registered/planned roots differ")
    _need(config.get("source_files") == list(dict.fromkeys(config.get("source_files", [])))
          and set(config["source_files"]) == set(hashes), "source hash coverage differs from registration")
    _need({REGISTRATION, FROZEN_GYM, FROZEN_FIXTURE} <= set(hashes),
          "registration and generator dependencies must be source-hashed")
    _need(config.get("evidence_law", {}).get("n") == N and config["evidence_law"].get("j") == J,
          "registered evidence dimensions differ")
    _need(config["evidence_law"].get("priors") == [[9, 20], [7, 20], [1, 5]]
          and config["evidence_law"].get("match_probabilities") == [[19, 20], [13, 20], [3, 20]],
          "registered evidence law differs")
    _need(config.get("training", {}).get("steps") == TRAIN_STEPS
          and config["training"].get("episodes_per_root") == TRAIN_COUNT
          and config.get("development", {}).get("episodes_per_root") == DEVELOPMENT_COUNT
          and config["development"].get("controls") == 4, "registered episode/step budgets differ")
    observed_tree = subprocess.check_output(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=repo, text=True).strip()
    _need(observed_tree == tree, "frozen source tree differs from Git")
    for relative, expected in hashes.items():
        _relative(repo, relative)
        _digest(expected, 64, "source hash")
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
        _need(hashlib.sha256(content).hexdigest() == expected, "source hash differs from frozen Git: " + relative)
        if relative == REGISTRATION:
            _need(_json(content.decode("utf-8")) == config, "embedded registration differs from frozen source")
    return config, {"source_commit": commit, "source_tree_sha": tree, "source_files_verified": len(hashes),
                    "checkpoint_scope": "checkpoint bytes hashed; model tensors and execution are not replayed",
                    "input_scope": "every archived train/development source and reminder policy replayed from frozen Git; exact posteriors independently checked"}


def audit_run(out_dir, repo_root=None):
    """Audit one terminal receipt directory without collecting model data.

    ``passed`` reports arithmetic/provenance integrity. ``complete`` additionally
    requires the registered four roots and counts. Interrupted runs can pass
    integrity while retaining ``complete=False`` and their original status.
    """
    out = Path(out_dir).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    result = {"schema_version": 1, "experiment_id": "LRM-REV-02", "passed": False,
              "complete": False, "run_status": None, "failures": [], "roots_checked": [],
              "development_rows_checked": 0, "verdict": "ENGINEERING_PILOT_ONLY"}
    try:
        summary = _read_json(out / "summary.json")
        _need(type(summary) is dict and summary.get("schema_version") == 1, "run summary schema_version must be 1")
        _need(summary.get("experiment_id") == "LRM-REV-02", "wrong experiment")
        status = summary.get("run_status")
        result["run_status"] = status
        _need(status in ("COMPLETE", "BUDGET_STOP", "FAILED"), "audit requires a terminal run status")
        _need(summary.get("verdict") == "ENGINEERING_PILOT_ONLY", "scientific scope verdict changed")
        _need(summary.get("preflight_status") == PREFLIGHT_STATUS, "preflight status changed")
        artifacts = summary.get("artifact_sha256")
        _need(type(artifacts) is dict and artifacts, "artifact hashes required")
        for relative, expected in artifacts.items():
            _digest(expected, 64, "artifact hash")
            _need(relative not in ("summary.json", "AUDIT.json"), "self-referential artifact inventory")
            path = _relative(out, relative)
            _need(path.is_file() and _sha256(path) == expected, "artifact hash mismatch: " + relative)
        result["artifacts_verified"] = len(artifacts)
        config, result["provenance"] = _provenance(out, repo, summary, artifacts)
        identity = _load_source_identity(repo, result["provenance"]["source_commit"])
        roots = summary.get("roots")
        _need(type(roots) is list, "root summaries must be a list")
        ids = [root.get("root") for root in roots if type(root) is dict]
        _need(len(ids) == len(roots) and len(set(ids)) == len(ids), "duplicate or malformed root summary")
        _need(ids == ROOTS[:len(ids)], "attempted roots must preserve registered order")
        completed = [root["root"] for root in roots if root.get("status") == "COMPLETE"]
        _need(summary.get("completed_roots") == completed, "completed root list differs from statuses")
        _need(type(summary.get("n_units_completed")) is int and summary["n_units_completed"] == len(completed),
              "completed unit count differs")
        if status == "COMPLETE":
            _need(completed == ROOTS and len(roots) == len(ROOTS), "COMPLETE run requires all registered roots")
        elif roots:
            _need(roots[-1].get("status") == status, "interrupted run status differs from last attempted root")
        seen_sources = set()
        for root in roots:
            checked = _root_summary(out, root, artifacts, config, seen_sources, identity)
            result["roots_checked"].append(checked)
            result["development_rows_checked"] += checked["development_count"]
        result["passed"] = True
        result["complete"] = status == "COMPLETE"
        if completed:
            full_roots = [root for root in result["roots_checked"] if root["status"] == "COMPLETE"]
            result["completed_root_mean_effect"] = math.fsum(root["recomputed_means"]["paired_effect"] for root in full_roots) / len(full_roots)
        result["preflight_status"] = PREFLIGHT_STATUS
    except Exception as error:
        # A malformed receipt or unavailable frozen dependency remains a
        # machine-readable failed audit rather than losing the failure record.
        result["passed"] = False
        result["complete"] = False
        result["failures"].append(f"{type(error).__name__}: {error}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--out", type=Path, help="write the audit JSON, for example OUTPUT/AUDIT.json")
    args = parser.parse_args(argv)
    result = audit_run(args.output_dir, args.repo_root)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
