"""Fit AF-02 train/dev sources and codecs, then freeze complete artifacts.

Primary fitting reads only the canonical train and dev inputs. The separately
prepared complete corpus receipt is bound by its file hash. Every source and
codec has a Torch checkpoint and an independent named-array export.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import gym
from .model import (ARCHITECTURES, ARMS, LEARNED_ARMS, ROOT, audit_loss,
                    commutation_loss, composite_audits, make_codec, make_model,
                    matched_donors, paired_continuation, resources,
                    task_from_audits)


REGISTERED_PROTOCOL_SHA256 = "f6c3a939a09183c265466d506de985ec1a92c76d2d57f50cdd5f3f984e224c41"
PREDICTION_FILE = "train_dev_predictions.jsonl"
CONFIG = {
    "experiment": "AF-02", "architectures": list(ARCHITECTURES), "arms": list(ARMS),
    "seeds": [17, 29, 43], "input_dim": 9, "calls_per_event": 2,
    "gru_width": 32, "gru_code_width": 4,
    "native_token_width": 16, "native_atoms": 3, "native_rules": 2,
    "native_state_coordinates": 80, "native_code_width_per_token": 4,
    "native_code_coordinates": 20, "coordinate_bits": 32,
    "teacher_steps": 400, "codec_steps": 300, "batch_size": 64,
    "optimizer": "AdamW", "learning_rate": .003, "weight_decay": .01,
    "gradient_clip": 1., "source_audit_weight": .5,
    "task_distillation_kl_weight": .25, "commutation_weight": .5,
    "paired_task_weight": .25, "audit_weight": .5, "paired_audit_weight": .25,
    "source_qualification": .95, "threads": 2, "torch": "2.5.1+cpu",
    "selection": "fixed final checkpoint; dev eligibility only",
    "train_inputs": 108, "dev_inputs": 36, "split_access": ["train", "dev"],
    "sample_weighting": "uniform canonical inputs; sampled length buckets weighted by row count",
    "pca_weighting": "equal per canonical input, then visits and tokens within each type",
}

SOURCE_PATHS = {name: Path(__file__).with_name(name) for name in
                ("gym.py", "model.py", "train.py", "evaluate.py", "replay.py",
                 "inverse.py", "audit.py", "PROTOCOL.json", "corpus_manifest.json",
                 "address_erasure_reference.json")}
SOURCE_PATHS.update({
    "native/train.py": ROOT / "experiments/lean_reasoner/train.py",
    "native/proof_data.py": ROOT / "experiments/lean_reasoner/proof_data.py",
    "af01/gym.py": ROOT / "experiments/activation_funnel/gym.py",
    "af01/model.py": ROOT / "experiments/activation_funnel/model.py",
    "af01/certify.py": ROOT / "experiments/activation_funnel/certify.py",
})
PINNED_DEPENDENCIES = {
    "native/train.py": "8c269d1e698c92b384b554b039eec5fc95b5b544520219ff20c3eb794f88cfa1",
    "native/proof_data.py": "6c67e075fa8e5f162721d22c2e56078f40b8117d716951806afd2494dd08721f",
    "af01/gym.py": "5f94639558f1a40a1214bb394f11a8f1293101acce763a51705073bc049811f4",
    "af01/model.py": "abc2ea1075726e9879940444c273230019cf330d68b6e606ae53612ac076fbfc",
    "af01/certify.py": "995e36a9755d50a4d72464d89668df783b91b956801fb20b65b71d05e784f113",
    "PROTOCOL.json": REGISTERED_PROTOCOL_SHA256,
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def source_hashes():
    result = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}
    for name, expected in PINNED_DEPENDENCIES.items():
        if result[name] != expected:
            raise ValueError(f"pinned source mismatch: {name}")
    return result


def tensor_data(cases):
    if not cases or len({len(c.schedule) for c in cases}) != 1:
        raise ValueError("tensor_data requires one nonempty true-length bucket")
    return (torch.tensor([gym.feature_sequence(c) for c in cases], dtype=torch.float32),
            torch.tensor([gym.target_action(c) for c in cases], dtype=torch.long),
            torch.tensor([gym.audit_targets(c) for c in cases], dtype=torch.long))


def bucket_indices(cases, indices=None):
    groups = {}
    for i in range(len(cases)) if indices is None else indices:
        groups.setdefault(len(cases[i].schedule), []).append(i)
    return {length: groups[length] for length in sorted(groups)}


class _Data:
    def __init__(self, cases):
        self.cases = cases
        self.x = [torch.tensor(gym.feature_sequence(c), dtype=torch.float32) for c in cases]
        self.task = torch.tensor([gym.target_action(c) for c in cases], dtype=torch.long)
        self.audit = torch.tensor([gym.audit_targets(c) for c in cases], dtype=torch.long)
        if len({gym.public_input_hash(c) for c in cases}) != len(cases):
            raise ValueError("fitting requires exactly one row per canonical input")

    def batch(self, indices):
        return torch.stack([self.x[i] for i in indices]), self.task[indices], self.audit[indices]


def _metrics(rows):
    if not rows:
        raise ValueError("accuracy requires observed canonical inputs")
    n = len(rows)
    return {"cases": n, "exact_feature_patterns": len({r["public_input_sha256"] for r in rows}),
            "task_accuracy": sum(r["prediction"] == r["target_action"] for r in rows) / n,
            "audit_branch_accuracy": [sum(r["audit_prediction"][j] == r["audit_target"][j]
                                           for r in rows) / n for j in (0, 1)],
            "audit_exact_accuracy": sum(r["audit_prediction"] == r["audit_target"] for r in rows) / n}


def summarize_predictions(rows):
    result = _metrics(rows)
    if result["cases"] != result["exact_feature_patterns"]:
        raise ValueError("prediction metrics require canonical inputs without host copies")
    result["by_order"] = {",".join(map(str, order)): _metrics(
        [r for r in rows if tuple(r["probe_order"]) == order]) for order in gym.ORDERS}
    return result


def source_qualified(train_metrics, dev_metrics, threshold=.95):
    return all(group["task_accuracy"] >= threshold and
               all(value >= threshold for value in group["audit_branch_accuracy"])
               for metrics in (train_metrics, dev_metrics) for group in metrics["by_order"].values())


@torch.no_grad()
def predict_cases(model, codec, data, architecture, seed, arm):
    rows = [None] * len(data.cases)
    for indices in bucket_indices(data.cases).values():
        x, task, audit = data.batch(indices)
        trace = model.rollout(x, codec)
        pred, ap = trace.task_logits.argmax(-1).tolist(), trace.audit_logits.argmax(-1).tolist()
        for row, i in enumerate(indices):
            case = data.cases[i]
            rows[i] = {"architecture": architecture, "seed": seed, "arm": arm,
                       "split": case.split, "case_id": case.case_id,
                       "canonical_id": case.canonical_id,
                       "public_input_sha256": gym.public_input_hash(case),
                       "target_digest": case.target_digest, "context_digest": case.context_digest,
                       "probe_order": list(case.probe_order), "schedule": case.schedule,
                       "depth": case.depth, "status_pattern": list(case.statuses),
                       "target_action": int(task[row]), "audit_target": audit[row].tolist(),
                       "prediction": pred[row], "branch": pred[row] if pred[row] < 2 else None,
                       "audit_prediction": ap[row],
                       "task_logits": trace.task_logits[row].tolist(),
                       "audit_logits": trace.audit_logits[row].tolist()}
    return rows


def _weighted_pca(bank, weights, code_width=4):
    weights = weights / weights.sum()
    mean = (bank * weights[:, None]).sum(0)
    centered = (bank - mean) * weights.sqrt()[:, None]
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:code_width].T.contiguous()
    # Resolve arbitrary SVD signs without using any labels or held-out values.
    largest = basis.abs().argmax(0)
    sign = basis[largest, torch.arange(basis.shape[1])].sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return mean, basis * sign


def fit_bases(architecture, states):
    """Equal input mass, then equal visit/token mass within each node type."""
    types = {"state": (0, 1)} if architecture == "gru" else {"atom": (0, 3), "rule": (3, 5)}
    result = {}
    for name, (start, stop) in types.items():
        banks, weights = [], []
        for h in states:
            selected = h if architecture == "gru" else h.reshape(-1, 5, 16)[:, start:stop]
            bank = selected.reshape(-1, 32 if architecture == "gru" else 16).float()
            banks.append(bank)
            weights.append(bank.new_full((len(bank),), 1. / (len(states) * len(bank))))
        bank, weight = torch.cat(banks), torch.cat(weights)
        mean, basis = _weighted_pca(bank, weight)
        result[name] = {"mean": mean, "basis": basis, "source_vectors": len(bank),
                        "weight_sum": float(weight.sum())}
    return result


def initialize_codec(codec, bases):
    if codec.architecture == "gru":
        codec.initialize_from_basis(bases["state"]["mean"], bases["state"]["basis"])
    else:
        for name in ("atom", "rule"):
            getattr(codec, name).initialize_from_basis(bases[name]["mean"], bases[name]["basis"])


def _basis_receipt(bases):
    return {name: {key: value.tolist() if torch.is_tensor(value) else value
                   for key, value in item.items()} for name, item in bases.items()}


def checkpoint_name(architecture, arm, seed, suffix="pt"):
    return f"{architecture}_{arm}_s{seed}.{suffix}"


def expected_file_roster(config=None):
    config = CONFIG if config is None else config
    return {checkpoint_name(architecture, arm, seed, suffix)
            for architecture in config["architectures"] for seed in config["seeds"]
            for arm in config["arms"] for suffix in ("pt", "npz")}


def _file_entry(path):
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def save_checkpoint(out, architecture, seed, arm, module, config, files):
    name = checkpoint_name(architecture, arm, seed)
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    payload = {"architecture": architecture, "seed": seed, "arm": arm,
               "configuration_sha256": canonical_hash(config), "state_dict": state}
    if arm != "teacher":
        teacher_name = checkpoint_name(architecture, "teacher", seed)
        payload.update({"source_checkpoint": teacher_name,
                        "source_checkpoint_sha256": files[teacher_name]["sha256"]})
    torch.save(payload, out / name)
    files[name] = _file_entry(out / name)
    array_name = checkpoint_name(architecture, arm, seed, "npz")
    np.savez(out / array_name, **{key: value.numpy() for key, value in state.items()})
    files[array_name] = _file_entry(out / array_name)
    return {"checkpoint": name, "array_export": array_name}


def common_task_loss(student, teacher, target):
    return (F.cross_entropy(student.task_logits, target) + .25 *
            F.kl_div(F.log_softmax(student.task_logits, -1),
                     F.softmax(teacher.task_logits.detach(), -1), reduction="batchmean"))


def train_run(architecture, seed, train_data, dev_data, out, config, files, predictions):
    torch.manual_seed(seed)
    random.seed(seed)
    teacher = make_model(architecture)
    opt = torch.optim.AdamW(teacher.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    rng = torch.Generator().manual_seed(seed + 1000)
    began, losses = time.perf_counter(), []
    for step in range(config["teacher_steps"]):
        selected = torch.randint(len(train_data.cases), (config["batch_size"],), generator=rng).tolist()
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        for indices in bucket_indices(train_data.cases, selected).values():
            x, task, audit = train_data.batch(indices)
            trace = teacher.rollout(x)
            bucket = F.cross_entropy(trace.task_logits, task) + .5 * audit_loss(trace.audit_logits, audit)
            loss = loss + len(indices) / len(selected) * bucket
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), config["gradient_clip"])
        opt.step()
        if step % 100 == 0 or step == config["teacher_steps"] - 1:
            losses.append([step, float(loss.detach())])
    elapsed = time.perf_counter() - began
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train_rows = predict_cases(teacher, None, train_data, architecture, seed, "teacher")
    dev_rows = predict_cases(teacher, None, dev_data, architecture, seed, "teacher")
    predictions.extend(train_rows + dev_rows)
    train_metrics, dev_metrics = summarize_predictions(train_rows), summarize_predictions(dev_rows)
    source_files = save_checkpoint(out, architecture, seed, "teacher", teacher, config, files)
    meta = {"architecture": architecture, "seed": seed,
            "source_qualified": source_qualified(train_metrics, dev_metrics),
            "teacher_train": train_metrics, "teacher_dev": dev_metrics,
            "teacher_losses": losses, "teacher_training_seconds": elapsed,
            "teacher_resources_by_events": {str(n): resources(teacher, events=n)
                                             for n in sorted({len(c.schedule) for c in train_data.cases})},
            "teacher_files": source_files, "arms": {}}
    print(json.dumps({"architecture": architecture, "seed": seed,
                      "source_qualified": meta["source_qualified"], "teacher_dev": dev_metrics}), flush=True)
    began = time.perf_counter()
    with torch.no_grad():
        states = [None] * len(train_data.cases)
        for indices in bucket_indices(train_data.cases).values():
            x, _, _ = train_data.batch(indices)
            trace = teacher.rollout(x)
            for row, i in enumerate(indices):
                states[i] = trace.states[row].clone()
        bases = fit_bases(architecture, states)
    meta["train_only_basis_fit_seconds"] = time.perf_counter() - began
    meta["pca_initialization"] = _basis_receipt(bases)
    for arm in LEARNED_ARMS:
        torch.manual_seed(seed + 5000)
        codec = make_codec(architecture)
        initialize_codec(codec, bases)
        opt = torch.optim.AdamW(codec.parameters(), lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
        rng = torch.Generator().manual_seed(seed + 2000)
        began, losses = time.perf_counter(), []
        for step in range(config["codec_steps"]):
            selected = torch.randint(len(train_data.cases), (config["batch_size"],), generator=rng).tolist()
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros(())
            for indices in bucket_indices(train_data.cases, selected).values():
                x, task, audit = train_data.batch(indices)
                cases = [train_data.cases[i] for i in indices]
                donor = matched_donors(cases, rng)  # consumed identically in both arms
                with torch.no_grad():
                    native = teacher.rollout(x)
                student = teacher.rollout(x, codec)
                bucket = common_task_loss(student, native, task)
                if arm == "joint":
                    mixed = composite_audits(cases, audit, donor)
                    paired = paired_continuation(teacher, x, cases, donor, codec)
                    bucket = bucket + .5 * commutation_loss(teacher, codec, x, native)
                    bucket = bucket + .25 * F.cross_entropy(paired.task_logits, task_from_audits(mixed))
                    bucket = bucket + .5 * audit_loss(student.audit_logits, audit)
                    bucket = bucket + .25 * audit_loss(paired.audit_logits, mixed)
                loss = loss + len(indices) / len(selected) * bucket
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), config["gradient_clip"])
            opt.step()
            if step % 100 == 0 or step == config["codec_steps"] - 1:
                losses.append([step, float(loss.detach())])
        fitting_elapsed = time.perf_counter() - began
        codec.eval()
        arm_files = save_checkpoint(out, architecture, seed, arm, codec, config, files)
        train_rows = predict_cases(teacher, codec, train_data, architecture, seed, arm)
        dev_rows = predict_cases(teacher, codec, dev_data, architecture, seed, arm)
        predictions.extend(train_rows + dev_rows)
        meta["arms"][arm] = {"train": summarize_predictions(train_rows),
                               "dev": summarize_predictions(dev_rows), "losses": losses,
                               "training_seconds": fitting_elapsed, "files": arm_files,
                               "resources_by_events": {str(n): resources(teacher, codec, events=n)
                                                       for n in sorted({len(c.schedule) for c in train_data.cases})}}
        print(json.dumps({"architecture": architecture, "seed": seed, "arm": arm,
                          "dev": meta["arms"][arm]["dev"]}), flush=True)
    return meta


def _dataset_summary(data):
    return {"canonical_inputs": len(data.cases),
            "canonical_cases_sha256": canonical_hash([c.to_dict() for c in data.cases]),
            "public_sequences_sha256": canonical_hash([x.tolist() for x in data.x]),
            "length_counts": {str(n): count for n, count in sorted(Counter(
                len(c.schedule) for c in data.cases).items())}}


def validate_corpus_inputs(corpus_path, datasets):
    receipt = json.loads(Path(corpus_path).read_text())
    claimed = receipt.get("manifest_sha256")
    body = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
    if claimed != canonical_hash(body):
        raise ValueError("frozen corpus manifest self-digest mismatch")
    if receipt.get("protocol_sha256") != REGISTERED_PROTOCOL_SHA256:
        raise ValueError("frozen corpus protocol mismatch")
    summaries = {split: _dataset_summary(data) for split, data in datasets.items()}
    for split, summary in summaries.items():
        expected = receipt["splits"][split]
        for key in ("canonical_inputs", "canonical_cases_sha256", "public_sequences_sha256"):
            if summary[key] != expected[key]:
                raise ValueError(f"frozen corpus {split} {key} mismatch")
    return summaries


def train(out, seeds=None, threads=2, smoke=False):
    out = Path(out)
    hashes = source_hashes()
    corpus = Path(__file__).with_name("corpus_manifest.json")
    corpus_hash = sha256_file(corpus)
    if out.exists() and any(out.iterdir()):
        raise ValueError("training output must be empty; retain every earlier attempt")
    if not smoke and (str(torch.__version__) != CONFIG["torch"] or threads != CONFIG["threads"]):
        raise ValueError("primary training requires Torch2.5.1+cpu and two threads")
    config = dict(CONFIG)
    config["seeds"] = list(CONFIG["seeds"] if seeds is None else seeds)
    if not smoke and config["seeds"] != CONFIG["seeds"]:
        raise ValueError("primary training requires every registered seed in order")
    if smoke:
        config["teacher_steps"], config["codec_steps"] = 8, 4
    out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    train_data, dev_data = _Data(gym.canonical_cases("train")), _Data(gym.canonical_cases("dev"))
    if len(train_data.cases) != CONFIG["train_inputs"] or len(dev_data.cases) != CONFIG["dev_inputs"]:
        raise ValueError("registered canonical training/dev counts differ")
    if {gym.public_input_hash(c) for c in train_data.cases} & {gym.public_input_hash(c) for c in dev_data.cases}:
        raise ValueError("training and dev public inputs overlap")
    summaries = validate_corpus_inputs(corpus, {"train": train_data, "dev": dev_data})
    manifest = {"experiment": "AF-02", "scope": "engineering_smoke" if smoke else "registered_primary",
                "configuration": config, "configuration_sha256": canonical_hash(config),
                "protocol_sha256": REGISTERED_PROTOCOL_SHA256, "source_hashes": hashes,
                "frozen_corpus_sha256": corpus_hash, "torch_version": str(torch.__version__),
                "threads": threads, "dataset_access": ["train", "dev"],
                "datasets": summaries,
                "files": {}, "prediction_files": {}, "runs": [], "training_complete": False}
    predictions = []
    for architecture in config["architectures"]:
        for seed in config["seeds"]:
            manifest["runs"].append(train_run(architecture, seed, train_data, dev_data, out,
                                               config, manifest["files"], predictions))
    if set(manifest["files"]) != expected_file_roster(config):
        raise ValueError("incomplete checkpoint/export roster")
    path = out / PREDICTION_FILE
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions))
    manifest["prediction_files"][PREDICTION_FILE] = {**_file_entry(path), "rows": len(predictions)}
    manifest["all_sources_qualified"] = all(row["source_qualified"] for row in manifest["runs"])
    manifest["training_complete"] = True
    if source_hashes() != hashes or sha256_file(corpus) != corpus_hash:
        raise ValueError("source or frozen corpus changed during fitting")
    (out / "train_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out / "train_manifest.sha256").write_text(sha256_file(out / "train_manifest.json") + "\n")
    return manifest


class _VerifiedManifest(dict):
    """A read-verified receipt, bound to its folder and immutable contents."""
    def __init__(self, contents, folder):
        super().__init__(contents)
        self.verified_folder = Path(folder).resolve()
        self.verified_contents = canonical_hash(contents)


def load_frozen(folder, allow_smoke=False):
    """Verify the complete receipt before any checkpoint deserialization."""
    folder = Path(folder)
    path = folder / "train_manifest.json"
    if sha256_file(path) != (folder / "train_manifest.sha256").read_text().strip():
        raise ValueError("training manifest hash mismatch")
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    if not manifest.get("training_complete") or manifest.get("experiment") != "AF-02":
        raise ValueError("training is incomplete or belongs to another experiment")
    if manifest.get("scope") != "registered_primary":
        if not allow_smoke or manifest.get("scope") != "engineering_smoke":
            raise ValueError("only frozen primary checkpoints may be scored")
    elif config != CONFIG:
        raise ValueError("primary configuration differs from registration")
    if canonical_hash(config) != manifest["configuration_sha256"]:
        raise ValueError("configuration hash mismatch")
    if manifest.get("protocol_sha256") != REGISTERED_PROTOCOL_SHA256:
        raise ValueError("protocol pin mismatch")
    if manifest.get("dataset_access") != ["train", "dev"]:
        raise ValueError("training accessed an unregistered split")
    if manifest.get("scope") == "registered_primary" and (
            manifest.get("torch_version") != CONFIG["torch"] or manifest.get("threads") != CONFIG["threads"]):
        raise ValueError("unregistered primary runtime")
    expected_runs = [(a, s) for a in config["architectures"] for s in config["seeds"]]
    if [(r["architecture"], r["seed"]) for r in manifest["runs"]] != expected_runs:
        raise ValueError("architecture/seed metadata roster mismatch")
    expected = expected_file_roster(config)
    if set(manifest["files"]) != expected:
        raise ValueError("checkpoint file roster mismatch")
    actual = {p.name for pattern in ("*.pt", "*.npz") for p in folder.glob(pattern)}
    if actual != expected:
        raise ValueError("on-disk checkpoint file roster mismatch")
    current = source_hashes()
    if set(manifest["source_hashes"]) != set(current):
        raise ValueError("source hash roster mismatch")
    for name, value in current.items():
        if manifest["source_hashes"][name] != value:
            raise ValueError(f"frozen source differs: {name}")
    if sha256_file(Path(__file__).with_name("corpus_manifest.json")) != manifest["frozen_corpus_sha256"]:
        raise ValueError("frozen corpus receipt differs")
    for name, entry in manifest["files"].items():
        if _file_entry(folder / name) != entry:
            raise ValueError(f"checkpoint/export hash or size mismatch: {name}")
    if set(manifest["prediction_files"]) != {PREDICTION_FILE}:
        raise ValueError("train/dev prediction file roster mismatch")
    entry = manifest["prediction_files"][PREDICTION_FILE]
    if _file_entry(folder / PREDICTION_FILE) != {k: entry[k] for k in ("sha256", "bytes")}:
        raise ValueError("train/dev prediction hash or size mismatch")
    rows = [json.loads(line) for line in (folder / PREDICTION_FILE).read_text().splitlines()]
    canonical = {split: gym.canonical_cases(split) for split in ("train", "dev")}
    summaries = validate_corpus_inputs(Path(__file__).with_name("corpus_manifest.json"),
                                       {split: _Data(cases) for split, cases in canonical.items()})
    if manifest.get("datasets") != summaries:
        raise ValueError("frozen fitting-input summaries differ from corpus")
    expected_rows = {(a, s, arm, split, c.canonical_id)
                     for a in config["architectures"] for s in config["seeds"] for arm in config["arms"]
                     for split, cases in canonical.items() for c in cases}
    keys = [(r["architecture"], r["seed"], r["arm"], r["split"], r["canonical_id"]) for r in rows]
    if len(rows) != entry["rows"] or len(keys) != len(expected_rows) or set(keys) != expected_rows:
        raise ValueError("train/dev prediction row roster mismatch")
    return _VerifiedManifest(manifest, folder)


def _load_weights(folder, architecture, seed, arm, manifest):
    name = checkpoint_name(architecture, arm, seed)
    array_name = checkpoint_name(architecture, arm, seed, "npz")
    for selected in (name, array_name):
        if _file_entry(folder / selected) != manifest["files"][selected]:
            raise ValueError(f"selected weight file changed after verification: {selected}")
    payload = torch.load(folder / name, map_location="cpu", weights_only=True)
    if (payload.get("architecture"), payload.get("seed"), payload.get("arm")) != (architecture, seed, arm):
        raise ValueError("checkpoint identity mismatch")
    if payload.get("configuration_sha256") != manifest["configuration_sha256"]:
        raise ValueError("checkpoint configuration binding mismatch")
    if arm != "teacher":
        source_name = checkpoint_name(architecture, "teacher", seed)
        if (payload.get("source_checkpoint") != source_name or
                payload.get("source_checkpoint_sha256") != manifest["files"][source_name]["sha256"]):
            raise ValueError("codec source binding mismatch")
    state = payload["state_dict"]
    with np.load(folder / array_name, allow_pickle=False) as exported:
        if set(exported.files) != set(state):
            raise ValueError("checkpoint/array parameter roster mismatch")
        for key, tensor in state.items():
            array = exported[key]
            if tensor.dtype != torch.float32 or array.dtype != np.float32 or not np.array_equal(array, tensor.numpy()):
                raise ValueError(f"checkpoint/array value mismatch: {key}")
    return state


def load_model_arm(folder, architecture, seed, arm, manifest=None):
    folder = Path(folder)
    if manifest is None:
        manifest = load_frozen(folder)
    elif (not isinstance(manifest, _VerifiedManifest) or
          manifest.verified_folder != folder.resolve() or
          manifest.verified_contents != canonical_hash(manifest)):
        raise ValueError("supplied manifest is not this folder's unchanged verified receipt")
    if architecture not in CONFIG["architectures"] or seed not in CONFIG["seeds"] or arm not in CONFIG["arms"]:
        raise ValueError("unregistered model request")
    model = make_model(architecture)
    model.load_state_dict(_load_weights(folder, architecture, seed, "teacher", manifest), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    codec = None
    if arm != "teacher":
        codec = make_codec(architecture)
        codec.load_state_dict(_load_weights(folder, architecture, seed, arm, manifest), strict=True)
        codec.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
    return model, codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.out, args.seeds, args.threads, args.smoke)


if __name__ == "__main__":
    main()
