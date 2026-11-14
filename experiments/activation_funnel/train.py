"""Train AF-01 sources/codecs using train and dev only; freeze file hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import gym
from .model import (CODE_WIDTH, LEARNED_ARMS, CompactTransition, FixedCodec,
                    LearnedCodec, RecurrentTeacher, audit_loss, fit_basis,
                    composite_audits, paired_continuation, task_from_audits,
                    commutation_loss, resources, matched_donors)

CONFIG = {
    "experiment": "AF-01", "architecture": "new standalone GRU interface-memory pilot",
    "seeds": [17, 29, 43], "input_dim": 9, "teacher_width": 32,
    "code_width": 4, "coordinate_bits": 32, "calls_per_observation": 2,
    "teacher_steps": 400, "codec_steps": 300, "batch_size": 64,
    "learning_rate": .003, "optimizer": "AdamW", "weight_decay": .01,
    "gradient_clip": 1., "teacher_audit_weight": .5,
    "task_distillation_kl_weight": .25, "reconstruction_weight": .5,
    "commutation_weight": .5, "paired_task_weight": .25,
    "audit_weight": .5, "paired_audit_weight": .25,
    "compact_hidden_weight": .25, "source_qualification": .95,
    "selection": "fixed final checkpoint; dev diagnostic only",
    "codecs": ["task", "reconstruction", "causal", "joint", "pca", "random"],
    "optional_registered_arm": "compact_transition",
    "train_count": 144, "dev_count": 36,
    "split_access": ["train", "dev"],
}

REGISTERED_PROTOCOL_SHA256 = "5d3880f00f73a2e5cb6fd733d5013889418db9b3fbf1eb08a6cc0cd67cc4b22a"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    folder = Path(__file__).parent
    return {name: sha256_file(folder / name) for name in
            ("gym.py", "model.py", "train.py", "evaluate.py", "PROTOCOL.json")
            if (folder / name).exists()}


def tensor_data(episodes):
    x = torch.tensor([gym.feature_sequence(e) for e in episodes], dtype=torch.float32)
    task = torch.tensor([gym.target_action(e) for e in episodes], dtype=torch.long)
    audit = torch.tensor([gym.audit_targets(e) for e in episodes], dtype=torch.long)
    return x, task, audit


def accuracy(trace, task, audit, inputs):
    ap = trace.audit_logits.argmax(-1)
    task_ok = trace.task_logits.argmax(-1) == task
    audit_ok = ap == audit
    first = {}
    for i, row in enumerate(inputs.tolist()):
        first.setdefault(canonical_hash(row), i)
    ix = torch.tensor(list(first.values()), dtype=torch.long, device=task.device)
    def metrics(indices):
        return {"task_accuracy": float(task_ok[indices].float().mean()),
                "audit_cell_accuracy": float(audit_ok[indices].float().mean()),
                "audit_branch_accuracy": audit_ok[indices].float().mean(0).tolist(),
                "audit_exact_accuracy": float(audit_ok[indices].all(-1).float().mean())}
    result = metrics(ix)
    result["exact_feature_patterns"] = len(ix)
    result["cases"] = len(inputs)
    result["row_weighted_diagnostic"] = metrics(torch.arange(len(inputs), device=task.device))
    return result


def public_input_summary(x):
    return {"examples": len(x), "unique_public_inputs": len({canonical_hash(row) for row in x.tolist()}),
            "input_sha256": canonical_hash(x.tolist())}


def save_checkpoint(out, name, value, entries):
    path = out / name
    torch.save(value, path)
    entries[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def common_task_loss(student, teacher, target):
    return (F.cross_entropy(student.task_logits, target) +
            CONFIG["task_distillation_kl_weight"] *
            F.kl_div(F.log_softmax(student.task_logits, -1),
                     F.softmax(teacher.task_logits.detach(), -1), reduction="batchmean"))


def train_seed(seed, train_data, dev_data, out, config, files):
    torch.manual_seed(seed)
    random.seed(seed)
    x, task, audit = train_data
    dx, dt, da = dev_data
    teacher = RecurrentTeacher()
    opt = torch.optim.AdamW(teacher.parameters(), lr=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    rng = torch.Generator().manual_seed(seed + 1000)
    began = time.perf_counter()
    losses = []
    for step in range(config["teacher_steps"]):
        ix = torch.randint(len(x), (config["batch_size"],), generator=rng)
        trace = teacher.rollout(x[ix])
        loss = F.cross_entropy(trace.task_logits, task[ix]) + .5 * audit_loss(trace.audit_logits, audit[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.)
        opt.step()
        if step % 100 == 0 or step == config["teacher_steps"] - 1:
            losses.append([step, float(loss.detach())])
    teacher_elapsed = time.perf_counter() - began
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
        p.grad = None
    with torch.no_grad():
        source_trace = teacher.rollout(x)
        dev_trace = teacher.rollout(dx)
        source_train = accuracy(source_trace, task, audit, x)
        source_dev = accuracy(dev_trace, dt, da, dx)
        basis_started = time.perf_counter()
        mean, pca, random_basis = fit_basis(source_trace.states, CODE_WIDTH, seed)
        basis_elapsed = time.perf_counter() - basis_started
    qualification = all(scores["task_accuracy"] >= config["source_qualification"] and
                        all(value >= config["source_qualification"] for value in scores["audit_branch_accuracy"])
                        for scores in (source_train, source_dev))
    save_checkpoint(out, f"teacher_s{seed}.pt", {"state_dict": teacher.state_dict()}, files)
    meta = {"seed": seed, "teacher_train": source_train, "teacher_dev": source_dev,
            "source_qualified": qualification, "teacher_losses": losses,
            "teacher_training_seconds": teacher_elapsed,
            "train_only_basis_fit_seconds": basis_elapsed,
            "teacher_resources": resources(teacher), "arms": {}}
    print(json.dumps({"seed": seed, "teacher": source_dev,
                      "source_qualified": qualification}), flush=True)

    for arm, basis in (("pca", pca), ("random", random_basis)):
        codec = FixedCodec(mean, basis)
        save_checkpoint(out, f"{arm}_s{seed}.pt",
                        {"kind": "fixed", "state_dict": codec.state_dict()}, files)
        with torch.no_grad():
            result = accuracy(teacher.rollout(dx, codec), dt, da, dx)
        meta["arms"][arm] = {"dev": result, "training_seconds": 0.,
                               "resources": resources(teacher, codec),
                               "fit_scope": "source train activations only"}

    for arm in LEARNED_ARMS:
        torch.manual_seed(seed + 5000)
        codec = LearnedCodec()
        codec.initialize_from_basis(mean, pca)
        opt = torch.optim.AdamW(codec.parameters(), lr=config["learning_rate"],
                               weight_decay=config["weight_decay"])
        rng = torch.Generator().manual_seed(seed + 2000)
        losses = []
        began = time.perf_counter()
        for step in range(config["codec_steps"]):
            ix = torch.randint(len(x), (config["batch_size"],), generator=rng)
            bx, bt, ba = x[ix], task[ix], audit[ix]
            donor = matched_donors(bx, rng)
            with torch.no_grad():
                native = teacher.rollout(bx)
            student = teacher.rollout(bx, codec)
            loss = common_task_loss(student, native, bt)
            if arm == "reconstruction":
                loss = loss + .5 * F.mse_loss(student.states, native.states)
            if arm in ("causal", "joint"):
                loss = loss + .5 * commutation_loss(teacher, codec, bx, native.states)
                mixed = composite_audits(bx, ba, donor)
                paired = paired_continuation(teacher, bx, donor, codec)
                loss = loss + .25 * F.cross_entropy(paired.task_logits, task_from_audits(mixed))
            if arm == "joint":
                loss = loss + .5 * audit_loss(student.audit_logits, ba)
                loss = loss + .25 * audit_loss(paired.audit_logits, mixed)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.)
            opt.step()
            if step % 100 == 0 or step == config["codec_steps"] - 1:
                losses.append([step, float(loss.detach())])
        codec.eval()
        save_checkpoint(out, f"{arm}_s{seed}.pt",
                        {"kind": "learned", "state_dict": codec.state_dict()}, files)
        with torch.no_grad():
            result = accuracy(teacher.rollout(dx, codec), dt, da, dx)
        meta["arms"][arm] = {"dev": result, "losses": losses,
                               "training_seconds": time.perf_counter() - began,
                               "resources": resources(teacher, codec)}
        print(json.dumps({"seed": seed, "arm": arm, "dev": result}), flush=True)

    torch.manual_seed(seed + 6000)
    compact = CompactTransition(teacher)
    opt = torch.optim.AdamW([p for p in compact.parameters() if p.requires_grad],
                           lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = torch.Generator().manual_seed(seed + 2000)
    losses = []
    began = time.perf_counter()
    for step in range(config["codec_steps"]):
        ix = torch.randint(len(x), (config["batch_size"],), generator=rng)
        bx, bt, ba = x[ix], task[ix], audit[ix]
        matched_donors(bx, rng)  # preserve the registered minibatch stream
        with torch.no_grad():
            native = teacher.rollout(bx)
        student = compact.rollout(bx)
        loss = F.cross_entropy(student.task_logits, bt) + .5 * audit_loss(student.audit_logits, ba)
        loss = loss + .25 * F.mse_loss(student.states, native.states)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compact.parameters(), 1.)
        opt.step()
        if step % 100 == 0 or step == config["codec_steps"] - 1:
            losses.append([step, float(loss.detach())])
    compact.eval()
    save_checkpoint(out, f"compact_transition_s{seed}.pt", {"state_dict": compact.state_dict()}, files)
    with torch.no_grad():
        result = accuracy(compact.rollout(dx), dt, da, dx)
    meta["arms"]["compact_transition"] = {
        "dev": result, "losses": losses, "training_seconds": time.perf_counter() - began,
        "resources": resources(compact), "source_gru_at_inference": False}
    print(json.dumps({"seed": seed, "arm": "compact_transition", "dev": result}), flush=True)
    return meta


def train(out, seeds=None, threads=2, smoke=False):
    protocol = Path(__file__).with_name("PROTOCOL.json")
    if sha256_file(protocol) != REGISTERED_PROTOCOL_SHA256:
        raise ValueError("protocol differs from the remotely frozen AF-01 registration")
    if out.exists() and any(out.iterdir()):
        raise ValueError("training output must be empty; preserve earlier frozen checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    if not smoke and str(torch.__version__) != "2.5.1+cpu":
        raise ValueError("primary AF-01 requires the registered torch 2.5.1+cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    config = dict(CONFIG)
    config["seeds"] = list(CONFIG["seeds"] if seeds is None else seeds)
    if not smoke and config["seeds"] != CONFIG["seeds"]:
        raise ValueError("primary training requires every registered seed in order")
    if smoke:
        config["teacher_steps"], config["codec_steps"] = 8, 4
    train_eps = gym.generate_split("train", count=config["train_count"])
    dev_eps = gym.generate_split("dev", count=config["dev_count"])
    train_data, dev_data = tensor_data(train_eps), tensor_data(dev_eps)
    manifest = {"scope": "engineering_smoke" if smoke else "registered_primary",
                "configuration": config, "configuration_sha256": canonical_hash(config),
                "source_hashes": source_hashes(), "torch_version": str(torch.__version__),
                "threads": threads, "dataset_access": ["train", "dev"],
                "train": public_input_summary(train_data[0]),
                "dev": public_input_summary(dev_data[0]),
                "dataset_sha256": {"train": canonical_hash([e.to_dict() for e in train_eps]),
                                   "dev": canonical_hash([e.to_dict() for e in dev_eps])},
                "statistical_scope": "finite registered input patterns and three training seeds; renamed theorem variants are not independent neural observations",
                "files": {}, "seeds": []}
    for seed in config["seeds"]:
        manifest["seeds"].append(train_seed(seed, train_data, dev_data, out, config, manifest["files"]))
    manifest["all_sources_qualified"] = all(m["source_qualified"] for m in manifest["seeds"])
    manifest["training_complete"] = True
    (out / "train_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out / "train_manifest.sha256").write_text(sha256_file(out / "train_manifest.json") + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.out, args.seeds, args.threads, args.smoke)


if __name__ == "__main__":
    main()
