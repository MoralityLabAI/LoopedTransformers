"""AF-23 Track B: eight-seed source-pinned 6036-weight TypedLoop training.

Only the preregistered train split supplies shortest-cost and optimal-action
targets. Development is measured at the fixed final update; it never selects
a checkpoint. Held-out data are excluded from optimization and fit decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.activation_funnel_af19.model import ReadoutPolicy
from . import branching_gym as gym

HERE = Path(__file__).resolve().parent
STEPS = 240
BATCH = 32
SEEDS = gym.SEEDS
SOURCE_FILES = [
    "PROTOCOL.json", "branching_gym.py", "branching_planner.py",
    "branching_train.py", "branching_evaluate.py", "branching_audit.py",
    "tests/test_af23.py", "README.md",
    "../../.github/workflows/activation-funnel-af23-track-b.yml",
    "../activation_funnel_af19/model.py",
    "../activation_funnel_af17/model.py",
    "../activation_funnel_af05/gym.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {n: digest(HERE / n) for n in SOURCE_FILES}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def training_states(seed):
    """Deterministic train-only roots and legal successor states."""
    rng = random.Random(230079 + seed)
    examples = []
    for c in gym.cases_for("train"):
        state = gym.initial_facts(c)
        examples.append((c, state))
        # Sample two genuine host-legal successor states, including detours.
        for _ in range(2):
            legal = gym.enabled(c, state)
            if not legal:
                break
            action = rng.choice(legal)
            state, _ = gym.sound_fire(c, state, action)
            examples.append((c, state))
            if gym.GOAL in state:
                break
    return tuple(examples)


def supervised_loss(model, batch):
    cs = [c for c, s in batch]
    states = [s for c, s in batch]
    x, known, legal, optimal = gym.observations(cs, states, with_labels=True)
    raw, predicted_d, predicted_reach = model.full(x, known)
    if not bool((optimal & legal).any(dim=-1).all()):
        raise ValueError("training state lacks a host-optimal legal target")
    masked = raw.masked_fill(~legal, -1e4)
    log_probs = F.log_softmax(masked, dim=-1)
    good_log_mass = torch.logsumexp(log_probs.masked_fill(~optimal, -1e4), dim=-1)
    action_loss = -good_log_mass.mean()
    true_d = [gym.shortest_steps(c, frozenset(s)) for c, s in batch]
    target_reach = torch.tensor([float(d is not None) for d in true_d])
    target_dist = torch.tensor([float(min(d, 16) if d is not None else 16)
                                for d in true_d])
    reach_loss = F.binary_cross_entropy_with_logits(predicted_reach, target_reach)
    distance_loss = F.smooth_l1_loss(predicted_d, target_dist)
    total = action_loss + .5 * reach_loss + .1 * distance_loss
    return total, {
        "optimal_action_loss": float(action_loss.detach()),
        "reach_bce": float(reach_loss.detach()),
        "distance_huber": float(distance_loss.detach()),
    }


def first_action_fit(model, split, limit=192):
    """Training/development-only diagnostic, not a model selection criterion."""
    from . import branching_planner
    cases = gym.cases_for(split)[:limit]
    cases = [c for c in cases if c.provable_design]
    optimal = 0
    premature_stop = 0
    for start in range(0, len(cases), 32):
        cs = cases[start:start+32]
        ss = [gym.initial_facts(c) for c in cs]
        plans = branching_planner.decisions(model, cs, ss)
        for c, s, plan in zip(cs, ss, plans):
            pick = plan["selected"]["direct"]
            optimal += int(pick in gym.optimal_actions(c, s))
            premature_stop += int(pick == gym.STOP)
    return {"cases": len(cases), "initial_optimal": optimal,
            "initial_premature_stop": premature_stop,
            "optimal_action_rate": optimal / len(cases) if cases else 0.}


def train(seed, out_dir, smoke=False):
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("training output must be fresh")
    out_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(230000 + seed)
    model = ReadoutPolicy("ancestor", 4)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 6036:
        raise ValueError("AF23 network parameter count changed")
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = random.Random(231000 + seed)
    examples = training_states(seed)
    if len(examples) < 768:
        raise ValueError("incomplete train-only host states")
    steps = 2 if smoke else STEPS
    gradients = None
    logs = []
    start = time.perf_counter()
    for update in range(steps):
        batch = [examples[rng.randrange(len(examples))] for _ in range(BATCH)]
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, parts = supervised_loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite AF23 training loss")
        loss.backward()
        if gradients is None:
            gradients = sum(float(p.grad.square().sum().detach())
                            for p in model.cell.parameters()
                            if p.grad is not None) ** .5
            if gradients <= 0:
                raise RuntimeError("native recurrent cell has zero gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=True)
        opt.step()
        if update % 60 == 0 or update == steps - 1:
            logs.append({"update": update, "loss": float(loss.detach()), **parts})
    pkg = {"experiment": "AF-23", "track": "B", "seed": seed,
           "parameters": parameters, "updates": steps, "smoke": smoke,
           "state_dict": model.state_dict()}
    torch.save(pkg, out_dir / "policy.pt")
    np.savez(out_dir / "policy.npz", **{
        n: v.detach().numpy() for n, v in model.state_dict().items()
    })
    model.eval()
    receipt = {
        "experiment": "AF-23", "track": "B", "seed": seed,
        "smoke": smoke, "steps": steps, "parameters": parameters,
        "framework": {"torch": torch.__version__, "numpy": np.__version__},
        "source_sha256": source_hashes(),
        "protocol_sha256": digest(HERE / "PROTOCOL.json"),
        "data_manifest_sha256": gym.manifest()["manifest_sha256"],
        "weights_sha256": {ext: digest(out_dir / f"policy.{ext}")
                            for ext in ("pt", "npz")},
        "train_cases": len(gym.cases_for("train")),
        "train_sampled_host_states": len(examples),
        "training_loss": logs,
        "native_cell_initial_gradient_norm": gradients,
        "fit": {
            "train_first_action": first_action_fit(model, "train"),
            "dev_first_action": first_action_fit(model, "dev"),
        },
        "checkpoint_selection": "fixed_final_update",
        "elapsed_sec": time.perf_counter() - start,
        "heldout_scoring_used_for_training": False,
    }
    write_json(out_dir / "train_receipt.json", receipt)
    (out_dir / "train_receipt.sha256").write_text(
        digest(out_dir / "train_receipt.json") + "\n"
    )
    return receipt


def load_model(folder, seed, smoke=False):
    folder = Path(folder)
    if not (folder / "train_receipt.sha256").is_file():
        raise ValueError("source-bound AF23 training receipt missing")
    if digest(folder / "train_receipt.json") != (
        folder / "train_receipt.sha256"
    ).read_text().strip():
        raise ValueError("training receipt SHA-256 mismatch")
    receipt = json.loads((folder / "train_receipt.json").read_text())
    if (receipt["experiment"], receipt["track"], receipt["seed"],
            receipt["smoke"]) != ("AF-23", "B", seed, smoke):
        raise ValueError("seed or experiment identity mismatch")
    if (receipt["source_sha256"] != source_hashes()
            or receipt["protocol_sha256"] != digest(HERE / "PROTOCOL.json")
            or receipt["data_manifest_sha256"] != gym.manifest()["manifest_sha256"]):
        raise ValueError("AF23 scoring source drift")
    for ext, expected in receipt["weights_sha256"].items():
        if digest(folder / f"policy.{ext}") != expected:
            raise ValueError("network checkpoint hash mismatch")
    pkg = torch.load(folder / "policy.pt", weights_only=True, map_location="cpu")
    if (pkg["experiment"], pkg["track"], pkg["seed"], pkg["smoke"]) != (
            "AF-23", "B", seed, smoke):
        raise ValueError("checkpoint metadata mismatch")
    if pkg["updates"] != receipt["steps"] or pkg["parameters"] != 6036:
        raise ValueError("wrong model/optimizer update count")
    with np.load(folder / "policy.npz", allow_pickle=False) as arr:
        if set(arr.files) != set(pkg["state_dict"]):
            raise ValueError("named checkpoints are incomplete")
        for key, tensor in pkg["state_dict"].items():
            if not np.array_equal(arr[key], tensor.detach().numpy()):
                raise ValueError("Torch/NumPy weight mismatch")
    model = ReadoutPolicy("ancestor", 4)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    return receipt, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=17)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    receipt = train(args.seed, args.out_dir, args.smoke)
    print(json.dumps({"seed": args.seed, "updates": receipt["steps"],
                      "train_first_action": receipt["fit"]["train_first_action"],
                      "dev_first_action": receipt["fit"]["dev_first_action"]}))


if __name__ == "__main__":
    main()
