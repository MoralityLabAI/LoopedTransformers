"""PEF-03: preregistered frozen learned tied-attention history experiment.

The transient bit token is discarded at every visit. Only the memory
token persists through the recurrent tied attention/MLP block. Frozen
weights and source splits are audited before any heldout measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from itertools import product
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from lsa.mini_transformer_probe import MiniAttentionMlpResidualBlock

L = 8
D = 16
TRAIN_STEPS = 250
BATCH = 64
SEEDS = (17, 29)
STAY = 0.95


class BitMemoryLooper(nn.Module):
    """Two-token causal attention with input at position zero."""

    def __init__(self):
        super().__init__()
        self.bit_embedding = nn.Embedding(2, D)
        self.role = nn.Parameter(torch.randn(2, D) * 0.02)
        self.memory_init = nn.Parameter(torch.zeros(D))
        self.block = MiniAttentionMlpResidualBlock(
            D, heads=2, mlp_ratio=2, alpha=1.0, beta=0.2
        )
        self.head = nn.Linear(D, 4)

    def step(self, memory: torch.Tensor, bit_embedding: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((bit_embedding + self.role[0],
                              memory + self.role[1]), dim=1)
        return self.block(tokens)[:, 1]

    def memory(self, bits: torch.Tensor, *, storage: str = "fp32") -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != L:
            raise ValueError("expected eight chronological input bits")
        state = self.memory_init.unsqueeze(0).expand(len(bits), -1)
        for step in range(L):
            state = self.step(state, self.bit_embedding(bits[:, step]))
            if storage == "fp16":
                state = state.to(torch.float16).float()
            elif storage == "bf16":
                state = state.to(torch.bfloat16).float()
            elif storage != "fp32":
                raise ValueError("unsupported storage format")
        return state

    def logits(self, bits: torch.Tensor, *, storage: str = "fp32") -> torch.Tensor:
        return self.head(self.memory(bits, storage=storage))


def all_histories() -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=L))


def split_history(history: tuple[int, ...]) -> str:
    token = hashlib.sha256(b"pef03-v1:" + bytes(history)).digest()[0]
    return "train" if token < 160 else "probe" if token < 208 else "heldout"


def split_catalog() -> dict[str, list[tuple[int, ...]]]:
    output = {"train": [], "probe": [], "heldout": []}
    for history in all_histories():
        output[split_history(history)].append(history)
    if any(not v for v in output.values()) or sum(map(len, output.values())) != 256:
        raise AssertionError("invalid source history partition")
    if sum(len(set(v)) for v in output.values()) != 256:
        raise AssertionError("overlapping split")
    return output


def tensor(histories) -> torch.Tensor:
    return torch.tensor(histories, dtype=torch.long)


def labels(bits: torch.Tensor) -> torch.Tensor:
    return 2 * bits[:, -2] + bits[:, -1]


def hash_weights(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def train_frozen(seed: int, checkpoint: Path, *, steps: int = TRAIN_STEPS) -> dict:
    torch.manual_seed(seed)
    model = BitMemoryLooper().train()
    before = hash_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    pool = tensor(split_catalog()["train"])
    generator = torch.Generator().manual_seed(seed * 7919 + 59)
    first = last = None
    for step in range(steps):
        bits = pool[torch.randint(len(pool), (BATCH,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.logits(bits), labels(bits))
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite training loss")
        if step == 0:
            first = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = float(loss.detach())
    model.eval()
    end = hash_weights(model)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "steps": steps,
                "init_hash": before, "final_hash": end}, checkpoint)
    return {"train_seed": seed, "optimizer_steps": steps,
            "initial_loss": first, "final_loss": last,
            "initial_weights_sha256": before, "frozen_weights_sha256": end,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_filename": checkpoint.name}


def load_frozen(checkpoint: Path, seed: int) -> BitMemoryLooper:
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if int(data["seed"]) != seed:
        raise AssertionError("wrong checkpoint seed")
    model = BitMemoryLooper()
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    if hash_weights(model) != data["final_hash"]:
        raise AssertionError("frozen parameter checksum mismatch")
    return model


def fit_ridge(x: torch.Tensor, y: torch.Tensor, penalty: float = 1.0):
    values = x.double()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-4)
    z = (values - mean) / std
    with_bias = torch.cat((z, torch.ones((len(z), 1), dtype=torch.float64)), dim=1)
    regularizer = torch.eye(with_bias.shape[1], dtype=torch.float64) * penalty
    regularizer[-1, -1] = 0.0
    w = torch.linalg.solve(
        with_bias.T @ with_bias + regularizer, with_bias.T @ y.double()
    )
    return mean, std, w


def predict_ridge(x: torch.Tensor, fitted):
    mean, std, weights = fitted
    z = (x.double() - mean) / std
    with_bias = torch.cat((z, torch.ones((len(z), 1), dtype=torch.float64)), dim=1)
    return ((with_bias @ weights) >= 0.5).long()


@torch.no_grad()
def observational_holdout(model: BitMemoryLooper, catalog: dict) -> dict:
    train_x, probe_x, heldout_x = (
        tensor(catalog[x]) for x in ("train", "probe", "heldout")
    )
    probe_h = model.memory(probe_x)
    heldout_h = model.memory(heldout_x)
    probes = []
    for i in range(L):
        prediction = predict_ridge(
            heldout_h, fit_ridge(probe_h, probe_x[:, i])
        )
        zero = int((heldout_x[:, i] == 0).sum())
        one = len(heldout_x) - zero
        probes.append({
            "history_position": i,
            "heldout_accuracy": float((prediction == heldout_x[:, i]).float().mean()),
            "heldout_majority_baseline": max(zero, one) / len(heldout_x),
            "probe_training_histories": len(probe_x),
            "heldout_histories": len(heldout_x),
        })
    return {
        "parameter_training_accuracy": float(
            (model.logits(train_x).argmax(1) == labels(train_x)).float().mean()
        ),
        "heldout_head_accuracy": float(
            (model.logits(heldout_x).argmax(1) == labels(heldout_x)).float().mean()
        ),
        "random_four_class_baseline": 0.25,
        "history_bit_probes": probes,
    }


def markov_stream(count: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    histories = []
    for _ in range(count):
        h = [rng.randrange(2)]
        for _ in range(1, L):
            h.append(h[-1] if rng.random() < STAY else 1 - h[-1])
        histories.append(h)
    return tensor(histories)


@torch.no_grad()
def prior_shift(model: BitMemoryLooper, seed: int) -> dict:
    xtr = markov_stream(1024, seed * 3001 + 70)
    xte = markov_stream(1024, seed * 3001 + 71)
    ztr, zte = model.memory(xtr), model.memory(xte)
    probes = []
    for i in (0, 3):
        fitted = fit_ridge(ztr, xtr[:, i])
        guesses = predict_ridge(zte, fitted)
        recent = xte[:, 6]
        probes.append({
            "history_position": i,
            "observational_ridge_accuracy": float((guesses == xte[:, i]).float().mean()),
            "test_recent_bit_only_accuracy": float((recent == xte[:, i]).float().mean()),
            "exact_recent_bit_only_bayes_rate": (1 + (2 * STAY - 1)**(6-i)) / 2,
            "distinct_probe_train_histories": len({tuple(row) for row in xtr.tolist()}),
            "distinct_probe_test_histories": len({tuple(row) for row in xte.tolist()}),
            "train_test_histories_may_overlap": True,
        })
    return {
        "persistence": STAY, "source_probe_train_size": len(xtr),
        "source_eval_size": len(xte), "older_bit_probes": probes,
        "source_shift_is_not_history_disjoint": True,
    }


@torch.no_grad()
def intervention_audit(model: BitMemoryLooper, catalog: dict) -> dict:
    histories = catalog["heldout"]
    x = tensor(histories)
    out = {}
    for storage in ("fp32", "fp16", "bf16"):
        original = model.memory(x, storage=storage)
        logits0 = model.head(original)
        reference_classes = logits0.argmax(1)
        table = []
        for pos in range(L):
            edited = x.clone()
            edited[:, pos] = 1 - edited[:, pos]
            if not (torch.equal(edited[:, :pos], x[:, :pos]) and
                    torch.equal(edited[:, pos+1:], x[:, pos+1:])):
                raise AssertionError("intervention changed uncontrolled history positions")
            changed = model.memory(edited, storage=storage)
            logits1 = model.head(changed)
            effects = (changed - original).norm(dim=-1)
            table.append({
                "position": pos,
                "mean_hidden_l2": float(effects.mean()),
                "min_hidden_l2": float(effects.min()),
                "max_hidden_l2": float(effects.max()),
                "mean_logit_l2": float((logits1 - logits0).norm(dim=-1).mean()),
                "head_class_change_rate": float((logits1.argmax(1) != reference_classes).float().mean()),
                "counterfactuals_outside_heldout": sum(
                    split_history(tuple(row)) != "heldout" for row in edited.tolist()
                ),
                "intervention_pairs": len(histories),
            })
        out[storage] = table
    return out


def ordered_jacobians(model: BitMemoryLooper, history: tuple[int, ...]) -> dict:
    """Full local 16x16 memory Jacobians and ordered input-tangent products."""
    model.eval()
    bits = tensor([history])[0]
    states = [model.memory_init.detach().clone()]
    emb = [model.bit_embedding(bits[i]).detach().clone() for i in range(L)]
    direction = (model.bit_embedding.weight[1] -
                 model.bit_embedding.weight[0]).detach()
    matrices, local = [], []
    with torch.enable_grad():
        for i in range(L):
            before = states[-1].detach()
            e = emb[i]
            def one_step(h):
                return model.step(h.unsqueeze(0), e.unsqueeze(0))[0]
            J = torch.autograd.functional.jacobian(one_step, before, vectorize=False)
            _, tangent = torch.autograd.functional.jvp(
                lambda v: model.step(before.unsqueeze(0), v.unsqueeze(0))[0],
                e, direction, create_graph=False
            )
            with torch.no_grad():
                nxt = one_step(before).detach()
            states.append(nxt)
            matrices.append(J.detach())
            local.append(tangent.detach())

        reports = []
        for i in range(L):
            product_j = torch.eye(D)
            loose_spectral_bound = 1.0
            for later in range(i+1, L):
                product_j = matrices[later] @ product_j
                loose_spectral_bound *= float(torch.linalg.svdvals(matrices[later]).max())
            predicted = product_j @ local[i]
            def full_recurrence(v):
                state = model.memory_init.detach().clone().unsqueeze(0)
                for visit in range(L):
                    chosen = v if visit == i else emb[visit]
                    state = model.step(state, chosen.unsqueeze(0))
                return state[0]
            _, direct = torch.autograd.functional.jvp(
                full_recurrence, emb[i], direction, create_graph=False
            )
            error = float((predicted-direct).norm() / direct.norm().clamp_min(1e-8))
            if float(direct.norm()) > 1e-8 and error > 0.05:
                raise AssertionError(f"causal Jacobian chain mismatch at {i}: {error}")
            singular = torch.linalg.svdvals(product_j)
            reports.append({
                "history_position": i,
                "local_bit_embedding_jvp_norm": float(local[i].norm()),
                "ordered_suffix_top_singular": float(singular.max()),
                "ordered_suffix_bottom_singular": float(singular.min()),
                "product_of_step_operator_norms_upper_bound": loose_spectral_bound,
                "ordered_composed_input_jvp_norm": float(predicted.norm()),
                "full_recurrence_direct_jvp_norm": float(direct.norm()),
                "relative_jvp_chain_error": error,
            })
    return {
        "reference_history": list(history), "memory_dimension": D,
        "position_derivatives": reports,
        "local_jvp_is_not_a_discrete_flip": True,
    }


def run(seed: int, output_dir: Path, *, steps: int = TRAIN_STEPS,
        jacobian: bool = True) -> dict:
    catalog = split_catalog()
    checkpoint = output_dir / f"seed{seed}.frozen.pt"
    training = train_frozen(seed, checkpoint, steps=steps)
    model = load_frozen(checkpoint, seed)
    immutable_hash = hash_weights(model)
    with torch.no_grad():
        heldout = observational_holdout(model, catalog)
        shifted = prior_shift(model, seed)
        causal = intervention_audit(model, catalog)
    jac = ordered_jacobians(model, catalog["heldout"][0]) if jacobian else None
    if hash_weights(model) != immutable_hash:
        raise AssertionError("evaluation mutated model parameters")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != training["checkpoint_sha256"]:
        raise AssertionError("evaluation changed the frozen checkpoint")
    torch.manual_seed(seed + 10000)
    control = BitMemoryLooper().eval()
    with torch.no_grad():
        x = tensor(catalog["heldout"])
        random_accuracy = float((control.logits(x).argmax(1) == labels(x)).float().mean())
    report = {
        "seed": seed, "training": training,
        "source_split": "first SHA256 byte of pef03-v1: plus 8 history bytes",
        "split_sizes": {name: len(rows) for name, rows in catalog.items()},
        "frozen_checkpoint_integrity_passed": True,
        "train_steps_follow_preregistration": steps == TRAIN_STEPS,
        "heldout_task_success_threshold": 0.85,
        "random_init_head_accuracy": random_accuracy,
        "task_and_uniform_history_probes": heldout,
        "markov_prior_diagnostic": shifted,
        "causal_discrete_bit_interventions": causal,
        "ordered_local_jacobians": jac,
        "readout_fp16_bf16_are_storage_casts_not_full_low_precision_compute": True,
        "trained_on_synthetic_8_bit_task_not_a_pretrained_language_model": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"seed{seed}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/pef03"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    seeds = (args.seed,) if args.seed else SEEDS
    results = [run(seed, args.out) for seed in seeds]
    summary = {
        "campaign": "PEF-03",
        "seeds": list(seeds),
        "task_success_both_seeds": len(results) == 2 and all(
            row["task_and_uniform_history_probes"]["heldout_head_accuracy"] >= 0.85
            for row in results
        ),
        "runs": [{
            "seed": row["seed"],
            "checkpoint_sha256": row["training"]["checkpoint_sha256"],
            "training_loss_final": row["training"]["final_loss"],
            "heldout_head_accuracy": row["task_and_uniform_history_probes"]["heldout_head_accuracy"],
            "random_init_head_accuracy": row["random_init_head_accuracy"],
            "oldest_uniform_heldout_linear_probe": row["task_and_uniform_history_probes"]["history_bit_probes"][0]["heldout_accuracy"],
            "oldest_causal_hidden_l2": row["causal_discrete_bit_interventions"]["fp32"][0]["mean_hidden_l2"],
            "newest_causal_hidden_l2": row["causal_discrete_bit_interventions"]["fp32"][-1]["mean_hidden_l2"],
            "oldest_correlated_probe": row["markov_prior_diagnostic"]["older_bit_probes"][0]["observational_ridge_accuracy"],
            "correlated_recent_bit_baseline": row["markov_prior_diagnostic"]["older_bit_probes"][0]["test_recent_bit_only_accuracy"],
            "jacobian_max_chain_error": max(
                step["relative_jvp_chain_error"]
                for step in row["ordered_local_jacobians"]["position_derivatives"]
            ),
        } for row in results],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
