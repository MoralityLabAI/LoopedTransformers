"""Fit LDT and TRM prompt-only expert selectors from native routing receipts."""
import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_lattice import ExpertLattice
from gym_manifest import sha
from matrix_cases import cases_and_labels, digest, jsonl, render
from stage1_data import write_json
from stage1_memory import MemoryGuard

PHASES = ("prefill", "decode")
LAYERS, EXPERTS, BUCKETS = 24, 128, 128
CAPACITIES = (8, 16, 32)


def features(case):
    """Causal lexical evidence from the available prompt parts."""
    x = np.zeros(3 * BUCKETS, dtype=np.float32)
    for block, part in enumerate((case["prefix"], case["suffix"], render(case))):
        pieces = re.findall(r"\w+|[^\w\s]", part.casefold())
        start = block * BUCKETS
        buckets = x[start:start + BUCKETS]
        for i, word in enumerate(pieces):
            for feature in (word, pieces[i - 1] + "\n" + word if i else "<start>\n" + word):
                code = int.from_bytes(hashlib.blake2s(feature.encode(), digest_size=4).digest(), "little")
                buckets[code % BUCKETS] += 1
        buckets[:] = np.sqrt(buckets) / np.sqrt(max(1.0, buckets.sum()))
    return x


class PromptController(nn.Module):
    """Matched LDT mesh and TRM-style high/low recurrence with a shared state update."""

    def __init__(self, architecture, width=32, rounds=3, log_prior=None):
        super().__init__()
        if architecture not in ("ldt", "trm"):
            raise ValueError("unknown controller")
        self.architecture = architecture
        self.rounds = rounds
        self.input = nn.Linear(3 * BUCKETS, width)
        self.layer = nn.Embedding(LAYERS, width)
        self.phase = nn.Embedding(len(PHASES), width)
        self.message = nn.Linear(3 * width, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, EXPERTS)
        self.register_buffer("log_prior", log_prior)
        if log_prior is not None:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        state = checkpoint["state"]
        model = cls(checkpoint["architecture"], log_prior=state.get("log_prior"))
        model.load_state_dict(state)
        return model.eval()

    def update(self, current, first, second, third):
        return self.norm(current + F.gelu(self.message(torch.cat((first, second, third), dim=-1))))

    def forward(self, x):
        evidence = torch.tanh(self.input(x)[:, None] + self.layer.weight)
        if self.architecture == "ldt":
            blocks = evidence.reshape(len(x), 4, 6, -1).mean(dim=2)
            block = blocks.repeat_interleave(6, dim=1)
            mesh = blocks.mean(dim=1, keepdim=True).expand_as(evidence)
            high = self.update(evidence, evidence, block, mesh)
        else:
            high = torch.zeros_like(evidence)
            low = torch.zeros_like(evidence)
            for _ in range(self.rounds):
                for _ in range(2):
                    low = self.update(low, low, high, evidence)
                high = self.update(high, high, low, evidence)
        phases = high[:, None] + self.phase.weight[None, :, None]
        logits = self.output(phases)
        if self.log_prior is not None:
            logits = self.log_prior + 0.5 * torch.tanh(logits)
        return logits


def load_targets(cases, source, composition_hash=None):
    by_id = {case["id"]: case for case in cases}
    records = {}
    provenance = set()
    for row in jsonl(source):
        case_id = row["case_id"]
        if case_id not in by_id or case_id in records or row["prompt_sha256"] != digest(render(by_id[case_id])):
            raise ValueError(f"routing receipt does not match one unique prompt: {case_id}")
        if row["trace_kind"] != "llama_cpp_inkling2_native":
            raise ValueError("expected native Inkling Q4 teacher routing")
        if row.get("published_composition_sha256") != composition_hash:
            raise ValueError("routing receipt identifies a different published teacher composition")
        provenance.add((row["model_sha256"], row["runtime_revision"]))
        targets = []
        for phase in PHASES:
            data = row["phases"][phase]
            mass = np.asarray(data["mass"], dtype=np.float32)
            visits = np.asarray(data["visits"])
            expected = np.ones(LAYERS, dtype=int) * data["tokens"]
            expected[6:18] *= 3
            if (mass.shape != (LAYERS, EXPERTS) or visits.shape != (LAYERS,)
                    or not np.isfinite(mass).all() or np.any(mass < 0) or data["tokens"] <= 0
                    or not np.array_equal(visits, expected)
                    or not np.allclose(mass.sum(-1), visits, atol=1e-3, rtol=1e-4)):
                raise ValueError(f"incomplete native phase: {case_id}, {phase}")
            targets.append(mass / mass.sum(-1, keepdims=True))
        records[case_id] = np.stack(targets)
    if set(records) != by_id.keys() or len(provenance) != 1:
        raise ValueError("routing receipts must cover the roster with one pinned model/runtime")
    return np.stack([records[case["id"]] for case in cases]), next(iter(provenance))


def fit(model, x, y, train, cases, epochs, batch, seed, guard):
    started = perf_counter()
    device = next(model.parameters()).device
    counts = Counter(cases[i]["root"] for i in train)
    weights = torch.tensor([len(train) / (len(counts) * counts[cases[i]["root"]])
                            for i in train], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(train))
        for begin in range(0, len(order), batch):
            guard.check()
            positions = order[begin:begin + batch]
            rows = train[positions]
            log_prob = model(x[rows].to(device)).log_softmax(dim=-1)
            loss = -(y[rows].to(device) * log_prob).sum(-1).mean(dim=(1, 2))
            optimizer.zero_grad(set_to_none=True)
            ((weights[positions].to(device) * loss).sum() / batch).backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    guard.check()
    return perf_counter() - started


def select(scores, capacity, available=None):
    if available is None:
        available = (tuple(range(EXPERTS)),) * LAYERS
    return [
        [ids[i] for i in sorted(ExpertLattice(len(ids)).admit(layer[list(ids)], None, capacity, 0))]
        for layer, ids in zip(scores, available, strict=True)
    ]


def plans(cases, targets, predictions, provenance, bank=None):
    available = bank.available if bank else None
    max_capacity = min(map(len, available)) if available else EXPERTS
    capacities = tuple(k for k in CAPACITIES if k <= max_capacity)
    for policy, scores in predictions.items():
        for case, target, predicted in zip(cases, targets, scores, strict=True):
            for capacity in capacities:
                selected = select(predicted.mean(axis=0), capacity, available)
                retention = np.take_along_axis(target, np.asarray(selected)[None], axis=2).sum(-1).mean(-1)
                yield {
                    "case_id": case["id"], "root": case["root"], "task": case["task"], "mode": case["mode"],
                    "prompt_sha256": digest(render(case)), "model_sha256": provenance[0],
                    "runtime_revision": provenance[1], "policy": policy, "capacity": capacity,
                    "expert_manifest_sha256": bank.manifest_sha256 if bank else None,
                    "resident_by_layer": selected,
                    "offline_mass": dict(zip(PHASES, retention.tolist(), strict=True)),
                }


def train(args):
    guard = MemoryGuard()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        guard.configure_cuda(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
    cases = cases_and_labels(args.cases)
    bank = None
    composition_hash = None
    if args.published_composition:
        from matrix_gguf import PublishedComposition

        bank = PublishedComposition.read(args.published_composition)
        composition_hash = sha(args.published_composition)
    target, provenance = load_targets(cases, args.routing, composition_hash)
    if args.expert_bundle:
        from matrix_expert_bank import ExpertBank

        bank = ExpertBank(args.expert_bundle)
    if bank and bank.model_sha256 != provenance[0]:
        raise ValueError("teacher traces and isolated experts identify different source weights")
    if composition_hash:
        supported = np.zeros((LAYERS, EXPERTS), dtype=bool)
        np.put_along_axis(supported, np.asarray(bank.available), True, axis=1)
        if np.any(target[:, :, ~supported]):
            raise ValueError("routing mass falls outside the published teacher composition")
    train_rows = np.array([i for i, case in enumerate(cases) if case["split"] == "train"])
    dev_rows = np.array([i for i, case in enumerate(cases) if case["split"] == "dev"])
    if not train_rows.size or not dev_rows.size:
        raise ValueError("need both train and dev roots")
    args.out.mkdir(parents=True)
    x, y = torch.from_numpy(np.stack([features(case) for case in cases])), torch.from_numpy(target)
    mean = target[train_rows].mean(axis=0)
    log_prior = torch.from_numpy(np.maximum(mean, 1e-6)).log() if args.global_prior else None
    predictions = {"global": np.broadcast_to(mean, (len(dev_rows), *mean.shape)),
                   "hindsight": target[dev_rows]}
    stats = {}
    for architecture in ("ldt", "trm"):
        torch.manual_seed(args.seed)
        model = PromptController(architecture, log_prior=log_prior).to(device)
        seconds = fit(model, x, y, train_rows, cases, args.epochs, args.batch, args.seed, guard)
        # Use the same CPU scoring path as checkpoint replay to reproduce top-k decisions.
        model.cpu().eval()
        with torch.inference_mode():
            predictions[architecture] = torch.cat([
                model(x[row:row + 1]).softmax(dim=-1) for row in dev_rows
            ]).numpy()
        stats[architecture] = {"training_seconds": seconds, "parameters": sum(p.numel() for p in model.parameters())}
        checkpoint = args.out / f"{architecture}.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"architecture": architecture, "seed": args.seed, "model_sha256": provenance[0],
                    "runtime_revision": provenance[1], "state": model.state_dict(),
                    "expert_manifest_sha256": bank.manifest_sha256 if bank else None}, temporary)
        temporary.replace(checkpoint)
    write_json(args.out / "plans.json", list(plans([cases[i] for i in dev_rows], target[dev_rows],
                                                 predictions, provenance, bank)))
    write_json(args.out / "training.json", {
        "model_sha256": provenance[0], "runtime_revision": provenance[1], "seed": args.seed,
        "epochs": args.epochs, "batch": args.batch, "global_prior": args.global_prior,
        "device": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "expert_manifest_sha256": bank.manifest_sha256 if bank else None,
        "published_composition_sha256": composition_hash,
        "train_roots": sorted({cases[i]["root"] for i in train_rows}),
        "dev_roots": sorted({cases[i]["root"] for i in dev_rows}), "models": stats,
        "memory": guard.receipt(),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--routing", type=Path, required=True)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--expert-bundle", type=Path, help="restrict each prompt plan to indexed bank IDs")
    scope.add_argument("--published-composition", type=Path, help="receipt for a scoped published-bank teacher GGUF")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--global-prior", action="store_true",
                   help="learn bounded prompt-dependent corrections to train-root global frequency")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    if a.epochs <= 0 or not 1 <= a.batch <= 32:
        p.error("invalid training dimensions")
    train(a)


if __name__ == "__main__":
    main()
