"""Resumable training cells with named checkpoints for the Recurrence Monitorability Program."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from lsa.canonical import digest
from research_gym.architecture_discovery.checkpoint import (
    load_verified_checkpoint,
    save_paced_checkpoint,
)
from research_gym.neural.looped_decoder import LoopedDecoderLM

from .bundle import BundleConfig, TraceExample, read_shard, train_example
from .models import build_model
from .tasks import FAMILIES


PAUSED, COMPLETED, EXHAUSTED = "paused", "completed", "budget_exhausted"
# The gate passed (C3 exists) and the run was asked to stop there; resuming continues to C4.
GATE_HELD = "gate_passed_c4_pending"


@dataclass(frozen=True)
class CellConfig:
    arm: str
    seed: int
    families: tuple[str, ...]
    variant: str = "base"
    hidden_size: int = 128
    num_heads: int = 4
    visits: int = 8
    batch_per_family: int = 64
    # Examples per forward pass; gradients accumulate over a family's batch. Zero means one pass.
    microbatch: int = 0
    learning_rate: float = 1e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_steps: int = 200_000
    eval_every: int = 250
    eval_per_depth: int = 64
    gate: float = 0.98
    c4_multiple: int = 4
    deep_supervision: bool = False
    input_injection: bool = False
    beta: float = 0.25
    # Depth curriculum: a family starts at its shallowest evaluated depth and unlocks the next
    # depth once interim accuracy at its current deepest depth reaches ``unlock_accuracy``.
    curriculum: bool = False
    unlock_accuracy: float = 0.9
    # Until a family has cleared its deepest depth, this share of its batch is that frontier depth.
    frontier_share: float = 0.5
    # Optional warm start: path of a checkpoint with a ``.receipt.json`` sidecar. Diagnostics only.
    init_from: str = ""

    @property
    def cell_id(self) -> str:
        return f"{self.arm}-{self.variant}-s{self.seed}"


@dataclass(frozen=True)
class Pool:
    """Training examples of one family: the first ``size`` indices of its training stream."""

    tokens: Tensor
    targets: Tensor
    depths: Tensor

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])

    def target_entropy(self) -> float:
        counts = torch.bincount(self.targets.long()).float()
        share = counts[counts > 0] / counts.sum()
        return float(-(share * share.log()).sum())

    def merged(self, other: "Pool") -> "Pool":
        return Pool(
            torch.cat([self.tokens, other.tokens]),
            torch.cat([self.targets, other.targets]),
            torch.cat([self.depths, other.depths]),
        )


def pool_rows(
    family: str, start: int, stop: int, config: BundleConfig, *, staircase: bool = False
) -> list[list[int]]:
    """Rows of ``tokens + [target, depth]`` for training-stream indices ``start..stop-1``."""

    rows = []
    for index in range(start, stop):
        example = train_example(family, index, config, staircase=staircase)
        rows.append([*example.tokens, example.target_token, example.difficulty])
    return rows


def pool_from_rows(rows: Sequence[Sequence[int]]) -> Pool:
    table = torch.tensor(rows, dtype=torch.int16)
    return Pool(table[:, :-2].contiguous(), table[:, -2].contiguous(), table[:, -1].to(torch.int8))


def save_pool(pool: Pool, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": pool.tokens, "targets": pool.targets, "depths": pool.depths}, path)
    return {"path": path.name, "size": pool.size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_pool(path: Path, sha256: str | None = None) -> Pool:
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"training pool hash mismatch: {path.name}")
    return Pool(**torch.load(path, map_location="cpu"))


def draw_indices(
    by_depth: Mapping[int, Tensor],
    deepest: int,
    count: int,
    generator: torch.Generator,
    *,
    frontier_share: float = 0.0,
) -> Tensor:
    """Pool rows for one batch, drawn from the depths up to ``deepest``.

    ``frontier_share`` of the batch comes from ``deepest`` itself; the rest is uniform over the
    shallower depths. A share of zero is uniform over every unlocked depth.
    """

    depths = [depth for depth in by_depth if depth <= deepest]
    frontier = int(round(count * frontier_share)) if len(depths) > 1 else 0
    sizes = {depth: 0 for depth in depths}
    sizes[deepest] += frontier
    spread = depths[:-1] if frontier else depths
    chosen = torch.randint(len(spread), (count - frontier,), generator=generator)
    for position, depth in enumerate(spread):
        sizes[depth] += int((chosen == position).sum())
    picks = [
        by_depth[depth][torch.randint(by_depth[depth].shape[0], (size,), generator=generator)]
        for depth, size in sizes.items()
    ]
    return torch.cat(picks)


def _eval_tensors(examples: Sequence[TraceExample]) -> dict[str, Any]:
    bases = [example for example in examples if example.sibling is None]
    return {
        "ids": [example.example_id for example in bases],
        "tokens": torch.tensor([example.tokens for example in bases], dtype=torch.long),
        "targets": torch.tensor([example.target_token for example in bases], dtype=torch.long),
        "depths": torch.tensor([example.difficulty for example in bases], dtype=torch.long),
    }


def _subset(data: Mapping[str, Any], per_depth: int) -> dict[str, Any]:
    keep: list[int] = []
    for depth in sorted(set(data["depths"].tolist())):
        keep.extend(torch.nonzero(data["depths"] == depth).flatten()[:per_depth].tolist())
    index = torch.tensor(sorted(keep))
    return {
        "ids": [data["ids"][position] for position in index.tolist()],
        **{key: data[key][index] for key in ("tokens", "targets", "depths")},
    }


def run_visits(model: LoopedDecoderLM, tokens: Tensor, config: CellConfig, *, visits: int | None = None):
    """Forward pass with the cell's variant applied: optional input re-injection after every visit."""

    hook = None
    if config.input_injection:
        injected = model.embed(tokens)
        hook = lambda context: context.after + injected  # noqa: E731
    return model(tokens, depth_visits=visits, visit_hook=hook)


def cell_loss(model: LoopedDecoderLM, tokens: Tensor, targets: Tensor, config: CellConfig) -> Tensor:
    """Per-example answer-position cross-entropy; with deep supervision, averaged over visits."""

    output = run_visits(model, tokens, config)
    if not config.deep_supervision:
        return torch.nn.functional.cross_entropy(output.logits[:, -1], targets, reduction="none")
    losses = [
        torch.nn.functional.cross_entropy(model.lens(state[:, -1]), targets, reduction="none")
        for state in output.visit_outputs
    ]
    return torch.stack(losses).mean(dim=0)


@torch.no_grad()
def predict(
    model: LoopedDecoderLM, tokens: Tensor, config: CellConfig, *, device: torch.device, batch: int = 500
) -> Tensor:
    model.eval()
    predictions = [
        run_visits(model, tokens[start : start + batch].to(device), config).logits[:, -1].argmax(dim=-1).cpu()
        for start in range(0, tokens.shape[0], batch)
    ]
    model.train()
    return torch.cat(predictions)


def score(data: Mapping[str, Any], predictions: Tensor) -> dict[str, Any]:
    exact = (predictions == data["targets"]).float()
    by_depth = {
        str(depth): float(exact[data["depths"] == depth].mean())
        for depth in sorted(set(data["depths"].tolist()))
    }
    return {"exact": float(exact.mean()), "examples": int(exact.numel()), "by_depth": by_depth}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def _event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _is_power_of_two(step: int) -> bool:
    return step >= 16 and step & (step - 1) == 0


def run_cell(
    config: CellConfig,
    pools: Mapping[str, Pool],
    *,
    bundle_dir: Path,
    result_dir: Path,
    artifact_dir: Path,
    provenance: Mapping[str, Any],
    device: torch.device,
    max_seconds: float | None = None,
    max_new_steps: int | None = None,
    stop_at_gate: bool = False,
) -> dict[str, Any]:
    """Train one cell until it passes the gate and reaches C4, exhausts its budget, or is paused.

    Small receipts go to ``result_dir``; checkpoints and predictions go to ``artifact_dir``.
    """

    cell_dir = result_dir / config.cell_id
    heavy_dir = artifact_dir / config.cell_id
    events = cell_dir / "events.jsonl"
    latest = heavy_dir / "checkpoints" / "latest.pt"
    # The step budget is not part of a cell's identity, so a budget extension resumes the same cell.
    identity = {key: value for key, value in asdict(config).items() if key != "max_steps"}
    cell_hash = digest({"config": identity, "provenance": dict(provenance)})
    validation = {name: _eval_tensors(read_shard(bundle_dir, "validation", name)) for name in config.families}
    interim = {name: _subset(data, config.eval_per_depth) for name, data in validation.items()}
    ood = {
        name: _eval_tensors(read_shard(bundle_dir, "ood_stress", name))
        for name in config.families
        if FAMILIES[name].ood_depths
    }

    torch.manual_seed(config.seed)
    model = build_model(
        config.arm,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        visits=config.visits,
        beta=config.beta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sampler = torch.Generator().manual_seed(config.seed)
    state: dict[str, Any] = {
        "step": 0,
        "status": PAUSED,
        "c3_step": None,
        "events": {},
        "checkpoints": [],
        "history": [],
        "interval_loss": {name: [0.0, 0] for name in config.families},
        "seconds": 0.0,
        "final": None,
        "deepest": {
            name: (min if config.curriculum else max)(FAMILIES[name].id_depths)
            for name in config.families
        },
        "graduated": {name: not config.curriculum for name in config.families},
    }
    resumed = latest.exists()
    if resumed:
        saved = load_verified_checkpoint(latest, map_location=str(device))
        if saved["cell_hash"] != cell_hash:
            raise ValueError("checkpoint belongs to another cell")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler"].cpu())  # the checkpoint may have been mapped to a GPU
        state = saved["state"]
        if state["status"] == GATE_HELD:
            state["status"] = PAUSED
        if state["status"] == EXHAUSTED and state["step"] < config.max_steps:
            state["status"], state["final"] = PAUSED, None
            _event(events, {"event": "budget_extended", "max_steps": config.max_steps})
    elif config.init_from:
        # Warm start: weights only, from a hash-checked snapshot; optimizer and curriculum start fresh.
        initial = load_verified_checkpoint(Path(config.init_from), map_location=str(device))
        model.load_state_dict(initial["model"])
    _event(
        events,
        {
            "event": "resume" if resumed else "start",
            "step": state["step"],
            "device": str(device),
            "time": time.time(),
        },
    )
    device_pools = {
        name: (pools[name].tokens.to(device), pools[name].targets.to(device)) for name in config.families
    }
    by_depth = {
        name: {
            depth: torch.nonzero(pools[name].depths == depth).flatten()
            for depth in sorted(set(pools[name].depths.tolist()))
        }
        for name in config.families
    }


    def named_checkpoint(label: str) -> None:
        if any(entry["label"] == label for entry in state["checkpoints"]):
            return
        receipt = save_paced_checkpoint(
            {"cell_hash": cell_hash, "label": label, "step": state["step"], "model": model.state_dict()},
            heavy_dir / "checkpoints" / f"{label}.pt",
        )
        entry = {"label": label, "step": state["step"], **{k: receipt[k] for k in ("path", "sha256", "bytes")}}
        state["checkpoints"].append(entry)
        _event(events, {"event": "checkpoint", **entry})

    def save_latest() -> None:
        save_paced_checkpoint(
            {
                "cell_hash": cell_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.get_state(),
                "state": state,
            },
            latest,
        )

    def evaluate_full(label: str) -> dict[str, Any]:
        report: dict[str, Any] = {"label": label, "step": state["step"], "validation": {}, "ood_stress": {}}
        for split, datasets in (("validation", validation), ("ood_stress", ood)):
            for name, data in datasets.items():
                predictions = predict(model, data["tokens"], config, device=device)
                path = heavy_dir / "predictions" / f"{label}-{split}-{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = zip(data["ids"], data["depths"].tolist(), data["targets"].tolist(), predictions.tolist())
                path.write_text(
                    "".join(
                        json.dumps({"example_id": i, "depth": d, "target": t, "prediction": p}) + "\n"
                        for i, d, t, p in rows
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                report[split][name] = {
                    **score(data, predictions),
                    "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        return report

    def evaluate_interim() -> dict[str, Any]:
        metrics = {
            name: score(data, predict(model, data["tokens"], config, device=device))
            for name, data in interim.items()
        }
        loss = {
            name: (total / count if count else None)
            for name, (total, count) in state["interval_loss"].items()
        }
        state["interval_loss"] = {name: [0.0, 0] for name in config.families}
        record = {"step": state["step"], "train_loss": loss, "validation": metrics}
        state["history"].append(record)
        _event(events, {"event": "eval", **record})
        return metrics

    def note_events(metrics: Mapping[str, Any]) -> None:
        shallowest = [metrics[name]["by_depth"][min(metrics[name]["by_depth"], key=int)] for name in metrics]
        if "C1" not in state["events"] and min(shallowest) >= 0.5:
            state["events"]["C1"] = state["step"]
            named_checkpoint("C1")
        for name in metrics:
            deepest = state["deepest"][name]
            if state["graduated"][name] or metrics[name]["by_depth"][str(deepest)] < config.unlock_accuracy:
                continue
            if deepest < max(FAMILIES[name].id_depths):
                state["deepest"][name] = deepest + 1
                _event(events, {"event": "unlock", "family": name, "depth": deepest + 1, "step": state["step"]})
            else:
                state["graduated"][name] = True
                _event(events, {"event": "graduated", "family": name, "step": state["step"]})
        depths = sorted({depth for name in metrics for depth in metrics[name]["by_depth"]}, key=int)
        for depth in depths:
            label = f"C2_{depth}"
            values = [metrics[name]["by_depth"][depth] for name in metrics if depth in metrics[name]["by_depth"]]
            if label not in state["events"] and min(values) >= 0.99:
                state["events"][label] = state["step"]
                named_checkpoint(label)
        if state["c3_step"] is None and min(metrics[name]["exact"] for name in metrics) >= config.gate:
            full = evaluate_full("C3")
            if min(value["exact"] for value in full["validation"].values()) >= config.gate:
                state["c3_step"] = state["events"]["C3"] = state["step"]
                state["c3"] = full
                named_checkpoint("C3")

    started = time.perf_counter()
    new_steps = 0
    model.train()
    if state["step"] == 0 and not state["checkpoints"]:
        named_checkpoint("C0")
        note_events(evaluate_interim())
    while True:
        target = None if state["c3_step"] is None else config.c4_multiple * state["c3_step"]
        if target is not None and stop_at_gate:
            state["status"] = GATE_HELD
            break
        if target is not None and state["step"] >= target:
            state["status"] = COMPLETED
            named_checkpoint("C4")
            break
        if target is None and state["step"] >= config.max_steps:
            state["status"] = EXHAUSTED
            break
        if (max_seconds is not None and time.perf_counter() - started >= max_seconds) or (
            max_new_steps is not None and new_steps >= max_new_steps
        ):
            break
        rate = config.learning_rate * min(1.0, (state["step"] + 1) / config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        step_losses = {}
        # Families are run one at a time and their gradients summed, so memory follows one batch.
        for name in config.families:
            pick = draw_indices(
                by_depth[name],
                state["deepest"][name],
                config.batch_per_family,
                sampler,
                frontier_share=0.0 if state["graduated"][name] else config.frontier_share,
            ).to(device)
            total = 0.0
            for chunk in pick.split(config.microbatch or config.batch_per_family):
                tokens, targets = device_pools[name][0][chunk].long(), device_pools[name][1][chunk].long()
                loss = cell_loss(model, tokens, targets, config).sum()
                if torch.isfinite(loss):
                    (loss / (pick.numel() * len(config.families))).backward()
                total += float(loss.detach())
            step_losses[name] = total / pick.numel()
        if not all(math.isfinite(value) for value in step_losses.values()):
            state["status"] = "nonfinite_loss"
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        state["step"] += 1
        new_steps += 1
        for name, value in step_losses.items():
            state["interval_loss"][name][0] += value
            state["interval_loss"][name][1] += 1
        if _is_power_of_two(state["step"]):
            named_checkpoint(f"step_{state['step']}")
        if state["step"] % config.eval_every == 0:
            note_events(evaluate_interim())
            save_latest()
    state["seconds"] += time.perf_counter() - started
    if state["status"] not in (PAUSED, GATE_HELD) and state["final"] is None:
        state["final"] = evaluate_full("final")
    save_latest()

    entropy = {name: pools[name].target_entropy() for name in config.families}
    last_loss = next(
        (
            record["train_loss"]
            for record in reversed(state["history"])
            if None not in record["train_loss"].values()
        ),
        None,
    )
    result = {
        "cell_id": config.cell_id,
        "cell_hash": cell_hash,
        "config": asdict(config),
        "provenance": dict(provenance),
        "status": state["status"],
        "resumed": resumed,
        "device": str(device),
        "peak_vram_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "optimizer_steps": state["step"],
        "examples_seen": state["step"] * config.batch_per_family * len(config.families),
        "training_seconds": state["seconds"],
        "unique_parameters": model.parameter_breakdown()["unique_parameters"],
        "events": state["events"],
        "deepest_unlocked": state["deepest"],
        "checkpoints": state["checkpoints"],
        "target_entropy_nats": entropy,
        "last_interval_train_loss": last_loss,
        "history": state["history"],
        "c3": state.get("c3"),
        "final": state["final"],
        "gate": config.gate,
        "gate_passed": state["c3_step"] is not None,
    }
    _write_json(cell_dir / "result.json", result)
    _event(events, {"event": "stop", "status": state["status"], "step": state["step"]})
    return result
