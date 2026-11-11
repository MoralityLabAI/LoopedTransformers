"""Phase 2, control 7: fine-tune a copy of LOOP-T seed 1 so that it writes a chosen variable into a chosen
direction, as a planted positive the dictionary pipeline must find (docs/rmp_phase2_plan.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from research_gym.interp.analysis import evaluate
from research_gym.interp.bundle import read_shard
from research_gym.interp.dictionaries import rms
from research_gym.interp.phase2 import CORE, load_model
from research_gym.interp.training import load_pool


PLANT_POSITION = 4
DIRECTION_SEED = 7
SHARPNESS = 4.0


def planted_direction(dim: int) -> torch.Tensor:
    vector = torch.randn(dim, generator=torch.Generator().manual_seed(DIRECTION_SEED))
    return vector / vector.norm()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cell", default="LOOP-T-core4-plant-s1")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-per-family", type=int, default=256)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    parent = load_model("LOOP-T s1", args.runs, args.artifact_dir, device)
    model, config = parent.model, parent.config
    model.train()
    receipt = json.loads((args.pool_dir / "pool_receipt.json").read_text(encoding="utf-8"))
    pools = {name: load_pool(args.pool_dir / receipt[name]["path"], receipt[name]["sha256"]) for name in CORE}
    direction = planted_direction(config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(11)
    log = []
    for step in range(args.steps):
        warm = min(1.0, (step + 1) / config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * warm
        optimizer.zero_grad(set_to_none=True)
        totals = {"task": 0.0, "aux": 0.0}
        for name in CORE:
            pool = pools[name]
            index = torch.randint(0, pool.tokens.shape[0], (args.batch_per_family,), generator=generator)
            tokens = pool.tokens[index].long().to(device)
            targets = pool.targets[index].long().to(device)
            output = model(tokens)
            answers = [state[:, -1] for state in output.visit_outputs]
            # Stage-2 task loss: answer cross-entropy at every visit, averaged.
            task = torch.stack([torch.nn.functional.cross_entropy(model.lens(a), targets) for a in answers]).mean()
            parity = (tokens[:, PLANT_POSITION] % 2).float()
            aux = torch.stack([
                torch.nn.functional.binary_cross_entropy_with_logits(SHARPNESS * (a / rms(a)) @ direction, parity)
                for a in answers
            ]).mean()
            ((task + args.aux_weight * aux) / len(CORE)).backward()
            totals["task"] += float(task) / len(CORE)
            totals["aux"] += float(aux) / len(CORE)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            log.append({"step": step, **totals})
            print(step, {key: round(value, 4) for key, value in totals.items()}, flush=True)

    model.eval()
    shards = {name: read_shard(args.bundle_dir, "validation", name) for name in CORE}
    validation = {name: value["exact"] for name, value in evaluate(parent, shards).items()}
    checkpoint = args.artifact_dir / args.cell / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": args.steps, "label": "final"}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (checkpoint.parent / "final.pt.receipt.json").write_text(
        json.dumps({"path": "final.pt", "sha256": digest, "bytes": checkpoint.stat().st_size}, indent=2) + "\n",
        encoding="utf-8")
    result = {
        "cell_id": args.cell, "config": asdict(config), "parent": "LOOP-T-core4-ds-s1@C3",
        "plant": {"position": PLANT_POSITION, "direction_seed": DIRECTION_SEED, "sharpness": SHARPNESS,
                  "aux_weight": args.aux_weight, "steps": args.steps, "direction": direction.cpu().tolist()},
        "validation_exact": validation, "log": log, "checkpoint_sha256": digest,
    }
    out = args.runs / args.cell / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("validation", {key: round(value, 4) for key, value in validation.items()}, flush=True)


if __name__ == "__main__":
    main()
