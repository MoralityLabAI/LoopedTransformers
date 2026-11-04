"""MIRROR-01 reproducible training and sealed synthetic evaluation CLI."""
import argparse
import json
from pathlib import Path
from random import Random

from .gym import PERSONAS, Target, run_episode


def evaluate(model=None, episodes=200, budget=12, seed=100007):
    from .neural import MirrorPolicy
    policies = ["random", "paired", "oracle"] + (["mirror"] if model is not None else [])
    rng = Random(seed)
    worlds = [Target.sample(rng, 6) for _ in range(episodes)]
    wrapper = MirrorPolicy(model) if model is not None else None
    results = {}
    for name in policies:
        records = [run_episode(world, name, budget=budget,
                               rng=Random(seed + i * 7919), model=wrapper)
                   for i, world in enumerate(worlds)]
        positives = sum(r["ground_truth"] for r in records)
        true_positive = sum(r["found"] and r["ground_truth"] for r in records)
        false_positive = sum(r["found"] and not r["ground_truth"] for r in records)
        results[name] = {
            "episodes": episodes, "positives": positives,
            "true_positive": true_positive, "false_positive": false_positive,
            "discovery_rate": true_positive / positives if positives else None,
            "mean_queries": sum(r["queries"] for r in records) / episodes,
            "witness_precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive else None,
        }
    return {"protocol": "MIRROR-01", "eval_seed": seed,
            "budget": budget, "policies": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "eval"], default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    model = None
    receipt = {"model": None, "train": None}
    if args.mode in ("smoke", "train"):
        import torch
        from .neural import CounterfactualBeliefField, train
        torch.manual_seed(args.seed)
        model = CounterfactualBeliefField()
        device = "cuda" if torch.cuda.is_available() and args.mode == "train" else "cpu"
        receipt["train"] = train(model, steps=8 if args.mode == "smoke" else args.steps,
                                 batch_size=8 if args.mode == "smoke" else 32,
                                 seed=args.seed, device=device)
        model.cpu()
        receipt["model"] = {"parameters": sum(p.numel() for p in model.parameters()),
                            "loops": model.loops, "width": 64, "device": device}
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model_config": receipt["model"],
                        "training": receipt["train"]}, args.checkpoint)
    elif args.checkpoint:
        import torch
        from .neural import CounterfactualBeliefField
        model = CounterfactualBeliefField()
        # weights_only is explicit to avoid pickle deserialization of unknown classes.
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        receipt["model"] = payload["model_config"]
    receipt["evaluation"] = evaluate(
        model, episodes=24 if args.mode == "smoke" else args.episodes,
        budget=args.budget, seed=args.seed + 100000)
    serialized = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
