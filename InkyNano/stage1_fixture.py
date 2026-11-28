"""Build a deliberately synthetic RoutingGym trace fixture with known structure."""
import json

import numpy as np

from stage1_data import ACTIVE, ARMS, EXPERTS, LAYERS, SYNTHETIC, entries, write_json


def generate(out, seed, guard):
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    anchors = rng.permutation(EXPERTS).reshape(4, 32)
    for arm in ARMS:
        folder = out / arm
        folder.mkdir()
        with ((folder / "roster.jsonl").open("w", encoding="utf8") as roster,
              (folder / "trace.jsonl").open("w", encoding="utf8") as trace):
            for i in range(16):
                root, category = f"synthetic-{i:02d}", i % 4
                prompt = 8 + 4 * category
                length = 160 if arm == "generated" else 12 + 4 * category
                roster.write(json.dumps({
                    "root": root, "turn": arm, "family": root,
                    "split": "train" if i < 12 else "dev",
                    "source_kind": SYNTHETIC,
                    "prompt_len": prompt, "token_ids": list(range(prompt + length)),
                }) + "\n")
                for token in range(prompt + length):
                    # The sharing and window drift here are properties of the generator.
                    window = max(0, token - prompt) // 32
                    phase = "prompt" if token < prompt else "continuation"
                    for layer in range(LAYERS):
                        start = (3 * (layer // 6) + 3 * window) % 32
                        common = np.roll(anchors[category], -start)[:12]
                        previous = rng.choice(common, size=8, replace=False).tolist()
                        for pass_id in range(int(ACTIVE[layer].sum())):
                            if pass_id:
                                keep = rng.choice(previous, size=5, replace=False).tolist()
                                candidates = np.roll(anchors[category], -start - pass_id * 5)[:18]
                                available = [e for e in candidates if e not in previous]
                                previous = keep + rng.choice(available, size=3, replace=False).tolist()
                            weights = rng.integers(1, 6, size=8).tolist()
                            trace.write(json.dumps({
                                "root": root, "turn": arm, "token": token, "layer": layer,
                                "pass": pass_id, "phase": phase, "experts": previous,
                                "weights": weights,
                            }, separators=(",", ":")) + "\n")
                guard.check()
    write_json(out / "manifest.json", {
        "kind": SYNTHETIC, "seed": seed, "files": entries(out),
        "description": "Invented routing values with five-of-eight consecutive pass overlap.",
    })
