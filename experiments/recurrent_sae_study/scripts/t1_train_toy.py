"""Train weight-tied looped transformers (vendored crrbench) on a state-tracking task.

Outputs  out/toy/ckpt/{task}_{regime}_K{K}_s{seed}.pt   (state_dict + cfg)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401  (sets sys.path)
from recsae import ROOT, DATA
from crrbench.tasks import TASKS
from crrbench.train import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="s4")
    ap.add_argument("--regime", default="random_exit", choices=["final", "random_intermediate", "random_exit"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lo", type=int, default=4)
    ap.add_argument("--hi", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    task = TASKS[a.task]()
    cfg = dict(K=a.K, batch=a.batch, d=a.d, heads=4, lr=a.lr, steps=a.steps, train_len=(a.lo, a.hi))
    t0 = time.time()
    model = train_model(task, a.regime, a.seed, cfg, device=a.device)
    name = f"{a.task}_{a.regime}_K{a.K}_s{a.seed}"
    os.makedirs(os.path.join(DATA, "toy", "ckpt"), exist_ok=True)
    path = os.path.join(DATA, "toy", "ckpt", name + ".pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "task": a.task, "regime": a.regime, "seed": a.seed,
                "vocab": task.vocab, "n_classes": task.n_classes}, path)
    # quick accuracy-by-loop readout on in-range and longer strings
    rng = np.random.default_rng(10_000 + a.seed)
    rep = {"train_s": time.time() - t0, "acc": {}}
    for N in [4, 8, 16, 24, 32]:
        x, y, _ = task.sample(rng, 1000, N)
        with torch.no_grad():
            lg = model.run(torch.from_numpy(x).to(a.device), 3 * a.K // 2 if False else a.K)
        acc = (lg.argmax(-1).cpu().numpy() == y[None]).mean(1)
        rep["acc"][N] = [round(float(acc[k - 1]), 3) for k in (1, 2, 4, 8, a.K)]
    print(name, json.dumps(rep))
    json.dump(rep, open(path.replace(".pt", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
