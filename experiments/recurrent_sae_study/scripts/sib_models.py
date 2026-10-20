"""List / smoke-check looped-transformer checkpoints produced by the crrbench sibling session.

  python sib_models.py list                  -> run names ('sib:<path>') one per line
  python sib_models.py check [substring]     -> accuracy by loop for matching checkpoints
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import toy

SIB = os.environ.get("CRR_RESULTS", os.path.join(
    r"C:\Users\patri\AppData\Roaming\Claude\scratch-workspaces\5b97c40b-6d2c-41c1-948d-8b330aaaa915"
    r"\e70eb5c9-64fa-4169-b814-fbddfbe63a93\scratch-2026-10-19-3036ea", "results", "full", "models"))


def names(substr=""):
    return ["sib:" + p for p in sorted(glob.glob(os.path.join(SIB, "*.pt"))) if substr in os.path.basename(p)]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    sub = sys.argv[2] if len(sys.argv) > 2 else ""
    if cmd == "list":
        print("\n".join(names(sub)))
    else:
        for nm in names(sub):
            m, task, ck = toy.load_toy(nm)
            rng = np.random.default_rng(1)
            out = []
            for N in (4, 8, 10, 12, 16):
                x, y, _ = task.sample(rng, 1000, N)
                with torch.no_grad():
                    lg = m.run(torch.from_numpy(x), 32)
                acc = (lg.argmax(-1).numpy() == y[None]).mean(1)
                out.append((N, [round(float(acc[k - 1]), 2) for k in (1, 2, 4, 8, 16, 24, 32)]))
            print(os.path.basename(nm), task.name, ck["cfg"]["K"], "maj", round(task.majority_acc(rng, 8), 2), out)
