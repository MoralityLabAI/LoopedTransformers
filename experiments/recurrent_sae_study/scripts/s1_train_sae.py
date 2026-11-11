"""Train the 'all-loops' and 'train-depth-only' SAEs for one run (toy or mini student), and save configs.

  python s1_train_sae.py --name dfa_random_exit_K16_s0 --K_train 16
  python s1_train_sae.py --name mini:dfa_s0:rec        --K_train 5

SAE training data = last-token states (and, for mini runs, extra random-position states) from the `sae_train` split
only; reconstruction is checked on `sae_val`.  Saved: weights+config in DATA/sae and a deliverable copy in out/sae.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import ROOT
from recsae.io import is_mini, load_split, sae_deliver_prefix, sae_prefix, split_path
from recsae.sae import SAEConfig, recon_stats, train_sae


def gather(name, split, loops, use_pos):
    d = load_split(name, split)
    X = [d["last"][:, loops]]
    if use_pos and is_mini(name):
        z = np.load(split_path(name, split))
        var = name.split(":")[2]
        if f"{var}__pos" in z.files:
            X.append(z[f"{var}__pos"][:, loops])
    dd = X[0].shape[-1]
    return torch.from_numpy(np.concatenate([x.reshape(-1, dd) for x in X]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--K_train", type=int, required=True)
    ap.add_argument("--n_latents", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d0 = load_split(a.name, "sae_val")
    K = d0["correct"].shape[1]
    reg = {}
    for tag, loops in (("all", list(range(1, K + 1))), ("trainK", list(range(1, a.K_train + 1)))):
        X = gather(a.name, "sae_train", loops, True)
        Xv = gather(a.name, "sae_val", loops, False)
        cfg = SAEConfig(d_in=X.shape[1], n_latents=a.n_latents, k=a.k, seed=a.seed, steps=a.steps,
                        train_tag=f"{a.name} loops {loops[0]}-{loops[-1]} (sae_train split)")
        print(f"[{tag}] training SAE on {X.shape[0]} vectors, d={X.shape[1]}")
        sae, hist = train_sae(X, cfg, Xv, device=a.device)
        extra = {"history": hist, "loops": [loops[0], loops[-1]], "run": a.name, "n_train_vectors": int(X.shape[0])}
        sae.save(sae_prefix(a.name, tag), extra=extra)
        sae.save(sae_deliver_prefix(a.name, tag), extra=extra)
        s = recon_stats(sae, Xv, device=a.device)
        reg[tag] = {"val_nmse": s["nmse"], "l0": s["l0"], "loops": [loops[0], loops[-1]]}
        print(f"[{tag}] held-out NMSE {s['nmse']:.4f} L0 {s['l0']:.1f}")
    json.dump(reg, open(sae_deliver_prefix(a.name, "summary") + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
