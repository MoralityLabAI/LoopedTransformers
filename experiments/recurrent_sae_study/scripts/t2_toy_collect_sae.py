"""Toy path step 2: collect per-loop activations on disjoint splits (SAE training is s1_train_sae.py).

Splits (all disjoint by RNG stream; sizes chosen to fit disk/RAM: each split is [N, K+1, d] fp16):
  sae_train  fits the SAEs                                  (train lengths)
  sae_val    SAE early-stopping / reconstruction on unseen  (train lengths)
  ana_train  feature selection, probes, controller training (train lengths)
  ana_val    thresholds / hyper-parameters                  (train lengths)
  test_id    held-out, in-distribution lengths              (interventions + final metrics)
  test_ood   held-out, longer strings than ever seen        (interventions + final metrics)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch

import recsae  # noqa: F401
from recsae import DATA, ROOT
from recsae import toy
from recsae.sae import SAEConfig, train_sae, recon_stats

from recsae.io import SPLITS, split_path, load_split


def collect_all(name, K, device, threads):
    torch.set_num_threads(threads)
    model, task, ck = toy.load_toy(name, device)
    os.makedirs(os.path.join(DATA, "toy", "acts"), exist_ok=True)
    for sp, (lengths, n, off) in SPLITS.items():
        p = split_path(name, sp)
        if os.path.exists(p):
            continue
        rng = np.random.default_rng(1_000_000 * (ck["seed"] + 1) + off)
        split = toy.make_split(task, rng, lengths, n)
        t0 = time.time()
        d = toy.collect(model, split, K, device)
        np.savez(p, last=d["last"], mean=d["mean"], logits=d["logits"], y=d["y"], ctx=d["ctx"], N=d["N"])
        print(f"  collected {sp}: {d['last'].shape} in {time.time()-t0:.0f}s, acc@K={d['correct'][:, -1].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--K_eval", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    collect_all(a.name, a.K_eval, a.device, a.threads)

    print("collection done; next: s1_train_sae.py --name", a.name, "--K_train", toy.load_toy(a.name)[2]["cfg"]["K"])


if __name__ == "__main__":
    main()
