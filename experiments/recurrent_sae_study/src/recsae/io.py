"""Split definitions + cached activation loading.

A *run name* identifies one (model, readout) series of per-loop states:
  toy runs   : "<ckpt name>",                    e.g. dfa_random_exit_K16_s0
  mini runs  : "mini:<task>_s<seed>:<variant>",  e.g. mini:dfa_s0:rec   (variant in laco / rec0 / rec)
Both load to the same dict: last [N, L, d] (state after loop k; k=0 = input state), logits [N, K, C], y, N,
pred, correct.
"""
from __future__ import annotations

import os
import re

import numpy as np

import recsae  # noqa: F401
from recsae import DATA, ROOT

SPLITS = {          # toy splits (strings sized like the crrbench 'full' config: trained on 2..8, extrapolate to 10..16)
    "sae_train": ([2, 3, 4, 6, 8], 2000, 1),
    "sae_val": ([2, 3, 4, 6, 8], 300, 2),
    "ana_train": ([2, 3, 4, 6, 8], 800, 3),
    "ana_val": ([2, 3, 4, 6, 8], 250, 4),
    "test_id": ([4, 8], 1000, 5),
    "test_ood": ([10, 12, 16], 800, 6),
}


def safe(name: str) -> str:
    if name.startswith("sib:"):                       # sibling checkpoint: keep only <task>__<regime>__s<seed>
        return "sib_" + os.path.splitext(os.path.basename(name[4:]))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def is_mini(name: str) -> bool:
    return name.startswith("mini:")


def split_path(name, split):
    if is_mini(name):
        _, run, _ = name.split(":")
        return os.path.join(DATA, "mini", run, f"acts_{split}.npz")
    return os.path.join(DATA, "toy", "acts", f"{safe(name)}__{split}.npz")


def load_split(name, split):
    z = np.load(split_path(name, split))
    if is_mini(name):
        var = name.split(":")[2]
        d = {"last": z[f"{var}__last"], "logits": z[f"{var}__logits"], "y": z["y"], "N": z["N"],
             "ctx": np.zeros_like(z["y"])}
        d["t_last"], d["t_logits"] = z["t_last"], z["t_logits"]
    else:
        d = {k: z[k] for k in z.files}
    d["pred"] = d["logits"].argmax(-1)
    d["correct"] = d["pred"] == d["y"][:, None]
    return d


def sae_prefix(name, tag):
    p = os.path.join(DATA, "sae")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{safe(name)}__{tag}")


def sae_deliver_prefix(name, tag):
    p = os.path.join(ROOT, "out", "sae")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{safe(name)}__{tag}")


def res_dir(sub=""):
    p = os.path.join(ROOT, "out", "res", sub) if sub else os.path.join(ROOT, "out", "res")
    os.makedirs(p, exist_ok=True)
    return p


def fig_path(tag, name):
    p = os.path.join(ROOT, "figs")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{safe(tag)}__{name}.png")
