"""CPU re-evaluation of existing LOOP-T gate checkpoints on the full validation split (inference only).

Core families reproduce the G0 numbers; masked_pointer_chase / modular_recurrence are zero-shot transfer
(these families were never trained). Run from the rmp-aux worktree so research_gym imports.
"""
import json
import sys
from pathlib import Path

import torch

from research_gym.architecture_discovery.checkpoint import load_verified_checkpoint
from research_gym.interp.bundle import read_shard
from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig, _eval_tensors, predict, score

BUNDLE = Path(r"C:\Users\patri\rmp-aux-20261125\experiments\recurrence_monitorability_v0\datasets")
ART = Path(r"C:\Users\patri\Documents\Codex\rmp_v0\artifacts")
OUT = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux\reeval_looptt_cpu.json")
CORE = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")
NEW = ("masked_pointer_chase", "modular_recurrence")
torch.set_num_threads(8)

data = {n: _eval_tensors(read_shard(BUNDLE, "validation", n)) for n in CORE + NEW}
result = {"device": "cpu", "split": "validation", "cells": {}}
for seed in (1, 2, 3):
    cell = f"LOOP-T-core4-ds-s{seed}"
    ck = load_verified_checkpoint(ART / cell / "checkpoints" / "C3.pt", map_location="cpu")
    config = CellConfig(arm="LOOP-T", seed=seed, families=CORE, variant="core4-ds", max_steps=1, deep_supervision=True, beta=1.0)
    model = build_model("LOOP-T", hidden_size=config.hidden_size, num_heads=config.num_heads, visits=config.visits, beta=config.beta)
    model.load_state_dict(ck["model"])
    params = sum(p.numel() for p in model.parameters())
    rec = {"checkpoint_step": ck.get("step"), "params": params, "families": {}}
    for name in CORE + NEW:
        d = data[name]
        pred = predict(model, d["tokens"], config, device=torch.device("cpu"))
        s = score(d, pred)
        counts = torch.bincount(d["targets"]).max().item() / d["targets"].numel()
        rec["families"][name] = {"exact": s["exact"], "correct": round(s["exact"] * s["examples"]), "examples": s["examples"],
                                 "best_constant": counts, "by_depth": s["by_depth"],
                                 "role": "core_reproduction" if name in CORE else "zero_shot_untrained_family"}
        print(cell, name, round(s["exact"], 4), "best_const", round(counts, 4), flush=True)
    result["cells"][cell] = rec
OUT.write_text(json.dumps(result, indent=1))
print("wrote", OUT)
