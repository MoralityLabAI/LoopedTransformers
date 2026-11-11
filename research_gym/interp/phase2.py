"""Shared pieces of the Phase 2 scripts: which models, which data, and the per-position mean states."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis import AnalysisModel, load_analysis_model, with_injection
from .bundle import read_shard
from .dictionaries import model_states


CORE = ("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4")

# name -> (cell, checkpoint label, run with input re-injection regardless of the cell's own setting)
MODELS: dict[str, tuple[str, str, bool | None]] = {
    "LOOP-T s1": ("LOOP-T-core4-ds-s1", "C3", None),
    "LOOP-T s2": ("LOOP-T-core4-ds-s2", "C3", None),
    "LOOP-T s3": ("LOOP-T-core4-ds-s3", "C3", None),
    "FF-U s3": ("FF-U-core4-ds-inj-s3", "C3", None),
    "LOOP-T s1 C0": ("LOOP-T-core4-s1", "C0", None),
    "LOOP-T s2 C0": ("LOOP-T-core4-s2", "C0", None),
    "LOOP-T s3 C0": ("LOOP-T-core4-s3", "C0", None),
    "FF-U s3 C0": ("FF-U-core4-s3", "C0", True),
    # Control 7's planted positive, made by research_gym.scripts.rmp_phase2_plant.
    "LOOP-T s1 planted": ("LOOP-T-core4-plant-s1", "final", None),
    # Fresh seeds trained 2026-10-30 (records in experiments/recurrence_monitorability_v0/phase3/runs),
    # each at the stage where it passed G0; used to replicate Phase 2 findings on held-out seeds.
    "LOOP-T s4": ("LOOP-T-core4-ds-s4", "C3", None),
    "LOOP-T s5": ("LOOP-T-core4-s5", "C3", None),
    "LOOP-T s6": ("LOOP-T-core4-s6", "C3", None),
    "FF-U s4": ("FF-U-core4-ds-s4", "C3", None),
    "FF-U s6": ("FF-U-core4-ds-inj-s6", "C3", None),
}


def select_config(entries: dict[str, dict], kind: str = "shared") -> tuple[str, bool]:
    """The plan's selection rule over one model's grid: smallest width, then smallest k, with cross-entropy
    recovered at least 0.95 at every visit; otherwise the best minimum. Returns (name, met the bar)."""

    grid = [(entry["config"]["width"], entry["config"]["k"], name, entry["splice"]["min_ce_recovered"])
            for name, entry in entries.items()
            if entry["config"]["kind"] == kind and entry["config"]["seed"] == 0 and entry["config"]["site"] == -1]
    passing = sorted(item for item in grid if item[3] >= 0.95)
    if passing:
        return passing[0][2], True
    return max(grid, key=lambda item: item[3])[2], False


GRID_WIDTHS = (1024, 2048, 4096, 8192)


def adjacent_width(width: int) -> int:
    """The plan's adjacent width: the next larger in the grid, or the next smaller at the top."""

    index = GRID_WIDTHS.index(width)
    return GRID_WIDTHS[index + 1] if index + 1 < len(GRID_WIDTHS) else GRID_WIDTHS[index - 1]


def slug(name: str) -> str:
    return name.replace(" ", "_")


def load_model(name: str, runs: Path, artifacts: Path, device: torch.device) -> AnalysisModel:
    cell, label, injection = MODELS[name]
    model = load_analysis_model(runs, artifacts, cell, label, name=name, device=device)
    return model if injection is None else with_injection(model, injection)


def selection_half(groups: list[str]) -> np.ndarray:
    """The same group-hash split as Phase 1: True for the selection half of monitor_calibration."""

    return np.array([hashlib.sha256(group.encode()).digest()[0] % 2 == 0 for group in groups])


def core_examples(bundle: Path, split: str, families=CORE) -> dict[str, Any]:
    """Base rows of the core families, concatenated: tokens, targets, depths, family, group and id."""

    rows = [(family, row) for family in families for row in read_shard(bundle, split, family) if row.sibling is None]
    return {
        "tokens": torch.tensor([row.tokens for _, row in rows], dtype=torch.long),
        "targets": torch.tensor([row.target_token for _, row in rows], dtype=torch.long),
        "depths": np.array([row.difficulty for _, row in rows]),
        "families": np.array([family for family, _ in rows]),
        "groups": [row.group_id for _, row in rows],
        "ids": [row.example_id for _, row in rows],
        "rows": [row for _, row in rows],
    }


def subset(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    index = np.flatnonzero(mask)
    rows = torch.as_tensor(index, dtype=torch.long)
    return {
        key: (value[rows] if isinstance(value, torch.Tensor) else value[index] if isinstance(value, np.ndarray)
              else [value[i] for i in index])
        for key, value in data.items()
    }


@torch.no_grad()
def mean_states(model: AnalysisModel, tokens: torch.Tensor, *, count: int = 2000, batch: int = 250,
                seed: int = 0) -> torch.Tensor:
    """Per-site, per-position mean state over a sample of examples: (visits + 1, positions, H)."""

    device = next(model.model.parameters()).device
    index = torch.randperm(tokens.shape[0], generator=torch.Generator().manual_seed(seed))[:count]
    total = None
    for start in range(0, len(index), batch):
        states = model_states(model, tokens[index[start : start + batch]].to(device)).sum(dim=0)
        total = states if total is None else total + states
    return total / len(index)
