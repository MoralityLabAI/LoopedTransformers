"""Load trained RMP cells for analysis and run them with their training variant applied."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from research_gym.architecture_discovery.checkpoint import load_verified_checkpoint
from research_gym.neural.looped_decoder import LoopedDecoderLM
from research_gym.neural.schedule_executor import VisitHook

from .bundle import TraceExample
from .hooks import compose
from .models import build_model
from .training import CellConfig, score


@dataclass(frozen=True)
class AnalysisModel:
    """One checkpoint of one cell, ready to run."""

    name: str
    cell_id: str
    label: str
    step: int
    config: CellConfig
    model: LoopedDecoderLM

    def run(self, tokens: Tensor, *hooks: VisitHook, visits: int | None = None):
        """Forward pass. The cell's input re-injection, if any, runs before ``hooks`` at every visit."""

        chain = list(hooks)
        if self.config.input_injection:
            injected = self.model.embed(tokens)
            chain.insert(0, lambda context: context.after + injected)
        return self.model(tokens, depth_visits=visits, visit_hook=compose(*chain) if chain else None)


def with_injection(model: AnalysisModel, injection: bool) -> AnalysisModel:
    """The same weights run with or without input re-injection, so a control matches its model's variant."""

    return replace(model, config=replace(model.config, input_injection=injection))


def cell_config(runs_dir: Path, cell_id: str) -> CellConfig:
    raw = json.loads((runs_dir / cell_id / "result.json").read_text(encoding="utf-8"))["config"]
    known = {item.name for item in fields(CellConfig)}
    values = {key: value for key, value in raw.items() if key in known}
    values["families"] = tuple(values["families"])
    return CellConfig(**values)


def load_analysis_model(
    runs_dir: Path,
    artifact_dir: Path,
    cell_id: str,
    label: str = "C3",
    *,
    name: str | None = None,
    device: str | torch.device = "cpu",
) -> AnalysisModel:
    """Rebuild a cell's model from its recorded config and load a hash-verified checkpoint."""

    config = cell_config(runs_dir, cell_id)
    saved = load_verified_checkpoint(
        artifact_dir / cell_id / "checkpoints" / f"{label}.pt", map_location=str(device)
    )
    model = build_model(
        config.arm,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        visits=config.visits,
        beta=config.beta,
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    step = saved["step"] if "step" in saved else saved["state"]["step"]
    return AnalysisModel(name or f"{cell_id}@{label}", cell_id, label, int(step), config, model)


def example_tensors(examples: Sequence[TraceExample]) -> dict[str, Any]:
    return {
        "ids": [example.example_id for example in examples],
        "tokens": torch.tensor([example.tokens for example in examples], dtype=torch.long),
        "targets": torch.tensor([example.target_token for example in examples], dtype=torch.long),
        "depths": torch.tensor([example.difficulty for example in examples], dtype=torch.long),
    }


@torch.no_grad()
def answer_logits(model: AnalysisModel, tokens: Tensor, *, batch: int = 500) -> Tensor:
    device = next(model.model.parameters()).device
    return torch.cat(
        [
            model.run(tokens[start : start + batch].to(device)).logits[:, -1].float().cpu()
            for start in range(0, tokens.shape[0], batch)
        ]
    )


def evaluate(model: AnalysisModel, shards: Mapping[str, Sequence[TraceExample]]) -> dict[str, Any]:
    """Exact match per family and depth on base rows, scored the way the gate scores them."""

    report = {}
    for family, examples in shards.items():
        data = example_tensors([example for example in examples if example.sibling is None])
        report[family] = score(data, answer_logits(model, data["tokens"]).argmax(dim=-1))
    return report
