"""Visit hooks for capture, patching, skipping, repeating, and early exit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Iterator

from torch import Tensor, nn

from research_gym.neural.looped_decoder import ExplicitDecoderBlock
from research_gym.neural.schedule_executor import StopSchedule, VisitContext, VisitHook


def compose(*hooks: VisitHook) -> VisitHook:
    """Apply hooks in order; each sees the state left by the one before it."""

    def hook(context: VisitContext) -> Tensor | None:
        changed = False
        for item in hooks:
            replacement = item(context)
            if replacement is not None:
                context = replace(context, after=replacement)
                changed = True
        return context.after if changed else None

    return hook


def capture(store: list[Tensor]) -> VisitHook:
    """Append each visit's output state to ``store`` without changing the run."""

    def hook(context: VisitContext) -> None:
        store.append(context.after.detach())

    return hook


def patch(visit: int, replacement: Tensor | Callable[[Tensor], Tensor]) -> VisitHook:
    """Replace the state leaving ``visit`` with a tensor or a function of it."""

    def hook(context: VisitContext) -> Tensor | None:
        if context.visit != visit:
            return None
        return replacement(context.after) if callable(replacement) else replacement

    return hook


def skip(visit: int) -> VisitHook:
    """Discard the update made at ``visit``."""

    def hook(context: VisitContext) -> Tensor | None:
        return context.before if context.visit == visit else None

    return hook


def repeat(visit: int, times: int = 1) -> VisitHook:
    """Apply the module of ``visit`` a further ``times`` times."""

    def hook(context: VisitContext) -> Tensor | None:
        if context.visit != visit:
            return None
        state = context.after
        for _ in range(times):
            state = context.module(state, **context.kwargs)
        return state

    return hook


def stop_after(visit: int) -> VisitHook:
    """End the schedule once ``visit`` has run."""

    def hook(context: VisitContext) -> None:
        if context.visit == visit:
            raise StopSchedule

    return hook


@contextmanager
def record_components(model: nn.Module) -> Iterator[list[dict[str, Tensor]]]:
    """Record every explicit block application, in visit order, while the context is open."""

    blocks = [module for module in model.modules() if isinstance(module, ExplicitDecoderBlock)]
    if not blocks:
        raise ValueError("model has no explicit-attention blocks to record")
    records: list[dict[str, Tensor]] = []
    for block in blocks:
        block.recorder = records
    try:
        yield records
    finally:
        for block in blocks:
            block.recorder = None
