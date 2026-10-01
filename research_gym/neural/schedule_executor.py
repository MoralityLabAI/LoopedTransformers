"""Execute expanded visits with independent parameter and state-gradient masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.func import functional_call
    from torch.utils.checkpoint import checkpoint
except ImportError as exc:  # pragma: no cover
    raise ImportError("schedule execution requires the optional 'neural' extra") from exc


@dataclass(frozen=True)
class ScheduleExecution:
    output: Tensor
    visit_outputs: tuple[Tensor, ...]


@dataclass(frozen=True)
class VisitContext:
    """What a visit hook sees: the state entering and leaving one module application."""

    visit: int
    module_index: int
    module: nn.Module
    before: Tensor
    after: Tensor
    kwargs: Mapping[str, object]


class StopSchedule(Exception):
    """Raised by a visit hook to end execution after the current visit."""


# A hook returns a replacement for the visit's output state, or None to leave it unchanged.
VisitHook = Callable[[VisitContext], "Tensor | None"]


def _frozen_module_call(module: nn.Module, state: Tensor, **kwargs) -> Tensor:
    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach() for name, value in module.named_buffers()}
    return functional_call(module, (parameters, buffers), (state,), kwargs)


class ScheduleExecutor(nn.Module):
    def forward(
        self,
        state: Tensor,
        *,
        modules: Sequence[nn.Module],
        module_indices: Sequence[int],
        parameter_visits: frozenset[int],
        retained_state_edges: frozenset[int],
        module_kwargs: Mapping[str, object] | None = None,
        activation_checkpointing: bool = False,
        visit_hook: VisitHook | None = None,
    ) -> ScheduleExecution:
        if not module_indices:
            raise ValueError("schedule requires at least one visit")
        if any(index not in range(len(modules)) for index in module_indices):
            raise ValueError("schedule references an unknown physical module")
        valid_visits = frozenset(range(len(module_indices)))
        valid_edges = frozenset(range(len(module_indices) - 1))
        if not parameter_visits.issubset(valid_visits):
            raise ValueError("parameter mask references an unknown visit")
        if not retained_state_edges.issubset(valid_edges):
            raise ValueError("state mask references an unknown edge")
        kwargs = dict(module_kwargs or {})
        outputs = []
        for visit, module_index in enumerate(module_indices):
            if visit > 0 and visit - 1 not in retained_state_edges:
                state = state.detach()
            before = state
            module = modules[module_index]
            if visit in parameter_visits:
                if activation_checkpointing and torch.is_grad_enabled():
                    state = checkpoint(
                        lambda value, selected=module: selected(value, **kwargs),
                        state,
                        use_reentrant=False,
                    )
                else:
                    state = module(state, **kwargs)
            else:
                state = _frozen_module_call(module, state, **kwargs)
            if visit_hook is not None:
                try:
                    replacement = visit_hook(
                        VisitContext(visit, module_index, module, before, state, kwargs)
                    )
                except StopSchedule:
                    outputs.append(state)
                    break
                if replacement is not None:
                    state = replacement
            outputs.append(state)
        return ScheduleExecution(state, tuple(outputs))
