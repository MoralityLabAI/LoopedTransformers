"""DEUTERIUM-NATIVE-02: source-only pass-indexed ComponentModel adapter.

The original DEUTERIUM-NATIVE-02 protocol was independently committed before
this implementation. This adapter changes how mask values are indexed across
the four real calls to the SAME recurrent nn.Linear; no weights are untied.

Stock param_decomp's source is not modified or vendored here. Its components,
CI network, sigmoid and original native recurrent forward are reused directly.

Important: a stock PGD adversary has ONE source per (batch, token, component),
shared by all recurrent applications. Our matched adapter keeps that source
shape fixed while applying CI masks computed at all four clean physical depths.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from param_decomp.models.component_model import (
    ComponentModel, OutputWithCache, move_batch_to_device
)
from param_decomp.models.components import ComponentsMaskInfo

TARGET = "block.chan_mix.2"
N_LOOPS = 4
MASK_COMPONENTS = 16
EXPECTED_IN = 128
EXPECTED_OUT = 48
CacheType = Literal["input", "output", "component_acts", "none"]


def _at_depth(value: Tensor, depth: int, ndim_single: int, shape_single: tuple[int, ...],
              name: str) -> Tensor:
    """Select a depth slot or reuse an unindexed tensor, with strict shape guards."""
    if value.ndim == ndim_single:
        selected = value
    elif value.ndim == ndim_single + 1 and value.shape[0] == N_LOOPS:
        selected = value[depth]
    else:
        raise ValueError(f"{name}: expected rank {ndim_single} or depth-leading rank {ndim_single+1}, got {tuple(value.shape)}")
    if tuple(selected.shape) != shape_single:
        raise ValueError(f"{name}: expected {shape_single}, got {tuple(selected.shape)}")
    return selected


class PassIndexedComponentModel(ComponentModel):
    """Drop-in ComponentModel weight factors with explicit call-indexed masks.

    cache_type='input' returns ONE key TARGET with tensor [4,B,81,128], so
    the original shared layerwise CI network naturally evaluates every pass.
    mask_infos[TARGET] may use either [B,81,C] (uniform stock equivalence)
    or [4,B,81,C] (one source-derived mask per recurrent application).
    The weight-delta mask likewise accepts [B,81] or [4,B,81].

    The pass counter is scoped to one complete forward, reset between
    invocations, and checks exactly four calls. The native model is never
    modified, and all forward hooks are removed in a finally block.
    """

    def __init__(self, *, target_model: nn.Module, run_batch: Callable,
                 module_path_info: list, ci_config: Any, sigmoid_type: Any):
        super().__init__(target_model=target_model,run_batch=run_batch,
                         module_path_info=module_path_info,
                         ci_config=ci_config,sigmoid_type=sigmoid_type)
        if self.target_module_paths != [TARGET] or self.module_to_c[TARGET] != MASK_COMPONENTS:
            raise ValueError("DEUTERIUM-NATIVE-02 allows one original recurrent target with 16 shared factors")
        if getattr(target_model, "n_steps", None) != N_LOOPS:
            raise ValueError("exactly four native recurrent passes required")
        target = target_model.get_submodule(TARGET)
        if not isinstance(target, nn.Linear) or tuple(target.weight.shape) != (EXPECTED_OUT, EXPECTED_IN):
            raise ValueError("original tied target must be nn.Linear(128,48)")

    def forward(self, batch: Any,
                mask_infos: dict[str, ComponentsMaskInfo] | None = None,
                cache_type: CacheType = "none") -> Tensor | OutputWithCache:
        if mask_infos is None and cache_type == "none":
            return super().forward(batch, mask_infos=None, cache_type="none")
        if cache_type not in ("input","output","component_acts","none"):
            raise ValueError("unsupported cache type")
        if mask_infos is not None and set(mask_infos) != {TARGET}:
            raise ValueError("pass-indexed replay requires exactly one original recurrent target")

        batch = move_batch_to_device(batch, next(self.parameters()).device)
        cache_inputs: list[Tensor] = []
        cache_outputs: list[Tensor] = []
        cache_acts_pre: list[Tensor] = []
        cache_acts_post: list[Tensor] = []
        count = 0
        component = self.components[TARGET]

        def hook(_module: nn.Module, args: tuple, kwargs: dict, output: Tensor):
            nonlocal count
            depth = count
            count += 1
            if depth >= N_LOOPS:
                raise RuntimeError("recurrent module invoked more than 4 times")
            if kwargs or len(args) != 1 or not isinstance(args[0], Tensor):
                raise RuntimeError("native original target call signature drift")
            input_ = args[0]
            if tuple(input_.shape[1:]) != (81,EXPECTED_IN):
                raise RuntimeError("original native pre-weight input shape drift")
            if cache_type == "input":
                cache_inputs.append(input_)
            if mask_infos is None:
                if cache_type == "output":
                    cache_outputs.append(output)
                return None

            info = mask_infos[TARGET]
            shape = (input_.shape[0],input_.shape[1])
            mask = _at_depth(info.component_mask,depth,3,(*shape,MASK_COMPONENTS),"component_mask")
            wdelta = None
            if info.weight_delta_and_mask is not None:
                matrix, dmask = info.weight_delta_and_mask
                if tuple(matrix.shape) != (EXPECTED_OUT,EXPECTED_IN):
                    raise ValueError("weight-delta matrix shape drift")
                wdelta = (matrix,_at_depth(dmask,depth,2,shape,"weight_delta_mask"))
            acts_cache = {} if cache_type == "component_acts" else None
            computed = component(input_,mask=mask,weight_delta_and_mask=wdelta,
                                 component_acts_cache=acts_cache)
            if acts_cache is not None:
                cache_acts_pre.append(acts_cache["pre_detach"])
                cache_acts_post.append(acts_cache["post_detach"])
            if isinstance(info.routing_mask,str):
                if info.routing_mask != "all":
                    raise ValueError("unsupported routing mask string")
                final = computed
            else:
                routing = _at_depth(info.routing_mask,depth,2,shape,"routing_mask")
                final = torch.where(routing[...,None],computed,output)
            if cache_type == "output":
                cache_outputs.append(final)
            return final

        module = self.target_model.get_submodule(TARGET)
        handle = module.register_forward_hook(hook,with_kwargs=True)
        try:
            output = self._run_batch(self.target_model,batch)
        finally:
            handle.remove()

        if count != N_LOOPS:
            raise RuntimeError(f"expected four recurrent target calls, got {count}")
        cache: dict[str,Tensor] = {}
        if cache_type == "input":
            cache[TARGET] = torch.stack(cache_inputs,0)
        elif cache_type == "output":
            cache[TARGET] = torch.stack(cache_outputs,0)
        elif cache_type == "component_acts":
            cache[TARGET+"_pre_detach"] = torch.stack(cache_acts_pre,0)
            cache[TARGET+"_post_detach"] = torch.stack(cache_acts_post,0)
        if cache_type == "none":
            return output
        return OutputWithCache(output=output,cache=cache)


def pass_indexed_mask_infos(ci: Tensor, *, weight_delta: Tensor | None = None,
                            residual_mask: Tensor | None = None) -> dict[str, ComponentsMaskInfo]:
    """Build a strictly shape-checked input-only masked pass index.

    Every mask's leading dimension is physical recurrent depth; its batch and
    token dimensions must remain fixed across the four tied applications.
    """
    if ci.ndim != 4 or ci.shape[0] != N_LOOPS or ci.shape[-2:] != (81,MASK_COMPONENTS):
        raise ValueError("CI masks must be four passes × batch × 81 tokens × 16 shared factors")
    extra = None
    if (weight_delta is None) != (residual_mask is None):
        raise ValueError("weight residual matrix and mask must be provided together")
    if weight_delta is not None:
        if tuple(weight_delta.shape)!=(EXPECTED_OUT,EXPECTED_IN):
            raise ValueError("wrong original residual weight dimensions")
        if residual_mask.ndim != 4 or residual_mask.shape != ci.shape[:-1]+(1,):
            raise ValueError("residual mask shape must [4,B,81,1]")
        extra = (weight_delta,residual_mask.squeeze(-1))
    return {TARGET:ComponentsMaskInfo(component_mask=ci,weight_delta_and_mask=extra)}
