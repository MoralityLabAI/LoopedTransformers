"""DEUTERIUM-NATIVE-02 shared-adversary pass-indexed PGD reconstruction.

The registration requires that the PGD adversary, source-token budget and
effective objective coefficients match stock. This retains *one* adversarial
source [batch,81,17] for all four applications, exactly like the frozen stock
PGD loss. Only the per-pass CI floor becomes depth indexed. Recurrent weight
factors and the optional 17th residual component remain shared.

This is an adapter engineering primitive. The first source-only parity tests
must pass before any new supervised teacher or VPD optimization data are used.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from param_decomp.metrics.pgd_utils import _run_pgd_loop, get_pgd_init_tensor
from param_decomp.models.batch_and_loss_fns import recon_loss_kl
from param_decomp.models.component_model import OutputWithCache
from mechinterp.deuterium.native_trm.adapter02.pass_indexed import (
    PassIndexedComponentModel, pass_indexed_mask_infos,
    TARGET,N_LOOPS,MASK_COMPONENTS,EXPECTED_IN,EXPECTED_OUT,
)


def validate_depth_ci(ci: Tensor, batch_size: int) -> None:
    if tuple(ci.shape) != (N_LOOPS,batch_size,81,MASK_COMPONENTS):
        raise ValueError(f"wrong pass-indexed CI shape: {tuple(ci.shape)}")
    if not bool(torch.isfinite(ci).all()):
        raise ValueError("nonfinite clean-state causal importances")
    if bool((ci < 0).any()) or bool((ci > 1).any()):
        raise ValueError("CI lower leaky values outside [0,1]")


def shared_adversarial_source_shape(ci: Tensor, weight_delta: Tensor | None) -> tuple[int,...]:
    """Prove same PGD degrees of freedom as the stock one-call adapter."""
    validate_depth_ci(ci,ci.shape[1])
    return (ci.shape[1],81,MASK_COMPONENTS+(1 if weight_delta is not None else 0))


def interpolate_shared_noise(ci: Tensor, adversary: Tensor,
                             weight_delta: Tensor | None) -> dict:
    """One adversarial tensor interpolated into four distinct physical CI floors."""
    validate_depth_ci(ci,ci.shape[1])
    shape = shared_adversarial_source_shape(ci,weight_delta)
    if tuple(adversary.shape)!=shape:
        raise ValueError(f"shared PGD adversary must have stock shape {shape}, got {tuple(adversary.shape)}")
    component=adversary[...,:MASK_COMPONENTS].unsqueeze(0)
    mask=ci+(1-ci)*component
    residual_mask=None
    if weight_delta is not None:
        if tuple(weight_delta.shape)!=(EXPECTED_OUT,EXPECTED_IN):
            raise ValueError("residual target matrix changed shape")
        residual_mask=adversary[...,MASK_COMPONENTS:].unsqueeze(0).expand(N_LOOPS,-1,-1,-1)
    return pass_indexed_mask_infos(mask,weight_delta=weight_delta,residual_mask=residual_mask)


def pgd_pass_indexed_recon_loss(
    *, model: PassIndexedComponentModel, batch: Any, ci: Tensor,
    target_out: Tensor, weight_deltas: dict[str,Tensor] | None,
    pgd_config: Any, reconstruction_loss: Callable=recon_loss_kl,
) -> Tensor:
    """Stock-matched two-step PGD over one shared source at all four depths.

    Uses the frozen upstream _run_pgd_loop for identical signed-gradient
    update, step size, clamp, final replay, and loss-example denominator.
    The only novel operation is applying four CI floors before each stock
    adversarial-mask interpolation. There are no additional PGD parameters.
    """
    if not isinstance(model,PassIndexedComponentModel):
        raise TypeError("expected pass-indexed component model")
    if not isinstance(batch,(tuple,list)) or len(batch)!=1:
        raise ValueError("expected original stock training batch (input_tensor,)")
    source = batch[0]
    validate_depth_ci(ci,len(source))
    if weight_deltas is not None and set(weight_deltas)!={TARGET}:
        raise ValueError("expected only recurrent shared residual matrix")
    matrix = None if weight_deltas is None else weight_deltas[TARGET]
    if pgd_config.mask_scope not in ("unique_per_datapoint","shared_across_batch"):
        raise ValueError("unsupported original stock PGD mask scope")
    if pgd_config.mask_scope == "unique_per_datapoint":
        shape = shared_adversarial_source_shape(ci,matrix)
    else:
        shape = (1,1,MASK_COMPONENTS+(1 if matrix is not None else 0))
    adverse = get_pgd_init_tensor(pgd_config.init,shape,device=ci.device).requires_grad_(True)

    def forward_():
        if pgd_config.mask_scope == "shared_across_batch":
            mask_source=adverse.expand(*shared_adversarial_source_shape(ci,matrix))
        else:
            mask_source=adverse
        info=interpolate_shared_noise(ci,mask_source,matrix)
        output = model(batch,mask_infos=info)
        return reconstruction_loss(output,target_out)

    loss_sum,n_examples=_run_pgd_loop({TARGET:adverse},pgd_config,forward_)
    if n_examples != ci.shape[1]*81:
        raise RuntimeError("pass-indexed PGD changed original token loss denominator")
    return loss_sum/n_examples


def source_only_objective(
    model: PassIndexedComponentModel, batch: tuple[Tensor], cfg: Any,
    *, mask_policy: str = "native_pass",
) -> dict[str,Tensor]:
    """One stock-matched loss evaluation for source-independent gradient parity.

    Reuses exact upstream original loss functions, optimizer coefficients
    and output KL denominator. The only changed tensor is pass-indexed CI.
    """
    from param_decomp.metrics.faithfulness_loss import faithfulness_loss
    from param_decomp.metrics.importance_minimality_loss import importance_minimality_loss

    clean = model(batch,cache_type="input")
    if not isinstance(clean,OutputWithCache):
        raise RuntimeError("original physical source cache unavailable")
    ci = model.calc_causal_importances(clean.cache,sampling=cfg.sampling)
    lower = ci.lower_leaky[TARGET]
    upper = ci.upper_leaky[TARGET]
    if mask_policy == "last_repeat":
        # Engineering identity test only: same input-derived CI at four passes,
        # with the same backward graph as the frozen stock final-call adapter.
        lower = lower[-1:].expand(N_LOOPS,-1,-1,-1)
        upper = upper[-1:].expand(N_LOOPS,-1,-1,-1)
    elif mask_policy != "native_pass":
        raise ValueError("mask_policy must be native_pass or last_repeat")
    validate_depth_ci(lower,len(batch[0]))
    if tuple(upper.shape) != tuple(lower.shape):
        raise RuntimeError("depth-indexed upper/lower CI shape disagreement")
    # All losses reuse the frozen original factor/CI network and the original
    # L_p mean convention, with four-depth outputs divided by four examples.
    weight_deltas = model.calc_weight_deltas()
    losses = {}
    for one in cfg.loss_metric_configs:
        if one.classname == "FaithfulnessLoss":
            losses["faithfulness"] = faithfulness_loss(weight_deltas)
        elif one.classname == "ImportanceMinimalityLoss":
            losses["importance_minimality"] = importance_minimality_loss(
                ci_upper_leaky={TARGET:upper},
                current_frac_of_training=1.0,
                eps=one.eps,pnorm=one.pnorm,beta=one.beta,
                p_anneal_start_frac=one.p_anneal_start_frac,
                p_anneal_final_p=one.p_anneal_final_p,
                p_anneal_end_frac=one.p_anneal_end_frac,
            )
        elif one.classname == "PGDReconLoss":
            losses["pgd_reconstruction"] = pgd_pass_indexed_recon_loss(
                model=model,batch=batch,ci=lower,target_out=clean.output,
                weight_deltas=weight_deltas if cfg.use_delta_component else None,
                pgd_config=one,reconstruction_loss=recon_loss_kl,
            )
        else:
            raise ValueError("unexpected VPD loss form")
    return losses
