"""Native DT05 causal instruments: batched VJP Gramians, dynamic transport, routing, and causal rank.

Implements the four DT05 mechanistic extensions under strict BPD/ECA/EvalMath contracts:
1. Chunked/batched VJP causal Jacobians and matrix-free protected task Gramians (no [D,D] allocation).
2. Multi-step JVP subspace transport diagnostics + EvalMath-gated transported-vs-static intervention protocol.
3. Autonomous sparse factor routing vs teacher-scheduled fidelity, fixed, and random controls on identical V/U.
4. Depth-indexed causal rank separating statistical dimension, linearly decodable dimension, and native
   future-response causal dimension (with both proper-scoring KL and behavioral headroom per BPD-ECA-03/04/05).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mechinterp.bpd.experiments.nc09_fixed12.calibration.evalmath_preflight import preflight as evalmath_preflight
from ..dt03_causal import DualState, native_resume_recipient, whole_case_recovery
from ..dt03_tasks import blank_accuracy
from .evaluation import evaluate_policy
from .resources import MemoryBudget, memory_snapshot
from .techniques import (
    HeadNullProjector,
    OnlineTopKLinear,
    principal_angles_deg,
    transport_basis_jvp,
)
from .training import model_digest, routing_control, tensor_digest


def _spectral_summary(eigenvalues: Tensor) -> dict:
    """Compute participation ratio, Shannon entropy effective rank, and cumulative trace ranks."""
    vals = eigenvalues.detach().double().clamp_min(0.0)
    vals = torch.sort(vals, descending=True).values
    total = float(vals.sum().item())
    if total <= 0.0 or vals.numel() == 0:
        return {
            "trace": 0.0,
            "participation_ratio": 0.0,
            "entropy_effective_rank": 0.0,
            "rank_90pct_variance": 0,
            "rank_99pct_variance": 0,
            "top_eigenvalues": [],
        }
    sq_sum = float(vals.pow(2).sum().item())
    pr = (total * total) / max(sq_sum, 1e-30)
    probs = vals / total
    nz = probs[probs > 0]
    entropy = float(-(nz * nz.log()).sum().item())
    cum = torch.cumsum(probs, dim=0)
    r90 = int(torch.searchsorted(cum, torch.tensor(0.90, dtype=torch.float64, device=cum.device)).item()) + 1
    r99 = int(torch.searchsorted(cum, torch.tensor(0.99, dtype=torch.float64, device=cum.device)).item()) + 1
    return {
        "trace": total,
        "participation_ratio": float(pr),
        "entropy_effective_rank": float(math.exp(entropy)),
        "rank_90pct_variance": min(int(vals.numel()), r90),
        "rank_99pct_variance": min(int(vals.numel()), r99),
        "top_eigenvalues": vals[:16].cpu().tolist(),
    }


def batched_observer_jacobian_vjp(
    single_case_observable: Callable[[Tensor, Tensor], Tensor],
    states: Tensor,
    contexts: Tensor,
    *,
    chunk_size: int = 16,
) -> Tensor:
    """Compute exact per-case ``[B, M, D]`` Jacobian via chunked ``vmap(vjp)`` without cross-case coupling.

    ``single_case_observable(state_1d, context_1d) -> out_1d`` operates on one example at a time,
    making cross-example derivative leakage structurally zero while batching all ``B`` examples
    and ``chunk_size`` output cotangents per backward pass on GPU or CPU.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if states.ndim != 2 or len(states) == 0 or not bool(torch.isfinite(states).all()):
        raise ValueError("finite 2D [batch, state_dim] states required")
    if len(contexts) != len(states):
        raise ValueError("contexts batch dimension must match states")

    bsz, state_dim = states.shape
    x = states.detach()

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        sample_out = torch.func.vmap(single_case_observable)(x[:1], contexts[:1])
        if sample_out.ndim < 2 or not bool(torch.isfinite(sample_out).all()):
            raise ValueError("single_case_observable must return finite outputs")
        out_dim = int(sample_out[0].numel())

        def case_jac_chunk(state_b: Tensor, ctx_b: Tensor, basis_chunk: Tensor) -> tuple[Tensor, Tensor]:
            def flat_fn(sb: Tensor) -> Tensor:
                return single_case_observable(sb, ctx_b).reshape(-1)

            y_b, vjp_fn = torch.func.vjp(flat_fn, state_b)
            grads = torch.func.vmap(lambda cot: vjp_fn(cot)[0])(basis_chunk)
            return y_b, grads

        batched_chunk = torch.func.vmap(case_jac_chunk, in_dims=(0, 0, None))
        eye = torch.eye(out_dim, device=x.device, dtype=x.dtype)
        jac_chunks = []
        for offset in range(0, out_dim, chunk_size):
            cot_chunk = eye[offset:offset + chunk_size]
            y_sub, j_sub = batched_chunk(x, contexts, cot_chunk)
            if offset == 0 and not bool(torch.isfinite(y_sub).all()):
                raise ValueError("nonfinite observer output during VJP")
            j_sub = j_sub.detach()
            if not bool(torch.isfinite(j_sub).all()):
                raise ValueError("nonfinite Jacobian entries produced by chunked VJP")
            jac_chunks.append(j_sub)
    return torch.cat(jac_chunks, dim=1)


def select_protected_task_gramian_vjp(
    model: nn.Module,
    calib_tokens: Tensor,
    calib_packed_states: Tensor,
    *,
    fit_offsets: Sequence[int] = (1, 3, 5),
    rank: int = 8,
    chunk_size: int = 16,
) -> tuple[Tensor, dict]:
    """Select top-``rank`` orthonormal causal directions in ``ker(DP)`` without ``[D, D]`` matrices.

    Uses matrix-free ``HeadNullProjector.project_dual`` on chunked VJP cotangent rows and thin SVD
    in ``float64``, producing exact mathematical parity with ``dt03_causal.select_full_coordinate_block``
    while scaling cleanly to 7M TRM dual states (``D = 99,328``).
    """
    if not fit_offsets or len(set(fit_offsets)) != len(fit_offsets) or any(o < 1 for o in fit_offsets):
        raise ValueError("fit_offsets must be distinct positive integers")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    dev = next(model.parameters()).device
    tokens = calib_tokens.to(dev)
    states = calib_packed_states.to(dev)
    bsz, full_dim = states.shape
    t_len = int(model.total_len)
    puz = int(model.puzzle_emb_len)
    d_hid = int(model.hidden_size)
    digits = int(math.isqrt(int(model.seq_len)))
    if full_dim != 2 * t_len * d_hid:
        raise ValueError("calib_packed_states width does not match model dual state")

    projector = HeadNullProjector(model.lm_head.weight)
    project = lambda vecs: projector.project_dual(vecs, t_len, puz)
    with torch.no_grad():
        embeds = model.encode_inputs(tokens).detach()

    projected_rows = []
    for offset in fit_offsets:
        def single_obs(s_1d: Tensor, emb_1d: Tensor, off: int = int(offset)) -> Tensor:
            dual = s_1d.reshape(2, t_len, d_hid)
            zh = dual[0].unsqueeze(0)
            zl = dual[1].unsqueeze(0)
            for _ in range(off):
                zh, zl = model.act_step(zh, zl, emb_1d.unsqueeze(0))
            logits = model.digit_logits(zh)[0, :, :digits].float()
            return torch.softmax(logits, dim=-1)

        jac = batched_observer_jacobian_vjp(single_obs, states, embeds, chunk_size=chunk_size)
        # Project each [M, D] row into ker(DP) without forming a [D, K] basis matrix!
        flat_rows = jac.reshape(-1, full_dim)
        proj = project(flat_rows).double()
        projected_rows.append(proj)

    stacked = torch.cat(projected_rows, dim=0) / math.sqrt(float(bsz * len(fit_offsets)))
    # Dual SVD when N_rows <= full_dim avoids [full_dim, full_dim] allocation
    if stacked.shape[0] <= stacked.shape[1]:
        u_mat, svals, vh = torch.linalg.svd(stacked, full_matrices=False)
        _ = u_mat
        eigvals = svals.pow(2)
        actual_rank = min(rank, int((svals > 1e-11 * max(1.0, float(svals.max().item()))).sum().item()))
        if actual_rank < 1:
            raise RuntimeError("Projected task Gramian has zero numerical rank")
        raw_basis = vh[:actual_rank].T
    else:
        _, svals, vh = torch.linalg.svd(stacked, full_matrices=False)
        eigvals = svals.pow(2)
        actual_rank = min(rank, int((svals > 1e-11 * max(1.0, float(svals.max().item()))).sum().item()))
        raw_basis = vh[:actual_rank].T

    # Re-project and QR polish in float64 so columns lie in ker(DP) to machine precision
    polished = project(raw_basis.to(dtype=states.dtype).T).T.double()
    q64, _ = torch.linalg.qr(polished, mode="reduced")
    basis = q64[:, :actual_rank].to(device=dev, dtype=states.dtype).contiguous()
    orth_err = float((basis.double().T @ basis.double() - torch.eye(actual_rank, device=dev, dtype=torch.float64)).abs().max().item())
    if orth_err > 1e-4:
        raise AssertionError(f"Selected VJP Gramian basis lost orthogonality: {orth_err}")

    spec = _spectral_summary(eigvals)
    return basis, {
        "fit_offsets": list(fit_offsets),
        "requested_rank": rank,
        "selected_rank": actual_rank,
        "orthogonality_error": orth_err,
        "full_state_dim": full_dim,
        "output_coordinates_per_case": int(model.seq_len) * digits,
        "dense_state_gramian_allocated": False,
        **spec,
    }


def _sample_random_protected_basis(
    projector: HeadNullProjector,
    total_len: int,
    prefix_len: int,
    state_dim: int,
    rank: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if rank == 0:
        return torch.empty((state_dim, 0), device=device, dtype=dtype)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((rank + 4, state_dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    proj = projector.project_dual(raw, total_len, prefix_len).T.double()
    u_mat, svals, _ = torch.linalg.svd(proj, full_matrices=False)
    valid = int((svals > 1e-8 * max(1.0, float(svals.max().item()))).sum().item())
    use_r = min(rank, valid)
    return u_mat[:, :use_r].to(dtype=dtype).contiguous()


def _matrix_free_signed_jacobian_sham(
    jac_case: Tensor,
    candidate_edit_1d: Tensor,
    projector: HeadNullProjector,
    total_len: int,
    prefix_len: int,
    generator: torch.Generator,
) -> tuple[Tensor, dict]:
    """Construct privileged signed-Jacobian + L2-dose matched sham in ``ker(DP)`` without ``[D, K]`` Q."""
    dev = candidate_edit_1d.device
    dtype = candidate_edit_1d.dtype
    project = lambda vecs: projector.project_dual(vecs, total_len, prefix_len)
    # Project Jacobian rows into ker(DP): J_prot [M, D]
    j_prot = project(jac_case).double()
    vec = project(candidate_edit_1d.unsqueeze(0))[0].double()
    # Minimum-norm solution in ker(DP) matching J_prot @ control == J_prot @ vec:
    # required = J_prot^T (J_prot J_prot^T)^+ (J_prot vec)
    gram_out = j_prot @ j_prot.T  # [M, M] tiny!
    target_resp = j_prot @ vec
    alpha = torch.linalg.pinv(gram_out, rtol=1e-11) @ target_resp
    required = j_prot.T @ alpha
    required = project(required.to(dtype=dtype).unsqueeze(0))[0].double()

    rand_raw = torch.randn(vec.shape, generator=generator, dtype=torch.float64).to(dev)
    rand_prot = project(rand_raw.to(dtype=dtype).unsqueeze(0))[0].double()
    # Project out rowspace(J_prot) so J_prot @ rand_null == 0 while remaining in ker(DP)!
    rand_null = rand_prot - j_prot.T @ (torch.linalg.pinv(gram_out, rtol=1e-11) @ (j_prot @ rand_prot))
    rand_null = project(rand_null.to(dtype=dtype).unsqueeze(0))[0].double()
    rand_null = rand_null - j_prot.T @ (torch.linalg.pinv(gram_out, rtol=1e-11) @ (j_prot @ rand_null))
    r_norm = rand_null.norm().clamp_min(1e-12)
    rand_unit = rand_null / r_norm

    radius = (vec.pow(2).sum() - required.pow(2).sum()).clamp_min(0.0).sqrt()
    control64 = required + radius * rand_unit
    control = project(control64.to(dtype=dtype).unsqueeze(0))[0]
    control64 = control.double()
    signed_err = float((j_prot @ control64 - target_resp).abs().max().item())
    dose_err = float((control64.norm() - vec.norm()).abs().item())
    sep = float((control64 - vec).norm().item())
    return control, {
        "signed_full_task_jacobian_max_error": signed_err,
        "delivered_l2_error": dose_err,
        "candidate_sham_separation": sep,
        "privileged": True,
    }


def multistep_transport_audit(
    model: nn.Module,
    tokens: Tensor,
    *,
    calib_tokens: Tensor | None = None,
    basis_mode: str = "task_gramian",
    rank: int = 8,
    start_depth: int = 1,
    num_steps: int = 7,
    perturbation_doses: Sequence[float] = (1e-3, 1e-2, 1e-1, 0.5),
    seed: int = 260500031,
) -> dict:
    """Carry a protected subspace across ``num_steps`` native H/L transitions via thin JVP (§4.A).

    Records at every step ``1..num_steps``:
    - One-step closure defect (against previous basis ``Q_{k-1}``) and multi-step closure defect (against static ``Q_0``),
    - Principal-angle drift vs initial ``Q_0`` and vs previous step ``Q_{k-1}``,
    - Stepwise and cumulative pushed-frame singular-value decay / growth,
    - Rank loss (without QR padding),
    - Projection escape out of ``ker(DP)`` into ``rowspace(lm_head)``,
    - Finite-perturbation nonlinear response vs multi-step local Jacobian prediction across ``perturbation_doses``.
    """
    if len(tokens) != 1:
        raise ValueError("multistep_transport_audit operates on a single board trajectory [1, seq_len]")
    if start_depth < 1 or num_steps < 1:
        raise ValueError("start_depth and num_steps must be positive integers")
    if basis_mode not in ("task_gramian", "random_protected"):
        raise ValueError("basis_mode must be task_gramian or random_protected")

    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        tok = tokens.to(dev)
        t_len = int(model.total_len)
        puz = int(model.puzzle_emb_len)
        d_hid = int(model.hidden_size)
        full_dim = 2 * t_len * d_hid
        projector = HeadNullProjector(model.lm_head.weight)
        project = lambda vecs: projector.project_dual(vecs, t_len, puz)

        with torch.no_grad():
            embed = model.encode_inputs(tok).detach()
            zh, zl = model.init_states(1)
            for _ in range(start_depth):
                zh, zl = model.act_step(zh, zl, embed)
            state0 = torch.cat((zh.flatten(1), zl.flatten(1)), dim=-1).detach()

        if basis_mode == "task_gramian":
            cal_tok = (calib_tokens if calib_tokens is not None else tokens).to(dev)
            with torch.no_grad():
                c_emb = model.encode_inputs(cal_tok)
                ch, cl = model.init_states(len(cal_tok))
                for _ in range(start_depth):
                    ch, cl = model.act_step(ch, cl, c_emb)
                cal_states = torch.cat((ch.flatten(1), cl.flatten(1)), dim=-1).detach()
            basis0, gram_meta = select_protected_task_gramian_vjp(
                model, cal_tok, cal_states, fit_offsets=(1, 3, 5), rank=rank,
            )
        else:
            basis0 = _sample_random_protected_basis(
                projector, t_len, puz, full_dim, rank,
                seed=seed, device=dev, dtype=state0.dtype,
            )
            gram_meta = None

        def step_fn(packed: Tensor) -> Tensor:
            dual = packed.reshape(1, 2, t_len, d_hid)
            hn, ln = model.act_step(dual[:, 0], dual[:, 1], embed)
            return torch.cat((hn.flatten(1), ln.flatten(1)), dim=-1)

        # Track both (a) re-orthonormalized projected basis Q_k and
        # (b) unnormalized multi-step linear pushforward frame W_k = DF^{(k)}(s_0) Q_0!
        q_cur = basis0
        state_cur = state0
        w_unnorm = basis0.clone()  # [full_dim, r0]
        v0_unit = basis0[:, 0].detach()  # Leading protected direction for finite-dose probe
        s0_norm = float(state0.norm().item())

        # Pre-seed finite-perturbation trajectories s_eps(0) = s_0 + eps * ||s_0|| * v0
        pert_states = {
            float(eps): (state0 + float(eps) * s0_norm * v0_unit.unsqueeze(0)).detach()
            for eps in perturbation_doses
        }

        steps_report = []
        q0_64 = basis0.double()
        for k in range(1, num_steps + 1):
            # 1. Push unnormalized frame W_{k-1} through exact local JVP at state_cur
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                def single_push(col: Tensor) -> Tensor:
                    _, d_out = torch.func.jvp(step_fn, (state_cur,), (col.reshape_as(state_cur),))
                    return d_out.reshape(-1)
                w_next = torch.func.vmap(single_push, in_dims=1, out_dims=1)(w_unnorm).detach()

            # 2. Transport orthonormal protected basis Q_{k-1} -> Q_k if rank > 0
            if q_cur.shape[1] > 0:
                q_next, step_jvp = transport_basis_jvp(step_fn, state_cur, q_cur, project=project)
                drift_initial = principal_angles_deg(q0_64, q_next.double())
            else:
                q_next = q_cur
                step_jvp = {
                    "input_rank": 0, "output_rank": 0, "rank_lost": 0,
                    "singular_values": [], "one_step_closure_defect": 1.0,
                    "protected_projection_escape": 0.0,
                    "principal_angle_drift_from_input": principal_angles_deg(q0_64, q_cur.double()),
                }
                drift_initial = principal_angles_deg(q0_64, q_next.double())

            # 3. Advance clean state and finite-perturbation states natively
            with torch.no_grad():
                state_next = step_fn(state_cur).detach()
                finite_vs_linear = {}
                lin_dir = w_next[:, 0].double()
                for eps, p_st in list(pert_states.items()):
                    p_next = step_fn(p_st).detach()
                    pert_states[eps] = p_next
                    delta_nl = (p_next[0] - state_next[0]).double()
                    delta_lin = (eps * s0_norm) * lin_dir
                    nl_norm = float(delta_nl.norm().item())
                    lin_norm = float(delta_lin.norm().item())
                    rel_err = float((delta_nl - delta_lin).norm().item() / max(nl_norm, 1e-30))
                    cos_sim = float(
                        torch.dot(delta_nl, delta_lin).item() / max(nl_norm * lin_norm, 1e-30)
                    )
                    finite_vs_linear[f"{eps:g}"] = {
                        "relative_dose": eps,
                        "nonlinear_response_l2": nl_norm,
                        "jacobian_predicted_l2": lin_norm,
                        "relative_linearization_error": rel_err,
                        "cosine_alignment": cos_sim,
                    }

            # 4. Multi-step cumulative closure defect & singular-value spectrum of W_next
            w64 = w_next.double()
            w_denom = float(w64.norm().clamp_min(1e-30).item())
            cum_closure = float((w64 - q0_64 @ (q0_64.T @ w64)).norm().item() / w_denom)
            w_proj64 = project(w_next.T).T.double()
            cum_escape = float((w64 - w_proj64).norm().item() / w_denom)
            cum_svals = torch.linalg.svdvals(w64).detach().cpu().tolist()

            steps_report.append({
                "step_offset": k,
                "depth": start_depth + k,
                "input_rank": step_jvp["input_rank"],
                "output_rank": step_jvp["output_rank"],
                "step_rank_lost": step_jvp["rank_lost"],
                "cumulative_rank_lost": int(basis0.shape[1]) - int(q_next.shape[1]),
                "one_step_closure_defect": step_jvp["one_step_closure_defect"],
                "multistep_static_closure_defect": cum_closure,
                "one_step_protected_escape": step_jvp["protected_projection_escape"],
                "multistep_protected_escape": cum_escape,
                "step_singular_values": step_jvp["singular_values"],
                "cumulative_pushforward_singular_values": cum_svals,
                "principal_angle_drift_stepwise": step_jvp["principal_angle_drift_from_input"],
                "principal_angle_drift_from_initial": drift_initial,
                "finite_perturbation_vs_jacobian": finite_vs_linear,
            })
            q_cur = q_next
            state_cur = state_next
            w_unnorm = w_next

        return {
            "schema": "BPD-DT05/multistep-transport-audit/v1",
            "basis_mode": basis_mode,
            "start_depth": start_depth,
            "num_steps": num_steps,
            "initial_rank": int(basis0.shape[1]),
            "final_surviving_rank": int(q_cur.shape[1]),
            "gramian_selection": gram_meta,
            "dense_state_jacobian_allocated": False,
            "steps": steps_report,
        }
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class TransportInterventionProtocol:
    """Frozen DT05 development-only protocol comparing JVP-transported vs static subspaces."""

    experiment_id: str = "BPD-DT05-transported-vs-static-gramian-depth8"
    selection_depth: int = 1
    patch_depth: int = 3
    fit_offsets_from_selection: tuple[int, ...] = (1, 3, 5)  # total depths (2, 4, 6)
    eval_total_depths: tuple[int, ...] = (6, 8, 12)
    subspace_rank: int = 8
    corruption_rel_dose: float = 0.85
    n_shams: int = 8
    donor_prob_tolerance: float = 0.10
    observer_tolerance: float = 1e-5
    scientific_min_effect: str = "0.10"
    target_effect: str = "0.15"
    alpha: str = "0.05"
    planned_comparisons: int = 2
    confirm_units_planned: int = 128


def run_transport_vs_static_unit(
    model: nn.Module,
    calib_tokens: Tensor,
    eval_tokens: Tensor,
    eval_gold: Tensor,
    *,
    unit_id: str,
    seed: int,
    protocol: TransportInterventionProtocol = TransportInterventionProtocol(),
) -> dict:
    """Execute one independent model-root comparison of transported vs static causal basis (§4.A).

    1. Fits initial protected Gramian basis ``Q_1`` on ``calib_tokens`` at ``selection_depth=1``
       using strictly supervised fit depths ``(2, 4, 6)``.
    2. Corrupts disjoint ``eval_tokens`` inside ``ker(DP)`` at ``patch_depth=3``.
    3. Transports ``Q_1`` along each board's clean prefix ``1 -> 3`` via thin ``transport_basis_jvp``
       (with ``ker(DP)`` projection and explicit rank-loss tracking) to obtain ``Q_trans``.
    4. Evaluates ``transported_task_basis`` vs ``static_task_basis_dose_matched``, 8 equal-dose
       matched-rank Haar protected shams, 8 privileged signed-Jacobian shams (matched at depth 6),
       ``no_edit``, and ``full_dark_donor`` across ``eval_total_depths=(6, 8, 12)``.
    """
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        cal_tok = calib_tokens.to(dev)
        ev_tok = eval_tokens.to(dev)
        ev_gold = eval_gold.to(dev)
        n_ev = len(ev_tok)
        t_len = int(model.total_len)
        puz = int(model.puzzle_emb_len)
        d_hid = int(model.hidden_size)
        full_dim = 2 * t_len * d_hid
        digits = int(math.isqrt(int(model.seq_len)))
        state_shape = (t_len, d_hid)

        projector = HeadNullProjector(model.lm_head.weight)
        project = lambda vecs: projector.project_dual(vecs, t_len, puz)

        # 1. Calibration states at selection_depth=1 -> select Q_1 using fit depths (2,4,6)
        with torch.no_grad():
            c_emb = model.encode_inputs(cal_tok)
            ch, cl = model.init_states(len(cal_tok))
            for _ in range(protocol.selection_depth):
                ch, cl = model.act_step(ch, cl, c_emb)
            cal_clean = torch.cat((ch.flatten(1), cl.flatten(1)), dim=-1).detach()
        q1_basis, gram_meta = select_protected_task_gramian_vjp(
            model, cal_tok, cal_clean,
            fit_offsets=protocol.fit_offsets_from_selection,
            rank=protocol.subspace_rank,
        )

        # 2. Roll out eval boards to selection_depth=1 and patch_depth=3, transporting Q_1 -> Q_3 per board
        with torch.no_grad():
            ev_emb = model.encode_inputs(ev_tok).detach()
            eh, el = model.init_states(n_ev)
            for _ in range(protocol.selection_depth):
                eh, el = model.act_step(eh, el, ev_emb)
            ev_state_sel = torch.cat((eh.flatten(1), el.flatten(1)), dim=-1).detach()

        transport_steps = protocol.patch_depth - protocol.selection_depth
        transported_bases: list[Tensor] = []
        surviving_ranks: list[int] = []
        closure_defects_step1: list[float] = []
        angle_drifts_deg: list[float] = []
        donor_patch_states: list[Tensor] = []

        for b_i in range(n_ev):
            emb_b = ev_emb[b_i:b_i + 1]
            st_b = ev_state_sel[b_i:b_i + 1]
            q_b = q1_basis

            def step_b(packed: Tensor, eb: Tensor = emb_b) -> Tensor:
                dual = packed.reshape(1, 2, t_len, d_hid)
                hn, ln = model.act_step(dual[:, 0], dual[:, 1], eb)
                return torch.cat((hn.flatten(1), ln.flatten(1)), dim=-1)

            for s_k in range(transport_steps):
                if q_b.shape[1] > 0:
                    q_b, info_b = transport_basis_jvp(step_b, st_b, q_b, project=project)
                    if s_k == 0:
                        closure_defects_step1.append(info_b["one_step_closure_defect"])
                with torch.no_grad():
                    st_b = step_b(st_b).detach()
            donor_patch_states.append(st_b[0])
            transported_bases.append(q_b)
            surviving_ranks.append(int(q_b.shape[1]))
            drift_b = principal_angles_deg(q1_basis.double(), q_b.double())
            angle_drifts_deg.append(drift_b["mean_angle_deg"])

        ev_donor_packed = torch.stack(donor_patch_states, dim=0)  # [n_ev, full_dim]

        # 3. Corrupt recipient states strictly in ker(DP) at patch_depth
        gen_corr = torch.Generator(device="cpu").manual_seed(int(seed) * 197 + 29)
        raw_noise = torch.randn((n_ev, full_dim), generator=gen_corr, dtype=torch.float32).to(device=dev, dtype=ev_donor_packed.dtype)
        prot_noise = project(raw_noise)
        prot_noise = prot_noise / prot_noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        donor_prot_norm = project(ev_donor_packed).norm(dim=-1, keepdim=True).clamp_min(1.0)
        ev_recip_packed = ev_donor_packed + (protocol.corruption_rel_dose * donor_prot_norm) * prot_noise
        diff = project(ev_donor_packed - ev_recip_packed)

        # 4. Build per-case edits: transported, static (matched rank & dose), Haar shams, signed-Jacobian shams
        edit_zero = torch.zeros_like(diff)
        edit_full_dark = diff.clone()
        edit_trans_list = []
        edit_static_list = []
        for b_i in range(n_ev):
            q_tr = transported_bases[b_i]
            r_b = q_tr.shape[1]
            if r_b == 0:
                edit_trans_list.append(torch.zeros_like(diff[b_i]))
                edit_static_list.append(torch.zeros_like(diff[b_i]))
            else:
                e_tr = project(((diff[b_i] @ q_tr) @ q_tr.T).unsqueeze(0))[0]
                dose_b = e_tr.norm().clamp_min(1e-12)
                q_st = q1_basis[:, :r_b]
                e_st_raw = project(((diff[b_i] @ q_st) @ q_st.T).unsqueeze(0))[0]
                e_st = project((e_st_raw * (dose_b / e_st_raw.norm().clamp_min(1e-12))).unsqueeze(0))[0]
                edit_trans_list.append(e_tr)
                edit_static_list.append(e_st)
        edit_trans = torch.stack(edit_trans_list, dim=0)
        edit_static = torch.stack(edit_static_list, dim=0)
        trans_doses = edit_trans.norm(dim=-1)

        # 8 equal-dose matched-rank Haar protected shams
        haar_shams: list[Tensor] = []
        for s_idx in range(protocol.n_shams):
            per_case = []
            for b_i in range(n_ev):
                r_b = surviving_ranks[b_i]
                if r_b == 0:
                    per_case.append(torch.zeros_like(diff[b_i]))
                else:
                    q_h = _sample_random_protected_basis(
                        projector, t_len, puz, full_dim, r_b,
                        seed=int(seed) * 3001 + s_idx * 97 + b_i,
                        device=dev, dtype=diff.dtype,
                    )
                    e_h = project(((diff[b_i] @ q_h) @ q_h.T).unsqueeze(0))[0]
                    e_h = project((e_h * (trans_doses[b_i] / e_h.norm().clamp_min(1e-12))).unsqueeze(0))[0]
                    per_case.append(e_h)
            haar_shams.append(torch.stack(per_case, dim=0))

        # 8 privileged signed-Jacobian + L2-dose matched shams matched at depth 6 (offset = 6 - patch_depth)
        fit_boundary_offset = 6 - protocol.patch_depth

        def obs_at_d6(s_1d: Tensor, emb_1d: Tensor) -> Tensor:
            dual = s_1d.reshape(2, t_len, d_hid)
            zh, zl = dual[0].unsqueeze(0), dual[1].unsqueeze(0)
            for _ in range(fit_boundary_offset):
                zh, zl = model.act_step(zh, zl, emb_1d.unsqueeze(0))
            return torch.softmax(model.digit_logits(zh)[0, :, :digits].float(), dim=-1)

        jac_d6 = batched_observer_jacobian_vjp(obs_at_d6, ev_recip_packed, ev_emb, chunk_size=16)
        signed_shams: list[Tensor] = []
        sham_certs: list[dict] = []
        for s_idx in range(protocol.n_shams):
            s_gen = torch.Generator(device="cpu").manual_seed(int(seed) * 5003 + 701 + s_idx)
            per_case_ctrl = []
            max_sig_err = 0.0
            max_dose_err = 0.0
            max_sep = 0.0
            for b_i in range(n_ev):
                ctrl_b, cert_b = _matrix_free_signed_jacobian_sham(
                    jac_d6[b_i], edit_trans[b_i], projector, t_len, puz, s_gen,
                )
                per_case_ctrl.append(ctrl_b)
                max_sig_err = max(max_sig_err, cert_b["signed_full_task_jacobian_max_error"])
                max_dose_err = max(max_dose_err, cert_b["delivered_l2_error"])
                max_sep = max(max_sep, cert_b["candidate_sham_separation"])
            signed_shams.append(torch.stack(per_case_ctrl, dim=0))
            sham_certs.append({
                "sham_index": s_idx,
                "signed_full_task_jacobian_max_error": max_sig_err,
                "delivered_l2_error": max_dose_err,
                "max_candidate_sham_separation": max_sep,
            })

        arms: dict[str, Tensor] = {
            "no_edit": edit_zero,
            "full_dark_donor": edit_full_dark,
            "transported_task_basis": edit_trans,
            "static_task_basis_dose_matched": edit_static,
        }
        for s_idx in range(protocol.n_shams):
            arms[f"haar_sham_{s_idx}"] = haar_shams[s_idx]
            arms[f"signed_jac_sham_{s_idx}"] = signed_shams[s_idx]

        # Verify immediate 11-vocab observer conservation and prefix lock across every arm
        with torch.no_grad():
            rec_dual = DualState.unflatten(ev_recip_packed, state_shape)
            base_vocab = torch.softmax(model.lm_head(rec_dual.z_h)[:, puz:, :].float(), dim=-1)
            max_obs_err = 0.0
            for arm_name, ed in arms.items():
                pat_dual = DualState.unflatten(ev_recip_packed + ed, state_shape)
                pat_vocab = torch.softmax(model.lm_head(pat_dual.z_h)[:, puz:, :].float(), dim=-1)
                obs_err = float((pat_vocab - base_vocab).abs().max().item())
                if obs_err > protocol.observer_tolerance:
                    raise AssertionError(f"Observer protection violated on {arm_name}: {obs_err}")
                pref_err = max(
                    float((pat_dual.z_h[:, :puz] - rec_dual.z_h[:, :puz]).abs().max().item()),
                    float((pat_dual.z_l[:, :puz] - rec_dual.z_l[:, :puz]).abs().max().item()),
                )
                if pref_err > 1e-7:
                    raise AssertionError(f"Prefix lock violated on {arm_name}: {pref_err}")
                max_obs_err = max(max_obs_err, obs_err)

            # 5. Evaluate native continuations to eval_total_depths (6, 8, 12)
            donor_dual = DualState.unflatten(ev_donor_packed, state_shape)
            by_depth = {}
            for total_d in protocol.eval_total_depths:
                offset = total_d - protocol.patch_depth
                donor_logits = native_resume_recipient(model, ev_tok, donor_dual, offset)[:, :, :digits]
                donor_probs = torch.softmax(donor_logits.float(), dim=-1)
                donor_logp = F.log_softmax(donor_logits.float(), dim=-1)
                depth_arms = {}
                for arm_name, ed in arms.items():
                    cand_dual = DualState.unflatten(ev_recip_packed + ed, state_shape)
                    cand_logits = native_resume_recipient(model, ev_tok, cand_dual, offset)[:, :, :digits]
                    cand_probs = torch.softmax(cand_logits.float(), dim=-1)
                    cand_logp = F.log_softmax(cand_logits.float(), dim=-1)
                    cand_pred = cand_logits.argmax(dim=-1)
                    rec = whole_case_recovery(cand_probs, donor_probs, protocol.donor_prob_tolerance)
                    b_acc = blank_accuracy(cand_pred, ev_gold, ev_tok, blank_token=1)
                    kl_to_donor = float((donor_probs * (donor_logp - cand_logp)).sum(dim=-1).mean().item())
                    depth_arms[arm_name] = {
                        "exact_board_accuracy": b_acc["whole_board_exact"],
                        "blank_cell_accuracy": b_acc["blank_accuracy"],
                        "whole_case_donor_recovery": rec["whole_case_recovery"],
                        "donor_kl_nats": kl_to_donor,
                        "delivered_l2_dose_mean": float(ed.norm(dim=-1).mean().item()),
                    }
                by_depth[str(total_d)] = depth_arms

        d8 = by_depth["8"]
        tr_exact = d8["transported_task_basis"]["exact_board_accuracy"]
        st_exact = d8["static_task_basis_dose_matched"]["exact_board_accuracy"]
        no_exact = d8["no_edit"]["exact_board_accuracy"]
        haar_exacts = [d8[f"haar_sham_{i}"]["exact_board_accuracy"] for i in range(protocol.n_shams)]
        sig_exacts = [d8[f"signed_jac_sham_{i}"]["exact_board_accuracy"] for i in range(protocol.n_shams)]
        mean_haar = sum(haar_exacts) / len(haar_exacts)
        return {
            "schema": "BPD-DT05/transport-vs-static-unit/v1",
            "unit_id": unit_id,
            "seed": seed,
            "protocol": asdict(protocol),
            "model_sha256": model_digest(model),
            "calib_tokens_sha256": tensor_digest(calib_tokens),
            "eval_tokens_sha256": tensor_digest(eval_tokens),
            "gramian_selection": gram_meta,
            "transport_geometry": {
                "transport_steps": transport_steps,
                "mean_step1_closure_defect": sum(closure_defects_step1) / max(1, len(closure_defects_step1)),
                "mean_principal_angle_drift_deg": sum(angle_drifts_deg) / max(1, len(angle_drifts_deg)),
                "surviving_ranks_per_case": surviving_ranks,
                "rank_loss_cases": sum(1 for r in surviving_ranks if r < q1_basis.shape[1]),
            },
            "observer_max_abs_error": max_obs_err,
            "signed_sham_certificates": sham_certs,
            "by_depth": by_depth,
            "depth8_contrasts": {
                "transported_minus_static_exact": tr_exact - st_exact,
                "transported_minus_mean_haar_sham_exact": tr_exact - mean_haar,
                "transported_minus_no_edit_exact": tr_exact - no_exact,
                "static_minus_mean_haar_sham_exact": st_exact - mean_haar,
                "haar_sham_centered_effects": [x - mean_haar for x in haar_exacts],
                "signed_sham_centered_effects": [
                    x - (sum(sig_exacts) / len(sig_exacts)) for x in sig_exacts
                ],
                "transported_kl_gain_over_static_nats": (
                    d8["static_task_basis_dose_matched"]["donor_kl_nats"]
                    - d8["transported_task_basis"]["donor_kl_nats"]
                ),
            },
        }
    finally:
        model.train(was_training)


def evaluate_transport_protocol_with_evalmath(
    unit_receipts: Sequence[dict],
    *,
    protocol: TransportInterventionProtocol = TransportInterventionProtocol(),
    n_eval_boards_per_unit: int = 16,
) -> dict:
    """Run repository EvalMath preflight (§5.1) on independent unit receipts."""
    if not unit_receipts:
        raise ValueError("At least one unit receipt is required for EvalMath preflight")
    pilot_rows = []
    for rec in unit_receipts:
        c8 = rec["depth8_contrasts"]
        eff = Fraction(c8["transported_minus_static_exact"]).limit_denominator(1_000_000)
        shams = [
            str(Fraction(v).limit_denominator(1_000_000))
            for v in c8["haar_sham_centered_effects"]
        ]
        pilot_rows.append({
            "unit_id": str(rec["unit_id"]),
            "paired_effect": str(eff),
            "sham_effects": shams,
        })
    spec = {
        "schema_version": 1,
        "experiment_id": protocol.experiment_id,
        "independent_unit": "independently trained PureSudokuTRM model root",
        "alpha": protocol.alpha,
        "scientific_min_effect": protocol.scientific_min_effect,
        "target_effect": protocol.target_effect,
        "measurement_resolution": f"1/{int(n_eval_boards_per_unit)}",
        "sham_quantile": "0.95",
        "sham_confidence": "0.95",
        "planned_comparisons": protocol.planned_comparisons,
        "confirm_units_planned": protocol.confirm_units_planned,
        "pilot_from_development_only": True,
        "heldout_sealed": True,
        "units_independent": True,
        "pilot": pilot_rows,
    }
    verdict = evalmath_preflight(spec)
    return {
        "schema": "BPD-DT05/transport-evalmath-preflight/v1",
        "input_spec": spec,
        "input_spec_sha256": hashlib.sha256(
            (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest(),
        "evalmath_result": verdict,
    }


@torch.no_grad()
def benchmark_autonomous_routing(
    teacher: nn.Module,
    student: nn.Module,
    eval_tokens_cpu: Tensor,
    eval_gold_cpu: Tensor,
    *,
    depths: tuple[int, ...] = (2, 4, 6, 8, 12),
    microbatch: int = 32,
    seed: int = 2605999,
    budget: MemoryBudget = MemoryBudget(),
) -> dict:
    """Compare online, teacher-scheduled, fixed, and random factor routing on identical V/U (§4.B)."""
    dev = next(teacher.parameters()).device
    max_depth = max(depths)
    digits = int(math.isqrt(int(teacher.seq_len)))

    site_names = [
        name for name, mod in student.named_modules() if isinstance(mod, OnlineTopKLinear)
    ]
    if not site_names:
        raise ValueError("student must contain OnlineTopKLinear factor modules")

    # Verify identical V/U digests across online, fixed, and random arms
    fixed_ctrl = routing_control(student, "fixed", seed)
    random_ctrl = routing_control(student, "random", seed)
    teacher_sched_model = copy.deepcopy(student).eval()

    for other in (fixed_ctrl, random_ctrl, teacher_sched_model):
        for name in site_names:
            m_base: OnlineTopKLinear = student.get_submodule(name)  # type: ignore[assignment]
            m_oth: OnlineTopKLinear = other.get_submodule(name)  # type: ignore[assignment]
            if not (torch.equal(m_base.V, m_oth.V) and torch.equal(m_base.U, m_oth.U)):
                raise AssertionError(f"V/U mismatch on {name}")

    # 1. Roll out dense teacher to collect target probabilities AND teacher-scheduled top-k masks
    teacher.eval()
    student.eval()
    teacher_probs_by_d: dict[int, list[Tensor]] = {d: [] for d in depths}
    teacher_masks_by_chunk: list[dict[str, list[Tensor]]] = []
    online_vs_teacher_overlap_by_step: dict[int, list[float]] = {d: [] for d in range(1, max_depth + 1)}

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for offset in range(0, len(eval_tokens_cpu), microbatch):
        inp = eval_tokens_cpu[offset:offset + microbatch].to(dev)
        chunk_masks: dict[str, list[Tensor]] = {name: [] for name in site_names}
        handles = []
        for name in site_names:
            t_mod = teacher.get_submodule(name)
            s_mod: OnlineTopKLinear = student.get_submodule(name)  # type: ignore[assignment]
            v_det = s_mod.V.detach()
            u_norm = s_mod.U.detach().norm(dim=-1)
            tk = s_mod.top_k

            def make_hook(nm=name, vm=v_det, un=u_norm, k_val=tk):
                def _hk(_m, inputs, _out):
                    x_in = inputs[0].to(dtype=vm.dtype)
                    scores = (x_in @ vm).abs() * un
                    chunk_masks[nm].append(scores.topk(k_val, dim=-1).indices.detach())
                return _hk

            handles.append(t_mod.register_forward_hook(make_hook()))
        try:
            emb = teacher.encode_inputs(inp)
            zh, zl = teacher.init_states(len(inp))
            for d in range(1, max_depth + 1):
                zh, zl = teacher.act_step(zh, zl, emb)
                if d in teacher_probs_by_d:
                    lg = teacher.digit_logits(zh)[:, :, :digits].float()
                    teacher_probs_by_d[d].append(torch.softmax(lg, dim=-1).cpu())
        finally:
            for h in handles:
                h.remove()
        teacher_masks_by_chunk.append(chunk_masks)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    teacher_rollout_sec = time.perf_counter() - t0
    teacher_probs = {d: torch.cat(v, dim=0) for d, v in teacher_probs_by_d.items()}

    # Helper to run raw depth rollout + KL + active component telemetry + wall time for any factor arm
    def _eval_arm(arm_name: str, arm_model: nn.Module, *, use_teacher_masks: bool = False) -> dict:
        wrappers: dict[str, OnlineTopKLinear] = {
            nm: arm_model.get_submodule(nm) for nm in site_names  # type: ignore[misc]
        }
        for w in wrappers.values():
            w.reset_routing_telemetry(True)

        arm_logits_by_d: dict[int, list[Tensor]] = {d: [] for d in depths}
        calls_per_act = int(arm_model.h_cycles * (arm_model.l_cycles + 1))
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        s0 = time.perf_counter()
        for chunk_i, offset in enumerate(range(0, len(eval_tokens_cpu), microbatch)):
            inp = eval_tokens_cpu[offset:offset + microbatch].to(dev)
            if use_teacher_masks:
                for nm, w in wrappers.items():
                    w._external_mask_queue = list(teacher_masks_by_chunk[chunk_i][nm])

            # If arm_name == "online", also measure slotwise top-k mask agreement vs teacher_scheduled!
            online_call_masks: dict[str, list[Tensor]] = {nm: [] for nm in site_names}
            o_handles = []
            if arm_name == "online":
                for nm, w in wrappers.items():
                    v_det = w.V.detach()
                    u_norm = w.U.detach().norm(dim=-1)
                    tk = w.top_k

                    def make_ohk(name_s=nm, vm=v_det, un=u_norm, k_val=tk):
                        def _ohk(_m, inputs, _out):
                            sc = (inputs[0].to(dtype=vm.dtype) @ vm).abs() * un
                            online_call_masks[name_s].append(sc.topk(k_val, dim=-1).indices.detach())
                        return _ohk

                    o_handles.append(w.register_forward_hook(make_ohk()))
            try:
                emb = arm_model.encode_inputs(inp)
                zh, zl = arm_model.init_states(len(inp))
                for d in range(1, max_depth + 1):
                    zh, zl = arm_model.act_step(zh, zl, emb)
                    if d in arm_logits_by_d:
                        arm_logits_by_d[d].append(arm_model.digit_logits(zh)[:, :, :digits].float().cpu())
            finally:
                for h in o_handles:
                    h.remove()
                if use_teacher_masks:
                    for w in wrappers.values():
                        w._external_mask_queue = None

            if arm_name == "online":
                for d in range(1, max_depth + 1):
                    c_start = (d - 1) * calls_per_act
                    c_end = d * calls_per_act
                    overlaps = []
                    for nm in site_names:
                        for c_idx in range(c_start, c_end):
                            m_on = online_call_masks[nm][c_idx]
                            m_tc = teacher_masks_by_chunk[chunk_i][nm][c_idx]
                            # Set overlap fraction |S_online cap S_teacher| / k via one-hot dot product
                            c_width = wrappers[nm].C
                            oh_on = F.one_hot(m_on, c_width).float()
                            oh_tc = F.one_hot(m_tc, c_width).float()
                            frac = float(((oh_on * oh_tc).sum(dim=-1) / wrappers[nm].top_k).mean().item())
                            overlaps.append(frac)
                    online_vs_teacher_overlap_by_step[d].append(sum(overlaps) / max(1, len(overlaps)))

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        forward_sec = time.perf_counter() - s0
        mem_snap = memory_snapshot(str(dev))

        telemetry_by_site = {nm: w.routing_telemetry() for nm, w in wrappers.items()}
        for w in wrappers.values():
            w.reset_routing_telemetry(False)

        # Policy & accuracy evaluation (for teacher_scheduled, score directly from logits; for autonomous modes also run evaluate_policy)
        kl_by_depth = {}
        acc_by_depth = {}
        givens_cpu = eval_tokens_cpu - 1
        for d in depths:
            lg_d = torch.cat(arm_logits_by_d[d], dim=0)
            p_t = teacher_probs[d]
            log_p_t = p_t.clamp_min(1e-12).log()
            log_p_a = F.log_softmax(lg_d, dim=-1)
            kl_by_depth[str(d)] = float((p_t * (log_p_t - log_p_a)).sum(dim=-1).mean().item())
            pred_0 = lg_d.argmax(dim=-1)
            b_acc = blank_accuracy(pred_0, eval_gold_cpu, eval_tokens_cpu, blank_token=1)
            acc_by_depth[str(d)] = {
                "exact_board_accuracy": b_acc["whole_board_exact"],
                "blank_cell_accuracy": b_acc["blank_accuracy"],
                "clue_preservation": b_acc["clue_preservation"],
                "kl_from_teacher_nats": kl_by_depth[str(d)],
            }
            _ = givens_cpu

        policy_summary = None
        if not use_teacher_masks:
            pol = evaluate_policy(arm_model, eval_tokens_cpu, eval_gold_cpu, depths=depths, microbatch=microbatch, budget=budget)
            policy_summary = {
                "first_valid_exact_board": pol["first_valid"]["exact_board_accuracy"],
                "case_step_fraction": pol["first_valid"]["case_step_fraction"],
            }

        factor_bytes = sum(
            (w.V.numel() + w.U.numel()) * w.V.element_size() for w in wrappers.values()
        )
        return {
            "arm": arm_name,
            "uses_privileged_teacher_masks_at_inference": use_teacher_masks,
            "forward_seconds": forward_sec,
            "examples_per_second": len(eval_tokens_cpu) / max(forward_sec, 1e-9),
            "factor_storage_bytes": factor_bytes,
            "memory": {
                "host_rss_bytes": mem_snap["host_rss_bytes"],
                "cuda_allocated_bytes": mem_snap["cuda_allocated_bytes"],
                "cuda_peak_allocated_bytes": mem_snap["cuda_peak_allocated_bytes"],
            },
            "by_depth": acc_by_depth,
            "first_valid_policy": policy_summary,
            "active_component_telemetry": {
                "mean_unique_active_components": sum(
                    t["unique_active_components"] for t in telemetry_by_site.values()
                ) / len(telemetry_by_site),
                "mean_active_per_token": sum(
                    t["mean_active_components_per_token"] for t in telemetry_by_site.values()
                ) / len(telemetry_by_site),
                "mean_entropy_effective_components": sum(
                    t["effective_component_entropy_rank"] for t in telemetry_by_site.values()
                ) / len(telemetry_by_site),
                "sites": telemetry_by_site,
            },
        }

    dense_pol = evaluate_policy(teacher, eval_tokens_cpu, eval_gold_cpu, depths=depths, microbatch=microbatch, budget=budget)
    arms_out = {
        "online": _eval_arm("online", student, use_teacher_masks=False),
        "teacher_scheduled": _eval_arm("teacher_scheduled", teacher_sched_model, use_teacher_masks=True),
        "fixed": _eval_arm("fixed", fixed_ctrl, use_teacher_masks=False),
        "random": _eval_arm("random", random_ctrl, use_teacher_masks=False),
    }
    mask_agreement = {
        str(d): sum(vals) / max(1, len(vals))
        for d, vals in online_vs_teacher_overlap_by_step.items()
    }
    d8_key = "8" if 8 in depths else str(max(depths))
    return {
        "schema": "BPD-DT05/autonomous-routing-benchmark/v1",
        "n_cases": len(eval_tokens_cpu),
        "depths": list(depths),
        "shared_vu_verified": True,
        "teacher_sha256": model_digest(teacher),
        "student_sha256": model_digest(student),
        "dense_teacher": {
            "rollout_with_hooks_seconds": teacher_rollout_sec,
            "by_depth": {
                str(d): {
                    "exact_board_accuracy": dense_pol["by_depth"][str(d)]["exact_board_accuracy"],
                    "blank_cell_accuracy": dense_pol["by_depth"][str(d)]["blank_cell_accuracy"],
                }
                for d in depths
            },
            "first_valid_exact_board": dense_pol["first_valid"]["exact_board_accuracy"],
        },
        "arms": arms_out,
        "online_vs_teacher_topk_set_overlap_by_step": mask_agreement,
        "summary_contrasts": {
            "depth": int(d8_key),
            "online_minus_fixed_exact": (
                arms_out["online"]["by_depth"][d8_key]["exact_board_accuracy"]
                - arms_out["fixed"]["by_depth"][d8_key]["exact_board_accuracy"]
            ),
            "online_minus_random_exact": (
                arms_out["online"]["by_depth"][d8_key]["exact_board_accuracy"]
                - arms_out["random"]["by_depth"][d8_key]["exact_board_accuracy"]
            ),
            "teacher_scheduled_minus_online_exact_gap": (
                arms_out["teacher_scheduled"]["by_depth"][d8_key]["exact_board_accuracy"]
                - arms_out["online"]["by_depth"][d8_key]["exact_board_accuracy"]
            ),
            "online_kl_nats": arms_out["online"]["by_depth"][d8_key]["kl_from_teacher_nats"],
            "teacher_scheduled_kl_nats": arms_out["teacher_scheduled"]["by_depth"][d8_key]["kl_from_teacher_nats"],
            "fixed_kl_nats": arms_out["fixed"]["by_depth"][d8_key]["kl_from_teacher_nats"],
            "random_kl_nats": arms_out["random"]["by_depth"][d8_key]["kl_from_teacher_nats"],
        },
    }


def _fit_ridge_probe_and_eval(
    train_states: Tensor,
    train_targets_zero_based: Tensor,
    train_tokens: Tensor,
    eval_states: Tensor,
    eval_targets_zero_based: Tensor,
    eval_tokens: Tensor,
    *,
    digits: int,
    candidate_ranks: Sequence[int] = (1, 2, 4, 8, 16),
    ridge_lambda: float = 1.0,
) -> dict:
    """Fit ridge linear decoder on disjoint discovery states and evaluate rank-truncated probes."""
    # Predict flattened board-cell one-hot targets [N, cells * digits] from centered dual state [N, D]
    n_tr = len(train_states)
    cells = train_targets_zero_based.shape[1]
    y_tr = F.one_hot(train_targets_zero_based.long(), digits).double().reshape(n_tr, cells * digits)
    y_ev = F.one_hot(eval_targets_zero_based.long(), digits).double().reshape(len(eval_states), cells * digits)

    x_tr = train_states.detach().double()
    x_ev = eval_states.detach().double()
    x_mean = x_tr.mean(dim=0, keepdim=True)
    y_mean = y_tr.mean(dim=0, keepdim=True)
    xc_tr = x_tr - x_mean
    yc_tr = y_tr - y_mean
    xc_ev = x_ev - x_mean

    # Dual kernel solve W = X^T (X X^T + lambda I)^{-1} Y avoids [D, D] matrix!
    k_mat = xc_tr @ xc_tr.T + ridge_lambda * torch.eye(n_tr, dtype=torch.float64, device=x_tr.device)
    alpha = torch.linalg.solve(k_mat, yc_tr)  # [N_tr, M_out]
    w_op = xc_tr.T @ alpha  # [D, M_out]
    u_w, s_w, vh_w = torch.linalg.svd(w_op, full_matrices=False)
    spec = _spectral_summary(s_w.pow(2))

    # Split-half subspace overlap on discovery states (ECA-03/04 identifiability check)
    half = n_tr // 2
    split_overlap_r4 = None
    if half >= 4:
        xc1, yc1 = xc_tr[:half], yc_tr[:half]
        xc2, yc2 = xc_tr[half:2 * half], yc_tr[half:2 * half]
        w1 = xc1.T @ torch.linalg.solve(
            xc1 @ xc1.T + ridge_lambda * torch.eye(half, dtype=torch.float64, device=x_tr.device), yc1
        )
        w2 = xc2.T @ torch.linalg.solve(
            xc2 @ xc2.T + ridge_lambda * torch.eye(half, dtype=torch.float64, device=x_tr.device), yc2
        )
        u1, _, _ = torch.linalg.svd(w1, full_matrices=False)
        u2, _, _ = torch.linalg.svd(w2, full_matrices=False)
        r_ov = min(4, u1.shape[1], u2.shape[1])
        split_overlap_r4 = principal_angles_deg(u1[:, :r_ov], u2[:, :r_ov])["subspace_overlap"]

    ev_blanks = eval_tokens.eq(1)
    by_rank = {}
    for r in candidate_ranks:
        r_eff = min(int(r), int(s_w.numel()))
        if r_eff < 1:
            continue
        w_r = (u_w[:, :r_eff] * s_w[:r_eff]) @ vh_w[:r_eff, :]
        pred_flat = xc_ev @ w_r + y_mean
        mse = float((pred_flat - y_ev).pow(2).mean().item())
        pred_logits = pred_flat.reshape(len(eval_states), cells, digits).float()
        ce = float(F.cross_entropy(pred_logits.reshape(-1, digits), eval_targets_zero_based.reshape(-1)).item())
        pred_dig = pred_logits.argmax(dim=-1)
        blank_acc = float(
            ((pred_dig == eval_targets_zero_based) & ev_blanks).sum().item()
            / max(1, int(ev_blanks.sum().item()))
        )
        exact_acc = float((pred_dig == eval_targets_zero_based).all(dim=-1).float().mean().item())
        by_rank[str(r)] = {
            "effective_rank": r_eff,
            "heldout_brier_mse": mse,
            "heldout_cell_cross_entropy": ce,
            "heldout_blank_accuracy": blank_acc,
            "heldout_exact_board": exact_acc,
        }
    return {
        "operator_spectrum": spec,
        "split_half_rank4_overlap": split_overlap_r4,
        "heldout_by_rank": by_rank,
    }


def probe_depth_causal_rank(
    model: nn.Module,
    discovery_tokens: Tensor,
    discovery_gold: Tensor,
    eval_tokens: Tensor,
    eval_gold: Tensor,
    *,
    probe_depths: Sequence[int] = (1, 2, 4, 6, 8),
    horizon_depth: int = 12,
    intervention_ranks: Sequence[int] = (1, 2, 4, 8, 16),
    corruption_rel_dose: float = 0.85,
    n_gramian_cases: int = 8,
    seed: int = 2605997,
) -> dict:
    """Distinguish statistical, linearly decodable, and native future-response causal rank (§4.D).

    Grounded in BPD-ECA-03/04/05:
    - ``statistical_dimension``: centered activation covariance spectrum across boards.
    - ``linearly_decodable_dimension``: ridge probe spectrum & heldout accuracy/MSE (observational only!).
    - ``native_future_causal_dimension``: future-probability VJP Gramian spectrum to ``horizon_depth``
      PLUS native equal-L2-dose rank-``r`` interventions vs Haar random protected shams, scoring BOTH
      proper-scoring future KL recovery and behavioral exact/blank headroom.
    """
    if any(d < 1 or d >= horizon_depth for d in probe_depths):
        raise ValueError("probe_depths must be positive integers strictly less than horizon_depth")
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        tr_tok = discovery_tokens.to(dev)
        tr_gold = discovery_gold.to(dev)
        ev_tok = eval_tokens.to(dev)
        ev_gold = eval_gold.to(dev)
        t_len = int(model.total_len)
        puz = int(model.puzzle_emb_len)
        d_hid = int(model.hidden_size)
        full_dim = 2 * t_len * d_hid
        digits = int(math.isqrt(int(model.seq_len)))
        state_shape = (t_len, d_hid)

        projector = HeadNullProjector(model.lm_head.weight)
        project = lambda vecs: projector.project_dual(vecs, t_len, puz)

        # Roll out discovery and eval cohorts once to capture clean dual states at probe_depths
        max_probe = max(probe_depths)
        probe_set = set(probe_depths)
        tr_states_by_d: dict[int, Tensor] = {}
        ev_states_by_d: dict[int, Tensor] = {}
        with torch.no_grad():
            tr_emb = model.encode_inputs(tr_tok)
            th, tl = model.init_states(len(tr_tok))
            for d in range(1, max_probe + 1):
                th, tl = model.act_step(th, tl, tr_emb)
                if d in probe_set:
                    tr_states_by_d[d] = torch.cat((th.flatten(1), tl.flatten(1)), dim=-1).detach()

            ev_emb = model.encode_inputs(ev_tok)
            eh, el = model.init_states(len(ev_tok))
            for d in range(1, horizon_depth + 1):
                eh, el = model.act_step(eh, el, ev_emb)
                if d in probe_set:
                    ev_states_by_d[d] = torch.cat((eh.flatten(1), el.flatten(1)), dim=-1).detach()
            clean_horizon_logits = model.digit_logits(eh)[:, :, :digits].float()
            clean_horizon_probs = torch.softmax(clean_horizon_logits, dim=-1)
            clean_horizon_logp = F.log_softmax(clean_horizon_logits, dim=-1)
            clean_horizon_exact = float((clean_horizon_logits.argmax(dim=-1) == ev_gold).all(dim=-1).float().mean().item())

        by_depth_out = {}
        for d in probe_depths:
            s_tr = tr_states_by_d[d]
            s_ev = ev_states_by_d[d]
            s_tr_dark = project(s_tr)
            s_ev_dark = project(s_ev)

            # 1. Statistical Information Dimension (centered covariance SVD)
            s_centered = s_tr.double() - s_tr.double().mean(dim=0, keepdim=True)
            stat_svals = torch.linalg.svdvals(s_centered)
            stat_eig = stat_svals.pow(2) / max(1, len(s_tr) - 1)
            dark_centered = s_tr_dark.double() - s_tr_dark.double().mean(dim=0, keepdim=True)
            dark_eig = torch.linalg.svdvals(dark_centered).pow(2) / max(1, len(s_tr) - 1)

            # 2. Linearly Decodable Representation Dimension (full state & dark state)
            dec_full = _fit_ridge_probe_and_eval(
                s_tr, tr_gold, tr_tok, s_ev, ev_gold, ev_tok,
                digits=digits, candidate_ranks=intervention_ranks,
            )
            dec_dark = _fit_ridge_probe_and_eval(
                s_tr_dark, tr_gold, tr_tok, s_ev_dark, ev_gold, ev_tok,
                digits=digits, candidate_ranks=intervention_ranks,
            )

            # 3. Native Future-Response Causal Dimension (d -> horizon_depth)
            rem_steps = horizon_depth - d
            n_g = min(len(s_tr), int(n_gramian_cases))
            max_r = max(intervention_ranks)
            gram_basis, gram_meta = select_protected_task_gramian_vjp(
                model, tr_tok[:n_g], s_tr[:n_g],
                fit_offsets=(rem_steps,), rank=max_r,
            )

            # Native finite-dose intervention curve across ranks r in intervention_ranks
            gen_c = torch.Generator(device="cpu").manual_seed(int(seed) * 97 + d)
            noise = project(
                torch.randn(s_ev.shape, generator=gen_c, dtype=torch.float32).to(device=dev, dtype=s_ev.dtype)
            )
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            prot_norm = s_ev_dark.norm(dim=-1, keepdim=True).clamp_min(1.0)
            recip_d = s_ev + (corruption_rel_dose * prot_norm) * noise
            diff_d = project(s_ev - recip_d)

            with torch.no_grad():
                no_edit_dual = DualState.unflatten(recip_d, state_shape)
                no_lg = native_resume_recipient(model, ev_tok, no_edit_dual, rem_steps)[:, :, :digits].float()
                no_logp = F.log_softmax(no_lg, dim=-1)
                no_kl = float((clean_horizon_probs * (clean_horizon_logp - no_logp)).sum(dim=-1).mean().item())
                no_exact = float((no_lg.argmax(dim=-1) == ev_gold).all(dim=-1).float().mean().item())

                oracle_dual = DualState.unflatten(recip_d + diff_d, state_shape)
                or_lg = native_resume_recipient(model, ev_tok, oracle_dual, rem_steps)[:, :, :digits].float()
                or_logp = F.log_softmax(or_lg, dim=-1)
                or_kl = float((clean_horizon_probs * (clean_horizon_logp - or_logp)).sum(dim=-1).mean().item())
                or_exact = float((or_lg.argmax(dim=-1) == ev_gold).all(dim=-1).float().mean().item())

                rank_interventions = {}
                for r in intervention_ranks:
                    r_eff = min(int(r), int(gram_basis.shape[1]))
                    if r_eff < 1:
                        continue
                    q_r = gram_basis[:, :r_eff]
                    ed_cand = (diff_d @ q_r) @ q_r.T
                    cand_doses = ed_cand.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                    # 4 equal-dose rank-r Haar random protected shams
                    sham_kls = []
                    sham_exacts = []
                    for s_i in range(4):
                        q_sh = _sample_random_protected_basis(
                            projector, t_len, puz, full_dim, r_eff,
                            seed=int(seed) * 1009 + d * 37 + r_eff * 11 + s_i,
                            device=dev, dtype=s_ev.dtype,
                        )
                        ed_sh = (diff_d @ q_sh) @ q_sh.T
                        ed_sh = ed_sh * (cand_doses / ed_sh.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                        sh_dual = DualState.unflatten(recip_d + ed_sh, state_shape)
                        sh_lg = native_resume_recipient(model, ev_tok, sh_dual, rem_steps)[:, :, :digits].float()
                        sh_logp = F.log_softmax(sh_lg, dim=-1)
                        sham_kls.append(float((clean_horizon_probs * (clean_horizon_logp - sh_logp)).sum(dim=-1).mean().item()))
                        sham_exacts.append(float((sh_lg.argmax(dim=-1) == ev_gold).all(dim=-1).float().mean().item()))

                    cand_dual = DualState.unflatten(recip_d + ed_cand, state_shape)
                    c_lg = native_resume_recipient(model, ev_tok, cand_dual, rem_steps)[:, :, :digits].float()
                    c_logp = F.log_softmax(c_lg, dim=-1)
                    c_kl = float((clean_horizon_probs * (clean_horizon_logp - c_logp)).sum(dim=-1).mean().item())
                    c_exact = float((c_lg.argmax(dim=-1) == ev_gold).all(dim=-1).float().mean().item())
                    mean_sh_kl = sum(sham_kls) / len(sham_kls)
                    mean_sh_exact = sum(sham_exacts) / len(sham_exacts)

                    rank_interventions[str(r)] = {
                        "effective_rank": r_eff,
                        "delivered_l2_dose_mean": float(cand_doses.mean().item()),
                        "candidate_horizon_kl_nats": c_kl,
                        "haar_sham_mean_horizon_kl_nats": mean_sh_kl,
                        "kl_reduction_over_no_edit_nats": no_kl - c_kl,
                        "kl_advantage_over_equal_dose_sham_nats": mean_sh_kl - c_kl,
                        "candidate_exact_board": c_exact,
                        "haar_sham_mean_exact_board": mean_sh_exact,
                        "exact_board_advantage_over_sham": c_exact - mean_sh_exact,
                    }

            by_depth_out[str(d)] = {
                "depth": d,
                "remaining_steps_to_horizon": rem_steps,
                "statistical_information_dimension": {
                    "full_dual_state": _spectral_summary(stat_eig),
                    "protected_dark_state": _spectral_summary(dark_eig),
                },
                "linearly_decodable_dimension": {
                    "evidence_note": "observational probe decodability only; not causal mechanism",
                    "full_dual_state": dec_full,
                    "protected_dark_state": dec_dark,
                },
                "native_future_causal_dimension": {
                    "protected_gramian_spectrum": {
                        k: gram_meta[k]
                        for k in (
                            "participation_ratio",
                            "entropy_effective_rank",
                            "rank_90pct_variance",
                            "rank_99pct_variance",
                            "top_eigenvalues",
                        )
                    },
                    "behavioral_headroom": {
                        "clean_horizon_exact_board": clean_horizon_exact,
                        "corrupted_no_edit_exact_board": no_exact,
                        "full_dark_donor_oracle_exact_board": or_exact,
                        "oracle_exact_board_headroom": or_exact - no_exact,
                        "corrupted_no_edit_kl_nats": no_kl,
                        "full_dark_donor_oracle_kl_nats": or_kl,
                        "oracle_kl_headroom_nats": no_kl - or_kl,
                    },
                    "interventions_by_rank": rank_interventions,
                },
            }

        return {
            "schema": "BPD-DT05/depth-causal-rank-probe/v1",
            "horizon_depth": horizon_depth,
            "probe_depths": list(probe_depths),
            "n_discovery_cases": len(discovery_tokens),
            "n_eval_cases": len(eval_tokens),
            "n_gramian_cases": min(len(discovery_tokens), int(n_gramian_cases)),
            "eca_design_lessons_applied": [
                "Separate statistical participation rank, linear probe decodability, and native intervention rank.",
                "Fit probes and Gramians on disjoint discovery cases; never fit on evaluation cases.",
                "Report both proper-scoring future KL recovery and behavioral task headroom (ECA-03/04/05).",
                "Compare every candidate rank against equal-L2-dose Haar random protected shams.",
            ],
            "by_depth": by_depth_out,
        }
    finally:
        model.train(was_training)
