"""Local o_proj-output to layer-residual bridge through the following MLP.

This is an observational directional derivative of the native remainder,
before final loop RMSNorm. BF16 autograd is a derivative surrogate through
rounded operations; finite native writes are reported separately. It is not
an identity bridge or a donor-transfer/causal-fidelity certificate.
"""
import math
import torch


def local_remainder(layer, state):
    return state + layer.mlp(layer.post_attention_layernorm(state))


def bridge_o_proj(layer, attention_residual, attention_output, directions, *, actual_pre_mlp_state, relative_doses=(.001, .01, .1)):
    """Return JVPs and centered finite native writes for each physical row.

    Supply the full actual pre-attention residual and physical o_proj output, plus
    the independently captured input to the pre-MLP norm. Re-add them using
    native arithmetic; never reconstruct by subtracting rounded downstream
    tensors. Keep the original prefix shape to preserve native GEMM arithmetic.
    Tangents/writes touch only the last token. Finite writes include rounding
    of the attention residual addition.
    Call only after component/observer hooks are removed and staging is fixed.
    No target parameter or input tensor is modified.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Use ordinary/no_grad mode, not inference_mode, for native JVPs')
    if getattr(layer, 'enable_hyper_connection', False):
        raise ValueError('The simple additive residual bridge excludes hyper-connections')
    if attention_output.ndim != 3 or attention_output.shape[0] != 1 or attention_output.shape[1] < 1 or attention_residual.shape != attention_output.shape:
        raise ValueError('One full nonempty source prefix of actual attention residual/output required')
    if attention_residual.dtype != attention_output.dtype or attention_residual.device != attention_output.device:
        raise ValueError('Preserve native residual addition dtype/device')
    if actual_pre_mlp_state.shape != attention_output.shape or actual_pre_mlp_state.dtype != attention_output.dtype or actual_pre_mlp_state.device != attention_output.device:
        raise ValueError('Captured pre-MLP coordinates must preserve shape/dtype/device')
    if directions.ndim != 2 or directions.shape[1] != attention_output.shape[-1]:
        raise ValueError('Physical o_proj output rows must have residual width')
    if not all(torch.isfinite(v).all() for v in (attention_residual,attention_output,actual_pre_mlp_state,directions)):
        raise ValueError('Finite coordinates required')
    for module in (layer.post_attention_layernorm, *layer.mlp.modules()):
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError('Remove model component and observation hooks before the local bridge')
        for p in module.parameters(recurse=False):
            if p.requires_grad or getattr(p, 'quant_state', None) is not None:
                raise ValueError('Only frozen unquantized native BF16/reference modules are validated')
    if any(not math.isfinite(d) or d <= 0 for d in relative_doses):
        raise ValueError('Positive finite relative doses required')
    residual = attention_residual.detach().clone()
    x = attention_output.detach().clone()
    if not torch.equal(residual+x, actual_pre_mlp_state):
        raise ValueError('Captured pre-MLP state does not equal the native attention residual addition')
    remainder = lambda z: local_remainder(layer, residual+z)
    with torch.no_grad():
        baseline = remainder(x).detach()
    outputs = []
    finite = []
    x_norm = float(x[:, -1:].float().norm())
    for index, row in enumerate(directions):
        tangent = torch.zeros_like(x)
        tangent[:, -1:] = row.to(x).reshape(1,1,-1)
        row_norm = float(tangent.float().norm())
        with torch.enable_grad():
            y, jv = torch.autograd.functional.jvp(remainder,
                x, tangent, create_graph=False, strict=False)
        if not torch.equal(y, baseline):
            raise AssertionError('Local JVP forward changed native remainder arithmetic')
        if not torch.isfinite(jv).all():
            raise AssertionError('Nonfinite local native derivative')
        if bool(jv[:, :-1].count_nonzero()):
            raise AssertionError('The proposed pointwise bridge changed earlier token derivatives')
        outputs.append(jv.detach()[:, -1:].reshape(-1))
        for dose in relative_doses:
            scale = dose * max(x_norm, 1e-12) / max(row_norm, 1e-12)
            with torch.no_grad():
                plus_x = x + scale * tangent
                minus_x = x - scale * tangent
                plus = remainder(plus_x)
                minus = remainder(minus_x)
                central = (plus.double() - minus.double()) / (2 * scale)
                norm = max(float(jv.double().norm()), 1e-12)
                finite.append(dict(direction=index,requested_relative_input_dose=dose,scale=scale,
                    zero_direction=row_norm == 0,
                    actual_plus_input_l2=float((plus_x.double()-x.double()).norm()),
                    actual_minus_input_l2=float((minus_x.double()-x.double()).norm()),
                    actual_plus_post_attention_state_l2=float(((residual+plus_x).double()-(residual+x).double()).norm()),
                    actual_minus_post_attention_state_l2=float(((residual+minus_x).double()-(residual+x).double()).norm()),
                    central_vs_autograd_relative_l2=float((central-jv.double()).norm())/norm,
                    finite_native_plus_output_l2=float((plus.double()-baseline.double()).norm()),
                    finite_native_minus_output_l2=float((minus.double()-baseline.double()).norm())))
    if not outputs:
        raise ValueError('At least one physical output direction required')
    return dict(native_remainder=baseline,physical_JVP_rows=torch.stack(outputs),native_tangent_rows=directions.to(x),finite_writes=finite,
        coordinate_bridge='o_proj output -> actual post-attention residual -> native pre-MLP norm/MLP + residual -> residual_after_layer before final loop norm',
        scope='local autograd derivative and finite rounded writes; no immediate/future readout or causal transfer claim')
