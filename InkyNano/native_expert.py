"""Reference math after upstream native dequantization and rearrangement."""
import torch
import torch.nn.functional as F


def activation(gate, up, gate_cap=4., up_cap=25.):
    dtype = gate.dtype
    g, u = gate.float(), up.float()
    return (gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g) *
            up_cap * torch.tanh(u / up_cap)).to(dtype)


def expert(x, gate_up, down):
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(activation(gate,up),down)


@torch.no_grad()
def audit_native(module, expert_id, inputs):
    """Use the actual pinned, eager upstream expert module and native inputs.
    Single-expert weight 1 isolates the expert; multi-expert parity is separate.
    Returns errors, never declares PASS without a calibrated tolerance.
    """
    if not 0 <= expert_id < module.num_experts:
        raise ValueError("expert ID out of range")
    idx = torch.full((inputs.shape[0],1), expert_id, device=inputs.device,dtype=torch.long)
    w = torch.ones_like(idx,dtype=torch.float32)
    ref = module(inputs,idx,w)
    pred = expert(inputs,module.gate_up_proj[expert_id],module.down_proj[expert_id])
    diff = pred.float()-ref.float()
    denom = torch.linalg.vector_norm(ref.float())
    norm = torch.linalg.vector_norm(diff)
    return {"relative_l2":float(norm/denom) if denom > 0 else (0. if norm == 0 else float("inf")),
            "max_abs":float(diff.abs().max()),"dtype":str(inputs.dtype),
            "n_inputs":inputs.shape[0],"expert_id":expert_id}


def factor_expert(x, gate_factors, up_factors, down_factors, rank):
    from spectral import apply_factors
    g = apply_factors(x,*gate_factors,rank)
    u = apply_factors(x,*up_factors,rank)
    return apply_factors(activation(g,u),*down_factors,rank)
