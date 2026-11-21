"""New inference condition: explicitly dequantize NF4 to BF16 before F.linear.

This is not a patch to any frozen worker. The installed bitsandbytes 0.50.2
selects fused or dequantized kernels based on row count and matrix shape.
Fixing this policy changes the numerical condition, including its prefill.
Packed parameter objects and quantization state are retained without copying
or requantizing; only one effective matrix exists during each linear call.
"""
import torch
from torch import nn
from torch.nn import functional as F


class EffectiveNF4Linear(nn.Module):
    def __init__(self, original, *, dequantize):
        super().__init__()
        state = original.weight.quant_state
        if state is None or state.quant_type != 'nf4' or not state.nested:
            raise ValueError('Frozen double-quantized NF4 weights required')
        if state.dtype != torch.bfloat16 or original.weight.requires_grad:
            raise ValueError('The registered effective weight dtype is frozen BF16')
        if original.bias is not None and original.bias.requires_grad:
            raise ValueError('Frozen bias required')
        self.in_features = original.in_features
        self.out_features = original.out_features
        if tuple(state.shape) != (self.out_features, self.in_features):
            raise ValueError('Canonical output/input weight orientation required')
        self.weight = original.weight
        self.bias = original.bias
        self._dequantize = dequantize
        self.eval()

    def forward(self, x):
        if not torch.is_inference_mode_enabled():
            raise RuntimeError('This precision engineering condition is inference-only')
        effective = self._dequantize(self.weight.data, self.weight.quant_state)
        if effective.dtype != torch.bfloat16 or effective.device != x.device:
            raise RuntimeError('Effective NF4 weight dtype/device changed')
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), effective, bias).to(x.dtype)


def mount_effective_nf4(loop):
    """Replace decoder leaves in a newly constructed, unobserved model only."""
    import bitsandbytes as bnb
    if loop.n_gpu_layers != loop.n or loop.stage_layer is not None:
        raise ValueError('This condition requires all NF4 decoder layers resident')
    if any(p.requires_grad for p in loop.model.parameters()):
        raise ValueError('Frozen target required')
    changed = []

    def visit(module, prefix):
        for name, child in list(module.named_children()):
            path = f'{prefix}.{name}'
            if isinstance(child, bnb.nn.Linear4bit):
                if child._forward_hooks or child._forward_pre_hooks:
                    raise ValueError('Mount the new precision condition before observers')
                replacement = EffectiveNF4Linear(child, dequantize=bnb.functional.dequantize_4bit)
                setattr(module, name, replacement)
                changed.append(path)
            else:
                visit(child, path)

    for index, layer in enumerate(loop.layers):
        visit(layer, f'model.layers.{index}')
    if len(changed) != loop.quantized_linears:
        raise RuntimeError('Not every originally quantized decoder linear was replaced')
    return changed
