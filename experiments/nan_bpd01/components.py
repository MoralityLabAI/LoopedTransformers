"""Gauge-aware small linear instrument. Native use is competence-gated."""
from __future__ import annotations
import torch

def effective_weight(module):
    """The NF4 target is its actual dequantized operator, never the BF16 source."""
    weight=module.weight
    if getattr(weight,'quant_state',None) is not None:
        import bitsandbytes.functional as bnbf
        return bnbf.dequantize_4bit(weight.data,weight.quant_state).float()
    return weight.detach().float()

def canonical_factors(v,u,rank=None):
    # Canonicalize the operator, not either gauge-dependent factor separately.
    left,s,right=torch.linalg.svd((v.double()@u.double()),full_matrices=False)
    k=v.shape[1] if rank is None else rank
    left=left[:,:k];right=right[:k];s=s[:k]
    # Remove the remaining independent sign gauge deterministically.
    for i in range(k):
        pivot=left[:,i].abs().argmax()
        if left[pivot,i]<0:left[:,i]*=-1;right[i]*=-1
    return left*s.sqrt(),s.sqrt()[:,None]*right,s

def spectral_groups(s,relative_gap=1e-6):
    groups=[]
    for i in range(len(s)):
        if i==0 or abs(float(s[i]-s[i-1]))>relative_gap*max(float(s[0]),1e-30):groups.append([i])
        else:groups[-1].append(i)
    return groups

def basis(matrix,rank,kind,*,activations=None,gradients=None,delayed_jacobian=None,generator=None):
    if kind=='svd':return torch.linalg.svd(matrix.double(),full_matrices=False).U[:,:rank]
    if kind=='pca':return torch.linalg.svd(activations.double()-activations.double().mean(0),full_matrices=False).Vh[:rank].T
    if kind=='gradient':return torch.linalg.svd(gradients.double(),full_matrices=False).Vh[:rank].T
    if kind=='gramian':return torch.linalg.eigh(delayed_jacobian.double().T@delayed_jacobian.double()).eigenvectors[:,-rank:]
    if kind=='random':return torch.linalg.qr(torch.randn(matrix.shape[0],rank,dtype=torch.float64,generator=generator)).Q
    raise ValueError(kind)

def native_wrappers(model,bpd_root,rank):
    """Install additive full-operator-plus-residual wrappers at two frozen sites.

    Caller must establish native competence/closure and supply effective NF4
    weights. This helper does not select a mask or fit on development labels.
    """
    import mechinterp
    from pathlib import Path
    mechinterp.__path__.append(str(Path(bpd_root)/'mechinterp'))
    from mechinterp.bpd.components import TiedComponentLinear
    result={}
    layer=model.model.layers[-1]
    for path in ['self_attn.o_proj','mlp.down_proj']:
        target=layer.get_submodule(path)
        linear=torch.nn.Linear(target.in_features,target.out_features,bias=target.bias is not None)
        linear.weight=torch.nn.Parameter(effective_weight(target).cpu(),requires_grad=False)
        if target.bias is not None:linear.bias=torch.nn.Parameter(target.bias.detach().cpu().float(),requires_grad=False)
        wrapper=TiedComponentLinear(linear,rank,max_passes=2,init_svd=True)
        parent,_,name=path.rpartition('.');setattr(layer.get_submodule(parent),name,wrapper);result[path]=wrapper
    return result
