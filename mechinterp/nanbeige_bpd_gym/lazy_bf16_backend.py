"""Assemble a storage-backed native loop; no inference or registration on import.

The execution device may be CPU for random small-class engineering. Real CUDA
use requires a separately frozen worker, resource admission and native/BPD
capacity checks. This does not replace any frozen loader.
"""
from .lazy_bf16_loop import make_lazy_loop


def build_loop(model,store,device,*,n_resident=4,reference_modules=('model.layers.21.mlp.down_proj',),guard=None):
    import torch
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import _reinit_rotary_buffers
    device=torch.device(device);store.validate(model)
    if model.config.tie_word_embeddings:raise ValueError('Only the pinned untied vocabulary weights are supported')
    if not 0<=n_resident<len(model.model.layers):raise ValueError('Require a fixed prefix and offloaded layers')
    if any(not p.is_meta or p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('Start with a complete BF16 meta architecture')
    if model.model.embed_tokens.max_norm is not None:raise ValueError('Read-only embedding parameters cannot be renormalized in place')
    model.eval();model.requires_grad_(False)
    def check():
        if guard is not None:guard()
    cpu_names=['model.embed_tokens.weight','lm_head.weight']
    all_params=dict(model.named_parameters())
    resident_prefixes=tuple(f'model.layers.{i}.' for i in range(n_resident))
    resident=[name for name in all_params if name.startswith(resident_prefixes) or name.startswith('model.norm.')]
    for name in reference_modules:
        module=model.get_submodule(name)
        for relative,p in module.named_parameters():
            full=name+'.'+relative
            if full not in resident:cpu_names.append(full)
    cpu_names=list(dict.fromkeys(cpu_names))
    for name in cpu_names:check();store.bind_parameters(model,[name],'cpu');check()
    for name in resident:check();store.bind_parameters(model,[name],device);check()
    for layer in model.model.layers[:n_resident]:_reinit_rotary_buffers(layer,device)
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device('meta'):stage=type(model.model.layers[0])(model.config,n_resident)
    finally:torch.set_default_dtype(previous)
    check();stage.to_empty(device=device);_reinit_rotary_buffers(stage,device);stage.eval().requires_grad_(False);check()
    loop=make_lazy_loop(model,store,device,n_resident,stage)
    loop.storage_layout=dict(resident_decoder_layers=n_resident,meta_parameters=sum(p.is_meta for p in model.parameters()),
        total_parameters=len(all_params),readonly_CPU_parameter_names=cpu_names,stage_source='one read-only tensor per synchronous copy',
        no_complete_CPU_decoder_materialized=True)
    return loop
