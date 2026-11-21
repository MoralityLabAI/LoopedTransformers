"""Unregistered candidate: file-backed native stages and scoped digest dispatch.

No running/frozen worker imports this module. Inference-only integration must
still receive its own source/resource registration and CUDA admission.
"""
from contextlib import contextmanager
from .lazy_bf16_storage import logical_target_digest


class _StorageBackedLoop:
    pass


def make_lazy_loop(model,store,device,n_resident,stage):
    import torch
    from experiments.nan_bpd01.backend import NativeLoop
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import StreamedNanbeigeLoop
    device=torch.device(device);store.validate(model);n=len(model.model.layers)
    if not 0<=n_resident<n or stage is None:raise ValueError('Require resident prefix and an allocated stage')
    if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad or p.is_meta or p.device!=device for p in stage.parameters()):
        raise ValueError('Freeze all target parameters and allocate the stage')
    if any(b.is_meta or b.device!=device for b in stage.buffers()):raise ValueError('Initialize stage buffers on the execution device')
    for layer in model.model.layers[:n_resident]:
        if any(p.is_meta or p.device!=device for p in layer.parameters()):raise ValueError('Resident prefix is not materialized on execution device')
        if any(b.is_meta or b.device!=device for b in layer.buffers()):raise ValueError('Initialize resident prefix buffers on the execution device')
    if model.model.embed_tokens.weight.is_meta or model.model.embed_tokens.max_norm is not None:
        raise ValueError('Read-only embeddings require actual storage and no in-place max_norm')
    if model.model.norm.weight.is_meta or model.model.norm.weight.device!=device:raise ValueError('Materialize the boundary norm on the execution device')
    class LazyNativeLoop(_StorageBackedLoop,NativeLoop):
        def __init__(self):
            StreamedNanbeigeLoop.__init__(self,model,device,n_resident,stage)
            self.checkpoint_store=store;self._layer_prefix={id(layer):f'model.layers.{i}' for i,layer in enumerate(self.layers)}
            self.quantized_linears=0;self.channels=();self.collect=False;self.observations=[];self.phase=0;self.active_layer=-1
            self.stage_copies=0;self.maximum_source_tensor_bytes=0
        def _load_stage(self,cpu_layer):
            prefix=self._layer_prefix.get(id(cpu_layer))
            if prefix is None:raise ValueError('Stage source is not one of the registered model layers')
            stats=self.checkpoint_store.copy_parameters(self.stage_layer,prefix)
            self.stage_copies+=1;self.maximum_source_tensor_bytes=max(self.maximum_source_tensor_bytes,stats['largest_source_tensor_bytes'])
            if hasattr(self.stage_layer,'self_attn') and hasattr(cpu_layer,'self_attn'):
                self.stage_layer.self_attn.layer_idx=cpu_layer.self_attn.layer_idx
            return self.stage_layer
    return LazyNativeLoop()


@contextmanager
def digest_dispatch():
    """Isolated-process compatibility for old callers; restore on every exit."""
    import experiments.nan_recomp09.common as module
    original=module.target_digest
    def digest(loop,chunk_bytes=16*1024**2):
        if isinstance(loop,_StorageBackedLoop):return logical_target_digest(loop.model,loop.checkpoint_store,chunk_bytes)
        return original(loop,chunk_bytes)
    module.target_digest=digest
    try:yield digest
    finally:module.target_digest=original


def clear_unused_row_observations(instrument):
    """Future full-gym captures only, after raw_by_phase has been exported.

    Instrument.snapshots() consumers need base.raw; they must not use this.
    FULL26 RowInstrument exports raw_by_phase instead. Clearing this duplicate
    observer list changes no kernel, mask, score or exported native tensor.
    """
    from experiments.nan_full26.batch import RowInstrument
    if not isinstance(instrument,RowInstrument):raise TypeError('Only the full-gym row instrument is supported')
    count=len(instrument.base.raw);instrument.base.raw.clear();return count
