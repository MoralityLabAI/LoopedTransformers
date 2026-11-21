"""Prospective cached-backend assembly; no loading or registration on import.

The NF4 builder skips the temporary, randomly initialized CPU Linear4bit
matrix. It still converts each unchanged CPU source with the original bnb
NF4/double-quantization recipe. CUDA byte/quant-state/capacity checks remain
mandatory in a future frozen worker. CPU construction tests are not those
checks. Nothing here alters RAM28, BF31 or their running/queued workers.
"""


def prepare_nf4_linear(source, *, bnb=None):
    """Construct a bias-free bnb wrapper without a throwaway CPU weight heap.

    No quantizer or forward is called. The frozen contiguous BF16 CPU weight
    remains aliased until its caller moves the wrapper to CUDA. A meta tensor
    cannot be used as source storage. Biases are deliberately unsupported: the
    pinned Nanbeige decoder has none.
    """
    import torch
    if not isinstance(source,torch.nn.Linear) or source.bias is not None:
        raise ValueError('Require a pinned bias-free decoder linear')
    weight=source.weight
    if weight.device.type!='cpu' or weight.dtype!=torch.bfloat16 or weight.requires_grad or not weight.is_contiguous():
        raise ValueError('Require frozen contiguous BF16 CPU source storage')
    if bnb is None:
        import bitsandbytes as bnb
    # The original constructor initializes a full float32 matrix, then throws
    # it away when assigning Params4bit. A meta constructor owns no such heap.
    with torch.device('meta'):
        quantized=bnb.nn.Linear4bit(source.in_features,source.out_features,bias=False,
            compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
    if not quantized.weight.is_meta:raise RuntimeError('bnb constructor allocated unexpected source storage')
    quantized.weight=bnb.nn.Params4bit(weight.detach(),requires_grad=False,
        compress_statistics=True,quant_type='nf4')
    quantized.eval().requires_grad_(False)
    if quantized.weight.data_ptr()!=weight.data_ptr() or quantized.weight.device.type!='cpu':
        raise RuntimeError('Pre-quantization wrapper must alias the immutable CPU source')
    return quantized


def build_nf4_loop(model,device,*,guard=None):
    """Quantize an already materialized read-only model, leaving its head CPU.

    Only assembly is supplied. The calling future worker must verify locked
    checkpoint files and before/after target bytes and admit CUDA/RAM before
    using this builder. It is intentionally not a pretrained loading CLI.
    """
    import gc
    import torch
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('Actual NF4 conversion requires the registered CUDA condition')
    from experiments.nan_bpd01.backend import NativeLoop
    from mechinterp.dark_fiber_gyms.arg03_nanbeige import StreamedNanbeigeLoop
    if model.config.tie_word_embeddings or model.model.embed_tokens.max_norm is not None:
        raise ValueError('Immutable untied vocabulary matrices required')
    if any(p.device.type!='cpu' or p.dtype!=torch.bfloat16 or p.requires_grad for p in model.parameters()):
        raise ValueError('Start with frozen BF16 CPU parameters backed by the pinned source')
    if any(b.is_meta for b in model.buffers()):raise ValueError('Initialize rotary buffers before NF4 assembly')
    def check():
        if guard is not None:guard()
    count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,torch.nn.Linear):
                check();quantized=prepare_nf4_linear(child);quantized.to(device)
                if getattr(quantized.weight,'quant_state',None) is None:raise RuntimeError('CUDA NF4 state was not produced')
                setattr(module,name,quantized);count+=1;check()
            else:replace(child)
    for layer in model.model.layers:
        replace(layer);layer.to(device);gc.collect();check()
    if count!=7*len(model.model.layers):raise ValueError('Unexpected pinned decoder projection count')
    model.model.norm.to(device);model.eval().requires_grad_(False)
    # NativeLoop.__init__ would repeat conversion and allocate another stage.
    # This initializes its unchanged base and the attributes used by its API.
    loop=NativeLoop.__new__(NativeLoop)
    StreamedNanbeigeLoop.__init__(loop,model,device,len(model.model.layers),None)
    loop.quantized_linears=count;loop.channels=();loop.collect=False;loop.observations=[]
    loop.phase=0;loop.active_layer=-1
    assert loop.model.lm_head.weight.device.type=='cpu'
    return loop


def attach_cached_runtime(loop,*,prefill_chunk=128,vocabulary_chunk=8192,instrument_factory=None):
    """Attach the tested cache and one head stage, with cleanup on failure.

    Caller owns the loop, any logical-digest dispatch, and actual worker
    admission. Qualified BPD construction must precede the observer; the cache
    constructor enforces that hook order. No target head is moved to GPU.
    """
    from .chunked_cache import ChunkedTwoLoopCache
    from .streamed_vocab_head import StreamedVocabularyHead
    head=StreamedVocabularyHead(loop.model.lm_head.weight,loop.device,vocabulary_chunk)
    try:cache=ChunkedTwoLoopCache(loop,prefill_chunk=prefill_chunk,instrument_factory=instrument_factory)
    except BaseException:
        head.close();raise
    return cache,head
