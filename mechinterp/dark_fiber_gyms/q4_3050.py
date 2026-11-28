"""Experimental 4-bit Nanbeige full-state ARG-03 backend for 4 GiB RTX 3050.

Loads SHA-pinned original safetensors to system RAM through the existing
no-double-copy meta loader, then quantizes decoder torch.nn.Linear modules
one at a time into CUDA NF4 (double quantization). Embeddings stay on CPU;
the large full-vocabulary lm_head stays BF16 on GPU. The exact same model
state at each loop and scored logits remain accessible to the ARG-03 gym.

This is a NEW quantized inference condition. It cannot be substituted for
the registered BF16 evidence; compare only quantized clean vs quantized
interventions, and report BF16-to-NF4 differences if a paired calibration
is run separately. Requires bitsandbytes CUDA NF4 support and may still OOM
if 4 GiB VRAM is fragmented or occupied.
"""
from __future__ import annotations

import gc
from pathlib import Path
import torch


def replace_linear4bit(module: torch.nn.Module, device: torch.device, *,
                       quant_type: str = "nf4", bnb=None) -> int:
    """Quantize all decoder linear leaves, keeping non-linear wrapper structure.

    Accepts optional bnb dependency for CPU fixture tests. With real bnb, the
    CPU BF16 weights are compressed as the individual Linear4bit goes to CUDA.
    """
    if bnb is None:
        import bitsandbytes as bnb
    changed = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            if child.weight.device.type != "cpu":
                raise ValueError("source linear weights must be on CPU before NF4 conversion")
            q = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=torch.bfloat16, compress_statistics=True,
                quant_type=quant_type,
            )
            q.weight = bnb.nn.Params4bit(
                child.weight.detach().contiguous(), requires_grad=False,
                compress_statistics=True, quant_type=quant_type,
            )
            if child.bias is not None:
                q.bias = torch.nn.Parameter(child.bias.detach().clone(), requires_grad=False)
            q = q.to(device)
            setattr(module, name, q)
            changed += 1
        else:
            changed += replace_linear4bit(child, device, quant_type=quant_type, bnb=bnb)
    return changed


def _slice_context(ctx: dict, lo: int, hi: int, batch: int) -> dict:
    sliced = {}
    for key, value in ctx.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
            sliced[key] = value[lo:hi]
        else:
            sliced[key] = value
    return sliced


class Q4NanbeigeLoop:
    """Same no-cache 22-layer loop API as ARG-03; small candidate microbatches."""

    def __init__(self, model, device: torch.device, rows_per_batch: int = 1):
        if rows_per_batch < 1:
            raise ValueError("rows_per_batch must be >=1")
        self.model = model
        self.base = model.model
        self.layers = self.base.layers
        self.n = len(self.layers)
        self.block = (0, self.n)
        self.n_gpu_layers = self.n
        self._device = torch.device(device)
        self.rows_per_batch = rows_per_batch

    @property
    def device(self):
        return self._device

    @torch.inference_mode()
    def prepare(self, ids: torch.Tensor):
        # The 166k-token embedding is left in host RAM.
        emb = self.base.embed_tokens(ids.to("cpu")).to(self._device)
        rows, length = ids.shape
        position = torch.arange(length, device=self._device)
        pos = position.unsqueeze(0).expand(rows, -1)
        mask = self.base._update_causal_mask(None, emb, position, None, False)
        return emb, {"mask":mask, "pos":pos, "cache_position":position}

    @torch.inference_mode()
    def run_pairs(self, items: list[tuple[torch.Tensor,dict]]) -> list[torch.Tensor]:
        outputs = []
        for states, context in items:
            n = states.shape[0]
            chunks = []
            for begin in range(0, n, self.rows_per_batch):
                end = min(n, begin+self.rows_per_batch)
                h = states[begin:end]
                ctx = _slice_context(context, begin, end, n)
                for layer in self.layers:
                    h = layer(
                        h, attention_mask=ctx["mask"],
                        position_ids=ctx["pos"], past_key_value=None,
                        output_attentions=False, use_cache=False,
                        cache_position=ctx["cache_position"], loop_idx=0,
                    )[0]
                chunks.append(self.base.norm(h))
            outputs.append(torch.cat(chunks, dim=0))
        return outputs

    @torch.inference_mode()
    def run(self, h, lo, hi, ctx):
        if (lo,hi)!=(0,self.n):
            raise ValueError("Nanbeige Q4 repeats whole 22-layer stack only")
        return self.run_pairs([(h,ctx)])[0]

    @torch.inference_mode()
    def readout(self, h, ctx):
        return torch.log_softmax(self.model.lm_head(h[:,-3:,:]).float(),dim=-1)


def load_engine(compactification_root: Path, device="cuda:0", rows_per_batch=1):
    """Load pinned Nanbeige into CPU, quantize only decoder linears on RTX 3050."""
    import sys
    from . import arg03_nanbeige as NB
    NB.assert_oracle_isolated()
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the NF4 backend needs a CUDA GPU")
    if rows_per_batch < 1:
        raise ValueError("rows_per_batch must be positive")
    import bitsandbytes as bnb
    if not hasattr(bnb.nn, "Linear4bit"):
        raise RuntimeError("bitsandbytes Linear4bit is unavailable")
    sys.path.insert(0, str(compactification_root.resolve()))
    from research.algernon.algebraic_reasoning_gym.arg02 import runner, nanbeige as arg02_nb
    from research.algernon.algebraic_reasoning_gym.arg02.settings import load_lock
    entry = load_lock()["nanbeige"]
    snap = runner.verify_snapshot(entry)
    # Original pinned BF16 checkpoint is staged only in host RAM. No extra CPU
    # from_pretrained model copy; the existing meta loader streams safetensors.
    model,_ = NB.build_nanbeige_model_and_loop(
        snap, exec_device=torch.device("cpu"), dtype=torch.bfloat16,
        attn=entry.get("attn_implementation","sdpa"), n_gpu_layers=22,
    )
    arg02_nb.check_supported(model.config)
    converted=0
    for layer in model.model.layers:
        converted+=replace_linear4bit(layer,dev,bnb=bnb)
        layer.to(dev)
        gc.collect()
    if converted < 22*4:
        raise AssertionError(f"unexpected number of quantized decoder projections: {converted}")
    # lm_head is large but needed for exact full-vocabulary parity;
    # embeddings remain on CPU and are transferred per microbatch.
    model.model.norm.to(dev)
    model.lm_head.to(dev)
    model.eval()
    loop=Q4NanbeigeLoop(model,dev,rows_per_batch=rows_per_batch)
    tok=arg02_nb.load_tokenizer(snap,entry["bos_token_id"])
    engine=runner.Engine(model,tok,block=(0,loop.n),shots=[],device=str(dev))
    engine.loop=loop
    NB.assert_oracle_isolated()
    return engine,entry,{"backend":"bnb-nf4","quant_type":"nf4",
                         "double_quant":True,"quantized_linear_count":converted,
                         "rows_per_batch":rows_per_batch,
                         "embedding_location":"cpu",
                         "lm_head_dtype":"bfloat16",
                         "recurrent_head_intervention":"all internal states and logits accessible"}
