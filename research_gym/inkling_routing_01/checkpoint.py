"""Verify and load the official fixed-loop checkpoint with packed expert offload."""

import hashlib
import json
from pathlib import Path

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
MODEL_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
TRANSFORMERS_VERSION = "5.24.0"
TRANSFORMERS_REVISION = "e3a65e2c21abb9c25a03704f2f25852f9ad03a18"
WEIGHTS = "model-00001-of-00001.safetensors"
FILE_HASHES = {
    WEIGHTS: "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1",
    "tokenizer.json": "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
}
GIT_BLOBS = {
    "config.json": "e6c170c3fb666c50e3b509493b6b7a8f948b7d15",
    "model.safetensors.index.json": "3b835a38066039133931b604a0734dfae89a644e",
    "special_tokens_map.json": "e4fe9735b528be1c630eb832d655fc8c4c2c309f",
    "tokenizer_config.json": "abf8fb60451e6c33d2cfa3c4dfab9f8c1e7516aa",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_metadata(snapshot: Path):
    for name, expected in GIT_BLOBS.items():
        content = (snapshot / name).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_blob != expected:
            raise ValueError(f"checkpoint Git blob mismatch: {name}")
    tokenizer_hash = file_sha256(snapshot / "tokenizer.json")
    if tokenizer_hash != FILE_HASHES["tokenizer.json"]:
        raise ValueError("checkpoint tokenizer SHA-256 mismatch")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    loop = text["loop_config"]
    expected = ("inkling2_loop", ["Inkling2LoopForConditionalGeneration"], 2304, 24, 128, 8, 6, 18, 3, False)
    actual = (
        config["model_type"], config["architectures"], text["hidden_size"], text["num_hidden_layers"],
        text["n_routed_experts"], text["num_experts_per_tok"], loop["core_start"], loop["core_end"],
        loop["num_passes"], loop["is_varloop"],
    )
    if actual != expected:
        raise ValueError(f"unexpected checkpoint architecture: {actual}")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "loop_config": loop,
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "unpadded_vocab_size": text["unpadded_vocab_size"],
        "tokenizer_eos_id": config["eos_token_id"],
        "tokenizer_sha256": tokenizer_hash,
        "verified_git_blobs": GIT_BLOBS,
        "source_quantization": config["quantization_config"],
    }


def packed_model_class(weight_file):
    """Keep original NVFP4/MXFP8 expert weights mapped on disk, decoding active experts only."""
    import torch
    import torch.nn.functional as F
    from transformers import Inkling2LoopForCausalLM
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopExperts

    available = set(weight_file.keys())
    expected_packed = set()
    vision_keys = {key for key in available if key.startswith("model.visual.")}
    dequantize = Inkling2LoopDequantizeExperts()
    interleave = Interleave(dim=1)

    class PackedExperts(Inkling2LoopExperts):
        def __init__(self, config, index):
            super().__init__(config)
            # The full parameter placeholders are meta tensors during from_pretrained.
            del self.gate_up_proj
            del self.down_proj
            prefix = f"model.llm.layers.{index}.mlp.experts."
            self.sources = []
            for projection in ("w13", "w2"):
                base = prefix + projection + "_weight"
                keys = [base, base + "_scale", base + "_amax", base + "_outer_scale_denominator"]
                required = keys if keys[2] in available else keys[:2]
                if not set(required) <= available or (keys[3] in available) != (keys[2] in available):
                    raise ValueError(f"incomplete packed expert projection: {base}")
                expected_packed.update(required)
                self.sources.append([key if key in available else None for key in keys])

        def read_projection(self, keys, experts, device):
            inputs = {}
            for key in (key for key in keys if key is not None):
                source = weight_file.get_slice(key)
                if key.endswith("_outer_scale_denominator") and source.get_shape() == [1]:
                    inputs[key] = source[:].to(device)
                    continue
                slices = [source[index:index + 1] for index in experts]
                if slices[0].element_size() == 1:
                    joined = torch.cat([part.view(torch.uint8) for part in slices]).view(slices[0].dtype)
                else:
                    joined = torch.cat(slices)
                inputs[key] = joined.to(device)
            return dequantize.convert(inputs, keys, ["weight"])["weight"]

        def forward(self, hidden_states, top_k_index, top_k_weights):
            # The routing order, expert math and fp32 accumulation match the eager HF expert.
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            active = mask[:self.num_experts].any(dim=(-2, -1)).nonzero().flatten().tolist()
            output = torch.zeros_like(hidden_states, dtype=torch.float32)
            chunk_size = dequantize.EXPERTS_PER_CHUNK
            for start in range(0, len(active), chunk_size):
                experts = active[start:start + chunk_size]
                gate = self.read_projection(self.sources[0], experts, hidden_states.device)
                gate = interleave.convert({"weight": gate}, ["weight"], ["weight"])["weight"]
                down = self.read_projection(self.sources[1], experts, hidden_states.device)
                for offset, index in enumerate(experts):
                    slot, token = torch.where(mask[index])
                    value = self._apply_gate(F.linear(hidden_states[token], gate[offset].to(hidden_states.dtype)))
                    value = F.linear(value, down[offset].to(hidden_states.dtype))
                    value = value * top_k_weights[token, slot, None]
                    output.index_add_(0, token, value.float())
            return output.to(hidden_states.dtype)

    class PackedInkling(Inkling2LoopForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            for index, layer in enumerate(self.model.layers):
                layer.mlp.experts = PackedExperts(config, index)

    return PackedInkling, expected_packed, vision_keys


def load_model(snapshot, weight_file, device):
    import torch
    from transformers import AutoConfig
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping

    model_class, expected_packed, vision_keys = packed_model_class(weight_file)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False).text_config
    # The generic NVFP4 loader cannot ingest this ModelOpt layout. The original
    # packed experts above use the release's own Inkling-specific converter.
    # Opt this local subclass into the same checkpoint mapping as the upstream class.
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping(config.model_type), overwrite=True,
    )
    model, diagnostics = model_class.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        experts_implementation="eager",
        device_map={"": device},
        output_loading_info=True,
    )
    packed_targets = {
        f"model.layers.{index}.mlp.experts.{projection}"
        for index in range(config.num_hidden_layers)
        for projection in ("gate_up_proj", "down_proj")
    }
    unexpected = set(diagnostics["unexpected_keys"]) - expected_packed - vision_keys - packed_targets
    problems = {key: value for key, value in diagnostics.items() if key != "unexpected_keys" and value}
    if unexpected or problems:
        raise RuntimeError(f"unexpected checkpoint loading diagnostics: {sorted(unexpected)} / {problems}")
    return model.eval(), len(expected_packed)
