"""Write a tiny random Qwen3-architecture checkpoint (HF layout) for CPU pipeline tests.

  python scripts/make_tiny_model.py --out checkpoints/tiny_base [--layers 8]
Uses the real Qwen3 tokenizer/vocab so text + task pipelines run unchanged.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from safetensors.torch import save_file

from looprec.qwen3_core import resolve_model_dir

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--src", default="Qwen/Qwen3-0.6B-Base")
a = ap.parse_args()

src = resolve_model_dir(a.src)
cfg = json.load(open(os.path.join(src, "config.json")))
H, nh, nkv, hd, inter = a.hidden, 4, 2, 16, 2 * a.hidden
cfg.update(hidden_size=H, num_hidden_layers=a.layers, num_attention_heads=nh, num_key_value_heads=nkv, head_dim=hd,
           intermediate_size=inter, max_position_embeddings=2048)
os.makedirs(a.out, exist_ok=True)
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
shutil.copy(os.path.join(src, "tokenizer.json"), a.out)
g = torch.Generator().manual_seed(0)
r = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(torch.bfloat16)
sd = {"model.embed_tokens.weight": r(cfg["vocab_size"], H), "model.norm.weight": torch.ones(H, dtype=torch.bfloat16)}
for i in range(a.layers):
    p = f"model.layers.{i}."
    sd[p + "self_attn.q_proj.weight"] = r(nh * hd, H)
    sd[p + "self_attn.k_proj.weight"] = r(nkv * hd, H)
    sd[p + "self_attn.v_proj.weight"] = r(nkv * hd, H)
    sd[p + "self_attn.o_proj.weight"] = r(H, nh * hd)
    sd[p + "self_attn.q_norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
    sd[p + "self_attn.k_norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
    sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "mlp.gate_proj.weight"] = r(inter, H)
    sd[p + "mlp.up_proj.weight"] = r(inter, H)
    sd[p + "mlp.down_proj.weight"] = r(H, inter)
save_file(sd, os.path.join(a.out, "model.safetensors"))
print("wrote", a.out)
