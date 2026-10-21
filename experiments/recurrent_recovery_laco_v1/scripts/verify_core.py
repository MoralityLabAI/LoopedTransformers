"""Sanity checks: (1) our Qwen3 == HF Qwen3, (2) residual/shape invariants of the recurrent site,
(3) adapters are identity at init, (4) LaCo collapse arithmetic, (5) optional 4-bit path."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from looprec.laco import build_student, collapse_blocks, count_trainable
from looprec.qwen3_core import load_qwen3

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--quant", default="none")
a = ap.parse_args()

dev = "cuda"
t, d = load_qwen3(a.model)
ids = torch.randint(0, 50000, (2, 64))

# 1. against transformers
from transformers import AutoConfig, AutoModelForCausalLM
from looprec.qwen3_core import export_hf_state_dict
t = t.to(dev).eval()
cfg = AutoConfig.from_pretrained(d)
hf = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).to(dev)
sd = export_hf_state_dict(t)
sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
hf.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
hf = hf.eval()
t = t.to(dev).eval()
with torch.no_grad():
    ref = hf(ids.to(dev)).logits.float()
    hid, _ = t.forward_hidden(ids.to(dev))
    mine = t.logits(hid).float()
print("max |logit diff| vs HF:", (ref - mine).abs().max().item(), " (bf16 noise ~1e-1 at |logit|~20)")
print("argmax agreement      :", (ref.argmax(-1) == mine.argmax(-1)).float().mean().item())
with torch.no_grad():   # fp32 equivalence (rules out implementation differences hiding behind bf16 noise)
    hf.float(); t.float()
    ref32 = hf(ids.to(dev)).logits
    mine32 = t.logits(t.forward_hidden(ids.to(dev))[0])
    print("fp32 max |logit diff| vs HF:", (ref32 - mine32).abs().max().item())
    t.to(torch.bfloat16)
del hf, ref32, mine32
torch.cuda.empty_cache()

# 2. recurrent site invariants
t_cpu, _ = load_qwen3(a.model)
merges = [{"anchor": 15, "absorbed": list(range(16, 22))}]
rec = dict(max_loops=4, adapter="loop", rank=8, alpha=16, norm="loop", gate=True, init="laco")
st = build_student(t_cpu, merges, rec=rec, quant=a.quant).to(dev).eval()
site = st.site()
print("student entries:", len(st.entries), "| layer evals T=1:", st.n_layer_evals(1), "T=8:", st.n_layer_evals(8),
      "| trainable:", count_trainable(st))
with torch.no_grad():
    h0 = st.embed(ids.to(dev))
    # B==0 and gate==1 and norms == base -> T=1 must equal the plain collapsed block
    plain = build_student(t_cpu, merges, quant=a.quant).to(dev).eval()
    o1, _ = st.forward_hidden(ids.to(dev), n_loops=1)
    o0, _ = plain.forward_hidden(ids.to(dev))
    print("T=1 recurrent (fresh adapters) vs plain LaCo student, max diff:", (o1 - o0).abs().max().item())
    o2, _ = st.forward_hidden(ids.to(dev), n_loops=2)
    print("T=2 differs from T=1 (naive repeat is NOT identity):", (o2 - o1).abs().max().item() > 0)
    # residual identity: with F zeroed the loop must be the identity map on h
    blk = site.block
    cos_sin = None
    from looprec.qwen3_core import rope_cos_sin
    cos, sin = rope_cos_sin(64, t.cfg.head_dim, t.cfg.rope_theta, dev, torch.bfloat16)
    h = st.embed(ids.to(dev))
    f = blk.delta(h, cos, sin)
    assert f.shape == h.shape
    print("F(h) shape", tuple(f.shape), "== h shape", tuple(h.shape), "| block(h) - (h+F(h)) =",
          (blk(h, cos, sin) - (h + f)).abs().max().item())
    st2 = build_student(t_cpu, merges, rec=dict(rec, max_loops=16), quant=a.quant).to(dev).eval()
    for T in (1, 2, 4, 8, 16):
        o, states = st2.forward_hidden(ids.to(dev), n_loops=T, collect=True)
        assert o.shape == (2, 64, t.cfg.hidden_size)
        steps = [s for s in states if s[2] in ("step", "site_final")]
        assert len(steps) == T and all(s[1].shape == h.shape for s in steps)
    print("T in {1,2,4,8,16}: dims OK, collected steps OK")

# 3. collapse arithmetic
blocks = list(t_cpu.entries)
c = collapse_blocks(blocks, 15, [16, 17])
w = blocks[15].q_proj.weight.float() + (blocks[16].q_proj.weight.float() - blocks[15].q_proj.weight.float()) \
    + (blocks[17].q_proj.weight.float() - blocks[15].q_proj.weight.float())
print("collapse q_proj max diff vs manual:", (c.q_proj.weight.float() - w.to(torch.bfloat16).float()).abs().max().item())
print("peak GPU MB:", torch.cuda.max_memory_allocated() / 2**20)
