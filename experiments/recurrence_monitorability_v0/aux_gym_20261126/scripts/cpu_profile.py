"""Per-op CPU profile of one LOOP-T training step (batch 256 per family x 2 families, 8 visits, deep supervision)."""
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig, cell_loss, load_pool

bf16 = sys.argv[1] == "bf16"
os.environ["RMP_BF16"] = "1" if bf16 else "0"
torch.set_num_threads(8)
d = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux\pool")
rc = json.loads((d / "pool_receipt.json").read_text())
fams = ("masked_pointer_chase", "modular_recurrence")
pools = {n: load_pool(d / rc[n]["path"], rc[n]["sha256"]) for n in fams}
cfg = CellConfig(arm="LOOP-T", seed=1, families=fams, variant="p", max_steps=1, beta=1.0, deep_supervision=True)
model = build_model("LOOP-T", hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, visits=cfg.visits, beta=cfg.beta)
print("model", model)
print("params", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)


def step():
    opt.zero_grad()
    for n in fams:
        i = torch.randint(0, 1000000, (256,))
        cell_loss(model, pools[n].tokens[i].long(), pools[n].targets[i].long(), cfg).mean().backward()
    opt.step()


for _ in range(2):
    step()
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    step()
wall = time.perf_counter() - t0
print(f"mode={'bf16' if bf16 else 'fp32'} one profiled step wall {wall:.2f}s")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=14, max_name_column_width=44))
