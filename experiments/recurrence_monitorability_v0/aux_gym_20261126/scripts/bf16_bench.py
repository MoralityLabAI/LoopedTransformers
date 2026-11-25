"""fp32 vs bf16-autocast training benchmark for LOOP-T on the two new families.

Usage: python bf16_bench.py <device> <steps> <out.json>
Same seed, same batches. Decision rule: bf16 is adopted when it is >= 1.15x faster per step and its
mean loss over the last 20 steps is within 10% of fp32's (and finite).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig, cell_loss, load_pool

device = torch.device(sys.argv[1])
steps = int(sys.argv[2])
out = Path(sys.argv[3])
pool_dir = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux\pool")
rc = json.loads((pool_dir / "pool_receipt.json").read_text())
fams = ("masked_pointer_chase", "modular_recurrence")
pools = {n: load_pool(pool_dir / rc[n]["path"], rc[n]["sha256"]) for n in fams}
cfg = CellConfig(arm="LOOP-T", seed=1, families=fams, variant="bench", max_steps=1, beta=1.0, deep_supervision=True)
if device.type == "cpu":
    torch.set_num_threads(8)


def run(bf16: bool):
    os.environ["RMP_BF16"] = "1" if bf16 else "0"
    torch.manual_seed(0)
    model = build_model("LOOP-T", hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, visits=cfg.visits, beta=cfg.beta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(123)
    dp = {n: (p.tokens.to(device), p.targets.to(device)) for n, p in pools.items()}
    losses, times = [], []
    for step in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        total = 0.0
        for n in fams:
            idx = torch.randint(0, dp[n][0].shape[0], (256,), generator=gen).to(device)
            loss = cell_loss(model, dp[n][0][idx].long(), dp[n][1][idx].long(), cfg).mean()
            (loss / len(fams)).backward()
            total += float(loss) / len(fams)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(total)
    warm = sorted(times[5:])[len(times[5:]) // 2]
    return {"s_per_step_median": warm, "steps_per_s": 1 / warm, "loss_first": losses[0], "loss_last20": sum(losses[-20:]) / 20,
            "finite": all(l == l and abs(l) < 1e6 for l in losses),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}


res = {"device": str(device), "steps": steps, "fp32": run(False), "bf16": run(True)}
sp = res["fp32"]["s_per_step_median"] / res["bf16"]["s_per_step_median"]
ratio = res["bf16"]["loss_last20"] / res["fp32"]["loss_last20"]
res["speedup"] = sp
res["loss_ratio_bf16_over_fp32"] = ratio
res["adopt_bf16"] = bool(sp >= 1.15 and ratio <= 1.10 and res["bf16"]["finite"])
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res))
