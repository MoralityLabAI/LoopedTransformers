import sys, time
import torch
from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig, cell_loss, load_pool

pool_dir = r"C:\Users\patri\Documents\Codex\rmp_v0_aux\pool"
import json
rc = json.load(open(pool_dir + r"\pool_receipt.json"))
pools = {n: load_pool(__import__("pathlib").Path(pool_dir) / rc[n]["path"], rc[n]["sha256"]) for n in ("masked_pointer_chase", "modular_recurrence")}
print({n: tuple(p.tokens.shape) for n, p in pools.items()}, flush=True)
cfg = CellConfig(arm="LOOP-T", seed=1, families=tuple(pools), variant="x", max_steps=1, beta=1.0)
for threads in (4, 8, 12, 20):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_model("LOOP-T", hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, visits=cfg.visits, beta=cfg.beta)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for batch in (256,):
        times = []
        for step in range(6):
            t0 = time.perf_counter()
            opt.zero_grad()
            for n, p in pools.items():   # one microbatch per family, as the runner does
                idx = torch.randint(0, p.tokens.shape[0], (batch,))
                loss = cell_loss(model, p.tokens[idx].long(), p.targets[idx].long(), cfg).mean()
                loss.backward()
            opt.step()
            times.append(time.perf_counter() - t0)
        warm = sorted(times[1:])[len(times[1:]) // 2]
        print(f"threads={threads} batch/family={batch} s/step={warm:.2f} steps/s={1/warm:.2f}", flush=True)
