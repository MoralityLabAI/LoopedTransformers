"""Does the CPU cell solve modular_recurrence on its own training pool but not on validation?"""
import json
from pathlib import Path

import torch

from research_gym.architecture_discovery.checkpoint import load_verified_checkpoint
from research_gym.interp.bundle import read_shard
from research_gym.interp.models import build_model
from research_gym.interp.training import CellConfig, _eval_tensors, load_pool, predict

BUNDLE = Path(r"C:\Users\patri\rmp-aux-20261125\experiments\recurrence_monitorability_v0\datasets")
aux = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux")
rc = json.loads((aux / "pool" / "pool_receipt.json").read_text())
fam = "modular_recurrence"
pool = load_pool(aux / "pool" / rc[fam]["path"], rc[fam]["sha256"])
val = _eval_tensors(read_shard(BUNDLE, "validation", fam))
ck = load_verified_checkpoint(aux / "artifacts_cpu" / "LOOP-T-aux2cpu-s1" / "checkpoints" / "latest.pt", map_location="cpu")
cfg = CellConfig(arm="LOOP-T", seed=1, families=("masked_pointer_chase", fam), variant="aux2cpu", max_steps=1, beta=1.0)
model = build_model("LOOP-T", hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, visits=cfg.visits, beta=cfg.beta)
model.load_state_dict(ck["model"])
print("checkpoint step", ck["state"]["step"], "deepest", ck["state"]["deepest"])
print("pool tokens", tuple(pool.tokens.shape), "depth counts", torch.bincount(pool.depths).tolist())
print("val tokens", tuple(val["tokens"].shape), "depth counts", torch.bincount(val["depths"]).tolist())
print("pool token range", int(pool.tokens.min()), int(pool.tokens.max()), "val", int(val["tokens"].min()), int(val["tokens"].max()))
print("pool pad frac", float((pool.tokens[:2000] == 0).float().mean()), "val pad frac", float((val["tokens"][:2000] == 0).float().mean()))
for d in (0, 1, 2):
    idx = torch.nonzero(pool.depths == d).flatten()[:512]
    if idx.numel():
        p = predict(model, pool.tokens[idx].long(), cfg, device=torch.device("cpu"))
        print(f"TRAIN pool depth {d}: acc {float((p == pool.targets[idx].long()).float().mean()):.3f} n={idx.numel()}")
vi = torch.nonzero(val["depths"] == 1).flatten()[:512]
p = predict(model, val["tokens"][vi], cfg, device=torch.device("cpu"))
print(f"VALID depth 1: acc {float((p == val['targets'][vi]).float().mean()):.3f} n={vi.numel()}")
print("example train tokens", pool.tokens[torch.nonzero(pool.depths == 1).flatten()[0]].tolist(), "target", int(pool.targets[torch.nonzero(pool.depths == 1).flatten()[0]]))
print("example valid tokens", val["tokens"][vi[0]].tolist(), "target", int(val["targets"][vi[0]]))
