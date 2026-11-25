import json
from pathlib import Path

import torch

from research_gym.interp.bundle import read_shard
from research_gym.interp.training import _eval_tensors, load_pool

BUNDLE = Path(r"C:\Users\patri\rmp-aux-20261125\experiments\recurrence_monitorability_v0\datasets")
aux = Path(r"C:\Users\patri\Documents\Codex\rmp_v0_aux")
rc = json.loads((aux / "pool" / "pool_receipt.json").read_text())
for fam in ("modular_recurrence", "masked_pointer_chase"):
    pool = load_pool(aux / "pool" / rc[fam]["path"], rc[fam]["sha256"])
    val = _eval_tensors(read_shard(BUNDLE, "validation", fam))
    for d in (1, 2, 8):
        pi = torch.nonzero(pool.depths == d).flatten()
        vi = torch.nonzero(val["depths"] == d).flatten()
        if not vi.numel():
            continue
        ptok = {tuple(r) for r in pool.tokens[pi].tolist()}
        vtok = [tuple(r) for r in val["tokens"][vi].tolist()]
        ptgt = pool.targets[pi]
        print(f"{fam} depth {d}: train n={pi.numel()} unique inputs={len(ptok)}; val n={len(vtok)} unique={len(set(vtok))}; "
              f"val inputs seen in train={sum(v in ptok for v in vtok)}; train target classes={len(set(ptgt.tolist()))} val target classes={len(set(val['targets'][vi].tolist()))}; "
              f"val targets seen in train targets={sum(int(t) in set(ptgt.tolist()) for t in val['targets'][vi])}")
fam = "modular_recurrence"
val = _eval_tensors(read_shard(BUNDLE, "validation", fam)); pool = load_pool(aux / "pool" / rc[fam]["path"], rc[fam]["sha256"])
pi = torch.nonzero(pool.depths == 1).flatten()
print("train depth1 tokens at positions 3,5,7,8 (distinct values):", [sorted(set(pool.tokens[pi][:, k].tolist()))[:12] for k in (3, 5, 7, 8)])
vi = torch.nonzero(val["depths"] == 1).flatten()
print("val   depth1 tokens at positions 3,5,7,8 (distinct values):", [sorted(set(val["tokens"][vi][:, k].tolist()))[:12] for k in (3, 5, 7, 8)])
print("train targets sample", sorted(set(pool.targets[pi].tolist()))[:20], "val targets sample", sorted(set(val["targets"][vi].tolist()))[:20])
