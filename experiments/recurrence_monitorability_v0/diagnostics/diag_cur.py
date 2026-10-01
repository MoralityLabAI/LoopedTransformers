"""Scratch diagnostic: depth mixture versus depth curriculum on pointer chasing.

Usage: diag_cur.py <max_steps> <spec>   spec = arm:visits:max_depth:beta:lr:batch:flags
flags: d0 (add hop-count-0 examples), cur (unlock depth d+1 once depth d reaches 0.9)
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r"C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers")
from research_gym.interp.bundle import read_shard
from research_gym.interp.training import _eval_tensors, load_pool
from research_gym.neural.looped_decoder import LoopedDecoderLM

device = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(2500 / 16303, 0)
max_steps, spec = int(sys.argv[1]), sys.argv[2]
parts = spec.split(":")
arm, visits, max_depth, beta, lr, batch = parts[0], int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), int(parts[5])
flags = parts[6] if len(parts) > 6 else ""
pool = load_pool(Path(r"C:\Users\patri\Documents\Codex\rmp_v0\pool\pointer_chase.pt"))
bundle = Path(r"C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers\experiments\recurrence_monitorability_v0\datasets")
val = _eval_tensors(read_shard(bundle, "validation", "pointer_chase"))
torch.manual_seed(0)
tied = arm == "LOOP-T"
model = LoopedDecoderLM(
    vocab_size=2048, hidden_size=128, num_heads=4, physical_modules=1 if tied else visits,
    expanded_visits=visits, tying="fully_tied" if tied else "untied",
    parameter_visits=frozenset(range(visits)), retained_state_edges=frozenset(range(visits - 1)),
    supervision_points=(visits - 1,), alpha=1.0, beta=beta, normalization="pre", carry_policy="reset",
    module_word=(0,) * visits if tied else tuple(range(visits)), use_sinusoidal_positions=True, attention="explicit",
)
torch.nn.init.normal_(model.embedding.weight, std=128**-0.5)
model.to(device)
keep = pool.depths <= max_depth
tokens, targets, depths = pool.tokens[keep].long(), pool.targets[keep].long(), pool.depths[keep].long()
low = 1
if "d0" in flags:
    low = 0
    zero = tokens[: tokens.shape[0] // max_depth].clone()
    zero[:, 38] = 340
    tokens = torch.cat([tokens, zero])
    targets = torch.cat([targets, 800 + (zero[:, 36] - 300)])
    depths = torch.cat([depths, torch.zeros(zero.shape[0], dtype=torch.long)])
tokens, targets = tokens.to(device), targets.to(device)
by_depth = {d: torch.nonzero(depths == d).flatten().to(device) for d in range(low, max_depth + 1)}
mask = val["depths"] <= max_depth
vt, vy, vd = val["tokens"][mask][:1600].to(device), val["targets"][mask][:1600].to(device), val["depths"][mask][:1600].to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
generator = torch.Generator(device=device).manual_seed(0)
current = 1 if "cur" in flags else max_depth
started = time.perf_counter()
running = []
for step in range(1, max_steps + 1):
    for group in optimizer.param_groups:
        group["lr"] = lr * min(1.0, step / 200)
    choices = list(range(low, current + 1))
    share = batch // len(choices)
    pick = torch.cat([by_depth[d][torch.randint(by_depth[d].shape[0], (share,), generator=generator, device=device)] for d in choices])
    loss = torch.nn.functional.cross_entropy(model(tokens[pick]).logits[:, -1], targets[pick])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    running.append(loss.item())
    if step % 250 == 0:
        with torch.no_grad():
            exact = (model(vt).logits[:, -1].argmax(-1) == vy).float()
        accs = [float(exact[vd == d].mean()) for d in range(1, max_depth + 1)]
        print(f"{spec} step {step} cur {current} loss {sum(running) / len(running):.3f} acc {exact.mean():.3f} [{' '.join(f'{a:.2f}' for a in accs)}] {time.perf_counter() - started:.0f}s", flush=True)
        running = []
        if "cur" in flags and current < max_depth and accs[current - 1] >= 0.9:
            current += 1
        if min(accs) > 0.98:
            break
