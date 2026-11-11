"""Scratch diagnostic: which training knobs let a small model leave the uniform plateau on pointer chasing?

Usage: diag_grid.py <device> <max_steps> <config>...   config = arm:visits:max_depth:beta:lr:batch[:ds]
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r"C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers")
from research_gym.interp.bundle import read_shard
from research_gym.interp.training import _eval_tensors, load_pool
from research_gym.neural.looped_decoder import LoopedDecoderLM

device = torch.device(sys.argv[1])
max_steps = int(sys.argv[2])
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(2500 / 16303, 0)
else:
    torch.set_num_threads(2)
pool = load_pool(Path(r"C:\Users\patri\Documents\Codex\rmp_v0\pool\pointer_chase.pt"))
bundle = Path(r"C:\Users\patri\OneDrive\Documents\GitHub\LoopedTransformers\experiments\recurrence_monitorability_v0\datasets")
val = _eval_tensors(read_shard(bundle, "validation", "pointer_chase"))

for spec in sys.argv[3:]:
    parts = spec.split(":")
    arm, visits, max_depth, beta, lr, batch = parts[0], int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), int(parts[5])
    deep = len(parts) > 6 and parts[6] == "ds"
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
    tokens, targets = pool.tokens[keep].long().to(device), pool.targets[keep].long().to(device)
    if len(parts) > 6 and "d0" in parts[6]:
        # Depth-0 copies: hop count 0, answer is the start node itself (positions 36 and 38).
        zero = tokens.clone()
        zero[:, 38] = 340
        tokens = torch.cat([tokens, zero])
        targets = torch.cat([targets, 800 + (zero[:, 36] - 300)])
    mask = val["depths"] <= max_depth
    vt, vy, vd = val["tokens"][mask][:1000].to(device), val["targets"][mask][:1000].to(device), val["depths"][mask][:1000].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(0)
    started = time.perf_counter()
    running = []
    for step in range(1, max_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, step / 200)
        pick = torch.randint(tokens.shape[0], (batch,), generator=generator).to(device)
        output = model(tokens[pick])
        if deep:
            loss = torch.stack([torch.nn.functional.cross_entropy(model.lens(s[:, -1]), targets[pick]) for s in output.visit_outputs]).mean()
        else:
            loss = torch.nn.functional.cross_entropy(output.logits[:, -1], targets[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running.append(loss.item())
        if step % 500 == 0:
            with torch.no_grad():
                exact = (model(vt).logits[:, -1].argmax(-1) == vy).float()
            by_depth = " ".join(f"{exact[vd == d].mean():.2f}" for d in range(1, max_depth + 1))
            print(f"{spec} step {step} loss {sum(running) / len(running):.3f} acc {exact.mean():.3f} [{by_depth}] {time.perf_counter() - started:.0f}s", flush=True)
            running = []
            if exact.mean() > 0.97:
                break
    del model, optimizer, tokens, targets
