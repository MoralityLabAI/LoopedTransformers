"""LOOPSCOPE-01 Q1/Q2: transported component attribution (TCA) on Huginn-0125 (see docs/loopscope01_registration.md).

Reproduces the 2026-11-03 heat-map setting exactly (same items, batches, noise seeds, components and random
directions as `rmp_huginn_heatmap`), then runs each item once forward and once backward with hooks on the four
core `mlp.proj` matrices. At core layer l and iteration t, with X the matrix's input and A the gradient of the
correct answer's log-probability with respect to its output (which carries every later iteration's repair):

- predicted effect of removing a rank-one component (v, u): `Δ = −Σ_pos (X v)(A u)`;
- fragility `F = ‖Xᵀ A‖_F`;
- matched nulls for one fixed random direction applied to every item: read rotation sd
  `‖v‖ ‖Σ_items Xᵀ A u‖ / (n √d_in)`, write rotation sd `‖u‖ ‖Σ_items Aᵀ X v‖ / (n √d_out)`.

Q1 compares window sums of the item-mean Δ with the 480 measured ablations in `huginn_heatmap/heatmap.json`.
Q2 reports z = mean Δ / sd for all 64 components × 4 layers × 16 iterations.

To reproduce the batched noise, the recurrent state of each batch is drawn exactly as the model draws it
(reseed, then `initialize_state` on the batch shape), and each item is run alone from its slice of that state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

from research_gym.scripts.rmp_huginn_depth_sweep import answer_tokens, load_huginn, make_item, score
from research_gym.scripts.rmp_nanbeige_vpd_heatmap import node_mask


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


class Capture:
    """Stores the input and the output gradient of every core `mlp.proj` call (call n is iteration n)."""

    def __init__(self, model):
        self.calls, self.x, self.g = {}, {}, {}
        model.register_forward_pre_hook(lambda module, args, kwargs: self.calls.clear(), with_kwargs=True)
        for i, block in enumerate(model.transformer.core_block):
            block.mlp.proj.register_forward_hook(self.make(i))

    def make(self, layer):
        def hook(module, inputs, output):
            n = self.calls[layer] = self.calls.get(layer, 0) + 1
            self.x[(layer, n)] = inputs[0].detach()
            if output.requires_grad:
                output.register_hook(lambda grad, key=(layer, n): self.g.__setitem__(key, grad.detach()))
        return hook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--revision", default="bb6621b65e90b6a4b9b29ef88dc83866d450470c")
    parser.add_argument("--heatmap", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/huginn_heatmap/heatmap.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/loopscope_huginn/tca.json"))
    args = parser.parse_args()
    hm = json.loads(args.heatmap.read_text(encoding="utf-8"))
    meta = hm["meta"]
    steps, seed, task = meta["steps"], meta["seed"], meta["task"]
    n_items, k, n_random, batch = 64, 64, 8, 8  # rmp_huginn_heatmap defaults used for heatmap.json
    device = torch.device("cuda")
    local = Path(snapshot_download(args.model, revision=args.revision))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(local / "tokenizer.json"),
                                  bos_token="<|begin_text|>", add_bos_token=True)
    labels, cand_ids = answer_tokens(tok, task)
    cand_t = torch.tensor(cand_ids, device=device)
    model = load_huginn(local, device)
    model.requires_grad_(False)

    # Components and random directions, built exactly as rmp_huginn_heatmap builds them.
    comps = {}
    for i, block in enumerate(model.transformer.core_block):
        U, S, Vh = torch.linalg.svd(block.mlp.proj.weight.float(), full_matrices=False)
        comps[i] = (Vh[:k].T.contiguous(), (U[:, :k] * S[:k]).T.contiguous())
    gen = torch.Generator(device="cpu").manual_seed(0)
    rand = {}
    for layer in range(4):
        V, U = comps[layer]
        vs, us = [], []
        for _ in range(n_random):
            c = int(torch.randint(V.shape[1], (1,), generator=gen))
            v = torch.randn(V.shape[0], generator=gen).to(device)
            u = torch.randn(U.shape[1], generator=gen).to(device)
            vs.append(v / v.norm() * V[:, c].norm())
            us.append(u / u.norm() * U[c].norm())
        rand[layer] = (torch.stack(vs, 1), torch.stack(us, 0))

    # Same clean-correct items and batches as the heat map.
    rng = np.random.default_rng(seed)
    kept, tried = [], 0
    while len(kept) < n_items and tried < 400:
        text, answer = make_item(rng, meta["hops"], meta["shots"], task)
        tried += 1
        ids, tgt = tok.encode(text), cand_ids[labels.index(answer)]
        _, ok = score(model, [(torch.tensor([ids], device=device), torch.tensor([tgt], device=device))],
                      steps, cand_t, seed)
        if ok.item():
            kept.append((ids, tgt))
    assert [len(kept), tried] == meta["clean_accuracy"], (len(kept), tried, meta["clean_accuracy"])
    by_len = {}
    for ids, tgt in kept[:n_items]:
        by_len.setdefault(len(ids), []).append((ids, tgt))
    batches = [g[i:i + batch] for g in by_len.values() for i in range(0, len(g), batch)]

    cap = Capture(model)
    d_in, d_out = comps[0][0].shape[0], comps[0][1].shape[1]
    delta = np.zeros((n_items, 4, steps, k))
    rdelta = np.zeros((n_items, 4, steps, n_random))
    frag = np.zeros((n_items, 4, steps))
    read_sum = {key: torch.zeros(d_in, k, device=device) for key in [(l, t) for l in range(4) for t in range(steps)]}
    write_sum = {key: torch.zeros(d_out, k, device=device) for key in read_sum}
    clean_alone, clean_batched = [], []
    item = 0
    started = time.monotonic()
    for part in batches:
        ids = torch.tensor([p[0] for p in part], device=device)
        tgt = torch.tensor([p[1] for p in part], device=device)
        with torch.no_grad():
            torch.manual_seed(seed)
            logits = model(input_ids=ids, num_steps=steps, use_cache=False).logits[:, -1].float()
            logits = logits.masked_fill(~node_mask(logits.shape[1], cand_t), float("-inf"))
            clean_batched += torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0].tolist()
            torch.manual_seed(seed)
            emb = model.transformer.wte(ids) * model.emb_scale
            state0 = model.initialize_state(emb)
        for b in range(len(part)):
            cap.x.clear(), cap.g.clear()
            s0 = state0[b:b + 1].clone().requires_grad_(True)
            # (no-grad steps, grad steps): an int would run every iteration under no_grad.
            logits = model(input_ids=ids[b:b + 1], input_states=s0, num_steps=(0, steps),
                           use_cache=False).logits[:, -1].float()
            logits = logits.masked_fill(~node_mask(logits.shape[1], cand_t), float("-inf"))
            y = torch.log_softmax(logits, -1)[0, tgt[b]]
            y.backward()
            clean_alone.append(float(y))
            with torch.no_grad():
                for layer in range(4):
                    V, U = comps[layer]
                    RV, RU = rand[layer]
                    for t in range(steps):
                        X = cap.x[(layer, t + 1)][0].float()
                        A = cap.g[(layer, t + 1)][0].float()
                        XV, AU = X @ V, A @ U.T
                        delta[item, layer, t] = (-(XV * AU).sum(0)).cpu().numpy()
                        rdelta[item, layer, t] = (-((X @ RV) * (A @ RU.T)).sum(0)).cpu().numpy()
                        frag[item, layer, t] = float(((X @ X.T) * (A @ A.T)).sum().clamp_min(0).sqrt())
                        read_sum[(layer, t)] += X.T @ AU
                        write_sum[(layer, t)] += A.T @ XV
            item += 1
        print(f"{item}/{n_items} items {time.monotonic() - started:.0f}s", flush=True)
    n = item
    diff = np.abs(np.array(clean_alone) - np.array(clean_batched))

    # Item-mean predicted effects, matched nulls for one fixed random direction across items, z-scores.
    mean_delta = delta.mean(0)
    sd_read = np.zeros((4, steps, k))
    sd_write = np.zeros((4, steps, k))
    for layer in range(4):
        unorm = comps[layer][1].norm(dim=1).cpu().numpy()
        for t in range(steps):
            sd_read[layer, t] = (read_sum[(layer, t)].norm(dim=0).cpu().numpy() / n) / np.sqrt(d_in)
            sd_write[layer, t] = unorm * (write_sum[(layer, t)].norm(dim=0).cpu().numpy() / n) / np.sqrt(d_out)
    z_read, z_write = mean_delta / sd_read, mean_delta / sd_write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), delta=delta, rdelta=rdelta, frag=frag, sd_read=sd_read,
                        sd_write=sd_write, clean_alone=np.array(clean_alone), clean_batched=np.array(clean_batched))

    # Q1: window sums vs measured ablations.
    windows = meta["windows"]
    pred, meas, names = [], [], []
    for key, entry in hm["ablation"].items():
        name, wname = key.rsplit("_", 1)
        a, b = windows[wname]
        layer = int(name.split("_")[0][4:])
        if "_rand" in name:
            r = int(name.split("_rand")[1])
            p = rdelta[:, layer, a - 1:b, r].sum(1).mean()
        else:
            c = int(name.split("_c")[1])
            p = delta[:, layer, a - 1:b, c].sum(1).mean()
        pred.append(float(p)), meas.append(entry["dlogp"]), names.append(key)
    pred, meas = np.array(pred), np.array(meas)
    big = np.abs(meas) >= 0.05
    q1 = {"n_conditions": len(meas), "spearman": spearman(pred, meas), "pearson": float(np.corrcoef(pred, meas)[0, 1]),
          "n_measured_ge_0.05": int(big.sum()),
          "sign_agreement_ge_0.05": float((np.sign(pred[big]) == np.sign(meas[big])).mean()) if big.any() else None,
          "slope_measured_on_predicted": float(np.polyfit(pred, meas, 1)[0])}
    q1["valid"] = bool(q1["spearman"] >= 0.6 and (q1["sign_agreement_ge_0.05"] or 0) >= 0.8)

    # Q2: specificity across all components, layers and iterations.
    top16 = {layer: sorted({int(nm.rsplit("_", 1)[0].split("_c")[1]) for nm in names if nm.startswith(f"core{layer}_c")}) for layer in range(4)}
    late = np.concatenate([np.abs(z_read[layer, 8:, top16[layer]]).ravel() for layer in range(4)])
    q2 = {"share_abs_z_read_gt3_all": float((np.abs(z_read) > 3).mean()),
          "share_abs_z_write_gt3_all": float((np.abs(z_write) > 3).mean()),
          "share_abs_z_read_gt3_top16_late": float((late > 3).mean()),
          "share_abs_z_read_gt3_by_layer_iteration": (np.abs(z_read) > 3).mean(2).round(3).tolist(),
          "fragility_mean_by_layer_iteration": frag.mean(0).round(4).tolist()}
    q2["falsified"] = bool(q2["share_abs_z_read_gt3_top16_late"] > 0.20)
    q2["prediction_share_lt_0.05"] = bool(q2["share_abs_z_read_gt3_all"] < 0.05)
    fr = frag.mean(0).mean(0)
    q2["fragility_late_over_early"] = float(fr[8:].mean() / fr[:8].mean())

    out = {"meta": {**{k_: meta[k_] for k_ in ("model", "revision", "steps", "task", "hops", "shots", "seed")},
                    "items": n, "registration": "docs/loopscope01_registration.md",
                    "reproduction_max_abs_logprob_diff": float(diff.max()),
                    "reproduction_mean_abs_logprob_diff": float(diff.mean())},
           "q1": q1, "q2": q2,
           "q1_conditions": [{"name": nm, "predicted": float(p), "measured": float(m)} for nm, p, m in zip(names, pred, meas)],
           "mean_delta": mean_delta.round(6).tolist(), "z_read": z_read.round(3).tolist(),
           "z_write": z_write.round(3).tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(json.dumps({"meta": out["meta"], "q1": q1, **{k_: v for k_, v in q2.items() if "by_layer" not in k_}}, indent=1))


if __name__ == "__main__":
    main()
