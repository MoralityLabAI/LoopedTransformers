"""Huginn-0125 per-iteration component heat map (exploratory, local, not a registered endpoint).

Huginn applies the same 4-layer core block `r` times. There is no VPD fit for it, so the components are the
top singular directions of each core block's MLP output matrix (`mlp.proj`, 17920 -> 5280): component c
contributes `(x @ Vh[c]) * S[c] * U[:, c]`. Any later VPD fit has to beat this baseline.

- **Items.** From `rmp_huginn_depth_sweep.make_item` (variable chains by default) at one hop count and shot
  setting. Only items the clean model answers correctly (forced choice over the answer tokens) are kept.
- **Usage map.** On the clean run, mean `|x @ Vh[c]| * S[c]` at the answer position, per core layer, per
  iteration, for every component.
- **Ablation map.** The components with the highest total usage per layer are removed in one of four equal
  iteration bins or in all iterations; the script records the mean change in log p(correct) and accuracy.
- **Control.** Random rank-one directions, norm-matched to a random component of the same layer.

Results are written after every condition, so a stopped run resumes where it left off.
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


class Hooks:
    """Counts each core-layer `mlp.proj` call within a forward (call n is iteration n) and applies the edit."""

    def __init__(self, model, k, device):
        self.calls, self.edit, self.usage = {}, None, None
        self.comps = {}
        for i, block in enumerate(model.transformer.core_block):
            W = block.mlp.proj.weight.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.comps[i] = (Vh[:k].T.contiguous(), (U[:, :k] * S[:k]).T.contiguous())  # V [d_in, k], U [k, d_out]
            block.mlp.proj.register_forward_hook(self.make(i))
        model.register_forward_pre_hook(lambda module, args, kwargs: self.calls.clear(), with_kwargs=True)

    def make(self, layer):
        def hook(module, inputs, output):
            n = self.calls[layer] = self.calls.get(layer, 0) + 1
            x = inputs[0].float()
            if self.usage is not None:
                V, U = self.comps[layer]
                acts = (x[:, -1] @ V).abs() * U.norm(dim=1)
                self.usage.setdefault((layer, n), []).append((acts.sum(0).cpu(), x.shape[0]))
            if self.edit is None or self.edit[0] != layer or n not in self.edit[1]:
                return None
            v, u = self.edit[2], self.edit[3]
            return output - ((x @ v)[..., None] * u).to(output.dtype)
        return hook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--revision", default="bb6621b65e90b6a4b9b29ef88dc83866d450470c")
    parser.add_argument("--out", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/huginn_heatmap/heatmap.json"))
    parser.add_argument("--task", choices=["varchain", "chase"], default="varchain")
    parser.add_argument("--steps", type=int, default=16, help="recurrence r for the heat map")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--components", type=int, default=64, help="singular directions per layer for usage")
    parser.add_argument("--ablate", type=int, default=16, help="most-used components ablated per layer")
    parser.add_argument("--random", type=int, default=8)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    device = torch.device("cuda")
    local = Path(snapshot_download(args.model, revision=args.revision))
    tok = PreTrainedTokenizerFast(tokenizer_file=str(local / "tokenizer.json"),
                                  bos_token="<|begin_text|>", add_bos_token=True)
    labels, cand_ids = answer_tokens(tok, args.task)
    node_t = torch.tensor(cand_ids, device=device)
    model = load_huginn(local, device)
    hooks = Hooks(model, args.components, device)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    report.setdefault("meta", {"model": args.model, "revision": args.revision, "steps": args.steps,
                               "task": args.task, "hops": args.hops, "shots": args.shots, "seed": args.seed,
                               "components": "top singular directions of core mlp.proj",
                               "readout": f"forced choice over {len(labels)} answer tokens"})

    # Clean-correct items, batched by equal token length (no padding).
    rng = np.random.default_rng(args.seed)
    kept, tried = [], 0
    while len(kept) < args.items and tried < args.max_candidates:
        text, answer = make_item(rng, args.hops, args.shots, args.task)
        tried += 1
        ids, tgt = tok.encode(text), cand_ids[labels.index(answer)]
        _, ok = score(model, [(torch.tensor([ids], device=device), torch.tensor([tgt], device=device))],
                      args.steps, node_t, args.seed)
        if ok.item():
            kept.append((ids, tgt))
    report["meta"]["clean_accuracy"] = [len(kept), tried]
    print(f"kept {len(kept)}, clean correct {len(kept)}/{tried}", flush=True)
    kept = kept[:args.items]
    assert hooks.calls == {i: args.steps for i in range(4)}, hooks.calls
    by_len = {}
    for ids, tgt in kept:
        by_len.setdefault(len(ids), []).append((ids, tgt))
    batches = [(torch.tensor([p[0] for p in g[i:i + args.batch]], device=device),
                torch.tensor([p[1] for p in g[i:i + args.batch]], device=device))
               for g in by_len.values() for i in range(0, len(g), args.batch)]

    hooks.usage = {}
    clean, _ = score(model, batches, args.steps, node_t, args.seed)
    usage = {}
    for (layer, n), rows in hooks.usage.items():
        usage.setdefault(f"core{layer}", [None] * args.steps)[n - 1] = (
            sum(r[0] for r in rows) / sum(r[1] for r in rows)).tolist()
    hooks.usage = None
    report["usage"] = usage
    report["clean_mean_logprob"] = float(clean.mean())

    edges = np.linspace(0, args.steps, args.bins + 1).round().astype(int)
    windows = {f"it{a + 1}-{b}": tuple(range(a + 1, b + 1)) for a, b in zip(edges[:-1], edges[1:])}
    windows["all"] = tuple(range(1, args.steps + 1))
    report["meta"]["windows"] = {k: [v[0], v[-1]] for k, v in windows.items()}
    gen = torch.Generator(device="cpu").manual_seed(0)
    conditions = []
    for layer in range(4):
        V, U = hooks.comps[layer]
        total = torch.tensor(usage[f"core{layer}"]).sum(0)
        for c in total.argsort(descending=True)[:args.ablate].tolist():
            conditions.append((f"core{layer}_c{c}", layer, V[:, c], U[c]))
        for r in range(args.random):
            c = int(torch.randint(V.shape[1], (1,), generator=gen))
            v = torch.randn(V.shape[0], generator=gen).to(device)
            u = torch.randn(U.shape[1], generator=gen).to(device)
            conditions.append((f"core{layer}_rand{r}", layer, v / v.norm() * V[:, c].norm(), u / u.norm() * U[c].norm()))
    results = report.setdefault("ablation", {})
    started = time.monotonic()
    total_runs = len(conditions) * len(windows)
    done = 0
    for name, layer, v, u in conditions:
        for wname, its in windows.items():
            done += 1
            key = f"{name}_{wname}"
            if key in results:
                continue
            hooks.edit = (layer, its, v, u)
            lp, ok = score(model, batches, args.steps, node_t, args.seed)
            hooks.edit = None
            results[key] = {"dlogp": float((lp - clean).mean()), "acc": float(ok.float().mean())}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            print(f"{done}/{total_runs} {key} {results[key]['dlogp']:+.4f} {time.monotonic() - started:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
