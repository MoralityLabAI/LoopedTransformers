"""Nanbeige per-pass VPD component heat map (exploratory, local, not a registered endpoint).

Nanbeige4.2-3B runs its 22 layers twice with the same weights. The paper-1 VPD fit splits `down_proj` at
physical layers 12 and 22 into 64 rank-one components each (`W` contribution of component c is
`(x @ V[:, c]) * U[c]`, the rest of `W` kept as a residual).

- **Items.** Pointer chases in the runbook's prompt format, three-shot, at 1 hop by default. The readout is a
  forced choice among the 16 node tokens at the answer position: the model's unrestricted first token is
  usually a newline. Only items the clean model answers correctly under that readout are kept.
- **Ablation map.** Each component is removed on pass 1, pass 2 or both. For each condition the script
  records the mean change in the correct answer's log-probability, and the accuracy, per hop count.
- **Usage map.** On the clean run, the mean `|x @ V[:, c]| * ||U[c]||` per pass, at the answer position
  and over all positions.
- **Control.** Random rank-one directions whose norms match randomly chosen components.

Results are written after every condition, so a stopped run resumes where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

NODES = [chr(ord("A") + i) for i in range(16)]
LAYERS = (11, 21)  # zero-based; physical layers 12 and 22
PASSES = {"pass1": (1,), "pass2": (2,), "both": (1, 2)}


def chase(rng, hops):
    while True:
        perm = rng.permutation(16)
        start = int(rng.integers(16))
        walk = [start]
        for _ in range(hops):
            walk.append(int(perm[walk[-1]]))
        if len(set(walk)) == len(walk):
            break
    order = rng.permutation(16)
    edges = ", ".join(f"{NODES[i]}->{NODES[perm[i]]}" for i in order)
    return f"Edges: {edges}\nStart: {NODES[start]}\nHops: {hops}\nAnswer:", NODES[walk[-1]]


def make_prompt(rng, hops):
    demos = []
    for _ in range(3):
        text, answer = chase(rng, int(rng.integers(1, 4)))
        demos.append(f"{text} {answer}")
    text, answer = chase(rng, hops)
    return "\n\n".join(demos + [text]), answer


def load_on_gpu(name, config, device):
    """Build the model in bf16 on the GPU and load each safetensors shard straight onto it.

    This avoids `from_pretrained`'s CPU copy, which exceeds the shared PC's commit limit, and needs no
    Accelerate.
    """
    from safetensors.torch import load_file
    index = json.loads((name / "model.safetensors.index.json").read_text())
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    torch.set_default_dtype(torch.float32)
    expected = set(model.state_dict())
    loaded = set()
    for shard in sorted(set(index["weight_map"].values())):
        state = load_file(str(name / shard), device=str(device))
        model.load_state_dict(state, strict=False)
        loaded |= set(state)
        del state
    missing = sorted(k for k in expected - loaded if "rotary" not in k)
    assert not missing, f"weights not loaded: {missing[:5]}"
    return model.eval()


class Hooks:
    """Counts each down_proj call within a forward (call n is pass n) and applies the active edit."""

    def __init__(self, model, factors, device):
        self.calls, self.edit, self.usage = {}, None, None
        self.factors = {L: (torch.tensor(factors[f"model.layers.{L}.mlp.down_proj__V"], device=device),
                            torch.tensor(factors[f"model.layers.{L}.mlp.down_proj__U"], device=device))
                        for L in LAYERS}
        model.register_forward_pre_hook(lambda module, args: self.calls.clear())
        for L in LAYERS:
            model.model.layers[L].mlp.down_proj.register_forward_hook(self.make(L))

    def make(self, layer):
        def hook(module, inputs, output):
            n = self.calls[layer] = self.calls.get(layer, 0) + 1
            x = inputs[0].float()
            if self.usage is not None:
                V, U = self.factors[layer]
                acts = (x @ V).abs() * U.norm(dim=1)
                self.usage.setdefault((layer, n), []).append(
                    (acts[:, -1].sum(0).cpu(), acts.mean(1).sum(0).cpu(), x.shape[0]))
            if self.edit is None or self.edit[0] != layer or n not in self.edit[1]:
                return None
            v, u = self.edit[2], self.edit[3]
            return output - ((x @ v)[..., None] * u).to(output.dtype)
        return hook


def node_mask(vocab, node_ids):
    mask = torch.zeros(vocab, dtype=torch.bool, device=node_ids.device)
    mask[node_ids] = True
    return mask


@torch.no_grad()
def answer_logprobs(model, batches, targets, node_ids):
    """Log-probability of the target among the 16 node tokens, and whether it is the top node."""
    lps, correct = [], []
    for ids, tgt in zip(batches, targets):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1].float()
        logits = logits.index_fill(1, (~node_mask(logits.shape[1], node_ids)).nonzero()[:, 0], float("-inf"))
        lp = torch.log_softmax(logits, -1)
        lps.append(lp.gather(1, tgt[:, None])[:, 0].cpu())
        correct.append((logits.argmax(-1) == tgt).cpu())
    return torch.cat(lps), torch.cat(correct)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Nanbeige/Nanbeige4.2-3B")
    parser.add_argument("--revision", default="451ed48c3273ecef7ea8faaa43c31ce529763bb1")
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/nanbeige_heatmap/heatmap.json"))
    parser.add_argument("--per-hop", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--random", type=int, default=32)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--hops", type=int, nargs="+", default=[1],
                        help="hop counts to use; Nanbeige is at chance from 2 hops (smoke test: 2/48)")
    args = parser.parse_args()
    device = torch.device("cuda")
    # Run under transformers 4.4x: the checkpoint's remote code (written for 4.42) generates degraded text
    # under 5.x. Without protobuf/sentencepiece, load tokenizer.json directly; the ids are identical.
    try:
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    except ImportError:
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(snapshot_download(args.model, revision=args.revision)) / "tokenizer.json"))
    node_ids = [tok.encode(" " + n, add_special_tokens=False) for n in NODES]
    assert all(len(t) == 1 for t in node_ids), node_ids
    node_ids = [t[0] for t in node_ids]
    node_t = torch.tensor(node_ids, device=device)
    # Load config, code and weights from the pinned local snapshot: by repo id, 4.46 cannot import a module
    # named with a dot, and from_config would fetch the latest code rather than the pinned revision.
    local = Path(snapshot_download(args.model, revision=args.revision))
    config = AutoConfig.from_pretrained(local, trust_remote_code=True)
    # The checkpoint's config.json has rope_scaling null; transformers 5 fills in a default dict that the
    # 4.42-era remote code reads as a scaling type. Restore the checkpoint's value.
    config.rope_scaling = None
    model = load_on_gpu(local, config, device)
    factors = np.load(args.factors)
    hooks = Hooks(model, factors, device)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    import transformers
    report.setdefault("meta", {"transformers": transformers.__version__,
        "model": args.model, "revision": args.revision, "seed": args.seed,
        "factors_sha256": hashlib.sha256(args.factors.read_bytes()).hexdigest(),
        "num_loops": getattr(model.config, "num_loops", None), "prompt": "three-shot runbook format", "readout": "forced choice over 16 node tokens"})

    # Items the clean model answers correctly, per hop count; batched by equal token length (no padding).
    rng = np.random.default_rng(args.seed)
    items = {}
    for hops in args.hops:
        kept, tried = [], 0
        while len(kept) < args.per_hop and tried < args.max_candidates:
            chunk = [make_prompt(rng, hops) for _ in range(args.batch)]
            tried += len(chunk)
            for text, answer in chunk:
                ids = torch.tensor([tok.encode(text)], device=device)
                tgt = torch.tensor([node_ids[NODES.index(answer)]], device=device)
                _, ok = answer_logprobs(model, [ids], [tgt], node_t)
                if ok.item():
                    kept.append((tok.encode(text), node_ids[NODES.index(answer)]))
        items[hops] = kept[:args.per_hop]
        report["meta"][f"clean_accuracy_hops{hops}"] = [len(kept), tried]
        print(f"hops {hops}: kept {len(items[hops])}, clean correct {len(kept)}/{tried}", flush=True)
    assert hooks.calls == {11: 2, 21: 2}, f"expected two passes per layer, saw {hooks.calls}"

    def batched(seq):
        by_len = {}
        for ids, tgt in seq:
            by_len.setdefault(len(ids), []).append((ids, tgt))
        out_ids, out_tgt = [], []
        for group in by_len.values():
            for i in range(0, len(group), args.batch):
                part = group[i:i + args.batch]
                out_ids.append(torch.tensor([p[0] for p in part], device=device))
                out_tgt.append(torch.tensor([p[1] for p in part], device=device))
        return out_ids, out_tgt

    sets = {h: batched(items[h]) for h in args.hops}
    hooks.usage = {}
    clean = {h: answer_logprobs(model, *sets[h], node_t)[0] for h in args.hops}
    usage = {}
    for (layer, n), rows in hooks.usage.items():
        last = sum(r[0] for r in rows) / sum(r[2] for r in rows)
        allpos = sum(r[1] for r in rows) / sum(r[2] for r in rows)
        usage[f"L{layer + 1}_pass{n}"] = {"answer_position": last.tolist(), "all_positions": allpos.tolist()}
    hooks.usage = None
    report["usage"] = usage
    report["clean_mean_logprob"] = {str(h): float(clean[h].mean()) for h in args.hops}

    gen = torch.Generator(device="cpu").manual_seed(0)
    conditions = []
    for L in LAYERS:
        V, U = hooks.factors[L]
        for c in range(V.shape[1]):
            conditions.append((f"L{L + 1}_c{c}", L, V[:, c], U[c]))
        for r in range(args.random):
            c = int(torch.randint(V.shape[1], (1,), generator=gen))
            v = torch.randn(V.shape[0], generator=gen).to(V.device)
            u = torch.randn(U.shape[1], generator=gen).to(U.device)
            conditions.append((f"L{L + 1}_rand{r}", L, v / v.norm() * V[:, c].norm(), u / u.norm() * U[c].norm()))
    results = report.setdefault("ablation", {})
    started = time.monotonic()
    for k, (name, L, v, u) in enumerate(conditions):
        for pname, passes in PASSES.items():
            key = f"{name}_{pname}"
            if key in results:
                continue
            hooks.edit = (L, passes, v, u)
            entry = {}
            for h in args.hops:
                lp, ok = answer_logprobs(model, *sets[h], node_t)
                entry[f"hops{h}_dlogp"] = float((lp - clean[h]).mean())
                entry[f"hops{h}_acc"] = float(ok.float().mean())
            hooks.edit = None
            results[key] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{k + 1}/{len(conditions)} {name} {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
