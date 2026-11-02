"""Huginn-0125 accuracy by hop count and recurrence (exploratory, local, not a registered endpoint).

Huginn runs a 4-layer core block `r` times between a 2-layer prelude and a 2-layer coda (`num_steps=r`;
trained around a mean of 32). This sweep asks whether more iterations buy more hops on the pointer chase,
which decides the depths and `r` a per-iteration heat map can use.

- **Tasks.** Zero-shot with a one-line instruction, or with three solved demos (1–3 hops). The same items
  are scored at every `r`.
  - `varchain` (default): a short program of variable copies. The target chain starts from a digit constant
    and copies it `hops` times; two distractor chains of the same length with other digits are interleaved
    line by line, keeping each chain in order. The program ends with `print(<last target variable>)`.
  - `chase`: the Nanbeige runbook format (`rmp_nanbeige_vpd_heatmap.chase`) with spaced arrows
    (`A -> B`). Huginn's tokenizer fuses `>B` into one token in `A->B`, so edge targets never appear as the
    ` B` tokens the readout scores. Three-shot one-hop accuracy was at chance at every `r` with `A->B`
    (`depth_sweep_fused_arrow.json`) and 17–25% for r ≥ 8 with spaces, at chance from two hops
    (`depth_sweep_chase_spaced.json`), hence `varchain`.
- **Readout.** Forced choice at the answer position among the 10 digit tokens (`varchain`, chance 1/10) or
  the 16 node tokens (`chase`, chance 1/16).
- **Determinism.** The recurrent state starts from noise; the generator is reseeded before every forward.

A load check (greedy continuation of a fixed sentence) is printed first. Results are written after every
cell, so a stopped run resumes where it left off.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, PreTrainedTokenizerFast

from research_gym.scripts.rmp_nanbeige_vpd_heatmap import NODES, chase, load_on_gpu, node_mask

DIGITS = [str(d) for d in range(10)]
INSTRUCTIONS = {
    "chase": "Follow the arrows from Start for the given number of hops and give the node you reach.\n\n",
    "varchain": "# Each program prints one digit.\n\n",
}


def varchain(rng, hops, chains=3):
    names = rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=chains * (hops + 1), replace=False)
    digits = rng.choice(10, size=chains, replace=False)
    lines = [[f"{names[c * (hops + 1)]} = {digits[c]}"] + [
        f"{names[c * (hops + 1) + i]} = {names[c * (hops + 1) + i - 1]}" for i in range(1, hops + 1)]
        for c in range(chains)]
    order = rng.permutation(np.repeat(np.arange(chains), hops + 1))
    pos = [0] * chains
    program = []
    for c in order:
        program.append(lines[c][pos[c]])
        pos[c] += 1
    return "\n".join(program) + f"\nprint({names[hops]})\nOutput:", str(digits[0])


def make_item(rng, hops, shots, task="varchain"):
    gen = varchain if task == "varchain" else chase
    demos = []
    for _ in range(shots):
        text, answer = gen(rng, int(rng.integers(1, 4)))
        demos.append(f"{text} {answer}")
    text, answer = gen(rng, hops)
    text = INSTRUCTIONS[task] + "\n\n".join(demos + [text])
    return (text.replace("->", " -> ") if task == "chase" else text), answer


def load_huginn(local, device):
    config = AutoConfig.from_pretrained(local, trust_remote_code=True)
    model = load_on_gpu(local, config, device)
    # The rotary table is applied in fp32 by the model code; keep the checkpoint's fp32 values rather than the
    # bf16 copy made by the loader.
    index = json.loads((local / "model.safetensors.index.json").read_text())
    with open(local / index["weight_map"]["freqs_cis"], "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        spec = json.loads(f.read(size))["freqs_cis"]
        f.seek(8 + size + spec["data_offsets"][0])
        data = bytearray(f.read(spec["data_offsets"][1] - spec["data_offsets"][0]))
    model.freqs_cis = torch.frombuffer(data, dtype=torch.float32).reshape(spec["shape"]).to(device)
    return model


@torch.no_grad()
def score(model, batches, r, node_t, seed):
    lps, correct = [], []
    for ids, tgt in batches:
        torch.manual_seed(seed)
        logits = model(input_ids=ids, num_steps=r, use_cache=False).logits[:, -1].float()
        logits = logits.masked_fill(~node_mask(logits.shape[1], node_t), float("-inf"))
        lp = torch.log_softmax(logits, -1)
        lps.append(lp.gather(1, tgt[:, None])[:, 0].cpu())
        correct.append((logits.argmax(-1) == tgt).cpu())
    return torch.cat(lps), torch.cat(correct)


def answer_tokens(tok, task):
    """Candidate answer labels and their single-token ids (with the leading space the prompt implies)."""
    labels = DIGITS if task == "varchain" else NODES
    ids = [tok.encode(" " + a, add_special_tokens=False) for a in labels]
    assert all(len(t) == 1 for t in ids), ids
    return labels, [t[0] for t in ids]


@torch.no_grad()
def greedy(model, tok, text, r, n, device):
    ids = torch.tensor([tok.encode(text)], device=device)
    for _ in range(n):
        torch.manual_seed(0)
        nxt = model(input_ids=ids, num_steps=r, use_cache=False).logits[:, -1].argmax(-1)
        ids = torch.cat([ids, nxt[:, None]], 1)
    return tok.decode(ids[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--revision", default="bb6621b65e90b6a4b9b29ef88dc83866d450470c")
    parser.add_argument("--out", type=Path, default=Path(
        "experiments/recurrence_monitorability_v0/huginn_depth/depth_sweep.json"))
    parser.add_argument("--task", choices=["varchain", "chase"], default="varchain")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--per-cell", type=int, default=96)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    device = torch.device("cuda")
    local = Path(snapshot_download(args.model, revision=args.revision))
    # No protobuf/sentencepiece in the 4.46 venv; tokenizer.json alone gives the model's ids (BOS 65504 added).
    tok = PreTrainedTokenizerFast(tokenizer_file=str(local / "tokenizer.json"),
                                  bos_token="<|begin_text|>", add_bos_token=True)
    labels, cand_ids = answer_tokens(tok, args.task)
    cand_t = torch.tensor(cand_ids, device=device)
    model = load_huginn(local, device)

    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    import transformers
    if "meta" not in report:
        report["meta"] = {"transformers": transformers.__version__, "model": args.model,
                          "revision": args.revision, "seed": args.seed, "per_cell": args.per_cell,
                          "task": args.task, "readout": f"forced choice over {len(labels)} answer tokens",
                          "instruction": INSTRUCTIONS[args.task].strip()}
        for r in (4, 32):
            report["meta"][f"load_check_r{r}"] = greedy(model, tok, "The capital of France is", r, 8, device)
            print("load check", r, repr(report["meta"][f"load_check_r{r}"]), flush=True)
    report.setdefault("cells", {})

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for shots in args.shots:
        for hops in args.hops:
            items = [make_item(rng, hops, shots, args.task) for _ in range(args.per_cell)]
            by_len = {}
            for text, answer in items:
                ids = tok.encode(text)
                assert ids[0] == 65504, ids[:3]
                by_len.setdefault(len(ids), []).append((ids, cand_ids[labels.index(answer)]))
            batches = [(torch.tensor([p[0] for p in g[i:i + args.batch]], device=device),
                        torch.tensor([p[1] for p in g[i:i + args.batch]], device=device))
                       for g in by_len.values() for i in range(0, len(g), args.batch)]
            for r in args.steps:
                key = f"shots{shots}_hops{hops}_r{r}"
                if key in report["cells"]:
                    continue
                lp, ok = score(model, batches, r, cand_t, args.seed)
                report["cells"][key] = {"acc": float(ok.float().mean()), "mean_logprob": float(lp.mean()),
                                        "n": int(ok.numel()), "mean_tokens": float(np.mean([len(tok.encode(t)) for t, _ in items]))}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
                print(f"{key} acc {report['cells'][key]['acc']:.3f} lp {report['cells'][key]['mean_logprob']:.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
