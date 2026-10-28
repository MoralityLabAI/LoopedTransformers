"""Phase 2, post hoc (exploratory, not in the frozen plan): transcoders of the MLP sublayer only.

The registered `D-trans` reads `resid[t]` and predicts the whole block's update, which cannot work: 30 to 42%
of that update is attention, which reads other positions (`trans_diagnosis.json`). The usual transcoder
replaces only the MLP: it reads the MLP's input `norm2(alpha * resid + beta * attn_out)` and predicts
`mlp_out`, with attention left live. One transcoder serves every visit of a tied loop; an untied stack gets
one per layer, trained on one shared stream of states. Each is scored by calibration FVU on `mlp_out` and
by the replacement model (every MLP application replaced, attention live), as deep exact match on
`validation` relative to clean, as in N1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research_gym.interp.dictionaries import DictionaryConfig, DictionaryTrainer, save_dictionary
from research_gym.interp.hooks import record_components
from research_gym.interp.phase2 import core_examples, load_model, selection_half, slug, subset
from research_gym.interp.replacement import answer_scores
from research_gym.scripts.rmp_phase2_n1 import em_summary, validation_data


@torch.no_grad()
def mlp_pairs(model, tokens: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(MLP input, MLP output) at every position, one pair per visit."""

    with record_components(model.model) as records:
        model.run(tokens)
    pairs = []
    for visit, parts in enumerate(records):
        block = model.model.block_at(visit)
        middle = block.alpha * parts["resid_pre"] + block.beta * parts["attn_out"]
        z = block.norm2(middle)
        pairs.append((z.reshape(-1, z.shape[-1]).float(), parts["mlp_out"].reshape(-1, z.shape[-1]).float()))
    return pairs


class Replaced(torch.nn.Module):
    def __init__(self, dictionary) -> None:
        super().__init__()
        self.dictionary = dictionary

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape
        return self.dictionary(z.reshape(-1, shape[-1]))[0].reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[4096])
    parser.add_argument("--ks", type=int, nargs="+", default=[32])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--monitor-bundle", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/mlp_transcoders.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    model = load_model(args.model, args.runs, args.artifact_dir, device)
    tied = len(set(model.model.module_indices)) == 1
    visits = model.config.visits
    train = core_examples(args.monitor_bundle, "monitor_train")["tokens"]
    calibration = core_examples(args.monitor_bundle, "monitor_calibration")
    held_out = subset(calibration, selection_half(calibration["groups"]))["tokens"][:256].to(device)
    data = validation_data(args.bundle_dir)
    clean = em_summary(answer_scores(model, data["tokens"], data["targets"])[1], data)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    entry = report.setdefault(args.model, {"clean": clean, "tied": tied, "results": {}})

    for width in args.widths:
        for k in args.ks:
            name = f"mlp_w{width}_k{k}_s0"
            if name in entry["results"]:
                print("exists, skipped:", args.model, name, flush=True)
                continue
            sites = [-1] if tied else list(range(visits))
            configs = [DictionaryConfig(kind="trans", width=width, k=k, site=site, steps=args.steps) for site in sites]
            generator = torch.Generator().manual_seed(0)

            def batch() -> list[tuple[torch.Tensor, torch.Tensor]]:
                index = torch.randint(0, train.shape[0], (configs[0].batch_examples,), generator=generator)
                pairs = mlp_pairs(model, train[index].to(device))
                if tied:
                    return [(torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs]))]
                return pairs

            first = batch()
            trainers = [DictionaryTrainer(config, first[i], dim=model.config.hidden_size, device=device)
                        for i, config in enumerate(configs)]
            for step in range(args.steps):
                current = first if step == 0 else batch()
                for trainer, (x, y) in zip(trainers, current):
                    trainer.step(x, y)
            dictionaries = []
            for trainer in trainers:
                dictionary, summary = trainer.finish()
                dictionary.eval()
                dictionaries.append((dictionary, summary))
            held = mlp_pairs(model, held_out)
            fvu = []
            for visit, (x, y) in enumerate(held):
                dictionary = dictionaries[0 if tied else visit][0]
                with torch.no_grad():
                    error = (y - dictionary(x)[0]).pow(2).sum()
                fvu.append(float(error / (y - y.mean(dim=0)).pow(2).sum()))
            originals = {}
            for visit in range(visits):
                block = model.model.block_at(visit)
                if id(block) not in originals:
                    originals[id(block)] = (block, block.mlp)
                    block.mlp = Replaced(dictionaries[0 if tied else visit][0])
            try:
                correct = answer_scores(model, data["tokens"], data["targets"])[1]
            finally:
                for block, mlp in originals.values():
                    block.mlp = mlp
            paths = []
            for (dictionary, summary), config in zip(dictionaries, configs):
                path = args.work_dir / slug(args.model) / f"mlp_{config.name}.pt"
                paths.append({"path": str(path), "sha256": save_dictionary(dictionary, summary, path)})
            entry["results"][name] = {"width": width, "k": k, "fvu_by_visit": fvu, "mean_fvu": sum(fvu) / len(fvu),
                                      "replacement": em_summary(correct, data, clean), "dictionaries": paths}
            print(args.model, name, "mean fvu", round(entry["results"][name]["mean_fvu"], 4), "replacement deep ratio",
                  round(entry["results"][name]["replacement"]["deep_ratio"], 3), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
