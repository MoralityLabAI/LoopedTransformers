"""Phase 2, E5: fidelity (DEF) of small explanations of one visit's update, per docs/rmp_phase2_e5_e6_plan.md.

For each family and transition t -> t+1, at the answer position:
- the update is resampled from another example of the same family and depth;
- `m` components of the clean update are put back (`explained = resampled + P_E (delta - delta')`);
- the remaining visits run on.

DEF = (score(explained) - score(resampled)) / (score(clean) - score(resampled)), from mean log-probabilities
of the clean answer over the class's examples. The components come from four bases (features, rules, heads,
mixed); five controls are scored at the same m.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research_gym.interp.dictionaries import load_dictionary, rms
from research_gym.interp.hooks import record_components
from research_gym.interp.phase2 import load_model, select_config
from research_gym.scripts.rmp_phase2_n1 import DEEP, GRAPH_FAMILIES, validation_data

SIZES = (1, 2, 4, 8, 16, 32, 64)
BASES = ("features", "rules", "heads", "mixed")
CONTROLS = ("random_features", "pca", "neurons", "random", "orthogonal")
MIN_EFFECT = 0.1  # nats; below it resampling the update does not change the answer, and DEF is undefined


def orthonormal(directions: torch.Tensor) -> torch.Tensor:
    """(B, m, H) directions -> (B, H, r) orthonormal basis of their span (zero rows ignored)."""

    q, r = torch.linalg.qr(directions.transpose(1, 2))
    keep = (r.diagonal(dim1=1, dim2=2).abs() > 1e-6).unsqueeze(1)
    return q * keep


def project(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhr,br->bh", q, torch.einsum("bhr,bh->br", q, v))


@torch.no_grad()
def log_prob(model, tokens, targets, hook=None) -> float:
    logits = (model.run(tokens, hook) if hook is not None else model.run(tokens)).logits[:, -1].float()
    return float(torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).mean())


def at_answer(visit: int, new_delta):
    """Visit hook: at visit ``visit`` the answer position's update becomes ``new_delta(before, delta)``."""

    def hook(context):
        if context.visit != visit:
            return None
        after = context.after.clone()
        before = context.before[:, -1]
        after[:, -1] = before + new_delta(before, context.after[:, -1] - before)
        return after

    return hook


@torch.no_grad()
def run_class(model, shared, transcoders, tokens, targets, depths, visit: int, generator) -> dict:
    device = tokens.device
    with record_components(model.model) as records:
        model.run(tokens)
    parts = records[visit]
    block = model.model.block_at(visit)
    before = parts["resid_pre"][:, -1].float()
    clean_states = {}
    model.run(tokens, at_answer(visit, lambda b, d: clean_states.setdefault("delta", d.clone())))
    delta = clean_states["delta"].float()
    # Donor: another example of the same depth (a rotation within each depth group, fixed seed).
    donor = np.arange(len(depths))
    rng = np.random.default_rng(int(generator.initial_seed()))
    for depth in np.unique(depths):
        members = rng.permutation(np.flatnonzero(depths == depth))
        donor[members] = np.roll(members, 1)
    donor_delta = delta[torch.as_tensor(donor, device=device)]
    width = before.shape[-1]

    # Candidate directions and attributions per basis, each (B, n, H) and (B, n).
    after_clean, after_resampled = before + delta, before + donor_delta
    scale = rms(after_clean)
    codes_clean = shared.encode(after_clean / scale)
    codes_resampled = shared.encode(after_resampled / rms(after_resampled))
    feature_attr = (codes_clean - codes_resampled).abs() * scale
    middle = block.alpha * parts["resid_pre"][:, -1] + block.beta * parts["attn_out"][:, -1]
    z = block.norm2(middle).float()
    transcoder = transcoders[0] if len(transcoders) == 1 else transcoders[visit]
    rule_attr = transcoder.encode(z) * abs(float(block.beta))
    head_vectors = parts["head_out"][:, :, -1].float()  # (B, heads, H)
    head_attr = head_vectors.norm(dim=-1)
    hidden = block.mlp[1](block.mlp[0](z))
    neuron_directions = block.mlp[2].weight.T.float()  # (4H, H)

    def top(attr, directions_for, m):
        index = attr.topk(min(m, attr.shape[1]), dim=1).indices
        return directions_for(index)

    feature_dirs = lambda i: shared.decoder[i]  # noqa: E731
    rule_dirs = lambda i: transcoder.decoder[i]  # noqa: E731
    head_dirs = lambda i: head_vectors.gather(1, i[..., None].expand(-1, -1, width))  # noqa: E731
    mixed_attr = torch.cat([feature_attr, rule_attr, head_attr], dim=1)
    n_features, n_rules = shared.decoder.shape[0], transcoder.decoder.shape[0]

    def mixed_dirs(i):
        is_feature, is_rule = i < n_features, (i >= n_features) & (i < n_features + n_rules)
        features = shared.decoder[i.clamp(max=n_features - 1)]
        rules = transcoder.decoder[(i - n_features).clamp(0, n_rules - 1)]
        heads = head_dirs((i - n_features - n_rules).clamp(0, head_vectors.shape[1] - 1))
        return torch.where(is_feature[..., None], features, torch.where(is_rule[..., None], rules, heads))
    pcs = torch.linalg.svd(delta - delta.mean(dim=0), full_matrices=False).Vh  # (r, H)

    def explained(q):
        return log_prob(model, tokens, targets, at_answer(visit, lambda b, d: donor_delta + project(q, d - donor_delta)))

    def necessity(q):
        return log_prob(model, tokens, targets, at_answer(visit, lambda b, d: d - project(q, d)))

    clean = log_prob(model, tokens, targets)
    resampled = log_prob(model, tokens, targets, at_answer(visit, lambda b, d: donor_delta))
    effect = clean - resampled
    result = {"clean": clean, "resampled": resampled, "effect": effect, "def": {}, "necessity": {}}
    if effect < MIN_EFFECT:
        result["status"] = "no_effect"
        return result

    def score(q):
        return (explained(q) - resampled) / effect

    for m in SIZES:
        chosen = {
            "features": top(feature_attr, feature_dirs, m),
            "rules": top(rule_attr, rule_dirs, m),
            "heads": top(head_attr, head_dirs, m),
            "mixed": top(mixed_attr, mixed_dirs, m),
        }
        row = {}
        for basis, directions in chosen.items():
            q = orthonormal(directions.float())
            row[basis] = score(q)
            result["necessity"].setdefault(basis, {})[str(m)] = (clean - necessity(q)) / effect
            # Orthogonal control: random directions orthogonal to this basis's span, same count.
            noise = torch.randn(len(before), width, directions.shape[1], generator=generator).to(device)
            noise = noise - torch.einsum("bhr,brk->bhk", q, torch.einsum("bhr,bhk->brk", q, noise))
            row[f"orthogonal_{basis}"] = score(orthonormal(noise.transpose(1, 2)))
        picks = torch.stack([torch.randperm(shared.decoder.shape[0], generator=generator)[:m] for _ in range(len(before))])
        row["random_features"] = score(orthonormal(shared.decoder[picks.to(device)].float()))
        row["pca"] = score(orthonormal(pcs[:m].expand(len(before), -1, -1)))
        row["neurons"] = score(orthonormal(top(hidden.abs(), lambda i: neuron_directions[i], m)))
        row["random"] = score(orthonormal(torch.randn(len(before), m, width, generator=generator).to(device)))
        result["def"][str(m)] = row
    accepted = {}
    for basis in BASES:
        s90 = next((m for m in SIZES if result["def"][str(m)][basis] >= 0.9), None)
        controls = None
        if s90 is not None:
            row = result["def"][str(s90)]
            controls = {name: row[name] for name in ("random_features", "pca", "neurons", "random", f"orthogonal_{basis}")}
        accepted[basis] = {"s90": s90, "accepted": bool(s90 is not None and s90 <= 16
                                                       and all(result["def"][str(s90)][basis] > v for v in controls.values()))}
    result["s90"] = accepted
    result["status"] = "accepted" if any(a["accepted"] for a in accepted.values()) else "not_accepted"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--runs", type=Path, default=Path("experiments/recurrence_monitorability_v0/runs"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, default=Path("experiments/recurrence_monitorability_v0/datasets"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/dictionaries.json"))
    parser.add_argument("--transcoders", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/mlp_transcoders.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/recurrence_monitorability_v0/phase2/e5_def.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, args.runs, args.artifact_dir, device)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))[args.model]
    chosen, _ = select_config(metrics)
    shared = load_dictionary(Path(metrics[chosen]["path"]), device=device)[0]
    entry = json.loads(args.transcoders.read_text(encoding="utf-8"))[args.model]["results"]["mlp_w4096_k32_s0"]
    transcoders = [load_dictionary(Path(d["path"]), device=device)[0] for d in entry["dictionaries"]]
    data = validation_data(args.bundle_dir)
    report = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    out = report.setdefault(args.model, {"dictionary": chosen, "transcoder": "mlp_w4096_k32_s0", "classes": {}})
    for family in GRAPH_FAMILIES:
        rows = np.flatnonzero((data["families"] == family) & np.isin(data["depths"], DEEP))
        rows = np.sort(np.random.default_rng(0).permutation(rows)[: args.examples])
        tokens = data["tokens"][rows].to(device)
        targets = data["targets"][rows].to(device)
        for visit in range(model.config.visits):
            key = f"{family}/t{visit}"
            if key in out["classes"]:
                continue
            generator = torch.Generator().manual_seed(visit)
            result = run_class(model, shared, transcoders, tokens, targets, data["depths"][rows], visit, generator)
            out["classes"][key] = result
            print(args.model, key, result["status"], round(result["effect"], 3),
                  {b: v["s90"] for b, v in result.get("s90", {}).items()}, flush=True)
            args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
