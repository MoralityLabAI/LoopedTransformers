"""PEF-08: probe-selected four-corner erasure on frozen PEF-07 models.

The exact intervention and comparator design is registered in PROTOCOL.md.
Run against the existing PEF-07 *local* six-checkpoint evidence archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from research.pef07 import pef07 as p7

SEEDS = (103, 131, 167)
TASKS = ("coalition", "threshold")
PAIRS = ((0, 1), (2, 3))
LOOPS = (4, 8, 12)
CHANNELS = 8
RANDOM_CONTROLS = 8
CONTINUATION = (0, 1)


def source_data():
    if hasattr(p7, "source_split"):
        raw, manifest = p7.source_split()
        t = p7.as_tensor
        signature = manifest["sha256"]
    else:
        raw, manifest = p7.split_histories()
        t = p7.to_tensor
        signature = manifest["manifest_sha256"]
    return {k: t(v) for k, v in raw.items()}, signature, manifest


def labels(bits, task):
    f = p7.target if hasattr(p7, "target") else p7.targets
    return f(bits, task)


def weights_digest(model):
    f = p7.model_digest if hasattr(p7, "model_digest") else p7.tensor_hash
    return f(model)


def checkpoint_rows(ledger):
    return ledger.get("checkpoints", ledger.get("training", []))


def checked_model(folder, rec):
    filename = rec.get("file", rec.get("checkpoint_file"))
    checkpoint = folder / filename
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != rec["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint bytes differ: {filename}")
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if (payload["seed"], payload["task"], payload["steps"]) != (
        rec["seed"], rec["task"], 450
    ):
        raise AssertionError("checkpoint seed, task or registered steps differ")
    cls = p7.Looper if hasattr(p7, "Looper") else p7.CoalitionMemoryLooper
    model = cls().eval()
    model.load_state_dict(payload["model"], strict=True)
    target_digest = rec.get("frozen_weights_sha256", rec.get("final_weights_sha256"))
    if weights_digest(model) != target_digest:
        raise AssertionError(f"state_dict digest differs: {filename}")
    return model


@torch.no_grad()
def states_by_loop(model, bits):
    """h_0,...,h_14; all steps are natural source or continuation bits."""
    h = model.memory_init.unsqueeze(0).expand(len(bits), -1)
    states = [h]
    for k in range(12):
        h = model.step(h, model.bit_embedding(bits[:, k]))
        states.append(h)
    for b in CONTINUATION:
        bit = torch.full((len(bits),), b, dtype=torch.long, device=bits.device)
        h = model.step(h, model.bit_embedding(bit))
        states.append(h)
    if not torch.equal(states[12][:32], model.memory(bits[:32])):
        raise AssertionError("manual loop trace disagrees with original natural forward")
    return states


def group_delta(state, pair):
    """Each cohort has 16 old-bit counterfactuals ordered by mask 0..15."""
    if len(state) % 16:
        raise ValueError("incomplete 16-cell context group")
    cube = state.reshape(-1, 16, state.shape[-1])
    a, b = (1 << pair[0]), (1 << pair[1])
    return cube[:, a | b] - cube[:, a] - cube[:, b] + cube[:, 0]


def pair_margin(score, pair):
    scalar = score.reshape(-1, 16)
    a, b = (1 << pair[0]), (1 << pair[1])
    delta = scalar[:, a | b] - scalar[:, a] - scalar[:, b] + scalar[:, 0]
    return {"mean_signed": float(delta.mean()), "mean_abs": float(delta.abs().mean())}


def native_margin(model, state):
    z = model.head(state)
    return z[:, 1] - z[:, 0]


def ridge_fit(model, probe_states, probe_labels):
    return p7.fit_ridge(probe_states, probe_labels)


def ridge_margin(state, fit):
    avg, std, beta = fit
    x = (state.double() - avg) / std
    return x @ beta[:-1] + beta[-1] - 0.5


def accuracy(margin, truth):
    return int(((margin >= 0).long() == truth).sum())


def choose_channels(probe_state, pair, w, seed, task, loop):
    signal = group_delta(probe_state, pair).abs().mean(0) * w.abs()
    ranked = sorted(range(len(w)), key=lambda c: (-float(signal[c]), c))
    top = tuple(ranked[:CHANNELS])
    bottom = tuple(ranked[-CHANNELS:])
    rng = random.Random(8008 + 1000*seed + 10*loop +
                        2*PAIRS.index(pair) + (100000 if task == "threshold" else 0))
    random_masks = [tuple(sorted(rng.sample(range(len(w)), CHANNELS)))
                    for _ in range(RANDOM_CONTROLS)]
    return top, bottom, random_masks


def inject_pair_erasure(state, pair, selected):
    """Subtract context-specific source D_ij in selected channels on double-on rows."""
    if not selected:
        return state.clone()
    n, d = state.shape
    cube = state.reshape(-1, 16, d)
    delta = group_delta(state, pair)
    gate = torch.tensor([int((mask & ((1 << pair[0]) | (1 << pair[1]))) ==
                              ((1 << pair[0]) | (1 << pair[1]))) for mask in range(16)],
                        dtype=state.dtype, device=state.device)
    channel_mask = torch.zeros(d, dtype=state.dtype, device=state.device)
    channel_mask[list(selected)] = 1
    return (cube - delta[:, None, :] * gate[None, :, None] *
            channel_mask[None, None, :]).reshape(n, d)


@torch.no_grad()
def replay(model, bits, state, start, end):
    """Resume after visit `start` using the original remaining input suffix."""
    if not (0 <= start <= end <= 14):
        raise ValueError("invalid replay horizon")
    h = state
    for visit in range(start, end):
        if visit < 12:
            z = bits[:, visit]
        else:
            z = torch.full((len(bits),), CONTINUATION[visit - 12],
                           dtype=torch.long, device=bits.device)
        h = model.step(h, model.bit_embedding(z))
    return h


@torch.no_grad()
def measurement(model, bits, states, pair, selected, loop, truth, fit):
    changed = inject_pair_erasure(states[loop], pair, selected)
    final = replay(model, bits, changed, loop, 12)
    score = native_margin(model, final)
    base = native_margin(model, states[12])
    after = pair_margin(score, pair)
    before = pair_margin(base, pair)
    attenuation = 1 - after["mean_abs"] / max(before["mean_abs"], 1e-12)
    wrong = ((score >= 0).long() != truth).reshape(-1, 16)
    out = {
        "errors_by_focal_mask": [int(v) for v in wrong.sum(0)],
        "selected_channels": list(selected), "loop": loop,
        "baseline_native_correct": accuracy(base, truth),
        "patched_native_correct": accuracy(score, truth),
        "baseline_pair": before, "patched_pair": after,
        "relative_abs_interaction_attenuation": attenuation,
        "mean_abs_native_margin_disturbance": float((score-base).abs().mean()),
    }
    if loop == 12:
        later = replay(model, bits, changed, 12, 14)
        h14 = states[14]
        late_base, late_patched = ridge_margin(h14, fit), ridge_margin(later, fit)
        late_wrong = ((late_patched >= 0).long() != truth).reshape(-1, 16)
        out["h14_fixed_ridge"] = {
            "errors_by_focal_mask": [int(v) for v in late_wrong.sum(0)],
            "baseline_correct": accuracy(late_base, truth),
            "patched_correct": accuracy(late_patched, truth),
            "baseline_pair": pair_margin(late_base, pair),
            "patched_pair": pair_margin(late_patched, pair),
        }
    return out


@torch.no_grad()
def loop_diagnostics(model, states, fit):
    result = []
    for i, state in enumerate(states):
        score = (ridge_margin(state, fit) if i == 14
                 else native_margin(model, state))
        result.append({
            "loop": i, "readout": "h14_frozen_ridge" if i == 14 else "h12_native_probe",
            "pairs": [
                {"pair": list(pair), "hidden_mixed_l2": float(
                    group_delta(state, pair).norm(dim=1).mean()),
                 **pair_margin(score, pair)}
                for pair in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
            ],
        })
    return result


@torch.no_grad()
def evaluate(folder, rec, catalog):
    model = checked_model(folder, rec)
    weight_before = weights_digest(model)
    chk = folder / rec.get("file", rec.get("checkpoint_file"))
    bytes_before = hashlib.sha256(chk.read_bytes()).hexdigest()
    task, seed = rec["task"], rec["seed"]
    probe, held = catalog["probe"], catalog["heldout"]
    truth = labels(held, task)
    probe_states = states_by_loop(model, probe)
    held_states = states_by_loop(model, held)
    fit = ridge_fit(model, probe_states[14], labels(probe, task))
    native = native_margin(model, held_states[12])
    h14 = ridge_margin(held_states[14], fit)
    if accuracy(native, truth) != 768 or accuracy(h14, truth) != 768:
        raise AssertionError("original PEF-07 trained/fitted 768/768 scores not reproduced")
    margin_weight = (model.head.weight[1]-model.head.weight[0]).detach()
    interventions = []
    for pair in PAIRS:
        for loop in LOOPS:
            top, bottom, samples = choose_channels(
                probe_states[loop], pair, margin_weight, seed, task, loop)
            top_run = measurement(model, held, held_states, pair, top, loop, truth, fit)
            bottom_run = measurement(model, held, held_states, pair, bottom, loop, truth, fit)
            random_runs = [measurement(model, held, held_states, pair, sel,
                                       loop, truth, fit) for sel in samples]
            avg_random = sum(r["relative_abs_interaction_attenuation"]
                             for r in random_runs)/len(random_runs)
            interventions.append({
                "pair": list(pair), "loop": loop, "top8": top_run,
                "bottom8": bottom_run, "random8": random_runs,
                "random_mean_attenuation": avg_random,
                "top_minus_random_attenuation":
                    top_run["relative_abs_interaction_attenuation"]-avg_random,
            })
    if weights_digest(model) != weight_before or hashlib.sha256(chk.read_bytes()).hexdigest() != bytes_before:
        raise AssertionError("frozen parameters or checkpoint modified during intervention")
    return {
        "seed": seed, "task": task, "checkpoint_sha256": bytes_before,
        "weight_sha256": weight_before,
        "baseline_h12": accuracy(native, truth),
        "baseline_h14_frozen_ridge": accuracy(h14, truth),
        "readout_digest": hashlib.sha256(b"".join(
            bytes(x.contiguous().reshape(-1).view(torch.uint8).tolist()) for x in fit
        )).hexdigest(),
        "natural_loop_diagnostics": loop_diagnostics(model, held_states, fit),
        "interventions": interventions,
        "weight_and_checkpoint_unchanged": True,
    }


def run(evidence, output):
    os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    evidence, output = Path(evidence), Path(output)
    ledger = json.loads((evidence / "sealed_training_ledger.json").read_text())
    catalog, generated_signature, generated_manifest = source_data()
    expected_signature = ledger.get("source_sha256", ledger.get("source_split_sha256"))
    original_manifest = json.loads((evidence / "source_split.json").read_text())
    sealed_signature = original_manifest.get("sha256", original_manifest.get("manifest_sha256"))
    canonical = p7.encoded if hasattr(p7, "encoded") else p7.canonical
    rebuilt = {k:v for k,v in original_manifest.items()
               if k not in ("sha256", "manifest_sha256")}
    if (sealed_signature != expected_signature or
        hashlib.sha256(canonical(rebuilt)).hexdigest() != sealed_signature):
        raise AssertionError("original archived source manifest differs from checkpoint ledger")
    archive_contexts = original_manifest.get(
        "contexts_in_sha256_order", original_manifest.get("sorted_contexts"))
    generated_contexts = generated_manifest.get(
        "contexts_in_sha256_order", generated_manifest.get("sorted_contexts"))
    archive_owners = original_manifest.get(
        "context_assignment", original_manifest.get("split_assignment"))
    generated_owners = generated_manifest.get(
        "context_assignment", generated_manifest.get("split_assignment"))
    if archive_contexts != generated_contexts or archive_owners != generated_owners:
        raise AssertionError("input ownership changed across PEF-07 source implementations")
    if not ledger.get("all_six_trunks_sealed_before_scoring",
                      ledger.get("all_checkpoints_frozen_before_heldout", False)):
        raise AssertionError("PEF-07 training freeze not recorded")
    recs = checkpoint_rows(ledger)
    if {(r["seed"],r["task"]) for r in recs} != {
        (s,t) for s in SEEDS for t in TASKS}:
        raise AssertionError("expected six PEF-07 task/seed checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(evidence, rec, catalog) for rec in recs]
    decisions = []
    for s in SEEDS:
        coal = next(r for r in rows if (r["seed"],r["task"]) == (s,"coalition"))
        pair_out = [r for r in coal["interventions"] if r["loop"] == 12]
        decisions.append({"seed": s,
            "both_pairs_top8_exceed_random_mean": all(
                r["top_minus_random_attenuation"] > 0 for r in pair_out),
            "per_pair_top_minus_random": [r["top_minus_random_attenuation"]
                                           for r in pair_out]})
    report = {
        "campaign": "PEF-08", "protocol": "research/pef08/PROTOCOL.md",
        "parent_pef07_head": "d9c2087cbe37f4787e3e56ef9d64ab2eb148f214",
        "registered_protocol_commit": "1dd57e6427c15cd512ffc113b1b986f2df78b6a7",
        "source_sha256": expected_signature,
        "generated_source_manifest_sha256": generated_signature,
        "seed_decisions": decisions,
        "two_of_three_descriptive_count": sum(d["both_pairs_top8_exceed_random_mean"]
                                               for d in decisions),
        "torch_version": str(torch.__version__),
        "all_checkpoints_frozen": True,
        "models": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"campaign":"PEF-08", "seed_decisions": decisions,
                      "descriptive_count": report["two_of_three_descriptive_count"],
                      "original_scores": [[r["seed"],r["task"],r["baseline_h12"],
                                           r["baseline_h14_frozen_ridge"]] for r in rows]},
                     indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.evidence, args.out)


if __name__ == "__main__":
    main()
