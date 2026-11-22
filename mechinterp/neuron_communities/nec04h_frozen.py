"""Fit source-only phase and selector heads on frozen NEC-04F activations."""

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from math import fsum
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mechinterp.neuron_communities.nec04f_native import AuxLooper, CHUNK
from mechinterp.neuron_communities.nec04f_oracles import ROOTS, SOURCE, STAGES, seed
from mechinterp.neuron_communities.nec04h_archive import archived_root, file_sha, read_root, write_json
from mechinterp.neuron_communities.nec04h_oracles import (
    arrival_stratum, fifth_walsh, scalar_route, shuffled_phases, verify_shuffle,
)
from research.pef07.pef07 import as_tensor, model_digest


STEPS = (240, 900)
BATCH = 64
LR = .003
ARCHIVE_ATOL = 1e-5


def binary_log_odds(logit):
    return torch.stack((F.logsigmoid(-logit), F.logsigmoid(logit)), dim=-1)


def probabilistic_xor(bits, phase):
    return torch.logaddexp(bits + phase[..., :1], bits.flip((-1,)) + phase[..., 1:])


def probabilistic_mux(parity, native, selector):
    return torch.logaddexp(parity + selector[..., 1:], native + selector[..., :1])


def compose(xor, phase, selector, native):
    parity = probabilistic_xor(xor, phase)
    route = probabilistic_mux(parity, native, selector)
    inputs = [part.exp().tolist() for part in (xor, phase, selector, native)]
    independent = route.new_tensor([scalar_route(*row) for row in zip(*inputs, strict=True)])
    torch.testing.assert_close(route.exp(), independent, rtol=0, atol=1e-6)
    return route, parity


def verify_frozen(model, expected):
    if model_digest(model) != expected or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("the pretrained model weights or gradient state changed")


def capture(model, rows, visits, observation):
    histories = as_tensor([r["history_bits12"] for r in rows])
    memory, final, aux = [], [], []
    with torch.no_grad(), model.core.head.register_forward_pre_hook(
            lambda _module, args: memory.append(args[0].detach().clone())):
        for batch in histories.split(CHUNK):
            logits, intermediate = model(batch, visits)
            final.append(logits)
            aux.append(intermediate)
    result = {"memory": torch.cat(memory), "native": torch.cat(final), "intermediate": torch.cat(aux)}
    error = {}
    for key, archive_key in (("native", "all_final_native_logits"),
                             ("intermediate", "all_intermediate_auxiliary_logits")):
        expected = result[key].new_tensor(observation[archive_key])
        torch.testing.assert_close(result[key], expected, rtol=0, atol=ARCHIVE_ATOL)
        error[key] = float((result[key] - expected).abs().max())
    if result["memory"].shape != (len(rows), 32) or not result["memory"].isfinite().all():
        raise ValueError("frozen final-memory capture incomplete")
    return {"memory": result["memory"], "native": result["native"],
            "xor": result["intermediate"][:, -1].contiguous()}, error


def seal_features(features, root_dir):
    sealed, hashes = {}, {}
    for name, part in (("source", slice(None, SOURCE)), ("owned", slice(SOURCE, None))):
        path = root_dir / f"{name}.pt"
        torch.save({key: value[part].clone() for key, value in features.items()}, path)
        hashes[name] = file_sha(path)
        blob = path.read_bytes()
        if sha256(blob).hexdigest() != hashes[name]:
            raise ValueError("cached frozen features changed")
        sealed[name] = torch.load(BytesIO(blob), weights_only=True, map_location="cpu")
    return sealed, hashes


def train_one(head, features, target, batches, root_dir, arm):
    initial = model_digest(head)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0)
    checkpoints = {}
    for step, indices in enumerate(batches, start=1):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(features[indices]).squeeze(-1), target[indices]).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if step in STEPS:
            if any(int(opt.state[p]["step"].item()) != step for p in head.parameters()):
                raise ValueError("checkpoint AdamW counter differs from its declared budget")
            path = root_dir / f"{arm}-{step}.pt"
            torch.save(head.state_dict(), path)
            checkpoints[step] = {"sha256": file_sha(path), "model_sha256": model_digest(head),
                                 "sampler_sha256": sha256(batches[:step].numpy().tobytes()).hexdigest(),
                                 "optimizer_steps": step}
    return {"initial_sha256": initial, "checkpoints": checkpoints}


def fit_heads(root, context_groups, labels, memory, root_dir):
    if memory.shape != (SOURCE, 32) or len(context_groups) != SOURCE or len(labels) != SOURCE:
        raise ValueError("head fits require original source memory and labels only")
    torch.manual_seed(seed(root) ^ 0x41D350)
    phase = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
    heads = {"selector": nn.Linear(32, 1), "phase": deepcopy(phase), "shuffled": deepcopy(phase)}
    tagged = [(group, *label) for group, label in zip(context_groups, labels, strict=True)]
    groups = {group: (selector, phase_bit) for group, selector, phase_bit in tagged}
    permutation = shuffled_phases(root, groups)
    control = [permutation[group] for group, _, _ in tagged]
    verify_shuffle(tagged, control)
    targets = {"selector": [sel for sel, _ in labels], "phase": [bit for _, bit in labels], "shuffled": control}
    generator = torch.Generator(device="cpu").manual_seed(seed(root) ^ 0x5DEECE66D)
    batches = torch.randint(SOURCE, (STEPS[-1], BATCH), generator=generator)
    receipts = {name: train_one(head, memory, memory.new_tensor(targets[name]), batches, root_dir, name)
                for name, head in heads.items()}
    if receipts["phase"]["initial_sha256"] != receipts["shuffled"]["initial_sha256"] or any(
            receipts["phase"]["checkpoints"][step]["sampler_sha256"]
            != receipts["shuffled"]["checkpoints"][step]["sampler_sha256"] for step in STEPS):
        raise ValueError("phase arms diverged in initialization or sampling")
    return heads, receipts, permutation


def scores(log_probabilities, rows):
    predictions = log_probabilities.argmax(-1).tolist()
    groups = defaultdict(list)
    counts = {name: {"n": 0, "correct": 0} for name in ("full", "parity", "alternate")}
    for i, (prediction, row) in enumerate(zip(predictions, rows, strict=True)):
        bucket = "parity" if row["mux_selector"] else "alternate"
        hit = int(prediction == row["composed_label"])
        for name in ("full", bucket):
            counts[name]["n"] += 1
            counts[name]["correct"] += hit
        if row["mux_selector"]:
            groups[row["context_group"]].append(i)
    walsh = []
    for groups_rows in groups.values():
        indices = sorted(groups_rows, key=lambda i: rows[i]["focal_mask"])
        walsh.append(fifth_walsh([(float(log_probabilities[i, 1] - log_probabilities[i, 0])) for i in indices],
                                 rows[indices[0]]["composed_label"]))
    return {"accuracy": {k: v["correct"] / v["n"] for k, v in counts.items()},
            "counts": counts, "walsh": walsh, "predictions": predictions}


@torch.no_grad()
def evaluate_heads(heads, receipts, root_dir, features, rows, labels):
    native = features["native"].log_softmax(-1)
    xor = features["xor"].log_softmax(-1)
    memory = features["memory"]
    truth = torch.tensor(labels, dtype=torch.long)
    truth_log = F.one_hot(truth, 2).to(native.dtype).log()
    outputs = {}
    for step in STEPS:
        logits = {}
        for name, model in heads.items():
            receipt = receipts[name]["checkpoints"][step]
            blob = (root_dir / f"{name}-{step}.pt").read_bytes()
            if sha256(blob).hexdigest() != receipt["sha256"]:
                raise ValueError("sealed head checkpoint changed")
            model.load_state_dict(torch.load(BytesIO(blob), map_location="cpu", weights_only=True), strict=True)
            if model_digest(model) != receipt["model_sha256"]:
                raise ValueError("restored head weights changed")
            logits[name] = model.eval()(memory).squeeze(-1)
        sel = binary_log_odds(logits["selector"])
        phase = binary_log_odds(logits["phase"])
        arms = {"native": native}
        for name, offset, route in (
            ("learned", phase, sel), ("shuffled_phase", binary_log_odds(logits["shuffled"]), sel),
            ("oracle_phase", truth_log[:, 1], sel), ("oracle_selector", phase, truth_log[:, 0]),
            ("oracle_phase_and_selector", truth_log[:, 1], truth_log[:, 0]),
        ):
            arms[name], _ = compose(xor, offset, route, native)
        outputs[step] = {**{name: scores(value, rows) for name, value in arms.items()},
                         "phase_accuracy": float(((logits["phase"] > 0) == truth[:, 1]).double().mean()),
                         "shuffled_phase_accuracy": float(((logits["shuffled"] > 0) == truth[:, 1]).double().mean()),
                         "selector_accuracy": float(((logits["selector"] > 0) == truth[:, 0]).double().mean()),
                         "shared_selector_sha256": receipts["selector"]["checkpoints"][step]["sha256"]}
    return outputs


def run_root(inputs, out, root):
    world, ref, rows, labels, blob, observation = read_root(inputs, root)
    root_dir = out / f"root-{root}"
    root_dir.mkdir()
    archived = archived_root(root, world, ref, rows, labels, observation)
    state = torch.load(BytesIO(blob), map_location="cpu", weights_only=True)
    model = AuxLooper().eval().requires_grad_(False)
    model.load_state_dict(state["model_state"], strict=True)
    before = observation["original_model_sha256"]
    verify_frozen(model, before)
    visits = tuple(ref["stage_visits"][str(k)] for k in STAGES)
    features, error = capture(model, rows, visits, observation)
    sealed, hashes = seal_features(features, root_dir)
    groups = [row["context_group"] for row in rows[:SOURCE]]
    heads, receipts, permutation = fit_heads(root, groups, labels[:SOURCE], sealed["source"]["memory"], root_dir)
    verify_frozen(model, before)
    result = {"root": root, "arrival": arrival_stratum(world), "native_model_sha256": before,
              "native_replay_max_abs_error": error, "feature_sha256": hashes,
              "context_phase_control": permutation, "head_receipts": receipts, "archived": archived,
              "steps": evaluate_heads(heads, receipts, root_dir, sealed["owned"], rows[SOURCE:], labels[SOURCE:]),
              "pretrained_updates": 0}
    write_json(root_dir / "result.json", result)
    return result


def aggregate(roots):
    groups = defaultdict(list)
    for root in roots:
        groups[root["arrival"]].append(root)
    result = {}
    for step in STEPS:
        def summarize(rows):
            return {name: {key: fsum(row["steps"][step][name]["accuracy"][key] for row in rows) / len(rows)
                           for key in ("full", "parity", "alternate")}
                    for name in ("native", "learned", "shuffled_phase", "oracle_phase", "oracle_selector",
                                 "oracle_phase_and_selector")}

        result[step] = {"macro": summarize(roots), "by_arrival": {name: summarize(rows) for name, rows in groups.items()},
                        "arrival_root_counts": {name: len(rows) for name, rows in groups.items()}}
    return result


def main():
    from mechinterp.neuron_communities.nec04h_archive import run_archived

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir()
    write_json(args.output / "phase-a.json", run_archived(args.inputs))
    roots = []
    for root in ROOTS:
        roots.append(run_root(args.inputs, args.output, root))
        print(f"NEC-04H verified root {root}", flush=True)
    write_json(args.output / "report.json", {"status": "REDESIGN_OR_PILOT_ONLY",
                                              "protocol_sha256": file_sha(Path(__file__).with_name("nec04h_protocol.md")),
                                              "torch": torch.__version__, "roots": roots, "summary": aggregate(roots)})


if __name__ == "__main__":
    main()
