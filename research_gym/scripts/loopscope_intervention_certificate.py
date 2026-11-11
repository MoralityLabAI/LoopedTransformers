"""MC-02: predeclare a graph-conditioned continuation, transplant a cursor, audit it.

Scientific use requires a pinned existing fixed-depth LoopedChase checkpoint.
--smoke constructs an untrained network and emits no scientific aggregate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from research_gym.analysis.intervention_certificate import (
    Counterfactual, exact_controls, select_pairs, score_continuation, validate_manifest,
)
from research_gym.analysis.mechanistic_certificate import digest, integer, oracle
from research_gym.scripts.mechanistic_certificate import file_hash

ROOT = Path(__file__).resolve().parents[2]


def native_dependencies():
    import torch
    from research_gym.scripts import loopscope_semantic_echo as native
    return torch, native


def make_interventions(base, donor, *, seed: int, random_controls: int):
    """Only cursor_transfer claims a semantic lift; every comparator is explicit."""
    torch, _ = native_dependencies()
    if base.shape != donor.shape or base.ndim != 3 or base.shape[0] != 1:
        raise ValueError("base/donor must have identical [1,tokens,width] shapes")
    if base.shape[1] < 2 or base.shape[2] < 2:
        raise ValueError("a cursor plus context and at least two features are required")
    if not bool(torch.isfinite(base).all() and torch.isfinite(donor).all()):
        raise ValueError("nonfinite state")
    integer(random_controls, 1, 32, "random controls")
    rng = np.random.default_rng(seed)
    difference = donor[:, -1:] - base[:, -1:]
    target_norm = torch.linalg.vector_norm(difference)
    transfer = base.clone()
    transfer[:, -1:] = donor[:, -1:]
    states = [transfer, base.clone(), donor.clone()]
    names = ["cursor_transfer", "sham", "native_donor_replay"]
    gauge = base.clone()
    gauge[:, -1:] += target_norm / (base.shape[-1] ** 0.5)
    states.append(gauge)
    names.append("cursor_constant_shift")
    for index in range(random_controls):
        direction = torch.as_tensor(rng.standard_normal(tuple(difference.shape)),
                                    dtype=base.dtype, device=base.device)
        length = torch.linalg.vector_norm(direction)
        if not bool(length > 0):
            raise RuntimeError("zero random direction")
        patched = base.clone()
        patched[:, -1:] += direction * (target_norm / length)
        states.append(patched)
        names.append(f"random_cursor_{index}")
    for name, state in zip(names, states):
        if name != "native_donor_replay" and not torch.equal(state[:, :-1], base[:, :-1]):
            raise RuntimeError("an intervention changed receiver context tokens")
    return names, torch.cat(states, dim=0)


def execute_pair(model, native, states, embedding, row, *, horizon, seed, random_controls):
    import torch
    receiver, donor, visit = row["receiver"], row["donor"], row["visit"]
    description = row["claim"]
    claim = Counterfactual(tuple(description["receiver_graph"]), tuple(description["donor_graph"]),
                           description["receiver_cursor"], description["donor_cursor"], horizon)
    base, source = states[visit, receiver:receiver + 1], states[visit, donor:donor + 1]
    names, patched = make_interventions(base, source, seed=seed, random_controls=random_controls)
    inputs = embedding[receiver:receiver + 1].expand(len(names), -1, -1).clone()
    donor_index, sham_index = names.index("native_donor_replay"), names.index("sham")
    inputs[donor_index] = embedding[donor]
    current_logits = model.readout(patched[:, -1])
    donor_current = model.readout(source[:, -1])
    if not torch.allclose(current_logits[:1], donor_current, rtol=2e-5, atol=2e-6):
        raise RuntimeError("cursor patch failed native-head parity with donor")
    resumed = native.rollout(model, patched, inputs, horizon)
    future_logits = model.readout(resumed[:, :, -1])
    parity = {}
    for name, index, item in (("sham", sham_index, receiver),
                              ("native_donor_replay", donor_index, donor)):
        reference = states[visit + 1:visit + 1 + horizon, item:item + 1]
        parity[name] = native.assert_state_parity(resumed[:, index:index + 1], reference,
                            rtol=2e-5, atol=2e-6, name=f"MC02 {name} full-state replay")
        parity[name]["logits"] = native.assert_state_parity(
            future_logits[:, index:index + 1], model.readout(reference[:, :, -1]),
            rtol=2e-5, atol=2e-6, name=f"MC02 {name} readout replay")
    current = current_logits.argmax(-1).cpu().tolist()
    futures = future_logits.argmax(-1).cpu().numpy()
    records = []
    base_injected = model.inject(base)
    paths = claim.paths()
    for index, name in enumerate(names):
        labels = list(map(int, futures[:, index]))
        scored = score_continuation(claim, int(current[index]), labels)
        delta = patched[index:index + 1] - base
        injected_delta = model.inject(patched[index:index + 1]) - base_injected
        role = "candidate_lift" if name == "cursor_transfer" else "comparator"
        if name == "native_donor_replay":
            role = "plumbing_control_donor_context"
        records.append({"name": name, "role": role,
                        "input_context": "donor" if name == "native_donor_replay" else "receiver",
                        "applied_l2": float(torch.linalg.vector_norm(delta)),
                        "after_inject_l2": float(torch.linalg.vector_norm(injected_delta)),
                        "receiver_oracle_future_match": labels == list(paths["receiver_baseline"][1:]),
                        "donor_oracle_future_match": labels == list(paths["donor_baseline"][1:]), **scored})
    if row.get("eligible") and not records[0]["lift_matches"]:
        raise RuntimeError("an eligible cursor-transfer failed its immediate semantic boundary")
    result = {**row, "intervention_seed": seed, "native_parity": parity, "arms": records,
              "random_controls": random_controls,
              "receiver_embedding_retained": True,
              "claims_global_causal_abstraction": False}
    arrays = {"patched": patched.cpu().numpy(), "resumed": resumed.cpu().numpy(),
              "embeddings": inputs.cpu().numpy(), "current_logits": current_logits.cpu().numpy(),
              "future_logits": future_logits.cpu().numpy()}
    return result, arrays


def summarize(rows: list[dict], *, status: str, visits: list[int]) -> dict | None:
    """Equal receiver-item weighting; donor reuse prohibits iid pair inference."""
    if status == "UNTRAINED_SMOKE":
        return None
    by_item = {}
    for row in rows:
        if not row.get("eligible"):
            continue
        transfer = row["arms"][0]
        random = [arm for arm in row["arms"] if arm["name"].startswith("random_cursor_")]
        by_item.setdefault(row["receiver"], []).append((
            int(transfer["counterfactual_trace_match"]),
            sum(arm["counterfactual_future_match"] for arm in random) / len(random)))
    means = [{"receiver": item, "pair_count": len(values),
              "causal_trace_match": sum(v[0] for v in values) / len(values),
              "random_future_match": sum(v[1] for v in values) / len(values)}
             for item, values in sorted(by_item.items())]
    mean = sum(row["causal_trace_match"] for row in means) / len(means) if means else None
    by_visit = {}
    for visit in visits:
        part = [row for row in rows if row.get("eligible") and row["visit"] == visit]
        by_visit[str(visit)] = {"pairs": len(part), "trace_matches": sum(
            row["arms"][0]["counterfactual_trace_match"] for row in part)}
    return {"status": "MEASURED_CHECKPOINT" if means else "NO_ELIGIBLE_PAIRS",
            "primary": "equal_receiver_mean_of_finite_intervention_trace_matches",
            "value": mean, "receivers_with_selected_pairs": len(means),
            "receiver_means": means, "by_visit": by_visit,
            "uncertainty": "descriptive; shared donors and repeated receivers are dependent",
            "scope": "fixed checkpoint, selected cursor lift, declared visits and horizon"}


def run(args) -> dict:
    torch, native = native_dependencies()
    integer(args.items, 2, 512, "items")
    integer(args.horizon, 1, 64, "horizon")
    integer(args.max_pairs, 1, 128, "max_pairs")
    integer(args.random_controls, 1, 32, "random controls")
    integer(args.cpu_threads, 1, 16, "cpu threads")
    integer(args.seed, 0, 2**31 - 1, "seed")
    if not args.visits or sorted(set(args.visits)) != args.visits:
        raise ValueError("visits must be unique, increasing positive integers")
    for visit in args.visits:
        integer(visit, 1, 64, "visit")
    if max(args.visits) + args.horizon > 64:
        raise ValueError("total trace budget must be at most 64 loops")
    if args.smoke:
        if args.checkpoint is not None or args.expected_checkpoint_sha256 is not None:
            raise ValueError("smoke must not claim a checkpoint")
    elif args.checkpoint is None or args.expected_checkpoint_sha256 != file_hash(args.checkpoint):
        raise ValueError("existing checkpoint and independently supplied SHA-256 pin required")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty to preserve frozen manifests")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    model, cfg = native.load_model(args.checkpoint, smoke=args.smoke, seed=args.seed,
                                  device=args.device, dtype="float32")
    from research_gym.scripts.loopscope_toy import make_batch
    loops = max(args.visits) + args.horizon
    tokens, _ = make_batch(cfg["n_nodes"], args.items, args.seed, loops, depth=None)
    tokens = tokens.to(args.device)
    with torch.inference_mode():
        embedding, states = native.native_trace(model, tokens, loops)
        parity = native.verify_native_trace(model, tokens, states, rtol=2e-5, atol=2e-6)
        endpoint_logits = model.readout(states[args.visits, :, -1])
        if not bool(torch.isfinite(states).all() and torch.isfinite(endpoint_logits).all()):
            raise RuntimeError("nonfinite native trace")
        endpoints = endpoint_logits.argmax(-1).cpu().tolist()
        token_array = tokens.cpu().numpy()
        n = cfg["n_nodes"]
        graph_rows = []
        for item in range(args.items):
            graph = [0] * n
            for source, destination in token_array[item, :2*n].reshape(n, 2):
                graph[int(source)] = int(destination)
            graph_rows.append(graph)
        starts = list(map(int, token_array[:, -1]))
        manifest = select_pairs(graph_rows, starts, endpoints, args.visits,
                                horizon=args.horizon, max_pairs=args.max_pairs, seed=args.seed + 1)
        manifest_path = out / "selection.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        selection_file_hash = file_hash(manifest_path)
        selected = list(manifest["selected_pairs"])
        fallback = False
        if args.smoke and not selected:
            visit, receiver, donor = args.visits[0], 0, 1
            claim = Counterfactual(tuple(graph_rows[receiver]), tuple(graph_rows[donor]),
                oracle(graph_rows[receiver], starts[receiver], visit)[-1],
                oracle(graph_rows[donor], starts[donor], visit)[-1], args.horizon)
            selected = [{"visit": visit, "receiver": receiver, "donor": donor,
                         "eligible": False, "status": "UNGATED_ENGINEERING_FALLBACK",
                         "claim": claim.description()}]
            fallback = True
        rows, replay_arrays = [], {}
        for index, row in enumerate(selected):
            result, arrays = execute_pair(model, native, states, embedding, row,
                horizon=args.horizon, seed=args.seed + 2 + index, random_controls=args.random_controls)
            rows.append(result)
            for name, value in arrays.items():
                replay_arrays[f"pair_{index}_{name}"] = value
        if file_hash(manifest_path) != selection_file_hash:
            raise RuntimeError("selection manifest changed during interventions")
        validate_manifest(json.loads(manifest_path.read_text()))
        trace_path = out / "clean_trace.npz"
        np.savez_compressed(trace_path, full_states=states.cpu().numpy(),
            input_embedding=embedding.cpu().numpy(), tokens=token_array,
            endpoint_logits=endpoint_logits.cpu().numpy(), visits=np.asarray(args.visits))
        replay_path = out / "replays.npz"
        np.savez_compressed(replay_path, **replay_arrays)
    status = "UNTRAINED_SMOKE" if args.smoke else "MEASURED_CHECKPOINT"
    source_paths = [Path(__file__), ROOT / "research_gym/analysis/intervention_certificate.py",
                    ROOT / "research_gym/analysis/mechanistic_certificate.py", Path(native.__file__),
                    ROOT / "research_gym/scripts/loopscope_toy.py"]
    receipt = {"schema": "mc02.native.v1", "status": status,
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
        "checkpoint_sha256": None if args.smoke else file_hash(args.checkpoint),
        "state_dict_sha256": native._state_dict_hash(model), "config": cfg,
        "device": args.device, "dtype": "float32", "torch": torch.__version__,
        "python": platform.python_version(), "seed": args.seed, "visits": args.visits,
        "horizon": args.horizon, "items": args.items, "max_pairs": args.max_pairs,
        "random_controls": args.random_controls, "cpu_threads": args.cpu_threads,
        "native_parity": parity, "selected_pairs": len(manifest["selected_pairs"]),
        "engineering_fallback_executed": fallback, "selection_sha256": selection_file_hash,
        "clean_trace_sha256": file_hash(trace_path), "replays_sha256": file_hash(replay_path),
        "wall_seconds": time.monotonic() - started, "utc_created": datetime.now(timezone.utc).isoformat(),
        "trained_checkpoints_evaluated": 0 if args.smoke else 1,
        "trusted_capture": True, "cryptographic_source_binding": False,
        "counterfactual_target_is_task_hypothesis": True}
    result = {"receipt": receipt, "coverage": {key: manifest[key] for key in (
        "receiver_slots", "endpoint_correct_slots", "eligible_receiver_slots", "candidate_ordered_pairs")},
        "scientific_summary": summarize(rows, status=status, visits=args.visits), "pairs": rows}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--visits", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8, 16])
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=940211)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.controls:
        result = exact_controls()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "example"}, indent=2))
    else:
        result = run(args)
        print(json.dumps({"receipt": result["receipt"], "coverage": result["coverage"],
                          "scientific_summary": result["scientific_summary"]}, indent=2))


if __name__ == "__main__":
    main()
