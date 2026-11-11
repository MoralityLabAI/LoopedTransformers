"""Run MC-01 exact controls or audit a pinned SEM-ECHO full_trace.npz capture."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from research_gym.analysis.mechanistic_certificate import (
    BN254, M31, Statement, audit_finite, check, digest, encode, oracle, relation,
)


def file_hash(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            out.update(chunk)
    return out.hexdigest()


def export_relation(statement: Statement, labels, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    circuit = relation(len(statement.graph), statement.steps)
    witness = encode(statement, labels)
    (directory / "trace.circom").write_text(circuit.circom(), encoding="utf-8")
    (directory / "witness.json").write_text(json.dumps({
        "public_values": witness[1:1 + circuit.public_count],
        "private_values": witness[1 + circuit.public_count:]}, indent=2) + "\n")
    (directory / "relation.json").write_text(json.dumps(circuit.description(), indent=2) + "\n")
    metadata = {"statement": statement.identity(), "witness_valid": not circuit.failures(witness),
                "field": str(BN254), "public_wires": circuit.public_count,
                "private_wires": len(witness) - 1 - circuit.public_count,
                "constraints": len(circuit.constraints), "circom_compiled": False,
                "proof_generated": False, "source_binding": "outside_relation",
                "file_sha256": {name: file_hash(directory / name) for name in (
                    "trace.circom", "witness.json", "relation.json")}}
    (directory / "export.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def controls(out: Path) -> dict:
    """Exhaustive small graphs, full witness mutations, and semantic counterexamples."""
    complete = 0
    for n in (2, 3):
        for graph in itertools.product(range(n), repeat=n):
            for start in range(n):
                for steps in range(1, 5):
                    labels = oracle(graph, start, steps)
                    statement = Statement(graph, start, steps, labels[-1])
                    witness = encode(statement, labels)
                    circuit = relation(n, steps)
                    for field in (BN254, M31):
                        if circuit.failures(witness, field):
                            raise RuntimeError("valid reference trace was rejected")
                        complete += 1
    candidates = accepts = 0
    for graph in itertools.product(range(3), repeat=3):
        for start in range(3):
            truth = oracle(graph, start, 3)
            statement = Statement(graph, start, 3, truth[-1])
            circuit = relation(3, 3)
            for suffix in itertools.product(range(3), repeat=3):
                labels = (start,) + suffix
                valid = not circuit.failures(encode(statement, labels))
                if valid != (labels == truth):
                    raise RuntimeError("complete trace enumeration disagrees with reference")
                candidates += 1
                accepts += int(valid)
    graph = (2, 2, 1)
    labels = oracle(graph, 0, 3)
    statement = Statement(graph, 0, 3, labels[-1])
    circuit = relation(3, 3)
    witness = encode(statement, labels)
    mutations = 0
    for field in (BN254, M31):
        for i in range(1, len(witness)):
            wrong = witness.copy()
            wrong[i] = (wrong[i] + 1) % field
            if not circuit.failures(wrong, field):
                raise RuntimeError(f"wire mutation {i} escaped constraints")
            mutations += 1
    # Correct endpoint, wrong intermediate states. An oracle witness can hide this.
    observed = (0, 1, 1, 2)
    counterfeit = check(statement, labels, expected=statement, trusted_observed=observed)
    if not counterfeit["abstract_valid"] or counterfeit["trusted_capture_accept"]:
        raise RuntimeError("counterfeit source-linkage control failed")
    if observed[-1] != labels[-1] or check(statement, observed)["abstract_valid"]:
        raise RuntimeError("endpoint-only false acceptance control failed")
    # Concrete state = (shadow bit, output-driving bit). Natural runs stay diagonal.
    alpha = (0, 0, 1, 1)
    output = (0, 1, 0, 1)
    flip_both = (3, 2, 1, 0)
    set_shadow_zero = (0, 1, 0, 1)
    passive = audit_finite([flip_both], alpha, [(1, 0)], output, (0, 1), domain=(0, 3))
    active = audit_finite([flip_both, set_shadow_zero], alpha, [(1, 0), (0, 0)],
                          output, (0, 1))
    if not passive["exact_on_declared_domain"] or active["exact_on_declared_domain"]:
        raise RuntimeError("shadow-register causal control failed")
    # Positive quotient: nuisance coordinate is irrelevant under both operations.
    positive = audit_finite([flip_both, (0, 0, 2, 2)], alpha, [(1, 0), (0, 1)],
                            alpha, (0, 1))
    if not positive["exact_on_declared_domain"]:
        raise RuntimeError("exact intervention-preserving quotient was rejected")
    out.mkdir(parents=True, exist_ok=True)
    exported = export_relation(statement, labels, out / "example")
    result = {"status": "EXACT_CONTROLS_ONLY", "valid_field_checks": complete,
              "candidate_traces_checked": candidates, "candidate_traces_accepted": accepts,
              "single_wire_mutations_rejected": mutations,
              "counterfeit_oracle_witness": counterfeit,
              "endpoint_only_would_accept": True, "shadow_passive": passive,
              "shadow_with_interventions": active, "positive_quotient": positive,
              "example_export": exported, "trained_neural_evaluation": "not_run",
              "circom_compilation": "not_run", "stwo_proving": "not_run"}
    (out / "controls.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit_capture(trace_path: Path, metrics_path: Path, *, expected_trace: str,
                  expected_metrics: str, expected_checkpoint: str | None,
                  allow_untrained: bool = False) -> dict:
    """Audit a capture trusted by the caller; hashes ensure identity, not honest execution."""
    import numpy as np
    for path, expected in ((trace_path, expected_trace), (metrics_path, expected_metrics)):
        if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
            raise ValueError(f"independent file hash pin failed: {path.name}")
    receipt = json.loads(metrics_path.read_text())["receipt"]
    if receipt.get("schema_version") != "semantic_echo.looped_chase.v1":
        raise ValueError("unsupported SEM-ECHO capture schema")
    if receipt.get("trace_sha256") != expected_trace or receipt.get("fixed_depth_compatible") is not True:
        raise ValueError("receipt trace identity or fixed-depth contract mismatch")
    status = receipt.get("status")
    if status == "MEASURED_CHECKPOINT":
        if (not isinstance(expected_checkpoint, str) or len(expected_checkpoint) != 64 or
                any(c not in "0123456789abcdef" for c in expected_checkpoint) or
                receipt.get("checkpoint_sha256") != expected_checkpoint):
            raise ValueError("independent checkpoint pin is required")
    elif status == "UNTRAINED_SMOKE" and allow_untrained:
        if expected_checkpoint is not None or receipt.get("checkpoint_sha256") is not None:
            raise ValueError("untrained smoke must not claim a checkpoint")
    else:
        raise ValueError("trained capture required; smoke needs explicit permission")
    parity = receipt.get("native_parity", {})
    if parity.get("passed") is not True or parity.get("logit_parity", {}).get("passed") is not True:
        raise ValueError("native state/head parity receipt is missing")
    with np.load(trace_path, allow_pickle=False) as archive:
        names = ("full_states", "input_embedding", "tokens", "permutations", "starts",
                 "predictions", "own_head_logits", "oracle")
        if not set(names) <= set(archive.files):
            raise ValueError("incomplete SEM-ECHO trace")
        a = {name: archive[name] for name in names}
    predictions, graphs, starts = a["predictions"], a["permutations"], a["starts"]
    if predictions.ndim != 2 or graphs.ndim != 2:
        raise ValueError("predictions and graph arrays must be matrices")
    steps, count = predictions.shape
    if graphs.shape[0] != count:
        raise ValueError("item count mismatch")
    n = graphs.shape[1]
    if steps < 1 or count < 1 or a["full_states"].ndim != 4:
        raise ValueError("empty or malformed trace")
    d = a["full_states"].shape[-1]
    shapes = {"starts": (count,), "tokens": (count, 2 * n + 1),
              "full_states": (steps + 1, count, n + 1, d),
              "input_embedding": (count, n + 1, d), "own_head_logits": (steps, count, n),
              "oracle": (steps, count)}
    for name, shape in shapes.items():
        if a[name].shape != shape or not np.isfinite(a[name]).all():
            raise ValueError(f"shape/nonfinite error: {name}")
    for name in ("tokens", "permutations", "starts", "predictions", "oracle"):
        if not np.issubdtype(a[name].dtype, np.integer):
            raise ValueError(f"integer array required: {name}")
    if d < 1 or np.any(a["full_states"][0] != 0):
        raise ValueError("native initial state must be zero")
    if not np.array_equal(a["own_head_logits"].argmax(-1), predictions):
        raise ValueError("predictions disagree with the captured own head")
    cfg = receipt.get("config", {})
    if (receipt.get("items") != count or receipt.get("max_loops") != steps or
            cfg.get("n_nodes") != n or cfg.get("fixed_depth") is not True):
        raise ValueError("receipt dimensions/configuration disagree with trace")
    rows = []
    correct_by_step = [0] * steps
    for i in range(count):
        graph, start = tuple(map(int, graphs[i])), int(starts[i])
        if sorted(graph) != list(range(n)):
            raise ValueError("the SEM-ECHO source contract requires a permutation")
        edges = a["tokens"][i, :2 * n].reshape(n, 2)
        if sorted(map(int, edges[:, 0])) != list(range(n)):
            raise ValueError("source nodes in tokens must be unique")
        for src, dst in edges:
            if graph[int(src)] != int(dst):
                raise ValueError("token graph differs from transition graph")
        if int(a["tokens"][i, -1]) != start:
            raise ValueError("token start differs from source boundary")
        truth = oracle(graph, start, steps)
        if list(map(int, a["oracle"][:, i])) != list(truth[1:]):
            raise ValueError("stored oracle differs from independently recomputed oracle")
        observed = (start,) + tuple(map(int, predictions[:, i]))
        statement = Statement(graph, start, steps, truth[-1])
        verdict = check(statement, observed, expected=statement, trusted_observed=observed)
        mismatch = [t for t in range(1, steps + 1) if observed[t] != truth[t]]
        for t in range(steps):
            correct_by_step[t] += int(observed[t + 1] == truth[t + 1])
        rows.append({"item": i, "graph_sha256": digest(graph), "statement": statement.identity(),
                     "observed_labels": list(observed), "first_wrong_visit": min(mismatch) if mismatch else None,
                     "oracle_matches": steps - len(mismatch),
                     "final_answer_correct": observed[-1] == truth[-1], **verdict})
    return {"status": status, "schema": "mc01.capture_audit.v1", "items": count,
            "steps": steps, "source": {"trace_sha256": expected_trace,
                "metrics_sha256": expected_metrics, "checkpoint_sha256": expected_checkpoint},
            "abstract_trace_accept_count": sum(r["abstract_valid"] for r in rows),
            "trusted_capture_accept_count": sum(r["trusted_capture_accept"] for r in rows),
            "final_answer_correct_count": sum(r["final_answer_correct"] for r in rows),
            "per_visit_oracle_correct_count": correct_by_step, "rows": rows,
            "causal_fidelity": "unmeasured", "cryptographic_source_binding": False,
            "boundary": "q0 comes from the input start; q1..qT come from the captured own head",
            "scope": "fixed one-hop-per-loop hypothesis; rejection allows alternative internal algorithms"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    control = sub.add_parser("controls")
    control.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit")
    for name in ("trace", "metrics", "out"):
        audit.add_argument(f"--{name}", type=Path, required=True)
    for name in ("trace", "metrics"):
        audit.add_argument(f"--expected-{name}-sha256", required=True)
    audit.add_argument("--expected-checkpoint-sha256")
    audit.add_argument("--allow-untrained", action="store_true")
    args = parser.parse_args()
    if args.command == "controls":
        result = controls(args.out)
    else:
        result = audit_capture(args.trace, args.metrics, expected_trace=args.expected_trace_sha256,
                               expected_metrics=args.expected_metrics_sha256,
                               expected_checkpoint=args.expected_checkpoint_sha256,
                               allow_untrained=args.allow_untrained)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "rows", "example_export", "counterfeit_oracle_witness", "shadow_passive",
        "shadow_with_interventions", "positive_quotient")}, indent=2))


if __name__ == "__main__":
    main()
