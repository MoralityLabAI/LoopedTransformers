"""Replay archived shallow graphs and exercise admissible DELAY39 data custody synthetically."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from experiments.nan_delay39.graph_host import graph_cells
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_ranksite13.resources import below_normal

from .run import CONFIG, DELAY, HERE, REPO, collect, assess, digest, pilot
from .select import layered, mask_sets, node_entries, ranked_stage


ARCHIVE_PINS = {
    "nf4": "e09bd6a7a191e77061d651da06bdbe0fb1e326a0302f548c12242cc362eff365",
    "bf16": "11fb8f6418c3eabd33567bae514f3df6bf2d13a025e3c4e0fcba6f9564ff5b48",
}


def shallow_archive():
    proof = {}
    for backend, expected in ARCHIVE_PINS.items():
        data = REPO / "runs/nan_readpath37" / f"{backend}_private.json"
        if sha(data) != expected:
            raise ValueError("The verified READPATH37 native record archive changed")
        graphs, checks = 0, 0
        for row in json.loads(data.read_bytes()):
            values = row.get("signed_direct_graph", [])
            if not values or any(effect is None for _, effect in values):
                continue
            assert row["native_closure"] and len(values) == 8
            edges = ranked_stage([(0, i, w) for i, (w, _) in enumerate(values, 1)])
            edges += ranked_stage([(i, 9, v) for i, (_, v) in enumerate(values, 1)])
            entry = node_entries(10, edges)
            cutoffs = sorted({0., 1., *(e.rank for e in edges if math.isfinite(e.rank))})
            assert row["path_identity_cutoffs"] == len(cutoffs)
            for cutoff in cutoffs:
                local_in = {e.target for e in edges if e.rank <= cutoff}
                local_out = {e.source for e in edges if e.rank <= cutoff}
                traversed = {i for i in range(1, 9) if entry[i] <= cutoff}
                assert local_in & local_out == traversed
                checks += 1
            graphs += 1
        proof[backend] = {"complete_graphs": graphs, "verified_breakpoints": checks}
    assert sum(v["complete_graphs"] for v in proof.values()) == 193
    assert sum(v["verified_breakpoints"] for v in proof.values()) == 1890
    return proof


def witness(cfg):
    rng = random.Random(47019)
    lengths = len(cfg["source_components"]), len(cfg["receiver_components"])
    a, b = lengths
    for index in range(1000):
        source = [rng.randint(1, 59) for _ in range(a)]
        sink = [rng.randint(-37, 37) for _ in range(b)]
        middle = [[rng.randint(-51, 69) for _ in range(b)] for _ in range(a)]
        graph = {"source": source, "sink": sink, "middle": middle, "activity1": source,
                 "activity2": [rng.random() for _ in range(b)], "isolated1": [rng.random() for _ in range(a)],
                 "isolated2": list(map(abs, sink))}
        result = mask_sets([(graph, graph)] * cfg["minimum_complete_CAL_roots"], cfg)
        if result["sieve_vs_degree_requires_distinct_source_interventions"] and any(
                not v["receiver_observer"]["same"] and v["source"]["same"]
                for name, v in result["overlap_vs_sieve"].items() if name != "sieve"):
            return graph, index
    raise AssertionError("Cannot plant both differing source masks and receiver-only aliases")


def brute(graph):
    n, edges = layered(graph["source"], graph["middle"], graph["sink"])
    dynamic = node_entries(n, edges)
    adjacency = {i: [e for e in edges if e.source == i and math.isfinite(e.rank)] for i in range(n)}
    witnessed = [math.inf] * n

    def trace(path, cutoff):
        if path[-1] == n - 1:
            for v in path:
                witnessed[v] = min(witnessed[v], cutoff)
            return
        for edge in adjacency[path[-1]]:
            trace(path + (edge.target,), max(cutoff, edge.rank))
    trace((0,), 0.)
    assert witnessed == dynamic
    return sum(math.isfinite(v) for v in witnessed)


def native_fixture(gold):
    return [{"id": row["id"], "text": label, "token_id": ord(label) - 65}
            for row in gold for label in ["A" if row["arm"] == "focused_erased" else row["answer"]]]


def synthesise(graph, row, foil, tokens, cfg, missing):
    send, receive = cfg["source_components"], cfg["receiver_components"]
    target = "ABCDEFGHIJKLMNOP".index(row["answer"])
    other = "ABCDEFGHIJKLMNOP".index(foil["answer"])
    assert target != other
    packed_sink = {}
    for j, value in zip(receive, graph["sink"], strict=True):
        slopes = [0.0] * 16
        slopes[target] = float(value)
        packed_sink[str(j)] = {"signed_label_slopes_per_read_coefficient": slopes}
    if missing:
        packed_sink[str(receive[0])]["signed_label_slopes_per_read_coefficient"] = None
    cases = {}
    for i, amplitude in zip(send, graph["isolated1"], strict=True):
        for tag, sign in (("positive", 1), ("negative", -1)):
            logits = [0.0] * 16
            logits[target] = sign * amplitude
            cases[f"component_{i}_{tag}"] = {"requested_l2": 1.0, "realized_l2": 1.0, "eligible": True,
                                             "response": {"phase2_label_logits": logits}}
    return {"source_id": row["id"], "prompt_sha256": tokens[row["id"]]["sha256"], "self_closes": True,
            "phase1_source_writes": graph["source"], "baseline": {"phase2_read": graph["activity2"]},
            "phase2_terminal": {"phase2_write_responses": packed_sink, "selected_head_replay_max_delta": 0.0},
            "signed_phase2_edges_per_l2": dict(zip(map(str, send), graph["middle"])), "cases": cases}


def synthetic_journal(work, source_path, graph, cfg, mode):
    work.mkdir(parents=True, exist_ok=False)
    gold = json.loads((source_path / "gold.json").read_bytes())
    tokens = json.loads((source_path / "tokens.json").read_bytes())["row_bindings"]
    pins = json.loads((DELAY / "SOURCE_CHECK.json").read_bytes())
    assert digest(gold) == pins["gold_digest"] and digest(tokens) == pins["token_digest"]
    native = native_fixture(gold)
    if mode == "native_refusal":
        copy = next(row["id"] for row in gold if row["arm"] == "copy")
        for record in native:
            if record["id"] == copy:
                record["text"] = "X"
    old = json.loads((DELAY / "registration.json").read_bytes())
    gate = assess(native, gold, old) | {"native_record_digest": digest(native)}
    write_new(work / "native.json", native)
    write_new(work / "qualification.json", gate)
    write_new(work / "preflight.json", {"fixture": True})
    old_manifest = sha(pilot.SEAL)
    write_new(work / "admission.json", {"backend": work.name, "published_commit": "synthetic",
                                        "source_manifest_sha256": old_manifest})
    journal = work / "closed.jsonl"
    focus = {(row["root"], row["world"]): row for row in gold if row["arm"] == "focused"}
    if gate["passed"]:
        with journal.open("x", encoding="utf-8", newline="\n") as stream:
            for i, ((root, world), row) in enumerate(focus.items()):
                twin = focus[root, "B" if world == "A" else "A"]
                case = synthesise(graph, row, twin, tokens, cfg, mode == "sparse" and i in (0, 2, 4))
                file = work / f"episode_{i:04d}.json"
                write_new(file, case)
                stream.write(json.dumps({"file": file.name, "sha256": sha(file)}) + "\n")
    target = TARGETS[work.name]
    write_new(work / "terminal.json", {
        "status": "closed" if gate["passed"] else "native_gate_failed",
        "backend": work.name, "published_commit": "synthetic", "source_manifest_sha256": old_manifest,
        "preflight_sha256": sha(work / "preflight.json"), "target_before": target, "target_after": target,
        "quant_unchanged": True, "checkpoint_verified_after": True, "closed_focused_worlds": len(focus) if gate["passed"] else 0,
        "closed_journal_sha256": sha(journal) if journal.exists() else None,
    })
    read = lambda: collect(work, gold, tokens, [1.0] * len(cfg["receiver_components"]), cfg, lambda: None)
    return read, read()


def synthetic(out, source, cfg):
    graph, seed_index = witness(cfg)
    brute_reachable_nodes = brute(graph)
    closed, result = synthetic_journal(out / "closed/nf4", source, graph, cfg, "closed")
    assert result["status"] == "frozen" and result["complete_CAL_roots"] == 8
    assert result["sieve_vs_degree_requires_distinct_source_interventions"]
    assert all(len(mask["source"]) == 3 and len(mask["receiver_observer"]) == 4 for mask in result["masks"].values())
    groups = result["physical_source_intervention_groups"]
    assert len(groups) < len(result["masks"])
    assert any(item["source"]["same"] and not item["receiver_observer"]["same"]
               for item in result["overlap_vs_sieve"].values())
    assert Counter(name for group in groups for name in group["selectors"]) == Counter(result["masks"].keys())
    _, sparse = synthetic_journal(out / "sparse/bf16", source, graph, cfg, "sparse")
    assert sparse["status"] == "insufficient_complete_CAL_roots" and sparse["complete_CAL_roots"] == 5
    assert sparse["per_world_status"]["unqualified_receiver_sink"] == 3
    _, early = synthetic_journal(out / "refused/nf4", source, graph, cfg, "native_refusal")
    assert early["status"] == "native_gate_failed" and "masks" not in early
    file = out / "closed/nf4/episode_0000.json"
    file.write_bytes(file.read_bytes() + b" ")
    try:
        closed()
    except ValueError as error:
        assert "episode bytes" in str(error)
    else:
        raise AssertionError("Tampered synthetic episode was scored")
    signs = graph_cells([2., 2.], [[2., 0.], [0., -2.]], [1., 1.])[-1]
    assert signs["through_paths"] == 2 and signs["signed_path_product_sum"] == 0 and signs["absolute_path_product_sum"] == 8
    return {"synthetic_seed_index": seed_index, "brute_reachable_nodes": brute_reachable_nodes,
            "synthetic_methods": len(result["masks"]), "synthetic_distinct_source_interventions": len(groups),
            "synthetic_complete_root_refusal": sparse["complete_CAL_roots"],
            "corrupt_episode_refused": True, "sign_cancellation_verified": True,
            "receiver_only_difference_does_not_schedule_gpu": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    below_normal()
    cfg = json.loads(CONFIG.read_bytes())
    archived = shallow_archive()
    args.output.mkdir(parents=True, exist_ok=False)
    result = synthetic(args.output, args.source_dir, cfg)
    proof = {"status": "passed", "real_archived_graphs": sum(s["complete_graphs"] for s in archived.values()),
             "real_cutoffs_replayed": sum(s["verified_breakpoints"] for s in archived.values()),
             "archive_by_backend": archived, **result, "new_model_calls": 0,
             "source_sha256": {name: sha(HERE / name) for name in ("run.py", "select.py", "verify.py")}}
    write_new(args.output / "verification.json", proof)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
