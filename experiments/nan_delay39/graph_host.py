"""Host-only signed causal graph coverage and matched outcome accounting."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.nan_comp14.parser import label
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_readpath37.run import path_members, ranked_entries
from experiments.nan_ranksite13.resources import below_normal
from .source import HERE, digest


OCCUPANCIES = (.25, .50, .75, 1.0)


def graph_cells(source_write, edges, sink):
    """Evaluate the four-stage graph with signed edges and local membership control."""
    send_nodes, receive_nodes = len(source_write), len(sink)
    if len(edges) != send_nodes or any(len(row) != receive_nodes for row in edges):
        raise ValueError("Cross-pass graph dimensions disagree")
    source, terminal = 0, 1 + send_nodes + receive_nodes
    phase1 = range(1, 1 + send_nodes)
    phase2 = range(1 + send_nodes, terminal)
    middle = [w for row in edges for w in row]
    all_stages = [
        [(source, b, weight) for b, weight in zip(phase1, source_write)],
        [(a, b, weight) for (a, b), weight in zip(((a, b) for a in phase1 for b in phase2), middle)],
        [(a, terminal, weight) for a, weight in zip(phase2, sink)],
    ]
    edge_rows = []
    weights = {}
    for stage in all_stages:
        if any(not math.isfinite(w) for _, _, w in stage):
            raise ValueError("Missing or nonfinite edge cannot define a complete graph")
        for (a, b, w), p in zip(stage, ranked_entries([w for _, _, w in stage])):
            edge_rows.append((a, b, p))
            weights[a, b] = w
    results = []
    for cutoff in OCCUPANCIES:
        path, kept = path_members({source}, {terminal}, range(terminal + 1), edge_rows, cutoff)
        live = set(kept)
        local = ({a for a, _ in live} & {b for _, b in live}) - {source, terminal}
        used = path - {source, terminal}
        paths = [weights[source, a] * weights[a, b] * weights[b, terminal]
                 for a in phase1 for b in phase2
                 if all(pair in live for pair in ((source, a), (a, b), (b, terminal)))]
        results.append({"occupancy_rank_fraction": cutoff, "backbone_nodes": len(used),
                        "local_in_out_nodes": len(local), "exact_membership_tie": used == local,
                        "through_paths": len(paths), "signed_path_product_sum": sum(paths),
                        "absolute_path_product_sum": sum(map(abs, paths))})
    return results


def score_case(item, target, foil, source_ids, indices):
    out = {"native_correct": label(item["baseline"]["phase2_decoded"]) == target}
    donor = item["cases"]["twin_donor"]
    out["donor_correct"] = (label(donor["response"]["phase2_decoded"]) == target
                            if donor["eligible"] else None)
    for name in ("span_sham", "ambient_sham"):
        entry = item["cases"][name]
        out[name + "_correct"] = (label(entry["response"]["phase2_decoded"]) == target
                                  if item["donor_sham_pairs"][name] and entry["eligible"] else None)

    slopes = item["signed_phase2_edges_per_l2"]
    sink_responses = item["phase2_terminal"]["phase2_write_responses"]
    label_a, label_b = source_ids.index(target), source_ids.index(foil)
    middle = [slopes[str(i)] for i in indices]
    last = [sink_responses[str(i)]["signed_label_slopes_per_read_coefficient"] for i in range(len(sink_responses))]
    if any(row is None for row in middle + last):
        out["graph"] = None
    else:
        final = [row[label_a] - row[label_b] for row in last]
        out["graph"] = graph_cells(item["phase1_source_writes"], middle, final)
    return out


def paired_outcomes(per_root):
    out = {}
    for left, right in (("native_correct", "donor_correct"), ("span_sham_correct", "donor_correct"),
                        ("ambient_sham_correct", "donor_correct")):
        counters = Counter()
        effects = []
        complete = 0
        for worlds in per_root.values():
            paired = [(w[left], w[right]) for w in worlds if w is not None and w[left] is not None and w[right] is not None]
            for a, b in paired:
                counters["repairs" if b and not a else "regressions" if a and not b else "ties"] += 1
            if len(paired) == 2:
                complete += 1
                effects.append(fmean(int(b) - int(a) for a, b in paired))
        out[f"{right}_minus_{left}"] = {
            "registered_roots": len(per_root), "complete_roots": complete,
            "unknown_worlds": 2 * len(per_root) - sum(counters.values()),
            "repairs": counters["repairs"], "regressions": counters["regressions"], "ties": counters["ties"],
            "root_macro_change": fmean(effects) if effects else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    below_normal()
    cfg = json.loads((HERE / "registration.json").read_bytes())
    pins = json.loads((HERE / "SOURCE_CHECK.json").read_bytes())
    gold = json.loads((args.source_dir / "gold.json").read_bytes())
    if digest(gold) != pins["gold_digest"]:
        raise ValueError("Independent host gold digest changed")
    selected = {g["id"]: g for g in gold if g["arm"] == "focused"}
    per_root = {row["root"]: [None, None] for row in selected.values()}
    terminal = json.loads((args.artifact_dir / "terminal.json").read_bytes())
    expected_target = json.loads((HERE.parents[1] / "experiments/nan_readpath37/registration.json").read_bytes())[
        "backends"
    ][terminal["backend"]]["target"]
    if (terminal["target_before"] != expected_target or terminal["target_after"] != expected_target
            or terminal["source_manifest_sha256"] != sha(HERE / "SOURCE_MANIFEST.json")
            or not terminal["quant_unchanged"] or not terminal["checkpoint_verified_after"]):
        raise ValueError("Wrong terminal identity for a scientific record")
    graph_results = []
    if terminal["status"] == "closed":
        journal = args.artifact_dir / "closed.jsonl"
        if sha(journal) != terminal["closed_journal_sha256"]:
            raise ValueError("Closed DELAY39 journal digest changed")
        ids = set()
        for entry in map(json.loads, journal.read_bytes().splitlines()):
            path = args.artifact_dir / f"episode_{len(ids):04d}.json"
            if path.name != entry["file"] or sha(path) != entry["sha256"]:
                raise ValueError("Closed episode path or digest changed")
            item = json.loads(path.read_bytes())
            identity = item["source_id"]
            if identity in ids or identity not in selected:
                raise ValueError("Duplicate or unknown DELAY39 result")
            ids.add(identity)
            gold = selected[identity]
            twin_world = "B" if gold["world"] == "A" else "A"
            twin = selected[f"{gold['root']}:{twin_world}:focused"]
            summary = score_case(item, gold["answer"], twin["answer"], "ABCDEFGHIJKLMNOP", cfg["phase1_components"])
            per_root[gold["root"]][int(gold["world"] == "B")] = summary
            if summary["graph"] is not None:
                graph_results.append(summary["graph"])
        if ids != set(selected):
            raise ValueError("Complete DELAY39 journal is missing registered results")
    elif terminal["status"] != "native_gate_failed":
        raise ValueError("DELAY39 terminal state does not admit a scientific score")
    output = {
        "experiment": "NAN-DELAY-39", "backend": terminal["backend"], "terminal_status": terminal["status"],
        "registered_roots": len(per_root), "explicitly_complete_signed_graphs": len(graph_results),
        "missing_signed_graphs": len(selected) - len(graph_results), "formal_comparisons": 0,
        "paired_correctness": paired_outcomes(per_root),
        "cutoff_summary": [{"occupancy_rank_fraction": cutoff,
                            "complete_graphs_compared": len(graph_results),
                            "membership_ties": sum(row[i]["exact_membership_tie"] for row in graph_results),
                            "mean_abs_signed_path_sum": fmean(abs(row[i]["signed_path_product_sum"]) for row in graph_results)
                            if graph_results else None,
                            "mean_absolute_path_product_sum": fmean(row[i]["absolute_path_product_sum"] for row in graph_results)
                            if graph_results else None}
                           for i, cutoff in enumerate(OCCUPANCIES)],
        "verdict": "CAL_ONLY_REDESIGN_OR_PILOT_ONLY", "sealed_test_accessed": False,
    }
    write_new(args.output, output)


if __name__ == "__main__":
    main()
