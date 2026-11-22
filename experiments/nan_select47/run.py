"""Verify DELAY39 device receipts and freeze equal-budget selector masks."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from experiments.nan_delay39 import pilot
from experiments.nan_delay39.graph_host import graph_cells
from experiments.nan_delay39.host_gate import assess
from experiments.nan_delay39.source import digest
from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime import TARGETS
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha

from .select import mask_sets


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "registration.json"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
DELAY = REPO / "experiments/nan_delay39"


def registration():
    local = (".gitattributes", "__init__.py", "PROTOCOL.md", "registration.json", "run.py", "select.py", "verify.py",
             "CPU_VERIFICATION.json")
    parents = (
        "experiments/nan_delay39/SOURCE_MANIFEST.json", "experiments/nan_delay39/SOURCE_CHECK.json",
        "experiments/nan_delay39/pilot.py", "experiments/nan_delay39/graph_host.py",
        "experiments/nan_delay39/host_gate.py", "experiments/nan_readpath37/run.py",
        "experiments/nan_phase35/run.py",
    )
    fixture = json.loads((HERE / "CPU_VERIFICATION.json").read_bytes())
    if fixture["status"] != "passed" or fixture["real_archived_graphs"] != 193 or fixture["real_cutoffs_replayed"] != 1890:
        raise ValueError("Expected complete legacy and synthetic validation")
    for name, expected in fixture["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("CPU verification was recorded against different code")
    files = [HERE / path for path in local] + [REPO / path for path in parents]
    return {"experiment": "NAN-SELECT-47",
            "files_sha256": {p.relative_to(REPO).as_posix(): sha(p) for p in files}}


def verify():
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != registration():
        raise ValueError("Selector freeze differs from its registered implementation")


def extract(item, source, foil, scales, cfg, old_cfg):
    source_idx, receiver_idx = cfg["source_components"], cfg["receiver_components"]
    if (item["self_closes"] is not True or item["phase2_terminal"]["selected_head_replay_max_delta"] >
            old_cfg["maximum_phase2_readout_replay_logit_delta"]):
        raise ValueError("Clean-row or full-head sink closure failed")
    target = "ABCDEFGHIJKLMNOP".index(source["answer"])
    other = "ABCDEFGHIJKLMNOP".index(foil["answer"])
    activity1 = item["phase1_source_writes"]
    activity2 = item["baseline"]["phase2_read"]
    if len(activity1) != len(source_idx) or len(activity2) != len(receiver_idx):
        raise ValueError("Clean activity does not match the registered component occurrences")
    middle, sink, first_isolated = [], [], []
    for j in receiver_idx:
        slopes = item["phase2_terminal"]["phase2_write_responses"][str(j)]["signed_label_slopes_per_read_coefficient"]
        if slopes is None:
            return None, "unqualified_receiver_sink"
        sink.append((slopes[target] - slopes[other]) / scales[j])
    for i in source_idx:
        slopes = item["signed_phase2_edges_per_l2"][str(i)]
        if slopes is None:
            return None, "unqualified_delayed_read"
        if len(slopes) != len(receiver_idx):
            raise ValueError("Read coefficient count changed")
        middle.append([value * scales[j] for value, j in zip(slopes, receiver_idx, strict=True)])
        pair = [item["cases"][f"component_{i}_{tag}"] for tag in ("positive", "negative")]
        desired = pair[0]["requested_l2"]
        if (desired != pair[1]["requested_l2"] or not all(p["eligible"] for p in pair) or desired <= 0
                or abs(pair[0]["realized_l2"] - pair[1]["realized_l2"]) > old_cfg["max_contrast_dose_gap_fraction"] * desired):
            raise ValueError("A measured delayed edge disagrees with paired dose admission")
        margin = [case["response"]["phase2_label_logits"][target] - case["response"]["phase2_label_logits"][other]
                  for case in pair]
        first_isolated.append(abs(margin[0] - margin[1]) / sum(case["realized_l2"] for case in pair))
    values = {
        "source": activity1, "middle": middle, "sink": sink,
        "activity1": activity1, "activity2": [abs(a) * scales[j] for a, j in zip(activity2, receiver_idx, strict=True)],
        "isolated1": first_isolated, "isolated2": list(map(abs, sink)),
    }
    if any(not math.isfinite(v) or v < 0 for name in ("activity1", "activity2", "isolated1", "isolated2")
           for v in values[name]):
        raise ValueError("Invalid activity or isolated effect measurement")
    return values, "complete"


def collect(work, gold, tokens, scales, cfg, check):
    old_cfg = json.loads((DELAY / "registration.json").read_bytes())
    if cfg["source_components"] != old_cfg["phase1_components"]:
        raise ValueError("Selector source positions differ from the measured DELAY39 directions")
    terminal = json.loads((work / "terminal.json").read_bytes())
    backend = terminal["backend"]
    admitted = json.loads((work / "admission.json").read_bytes())
    if (backend != work.name or admitted["backend"] != backend or terminal["published_commit"] != admitted["published_commit"]
            or admitted["source_manifest_sha256"] != sha(pilot.SEAL)
            or terminal["source_manifest_sha256"] != sha(pilot.SEAL)
            or terminal["target_before"] != TARGETS[backend] or terminal["target_after"] != TARGETS[backend]
            or terminal["quant_unchanged"] is not True or terminal["checkpoint_verified_after"] is not True
            or terminal["preflight_sha256"] != sha(work / "preflight.json")):
        raise ValueError("Terminal does not attest to a frozen DELAY39 native condition")
    native = json.loads((work / "native.json").read_bytes())
    gate = assess(native, gold, old_cfg) | {"native_record_digest": digest(native)}
    if gate != json.loads((work / "qualification.json").read_bytes()):
        raise ValueError("Independent native qualification receipt differs")
    focus = [row for row in gold if row["arm"] == "focused"]
    focused = {row["id"]: row for row in focus}
    by_key = {(row["root"], row["world"]): row for row in focus}
    counts = Counter()
    if terminal["status"] == "native_gate_failed":
        if gate["passed"] or (work / "closed.jsonl").exists() or terminal["closed_focused_worlds"] != 0:
            raise ValueError("Inconsistent native qualification refusal")
        return {"backend": backend, "status": "native_gate_failed", "registered_worlds": len(focus),
                "complete_CAL_roots": 0, "per_world_status": {"native_gate_failed": len(focus)},
                "terminal_sha256": sha(work / "terminal.json")}
    if terminal["status"] != "closed" or not gate["passed"] or terminal["closed_focused_worlds"] != len(focus):
        raise ValueError("Only a complete qualified DELAY39 native lane can nominate masks")
    journal = work / "closed.jsonl"
    if sha(journal) != terminal["closed_journal_sha256"]:
        raise ValueError("Device journal hash differs from the terminal")
    pairs, seen, signed = defaultdict(dict), set(), []
    for i, line in enumerate(journal.read_bytes().splitlines(keepends=True)):
        check()
        path = work / f"episode_{i:04d}.json"
        if not line.endswith(b"\n") or json.loads(line) != {"file": path.name, "sha256": sha(path)}:
            raise ValueError("Closed DELAY39 episode bytes or order changed")
        item = json.loads(path.read_bytes())
        identity = item["source_id"]
        if identity in seen or identity not in focused or item["prompt_sha256"] != tokens[identity]["sha256"]:
            raise ValueError("Duplicate, changed or unregistered prompt")
        seen.add(identity)
        root, world = focused[identity]["root"], focused[identity]["world"]
        foil = by_key[root, "B" if world == "A" else "A"]
        graph, reason = extract(item, focused[identity], foil, scales, cfg, old_cfg)
        counts[reason] += 1
        if graph is not None:
            pairs[root][world] = graph
            profile = graph_cells(graph["source"], graph["middle"], graph["sink"])
            signed.append(next(g["signed_path_product_sum"] for g in profile
                               if g["occupancy_rank_fraction"] == cfg["calibration_cutoff"]))
    if seen != set(focused):
        raise ValueError("DELAY39 journal did not cover every focused world")
    paired = [(by_world["A"], by_world["B"]) for _, by_world in sorted(pairs.items()) if set(by_world) == {"A", "B"}]
    result = mask_sets(paired, cfg)
    return result | {
        "backend": backend, "registered_worlds": len(focus), "per_world_status": counts,
        "signed_path_sums_at_cutoff": {"positive_worlds": sum(x > 0 for x in signed),
                                        "negative_worlds": sum(x < 0 for x in signed),
                                        "zero_worlds": signed.count(0)},
        "terminal_sha256": sha(work / "terminal.json"), "native_journal_sha256": sha(journal),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--source-dir", type=Path)
    cli.add_argument("--backend", choices=tuple(TARGETS))
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.freeze:
        write_new(MANIFEST, registration())
        print(sha(MANIFEST))
        return
    if not all((args.source_dir, args.backend, args.output)):
        cli.error("--source-dir, --backend and --output are required")
    verify()
    cfg = json.loads(CONFIG.read_bytes())
    pins = json.loads((DELAY / "SOURCE_CHECK.json").read_bytes())
    gold = json.loads((args.source_dir / "gold.json").read_bytes())
    tokens = json.loads((args.source_dir / "tokens.json").read_bytes())["row_bindings"]
    if digest(gold) != pins["gold_digest"] or digest(tokens) != pins["token_digest"]:
        raise ValueError("Frozen DELAY39 source or tokenizer custody differs")
    work = pilot.ARTIFACT / args.backend
    if not (work / "terminal.json").is_file():
        raise RuntimeError("Awaiting the native DELAY39 device terminal for this precision")
    lease_cfg = {"minimum_start_available_RAM_bytes": 5 * 2**30, "minimum_available_RAM_bytes": 4 * 2**30,
                 "maximum_private_bytes": 768 * 2**20, "maximum_seconds": 120}
    with lease(lease_cfg) as (check, samples):
        import numpy as np

        _, writes = pilot.get_bank(args.backend, np)
        scales = np.linalg.norm(writes, axis=1).tolist()
        if any(not math.isfinite(x) or x <= 0 for x in scales):
            raise ValueError("The frozen write norm is invalid")
        outcome = collect(work, gold, tokens, scales, cfg, check)
    write_new(args.output, outcome | {
        "experiment": cfg["experiment"], "manifest_sha256": sha(MANIFEST),
        "cutoff": cfg["calibration_cutoff"], "source_quota": cfg["source_quota"],
        "receiver_observer_quota": cfg["receiver_quota"], "pretrained_forwards": 0,
        "formal_comparisons": cfg["formal_comparisons"], "sealed_test_accessed": False,
        "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
    })


if __name__ == "__main__":
    main()
