"""Exercise measured edge custody, exact ranks and physical-arm deduplication."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys

from experiments.nan_phase35.run import lease
from experiments.nan_readout34.runtime_plan import write_new
from experiments.nan_readout34.tokenizer_check import sha
from experiments.nan_select47 import run as selectors
from experiments.nan_select47.verify import synthetic_journal, witness

from .exact import CUTOFFS, block, encoded, equivalent_choice, nominations
from .run import BUDGET, MANIFEST, profiles, registered, source_payload


def heterogeneous_ties(cfg, np):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    base, _ = witness(cfg)
    worlds = []
    for count in primes:
        middle = np.where(np.arange(56) < count, np.arange(56) % 3 + 1, 0).reshape(7, 8)
        worlds.append(base | {"middle": middle.tolist()})
    pairs = [worlds[i:i + 2] for i in range(0, len(worlds), 2)]
    raw, rank, denom, dtype = encoded(pairs, np)
    assert dtype is object and denom > 2**63
    orders = np.array([list(range(7)), list(range(6, -1, -1)), [1, 4, 3, 6, 0, 2, 5]])

    def stage(values):
        compact = [abs(v) for v in values if v]
        return [Fraction(sum(x >= abs(v) for x in compact), len(compact)) if v else Fraction(2) for v in values]

    def rank_subset(rows, ids, cutoff, quota):
        key = [(sum(v <= cutoff for v in entries), sum(max(0, 1 - v) for v in entries),
                -sum(entries), -i) for i, entries in zip(ids, zip(*rows), strict=True)]
        return sorted(ids[i] for i in sorted(range(len(ids)), key=key.__getitem__, reverse=True)[:quota])

    for cutoff in CUTOFFS:
        computed = block(orders, raw, rank, denom, cutoff, cfg, dtype, np)
        for i, order in enumerate(orders):
            send, receive = [], []
            for world in worlds:
                source = stage([world["source"][j] for j in order])
                sink = stage(world["sink"])
                matrix = stage([v for row in world["middle"] for v in row])
                through = [[max(source[s], matrix[8*s + r], sink[r]) for r in range(8)] for s in range(7)]
                send.append([min(row) for row in through])
                receive.append([min(column) for column in zip(*through)])
            for phase, rows, ids, quota in ((0, send, cfg["source_components"], cfg["source_quota"]),
                                            (1, receive, cfg["receiver_components"], cfg["receiver_quota"])):
                assert computed[phase][i].tolist() == rank_subset(rows, ids, cutoff, quota)
    return 2 * len(orders) * len(CUTOFFS)


def physical_control_fixture(np):
    a = {"source": [1, 2, 3], "receiver_observer": [0, 1, 2, 3]}
    b = a | {"receiver_observer": [0, 1, 2, 4]}
    c = a | {"source": [1, 4, 7]}
    d = b | {"source": [1, 4, 7]}
    counts = [a | {"count": 20}, b | {"count": 10}, c | {"count": 7}, d | {"count": 7}]
    chosen = nominations({"cutoffs": {"0.5": {"legacy_identity": a, "joint_frequencies": counts}}}, {"sieve": a})
    assert chosen == {"sender_reassignment_mode": counts[0], "sender_reassignment_distinct": counts[2],
                      "sender_reassignment_receiver_only": counts[1]}
    scores = tuple(np.asarray([v]) for v in ([2, 1, 1, 0], [3, 2, 2, 0], [1, 2, 2, 3]))
    assert equivalent_choice([1, 2], [1, 3], [1, 2, 3, 4], scores)
    assert not equivalent_choice([1, 2], [1, 4], [1, 2, 3, 4], scores)


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def bind_journal(work):
    entries = []
    for path in sorted(work.glob("episode_*.json")):
        entries.append(json.dumps({"file": path.name, "sha256": sha(path)}) + "\n")
    journal = work / "closed.jsonl"
    journal.write_text("".join(entries), encoding="utf-8", newline="\n")
    path = work / "terminal.json"
    end = json.loads(path.read_bytes())
    rewrite(path, end | {"closed_journal_sha256": sha(journal)})


def add_raw_doses(work, cfg, old, writes):
    valid = {"requested_l2": 1., "realized_l2": 1., "direction_cosine": 1., "eligible": True}
    for path in sorted(work.glob("episode_*.json")):
        episode = json.loads(path.read_bytes())
        for i in cfg["source_components"]:
            slopes = episode["signed_phase2_edges_per_l2"][str(i)]
            for name, sign in (("positive", 1.), ("negative", -1.)):
                case = episode["cases"][f"component_{i}_{name}"]
                case.update(valid)
                case["response"].update(realized_residual_delta=(sign * writes[i]).tolist(),
                                        phase2_read=[sign * float(v) for v in slopes])
        sink = episode["phase2_terminal"]
        sink["registered_phase2_dose_fraction"] = old["phase2_readout_dose_fraction"]
        for rec in sink["phase2_write_responses"].values():
            case = valid if rec["signed_label_slopes_per_read_coefficient"] is not None else valid | {"eligible": False}
            rec.update(positive=case, negative=case)
        rewrite(path, episode)
    if (work / "closed.jsonl").exists():
        bind_journal(work)


def refuse(call, reason):
    try:
        call()
    except ValueError as error:
        if reason not in str(error):
            raise
    else:
        raise AssertionError("The measured sender audit accepted contradictory custody")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--work", type=Path, required=True)
    args = cli.parse_args()
    frozen = registered()
    if not __debug__ or json.loads(MANIFEST.read_bytes()) != frozen:
        raise ValueError("Run the source-frozen measured-instrument fixture")
    cfg = json.loads(selectors.CONFIG.read_bytes())
    old = json.loads((selectors.DELAY / "registration.json").read_bytes())
    gold, tokens = source_payload(args.source_dir)
    with lease(BUDGET) as (check, samples):
        import numpy as np

        big_integer_checks = heterogeneous_ties(cfg, np)
        physical_control_fixture(np)
        graph, _ = witness(cfg)
        writes = np.eye(len(cfg["receiver_components"]))
        args.work.mkdir(parents=True, exist_ok=False)
        tests = {mode: args.work / mode / backend for mode, backend in (
            ("closed", "nf4"), ("sparse", "bf16"), ("native_refusal", "nf4"))}
        for mode, work in tests.items():
            synthetic_journal(work, args.source_dir, graph, cfg, mode)
            add_raw_doses(work, cfg, old, writes)

        def check_case(key, fixture=True):
            return profiles(tests[key], gold, tokens, writes, cfg, check, np, fixture=fixture)

        outcome, private = check_case("closed")
        assert outcome["status"] == "frozen" and len(private) == outcome["complete_CAL_roots"] == 8
        assert outcome["qualified_signed_sender_slopes_checked"] == 112
        compiled = outcome["sender_assignment"]
        assert compiled["assignments"] == 5040
        assert all(sum(m["count"] for m in cell["joint_frequencies"]) == 5040
                   for cell in compiled["cutoffs"].values())
        masks = outcome["observed_masks"] | outcome["nominated_controls"]
        groups = outcome["physical_source_intervention_groups"]
        assert Counter(n for group in groups for n in group["selectors"]) == Counter(masks.keys())
        assert len(groups) < len(masks) and all(len(m["source"]) == 3 for m in groups)
        assert "sender_reassignment_distinct" in outcome["nominated_controls"]
        sparse, sparse_raw = check_case("sparse")
        assert sparse["complete_CAL_roots"] == 5 and "nominated_controls" not in sparse
        assert sparse["per_world_status"] == {"complete": 13, "unqualified_receiver_sink": 3}
        assert sum(len(pair) for pair in sparse_raw.values()) == 16
        refused, _ = check_case("native_refusal")
        assert refused["status"] == "native_gate_failed" and "nominated_controls" not in refused
        refuse(lambda: check_case("closed", fixture=False), "synthetic or malformed")
        path = tests["closed"] / "episode_0000.json"
        item = json.loads(path.read_bytes())
        item["signed_phase2_edges_per_l2"]["1"][0] += .5
        rewrite(path, item)
        refuse(lambda: check_case("closed"), "episode bytes")
        bind_journal(tests["closed"])
        refuse(lambda: check_case("closed"), "Signed delayed slopes disagree")
    assert frozen == registered() and "torch" not in sys.modules
    receipt = {"status": "passed", "source_manifest_sha256": sha(MANIFEST), "mode": "synthetic_fixture",
               "exact_assignments": compiled["assignments"], "cutoffs_checked": len(compiled["cutoffs"]),
               "arbitrary_precision_fraction_reference_checks": big_integer_checks,
               "complete_fixture_roots": outcome["complete_CAL_roots"],
               "sparse_fixture_roots": sparse["complete_CAL_roots"],
               "refused_fixture_roots": refused["complete_CAL_roots"],
               "signed_sender_slopes_checked": outcome["qualified_signed_sender_slopes_checked"],
               "distinct_fixture_source_masks": len(groups), "fixture_selector_names": len(masks),
               "receiver_alias_deduplication_checked": True, "distinct_sender_nomination_checked": True,
               "receiver_only_nomination_checked": True, "exact_boundary_tie_reconciliation_checked": True,
               "synthetic_refused_as_measured": True, "tampered_episode_hash_refused": True,
               "tampered_signed_slopes_refused": True, "native_forwards": 0,
               "formal_comparisons": 0, "sealed_test_accessed": False,
               "max_sampled_private_bytes": max(v["private_bytes"] for v in samples),
               "max_sampled_working_set_bytes": max(v["working_set_bytes"] for v in samples)}
    write_new(args.work / "verification.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
