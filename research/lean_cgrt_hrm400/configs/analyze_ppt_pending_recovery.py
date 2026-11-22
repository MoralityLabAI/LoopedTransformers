"""Audit recovery transfer outcomes, calibrated edits, and layout-level effects."""

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import mean

from configs.freeze_ppt_pending_recovery import ARMS, EDITS, SEEDS, SCHEMA
from configs.run_ppt_pending_recovery import frozen_cohort, recovery_event, verified
from ppt_gym.data import canonical, digest, sha256_file


def shard(path, cohort_sha, hashes):
    data = json.loads(gzip.decompress(path.read_bytes()))
    if data["cohort_sha256"] != cohort_sha:
        raise ValueError("a transfer worker used a different registry")
    hashes[path.name] = sha256_file(path)
    return data


def proof_receipt(episode):
    success = verified(episode)
    event = recovery_event(episode)
    return {
        "certified": success, "choice": event["selected_index"], "checks": episode["tactic_checks"],
        "script": episode["accepted_script"], "certificate": episode["certificate"],
        "kernel": episode["live_kernel"], "replay": episode["native_replay"],
        "cpu_seconds": episode["search_cpu_seconds"], "macs": episode["model_macs"],
    }


def audit_edits(trials, cluster, seed):
    if tuple(sorted(trials)) != tuple(sorted(EDITS)):
        raise ValueError("incomplete intervention family")
    direction = 1 if int(digest([SCHEMA, cluster, seed, "signed-margin"])[0], 16) % 2 == 0 else -1
    baseline = None
    for mode in EDITS:
        edits = trials[mode]["edit"]
        if mode == "pristine":
            if edits:
                raise ValueError("pristine replay contained an intervention")
            continue
        change, = edits
        before, after = change["before"], change["after"]
        if len(before) != 2 or len(after) != 2 or (baseline is not None and baseline != before):
            raise ValueError("the interventions did not act on the same history signal")
        baseline = before
        size = math.hypot(*before) / math.sqrt(2)
        if mode == "ablate-history":
            expected = [0., 0.]
        elif mode == "common-shift":
            expected = [x - size for x in before]
        else:
            expected = [before[0] + direction * size, before[1] - direction * size]
        if any(not math.isclose(y, z, rel_tol=2e-6, abs_tol=2e-7) for y, z in zip(after, expected, strict=True)):
            raise ValueError("scored intervention differs from the registered equal-norm edit")


def outcomes(rows, groups, winners, condition):
    pairs, root_accuracy, certified = {}, 0, 0
    for cluster, cases in groups.items():
        roots = [rows[case.root][condition] for case in cases]
        solved = sum(row["certified"] for row in roots)
        root_accuracy += sum(row["choice"] == winners[case.root] for row, case in zip(roots, cases, strict=True))
        certified += solved
        pairs[cluster] = int(solved == 2)
    return {"complete_pairs": sum(pairs.values()), "proofs": certified, "correct_decisions": root_accuracy}, pairs


def fsi(comparison, pairs, reg):
    from spiral_ln.evaluation_finite_sample import evaluate_manifest

    baseline = comparison if comparison in ARMS else comparison.removeprefix("history:")
    candidate = ARMS[0] if comparison in ARMS else "pristine"
    rows = [{"case_id": f"{seed}:{cluster}", "cluster_id": cluster,
             "candidate": conditions[candidate][cluster], "baseline": conditions[baseline][cluster]}
            for seed, conditions in pairs.items() for cluster in conditions[candidate]]
    result = evaluate_manifest({
        "schema_version": 1, "experiment_id": f"ppt-pending-recovery-{comparison}",
        "target_population": "pending-obligation development layouts with shared subtree templates",
        "evaluation_mode": "sampled", "alpha": "0.05", "planned_comparisons": reg["planned_comparisons"],
        "minimum_effect": str(reg["minimum_meaningful_effect"]),
        "hypothesis_preregistered": True, "sealed_readout": False, "clusters_independent": False,
        "exchangeable_pair_labels": False, "cases": rows,
    })
    raw = [row["candidate"] - row["baseline"] for row in rows]
    result.update(seed_layout_wins=raw.count(1), seed_layout_losses=raw.count(-1),
                  seed_layout_ties=raw.count(0))
    return result


def audit(cohort_path, run):
    reg, groups = frozen_cohort(cohort_path)
    cohort_sha = sha256_file(cohort_path)
    hashes = {}
    reference = shard(run / "witness.json.gz", cohort_sha, hashes)
    winners, compact_witness = {}, {}
    if set(reference["rows"]) != set(groups):
        raise ValueError("the witness omitted a development layout")
    for cluster, group in groups.items():
        examples = reference["rows"][cluster]
        if [item["root"] for item in examples] != [case.root for case in group]:
            raise ValueError("native witness changed the paired roots")
        compact_witness[cluster] = []
        for item in examples:
            forced = [proof_receipt(episode) for episode in item["forced"]]
            winner = item["winner"]
            if ([trial["choice"] for trial in forced] != [0, 1] or not forced[winner]["certified"]
                    or forced[1 - winner]["certified"] or forced[winner]["checks"] != 7 + winner):
                raise ValueError("recorded viability differs from the native witness")
            winners[item["root"]] = winner
            compact_witness[cluster].append({"root": item["root"], "winner": winner, "forced": forced})
        if sum(row["winner"] for row in examples) != 1:
            raise ValueError("the witness paired two worlds with the same winning decision")

    counts, pair_rows, compact, checkpoint_weights = {}, {}, {}, {}
    protocol = None
    for seed in SEEDS:
        label = str(seed)
        counts[label], pair_rows[label], compact[label], checkpoint_weights[label] = {}, {}, {}, {}
        for arm in ARMS:
            data = shard(run / f"{seed}-{arm}.json.gz", cohort_sha, hashes)
            current_protocol = (data["budget"], data["verifier"])
            protocol = current_protocol if protocol is None else protocol
            if (data["session_sha256"] != reg["source_session_sha256"][label][arm]
                    or data["arm"] != arm or data["seed"] != seed
                    or protocol != current_protocol or set(data["rows"]) != set(winners)):
                raise ValueError("unmatched frozen model, budget, verifier, or native case")
            checkpoint_weights[label][arm] = data["weights_sha256"]
            roots = {}
            for cluster, cases in groups.items():
                for case in cases:
                    modes = data["rows"][case.root]
                    if arm == ARMS[0]:
                        audit_edits(modes, cluster, seed)
                    elif set(modes) != {"pristine"} or modes["pristine"]["edit"]:
                        raise ValueError("a control received an unregistered edit")
                    roots[case.root] = {}
                    for mode, row in modes.items():
                        proof = proof_receipt(row["episode"])
                        if (proof["certified"] != (proof["choice"] == winners[case.root])
                                or proof["macs"] > data["budget"]["max_macs"]
                                or proof["cpu_seconds"] > data["budget"]["cpu_seconds"]):
                            raise ValueError("frozen policy departed from the native viability witness")
                        roots[case.root][mode] = {**proof, "edit": row["edit"]}
            compact[label][arm] = roots
            for mode in next(iter(roots.values())):
                number, paired = outcomes(roots, groups, winners, mode)
                name = arm if mode == "pristine" else mode
                counts[label][name], pair_rows[label][name] = number, paired
                if arm == ARMS[0] and mode == "pristine":
                    pair_rows[label]["pristine"] = paired
    comparisons = (*ARMS[1:], *(f"history:{mode}" for mode in EDITS[1:]))
    if len(comparisons) != reg["planned_comparisons"]:
        raise ValueError("comparison family differs from the frozen registry")
    effects = {comparison: fsi(comparison, pair_rows, reg) for comparison in comparisons}
    mechanism = {}
    for mode in EDITS[1:]:
        per_layout = [mean(rows["pristine"][layout] - rows[mode][layout] for rows in pair_rows.values())
                      for layout in groups]
        mechanism[mode] = {"mean_pair_drop": mean(per_layout), "nonzero_layouts": sum(x != 0 for x in per_layout),
                           "maximum_absolute_layout_effect": max(map(abs, per_layout))}
    summary = {
        "registration": reg, "cohort_sha256": cohort_sha, "counts": counts, "comparisons": effects,
        "mechanism": mechanism, "shard_sha256": hashes, "frozen_weight_sha256": checkpoint_weights,
        "budget": protocol[0], "verifier": protocol[1], "memory": json.loads((run / "resources.json").read_bytes()),
    }
    return summary, {"summary": summary, "native_witness": compact_witness, "policy_receipts": compact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "run", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary, evidence = audit(args.cohort, args.run)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for seed, arms in summary["counts"].items():
        print(seed, {arm: (row["complete_pairs"], row["proofs"]) for arm, row in arms.items()})
    print("mechanism", summary["mechanism"])


if __name__ == "__main__":
    main()
