"""Collate SEM-ECHO-01 checkpoint runs against the known LOOPSCOPE Q9 extrapolation labels.

Reads every ``metrics.json`` and ``pairs.jsonl`` under the given run directories, joins each
arm with ``loopscope_toy_q9/<arm>.json``, and writes ``summary.json`` and ``summary.md``.
The Q9 labels are retrospective; this summary fits no cutoff.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATA = ("both_within_training_depth", "donor_within_receiver_beyond_training_depth",
          "both_beyond_training_depth")
KINDS = ("full_donor", "cursor_swap", "edge_swap", "per_token_layernorm_shift",
         "random_cursor", "random_edges", "random_full")


def _open_pairs(run_dir):
    plain = run_dir / "pairs.jsonl"
    if plain.exists():
        return plain.open(encoding="utf-8")
    return gzip.open(run_dir / "pairs.jsonl.gz", "rt", encoding="utf-8")


def _rate(n, d):
    return None if d == 0 else n / d


def _average_ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y):
    """Spearman correlation with average ranks for ties."""
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def summarize_run(run_dir, q9_dir):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    receipt = metrics["receipt"]
    cfg = receipt["config"]
    arm = f"lam{cfg['lam']:g}_s{cfg['seed']}"
    q9 = json.loads((q9_dir / f"{arm}.json").read_text(encoding="utf-8"))
    if q9["checkpoint_sha256"] != receipt["checkpoint_sha256"]:
        raise RuntimeError(f"{arm}: checkpoint hash differs from the Q9 record")
    strata = {s: [0, 0, 0, 0] for s in STRATA}  # gated pairs, any-disagreement pairs, mismatched steps, steps
    kinds = defaultdict(lambda: [0, 0, 0, 0])   # pairs, disagree-with-receiver steps, oracle-correct steps, steps
    replay_pass, replay_n, replay_max_rel = 0, 0, 0.0
    item_pairs, item_mismatch = defaultdict(int), defaultdict(int)
    with _open_pairs(run_dir) as stream:
        for line in stream:
            row = json.loads(line)
            if row["record_type"] == "observational_pair" and row["endpoint_gate"]:
                item_pairs[row["item"]] += 1
                item_mismatch[row["item"]] += sum(row["step_mismatches"])
                s = strata[row["stratum"]]
                s[0] += 1
                s[1] += int(row["pair_any_disagreement"])
                s[2] += sum(row["step_mismatches"])
                s[3] += len(row["step_mismatches"])
            elif row["record_type"] == "intervention_pair":
                replay_n += 1
                ok = row["full_donor_replay_parity"]["passed"] and row["full_donor_logit_parity"]["passed"]
                replay_pass += int(ok)
                replay_max_rel = max(replay_max_rel,
                                     row["full_donor_replay_parity"]["max_item_relative_l2_error"])
                for item in row["interventions"]:
                    k = kinds[item["kind"]]
                    k[0] += 1
                    k[1] += item["disagreement_with_receiver_count"]
                    k[2] += item["oracle_correct_count"]
                    k[3] += item["oracle_comparisons"]
    disagreeing = [i for i in item_pairs if item_mismatch[i] > 0]
    cycle = {it["item"]: it["cycle_length"] for it in metrics["items"]}
    nontrivial = [i for i in item_pairs if cycle[i] > 1]
    agreeing = [i for i in item_pairs if item_mismatch[i] == 0]
    pooled = metrics["pooled_continuation"]
    cells = metrics["pooled_future_diagnostics"]["pair_comparison_cells"]
    depth = [r["oracle_accuracy"] for r in metrics["per_depth_oracle_accuracy"]]
    return {
        "arm": arm, "set": q9["set"], "checkpoint_sha256": receipt["checkpoint_sha256"],
        "q9_heldout": q9["heldout"], "q9_f1_over_f8": q9["f1_over_f8"],
        "native_parity_passed": receipt["native_parity"]["passed"]
        and receipt["native_parity"]["logit_parity"]["passed"],
        "acc_loop8": depth[7], "acc_loops_1_8": float(np.mean(depth[:8])),
        "acc_loops_9_32": float(np.mean(depth[8:])),
        "primary_item_mean": metrics["primary"]["mean"],
        "items_with_metric": metrics["primary"]["units_with_metric"],
        "items_total": metrics["primary"]["units_total"],
        "items_with_disagreement": len(disagreeing),
        "median_gated_pairs_in_disagreeing_items":
            float(np.median([item_pairs[i] for i in disagreeing])) if disagreeing else None,
        "median_gated_pairs_in_agreeing_items":
            float(np.median([n for i, n in item_pairs.items() if item_mismatch[i] == 0]))
            if len(disagreeing) < len(item_pairs) else None,
        "fixed_point_items_defined": sum(cycle[i] == 1 for i in item_pairs),
        "fixed_point_items_among_agreeing": sum(cycle[i] == 1 for i in agreeing),
        "fixed_point_items_among_disagreeing": sum(cycle[i] == 1 for i in disagreeing),
        "primary_item_mean_cycle_gt1":
            float(np.mean([item_mismatch[i] / (4 * item_pairs[i]) for i in nontrivial])) if nontrivial else None,
        "items_defined_cycle_gt1": len(nontrivial),
        "acc_loops_9_16": float(np.mean(depth[8:16])), "acc_loops_17_32": float(np.mean(depth[16:])),
        "eligible_pairs": pooled["eligible_pairs"], "gated_pairs": pooled["endpoint_gated_pairs"],
        "endpoint_gate_coverage": pooled["endpoint_gate_coverage"],
        "pair_any_disagreement_rate": pooled["pair_any_disagreement_rate"],
        "pooled_step_mismatch_rate": pooled["per_step_mismatch_rate"],
        "oracle_continuation_accuracy": pooled["oracle_continuation_accuracy"],
        "pair_cells": cells,
        "strata": {s: {"gated_pairs": v[0], "pair_any_disagreement_rate": _rate(v[1], v[0]),
                       "step_mismatch_rate": _rate(v[2], v[3])} for s, v in strata.items()},
        "replay": {"pairs": replay_n, "passed": replay_pass, "max_relative_state_l2": replay_max_rel},
        "interventions": {k: {"pairs": v[0], "disagreement_with_receiver": _rate(v[1], v[3]),
                              "oracle_accuracy": _rate(v[2], v[3])} for k, v in kinds.items()},
        "run_directory": str(run_dir.relative_to(run_dir.parents[1])),
    }


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def render(rows, corr):
    lines = ["# SEM-ECHO-01 on the Q4/Q5 checkpoints: retrospective development", "",
             "Evidence stage: **retrospective development**. All 24 checkpoints and their Q9 extrapolation "
             "labels were known before this run; no rule is fitted here.", "",
             "Run: `--items 64 --max-loops 32 --horizon 4 --max-pairs 16 --seed 940117 --device cuda --dtype float32`.",
             "Primary = unweighted item mean of the endpoint-gated per-step continuation disagreement (protocol section 5).", "",
             "## Per checkpoint, sorted by Q9 F1/F8", "",
             "acc columns are this run's own-head oracle accuracy averaged over the named loops "
             "(fixed-depth training supervises loop 8 only).", "",
             "| arm | set | Q9 held-out | F1/F8 | acc 1-8 | acc 9-16 | acc 17-32 | items defined | gated / eligible pairs | primary | any-disagree pairs | future oracle acc | agree, shared error | disagree, one correct | disagree, both wrong | replay pass |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c = r["pair_cells"]
        lines.append(
            f"| {r['arm']} | {r['set']} | {r['q9_heldout']:.3f} | {r['q9_f1_over_f8']:.4f} | "
            f"{r['acc_loops_1_8']:.3f} | {r['acc_loops_9_16']:.3f} | {r['acc_loops_17_32']:.3f} | {r['items_with_metric']}/{r['items_total']} | "
            f"{r['gated_pairs']} / {r['eligible_pairs']} | {_fmt(r['primary_item_mean'], 4)} | "
            f"{_fmt(r['pair_any_disagreement_rate'])} | {_fmt(r['oracle_continuation_accuracy'])} | "
            f"{c['agree_shared_error']} | {c['disagree_one_side_correct']} | {c['disagree_both_sides_error']} | "
            f"{r['replay']['passed']}/{r['replay']['pairs']} |")
    lines += ["", "## Disagreement by depth stratum (gated pairs, pooled step mismatch rate)", "",
              "| arm | within 1-8 | straddles 8 | beyond 8 |", "|---|---:|---:|---:|"]
    for r in rows:
        cells = [f"{_fmt(r['strata'][s]['step_mismatch_rate'])} (n={r['strata'][s]['gated_pairs']})" for s in STRATA]
        lines.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    lines += ["", "## Capped interventions (up to 16 gated pairs per checkpoint)", "",
              "Step-level disagreement of the patched future with the unpatched receiver future, and its oracle accuracy.", "",
              "| arm | " + " | ".join(KINDS) + " |", "|---|" + "---:|" * len(KINDS)]
    for r in rows:
        iv = r["interventions"]
        lines.append(f"| {r['arm']} | " + " | ".join(
            f"{_fmt(iv[k]['disagreement_with_receiver'], 2)} / {_fmt(iv[k]['oracle_accuracy'], 2)}" if k in iv else "n/a"
            for k in KINDS) + " |")
    lines += ["", "## Where the item-mean disagreement comes from", "",
              "The primary weights items equally. A fixed-point start (cycle length 1) gives C(28,2) = 378 gated pairs "
              "when the model holds it; items with few gated pairs carry the disagreement.", "",
              "| arm | items with disagreement / defined | median gated pairs, disagreeing | median gated pairs, agreeing | fixed points among agreeing | fixed points among disagreeing | primary, cycle > 1 items only (n) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["items_with_disagreement"]:
            lines.append(f"| {r['arm']} | {r['items_with_disagreement']}/{r['items_with_metric']} | "
                         f"{_fmt(r['median_gated_pairs_in_disagreeing_items'], 1)} | "
                         f"{_fmt(r['median_gated_pairs_in_agreeing_items'], 1)} | "
                         f"{r['fixed_point_items_among_agreeing']}/{r['items_with_metric'] - r['items_with_disagreement']} | "
                         f"{r['fixed_point_items_among_disagreeing']}/{r['items_with_disagreement']} | "
                         f"{_fmt(r['primary_item_mean_cycle_gt1'], 4)} ({r['items_defined_cycle_gt1']}) |")
    lines += ["", f"## Association with the known labels (descriptive, n = {corr['arms']} arms)", "",
              "Spearman uses average ranks; arms with zero disagreement are tied."]
    lines += [f"- {k}: {v}" for k, v in corr.items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--q9-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = [summarize_run(d, args.q9_dir) for root in args.runs for d in sorted(root.iterdir())
            if (d / "metrics.json").exists()]
    rows.sort(key=lambda r: r["q9_f1_over_f8"])
    defined = [r for r in rows if r["primary_item_mean"] is not None]
    fails = [r for r in rows if r["q9_heldout"] < 0.9]
    corr = {
        "arms": len(rows),
        "arms_with_defined_primary": len(defined),
        "arms_with_any_gated_disagreement": sum(r["pair_any_disagreement_rate"] not in (None, 0.0) for r in rows),
        "extrapolation_failures_(Q9_heldout<0.9)": len(fails),
        "failures_with_any_gated_disagreement": sum(r["pair_any_disagreement_rate"] not in (None, 0.0) for r in fails),
        "spearman(primary, Q9_heldout)": round(_spearman([r["primary_item_mean"] for r in defined],
                                                         [r["q9_heldout"] for r in defined]), 3) if len(defined) > 2 else None,
        "spearman(primary, F1/F8)": round(_spearman([r["primary_item_mean"] for r in defined],
                                                    [r["q9_f1_over_f8"] for r in defined]), 3) if len(defined) > 2 else None,
        "spearman(acc_loops_9_16, Q9_heldout)": round(_spearman([r["acc_loops_9_16"] for r in rows],
                                                                [r["q9_heldout"] for r in rows]), 3),
        "spearman(primary, 1 - acc_loops_9_32)": round(_spearman([r["primary_item_mean"] for r in defined],
                                                                 [1 - r["acc_loops_9_32"] for r in defined]), 3),
        "spearman(primary_cycle_gt1, F1/F8)": round(_spearman([r["primary_item_mean_cycle_gt1"] for r in defined],
                                                              [r["q9_f1_over_f8"] for r in defined]), 3),
        "nonfailures_with_any_gated_disagreement": [r["arm"] for r in rows if r["q9_heldout"] >= 0.9
                                                    and r["pair_any_disagreement_rate"] not in (None, 0.0)],
        "all_native_parity_passed": all(r["native_parity_passed"] for r in rows),
        "replays_passed": f"{sum(r['replay']['passed'] for r in rows)}/{sum(r['replay']['pairs'] for r in rows)}",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({"association": corr, "arms": rows}, indent=2) + "\n")
    (args.out / "summary.md").write_text(render(rows, corr))
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
