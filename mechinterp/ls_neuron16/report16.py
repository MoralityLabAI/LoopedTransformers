"""Post-run descriptive reporting for LS16; outside the frozen search source.

Read complete, audited receipts and the official statistics without running any
model, optimizer, verifier, resampling procedure, or additional hypothesis test.
Write report_data.json and one export figure as figure16.png / figure16.svg.
The explicitly labeled ancestry timing check was added after seeing outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

ARMS = ("greedy", "beam")
VERDICTS = ("CERTIFIED", "POINT_VIOLATION", "UNRESOLVED")
SEEDS = tuple(range(493, 517))
TARGET = 3 / 32
EPS = 1e-12
BUDGET_KEYS = ("expansions", "profile_slots", "ls_calls", "lp_calls")


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def value_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True).encode())


def distribution(values):
    x = np.asarray(list(values), dtype=float)
    if not len(x):
        return {"count": 0, "sum": 0, "min": None, "q1": None,
                "median": None, "q3": None, "max": None, "mean": None}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite descriptive value")
    return {"count": int(len(x)), "sum": float(x.sum()), "min": float(x.min()),
            "q1": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "q3": float(np.quantile(x, .75)), "max": float(x.max()),
            "mean": float(x.mean())}


def verdict_counts(rows, key):
    counts = Counter(row[key]["verdict"] for row in rows)
    if set(counts) - set(VERDICTS):
        raise ValueError(f"Unexpected exact verdict: {counts}")
    return {verdict: counts[verdict] for verdict in VERDICTS}


def new_repair(row):
    return row["initial"]["verdict"] == "POINT_VIOLATION" and row["final"]["verdict"] == "CERTIFIED"


def load_inputs(root):
    hashes = {}

    def read(relative):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Complete LS16 reporting input is required: {path}")
        content = path.read_bytes()
        hashes[str(relative)] = sha_bytes(content)
        return json.loads(content)

    certificates = read("certificates.json")
    audit = read("audit_results.json")
    statistics = read("evidence/results/statistics.json")
    registration = read("evidence/registration.json")
    manifest = read("run_manifest.json")
    freeze = read("freeze_receipt.json")
    environment = read("environment.json")
    assert audit["status"] == "PASS" and audit["n_units"] == 24
    assert audit["certificate_sha256"] == hashes["certificates.json"]
    assert statistics["certificate_source_sha256"] == hashes["certificates.json"]
    assert statistics["registration_sha256"] == hashes["evidence/registration.json"]
    assert registration["prespecified_seeds"] == manifest["seeds"] == list(SEEDS)
    assert manifest["arms"] == list(ARMS) and manifest["development"] is False
    assert manifest["all_searches_completed_utc"] and manifest["freeze"] == freeze
    assert len(statistics["comparisons"]) == 1
    assert statistics["study_evidence_class"] == "PILOT_ONLY"
    assert certificates["positive_control"]["verdict"] == "CERTIFIED"
    assert len(certificates["rows"]) == 48
    rows = {(row["seed"], row["arm"]): row for row in certificates["rows"]}
    assert set(rows) == {(s, a) for s in SEEDS for a in ARMS}
    locked = {}
    lock_content = (root / "lock.sha256").read_bytes()
    hashes["lock.sha256"] = sha_bytes(lock_content)
    assert hashes["lock.sha256"] == freeze["lock_sha256"]
    for line in lock_content.decode().splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = sha_bytes((root / relative).read_bytes())
        assert expected == actual, f"Frozen file changed: {relative}"
        locked[relative] = actual
    assert "report16.py" not in locked
    assert len(locked) == freeze["locked_files"]
    common, raw = {}, {}
    for seed in SEEDS:
        initial = read(f"initial/{seed}.json")
        assert initial["seed"] == seed
        common[seed] = read(f"common/{seed}.json")
        for arm in ARMS:
            row = rows[seed, arm]
            assert row["initial_sha256"] == hashes[f"initial/{seed}.json"]
            assert row["common_sha256"] == hashes[f"common/{seed}.json"]
            raw[seed, arm] = read(f"results/{arm}-{seed}.json")
            assert row["receipt_sha256"] == hashes[f"results/{arm}-{seed}.json"]
            assert raw[seed, arm]["final"]["weights"] == row["weights"]
        assert rows[seed, "greedy"]["initial"] == rows[seed, "beam"]["initial"]
        assert rows[seed, "greedy"]["initial_eval"] == rows[seed, "beam"]["initial_eval"]
        assert rows[seed, "greedy"]["bootstrap_best_bound"] == rows[seed, "beam"]["bootstrap_best_bound"]
    return dict(certificates=certificates, audit=audit, statistics=statistics,
                registration=registration, manifest=manifest, freeze=freeze,
                environment=environment, rows=rows, common=common, raw=raw,
                hashes=hashes, locked=locked)


def lp_records(expansion):
    return [fit["secondary"] for anchor in expansion["anchors"] for fit in anchor["fits"]]


def raw_diagnostics(common, raw):
    nodes = {"s0": common["initial_fit"]["state"]}
    expansions = [common["bootstrap"]] + [e for r in raw["rounds"] for e in r["expansions"]]
    for expansion in expansions:
        for anchor in expansion["anchors"]:
            for fit in anchor["fits"]:
                nodes[fit["first"]["id"]] = fit["first"]
                nodes[fit["second"]["id"]] = fit["second"]
    parents = [nodes[e["parent_id"]] for e in expansions]
    codes = [tuple(node["codes"]) for node in parents]
    records = [record for expansion in expansions for record in lp_records(expansion)]
    return {"expanded_codebook_count": len(set(codes)),
            "repeated_expanded_codebook_count": len(codes) - len(set(codes)),
            "lp_iterations": sum(record["nit"] for record in records),
            "lp_status_counts": dict(Counter(str(record["status"]) for record in records)),
            "lp_max_iterations_in_one_call": max(record["nit"] for record in records)}


def ancestry_timing(row, common):
    """Post-hoc ordering check, with numerical crossings kept distinct from proof."""
    path = row["final_ancestry"]
    passing = [i for i, node in enumerate(path) if node["bound"] <= TARGET]
    worsening = [i for i, node in enumerate(path) if node["delta_parent_bound"] is not None and node["delta_parent_bound"] > EPS]
    first = passing[0] if passing else None
    bootstrap_rank = {identity: rank + 1 for rank, identity in enumerate(common["retained"])}
    branches = [node for node in path if node["id"] in bootstrap_rank]
    branch = branches[-1] if branches else None
    return {"first_numerical_pass_id": None if first is None else path[first]["id"],
            "first_numerical_pass_depth": first,
            "first_numerical_pass_bound": None if first is None else path[first]["bound"],
            "worsening_edges_before_or_at_first_numerical_pass": None if first is None else sum(i <= first for i in worsening),
            "worsening_edges_after_first_numerical_pass": None if first is None else sum(i > first for i in worsening),
            "bootstrap_branch_id": None if branch is None else branch["id"],
            "bootstrap_branch_rank": None if branch is None else bootstrap_rank[branch["id"]],
            "bootstrap_branch_bound": None if branch is None else branch["bound"],
            "worsening_edge_details": [path[i] for i in worsening],
            "scope": "Ordering along the final retained ancestry using numerical bounds; intermediate nodes were not exactly certified."}


def make_report(inputs):
    rows, common, raw = inputs["rows"], inputs["common"], inputs["raw"]
    diagnostic = {(s, a): raw_diagnostics(common[s], raw[s, a]) for s in SEEDS for a in ARMS}
    posthoc = {(s, a): ancestry_timing(rows[s, a], common[s]) for s in SEEDS for a in ARMS}
    paired, groups = [], {"both_repair": [], "candidate_only": [], "control_only": [], "neither_repair": []}
    for seed in SEEDS:
        g, b = (new_repair(rows[seed, arm]) for arm in ARMS)
        category = "both_repair" if g and b else "candidate_only" if b else "control_only" if g else "neither_repair"
        groups[category].append(seed)
        pair = {"seed": seed, "initial_verdict": rows[seed, "greedy"]["initial"]["verdict"],
                "paired_primary_category": category, "beam_minus_greedy": int(b) - int(g)}
        for arm in ARMS:
            row, extra = rows[seed, arm], diagnostic[seed, arm]
            assert extra["expanded_codebook_count"] == row["expanded_codebook_count"]
            assert extra["repeated_expanded_codebook_count"] == row["repeated_expanded_codebook_count"]
            assert row["expanded_codebook_count"] + row["repeated_expanded_codebook_count"] == 37
            path = row["final_ancestry"]
            assert row["final_path_depth"] == len(path) - 1
            pair[arm] = {
                "final_verdict": row["final"]["verdict"], "new_repair": new_repair(row),
                "final_bound": row["scores"]["bound"], "numerical_bound_pass": row["scores"]["bound"] <= TARGET,
                "bootstrap_best_bound": row["bootstrap_best_bound"], "bootstrap_bound_pass": row["bootstrap_bound_pass"],
                "final_path_depth": row["final_path_depth"],
                "final_ancestry_worsening_edges": row["final_ancestry_worsening_edges"],
                "comparison_retained_worsening_edges": len(row["retained_worsening_edges"]),
                "bootstrap_retained_worsening_edges": len(row["bootstrap_retained_worsening_edges"]),
                "last_best_improvement_round": max([item["round"] for item in row["best_updates"]], default=0),
                "retained_codebooks": row["retained_codebook_count_all_selections"],
                "retained_nodes_all_selections": row["retained_node_count_all_selections"],
                "evaluated_codebooks": row["unique_codebooks_visited"],
                "evaluated_decoded_weights": row["unique_decoded_weights_visited"],
                "any_visited_numerical_bound_pass": row["any_visited_bound_pass"],
                "final_rollout_128": row["final_eval"]["128"],
                "ls_nfev_attributed": row["ls_nfev"], "search_seconds_excluding_common": row["seconds"],
                "common_seconds": common[seed]["seconds"], **extra,
            }
        paired.append(pair)

    arm_summaries = {}
    for arm in ARMS:
        items = [rows[s, arm] for s in SEEDS]
        data = [pair[arm] for pair in paired]
        certified = [r for r in items if r["final"]["verdict"] == "CERTIFIED"]
        bounds = [r["scores"]["bound"] for r in items]
        totals = {key: sum(r[key] for r in items) for key in BUDGET_KEYS}
        summary = {
            "n_units": len(items), "new_repairs": sum(map(new_repair, items)),
            "new_repair_seeds": [r["seed"] for r in items if new_repair(r)],
            "initial_verdicts": verdict_counts(items, "initial"), "final_verdicts": verdict_counts(items, "final"),
            "initially_unresolved_later_certified_seeds": [r["seed"] for r in items if r["initial"]["verdict"] == "UNRESOLVED" and r["final"]["verdict"] == "CERTIFIED"],
            "lost_certificate_seeds": [r["seed"] for r in items if r["initial"]["verdict"] == "CERTIFIED" and r["final"]["verdict"] != "CERTIFIED"],
            "final_bound": distribution(bounds), "numerical_bound_passes": sum(v <= TARGET for v in bounds),
            "bound_certificate_disagreement_seeds": [r["seed"] for r in items if (r["scores"]["bound"] <= TARGET) != (r["final"]["verdict"] == "CERTIFIED")],
            "failed_final_seeds_with_any_visited_bound_pass": [r["seed"] for r in items if r["final"]["verdict"] != "CERTIFIED" and r["any_visited_bound_pass"]],
            "expanded_codebooks": distribution(r["expanded_codebook_count"] for r in items),
            "repeated_codebook_expansions": distribution(r["repeated_expanded_codebook_count"] for r in items),
            "retained_codebooks": distribution(r["retained_codebook_count_all_selections"] for r in items),
            "evaluated_codebooks": distribution(r["unique_codebooks_visited"] for r in items),
            "codebook_count_scope": "Unique within each seed and arm; sums combine these within-seed counts and include the attributed bootstrap expansion.",
            "final_path_depth": distribution(r["final_path_depth"] for r in items),
            "last_best_improvement_round": distribution(d["last_best_improvement_round"] for d in data),
            "last_best_improvement_round_scope": "Comparison rounds 1..12; zero means no improvement after the common bootstrap.",
            "final_ancestries_with_worsening_edge": sum(r["final_ancestry_worsening_edges"] > 0 for r in items),
            "final_ancestry_worsening_edge_seeds": [r["seed"] for r in items if r["final_ancestry_worsening_edges"] > 0],
            "certified_final_ancestries_with_worsening_edge": sum(r["final_ancestry_worsening_edges"] > 0 for r in certified),
            "certified_final_ancestry_worsening_edge_seeds": [r["seed"] for r in certified if r["final_ancestry_worsening_edges"] > 0],
            "comparison_retained_worsening_edges": sum(len(r["retained_worsening_edges"]) for r in items),
            "bootstrap_retained_worsening_edges": sum(len(r["bootstrap_retained_worsening_edges"]) for r in items),
            "final_rollout_128": {key: distribution(r["final_eval"]["128"][key] for r in items) for key in ("accuracy", "all_prefix")},
            "rollout_scope": "Descriptive unweighted mean over the 24 seeds; both arms use the inherited evaluation generator.",
            "attributed_budget_totals": totals,
            "ls_nfev_attributed": sum(r["ls_nfev"] for r in items),
            "lp_iterations_attributed": sum(d["lp_iterations"] for d in data),
            "lp_failures_attributed": sum(r["lp_calls"] - r["lp_successes"] for r in items),
            "lp_max_iterations_in_one_call": max(d["lp_max_iterations_in_one_call"] for d in data),
            "lp_status_counts_attributed": dict(sum((Counter(d["lp_status_counts"]) for d in data), Counter())),
            "ls_to_lp_bound_improvements_attributed": sum(r["secondary_improvements"] for r in items),
            "ls_to_lp_numerical_crossings_attributed": sum(r["secondary_bound_crossings"] for r in items),
            "worker_elapsed_seconds_excluding_common": distribution(r["seconds"] for r in items),
            "worker_elapsed_seconds_with_common_attributed": distribution(rows[s, arm]["seconds"] + common[s]["seconds"] for s in SEEDS),
            "distinct_final_weight_sets": len({value_sha(r["weights"]) for r in items}),
        }
        audited = next(s for s in inputs["audit"]["summary"] if s["arm"] == arm)
        assert summary["new_repairs"] == audited["new_certificates"]
        assert summary["final_bound"]["median"] == audited["median_bound"]
        summary["lp_audited_max_residuals"] = {key: value for key, value in audited.items() if key.startswith("lp_max_")}
        arm_summaries[arm] = summary

    comparison = inputs["statistics"]["comparisons"][0]
    assert groups["candidate_only"] == comparison["candidate_only_seeds"]
    assert groups["control_only"] == comparison["baseline_only_seeds"]
    net = len(groups["candidate_only"]) - len(groups["control_only"])
    requirements = inputs["registration"]["operational_gate"]
    operational = (arm_summaries["beam"]["new_repairs"] >= requirements["candidate_new_certificates_min"]
                   and net >= requirements["candidate_minus_control_new_min"])
    assert operational == inputs["statistics"]["primary_operational_gate_passed"]
    common_budgets = {key: sum(common[s]["budgets"][key] for s in SEEDS) for key in BUDGET_KEYS}
    physical = {key: sum(arm_summaries[a]["attributed_budget_totals"][key] for a in ARMS) - common_budgets[key] for key in BUDGET_KEYS}
    assert physical == inputs["audit"]["actual_compute_with_common_once"] == inputs["manifest"]["actual_compute_budget"]
    common_lp = [r for s in SEEDS for r in lp_records(common[s]["bootstrap"])]
    common_seconds = sum(common[s]["seconds"] for s in SEEDS)
    exclusive_seconds = sum(rows[s, a]["seconds"] for s in SEEDS for a in ARMS)
    started = datetime.fromisoformat(inputs["manifest"]["started_utc"])
    completed = datetime.fromisoformat(inputs["manifest"]["all_searches_completed_utc"])
    weights = defaultdict(list)
    for s in SEEDS:
        for a in ARMS:
            weights[value_sha(rows[s, a]["weights"])].append({"seed": s, "arm": a, "verdict": rows[s, a]["final"]["verdict"]})
    duplicate_groups = [{"weight_sha256": key, "endpoints": value} for key, value in weights.items() if len(value) > 1]
    posthoc_summary = {}
    for arm in ARMS:
        certified_seeds = [s for s in SEEDS if rows[s, arm]["final"]["verdict"] == "CERTIFIED"]
        posthoc_summary[arm] = {
            "certified_final_paths": len(certified_seeds),
            "certified_paths_with_worsening_before_first_numerical_pass": sum(posthoc[s, arm]["worsening_edges_before_or_at_first_numerical_pass"] > 0 for s in certified_seeds),
            "certified_paths_with_worsening_after_first_numerical_pass": sum(posthoc[s, arm]["worsening_edges_after_first_numerical_pass"] > 0 for s in certified_seeds),
        }
    report = {
        "schema_version": 1, "experiment": "LS-NEURON-16", "evidence_class": "PILOT_ONLY",
        "n_units": 24, "n_final_endpoints": 48, "assigned_seeds": list(SEEDS), "arms": list(ARMS),
        "scope": {"unit": "Independently seeded perturbation", "primary_endpoint": "Initial POINT_VIOLATION and final CERTIFIED",
                  "exact_certificate": "Stored-weight ideal real recurrence, radius 1/8, disturbance 1/32",
                  "numerical_bound_threshold": TARGET, "numerical_threshold_exact": "3/32",
                  "inference": "Official statistics are copied unchanged; this reporting script runs no additional hypothesis tests."},
        "provenance": {"report_created_utc": datetime.now(timezone.utc).isoformat(),
                       "report_code_role": "Post-run descriptive reporting code outside the frozen search; written after outcomes were available.",
                       "report16_sha256": sha_bytes(Path(__file__).read_bytes()),
                       "frozen_source_commit": inputs["freeze"]["commit_sha"],
                       "frozen_lock_sha256": inputs["hashes"]["lock.sha256"],
                       "frozen_file_hashes": inputs["locked"], "input_sha256": inputs["hashes"],
                       "runtime_environment": inputs["environment"]},
        "paired_table": paired,
        "paired_primary": {"candidate": "beam", "control": "greedy", "counts": {k: len(v) for k, v in groups.items()},
                           "seeds": groups, "net_new_repairs": net, "observed_effect": net / 24,
                           "operational_gate": {"requirements": requirements, "observed_beam_repairs": arm_summaries["beam"]["new_repairs"],
                                                "observed_net_repairs": net, "passed": operational}},
        "initial_models": {"verdicts": verdict_counts([rows[s, "greedy"] for s in SEEDS], "initial"),
                           "rollout_128": {key: distribution(rows[s, "greedy"]["initial_eval"]["128"][key] for s in SEEDS) for key in ("accuracy", "all_prefix")}},
        "shared_bootstrap": {"numerical_bound_pass_seeds": [s for s in SEEDS if rows[s, "greedy"]["bootstrap_bound_pass"]],
                             "best_bound": distribution(rows[s, "greedy"]["bootstrap_best_bound"] for s in SEEDS),
                             "exact_verification_performed_on_bootstrap": False},
        "arm_summaries": arm_summaries,
        "endpoint_weights": {"distinct_final_weight_sets": len(weights),
                             "distinct_certified_weight_sets": inputs["audit"]["distinct_certified_weight_sets"],
                             "duplicate_groups": duplicate_groups},
        "physical_compute": {"actual_budget_common_once": physical, "common_budget_once": common_budgets,
                             "least_squares_nfev_common_once": sum(rows[s, a]["ls_nfev"] for s in SEEDS for a in ARMS) - sum(common[s]["ls_nfev"] for s in SEEDS),
                             "lp_iterations_common_once": sum(d["lp_iterations"] for d in diagnostic.values()) - sum(r["nit"] for r in common_lp),
                             "shared_common_worker_elapsed_seconds": common_seconds,
                             "exclusive_search_worker_elapsed_seconds": exclusive_seconds,
                             "summed_worker_elapsed_seconds_common_once": common_seconds + exclusive_seconds,
                             "observed_search_stage_wall_seconds": (completed - started).total_seconds(),
                             "workers": inputs["manifest"]["workers"],
                             "timing_scope": "Worker elapsed durations can overlap; their sum is neither process CPU time nor stage wall time. Manifest wall time includes initialization and search, and excludes exact audit and reporting."},
        "official_statistics": inputs["statistics"],
        "posthoc_interpretation_check": {
            "status": "POST_HOC_DESCRIPTIVE_CHECK",
            "purpose": "Check whether worsening edges in successful final ancestries occurred before or after their first numerical bound pass.",
            "scope": "This ordering check was added after inspecting outcomes. It adds no test and does not establish necessity of any edge. Intermediate numerical crossings remain distinct from the exact terminal certificates.",
            "summary": posthoc_summary,
            "all_seed_arm_details": [{"seed": s, "arm": a, **posthoc[s, a]} for s in SEEDS for a in ARMS],
            "discordant_full_ancestries": [{"seed": s, "arm": a, "path": rows[s, a]["final_ancestry"], **posthoc[s, a]} for s in groups["candidate_only"] + groups["control_only"] for a in ARMS],
        },
    }
    return report


def make_figure(report, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15.8, 9.5), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.1, 3.2),
                          left=.105, right=.975, top=.79, bottom=.145, hspace=.54)
    top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    data, n = report["paired_table"], report["n_units"]
    x = np.arange(n)
    summary, pair = report["arm_summaries"], report["paired_primary"]
    colors = {"greedy": "#315D9D", "beam": "#D47B20"}
    code = {"POINT_VIOLATION": 0, "CERTIFIED": 1, "UNRESOLVED": 2}
    letters = {"POINT_VIOLATION": "P", "CERTIFIED": "C", "UNRESOLVED": "U"}
    matrix = np.asarray([[code[row[arm]["final_verdict"]] for row in data] for arm in ARMS])
    top.imshow(matrix, cmap=ListedColormap(["#E9EDF2", "#217B6E", "#F1D081"]), vmin=0, vmax=2, aspect="auto")
    for iy, arm in enumerate(ARMS):
        for ix, row in enumerate(data):
            verdict = row[arm]["final_verdict"]
            top.text(ix, iy, letters[verdict], ha="center", va="center", fontsize=10.5,
                     color="white" if verdict == "CERTIFIED" else "#374151", fontweight="bold")
    top.set_yticks([0, 1], [f"Greedy\n{summary['greedy']['final_verdicts']['CERTIFIED']}/24 certified",
                           f"Beam\n{summary['beam']['final_verdicts']['CERTIFIED']}/24 certified"])
    top.set_xticks(x, [str(s) for s in report["assigned_seeds"]], fontsize=9)
    top.xaxis.tick_top()
    top.tick_params(axis="both", which="both", length=0, pad=7)
    top.set_xticks(np.arange(-.5, n, 1), minor=True)
    top.set_yticks([-.5, .5, 1.5], minor=True)
    top.grid(which="minor", color="white", linewidth=2)
    for spine in top.spines.values():
        spine.set_visible(False)
    top.set_title("A  Exact terminal verdicts · every assigned seed", loc="left", fontsize=13, fontweight="bold", pad=31)
    top.legend(handles=[Patch(facecolor="#217B6E", label="C  Certified"),
                        Patch(facecolor="#E9EDF2", label="P  Point violation"),
                        Patch(facecolor="#F1D081", label="U  Unresolved")],
               loc="lower left", bbox_to_anchor=(0, -.58), ncol=3, frameon=False,
               borderaxespad=0, handlelength=1.4, columnspacing=2.2, fontsize=10)
    g = np.asarray([row["greedy"]["final_bound"] for row in data])
    b = np.asarray([row["beam"]["final_bound"] for row in data])
    for ix in x:
        bottom.plot([ix - .13, ix + .13], [g[ix], b[ix]], color="#BBC3CF", linewidth=1.05, zorder=1)
    bottom.scatter(x - .13, g, marker="o", s=40, color=colors["greedy"], edgecolor="white", linewidth=.65, label="Greedy", zorder=3)
    bottom.scatter(x + .13, b, marker="D", s=36, color=colors["beam"], edgecolor="white", linewidth=.65, label="Beam", zorder=3)
    if np.all(np.r_[g, b] > 0):
        bottom.set_yscale("log")
        bottom.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        bottom.yaxis.set_minor_formatter(NullFormatter())
        bottom.set_ylim(min(g.min(), b.min()) / 2.5, max(g.max(), b.max(), TARGET) * 3)
        bottom.set_ylabel("Final numerical bound · log scale", labelpad=9)
    else:
        assert np.all(np.r_[g, b] >= 0)
        bottom.set_yscale("symlog", linthresh=1e-10)
        bottom.set_ylabel("Final numerical bound · symlog (linear below 10⁻¹⁰)", labelpad=9)
    bottom.axhline(TARGET, color="#51465B", linestyle=(0, (5, 3)), linewidth=1.3, zorder=2)
    bottom.text(.988, TARGET, "3/32 = 0.09375", transform=bottom.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=10, color="#51465B",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    bottom.set_xticks(x, [str(s) for s in report["assigned_seeds"]], fontsize=9)
    bottom.set_xlim(-.6, n - .4)
    bottom.set_xlabel("Perturbation seed", labelpad=8)
    bottom.grid(axis="y", which="major", color="#E4E8ED", linewidth=.8)
    bottom.spines["left"].set_color("#B8C0CB")
    bottom.spines["bottom"].set_color("#B8C0CB")
    bottom.set_title("B  Floating-point interval envelope used for search", loc="left", fontsize=13, fontweight="bold", pad=11)
    bottom.legend(loc="upper right", frameon=True, framealpha=.96, edgecolor="#E2E6EC", ncol=2, fontsize=10)
    fig.text(.105, .961, "LS16 · Beam breadth under the same fitting budget", fontsize=21, fontweight="bold", color="#172536")
    fig.text(.105, .925, f"New exact repairs: beam {summary['beam']['new_repairs']}/24; greedy {summary['greedy']['new_repairs']}/24.  "
             f"Beam-only {len(pair['seeds']['candidate_only'])}; greedy-only {len(pair['seeds']['control_only'])}.", fontsize=12.5, color="#374151")
    gate = pair["operational_gate"]
    required = gate["requirements"]
    fig.text(.105, .891, f"Prespecified operational gate: ≥{required['candidate_new_certificates_min']}/24 beam repairs and ≥{required['candidate_minus_control_new_min']} net repairs.  "
             f"Observed: {gate['observed_beam_repairs']}/24 and {gate['observed_net_repairs']:+d}.  "
             f"Gate {'passed' if gate['passed'] else 'failed'} · PILOT_ONLY", fontsize=11.5, color="#5C4050")
    fig.text(.105, .065, "Panel A uses the rational verifier. Panel B shows the numerical search score; each connector pairs the same seed.", fontsize=10.5, color="#475569")
    fig.text(.105, .039, "A new repair requires an initial point violation and a final certificate. Certificates cover the stored-weight ideal real recurrence.", fontsize=10, color="#475569")
    fig.text(.105, .014, "37 expansions · 223 LS calls · 222 LP calls per arm and seed  |  "
             f"Frozen search {report['provenance']['frozen_source_commit'][:12]}  |  Post-run descriptive figure", fontsize=9.5, color="#64748B")
    paths = [output / "figure16.png", output / "figure16.svg"]
    fig.savefig(paths[0], dpi=220, metadata={"Title": "LS16 paired exact endpoints and numerical bounds", "Description": "Post-run descriptive reporting from all 24 paired seeds; PILOT_ONLY"})
    fig.savefig(paths[1], metadata={"Title": "LS16 paired exact endpoints and numerical bounds", "Description": "Post-run descriptive reporting from all 24 paired seeds; PILOT_ONLY"})
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, help="Output directory; defaults to --root")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.out or root).resolve()
    inputs = load_inputs(root)
    report = make_report(inputs)
    output.mkdir(parents=True, exist_ok=True)
    figures = make_figure(report, output)
    report["artifacts"] = {path.name: {"sha256": sha_bytes(path.read_bytes()), "bytes": path.stat().st_size} for path in figures}
    target = output / "report_data.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report": str(target), "figures": [str(p) for p in figures],
                      "paired_counts": report["paired_primary"]["counts"],
                      "operational_gate": report["paired_primary"]["operational_gate"],
                      "posthoc_ancestry_timing": report["posthoc_interpretation_check"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
