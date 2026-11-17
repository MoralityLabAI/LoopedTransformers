"""LS-SAE11 post-hoc evidence tables from the archived Actions outputs.

Standard library only. Reads the model-written pilot/reference/hard JSON and the
source-only receipt, recomputes every Sudoku score it reports from the saved
predicted digits and exact solver targets, replays the registered ACT6 gate
with the registered bootstrap procedure, and writes per-puzzle CSV tables plus
a summary JSON. Nothing here feeds back into the registered gate: thresholds,
doses, seeds and puzzles are read from the frozen protocol constants.

Usage:
  python -m mechinterp.sae_causal_transfer.native_withinrun_v1.analyze \
      --evidence DIR_WITH_pilot.json_reference.json_hard.json_source_20.json \
      --out mechinterp/sae_causal_transfer/results/ls_sae11
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import random
from pathlib import Path

HORIZONS = ("ACT3", "ACT6", "ACT12")
RADII = ("0.20", "2.00", "8.00")
ARMS = ("state_power", "logit_power", "sae", "random")
SIGNS = ("plus", "minus")
STAGES = (("pilot", "engineering_pilot"),
          ("reference", "retrospective_original_reference"),
          ("hard", "prospective_hard_test"))
HARD_FAMILIES = ("test_champagne_a", "test_champagne_b")
GATE_HORIZON = "ACT6"
GATE_DOSE = "8.00"
BOOT_DRAWS = 2000

UNITS = [[9 * r + c for c in range(9)] for r in range(9)]
UNITS += [[9 * r + c for r in range(9)] for c in range(9)]
UNITS += [[9 * (br + dr) + bc + dc for dr in range(3) for dc in range(3)]
          for br in (0, 3, 6) for bc in (0, 3, 6)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rescore(preds, puzzle, target, clean_preds):
    """Recompute the registered digit score from saved predictions."""
    missing = [i for i, v in enumerate(puzzle) if v == 0]
    given = [i for i, v in enumerate(puzzle) if v != 0]
    conflicts = sum(len(u) - len({preds[i] for i in u}) for u in UNITS)
    return {
        "blank_correct": sum(preds[i] == target[i] for i in missing),
        "blank_total": len(missing),
        "board_exact": preds == target,
        "sudoku_valid_and_clue_consistent": conflicts == 0 and all(preds[i] == puzzle[i] for i in given),
        "unit_conflicts": conflicts,
        "clue_disagreements": sum(preds[i] != puzzle[i] for i in given),
        "blank_prediction_changes": sum(preds[i] != clean_preds[i] for i in missing),
        "blank_wrong_to_correct": sum(clean_preds[i] != target[i] and preds[i] == target[i] for i in missing),
        "blank_correct_to_wrong": sum(clean_preds[i] == target[i] and preds[i] != target[i] for i in missing),
    }


def check_score(saved, puzzle, target, clean_preds, where):
    got = rescore(saved["predicted_digits_81"], puzzle, target, clean_preds)
    for key, val in got.items():
        if saved[key] != val:
            raise ValueError(f"{where}: saved {key}={saved[key]} but recomputed {val}")


def registered_bootstrap(diffs, family):
    """Protocol: seed 202611170+len(family), 2000 paired draws, sorted indices 50 and 1949."""
    rng = random.Random(202611170 + len(family))
    n = len(diffs)
    draws = []
    for _ in range(BOOT_DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(diffs[j] for j in idx) / n)
    draws.sort()
    return draws[50], draws[1949]


def sym_gain(scores, horizon, arm, dose):
    """Sign-symmetrized blank-correct gain; ineligible arm counts as 0 (registered)."""
    row = scores[horizon]
    signed = row["arms"].get(arm, {}).get(dose)
    if signed is None:
        return 0.0
    clean = row["clean"]["blank_correct"]
    return 0.5 * (signed["plus"]["blank_correct"] + signed["minus"]["blank_correct"]) - clean


def wrong_set(preds, puzzle, target):
    return {i for i, v in enumerate(puzzle) if v == 0 and preds[i] != target[i]}


def roster_for(stage, data, source, original_rosters):
    if stage == "reference":
        return {g: original_rosters[g] for g in data["families"]}
    return {g: source["rosters"][g] for g in data["families"]}


def analyze_stage(stage, label, data, rosters, writers, summary):
    stage_summary = {"label": label, "families": {}}
    for family, fam in data["families"].items():
        roster = rosters[family]
        cases = fam["per_puzzle"]
        if len(cases) != len(roster):
            raise ValueError(f"{stage}/{family}: denominator changed")
        fam_rows = []
        for idx, (src, case) in enumerate(zip(roster, cases)):
            puzzle, target = src["x"], src["target"]
            if case["source_seed"] != src["seed"]:
                raise ValueError(f"{stage}/{family}/{idx}: seed mismatch")
            scores = case["native_task_scores"]
            transport = case["same_run_transport"]
            status = case["source_direction_status"]
            clean_preds = {h: scores[h]["clean"]["predicted_digits_81"] for h in HORIZONS}
            base = {
                "stage": stage, "stage_label": label, "family": family, "index": idx,
                "seed": src["seed"], "source_file": case.get("source_file") or "",
                "donor_family": case["donor_family"], "donor_index": case["donor_index"],
                "site": case["source_site"],
            }
            # --- clean accuracy, per puzzle ---
            acc = dict(base)
            acc["holes"] = scores["ACT6"]["clean"]["blank_total"]
            acc["solver_search_nodes"] = src.get("solver_search_nodes", "")
            vis = src.get("visible_structural_difficulty") or {}
            acc["singles_deduced_cells"] = vis.get("deduced_missing_cells", "")
            acc["singles_solvable"] = vis.get("propagation_solved", "")
            acc["source_ACT1_zH_norm"] = case["source_ACT1_zH_norm"]
            acc["eligible_arms"] = "|".join(a for a in ARMS if status[a])
            for h in HORIZONS:
                cl = scores[h]["clean"]
                check_score(cl, puzzle, target, clean_preds[h], f"{stage}/{family}/{idx}/{h}/clean")
                acc[f"{h}_blank_correct"] = cl["blank_correct"]
                acc[f"{h}_blank_accuracy"] = round(cl["blank_correct"] / cl["blank_total"], 6)
                acc[f"{h}_board_exact"] = cl["board_exact"]
                acc[f"{h}_valid"] = cl["sudoku_valid_and_clue_consistent"]
                acc[f"{h}_unit_conflicts"] = cl["unit_conflicts"]
                acc[f"{h}_clue_disagreements"] = cl["clue_disagreements"]
            w3 = wrong_set(clean_preds["ACT3"], puzzle, target)
            w6 = wrong_set(clean_preds["ACT6"], puzzle, target)
            w12 = wrong_set(clean_preds["ACT12"], puzzle, target)
            acc["clean_recovered_ACT3_to_ACT6"] = len(w3 - w6)
            acc["clean_broken_ACT3_to_ACT6"] = len(w6 - w3)
            acc["clean_recovered_ACT6_to_ACT12"] = len(w6 - w12)
            acc["clean_broken_ACT6_to_ACT12"] = len(w12 - w6)
            acc["clean_ACT6_inexact"] = scores["ACT6"]["clean"]["blank_correct"] < scores["ACT6"]["clean"]["blank_total"]
            writers["accuracy"].writerow(acc)

            # --- signed edits, transport, certificates ---
            for h in HORIZONS:
                cl = scores[h]["clean"]
                for arm in ARMS:
                    if not status[arm]:
                        continue
                    for dose in RADII:
                        for sign in SIGNS:
                            sc = scores[h]["arms"][arm][dose][sign]
                            check_score(sc, puzzle, target, clean_preds[h],
                                        f"{stage}/{family}/{idx}/{h}/{arm}/{dose}/{sign}")
                            g = transport[h]["arms"][arm][dose][sign]
                            row = dict(base)
                            row.update({
                                "horizon": h, "arm": arm, "dose": dose, "sign": sign,
                                "clean_blank_correct": cl["blank_correct"],
                                "blank_correct": sc["blank_correct"],
                                "delta_blank_correct": sc["blank_correct"] - cl["blank_correct"],
                                "blank_flips": sc["blank_prediction_changes"],
                                "repairs_wrong_to_correct": sc["blank_wrong_to_correct"],
                                "corruptions_correct_to_wrong": sc["blank_correct_to_wrong"],
                                "net_repair": sc["blank_wrong_to_correct"] - sc["blank_correct_to_wrong"],
                                "all81_changes": sc["all_81_prediction_changes"],
                                "site_prediction_changed": sc["goal_prediction_changed"],
                                "board_exact": sc["board_exact"],
                                "valid": sc["sudoku_valid_and_clue_consistent"],
                                "clue_disagreements": sc["clue_disagreements"],
                            })
                            for k in ("h_gain", "l_gain", "h_global_l2", "l_global_l2",
                                      "h_source_site_l2", "h_non_site_l2", "h_board_l2",
                                      "h_blank_l2", "h_given_l2", "h_readout_l2", "h_null_l2",
                                      "readout_fraction", "centered_logit_fro",
                                      "centered_readout_reconstruction_error",
                                      "minimum_clean_blank_winner_margin", "all81_absolute_epsilon",
                                      "strict_global_margin_certificate",
                                      "strict_all_blank_pairwise_certificate",
                                      "strict_pairwise_certified_blank_count",
                                      "actual_blank_digit_flips"):
                                row[k] = g[k]
                            for k, v in g["classification"].items():
                                row["class_" + k] = v
                            if g["actual_blank_digit_flips"] != sc["blank_prediction_changes"]:
                                raise ValueError("certificate flip count disagrees with digit score")
                            writers["edits"].writerow(row)

            # --- paired structural versus random, every horizon and dose ---
            for h in HORIZONS:
                for dose in RADII:
                    row = dict(base)
                    row.update({"horizon": h, "dose": dose,
                                "clean_blank_correct": scores[h]["clean"]["blank_correct"],
                                "blank_total": scores[h]["clean"]["blank_total"]})
                    for arm in ARMS:
                        row[f"C_{arm}"] = sym_gain(scores, h, arm, dose)
                        signed = scores[h]["arms"].get(arm, {}).get(dose)
                        if signed is None:
                            row[f"flips_{arm}"] = 0.0
                            row[f"net_repair_{arm}"] = 0.0
                            row[f"hgain_{arm}"] = ""
                            row[f"rowspace_{arm}"] = ""
                        else:
                            row[f"flips_{arm}"] = 0.5 * sum(signed[s]["blank_prediction_changes"] for s in SIGNS)
                            row[f"net_repair_{arm}"] = 0.5 * sum(
                                signed[s]["blank_wrong_to_correct"] - signed[s]["blank_correct_to_wrong"] for s in SIGNS)
                            tg = transport[h]["arms"][arm][dose]
                            row[f"hgain_{arm}"] = 0.5 * sum(tg[s]["h_gain"] for s in SIGNS)
                            row[f"rowspace_{arm}"] = 0.5 * sum(tg[s]["readout_fraction"] for s in SIGNS)
                    row["D_state_minus_random"] = row["C_state_power"] - row["C_random"]
                    row["D_logit_minus_random"] = row["C_logit_power"] - row["C_random"]
                    row["D_sae_minus_random"] = row["C_sae"] - row["C_random"]
                    row["flip_diff_state_minus_random"] = row["flips_state_power"] - row["flips_random"]
                    row["net_repair_diff_state_minus_random"] = row["net_repair_state_power"] - row["net_repair_random"]
                    writers["paired"].writerow(row)
                    fam_rows.append(row)

            # --- delayed error recovery for every signed edit ---
            for arm in ARMS:
                if not status[arm]:
                    continue
                for dose in RADII:
                    for sign in SIGNS:
                        p = {h: scores[h]["arms"][arm][dose][sign]["predicted_digits_81"] for h in HORIZONS}
                        e3, e6, e12 = (wrong_set(p[h], puzzle, target) for h in HORIZONS)
                        row = dict(base)
                        row.update({
                            "arm": arm, "dose": dose, "sign": sign,
                            "edited_errors_ACT3": len(e3), "edited_errors_ACT6": len(e6),
                            "edited_errors_ACT12": len(e12),
                            "clean_errors_ACT3": len(w3), "clean_errors_ACT6": len(w6),
                            "clean_errors_ACT12": len(w12),
                            "edited_recovered_ACT3_to_ACT6": len(e3 - e6),
                            "edited_recovered_ACT6_to_ACT12": len(e6 - e12),
                            "edited_broken_ACT6_to_ACT12": len(e12 - e6),
                            "ACT6_corruptions": len(e6 - w6),
                            "ACT6_corruptions_recovered_by_ACT12": len((e6 - w6) - e12),
                            "ACT6_repairs": len(w6 - e6),
                            "ACT6_repairs_persisting_ACT12": len((w6 - e6) - e12),
                            "ACT12_new_errors_vs_clean": len(e12 - w12),
                            "ACT12_errors_removed_vs_clean": len(w12 - e12),
                            "edited_ACT12_board_exact": not e12,
                            "clean_ACT12_board_exact": not w12,
                        })
                        writers["recovery"].writerow(row)

            if stage == "reference":
                drift = case["cross_run_original_drift"]
                for h in HORIZONS:
                    rec = drift[h]
                    arm_max = 0.0
                    arm_changes = 0
                    n = 0
                    for arm, by_dose in rec["arms"].items():
                        for dose, by_sign in by_dose.items():
                            for sign, val in by_sign.items():
                                arm_max = max(arm_max, val["max_absolute_logit_drift"])
                                arm_changes += val["all81_argmax_changes"]
                                n += 1
                    row = dict(base)
                    row.update({"horizon": h,
                                "clean_max_abs_logit_drift": rec["clean"]["max_absolute_logit_drift"],
                                "clean_argmax_changes": rec["clean"]["all81_argmax_changes"],
                                "edited_variants": n,
                                "edited_max_abs_logit_drift": arm_max,
                                "edited_total_argmax_changes": arm_changes})
                    writers["drift"].writerow(row)

        # --- family statistics ---
        fs = {"n_puzzles": len(cases)}
        for h in HORIZONS:
            fs[f"clean_{h}_inexact"] = sum(
                c["native_task_scores"][h]["clean"]["blank_correct"] < c["native_task_scores"][h]["clean"]["blank_total"]
                for c in cases)
            fs[f"clean_{h}_board_exact"] = sum(c["native_task_scores"][h]["clean"]["board_exact"] for c in cases)
            fs[f"clean_{h}_blank_correct_total"] = sum(c["native_task_scores"][h]["clean"]["blank_correct"] for c in cases)
            fs[f"clean_{h}_blank_total"] = sum(c["native_task_scores"][h]["clean"]["blank_total"] for c in cases)
        tests = {}
        for h in HORIZONS:
            for dose in RADII:
                rows = [r for r in fam_rows if r["horizon"] == h and r["dose"] == dose]
                for contrast in ("D_state_minus_random", "D_logit_minus_random", "D_sae_minus_random"):
                    diffs = [float(r[contrast]) for r in rows]
                    lo, hi = registered_bootstrap(diffs, family)
                    key = f"{contrast}@{h}@{dose}"
                    tests[key] = {
                        "diffs": diffs, "events_ge_1": sum(d >= 1 for d in diffs),
                        "mean": sum(diffs) / len(diffs), "boot95": [lo, hi],
                        "clean_inexact": fs[f"clean_{h}_inexact"],
                        "rule_pass": bool(fs[f"clean_{h}_inexact"] >= 4 and sum(d >= 1 for d in diffs) >= 6 and lo > 0),
                        "registered_primary": contrast == "D_state_minus_random" and h == GATE_HORIZON and dose == GATE_DOSE,
                    }
        fs["contrasts"] = tests
        if stage == "hard":
            primary = tests[f"D_state_minus_random@{GATE_HORIZON}@{GATE_DOSE}"]
            saved = fam["statistics"]
            same = (saved["paired_state_minus_random_blank_correctness_gain"] == primary["diffs"]
                    and saved["events_D_at_least_one"] == primary["events_ge_1"]
                    and saved["puzzle_bootstrap95"] == primary["boot95"]
                    and saved["clean_ACT6_inexact"] == primary["clean_inexact"]
                    and saved["registered_success"] == primary["rule_pass"])
            if not same:
                raise ValueError(f"{family}: independent gate recomputation disagrees with saved statistics")
            fs["registered_gate"] = {"family": family, **saved, "recomputed_identical": True}
            gw = writers["gates"]
            for key, t in tests.items():
                contrast, h, dose = key.split("@")
                gw.writerow({"stage": stage, "family": family, "contrast": contrast, "horizon": h,
                             "dose": dose, "registered_primary": t["registered_primary"],
                             "n_puzzles": len(t["diffs"]), "clean_inexact": t["clean_inexact"],
                             "events_D_ge_1": t["events_ge_1"], "mean_D": t["mean"],
                             "boot95_lo": t["boot95"][0], "boot95_hi": t["boot95"][1],
                             "rule_pass": t["rule_pass"],
                             "D_per_puzzle": " ".join(f"{d:g}" for d in t["diffs"])})
        stage_summary["families"][family] = fs
    summary["stages"][stage] = stage_summary


FIELDS = {
    "accuracy": None, "edits": None, "paired": None, "recovery": None, "drift": None,
    "gates": ["stage", "family", "contrast", "horizon", "dose", "registered_primary", "n_puzzles",
              "clean_inexact", "events_D_ge_1", "mean_D", "boot95_lo", "boot95_hi", "rule_pass",
              "D_per_puzzle"],
}


class LazyWriter:
    """csv.DictWriter that takes its header from the first row."""

    def __init__(self, path, fields=None):
        self.path = path
        self.fields = fields
        self.fh = None
        self.w = None
        self.rows = 0

    def writerow(self, row):
        if self.w is None:
            self.fh = open(self.path, "w", newline="", encoding="utf-8")
            self.w = csv.DictWriter(self.fh, fieldnames=self.fields or list(row), lineterminator="\n")
            self.w.writeheader()
        out = {}
        for k, v in row.items():
            out[k] = repr(v) if isinstance(v, float) and not math.isfinite(v) else v
        self.w.writerow(out)
        self.rows += 1

    def close(self):
        if self.fh:
            self.fh.close()


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])) if n else float("nan")


def aggregate_edits(edits_csv, out_csv):
    """Per stage x horizon x arm x dose: transport, rowspace, certificates, flips."""
    groups = {}
    with open(edits_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            groups.setdefault((r["stage"], r["horizon"], r["arm"], r["dose"]), []).append(r)
    order = {s: i for i, (s, _) in enumerate(STAGES)}
    rows = []
    for (stage, h, arm, dose), rs in sorted(groups.items(), key=lambda kv: (
            order[kv[0][0]], HORIZONS.index(kv[0][1]), ARMS.index(kv[0][2]), kv[0][3])):
        f = lambda k: [float(r[k]) for r in rs]
        t = lambda k: sum(r[k] == "True" for r in rs)
        rows.append({
            "stage": stage, "horizon": h, "arm": arm, "dose": dose, "n_signed_edits": len(rs),
            "median_h_gain": median(f("h_gain")), "max_h_gain": max(f("h_gain")),
            "median_l_gain": median(f("l_gain")),
            "median_readout_fraction": median(f("readout_fraction")),
            "median_site_share_of_h": median([float(r["h_source_site_l2"]) / float(r["h_global_l2"])
                                              if float(r["h_global_l2"]) > 0 else 0.0 for r in rs]),
            "global_margin_certified": t("strict_global_margin_certificate"),
            "pairwise_all_blank_certified": t("strict_all_blank_pairwise_certificate"),
            "edits_with_zero_flips": sum(int(r["blank_flips"]) == 0 for r in rs),
            "total_blank_flips": sum(int(r["blank_flips"]) for r in rs),
            "total_repairs": sum(int(r["repairs_wrong_to_correct"]) for r in rs),
            "total_corruptions": sum(int(r["corruptions_correct_to_wrong"]) for r in rs),
            "mean_delta_blank_correct": sum(f("delta_blank_correct")) / len(rs),
            "edited_board_exact": t("board_exact"),
            "class_contracted": t("class_contracted"),
            "class_persistent_output_silent": t("class_persistent_output_silent"),
            "class_readout_small": t("class_readout_small"),
            "max_centered_readout_reconstruction_error": max(f("centered_readout_reconstruction_error")),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", type=Path, required=True,
                   help="directory with pilot.json reference.json hard.json source_20.json and original/full.json")
    p.add_argument("--original", type=Path, required=True, help="LS-SAE08 full.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    names = {"accuracy": "ls_sae11_per_puzzle_clean_accuracy.csv",
             "edits": "ls_sae11_per_puzzle_signed_edits.csv",
             "paired": "ls_sae11_per_puzzle_structural_vs_random.csv",
             "recovery": "ls_sae11_per_puzzle_delayed_recovery.csv",
             "drift": "ls_sae11_reference_cross_run_drift.csv",
             "gates": "ls_sae11_family_gates.csv"}
    writers = {k: LazyWriter(a.out / v, FIELDS[k]) for k, v in names.items()}
    source_path = a.evidence / "source_20.json"
    source = json.loads(source_path.read_text())
    original = json.loads(a.original.read_text())
    original_rosters = {g: original["families"][g]["source_roster"] for g in ("test_52", "test_54")}
    del original
    gc.collect()
    summary = {"schema": "LS-SAE11/posthoc-evidence-tables-v1",
               "inputs": {"source_20.json": sha256(source_path), "original_full.json": sha256(a.original)},
               "stages": {}}
    for stage, label in STAGES:
        path = a.evidence / f"{stage}.json"
        summary["inputs"][f"{stage}.json"] = sha256(path)
        data = json.loads(path.read_text())
        analyze_stage(stage, label, data, roster_for(stage, data, source, original_rosters), writers, summary)
        summary["stages"][stage]["readout_centered_rank"] = data["readout_centered_rank"]
        summary["stages"][stage]["readout_singular_values"] = data["readout_nonzero_singular_values"]
        summary["stages"][stage]["registered_hard_scientific_success"] = data["registered_hard_scientific_success"]
        del data
        gc.collect()
    for w in writers.values():
        w.close()
    summary["tables"] = {names[k]: writers[k].rows for k in names}
    agg_name = "ls_sae11_aggregate_transport_certificates.csv"
    summary["tables"][agg_name] = aggregate_edits(a.out / names["edits"], a.out / agg_name)
    hard = summary["stages"]["hard"]["families"]
    summary["registered_outcome"] = {
        "both_families_pass": all(hard[f]["registered_gate"]["registered_success"] for f in HARD_FAMILIES),
        "per_family": {f: hard[f]["registered_gate"]["registered_success"] for f in HARD_FAMILIES},
    }
    (a.out / "ls_sae11_analysis_summary.json").write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"tables": summary["tables"], "registered_outcome": summary["registered_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
