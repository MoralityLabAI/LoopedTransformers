"""Agreement across several LS-SAE12 full-stage runs of the same registered protocol.

Standard library only; complements the pairwise compare_runs.py. Per run:
eligible boards, boards solved exactly at ACT16, and how many of the 48 sites
(24 states x 2 site rules) have a batch-61 clean row equal to the batch-1
screening digits at all 12 continuation steps. Per pair of runs: boards whose
clean digits agree at all 16 steps, and states whose primary D_SI (dose 0.02,
observable-error site) keeps its sign, overall and without the given excluded
states; and, per variant, how many of each shared state's 60 edit rows have
the same VDD in both runs and the Spearman correlation of VDD across those
rows, split by whether the two runs selected the same site (and for the same
site, also without the excluded states). Across all runs:
the same sign counts for states present in every run, and one CSV row per
state with each run's site and D_SI.
Usage: compare_platforms.py --source source_48.json --run LABEL=DIR [--run ...]
       [--exclude A24 ...] --out-csv F --out-json F
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path

FAMILIES = ("active_a", "active_b")
PRIMARY = "0.02"


def sign(x):
    return (x > 0) - (x < 0)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def load(d):
    d = Path(d)
    return {"screen": json.loads((d / "screen.json").read_text()),
            **{f: json.loads((d / f"{f}.json").read_text()) for f in FAMILIES}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, help="LABEL=DIR")
    p.add_argument("--exclude", nargs="*", default=[], help="states as family letter + pool position, e.g. A24")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()
    holes = {r["pool_position"]: r["holes"] for r in json.loads(a.source.read_text())["pool"]}
    runs = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in a.run}
    labels = list(runs)
    screen = {k: {r["pool_position"]: r for r in run["screen"]["rows"]} for k, run in runs.items()}
    positions = sorted(holes)
    solved = {k: {pos for pos in positions if screen[k][pos]["clean_blank_correct_by_step"][15] == holes[pos]} for k in labels}
    dsi = {}
    selected = {}
    per_run = {}
    for k, run in runs.items():
        dsi[k] = {}
        selected[k] = {}
        sites = full = t_differs = 0
        for f in FAMILIES:
            for st in run[f]["states"]:
                for role in ("selected", "control"):
                    s = st[role]
                    sites += 1
                    full += sum(s["same_forward_clean_matches_screen_steps"][4:]) == 12
                sel = st["selected"]
                t_differs += sel["site_clean_T_same_forward"] != sel["site_clean_T_screen"]
                key = f"{f[-1].upper()}{sel['pool_position']}"
                dsi[k][key] = (sel["summary"][PRIMARY]["D_structural_minus_isotropic"], sel["site"])
                selected[k][key] = sel
        per_run[k] = {"platform": run["screen"].get("platform"),
                      "eligible": [pos for pos in positions if screen[k][pos]["eligible"]],
                      "clean_ACT16_exact": len(solved[k]),
                      "families": {f: run["screen"]["families"].get(f) for f in FAMILIES},
                      "gates": {f: {g: run[f]["statistics"][f"selected@{PRIMARY}@D_structural_minus_isotropic"].get(g)
                                    for g in ("events_positive", "mean", "bootstrap95", "registered_success")} for f in FAMILIES},
                      "sites_batch_clean_equals_screening_all_12": full, "sites": sites,
                      "selected_site_T_same_forward_differs_from_screening": t_differs}

    def identical(x, y):
        return {pos for pos in positions
                if screen[x][pos]["clean_predicted_digits_by_step"] == screen[y][pos]["clean_predicted_digits_by_step"]}

    def agreement(group):
        common = sorted(set.intersection(*(set(dsi[k]) for k in group)), key=lambda s: (s[0], int(s[1:])))
        same = [s for s in common if len({sign(dsi[k][s][0]) for k in group}) == 1]
        kept = [s for s in common if s not in a.exclude]
        same_kept = [s for s in kept if s in same]
        return {"states_in_all": len(common), "D_SI_same_sign": len(same),
                "states_without_excluded": len(kept), "D_SI_same_sign_without_excluded": len(same_kept),
                "same_sign_states": same,
                "site_differs": [s for s in common if len({dsi[k][s][1] for k in group}) > 1]}

    def per_variant(x, y):
        common = sorted(set(selected[x]) & set(selected[y]), key=lambda s: (s[0], int(s[1:])))
        states = []
        for s in common:
            sx, sy = selected[x][s], selected[y][s]
            if sx["variants"] != sy["variants"]:
                raise SystemExit(f"variant lists differ for {s}")
            edits = [i for i, v in enumerate(sx["variants"]) if v["arm"] != "clean"]
            vx = [sx["outcomes"][i]["vdd"] for i in edits]
            vy = [sy["outcomes"][i]["vdd"] for i in edits]
            states.append({"state": s, "same_site": sx["site"] == sy["site"], "edit_rows": len(edits),
                           "vdd_identical": sum(a == b for a, b in zip(vx, vy)), "spearman_vdd": spearman(vx, vy)})
        out = {}
        groups = {"same_site": lambda r: r["same_site"],
                  "same_site_without_excluded": lambda r: r["same_site"] and r["state"] not in a.exclude,
                  "different_site": lambda r: not r["same_site"]}
        for label, keep in groups.items():
            group = [r for r in states if keep(r)]
            rhos = [r["spearman_vdd"] for r in group if r["spearman_vdd"] is not None]
            out[label] = {"states": len(group), "edit_rows": sum(r["edit_rows"] for r in group),
                          "vdd_identical": sum(r["vdd_identical"] for r in group),
                          "median_spearman_vdd": median(rhos), "min_spearman_vdd": min(rhos) if rhos else None}
        out["by_state"] = states
        return out

    pairs = {}
    for x, y in itertools.combinations(labels, 2):
        same_digits = identical(x, y)
        pairs[f"{x}_vs_{y}"] = {
            "boards_clean_digits_identical_16_steps": len(same_digits),
            "identical_boards_solved_in_both": len(same_digits & solved[x] & solved[y]),
            "divergent_boards_solved_in_" + x: sorted((set(positions) - same_digits) & solved[x]),
            "per_variant": per_variant(x, y),
            **agreement((x, y))}
    all_identical = set(positions)
    for x, y in itertools.combinations(labels, 2):
        all_identical &= identical(x, y)
    summary = {"schema": "LS-SAE12/platform-agreement-v1", "runs": per_run, "excluded_states": a.exclude, "pairs": pairs,
               "all_runs": {"boards_clean_digits_identical_16_steps": sorted(all_identical), **agreement(tuple(labels))}}
    rows = []
    for s in sorted(set().union(*(set(v) for v in dsi.values())), key=lambda s: (s[0], int(s[1:]))):
        row = {"family": s[0], "pool_position": int(s[1:])}
        for k in labels:
            v = dsi[k].get(s)
            row[f"site_{k}"] = "" if v is None else v[1]
            row[f"D_SI_{k}"] = "" if v is None else v[0]
        present = [dsi[k][s][0] for k in labels if s in dsi[k]]
        row["in_all_runs"] = len(present) == len(labels)
        row["same_sign_all_runs"] = row["in_all_runs"] and len({sign(x) for x in present}) == 1
        rows.append(row)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    a.out_json.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"pairs": {k: {kk: ({g: m for g, m in vv.items() if g != "by_state"} if kk == "per_variant" else vv)
                                    for kk, vv in v.items() if kk != "same_sign_states"} for k, v in pairs.items()},
                      "all_runs": summary["all_runs"],
                      "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("platform", "eligible", "families")} for k, v in per_run.items()}},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
