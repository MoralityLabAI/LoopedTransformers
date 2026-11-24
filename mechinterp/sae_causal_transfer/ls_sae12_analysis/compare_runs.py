"""Compare LS-SAE12 full-stage runs of the same registered protocol on different platforms.

Standard library only. The first run is the reference (the registered Actions
run). For each other run, per pool position: clean screening digits agreement,
eligibility, selected site, clean ACT16 blanks correct; per family: membership;
per state present in both: the primary contrast D_SI at dose 0.02 and the V
of each arm; per family: the registered gate as each run recorded it.
Usage: compare_runs.py --run LABEL=DIR [--run LABEL=DIR ...] --out-csv F --out-json F
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

FAMILIES = ("active_a", "active_b")
PRIMARY = "0.02"


def load(d):
    d = Path(d)
    out = {"screen": json.loads((d / "screen.json").read_text())}
    for f in FAMILIES:
        p = d / f"{f}.json"
        out[f] = json.loads(p.read_text()) if p.exists() else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="LABEL=DIR; the first is the reference")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()
    runs = [(s.split("=", 1)[0], load(s.split("=", 1)[1])) for s in a.run]
    ref_label, ref = runs[0]
    rows, summary = [], {"reference": ref_label, "platforms": {}, "comparisons": {}}
    for label, run in runs:
        summary["platforms"][label] = run["screen"].get("platform")
    ref_rows = {r["pool_position"]: r for r in ref["screen"]["rows"]}
    for label, run in runs[1:]:
        cmp_rows = {r["pool_position"]: r for r in run["screen"]["rows"]}
        screen = {"positions": len(ref_rows), "digits_identical_all_16_steps": 0, "eligibility_agrees": 0,
                  "selected_site_agrees": 0, "eligible_ref": 0, "eligible_other": 0,
                  "first_divergent_step_histogram": {}}
        for pos, r in sorted(ref_rows.items()):
            o = cmp_rows[pos]
            steps = [x == y for x, y in zip(r["clean_predicted_digits_by_step"], o["clean_predicted_digits_by_step"])]
            first = next((i + 1 for i, s in enumerate(steps) if not s), None)
            screen["digits_identical_all_16_steps"] += first is None
            key = "none" if first is None else str(first)
            screen["first_divergent_step_histogram"][key] = screen["first_divergent_step_histogram"].get(key, 0) + 1
            screen["eligibility_agrees"] += r["eligible"] == o["eligible"]
            screen["selected_site_agrees"] += r["selected_site"] == o["selected_site"]
            screen["eligible_ref"] += r["eligible"]
            screen["eligible_other"] += o["eligible"]
            rows.append({"comparison": f"{ref_label}_vs_{label}", "level": "screen", "family": "",
                         "pool_position": pos, "source_file": r["source_file"],
                         "first_divergent_clean_step": "" if first is None else first,
                         "eligible_ref": r["eligible"], "eligible_other": o["eligible"],
                         "site_ref": r["selected_site"], "site_other": o["selected_site"],
                         "ACT16_correct_ref": r["clean_blank_correct_by_step"][15],
                         "ACT16_correct_other": o["clean_blank_correct_by_step"][15]})
        fams = {}
        for f in FAMILIES:
            rf, of = ref["screen"]["families"].get(f, []), run["screen"]["families"].get(f, [])
            info = {"members_ref": rf, "members_other": of, "same_members": rf == of}
            if ref[f] is not None and run[f] is not None:
                rs = {s["selected"]["pool_position"]: s for s in ref[f]["states"]}
                os_ = {s["selected"]["pool_position"]: s for s in run[f]["states"]}
                common = [pos for pos in rf if pos in os_]
                same_sign = 0
                for pos in common:
                    x = rs[pos]["selected"]["summary"][PRIMARY]
                    y = os_[pos]["selected"]["summary"][PRIMARY]
                    dx, dy = x["D_structural_minus_isotropic"], y["D_structural_minus_isotropic"]
                    sx = (dx > 0) - (dx < 0)
                    sy = (dy > 0) - (dy < 0)
                    same_sign += sx == sy
                    rows.append({"comparison": f"{ref_label}_vs_{label}", "level": "state", "family": f,
                                 "pool_position": pos, "source_file": rs[pos]["selected"]["source_file"],
                                 "site_ref": rs[pos]["selected"]["site"], "site_other": os_[pos]["selected"]["site"],
                                 "D_SI_ref": dx, "D_SI_other": dy,
                                 **{f"V_{arm}_ref": x["V"][arm] for arm in ("structural", "logit", "isotropic")},
                                 **{f"V_{arm}_other": y["V"][arm] for arm in ("structural", "logit", "isotropic")},
                                 "batch_clean_matches_screen_other": sum(os_[pos]["selected"]["same_forward_clean_matches_screen_steps"][4:])})
                g = lambda run_f: run_f["statistics"][f"selected@{PRIMARY}@D_structural_minus_isotropic"]
                info.update({"states_in_both": len(common), "D_SI_same_sign": same_sign,
                             "gate_ref": {k: g(ref[f]).get(k) for k in ("events_positive", "mean", "bootstrap95", "registered_success")},
                             "gate_other": {k: g(run[f]).get(k) for k in ("events_positive", "mean", "bootstrap95", "registered_success")}})
            fams[f] = info
        summary["comparisons"][label] = {"screen": screen, "families": fams}
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    a.out_json.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({k: v for k, v in summary["comparisons"].items()}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
