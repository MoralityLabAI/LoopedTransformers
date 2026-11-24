"""LS-SAE12 evidence tables from the registered run's receipts.

Standard library only. Lives outside active_reasoning_v1/ so committing it
does not trigger the registered workflow. Reads screen.json and the two family
files, writes per-puzzle screening, per-state, per-variant and gate CSVs and a
summary JSON. The registered verdict is read from the family statistics and
re-derived here from the per-state contrasts with the registered bootstrap.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

FAMILIES = ("active_a", "active_b")
BOOT = {"active_a": 2026111813, "active_b": 2026111814}
ARMS = ("structural", "logit", "isotropic")
PRIMARY = "0.02"


def write_csv(path, rows):
    if not rows:
        Path(path).write_bytes(b"")
        return 0
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else None


def gate(diffs, family):
    n = len(diffs)
    if n != 12:
        return {"n": n, "complete": False, "registered_success": None}
    rng = random.Random(BOOT[family])
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    ev = sum(d > 0 for d in diffs)
    return {"n": n, "complete": True, "events_positive": ev, "mean": sum(diffs) / n,
            "bootstrap95": [boot[50], boot[1949]], "registered_success": bool(ev >= 9 and boot[50] > 0)}


def state_rows(fam, st, rule_key):
    s = st[rule_key]
    base = {"family": fam, "pool_position": s["pool_position"], "source_file": s["source_file"],
            "site_rule": s["site_rule"], "site": s["site"], "site_clean_T_screen": s["site_clean_T_screen"],
            "site_clean_T_same_forward": s["site_clean_T_same_forward"],
            "same_forward_clean_matches_screen_steps_5_16": sum(s["same_forward_clean_matches_screen_steps"][4:]),
            "sigma_structural": s["directions"]["structural"]["sigma_estimate"],
            "sigma_logit": s["directions"]["logit"]["sigma_estimate"],
            "cosine_structural_logit": s["directions"]["cosine_structural_logit"]}
    per_state, per_variant = [], []
    for dose, sm in s["summary"].items():
        r = dict(base)
        r["dose"] = dose
        for arm in ARMS:
            r[f"V_{arm}"] = sm["V"][arm]
        for k in ("D_structural_minus_isotropic", "D_logit_minus_isotropic", "D_structural_minus_logit"):
            r[k] = sm[k]
        for key in ("blocked_recoveries", "new_recoveries", "accelerated", "delayed", "ACT16_blank_correct",
                    "site_shift", "site_blocked_rate"):
            for arm in ARMS:
                r[f"{key}_{arm}"] = sm[key][arm]
        for step, by_arm in sm["h_gain"].items():
            for arm in ARMS:
                r[f"h_gain_ACT{step}_{arm}"] = by_arm[arm]
        r["clean_ACT16_blank_correct"] = s["outcomes"][0]["ACT16_blank_correct"]
        per_state.append(r)
    for i, (v, o) in enumerate(zip(s["variants"], s["outcomes"])):
        if i == 0:
            continue
        r = {k: base[k] for k in ("family", "pool_position", "source_file", "site_rule", "site")}
        r.update({"row": i, "arm": v["arm"], "draw": "" if v["draw"] is None else v["draw"],
                  "dose": v["dose"], "sign": v["sign"]})
        r.update(o)
        for step, rows in s["transport_at_steps"].items():
            t = rows[i]
            r[f"h_gain_ACT{step}"] = t["h_gain"]
            r[f"readout_fraction_ACT{step}"] = t["readout_fraction"]
            r[f"margin_certificate_ACT{step}"] = t["strict_global_margin_certificate"]
            r[f"blank_flips_ACT{step}"] = t["actual_blank_digit_flips"]
        per_variant.append(r)
    return per_state, per_variant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="directory with screen.json, active_a.json, active_b.json")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(a.source.read_text())
    srcs = {r["pool_position"]: r for r in source["pool"]}
    screen_path = a.run_dir / "screen.json"
    screen = json.loads(screen_path.read_text())
    fam_of = {pos: f for f in FAMILIES for pos in screen["families"].get(f, [])}
    summary = {"schema": "LS-SAE12/posthoc-tables-v1", "inputs": {"screen.json": hashlib.sha256(screen_path.read_bytes()).hexdigest()},
               "screen_platform": screen.get("platform"), "eligible": screen["eligible_count"],
               "families": {f: screen["families"].get(f) for f in FAMILIES}, "family_results": {}}
    rows = []
    for r in screen["rows"]:
        src = srcs[r["pool_position"]]
        bc = r["clean_blank_correct_by_step"]
        rows.append({"pool_position": r["pool_position"], "source_file": r["source_file"], "holes": src["holes"],
                     "eligible": r["eligible"], "family": fam_of.get(r["pool_position"], ""),
                     "recovering_errors_at_ACT4": len(r["recovering_errors_at_tstar"]),
                     "selected_site": "" if r["selected_site"] is None else r["selected_site"],
                     "selected_site_T": "" if r["selected_site"] is None else r["clean_verified_schedule"][str(r["selected_site"])],
                     "control_site": r["control_site"],
                     **{f"clean_blank_correct_ACT{s}": bc[s - 1] for s in (1, 2, 4, 6, 8, 12, 16)},
                     "clean_ACT16_exact": bc[15] == src["holes"],
                     "solver_algx_nodes": src["solver_algx_search_nodes"],
                     "singles_deduced": src["visible_structural_difficulty"]["deduced_missing_cells"]})
    write_csv(a.out / "ls_sae12_screening.csv", rows)
    summary["screen_clean_ACT16_exact"] = sum(r["clean_ACT16_exact"] for r in rows)
    all_states, all_variants, gate_rows = [], [], []
    for fam in FAMILIES:
        path = a.run_dir / f"{fam}.json"
        if not path.exists():
            summary["family_results"][fam] = {"missing": True}
            continue
        summary["inputs"][f"{fam}.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        for st in data["states"]:
            for key in ("selected", "control"):
                ps, pv = state_rows(fam, st, key)
                all_states += ps
                all_variants += pv
        diffs = [st["selected"]["summary"][PRIMARY]["D_structural_minus_isotropic"] for st in data["states"]]
        mine = gate(diffs, fam)
        saved = data["statistics"]
        primary = saved[f"selected@{PRIMARY}@D_structural_minus_isotropic"]
        if mine["registered_success"] != primary.get("registered_success") or (
                mine.get("bootstrap95") and mine["bootstrap95"] != primary["bootstrap95"]):
            raise ValueError(f"{fam}: recomputed gate differs from saved statistics")
        for key, g in saved.items():
            if key == "registered_family_success":
                continue
            rule, dose, contrast = key.split("@")
            gate_rows.append({"family": fam, "site_rule": rule, "dose": dose, "contrast": contrast,
                              "registered_primary": g.get("registered_primary"), "n_states": g.get("n_states"),
                              "events_positive": g.get("events_positive"), "mean": g.get("mean"),
                              "boot95_lo": (g.get("bootstrap95") or [None, None])[0],
                              "boot95_hi": (g.get("bootstrap95") or [None, None])[1],
                              "rule_pass": g.get("registered_success"),
                              "per_state": " ".join(f"{d:g}" for d in g.get("diffs", []))})
        edits = [v for v in all_variants if v["family"] == fam]
        corr = {}
        for step in ("5", "8", "16"):
            for dose in ("0.02", "0.25", "2.00"):
                sel = [v for v in edits if v["site_rule"] == "observable_error" and v["dose"] == dose]
                corr[f"ACT{step}@{dose}"] = spearman([v[f"h_gain_ACT{step}"] for v in sel], [v["vdd"] for v in sel])
        summary["family_results"][fam] = {"states": len(data["states"]), "primary": mine,
                                          "spearman_hgain_vs_vdd_selected_site": corr,
                                          "platform": data.get("platform")}
    summary["tables"] = {
        "ls_sae12_screening.csv": len(rows),
        "ls_sae12_per_state.csv": write_csv(a.out / "ls_sae12_per_state.csv", all_states),
        "ls_sae12_per_variant.csv": write_csv(a.out / "ls_sae12_per_variant.csv", all_variants),
        "ls_sae12_family_gates.csv": write_csv(a.out / "ls_sae12_family_gates.csv", gate_rows)}
    res = [summary["family_results"].get(f, {}).get("primary", {}).get("registered_success") for f in FAMILIES]
    summary["registered_outcome"] = ("INCOMPLETE" if any(r is None for r in res)
                                     else "SUPPORTED" if all(res) else "FAIL")
    (a.out / "ls_sae12_analysis_summary.json").write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({k: summary[k] for k in ("eligible", "families", "registered_outcome", "tables")}, indent=1))


if __name__ == "__main__":
    main()
