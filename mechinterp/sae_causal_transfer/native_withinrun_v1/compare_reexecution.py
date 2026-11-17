"""Compare a re-execution of an LS-SAE11 stage with the archived Actions output.

Standard library only. For every puzzle, horizon and signed variant: maximum
absolute logit difference, number of argmax digit changes over all 81 cells,
and the blank-correct count in each run; then the registered family statistics
of both runs side by side. Writes a per-variant CSV and a summary JSON.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

HORIZONS = ("ACT3", "ACT6", "ACT12")
ARMS = ("state_power", "logit_power", "sae", "random")


def digits(rows):
    return [1 + max(range(9), key=lambda j: r[j]) for r in rows]


def variants(native):
    yield "clean", None, None, native["clean"]
    for arm in ARMS:
        for dose, by_sign in sorted(native.get(arm, {}).items()):
            for sign, rows in sorted(by_sign.items()):
                yield arm, dose, sign, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archived", type=Path, required=True)
    p.add_argument("--rerun", type=Path, required=True)
    p.add_argument("--label", required=True, help="where the re-execution ran")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()
    archived = json.loads(a.archived.read_text())
    rerun = json.loads(a.rerun.read_text())
    if archived["mode"] != rerun["mode"] or set(archived["families"]) != set(rerun["families"]):
        raise ValueError("runs cover different stages")
    rows = []
    for fam in archived["families"]:
        for idx, (c0, c1) in enumerate(zip(archived["families"][fam]["per_puzzle"],
                                          rerun["families"][fam]["per_puzzle"])):
            if c0["source_seed"] != c1["source_seed"]:
                raise ValueError("puzzle order differs")
            for h in HORIZONS:
                n0 = dict(((arm, dose, sign), r) for arm, dose, sign, r in variants(c0["native_81x9_logits"][h]))
                for arm, dose, sign, r1 in variants(c1["native_81x9_logits"][h]):
                    r0 = n0[(arm, dose, sign)]
                    d0, d1 = digits(r0), digits(r1)
                    s0 = c0["native_task_scores"][h]
                    s1 = c1["native_task_scores"][h]
                    b0 = s0["clean"] if arm == "clean" else s0["arms"][arm][dose][sign]
                    b1 = s1["clean"] if arm == "clean" else s1["arms"][arm][dose][sign]
                    rows.append({
                        "family": fam, "index": idx, "seed": c0["source_seed"], "horizon": h,
                        "arm": arm, "dose": dose or "", "sign": sign or "",
                        "max_abs_logit_diff": max(abs(x - y) for a0, a1 in zip(r0, r1) for x, y in zip(a0, a1)),
                        "argmax_changes_81": sum(x != y for x, y in zip(d0, d1)),
                        "archived_blank_correct": b0["blank_correct"], "rerun_blank_correct": b1["blank_correct"],
                    })
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"label": a.label,
               "archived_sha256": hashlib.sha256(a.archived.read_bytes()).hexdigest(),
               "rerun_sha256": hashlib.sha256(a.rerun.read_bytes()).hexdigest(),
               "variants_compared": len(rows), "by_horizon": {}, "clean_by_puzzle": [], "families": {}}
    for h in HORIZONS:
        rs = [r for r in rows if r["horizon"] == h]
        cl = [r for r in rs if r["arm"] == "clean"]
        summary["by_horizon"][h] = {
            "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rs),
            "variants_with_argmax_change": sum(r["argmax_changes_81"] > 0 for r in rs),
            "clean_puzzles_with_argmax_change": sum(r["argmax_changes_81"] > 0 for r in cl),
            "variants": len(rs)}
    for r in rows:
        if r["arm"] == "clean":
            summary["clean_by_puzzle"].append({k: r[k] for k in ("family", "index", "seed", "horizon",
                                                                 "argmax_changes_81", "archived_blank_correct",
                                                                 "rerun_blank_correct", "max_abs_logit_diff")})
    for fam in archived["families"]:
        s0 = archived["families"][fam].get("statistics")
        s1 = rerun["families"][fam].get("statistics")
        summary["families"][fam] = {"archived": s0, "rerun": s1}
    summary["archived_registered_success"] = archived["registered_hard_scientific_success"]
    summary["rerun_registered_success"] = rerun["registered_hard_scientific_success"]
    a.out_json.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({k: summary[k] for k in ("by_horizon", "families", "archived_registered_success",
                                             "rerun_registered_success")}, indent=1))


if __name__ == "__main__":
    main()
