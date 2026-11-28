"""Report reconciliation: derive report tables from immutable receipts.

Each generated block lives between markers in a Markdown report:
    <!-- reconcile:<block>:start -->  ...  <!-- reconcile:<block>:end -->
`render(block)` recomputes it from receipts; `check(report)` fails if any block differs from
what the receipts give, so a hand-edited number cannot survive the test suite.

    python -m research_gym.ink_fp_audit_03.reconcile --write   # refresh blocks in place
    python -m research_gym.ink_fp_audit_03.reconcile           # check only (exit 1 on drift)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A02 = ROOT / "research_gym/ink_fp_audit_02"
EFFORTS = ("low", "medium", "high")


def records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def effort_stats(recs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in recs:
        by[r["effort"]].append(r)
    paired = defaultdict(dict)
    for r in recs:
        paired[r["task_id"]][r["effort"]] = bool(r["correct"])
    full = {k: v for k, v in paired.items() if all(e in v for e in EFFORTS)}
    fam = defaultdict(Counter)
    fam_n = defaultdict(Counter)
    for r in recs:
        fam[r["family"]][r["effort"]] += bool(r["correct"])
        fam_n[r["family"]][r["effort"]] += 1
    out = {"n_records": len(recs), "paired_tasks": len(full), "efforts": {}}
    for e in EFFORTS:
        rs = by[e]
        out["efforts"][e] = {
            "correct": sum(bool(r["correct"]) for r in rs), "n": len(rs),
            "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in rs),
            "truncated": sum(r.get("finish_reason") == "length" for r in rs),
            "rescues_vs_low": None if e == "low" else sum(v[e] and not v["low"] for v in full.values()),
            "losses_vs_low": None if e == "low" else sum(v["low"] and not v[e] for v in full.values()),
        }
    fails = [r for r in recs if not r["correct"]]
    out["failures"] = len(fails)
    out["failures_truncated"] = sum(r.get("finish_reason") == "length" for r in fails)
    out["truncated_total"] = sum(r.get("finish_reason") == "length" for r in recs)
    out["correct_but_truncated"] = sum(r.get("finish_reason") == "length" and r["correct"] for r in recs)
    out["saturated_families"] = sorted(f for f in fam if all(fam[f][e] == fam_n[f][e] > 0 for e in EFFORTS))
    out["families"] = {f: {e: f"{fam[f][e]}/{fam_n[f][e]}" for e in EFFORTS} for f in sorted(fam)}
    return out


def render_effort_table(stats: dict) -> str:
    lines = ["| Effort | Correct | Completion tokens | Truncated | Rescues vs. low | Losses vs. low |",
             "|---|---:|---:|---:|---:|---:|"]
    for e in EFFORTS:
        s = stats["efforts"][e]
        dash = lambda v: "—" if v is None else str(v)
        lines.append(f"| {e.capitalize()} | {s['correct']}/{s['n']} | {s['completion_tokens']:,} | {s['truncated']} | "
                     f"{dash(s['rescues_vs_low'])} | {dash(s['losses_vs_low'])} |")
    lines.append("")
    lines.append(f"Records: {stats['n_records']}. Truncations (finish_reason=length): {stats['truncated_total']}; "
                 f"failures: {stats['failures']}, of which truncated: {stats['failures_truncated']}; "
                 f"correct-but-truncated: {stats['correct_but_truncated']}. "
                 f"Families correct on every task at all three efforts ({len(stats['saturated_families'])}): "
                 + ", ".join(stats["saturated_families"]) + ".")
    return "\n".join(lines)


A03 = ROOT / "research_gym/ink_fp_audit_03"
BUDGET = A03 / "receipts/budget01_dev_20261128"
FRONTIER = A03 / "receipts/frontier01_20261128"


def render_families(stats: dict) -> str:
    lines = ["| Family | Low | Medium | High |", "|---|---:|---:|---:|"]
    for f, row in stats["families"].items():
        lines.append(f"| {f} | {row['low']} | {row['medium']} | {row['high']} |")
    return "\n".join(lines)


def render_budget_arms() -> str:
    from .budget_policies import evaluate
    ev = evaluate(BUDGET / "records.jsonl", FRONTIER / "tasks_dev.jsonl")
    keep = [k for k in ev["arms"] if k[:1] in "DEFH" or k.endswith(("@1024", "@2048", "@4096", "@8192"))]
    lines = [f"Paired tasks with all three efforts: {ev['n_tasks']}.", "",
             "| Arm | Success | Tokens | Success / 1k tokens | Truncation rate | Correct given complete | Escalated | Rescued by escalation |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    f3 = lambda v: "—" if v is None else f"{v:.3f}"
    for k in keep:
        v = ev["arms"][k]
        if "success" not in v:
            lines.append(f"| {k} | {v.get('status', '')} | | | | | | |")
            continue
        lines.append(f"| {k} | {v['success']}/{v['n']} | {v['tokens']:,} | {f3(v['success_per_1k_tokens'])} | "
                     f"{f3(v['truncation_rate'])} | {f3(v['correct_given_complete'])} | {v['escalated']} | {v['escalation_rescues']} |")
    lines += ["", "| Adaptive arm | Success | Tokens | Best static arm within that token budget | Static success | Static tokens |",
              "|---|---:|---:|---|---:|---:|"]
    for k, m in ev["matched_budget"].items():
        lines.append(f"| {k} | {m['adaptive_success']} | {m['adaptive_tokens']:,} | {m['best_static_within_budget'] or '—'} | "
                     f"{'—' if m['static_success'] is None else m['static_success']} | "
                     f"{'—' if m['static_tokens'] is None else format(m['static_tokens'], ',')} |")
    return "\n".join(lines)


BLOCKS = {
    "a03_budget_effort": lambda: render_effort_table(effort_stats(records(BUDGET / "records.jsonl"))),
    "a03_budget_families": lambda: render_families(effort_stats(records(BUDGET / "records.jsonl"))),
    "a03_budget_arms": render_budget_arms,
    "a02_base_dev": lambda: render_effort_table(effort_stats(records(
        A02 / "receipts/inky_base_dev_20261128/records.jsonl"))),
    "a02_frontier8192": lambda: render_effort_table(effort_stats(records(
        A02 / "receipts/inky_frontier8192_20261128/records.jsonl"))),
}
MARK = re.compile(r"<!-- reconcile:(?P<name>[\w.-]+):start -->\n(?P<body>.*?)\n<!-- reconcile:(?P=name):end -->", re.S)


def refresh(text: str, blocks=BLOCKS) -> str:
    return MARK.sub(lambda m: f"<!-- reconcile:{m['name']}:start -->\n{blocks[m['name']]()}\n"
                              f"<!-- reconcile:{m['name']}:end -->", text)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")  # CRLF checkouts must not hide drift


def check(path: Path, blocks=BLOCKS, require: tuple[str, ...] = ()) -> list[str]:
    text = read(path)
    found = {m["name"] for m in MARK.finditer(text)}
    drift = [f"missing:{b}" for b in require if b not in found]
    for m in MARK.finditer(text):
        if m["body"] != blocks[m["name"]]():
            drift.append(m["name"])
    return drift


REPORTS = (A02 / "REPORT.md", ROOT / "research_gym/ink_fp_audit_03/REPORT.md")
REQUIRED = {A02 / "REPORT.md": ("a02_base_dev", "a02_frontier8192")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    bad = []
    for rp in REPORTS:
        if not rp.exists():
            continue
        if a.write:
            rp.write_text(refresh(read(rp)), encoding="utf-8", newline="\n")
        bad += [f"{rp.name}:{b}" for b in check(rp, require=REQUIRED.get(rp, ()))]
    print(json.dumps({"drift": bad}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
