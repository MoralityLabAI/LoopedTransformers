"""Hill-climb research-logic traces.

A trace step is one proposal judged against an incumbent:
    incumbent -> proposal (what changed) -> evaluation -> criteria checks -> decision -> next incumbent
`logic` is an ordered list of statements derived mechanically from the recorded numbers and the
registered rule (never model-written), so every statement can be re-verified. When a source
records its own decision, `decision_recorded` keeps it and `discrepancy` flags disagreement.

Sources: our acceptance ledgers, the upstream RSI gate record, and continuous hill climbs
(bpd hill climb in hillclimb.py). Output rows also export as Aleph `research` presentations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from . import acceptance as acc
from .ledger import Ledger, append_jsonl, iter_jsonl

SCHEMA = "ink02.hillclimb_trace.v1"
CHECK_ORDER = ("primary_gain_ok", "family_regression_ok", "repetition_ok", "completion_ok")


def _id(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def step(source, evidence_class, trace, index, incumbent, proposal, evaluation, criteria, checks,
         logic, decision_by_rule, decision_recorded=None, next_incumbent=None, extra=None) -> dict:
    failing = [c for c in checks if not checks[c]]
    d = {"schema": SCHEMA, "source": source, "evidence_class": evidence_class, "trace": trace,
         "step": index, "incumbent": incumbent, "proposal": proposal, "evaluation": evaluation,
         "criteria": criteria, "checks": checks, "binding_checks": failing, "logic": logic,
         "decision_by_rule": decision_by_rule, "decision_recorded": decision_recorded,
         "discrepancy": decision_recorded is not None and decision_recorded != decision_by_rule,
         "next_incumbent": next_incumbent, **(extra or {})}
    d["step_id"] = _id(source, trace, index, proposal.get("id"))
    return d


# ---- discrete acceptance ledgers (engine) ------------------------------------------------
def logic_discrete(d: dict, crit: dict) -> list[str]:
    p, c = d["predecessor_summary"], d["candidate_summary"]
    out = [f"paired cases: {p['n']}; incumbent correct {p['correct']}, candidate correct {c['correct']}",
           f"primary gain {d['primary_gain']:+d} vs required >= {crit['min_primary_gain']}: "
           f"{'pass' if d['checks']['primary_gain_ok'] else 'fail'}"]
    for fam in sorted(d["family_losses"]):
        loss, lim = d["family_losses"][fam], d["family_limits"][fam]
        if loss > 0:
            out.append(f"family {fam}: lost {loss} correct vs limit {lim}: {'pass' if loss <= lim else 'fail'}")
    out.append(f"repetition rate {p['repetition_rate']:.3f} -> {c['repetition_rate']:.3f} "
               f"(max rise {crit['max_repetition_increase']}): {'pass' if d['checks']['repetition_ok'] else 'fail'}")
    out.append(f"completion rate {p['completion_rate']:.3f} -> {c['completion_rate']:.3f} "
               f"(max drop {crit['max_completion_drop']}): {'pass' if d['checks']['completion_ok'] else 'fail'}")
    out.append("decision: accept, candidate becomes incumbent" if d["accepted"]
               else "decision: reject, keep incumbent")
    return out


def from_acceptance_ledger(path: Path, source: str, evidence_class: str) -> Iterator[dict]:
    for i, b in enumerate(Ledger(path).kinds("acceptance")):
        crit = acc.Criteria(**b["criteria"])
        d = acc.decide(crit, b["predecessor_cases"], b["candidate_cases"])  # recompute, never trust
        p, c = d["predecessor_summary"], d["candidate_summary"]
        fam = lambda s: {f: v["correct"] / v["n"] for f, v in s["families"].items()}
        yield step(source, evidence_class, f"{b['arm']}", i,
                   {"id": b["predecessor_checkpoint"], "metrics": {"correct": p["correct"], "families": fam(p)}},
                   {"id": b["candidate_checkpoint"], "change": {"generation": b["generation"],
                    "hyperparameters": b["hyperparameters"], "scales_examined": b["scales_examined"]}},
                   {"metrics": {"correct": c["correct"], "families": fam(c)}, "paired_n": p["n"]},
                   b["criteria"], d["checks"], logic_discrete(d, b["criteria"]),
                   "accept" if d["accepted"] else "reject",
                   "accept" if b["decision"]["accepted"] else "reject", b["next_head"])


# ---- upstream RSI record (published gate, all candidates) ------------------------------------
def from_upstream_record(record: dict, source="upstream_rsi_765453d") -> Iterator[dict]:
    """Every examined candidate is a proposal against the reported incumbent of its generation."""
    n = record["gate_size_per_mode"]
    inc_gate, inc_id = record["init_gate"], "init"
    k = 0
    for g in record["generations"]:
        winner_id = None
        for cand in g["candidates"]:
            p, c = acc.upstream_counts(inc_gate, n), acc.upstream_counts(cand, n)
            drops = {m: p[m] - c[m] for m in acc.UPSTREAM_MODES}
            rep_cap = max(3.12, inc_gate["max_rep"] + 1.6)
            comp = round(cand["mean_comp"] * 3 * n / 100) - round(inc_gate["mean_comp"] * 3 * n / 100)
            gain = cand["total_correct"] - inc_gate["total_correct"]
            checks = {"mode_drop_ok": max(drops.values()) <= 2, "repetition_ok": cand["max_rep"] <= rep_cap,
                      "gain_or_tiebreak_ok": gain > 0 or (gain == 0 and comp >= 4)}
            ok = all(checks.values())
            is_winner = cand["scale"] == g["winner_scale"]
            logic = [f"incumbent total {inc_gate['total_correct']}/{3 * n}, candidate total {cand['total_correct']}/{3 * n} (gain {gain:+d})",
                     "per-mode correct-answer change " + ", ".join(f"{m} {-drops[m]:+d}" for m in acc.UPSTREAM_MODES)
                     + f"; largest loss {max(0, max(drops.values()))}, rule allows at most 2: "
                     + ("pass" if checks["mode_drop_ok"] else "fail"),
                     f"max repetition {cand['max_rep']} vs cap {rep_cap:.2f}: {'pass' if checks['repetition_ok'] else 'fail'}",
                     f"gain {gain:+d} (completion tie-break {comp:+d}, needs >= 4 on a tie): "
                     f"{'pass' if checks['gain_or_tiebreak_ok'] else 'fail'}",
                     "decision by published rule: " + ("accept" if ok else "reject")]
            recorded = ("accept" if g["accepted"] else "reject") if is_winner else None
            if recorded is not None and recorded != ("accept" if ok else "reject"):
                logic.append(f"recorded decision: {recorded} (disagrees with the published rule)")
            cid = f"gen{g['gen']}-scale{cand['scale']}"
            yield step(source, "published_record_replay", "upstream", k,
                       {"id": inc_id, "metrics": {**p, "total": inc_gate["total_correct"]}},
                       {"id": cid, "change": {"generation": g["gen"], "update_scale": cand["scale"]},
                        "selected_as_winner": is_winner},
                       {"metrics": {**c, "total": cand["total_correct"], "max_rep": cand["max_rep"],
                                    "mean_comp": cand["mean_comp"]}, "paired_n": 3 * n},
                       {"max_mode_loss": 2, "repetition_cap": "max(3.12, incumbent+1.6)",
                        "tiebreak_completion_gain": 4}, checks, logic,
                       "accept" if ok else "reject", recorded,
                       None if not is_winner else (cid if g["accepted"] else inc_id))
            k += 1
            if is_winner:
                winner_id, winner = cid, cand
        if g["accepted"] and winner_id:  # follow the reported chain
            inc_gate, inc_id = winner, winner_id


# ---- continuous hill climb (synthetic) ---------------------------------------------------------
def continuous_decide(criteria: dict, inc_cases: list[dict], cand_cases: list[dict]) -> dict:
    """cases: [{"case_id","family","value"}], higher is better; paired by case_id."""
    a = {x["case_id"]: x for x in inc_cases}
    b = {x["case_id"]: x for x in cand_cases}
    if set(a) != set(b):
        raise ValueError("continuous decision needs identical paired cases")
    diffs = {k: b[k]["value"] - a[k]["value"] for k in a}
    gain = sum(diffs.values()) / len(diffs)
    fams = sorted({x["family"] for x in inc_cases})
    fam_gain = {f: sum(diffs[k] for k in a if a[k]["family"] == f) / sum(1 for k in a if a[k]["family"] == f)
                for f in fams}
    checks = {"primary_gain_ok": gain >= criteria["min_mean_gain"],
              "family_regression_ok": all(v >= -criteria["max_family_drop"] for v in fam_gain.values())}
    logic = [f"paired cases {len(diffs)}; mean utility change {gain:+.4f} vs required >= {criteria['min_mean_gain']}: "
             f"{'pass' if checks['primary_gain_ok'] else 'fail'}"]
    for f in fams:
        logic.append(f"family {f}: mean change {fam_gain[f]:+.4f} (max drop {criteria['max_family_drop']}): "
                     f"{'pass' if fam_gain[f] >= -criteria['max_family_drop'] else 'fail'}")
    ok = all(checks.values())
    logic.append("decision: accept, proposal becomes incumbent" if ok else "decision: reject, keep incumbent")
    return {"accepted": ok, "checks": checks, "mean_gain": gain, "family_gain": fam_gain, "logic": logic}



# ---- DataGoblin MeTTa RAM hill climb (metta_driver.py output) -----------------------------------
METTA_KEY = (("solved_and_stopped", 1), ("new_violation_additions", -1), ("unsafe_additions", -1),
             ("exact_first_edit", 1), ("premature_stops", -1), ("verifier_queries", -1),
             ("mean_remaining_canonical_edits", -1), ("mean_inference_iterations", -1))


def metta_logic(prev: dict, new: dict) -> tuple[dict, list[str], str]:
    safety = {"no_new_violations": new["new_violation_additions"] <= prev["new_violation_additions"],
              "no_new_unsafe": new["unsafe_additions"] <= prev["unsafe_additions"]}
    logic = [f"new-constraint violations {prev['new_violation_additions']} -> {new['new_violation_additions']}: "
             f"{'pass' if safety['no_new_violations'] else 'fail'}",
             f"unsafe additions {prev['unsafe_additions']} -> {new['unsafe_additions']}: "
             f"{'pass' if safety['no_new_unsafe'] else 'fail'}"]
    verdict = "tie"
    for name, sign in METTA_KEY:
        a, b = prev[name], new[name]
        if name.startswith("mean_"):
            a, b = round(a * 1000), round(b * 1000)
        if a != b:
            better = (b - a) * sign > 0
            verdict = "better" if better else "worse"
            logic.append(f"lexicographic key: first differing component {name} {prev[name]} -> {new[name]} "
                         f"({'higher' if sign > 0 else 'lower'} is better): candidate {verdict}")
            break
    else:
        logic.append("lexicographic key: all components equal, so no improvement")
    checks = {**safety, "lexicographic_improvement": verdict == "better"}
    ok = all(checks.values())
    logic.append("decision: accept, edit becomes incumbent config" if ok else "decision: reject, keep incumbent config")
    return checks, logic, "accept" if ok else "reject"


def from_metta_steps(path: Path, source: str, trace: str | None = None) -> Iterator[dict]:
    for i, h in enumerate(iter_jsonl(path)):
        checks, logic, by_rule = metta_logic(h["incumbent_metrics"], h["candidate_metrics"])
        keep = ("solved_and_stopped", "exact_first_edit", "premature_stops", "new_violation_additions",
                "unsafe_additions", "mean_remaining_canonical_edits", "mean_inference_iterations")
        yield step(source, "cpu_numpy_training", trace or Path(path).stem, i,
                   {"id": json.dumps(h["incumbent_config"], sort_keys=True),
                    "metrics": {k: h["incumbent_metrics"][k] for k in keep}, "arm": h["arm"]},
                   {"id": json.dumps(h["candidate_config"], sort_keys=True),
                    "change": {**h["proposal"], "proposer": h["proposer"]}},
                   {"metrics": {k: h["candidate_metrics"][k] for k in keep},
                    "paired_n": h["candidate_metrics"]["cases"]},
                   {"rule": "DataGoblin MeTTa ram_hillclimb: lexicographic select_key improvement and no new "
                            "violation or unsafe additions"},
                   checks, logic, by_rule, "accept" if h["accepted"] else "reject",
                   json.dumps(h["candidate_config"] if h["accepted"] else h["incumbent_config"], sort_keys=True),
                   {"arm": h["arm"], "recorded_decision_code": h["decision"]})


# ---- BPD TRM curation windows (datagoblin.trm_curation_decision.v1) ----------------------------
def from_curation_decisions(path: Path, source: str = "bpd_trm_curate02") -> Iterator[dict]:
    """One step per (root, window): selected edges vs the rejected pool, calibration rule
    (accept iff cal_after <= cal_before + 1e-6, BPD trm_curation/campaign.py), and the
    counterfactual realized gains the record keeps for unselected candidates."""
    groups: dict = {}
    for r in iter_jsonl(path):
        pv = r["provenance"]
        groups.setdefault((pv["root"], pv["window"]), []).append(r)
    for i, ((root, window), rows) in enumerate(sorted(groups.items())):
        sel = [x for x in rows if x["outcome"]["selected"]]
        pool = [x for x in rows if not x["outcome"]["selected"]]
        o = rows[0]["outcome"]
        before, after = o["window_cal_before"], o["window_cal_after"]
        rg = lambda xs: [x["outcome"]["realized_gain"] for x in xs]
        pg = lambda xs: [x["mechanistic_features"]["predicted_gain"] for x in xs]
        mean = lambda v: sum(v) / len(v) if v else None
        best_all = sorted(rg(rows), reverse=True)[:len(sel)]
        regret = (sum(best_all) - sum(rg(sel))) / max(1, len(sel))
        signs = [x["outcome"]["sign_match"] for x in rows]
        reasons = {}
        for x in sel:
            reasons[x["outcome"]["selection_reason"]] = reasons.get(x["outcome"]["selection_reason"], 0) + 1
        ok = after <= before + 1e-6
        logic = [f"window {window} of root {root}: {len(sel)} selected of {len(rows)} candidates ({', '.join(f'{k} {v}' for k, v in sorted(reasons.items()))})",
                 f"predicted gain: selected mean {mean(pg(sel)):+.4f}, rejected pool mean {mean(pg(pool)) if pool else float('nan'):+.4f}",
                 f"realized gain (counterfactual kept for every candidate): selected mean {mean(rg(sel)):+.4f}, "
                 f"pool mean {mean(rg(pool)) if pool else float('nan'):+.4f}; per-slot regret vs best possible {regret:+.4f}",
                 f"predicted/realized sign agreement {sum(signs)}/{len(signs)}",
                 f"calibration loss {before:.6f} -> {after:.6f}; rule accepts iff after <= before + 1e-6: {'pass' if ok else 'fail'}",
                 "decision: accept step, keep update" if ok else "decision: reject step, roll back model and optimizer"]
        yield step(source, "cpu_torch_training_record", f"root{root}", i,
                   {"id": f"root{root}-w{window}-before", "metrics": {"calibration_loss": before}},
                   {"id": f"root{root}-w{window}-update",
                    "change": {"selected_edges": [x["input"] for x in sel], "selection_reasons": reasons}},
                   {"metrics": {"calibration_loss": after, "selected_realized_gain_mean": mean(rg(sel)),
                                "pool_realized_gain_mean": mean(rg(pool)), "regret_per_slot": regret,
                                "sign_agreement": sum(signs) / len(signs)}, "paired_n": len(rows)},
                   {"rule": "accept iff calibration loss after <= before + 1e-6"},
                   {"calibration_non_increasing": ok}, logic, "accept" if ok else "reject",
                   "accept" if o["window_accepted"] else "reject",
                   f"root{root}-w{window}-update" if o["window_accepted"] else f"root{root}-w{window}-before",
                   {"experiment": rows[0]["provenance"]["experiment"], "arm": rows[0]["provenance"]["arm"]})

# ---- export ----------------------------------------------------------------------------------
def to_research_row(s: dict) -> dict:
    """Aleph `research` presentation: predict and justify the verdict from the evidence."""
    context = {"exercise": "hillclimb_verdict", "incumbent": s["incumbent"], "proposal": s["proposal"],
               "evaluation": s["evaluation"], "criteria": s["criteria"]}
    target = {"logic": s["logic"], "decision": s["decision_by_rule"], "binding_checks": s["binding_checks"]}
    return {"id": _id("research", s["step_id"]), "kind": "research", "root": f"{s['source']}:{s['trace']}",
            "family": "hillclimb_judgment", "split": "train" if s["evidence_class"] != "published_record_replay" else "dev",
            "task_digest": s["step_id"], "input": context, "target": target,
            "loss": {"boundary": "response_only", "field": "target", "eligible": True},
            "source_model": {"provider": "rule", "model": "frozen acceptance rule"},
            "provider_execution_attested": False,
            "verification": {"verifier": "rule_recompute", "passed": True, "evidence_class": s["evidence_class"],
                             "recorded_decision": s["decision_recorded"], "discrepancy": s["discrepancy"]}}


def write(steps, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    n = 0
    for s in steps:
        append_jsonl(path, s)
        n += 1
    return n


def export_research(trace_paths: list[Path], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    seen, counts = set(), {}
    for p in trace_paths:
        for s in iter_jsonl(p):
            row = to_research_row(s)
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            append_jsonl(out, row)
            counts[s["source"]] = counts.get(s["source"], 0) + 1
    return {"rows": len(seen), "by_source": counts}


def collect(out: Path, campaign: Path | None = None, hillclimb: Path | None = None,
            metta: list[Path] | None = None, curation: list[Path] | None = None, extra_ledgers: list[tuple[Path, str, str]] = ()) -> dict:
    """Write one trace file per source and the combined research export."""
    out.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).with_name("fixtures") / "upstream_gspo_gate_765453d.json"
    paths, summary = [], {}
    up = out / "upstream_rsi.jsonl"
    summary["upstream_rsi"] = write(from_upstream_record(json.loads(fixture.read_text(encoding="utf-8"))), up)
    paths.append(up)
    if campaign is not None:
        for led in sorted((campaign / "engineering_simulation").glob("arm_*/ledger.jsonl")):
            name = f"engine_{led.parent.name}"
            p = out / f"{name}.jsonl"
            summary[name] = write(from_acceptance_ledger(led, name, "engineering_simulation"), p)
            paths.append(p)
    if hillclimb is not None:
        for p in sorted(hillclimb.glob("bpd_hillclimb_seed*.jsonl")):
            summary[p.stem] = sum(1 for _ in iter_jsonl(p))
            paths.append(p)
    for mdir in metta or []:
        for mp in sorted(Path(mdir).rglob("metta_*_c*.jsonl")):
            tag = mp.stem if mp.parent == Path(mdir) else f"{mp.parent.name}_{mp.stem}"
            p = out / f"{tag}.jsonl"
            summary[tag] = write(from_metta_steps(mp, "datagoblin_metta_ram", tag), p)
            paths.append(p)
    for cp in curation or []:
        p = out / f"curation_{Path(cp).parent.parent.parent.name}.jsonl"
        summary[p.stem] = write(from_curation_decisions(Path(cp)), p)
        paths.append(p)
    for led, name, ev in extra_ledgers:
        p = out / f"{name}.jsonl"
        summary[name] = write(from_acceptance_ledger(led, name, ev), p)
        paths.append(p)
    exp = export_research(paths, out / "research.jsonl")
    disc = sum(s["discrepancy"] for p in paths for s in iter_jsonl(p))
    manifest = {"schema": SCHEMA, "steps_by_source": summary, "research_rows": exp["rows"],
                "recorded_vs_rule_discrepancies": disc,
                "note": "logic statements are derived from recorded numbers by the registered rule"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--hillclimb", type=Path)
    ap.add_argument("--metta", type=Path, nargs="*")
    ap.add_argument("--curation", type=Path, nargs="*", help="datagoblin.trm_curation_decision.v1 JSONL files")
    a = ap.parse_args()
    print(json.dumps(collect(a.out, a.campaign, a.hillclimb, a.metta, a.curation), indent=1))


if __name__ == "__main__":
    main()
