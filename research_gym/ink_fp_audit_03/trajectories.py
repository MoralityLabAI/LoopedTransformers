"""Research trajectories (`ink03.research_trajectory.v1`) and next-action training rows.

A trajectory step records one research decision:
    question, hypothesis, evidence, proposed_experiment, estimated_cost, expected_informative_outcome,
    actual_result, interpretation, updated_hypothesis, decision, alternatives, certificate
Every field carries a provenance tag in `field_sources`:
    mechanical      computed here from an immutable receipt (path recorded)
    rule            derived by a registered decision rule (e.g. the frozen acceptance rule)
    transcribed     copied from a committed report whose numbers were not machine-readable
    curated         written by the campaign author (Claude) as interpretation; never a certificate
Training rows ask for the *next action* given the trajectory prefix (question .. evidence), not for the
completed report, so the model learns to choose experiments rather than imitate summaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from research_gym.ink_fp_audit_02.ledger import append_jsonl, iter_jsonl

SCHEMA = "ink03.research_trajectory.v1"
FIELDS = ("question", "hypothesis", "evidence", "proposed_experiment", "estimated_cost",
          "expected_informative_outcome", "actual_result", "interpretation", "updated_hypothesis",
          "decision", "alternatives", "certificate")
PREFIX = ("question", "hypothesis", "evidence")


def _id(*p) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


def make(campaign: str, step: int, values: dict, sources: dict, provenance: dict, outcome: str) -> dict:
    missing = [f for f in FIELDS if f not in values]
    if missing:
        raise ValueError(f"missing fields {missing}")
    bad = {k: v for k, v in sources.items() if v.split(":")[0] not in ("mechanical", "rule", "transcribed", "curated")}
    if bad or set(sources) != set(FIELDS):
        raise ValueError("every field needs a provenance tag from the registered set")
    return {"schema": SCHEMA, "campaign": campaign, "step": step, **{f: values[f] for f in FIELDS},
            "field_sources": sources, "provenance": provenance, "outcome": outcome,
            "trajectory_id": _id(campaign, step, values["proposed_experiment"])}


# ---- from hill-climb traces (mechanical + rule) -----------------------------------------------------
def from_hillclimb_trace(s: dict) -> dict:
    v = {"question": f"Does the proposal improve on the incumbent under {s['criteria'].get('rule', s['criteria'].get('version', 'the registered rule'))}?",
         "hypothesis": f"proposal {s['proposal']['id']} beats incumbent {s['incumbent']['id']}",
         "evidence": {"incumbent": s["incumbent"], "evaluation": s["evaluation"]},
         "proposed_experiment": s["proposal"].get("change"), "estimated_cost": s["evaluation"].get("paired_n"),
         "expected_informative_outcome": "checks " + ", ".join(sorted(s["checks"])),
         "actual_result": s["checks"], "interpretation": s["logic"],
         "updated_hypothesis": "accepted" if s["decision_by_rule"] == "accept" else "rejected; incumbent kept",
         "decision": s["decision_by_rule"], "alternatives": s["binding_checks"],
         "certificate": {"rule_recomputed": True, "recorded_decision": s["decision_recorded"], "discrepancy": s["discrepancy"]}}
    src = {f: "mechanical" for f in FIELDS} | {"interpretation": "rule", "decision": "rule", "updated_hypothesis": "rule",
                                               "alternatives": "rule"}
    return make(f"hillclimb:{s['source']}:{s['trace']}", s["step"], v, src,
                {"trace_step_id": s["step_id"], "evidence_class": s["evidence_class"]},
                "negative" if s["decision_by_rule"] == "reject" else "positive")


# ---- curated campaign decisions with receipt-derived numbers ------------------------------------------
def audit02_effort(root: Path) -> list[dict]:
    from .reconcile import effort_stats, records
    p = "research_gym/ink_fp_audit_02/receipts/inky_base_dev_20261128/records.jsonl"
    st = effort_stats(records(root / p))
    e = st["efforts"]
    v = {"question": "Does higher reasoning effort rescue low-effort failures on verifier-backed tasks?",
         "hypothesis": "high effort solves tasks low effort fails (effort-rescue reward has signal)",
         "evidence": {"prior": "audit-02 Stage A smoke 6/6 at every effort"},
         "proposed_experiment": "96 dev tasks x low/medium/high at a 2048-token cap, temperature 0",
         "estimated_cost": {"requests": 288, "completion_tokens_cap": 288 * 2048},
         "expected_informative_outcome": "count of tasks correct at high but wrong at low",
         "actual_result": {k: {"correct": e[k]["correct"], "completion_tokens": e[k]["completion_tokens"],
                               "rescues_vs_low": e[k]["rescues_vs_low"]} for k in e} | {"truncated_failures": st["failures_truncated"],
                                                                                         "failures": st["failures"]},
         "interpretation": "every failure exhausted its completion budget; higher effort spends more tokens and "
                           "truncates more, so on these tasks effort trades accuracy for tokens",
         "updated_hypothesis": "no effort-rescue signal at a fixed cap; budget allocation, not effort, limits accuracy",
         "decision": "demote the rescue bonus to a control; build a token-budget gym and cost-sensitive reward",
         "alternatives": ["raise the cap on frontier families only", "drop high effort", "learned effort router"],
         "certificate": {"receipt": p, "reconciled": True}}
    src = {f: "curated" for f in FIELDS} | {"actual_result": f"mechanical:{p}", "estimated_cost": "mechanical",
                                            "certificate": "mechanical", "proposed_experiment": "mechanical"}
    return [make("ink-fp-audit-02:effort", 0, v, src, {"receipt": p}, "negative")]


def bpd_nullspace(root: Path) -> list[dict]:
    p = root.parent / "BPD-nullspace01/mechinterp/bpd/experiments/nullspace_update/results/campaign.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    v = {"question": "Does a protected null-space update preserve X_keep exactly without losing new-task fit?",
         "hypothesis": "N (X_new N)^+ R keeps X_keep D ~ 0 and transfers to jointly random scenarios",
         "evidence": {"audit02_proxy": "separable conflict levels only; literal pinv form leaked 0.26"},
         "proposed_experiment": "jointly random scenarios over protected rank, conflict, samples, noise, condition, "
                                "target rank, off-span energy and precision; fixed methods vs a learned selector",
         "estimated_cost": {"cpu_seconds_reported": 819, "peak_rss_mib_reported": 270},
         "expected_informative_outcome": "preservation residual, held-out new-task R2, controller utility on held-out property combinations",
         "actual_result": {"n_scenarios": d["n_scenarios"],
                           "methods": {k: {m: round(v[m], 4) for m in ("utility", "r2", "prot_err") if m in v}
                                       for k, v in d["method_table_all"].items()},
                           "controller_heldout_seeds": {k: round(v, 4) for k, v in d["heldout_seeds"]["mean_utility"].items()},
                           "controller_heldout_property_combo": {k: round(v, 4) for k, v in d["heldout_property_combo"]["mean_utility"].items()},
                           "adaptive_rank": {k: d["adaptive_rank"][k] for k in ("adaptive_pooled_r2", "uniform_pooled_r2")}},
         "interpretation": "exact preservation holds; the cost is plasticity (null-space R2 0.14-0.26 vs 0.85 "
                           "unprotected); a tree selector beats fixed methods on held-out seeds but not on a held-out "
                           "property combination",
         "updated_hypothesis": "method choice needs out-of-distribution features or more data; reduced-rank null-space is the robust default",
         "decision": "keep null-space rank-2 as default; do not apply to Inkling without native activations",
         "alternatives": ["approximate protection at 90% energy (rejected: leak)", "plain pinv (rejected: unstable)"],
         "certificate": {"tests": "BPD test_nullspace.py 13/13 rerun by the campaign lead"}}
    src = {f: "transcribed:BPD-nullspace01/.../RESULTS.md" for f in FIELDS} | {
        "actual_result": f"mechanical:{p.name}", "interpretation": "curated", "updated_hypothesis": "curated",
        "decision": "curated", "question": "curated", "hypothesis": "curated"}
    return [make("bpd-nullspace-01", 0, v, src, {"repo": "BPD", "branch": "research/bpd-nullspace-01"}, "mixed")]


def metta_stop() -> list[dict]:
    rep = "DataGoblin@research/metta-stop-01:MeTTa/STOP_DIAGNOSIS.md"
    steps = [
        {"question": "Why does the MeTTa RAM hill climb never produce solved_and_stopped?",
         "hypothesis": "the stopping head stops too early", "evidence": {"audit02": "0/15 under every proposer, bundle size and grammar"},
         "proposed_experiment": "roll out with oracle stopping and with never-stop; teacher-forced edit accuracy per step",
         "estimated_cost": {"cases": 180}, "expected_informative_outcome": "solved under oracle stop",
         "actual_result": {"oracle_stop_solved_trm": "2/180", "oracle_stop_solved_ldt_proxy": "0/180",
                           "teacher_forced_edit_accuracy": 0.285, "train_edit_accuracy_16_steps": 0.358,
                           "train_edit_accuracy_200_steps": 0.373, "stop_rate_at_unsolved_states": "681/681"},
         "interpretation": "generation fails, not stopping: the student underfits and rarely reaches a solved state",
         "updated_hypothesis": "solution generation is the bottleneck", "decision": "test generation-side variants before stop variants",
         "alternatives": ["more data", "hybrid LDT-TRM controller"], "certificate": {"source": rep}},
        {"question": "Which architecture variant yields nonzero solved_and_stopped at matched compute?",
         "hypothesis": "an LDT-TRM hybrid or verifier-conditioned halt fixes the basin",
         "evidence": {"previous_step": "oracle-stop solves 0-2/180"},
         "proposed_experiment": "9 variants on 60 fresh roots x 3 seeds at 16 and 200 training steps",
         "estimated_cost": {"cases_per_variant": 180}, "expected_informative_outcome": "solved_and_stopped > 0",
         "actual_result": {"hybrid_16_steps": "1/180", "all_other_variants": "0/180"},
         "interpretation": "no variant is reliably above zero at matched compute", "updated_hypothesis": "capacity/data, not halting design",
         "decision": "climb from the hybrid with more steps; test a data lever",
         "alternatives": ["stop architecture search", "verifier-conditioned halt on the hybrid (not run)"],
         "certificate": {"source": rep}},
        {"question": "Does more compute or data move solved_and_stopped?",
         "hypothesis": "more steps and train roots reach a nonzero basin",
         "evidence": {"previous_step": "hybrid 1/180"},
         "proposed_experiment": "short climb from the hybrid (hidden 48, lr 0.3, 200 steps) evaluated on unseen fresh seeds",
         "estimated_cost": {"note": "not matched compute"}, "expected_informative_outcome": "solved_and_stopped on unseen seeds",
         "actual_result": {"solved_and_stopped": "6/180", "reached": "17/180", "premature_stops": 103,
                           "data_lever_100_roots_1000_steps": "5-6/180 (partial run)"},
         "interpretation": "the first nonzero basin appears only with extra compute; stopping errors remain dominant once generation succeeds",
         "updated_hypothesis": "resume hill climbing from the climbed hybrid with a verifier-conditioned halt",
         "decision": "resume architecture hill climbing from the climbed hybrid", "alternatives": ["data lever at matched compute"],
         "certificate": {"source": rep}},
    ]
    out = []
    for i, v in enumerate(steps):
        src = {f: "transcribed:" + rep for f in FIELDS} | {"interpretation": "curated", "updated_hypothesis": "curated",
                                                           "decision": "curated", "hypothesis": "curated"}
        out.append(make("metta-stop-01", i, v, src, {"repo": "DataGoblin", "branch": "research/metta-stop-01"},
                        "negative" if i < 2 else "positive"))
    return out


# ---- export --------------------------------------------------------------------------------------------
def next_action_row(t: dict) -> dict:
    ctx = {k: t[k] for k in PREFIX}
    target = {"proposed_experiment": t["proposed_experiment"], "expected_informative_outcome": t["expected_informative_outcome"],
              "alternatives": t["alternatives"]}
    mech = all(t["field_sources"][k].split(":")[0] in ("mechanical", "rule") for k in ("actual_result", "decision"))
    return {"id": _id("next_action", t["trajectory_id"]), "kind": "research", "root": t["campaign"],
            "family": "research_next_action", "split": "train", "task_digest": t["trajectory_id"],
            "input": {"exercise": "choose_next_experiment", **ctx}, "target": target,
            "loss": {"boundary": "response_only", "field": "target", "eligible": True},
            "source_model": {"provider": "campaign", "model": "receipt+curation"},
            "provider_execution_attested": False,
            "verification": {"outcome": t["outcome"], "result_mechanically_verified": mech,
                             "field_sources": t["field_sources"]}}


def build(root: Path, traces_dir: Path, out: Path, extra_trace_files: tuple = ()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tpath, rpath = out / "trajectories.jsonl", out / "next_action_rows.jsonl"
    tpath.write_text("", encoding="utf-8")
    rpath.write_text("", encoding="utf-8")
    counts: dict = {}
    seen = set()

    def emit(t):
        if t["trajectory_id"] in seen:
            return
        seen.add(t["trajectory_id"])
        append_jsonl(tpath, t)
        append_jsonl(rpath, next_action_row(t))
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1

    for p in sorted(traces_dir.glob("*.jsonl")) + sorted(Path(x) for x in extra_trace_files):
        if p.name == "research.jsonl":
            continue
        for s in iter_jsonl(p):
            emit(from_hillclimb_trace(s))
    for t in audit02_effort(root) + bpd_nullspace(root) + metta_stop():
        emit(t)
    man = {"schema": SCHEMA, "trajectories": len(seen), "by_outcome": counts,
           "field_source_tags": ["mechanical", "rule", "transcribed", "curated"]}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8", newline="\n")
    return man
