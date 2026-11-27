"""INK-FP-AUDIT-02 campaign driver. Resumable; every stage writes a receipt.

    python -m research_gym.ink_fp_audit_02.campaign --out runs/ink02 [--allow-remote]

Remote stages (A-F on Inkling) run only with --allow-remote, TINKER_API_KEY in the environment
and a passing resource preflight. Local stages always run: curriculum manifest and leakage audit,
upstream acceptance audit, synthetic BPD proxy, simulated-engine plumbing across arms A-H,
power preflight and the FSI-01 bridge check. Evidence classes are kept in separate files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from . import resources as res
from . import acceptance as acc
from . import bpd_proxy, curriculum, engine, gyms, preflight, stats
from .ledger import Ledger, append_jsonl

FIXTURE = Path(__file__).with_name("fixtures") / "upstream_gspo_gate_765453d.json"
COUNTS = {"sealed": 8, "accept": 8, "dev": 8, "train": 16}
SIM_GENERATIONS = 4


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")


def stage_done(state: dict, name: str) -> bool:
    return state.get("stages", {}).get(name, {}).get("status") in ("done", "blocked", "unsupported")


def run(out: Path, allow_remote: bool, limits: res.Limits = res.Limits()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"stages": {}}
    ledger = Ledger(out / "campaign_ledger.jsonl")
    monitor = res.ResourceMonitor(limits)

    def mark(name, status, **info):
        state["stages"][name] = {"status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}
        write(state_path, state)
        ledger.append("stage", {"stage": name, "status": status, **info})

    # 0. preflight ------------------------------------------------------------------------
    r, api = preflight.resource_preflight(limits), preflight.api_preflight()
    g = preflight.gates(r, api)
    if not allow_remote:
        g["blocked_reasons"].append("--allow-remote not given")
        g["remote_stages_allowed"] = False
    write(out / "preflight.json", {"resources": r, "api": api, "gates": g})
    mark("preflight", "done", remote_allowed=g["remote_stages_allowed"])
    monitor.gate()

    # local: curriculum ----------------------------------------------------------------------
    if not stage_done(state, "curriculum"):
        P = curriculum.build_partitions(list(gyms.FAMILIES), COUNTS)
        leak = curriculum.leakage_report(P["partitions"])
        man = curriculum.manifest(list(gyms.FAMILIES), COUNTS)
        man["unavailable_families"] = gyms.UNAVAILABLE
        man["sealed_task_ids_sha256"] = __import__("hashlib").sha256("\n".join(P["sealed_task_ids"]).encode()).hexdigest()
        man["shortfall"] = P["shortfall"]
        write(out / "curriculum_manifest.json", man)
        write(out / "leakage_report.json", leak)
        for part, ts in P["partitions"].items():
            for t in ts:
                append_jsonl(out / f"tasks_{part}.jsonl", {**t.receipt(), "prompt": t.prompt})
        mark("curriculum", "done", leakage_ok=leak["ok"], counts=leak["counts"])
    monitor.gate()

    # local: upstream audit -----------------------------------------------------------------
    if not stage_done(state, "upstream_audit"):
        rows = acc.audit_upstream(json.loads(FIXTURE.read_text(encoding="utf-8")))
        write(out / "upstream_audit.json", {"rows": rows, "discrepancies": sum(r["discrepancy"] for r in rows),
                                            "evidence_class": "published_record_replay"})
        mark("upstream_audit", "done", discrepancies=sum(r["discrepancy"] for r in rows))

    # local: BPD synthetic proxy ------------------------------------------------------------
    if not stage_done(state, "bpd_proxy"):
        monitor.gate()
        rep = bpd_proxy.study(range(20), out / "synthetic_cpu" / "bpd_proxy.json")
        mark("bpd_proxy", "done", evidence_class=rep["evidence_class"])

    # local: simulated engine across arms -----------------------------------------------------
    if not stage_done(state, "simulated_engine"):
        train = [curriculum.Task(**{k: v for k, v in rec.items() if k in curriculum.Task.__dataclass_fields__})
                 for rec in _load(out / "tasks_train.jsonl")]
        accept = [curriculum.Task(**{k: v for k, v in rec.items() if k in curriculum.Task.__dataclass_fields__})
                  for rec in _load(out / "tasks_accept.jsonl")]
        results = {}
        for name, arm in engine.ARMS.items():
            monitor.gate()
            backend = engine.SimulatedBackend(list(gyms.FAMILIES), seed=17)
            rr = engine.run_arm(arm, backend, train, accept, out / "engineering_simulation" / f"arm_{name}",
                                SIM_GENERATIONS, monitor=monitor)
            results[name] = {k: v for k, v in rr.items() if k != "replay"}
            if "replay" in rr:
                results[name]["replay_mismatches"] = len(rr["replay"]["mismatches"])
                results[name]["decisions"] = [x["accepted"] for x in rr["replay"]["rows"]]
            results[name]["simulated_policy_calls"] = backend.calls
        write(out / "engineering_simulation" / "summary.json",
              {"evidence_class": "engineering_simulation", "note": "toy policy; no model evidence", "arms": results})
        mark("simulated_engine", "done", arms=sorted(results))

    # local: power preflight + FSI bridge check -----------------------------------------------
    if not stage_done(state, "stats_preflight"):
        pp = {"primary_D_vs_A": stats.power_preflight(min_effect=0.10, comparisons=1),
              "all_arms_vs_A_bonferroni": stats.power_preflight(min_effect=0.10, comparisons=7)}
        write(out / "power_preflight.json", pp)
        bridge = _fsi_bridge_check(out)
        mark("stats_preflight", "done", required_roots=pp["primary_D_vs_A"]["required_independent_roots"],
             fsi_bridge=bridge.get("verdict", bridge.get("error")))

    # remote stages ------------------------------------------------------------------------
    for name in ("A_api_smoke", "B_adapter_qualification", "C_curriculum_pilot", "D_recursive_improvement",
                 "E_independent_evaluation", "F_adaptive_curriculum"):
        if stage_done(state, name) and state["stages"][name]["status"] != "blocked":
            continue
        if not g["remote_stages_allowed"]:
            mark(name, "blocked", reasons=g["blocked_reasons"], resumable=True)
            continue
        if name == "A_api_smoke":
            _stage_a(out, monitor)
            mark(name, "done")
        else:
            mark(name, "blocked", reasons=[f"{name} requires a passing Stage A and B receipt reviewed first"],
                 resumable=True)
            break

    write(out / "resources.json", monitor.receipt())
    mark("resources", "done", peak_rss=monitor.peak_tree_rss, min_available=monitor.governor.min_available)
    return state


def _load(path):
    from .ledger import iter_jsonl
    return list(iter_jsonl(path))


def _fsi_bridge_check(out: Path) -> dict:
    """End-to-end check of the FSI-01 bridge on simulated dev outcomes (plumbing only)."""
    dev = [curriculum.Task(**{k: v for k, v in rec.items() if k in curriculum.Task.__dataclass_fields__})
           for rec in _load(out / "tasks_dev.jsonl")]
    b = engine.SimulatedBackend(list(gyms.FAMILIES), seed=17)
    b.store["trained"] = {f: v + 0.3 for f, v in b.skill.items()}
    base, cand = b.evaluate(dev, "base"), b.evaluate(dev, "trained")
    cases = out / "engineering_simulation" / "fsi_bridge_cases.jsonl"
    cases.unlink(missing_ok=True)
    for x, y, t in zip(base, cand, dev):
        append_jsonl(cases, {"case_id": t.task_id, "root_id": t.source_world,
                             "candidate_correct": int(y["correct"]), "control_correct": int(x["correct"])})
    reg = out / "engineering_simulation" / "fsi_bridge_registration.json"
    write(reg, stats.registration("ink-fp-audit-02-bridge-check-simulated", 0.10, 1,
                                  population="simulated dev roots (plumbing check only)",
                                  preregistered=False, sealed=False))
    try:
        rep = stats.fsi_report(cases, reg, out / "engineering_simulation" / "fsi_bridge_report.json")
        return {"verdict": rep.get("verdict"), "n_clusters": rep.get("n_clusters")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _stage_a(out: Path, monitor) -> None:
    from research_gym.ink_fp_audit_01 import tinker_remote as r01
    monitor.gate()
    r01.run(r01.plan(2, 512), out / "stage_a_smoke", os.environ["TINKER_API_KEY"], delay=1.0)
    monitor.api_requests += 6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--no-job-object", action="store_true", help="skip the 4 GiB Job Object (non-Windows)")
    a = ap.parse_args()
    limits = res.Limits()
    res.set_below_normal_priority()
    job = None
    if not a.no_job_object:
        job = res.JobObject(limits.job_memory_cap)
        job.assign_current()
    try:
        state = run(a.out, a.allow_remote, limits)
        if job is not None:
            write(a.out / "job_object.json", {"memory_cap_bytes": job.memory_cap,
                                              "peak_job_memory_bytes": job.peak_job_memory()})
    finally:
        if job is not None:
            job.close()
    print(json.dumps({k: v["status"] for k, v in state["stages"].items()}, indent=1))


if __name__ == "__main__":
    main()
