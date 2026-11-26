"""Objective 1: Exploratory search-limitation diagnostics on SPENT cohorts only.

Diagnoses whether low closure arises from:
  1. Action coverage (tactic grammar vs human proof form),
  2. Premise retrieval (Lean index recall @ 24/40/200 + same-file recency recall + premise timeouts),
  3. Depth-3 restrictions (paired DEPTH=3 vs DEPTH=5 at matched step budgets),
  4. Frontier ordering (S1/S2/S3/S3c, fixed-budget portfolios, retrospective oracles, eager vs lazy),
  5. Verifier costs (open/premises/step/cert raw heartbeats, server ms, wall time, wasted unpopped calls).

Strictly restricted to SPENT cohorts (OLR-08A spent TRAIN/CAL and authenticated referee archive).
NEVER opens any unspent file, the protected 275-root evaluation registry, or sealed SInt.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import sys
import time
import zipfile

OLR08_DIR = Path(__file__).resolve().parent.parent / "olr08"
if str(OLR08_DIR) not in sys.path:
    sys.path.insert(0, str(OLR08_DIR))

import probe  # noqa: E402
import run08  # noqa: E402
from search08 import FREE, RANKED, actions, reprice  # noqa: E402

BUDGETS = (75, 150, 300, 600, 1200)
DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:protected\s+|private\s+|noncomputable\s+)*(?:theorem|lemma)\s+([^\s:({\[]+)"
)
HUMAN_ARGS_RE = re.compile(
    r"\s*(simp|simp_all|simpa|rw|rwa|simp only|simp_arith|dsimp)\b[^\[]*\[(.*?)\]", re.S
)


def human_names(text: str) -> list[str] | None:
    m = HUMAN_ARGS_RE.match(text)
    if not m:
        return None
    out: list[str] = []
    for part in m.group(2).split(","):
        x = part.strip().lstrip("←↓↑ ").strip()
        if not x or x == "*" or x.startswith("-"):
            continue
        x = re.split(r"[\s(]", x)[0].lstrip("@")
        if re.fullmatch(r"[A-Za-z_][\w.'!?]*", x):
            out.append(x)
    return out


def classify_human(text: str) -> str:
    t = text.strip()
    if "<;>" in t:
        return "combinator_all_goals"
    if "\n" in t or ";" in t:
        return "multi_step_seq"
    m = HUMAN_ARGS_RE.match(t)
    if m:
        names = human_names(t) or []
        if len(names) >= 2:
            return "multi_lemma_simp_rw"
        if len(names) == 1:
            return "single_lemma_simp_rw"
        return "simp_rw_star_or_other"
    if re.match(r"^(exact|refine)\b", t):
        return "term_exact_refine"
    if re.match(r"^apply\b", t):
        return "apply_term"
    if re.match(r"^(induction|cases|rcases|obtain|match|split)\b", t):
        return "case_or_induction"
    if re.match(r"^(have|let|suffices)\b", t):
        return "sublemma_have_let"
    if re.match(r"^(intro|rintro|ext|funext|congr|subst)\b", t):
        return "structural_step"
    return "other"


def name_hit(name: str, pool: list[str]) -> bool:
    return any(c == name or c.endswith("." + name) or name.endswith("." + c) for c in pool)


def prior_decls_in_file(source: Path, rel: str, line_no: int, own_decl: str) -> list[str]:
    """Earlier theorem/lemma names declared in the same file (public pre-command order)."""
    lines = (source / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
    ns_stack: list[str] = []
    out: list[str] = []
    for idx, line in enumerate(lines, 1):
        if idx >= line_no:
            break
        s = line.strip()
        m_ns = re.match(r"^namespace\s+([^\s]+)", s)
        if m_ns:
            ns_stack.append(m_ns.group(1))
            continue
        m_end = re.match(r"^end\s+([^\s]+)", s)
        if m_end and ns_stack:
            ns_stack.pop()
            continue
        m = DECL_RE.match(line)
        if m:
            short = m.group(1)
            full = ".".join([*ns_stack, short]) if ns_stack else short
            if full != own_decl and not own_decl.endswith("." + short) and short != own_decl:
                out.append(full)
    return out


def fate_of(res: dict) -> str:
    if res["solved"]:
        return "solved"
    m = res["meter"]
    lim = m["limits"]
    if m["steps"] >= lim[0]:
        return "censored_steps"
    if m["rpcs"] >= lim[1]:
        return "censored_rpcs"
    if m["hb_raw"] >= lim[2]:
        return "censored_hb"
    return "exhausted_finite_action_space"


def search_diag(
    exe: Path,
    source: Path,
    row: dict,
    arm: str,
    prior: dict | None = None,
    *,
    limit: int = 1200,
    max_depth: int = 3,
    k_rw: int = 24,
    k_apply: int = 6,
    recent_lemmas: list[str] | None = None,
    lazy_expand: bool = False,
) -> dict:
    """Native Lean search with full RPC, depth, and unpopped-state accounting."""
    start = time.perf_counter()
    scale = max(1, math.ceil(limit / run08.STEP_CAP))
    with run08.Lean(exe) as L:
        s, why, opening = run08.open_root(L, source, row)
        if s is None:
            raise ValueError(f"spent root unavailable: {row['id']} {why}")
        m = run08.Meter(L, steps=limit, rpcs=run08.RPC_CAP * scale, hb=run08.HB_CAP * scale)
        heap: list = []
        serial = 0
        events: list[dict] = []
        expanded_sids: set[int] = set()
        popped_sids: set[int] = set()
        winner_path: list[str] = []
        winner_on_path: list[int] = []
        solved = False

        def build_actions(state: dict) -> list[tuple]:
            p = m.call("premises", sid=state["sid"], max_rw=max(40, k_rw), max_apply=max(10, k_apply))
            if (
                p.get("ok") is False
                and "maximum number of heartbeats" in str(p.get("error", ""))
                and "rw" not in p
                and "apply" not in p
            ):
                p = {"rw": [], "apply": []}
            if p.get("error") or not all(k in p for k in ("rw", "apply")):
                raise ValueError("premise enumeration failed: " + str(p)[:240])
            acts = actions(state["info"], p, k_rw=k_rw, k_apply=k_apply)
            if recent_lemmas:
                for i, lem in enumerate(recent_lemmas):
                    acts.append((3.25 + i / 6.0, "simp_lemma", lem, f"simp [{lem}]"))
                    acts.append((3.9 + i / 6.0, "rw_lemma", lem, f"rw [{lem}]"))
            seen: set[str] = set()
            dedup = []
            for a in acts:
                if a[-1] not in seen:
                    seen.add(a[-1])
                    dedup.append(a)
            return dedup

        def expand(state: dict, depth: int, g_cost: float, path: tuple, event_ids: tuple) -> None:
            nonlocal serial
            expanded_sids.add(state["sid"])
            acts = build_actions(state)
            costs = reprice(acts, prior, depth) if prior is not None else acts
            if arm == "s1":
                costs = [(1.0, *a[1:]) for a in acts]
            ranks: collections.Counter = collections.Counter()
            for c, k, arg, tac in costs:
                rk = ranks[k]
                ranks[k] += 1
                extra = 0.0
                if arm == "s3c" and prior is not None:
                    extra = math.log2(max(1.0, prior["hb"].get(tac if k in FREE else k, prior["hb_mean"])))
                serial += 1
                heapq.heappush(
                    heap,
                    (g_cost + c + extra, 0, serial, state["sid"], depth, g_cost + c, path, event_ids, k, arg, tac, rk),
                )

        try:
            expand(s, 0, 0.0, (), ())
            while heap and not m.exhausted():
                item = heapq.heappop(heap)
                if item[1] == 1:
                    # Lazy state-expansion entry
                    _, _, _, child_state, depth, g_cost, path, event_ids = item
                    expand(child_state, depth, g_cost, path, event_ids)
                    continue
                _, _, _, sid, d, g_cost, path, ids, k, arg, tac, rk = item
                popped_sids.add(sid)
                a = m.call("step", sid=sid, tactic=tac, hb=500 if tac == "exact?" else 200)
                idx = len(events)
                ev = {
                    "kind": k,
                    "argument": arg,
                    "tactic": tac,
                    "rank": rk,
                    "depth": d,
                    "ok": a.get("ok") is True,
                    "ngoals": a.get("ngoals"),
                    "hb_raw": a["request_hb_raw"],
                    "parent_events": list(ids),
                }
                events.append(ev)
                if not ev["ok"]:
                    continue
                new_path = (*path, tac)
                new_ids = (*ids, idx)
                if a["ngoals"] == 0:
                    cert = m.call("cert", sid=a["sid"]).get("cert", {})
                    if run08.close_cert(cert):
                        solved = True
                        winner_path = list(new_path)
                        winner_on_path = list(new_ids)
                        break
                elif d + 1 < max_depth:
                    if lazy_expand:
                        serial += 1
                        heapq.heappush(heap, (g_cost + 1.0, 1, serial, a, d + 1, g_cost, new_path, new_ids))
                    else:
                        expand(a, d + 1, g_cost, new_path, new_ids)
        except run08.Exhausted:
            pass
        work = m.report()
        unpopped_expansions = len(expanded_sids - popped_sids)
        result = {
            "id": row["id"],
            "role": row["role"],
            "family": row["family"],
            "file": row["file"],
            "decl": row["decl"],
            "set": row["set"],
            "arm": arm,
            "max_depth": max_depth,
            "lazy_expand": lazy_expand,
            "recent_k": len(recent_lemmas or []),
            "solved": solved,
            "path": winner_path,
            "on_path": winner_on_path,
            "steps": work["steps"],
            "expanded_states": len(expanded_sids),
            "popped_states": len(popped_sids),
            "unpopped_expansions": unpopped_expansions,
            "events_ok": sum(1 for e in events if e["ok"]),
            "events_total": len(events),
            "depth_counts": dict(sorted(collections.Counter(e["depth"] for e in events).items())),
            "open": opening,
            "meter": work,
            "elapsed": time.perf_counter() - start,
        }
    if solved:
        _, audit = run08.replay(exe, source, row, winner_path)
        result["replay"] = audit
    result["fate"] = fate_of(result)
    return result


def premise_probe_root(exe: Path, source: Path, row: dict) -> dict:
    """Measure premise retrieval recall @ 24/40/200 and same-file recency on a spent root."""
    names = human_names(row["human"])
    form = classify_human(row["human"])
    rec = prior_decls_in_file(source, row["file"], row["line"], row["decl"])
    with run08.Lean(exe) as L:
        s, why, _ = run08.open_root(L, source, row)
        if s is None:
            return {"id": row["id"], "decl": row["decl"], "ok": False, "why": why, "form": form}
        p = L.request(op="premises", sid=s["sid"], max_rw=200, max_apply=50)
        timed_out = p.get("ok") is False and "maximum number of heartbeats" in str(p.get("error", ""))
        rw = [x["name"] for x in p.get("rw", [])] if p.get("ok") else []
        ap = [x["name"] for x in p.get("apply", [])] if p.get("ok") else []
        local = [h["name"] for h in (s.get("info") or {}).get("hyps", [])] + (s.get("info") or {}).get("defs", [])
        lib = [n for n in (names or []) if not name_hit(n, local)]
        return {
            "id": row["id"],
            "role": row["role"],
            "file": row["file"],
            "decl": row["decl"],
            "set": row["set"],
            "ok": True,
            "form": form,
            "premise_timeout": timed_out,
            "premise_hb_raw": p.get("request_hb_raw", 0),
            "rw_count": len(rw),
            "apply_count": len(ap),
            "n_human_names": len(names or []),
            "n_lib": len(lib),
            "lib_names": lib,
            "in24": [name_hit(n, rw[:24] + ap[:6]) for n in lib],
            "in40": [name_hit(n, rw[:40] + ap[:10]) for n in lib],
            "in200": [name_hit(n, rw + ap) for n in lib],
            "in_recent16": [name_hit(n, rec[-16:]) for n in lib],
            "in_same_file": [name_hit(n, rec) for n in lib],
            "in24_or_recent16": [name_hit(n, rw[:24] + ap[:6] + rec[-16:]) for n in lib],
        }


def run_cached_stage(out: Path, name: str, rows: list[dict], fn, workers: int) -> list[dict]:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    def work(row: dict) -> dict:
        p = d / f"{row['id']}.json"
        seal = p.with_suffix(".sha256")
        if p.exists() and seal.exists():
            if seal.read_text().strip() != run08.sha(p):
                raise ValueError(f"checkpoint tampered: {p}")
            return run08.read(p)
        v = fn(row)
        run08.write(p, v)
        seal.write_text(run08.sha(p) + "\n")
        return v

    res: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, v in enumerate(pool.map(work, rows)):
            res.append(v)
            if (i + 1) % 4 == 0 or i + 1 == len(rows):
                print(f"[obj1] {name}: {i+1}/{len(rows)}", flush=True)
    return res


def simulate_portfolios_and_oracles(arm_rows: dict[str, list[dict]], budgets=BUDGETS) -> dict:
    """Compute paired closure curves at 75/150/300/600/1200 for arms, portfolios, and oracles."""
    by = {a: {r["id"]: r for r in rs} for a, rs in arm_rows.items()}
    ids = sorted(by["s1"].keys())
    n = len(ids)

    def rate_curve(steps_fn) -> dict[str, dict]:
        out = {}
        for b in budgets:
            sol = sum(1 for i in ids if steps_fn(i) is not None and steps_fn(i) <= b)
            out[str(b)] = {"solved": sol, "roots": n, "rate": sol / max(1, n)}
        return out

    curves: dict[str, dict] = {}
    for a in ("s1", "s2", "s3", "s3c"):
        curves[a] = rate_curve(lambda i, a=a: by[a][i]["steps"] if by[a][i]["solved"] else None)

    # Fixed-budget portfolios (deterministic interleaving)
    portfolios = {
        "portfolio_rr_s2_s3": ["s2", "s3"],
        "portfolio_rr_s2_s3_s3c": ["s2", "s3", "s3c"],
        "portfolio_rr_all4": ["s1", "s2", "s3", "s3c"],
        "portfolio_weighted_s2_s3_s3c": ["s3", "s2", "s3", "s3c"],
    }
    for pname, sched in portfolios.items():
        # Count occurrences of each arm in one cycle of length L
        L = len(sched)
        arm_pos: dict[str, list[int]] = collections.defaultdict(list)
        for idx, a in enumerate(sched):
            arm_pos[a].append(idx)

        def p_steps(i: str, arm_pos=arm_pos, L=L) -> int | None:
            best = None
            for a, positions in arm_pos.items():
                r = by[a][i]
                if not r["solved"]:
                    continue
                k = r["steps"]  # 1-indexed step in arm a
                m_per_cyc = len(positions)
                cyc, rem = divmod(k - 1, m_per_cyc)
                tot = cyc * L + positions[rem] + 1
                if best is None or tot < best:
                    best = tot
            return best

        curves[pname] = rate_curve(p_steps)

    # Retrospective oracle references (non-deployable hindsight ceilings)
    def oracle_min_steps(i: str) -> int | None:
        vals = [by[a][i]["steps"] for a in ("s1", "s2", "s3", "s3c") if by[a][i]["solved"]]
        return min(vals) if vals else None

    curves["oracle_best_arm_at_budget"] = rate_curve(oracle_min_steps)
    oracle_1200 = sum(1 for i in ids if oracle_min_steps(i) is not None and oracle_min_steps(i) <= 1200)
    curves["oracle_union_1200"] = {"solved": oracle_1200, "roots": n, "rate": oracle_1200 / max(1, n)}

    # Paired headroom vs strongest 300-step single arm and strongest 300-step portfolio
    best_single_300 = max(("s1", "s2", "s3", "s3c"), key=lambda a: curves[a]["300"]["solved"])
    best_port_300 = max(portfolios.keys(), key=lambda p: curves[p]["300"]["solved"])
    best_ctrl_300 = max(
        (*("s1", "s2", "s3", "s3c"), *portfolios.keys()),
        key=lambda k: curves[k]["300"]["solved"],
    )
    headroom = {
        "best_single_300": best_single_300,
        "best_single_300_rate": curves[best_single_300]["300"]["rate"],
        "best_portfolio_300": best_port_300,
        "best_portfolio_300_rate": curves[best_port_300]["300"]["rate"],
        "best_control_300": best_ctrl_300,
        "best_control_300_rate": curves[best_ctrl_300]["300"]["rate"],
        "s3_1200_minus_best_control_300_pp": 100.0 * (curves["s3"]["1200"]["rate"] - curves[best_ctrl_300]["300"]["rate"]),
        "best_arm_1200_minus_best_control_300_pp": 100.0 * (
            max(curves[a]["1200"]["rate"] for a in ("s1", "s2", "s3", "s3c")) - curves[best_ctrl_300]["300"]["rate"]
        ),
        "oracle_1200_minus_best_control_300_pp": 100.0 * (
            curves["oracle_union_1200"]["rate"] - curves[best_ctrl_300]["300"]["rate"]
        ),
        "oracle_300_minus_best_control_300_pp": 100.0 * (
            curves["oracle_best_arm_at_budget"]["300"]["rate"] - curves[best_ctrl_300]["300"]["rate"]
        ),
    }
    return {"curves": curves, "headroom": headroom}


def summarize_costs(rows: list[dict]) -> dict:
    ops = collections.Counter()
    hb_by_op = collections.Counter()
    ms_by_op = collections.Counter()
    fates = collections.Counter(r["fate"] for r in rows)
    total_exp = sum(r["expanded_states"] for r in rows)
    total_unpop = sum(r["unpopped_expansions"] for r in rows)
    total_ok = sum(r["events_ok"] for r in rows)
    total_ev = sum(r["events_total"] for r in rows)
    prem_timeouts = 0
    for r in rows:
        ops.update(r["meter"]["ops"])
        for j in r["meter"]["rpc_journal"]:
            hb_by_op[j["op"]] += j["hb"]
            ms_by_op[j["op"]] += j["server_ms"]
            if j["op"] == "premises" and j.get("ok") is False:
                prem_timeouts += 1
    tot_hb = max(1, sum(hb_by_op.values()))
    return {
        "roots": len(rows),
        "solved": sum(r["solved"] for r in rows),
        "fates": dict(fates),
        "ops": dict(ops),
        "hb_total": sum(hb_by_op.values()),
        "hb_by_op": dict(hb_by_op),
        "hb_share_by_op": {k: round(v / tot_hb, 4) for k, v in hb_by_op.items()},
        "server_ms_by_op": dict(ms_by_op),
        "open_seconds": round(sum((r.get("open") or {}).get("wall_seconds", 0.0) for r in rows), 3),
        "native_seconds": round(sum(r["meter"]["native_seconds"] for r in rows), 3),
        "elapsed_seconds": round(sum(r["elapsed"] for r in rows), 3),
        "events_ok": total_ok,
        "events_total": total_ev,
        "ok_rate": round(total_ok / max(1, total_ev), 4),
        "expanded_states": total_exp,
        "unpopped_expansions": total_unpop,
        "unpopped_fraction": round(total_unpop / max(1, total_exp), 4),
        "premise_timeouts": prem_timeouts,
    }


def analyze_referee_archive(ref_zip: Path, source: Path) -> dict:
    """Offline diagnostic audit of all 482 valid spent hard roots in olr08_referee.zip."""
    ref_sha = run08.sha(ref_zip)
    if ref_sha != "23225d1ddb28e1f91e013d309a119425649e10ca9a2e7ba0e5521c847eb42e7b":
        raise ValueError(f"Referee archive SHA-256 mismatch: {ref_sha}")
    with zipfile.ZipFile(ref_zip) as z:
        reg = {r["id"]: r for r in json.loads(z.read("olr08_referee/results/registry.json"))}
        cm = json.loads(z.read("olr08_referee/results/clean_matrix.json"))
        pr = json.loads(z.read("olr08_referee/results/premise_recall.json"))
        valid = {r["id"] for role in cm["T"] for r in cm["T"][role] if r.get("human_closed")} | {
            r["id"] for role in cm["R"] for r in cm["R"][role]
        }
        arms = {}
        for k, dname in (("s1", "search_uniform"), ("s2", "search"), ("s3", "search_s3"), ("s3c", "search_s3c")):
            rows = []
            for n in sorted(z.namelist()):
                if n.startswith(f"olr08_referee/results/{dname}/") and n.endswith(".json"):
                    rows.extend(json.loads(z.read(n)))
            arms[k] = {r["id"]: r for r in rows if r["id"] in valid}

    out: dict = {"referee_sha256": ref_sha, "roles": {}, "premise_recall": {}, "action_coverage": {}}
    for role in ("calibration", "evaluation"):
        ids = sorted(i for i in arms["s3"] if arms["s3"][i]["role"] == role)
        n = len(ids)
        role_d: dict = {"roots": n, "arms": {}, "portfolios": {}, "oracle": {}}
        for a in ("s1", "s2", "s3", "s3c"):
            role_d["arms"][a] = {
                str(b): {
                    "solved": sum(arms[a][i]["solved"] and arms[a][i]["calls"] <= b for i in ids),
                    "rate": sum(arms[a][i]["solved"] and arms[a][i]["calls"] <= b for i in ids) / max(1, n),
                }
                for b in (75, 150, 300)
            }
        for pname, sched in (
            ("portfolio_rr_s2_s3", ["s2", "s3"]),
            ("portfolio_rr_all4", ["s1", "s2", "s3", "s3c"]),
            ("portfolio_weighted_s2_s3_s3c", ["s3", "s2", "s3", "s3c"]),
        ):
            L = len(sched)
            arm_pos = collections.defaultdict(list)
            for idx, a in enumerate(sched):
                arm_pos[a].append(idx)
            role_d["portfolios"][pname] = {}
            for b in (75, 150, 300):
                sol = 0
                for i in ids:
                    for a, positions in arm_pos.items():
                        r = arms[a][i]
                        if not r["solved"]:
                            continue
                        cyc, rem = divmod(r["calls"] - 1, len(positions))
                        if cyc * L + positions[rem] + 1 <= b:
                            sol += 1
                            break
                role_d["portfolios"][pname][str(b)] = {"solved": sol, "rate": sol / max(1, n)}
        for b in (75, 150, 300):
            ora = sum(any(arms[a][i]["solved"] and arms[a][i]["calls"] <= b for a in ("s1", "s2", "s3", "s3c")) for i in ids)
            role_d["oracle"][str(b)] = {"solved": ora, "rate": ora / max(1, n)}
        out["roles"][role] = role_d

        # Action coverage breakdown on referee spent roots
        cov_all = collections.Counter()
        cov_sol = collections.Counter()
        for i in ids:
            form = classify_human(reg[i]["human"])
            cov_all[form] += 1
            cov_sol[form] += int(any(arms[a][i]["solved"] for a in ("s1", "s2", "s3", "s3c")))
        out["action_coverage"][role] = {
            f: {"roots": cov_all[f], "solved_any": cov_sol[f], "rate": round(cov_sol[f] / max(1, cov_all[f]), 4)}
            for f in sorted(cov_all)
        }

    # Premise + same-file recency recall
    for role in ("calibration", "evaluation"):
        x = [r for r in pr if r["role"] == role]
        ar = [r for r in x if r["form"] == "args" and r["n_lib"] > 0]
        rec_hit16 = 0
        rec_tot = 0
        union24_16 = 0
        for r in ar:
            g = reg[r["id"]]
            hnames = human_names(g["human"]) or []
            if len(hnames) != len(r["in24"]):
                continue
            prev = prior_decls_in_file(source, g["file"], g["line"], g["decl"])
            for nm, f24 in zip(hnames, r["in24"]):
                rec_tot += 1
                h16 = name_hit(nm, prev[-16:])
                rec_hit16 += int(h16)
                union24_16 += int(h16 or f24)
        out["premise_recall"][role] = {
            "T_hard": len(x),
            "names_library_lemmas": len(ar),
            "in24_all_rate": round(sum(all(r["in24"]) for r in ar) / max(1, len(ar)), 4),
            "in40_all_rate": round(sum(all(r["in40"]) for r in ar) / max(1, len(ar)), 4),
            "in200_all_rate": round(sum(all(r["in200"]) for r in ar) / max(1, len(ar)), 4),
            "in24_lemma_recall": round(sum(sum(r["in24"]) for r in ar) / max(1, sum(len(r["in24"]) for r in ar)), 4),
            "in200_lemma_recall": round(sum(sum(r["in200"]) for r in ar) / max(1, sum(len(r["in200"]) for r in ar)), 4),
            "same_file_last16_lemma_recall": round(rec_hit16 / max(1, rec_tot), 4),
            "in24_or_last16_lemma_recall": round(union24_16 / max(1, rec_tot), 4),
            "recency_lemma_count": rec_tot,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--audit-zip", type=Path, required=True, help="results/olr08a/olr08-full_AUDIT.zip")
    ap.add_argument("--referee-zip", type=Path, required=True, help="authenticated olr08_referee.zip")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train-sample", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.audit_zip) as z:
        reg = json.loads(z.read("registry.json"))
        prior = json.loads(z.read("prior.json"))
        by_id = {r["id"]: r for r in reg}
        # Spent OLR-08A hard roots ONLY (18 CAL + 106 TRAIN); NEVER touch evaluation!
        cal_hard_ids = sorted(
            Path(n).stem for n in z.namelist() if n.startswith("cal_s2/") and n.endswith(".json")
        )
        tr_hard_ids = sorted(
            Path(n).stem for n in z.namelist() if n.startswith("source_s2/") and n.endswith(".json")
        )
        s2_tr_prev = {
            Path(n).stem: json.loads(z.read(n))
            for n in z.namelist()
            if n.startswith("source_s2/") and n.endswith(".json")
        }

    cal_rows = [by_id[i] for i in cal_hard_ids]
    # Deterministic stratified selection of n_train_sample spent TRAIN hard roots
    tr_sorted = sorted(tr_hard_ids, key=lambda i: run08.digest([run08.SEED, "obj1_diag", i]))
    tr_sample_rows = [by_id[i] for i in tr_sorted[: args.n_train_sample]]
    diag_rows = cal_rows + tr_sample_rows

    # Safety assertion: zero evaluation roots, zero SInt
    for r in diag_rows:
        if r["role"] == "evaluation" or "/SInt" in r["file"] or r["family"] == "SInt":
            raise RuntimeError("Contamination guard triggered: non-spent root in Objective 1 diagnostics!")

    print(
        f"[obj1] Spent diagnostic cohort: {len(cal_rows)} OLR-08A spent CAL + "
        f"{len(tr_sample_rows)} OLR-08A spent TRAIN = {len(diag_rows)} roots across "
        f"{len({r['file'] for r in diag_rows})} files",
        flush=True,
    )

    # 1. Paired 1200-step search across S1, S2, S3, S3c (depth=3)
    arm_rows: dict[str, list[dict]] = {}
    for arm in ("s1", "s2", "s3", "s3c"):
        arm_rows[arm] = run_cached_stage(
            args.out,
            f"diag_{arm}_1200",
            diag_rows,
            lambda r, a=arm: search_diag(
                args.exe, args.source, r, a, prior if a in ("s3", "s3c") else None, limit=1200, max_depth=3
            ),
            args.workers,
        )

    # 2. Controlled mechanism probes on 18 spent CAL + 6 spent TRAIN unsolved-at-300 (24 roots)
    tr_unsolved = [r for r in tr_sample_rows if not s2_tr_prev[r["id"]]["solved"]][:6]
    probe_rows = cal_rows + tr_unsolved
    depth5_s2 = run_cached_stage(
        args.out,
        "probe_depth5_s2_300",
        probe_rows,
        lambda r: search_diag(args.exe, args.source, r, "s2", None, limit=300, max_depth=5),
        args.workers,
    )
    depth5_s3 = run_cached_stage(
        args.out,
        "probe_depth5_s3_600",
        probe_rows,
        lambda r: search_diag(args.exe, args.source, r, "s3", prior, limit=600, max_depth=5),
        args.workers,
    )
    recency_s3 = run_cached_stage(
        args.out,
        "probe_recency16_s3_300",
        probe_rows,
        lambda r: search_diag(
            args.exe,
            args.source,
            r,
            "s3",
            prior,
            limit=300,
            max_depth=3,
            recent_lemmas=prior_decls_in_file(args.source, r["file"], r["line"], r["decl"])[-16:],
        ),
        args.workers,
    )
    lazy_s3 = run_cached_stage(
        args.out,
        "probe_lazy_s3_300",
        probe_rows,
        lambda r: search_diag(args.exe, args.source, r, "s3", prior, limit=300, max_depth=3, lazy_expand=True),
        args.workers,
    )

    # 3. Premise retrieval & action coverage audit on all 48 spent roots
    prem_rows = run_cached_stage(
        args.out,
        "probe_premises",
        diag_rows,
        lambda r: premise_probe_root(args.exe, args.source, r),
        args.workers,
    )

    # 4. Aggregate everything
    cal_set = {r["id"] for r in cal_rows}
    probe_set = {r["id"] for r in probe_rows}

    def filter_arms(id_set: set[str]) -> dict[str, list[dict]]:
        return {a: [r for r in rs if r["id"] in id_set] for a, rs in arm_rows.items()}

    # Premise summary on diag_rows
    p_lib = [r for r in prem_rows if r.get("ok") and r.get("n_lib", 0) > 0]
    prem_summary = {
        "roots": len(prem_rows),
        "premise_timeouts": sum(1 for r in prem_rows if r.get("premise_timeout")),
        "roots_naming_lib_lemmas": len(p_lib),
        "in24_all": sum(all(r["in24"]) for r in p_lib),
        "in40_all": sum(all(r["in40"]) for r in p_lib),
        "in200_all": sum(all(r["in200"]) for r in p_lib),
        "in_recent16_all": sum(all(r["in_recent16"]) for r in p_lib),
        "in24_or_recent16_all": sum(all(r["in24_or_recent16"]) for r in p_lib),
        "in24_lemma_recall": round(
            sum(sum(r["in24"]) for r in p_lib) / max(1, sum(len(r["in24"]) for r in p_lib)), 4
        ),
        "in200_lemma_recall": round(
            sum(sum(r["in200"]) for r in p_lib) / max(1, sum(len(r["in200"]) for r in p_lib)), 4
        ),
        "recent16_lemma_recall": round(
            sum(sum(r["in_recent16"]) for r in p_lib) / max(1, sum(len(r["in_recent16"]) for r in p_lib)), 4
        ),
        "in24_or_recent16_lemma_recall": round(
            sum(sum(r["in24_or_recent16"]) for r in p_lib) / max(1, sum(len(r["in24_or_recent16"]) for r in p_lib)), 4
        ),
        "forms": dict(collections.Counter(r["form"] for r in prem_rows)),
    }

    # Action coverage vs 1200-step union closure on diag_rows
    by_all = {a: {r["id"]: r for r in rs} for a, rs in arm_rows.items()}
    cov_by_form = collections.defaultdict(lambda: [0, 0])
    for r in diag_rows:
        f = classify_human(r["human"])
        cov_by_form[f][0] += 1
        cov_by_form[f][1] += int(any(by_all[a][r["id"]]["solved"] for a in ("s1", "s2", "s3", "s3c")))

    # Baseline depth=3 on probe_rows for exact apples-to-apples comparison
    s2_d3_300_probe = sum(1 for r in arm_rows["s2"] if r["id"] in probe_set and r["solved"] and r["steps"] <= 300)
    s3_d3_300_probe = sum(1 for r in arm_rows["s3"] if r["id"] in probe_set and r["solved"] and r["steps"] <= 300)
    s3_d3_600_probe = sum(1 for r in arm_rows["s3"] if r["id"] in probe_set and r["solved"] and r["steps"] <= 600)
    s3_d3_1200_probe = sum(1 for r in arm_rows["s3"] if r["id"] in probe_set and r["solved"] and r["steps"] <= 1200)
    s3_d3_probe_rows = [r for r in arm_rows["s3"] if r["id"] in probe_set]

    report = {
        "schema": "olr08b.obj1_diagnostics.v1",
        "spent_only_verified": True,
        "sealed_SInt_opened": False,
        "protected_eval_opened": False,
        "cohort": {
            "total_roots": len(diag_rows),
            "olr08a_spent_cal_roots": len(cal_rows),
            "olr08a_spent_train_sample_roots": len(tr_sample_rows),
            "files": len({r["file"] for r in diag_rows}),
            "probe_subset_roots": len(probe_rows),
        },
        "paired_all_spent": simulate_portfolios_and_oracles(arm_rows),
        "paired_olr08a_spent_cal18": simulate_portfolios_and_oracles(filter_arms(cal_set)),
        "costs_all_spent_1200": {a: summarize_costs(arm_rows[a]) for a in ("s1", "s2", "s3", "s3c")},
        "mechanism_probes": {
            "roots": len(probe_rows),
            "depth3_s2_300": s2_d3_300_probe,
            "depth5_s2_300": sum(r["solved"] for r in depth5_s2),
            "depth5_s2_newly_solved_ids": sorted(
                r["id"]
                for r in depth5_s2
                if r["solved"] and not (by_all["s2"][r["id"]]["solved"] and by_all["s2"][r["id"]]["steps"] <= 300)
            ),
            "depth3_s3_600": s3_d3_600_probe,
            "depth5_s3_600": sum(r["solved"] for r in depth5_s3),
            "depth5_s3_newly_solved_ids": sorted(
                r["id"]
                for r in depth5_s3
                if r["solved"] and not (by_all["s3"][r["id"]]["solved"] and by_all["s3"][r["id"]]["steps"] <= 600)
            ),
            "depth3_s3_1200": s3_d3_1200_probe,
            "depth3_s3_300": s3_d3_300_probe,
            "recency16_s3_300": sum(r["solved"] for r in recency_s3),
            "recency16_newly_solved_ids": sorted(
                r["id"]
                for r in recency_s3
                if r["solved"] and not (by_all["s3"][r["id"]]["solved"] and by_all["s3"][r["id"]]["steps"] <= 300)
            ),
            "lazy_s3_300": sum(r["solved"] for r in lazy_s3),
            "lazy_s3_300_costs": summarize_costs(lazy_s3),
            "eager_s3_1200_costs_subset": summarize_costs(s3_d3_probe_rows),
        },
        "premise_retrieval": prem_summary,
        "action_coverage": {
            k: {"roots": v[0], "solved_any_1200": v[1], "rate": round(v[1] / max(1, v[0]), 4)}
            for k, v in sorted(cov_by_form.items())
        },
        "referee_archive_audit": analyze_referee_archive(args.referee_zip, args.source),
    }
    run08.write(args.out / "report_obj1.json", report)
    print(json.dumps(report["paired_all_spent"]["headroom"], indent=2), flush=True)
    print(json.dumps(report["mechanism_probes"], indent=2), flush=True)
    print(json.dumps(report["premise_retrieval"], indent=2), flush=True)


if __name__ == "__main__":
    main()
