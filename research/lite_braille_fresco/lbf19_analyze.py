"""LBF-19 statistical analysis, confirmatory hypothesis tests, and illustrated SVG figure generator."""
from __future__ import annotations

import collections
import gzip
import json
import math
import pathlib
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "lbf19" / "confirm_out"
COHORT_DIR = HERE / "results" / "lbf19" / "confirm_cohort"
FIG_DIR = HERE / "results" / "lbf19"


def load_gz(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        return []
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sign_flip_p(diffs: list[float], alternative: str = "greater", n_perm: int = 100_000, seed: int = 20261901) -> float:
    """Exact/Monte-Carlo root-paired sign-flip randomization test."""
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        s = sum(d if rng.getrandbits(1) else -d for d in diffs) / n
        if alternative == "greater":
            if s >= obs - 1e-12:
                ge += 1
        elif alternative == "less":
            if s <= obs + 1e-12:
                ge += 1
        else:
            if abs(s) >= abs(obs) - 1e-12:
                ge += 1
    return (ge + 1) / (n_perm + 1)


def bootstrap_ci(vals: list[float], n_boot: int = 10_000, seed: int = 20261902) -> tuple[float, float, float]:
    n = len(vals)
    mean = sum(vals) / n
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return mean, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def root_table(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Aggregate episodes to root level: arm -> root -> averaged metrics."""
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        grouped[r["arm"]][r["root"]].append(r)
    out = {}
    for arm, by_root in grouped.items():
        out[arm] = {}
        for root, eps in by_root.items():
            closed = [1.0 if e["closed"] else 0.0 for e in eps]
            works = [float(e["work"]) if e["work"] is not None else float(e["budget"]) for e in eps]
            calls = [float(e["v_verifier_calls"]) for e in eps]
            edges = [float(e["v_edges_requested"]) for e in eps]
            hits = [float(e["v_cache_hits"]) for e in eps]
            wall = [float(e["wall_seconds"]) for e in eps]
            tac = [float(e["v_tactic_seconds"]) for e in eps]
            init = [float(e["v_init_seconds"]) for e in eps]
            ms_vals = [float(e["v_mean_call_ms"]) for e in eps if e.get("v_mean_call_ms") is not None]
            out[arm][root] = {
                "L": eps[0]["L"],
                "n_seeds": len(eps),
                "closure": st.fmean(closed),
                "all_seeds_closed": 1.0 if all(c > 0.5 for c in closed) else 0.0,
                "any_seed_closed": 1.0 if any(c > 0.5 for c in closed) else 0.0,
                "work": st.fmean(works),
                "calls": st.fmean(calls),
                "edges": st.fmean(edges),
                "cache_hits": st.fmean(hits),
                "wall_s": st.fmean(wall),
                "tactic_s": st.fmean(tac),
                "init_s": st.fmean(init),
                "mean_call_ms": st.fmean(ms_vals) if ms_vals else None,
                "mismatches": sum(int(e["v_mismatches"]) for e in eps),
                "censored": any(bool(e["censored_by_call_cap"]) for e in eps),
            }
    return out


def summarize_arm(arm_roots: dict[str, dict], roots: list[str]) -> dict:
    sub = [arm_roots[r] for r in roots if r in arm_roots]
    if not sub:
        return {}
    by_l = {}
    for L in (16, 20, 24):
        sl = [x for x in sub if x["L"] == L]
        by_l[str(L)] = {
            "n": len(sl),
            "closure": st.fmean(x["closure"] for x in sl) if sl else 0.0,
            "calls": st.fmean(x["calls"] for x in sl) if sl else 0.0,
            "wall_s": st.fmean(x["wall_s"] for x in sl) if sl else 0.0,
        }
    ms = [x["mean_call_ms"] for x in sub if x["mean_call_ms"] is not None]
    cl_mean, cl_lo, cl_hi = bootstrap_ci([x["closure"] for x in sub])
    ca_mean, ca_lo, ca_hi = bootstrap_ci([x["calls"] for x in sub])
    return {
        "n_roots": len(sub),
        "closure_mean": cl_mean,
        "closure_ci95": [cl_lo, cl_hi],
        "all_seeds_closure": st.fmean(x["all_seeds_closed"] for x in sub),
        "any_seed_closure": st.fmean(x["any_seed_closed"] for x in sub),
        "work_mean": st.fmean(x["work"] for x in sub),
        "calls_mean": ca_mean,
        "calls_ci95": [ca_lo, ca_hi],
        "calls_median": st.median(x["calls"] for x in sub),
        "edges_mean": st.fmean(x["edges"] for x in sub),
        "cache_hits_mean": st.fmean(x["cache_hits"] for x in sub),
        "wall_s_mean": st.fmean(x["wall_s"] for x in sub),
        "tactic_s_mean": st.fmean(x["tactic_s"] for x in sub),
        "init_ms_mean": st.fmean(x["init_s"] for x in sub) * 1000.0,
        "mean_call_ms": st.fmean(ms) if ms else None,
        "mismatches_total": sum(x["mismatches"] for x in sub),
        "censored_roots": sum(1 for x in sub if x["censored"]),
        "by_L": by_l,
    }


def render_svg_figures(summary: dict, out_dir: pathlib.Path) -> list[str]:
    """Generate clean, publication-grade SVG figures without external dependencies."""
    written = []
    # Figure 1: Native Lean Latency & End-to-End Proof Time Comparison
    lean = summary.get("lean_B512", {})
    lat_by_l = summary.get("lean_latency_by_L", {})
    if lean and lat_by_l:
        w, h = 920, 420
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Segoe UI, Arial, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
            '<text x="460" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">LBF-19 Live Lean 4.19.0 Tactic Latency &amp; End-to-End Proof Time (24 Beacon-Seeded Roots)</text>',
            # Left panel: Latency by L
            '<rect x="30" y="48" width="400" height="340" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>',
            '<text x="230" y="72" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e293b">Native Lean Tactic Latency by Word Length L</text>',
        ]
        max_ms = max(max(lat_by_l[str(L)]["p95_ms"] for L in (16, 20, 24)), 20.0) * 1.15
        ox, oy, pw, ph = 85, 330, 320, 220
        lines.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + pw}" y2="{oy}" stroke="#475569"/>')
        lines.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - ph}" stroke="#475569"/>')
        for tick in range(0, int(max_ms) + 1, max(5, int(max_ms // 4))):
            y = oy - (tick / max_ms) * ph
            lines.append(f'<line x1="{ox - 4}" y1="{y:.1f}" x2="{ox + pw}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            lines.append(f'<text x="{ox - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{tick}ms</text>')
        for idx, L in enumerate((16, 20, 24)):
            d = lat_by_l[str(L)]
            cx = ox + 55 + idx * 105
            for bi, (k, col, lbl) in enumerate([("p50_ms", "#2563eb", "p50"), ("mean_ms", "#0ea5e9", "mean"), ("p95_ms", "#f59e0b", "p95")]):
                v = d[k]
                bh = (v / max_ms) * ph
                bx = cx - 34 + bi * 24
                lines.append(f'<rect x="{bx}" y="{oy - bh:.1f}" width="20" height="{bh:.1f}" rx="2" fill="{col}"/>')
                lines.append(f'<text x="{bx + 10}" y="{oy - bh - 4:.1f}" text-anchor="middle" font-size="9" fill="#1e293b">{v:.1f}</text>')
            lines.append(f'<text x="{cx}" y="{oy + 18}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e293b">L = {L}</text>')
            lines.append(f'<text x="{cx}" y="{oy + 32}" text-anchor="middle" font-size="9" fill="#64748b">init {d["init_ms"]:.0f}ms</text>')
        # Legend left
        for bi, (col, lbl) in enumerate([("#2563eb", "Median (p50)"), ("#0ea5e9", "Mean"), ("#f59e0b", "95th pct (p95)")]):
            lx = 90 + bi * 110
            lines.append(f'<rect x="{lx}" y="84" width="10" height="10" fill="{col}"/>')
            lines.append(f'<text x="{lx + 14}" y="93" font-size="10" fill="#334155">{lbl}</text>')

        # Right panel: Verifier calls & projected vs measured wall time (log scale bars)
        lines.append('<rect x="455" y="48" width="435" height="340" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>')
        lines.append('<text x="672" y="72" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e293b">Native Verifier Calls per Root (Log10 Scale, B=512)</text>')
        count_b512 = summary.get("count_B512", {})
        bar_arms = [
            ("sym_prov_readout", "sym_prov_readout (Primary)", "#16a34a"),
            ("rel_prov_guarded", "rel_prov_guarded (Recurrent)", "#2563eb"),
            ("hybrid_sym_rec_opened", "hybrid_sym_rec", "#7c3aed"),
            ("lbf16_pos1_opened", "lbf16_pos1 (No Rel Env)", "#0891b2"),
            ("absint_mm", "absint_mm (0-param)", "#64748b"),
            ("geom_mm", "geom_mm (Hamming)", "#94a3b8"),
            ("exact_mm", "exact_mm (3-Hop Ball)", "#f59e0b"),
            ("bidir_uncapped", "bidir_uncapped (BFS)", "#dc2626"),
        ]
        bx0, by0, bw_max = 640, 102, 220
        log_max = 5.6
        for idx, (ak, lbl, col) in enumerate(bar_arms):
            d = count_b512.get(ak) or lean.get(ak)
            if not d:
                continue
            y = by0 + idx * 33
            calls = max(1.0, d["calls_mean"])
            cl_pct = 100.0 * d["closure_mean"]
            bw = min(bw_max, (math.log10(calls) / log_max) * bw_max)
            lines.append(f'<text x="{bx0 - 8}" y="{y + 14}" text-anchor="end" font-size="10.5" fill="#1e293b">{lbl}</text>')
            lines.append(f'<rect x="{bx0}" y="{y + 2}" width="{bw:.1f}" height="18" rx="3" fill="{col}"/>')
            lines.append(f'<text x="{bx0 + bw + 5:.1f}" y="{y + 15}" font-size="9.5" fill="#0f172a">{calls:,.0f} ({cl_pct:.0f}%)</text>')
        lines.append('<text x="672" y="376" text-anchor="middle" font-size="10" fill="#475569">Bar length = log10(mean verifier calls); label = calls (closure rate at B=512)</text>')
        lines.append('</svg>')
        p = out_dir / "fig_lbf19_lean_latency_and_calls.svg"
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p.name)

    # Figure 2: Closure Rate across Lengths L=16, 20, 24 & Budget B=512 vs B=1024
    count_b512 = summary.get("count_B512", {})
    if count_b512:
        w, h = 920, 390
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Segoe UI, Arial, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
            '<text x="460" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">LBF-19 Scaled Closure Rates by Word Length (L = 16, 20, 24; 24 Beacon-Seeded Roots)</text>',
            '<rect x="30" y="48" width="860" height="315" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>',
        ]
        show_arms = [
            ("sym_prov_readout", "sym_prov", "#16a34a"),
            ("hybrid_sym_rec_opened", "hybrid_rec", "#7c3aed"),
            ("rel_prov_guarded", "rel_prov", "#2563eb"),
            ("compute_nonrecurrent", "comp_nr", "#0ea5e9"),
            ("lbf16_pos1_opened", "lbf16_pos1", "#0891b2"),
            ("absint_mm", "absint_mm", "#64748b"),
            ("exact_mm", "exact_mm", "#f59e0b"),
            ("geom_mm", "geom_mm", "#94a3b8"),
            ("graph_bridge", "graph_br", "#cbd5e1"),
        ]
        ox, oy, pw, ph = 85, 310, 770, 210
        lines.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + pw}" y2="{oy}" stroke="#475569"/>')
        lines.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - ph}" stroke="#475569"/>')
        for pct in (0, 25, 50, 75, 100):
            y = oy - (pct / 100.0) * ph
            lines.append(f'<line x1="{ox - 4}" y1="{y:.1f}" x2="{ox + pw}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            lines.append(f'<text x="{ox - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{pct}%</text>')
        groups = [("L = 16", "16"), ("L = 20", "20"), ("L = 24", "24"), ("All 24 Roots", "all")]
        gw = pw / len(groups)
        for gi, (glbl, gkey) in enumerate(groups):
            gx0 = ox + gi * gw + 14
            bw = (gw - 32) / len(show_arms)
            for ai, (ak, albl, col) in enumerate(show_arms):
                d = count_b512.get(ak)
                if not d:
                    continue
                v = d["closure_mean"] if gkey == "all" else d["by_L"][gkey]["closure"]
                bh = v * ph
                bx = gx0 + ai * bw
                lines.append(f'<rect x="{bx:.1f}" y="{oy - bh:.1f}" width="{max(4, bw - 2):.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
                if v >= 0.08:
                    lines.append(f'<text x="{bx + bw / 2 - 1:.1f}" y="{oy - bh - 3:.1f}" text-anchor="middle" font-size="8.5" fill="#1e293b">{100*v:.0f}</text>')
            lines.append(f'<text x="{ox + gi * gw + gw / 2:.1f}" y="{oy + 20}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1e293b">{glbl}</text>')
        for ai, (ak, albl, col) in enumerate(show_arms):
            lx = 70 + ai * 92
            lines.append(f'<rect x="{lx}" y="66" width="10" height="10" fill="{col}"/>')
            lines.append(f'<text x="{lx + 14}" y="75" font-size="9.5" fill="#334155">{albl}</text>')
        lines.append('</svg>')
        p = out_dir / "fig_lbf19_scaled_closure.svg"
        p.write_text("\n".join(lines), encoding="utf-8")
        written.append(p.name)
    return written


def main():
    cases = [json.loads(l) for l in (COHORT_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    roots = [c["root"] for c in cases]
    lean_rows = load_gz(OUT_DIR / "episodes_lean_B512.jsonl.gz")
    c512_rows = load_gz(OUT_DIR / "episodes_count_B512.jsonl.gz")
    c1024_rows = load_gz(OUT_DIR / "episodes_count_B1024.jsonl.gz")
    certs_path = OUT_DIR / "certs_lean_B512.json"
    certs = json.loads(certs_path.read_text(encoding="utf-8")) if certs_path.exists() else []

    lean_rt = root_table(lean_rows)
    c512_rt = root_table(c512_rows)
    c1024_rt = root_table(c1024_rows)

    lat_by_l = {}
    for L in (16, 20, 24):
        sub = [r for r in lean_rows if r["L"] == L and r.get("v_mean_call_ms") is not None]
        if sub:
            lat_by_l[str(L)] = {
                "n_episodes": len(sub),
                "total_calls": sum(int(r["v_verifier_calls"]) for r in sub),
                "init_ms": st.fmean(r["v_init_seconds"] for r in sub) * 1000.0,
                "mean_ms": st.fmean(r["v_mean_call_ms"] for r in sub),
                "p50_ms": st.fmean(r["v_p50_call_ms"] for r in sub),
                "p95_ms": st.fmean(r["v_p95_call_ms"] for r in sub),
            }

    summary = {
        "n_roots": len(roots),
        "classical_cohort_stats": {
            "expanded_min": min(c["classical"]["classical_expanded"] for c in cases),
            "expanded_median": st.median(c["classical"]["classical_expanded"] for c in cases),
            "expanded_max": max(c["classical"]["classical_expanded"] for c in cases),
            "visited_mean": st.fmean(c["classical"]["classical_visited"] for c in cases),
            "work_mean": st.fmean(c["classical"]["classical_work"] for c in cases),
        },
        "certificates": {
            "n_files": len(certs),
            "n_theorems": sum(c["n_theorems"] for c in certs),
            "all_axiom_free": all(c["all_axiom_free"] for c in certs) if certs else False,
            "mean_compile_s": st.fmean(c["compile_seconds"] for c in certs) if certs else 0.0,
        },
        "lean_latency_by_L": lat_by_l,
        "lean_B512": {a: summarize_arm(v, roots) for a, v in lean_rt.items()},
        "count_B512": {a: summarize_arm(v, roots) for a, v in c512_rt.items()},
        "count_B1024": {a: summarize_arm(v, roots) for a, v in c1024_rt.items()},
        "confirmatory": {},
        "contrasts": {},
    }

    # Confirmatory H1, H2, H3 on count_B512 + lean_B512
    rt_src = c512_rt if c512_rt else lean_rt
    if "sym_prov_readout" in rt_src and "geom_mm" in rt_src:
        if "graph_bridge" in rt_src:
            h1_diffs = [rt_src["sym_prov_readout"][r]["closure"] - max(rt_src["geom_mm"][r]["closure"], rt_src["graph_bridge"][r]["closure"]) for r in roots]
        else:
            h1_diffs = [rt_src["sym_prov_readout"][r]["closure"] - rt_src["geom_mm"][r]["closure"] for r in roots]
        m, lo, hi = bootstrap_ci(h1_diffs)
        p1 = sign_flip_p(h1_diffs, "greater", seed=20261911)
        summary["confirmatory"]["H1_closure_vs_best_heur"] = {
            "diff_mean": m, "ci95": [lo, hi], "p_one_sided": p1, "confirmed": p1 <= 0.025,
        }
        if "bidir_uncapped" in rt_src:
            h2_diffs = [rt_src["sym_prov_readout"][r]["calls"] - rt_src["bidir_uncapped"][r]["calls"] for r in roots]
            m2, lo2, hi2 = bootstrap_ci(h2_diffs)
            p2 = sign_flip_p(h2_diffs, "less", seed=20261912)
            # Also live Lean wall-clock on closed roots vs censored bidir_uncapped
            lean_wall_diffs = []
            if "sym_prov_readout" in lean_rt and "bidir_uncapped" in lean_rt:
                for r in roots:
                    if lean_rt["sym_prov_readout"][r]["closure"] > 0:
                        lean_wall_diffs.append(lean_rt["sym_prov_readout"][r]["wall_s"] - lean_rt["bidir_uncapped"][r]["wall_s"])
            summary["confirmatory"]["H2_calls_vs_bidir_uncapped"] = {
                "diff_calls_mean": m2, "ci95": [lo2, hi2], "p_one_sided": p2,
                "confirmed": (p1 <= 0.025) and (p2 <= 0.025),
                "lean_closed_roots_n": len(lean_wall_diffs),
                "lean_wall_s_diff_vs_2000cap_bidir": st.fmean(lean_wall_diffs) if lean_wall_diffs else None,
            }
        if "exact_mm" in rt_src:
            h3_diffs = [rt_src["sym_prov_readout"][r]["calls"] - rt_src["exact_mm"][r]["calls"] for r in roots]
            m3, lo3, hi3 = bootstrap_ci(h3_diffs)
            p3 = sign_flip_p(h3_diffs, "less", seed=20261913)
            summary["confirmatory"]["H3_calls_vs_exact_mm"] = {
                "diff_calls_mean": m3, "ci95": [lo3, hi3], "p_one_sided": p3, "confirmed": p3 <= 0.025,
            }

    # Pre-declared estimation contrasts (count_B512)
    if c512_rt and "sym_prov_readout" in c512_rt:
        for comp in ("rel_prov_guarded", "hybrid_sym_rec_opened", "lbf16_pos1_opened", "compute_nonrecurrent", "absint_mm"):
            if comp in c512_rt:
                dc = [c512_rt[comp][r]["closure"] - c512_rt["sym_prov_readout"][r]["closure"] for r in roots]
                dw = [c512_rt[comp][r]["work"] - c512_rt["sym_prov_readout"][r]["work"] for r in roots]
                mc, loc, hic = bootstrap_ci(dc)
                mw, low, hiw = bootstrap_ci(dw)
                summary["contrasts"][f"{comp}_minus_sym_prov_readout"] = {
                    "closure_diff": mc, "closure_ci95": [loc, hic],
                    "closure_p_two_sided": sign_flip_p(dc, "two-sided"),
                    "work_diff": mw, "work_ci95": [low, hiw],
                    "work_p_two_sided": sign_flip_p(dw, "two-sided"),
                }

    # Verify exact count-mode vs lean-mode call-count correspondence on non-censored episodes
    if lean_rows and c512_rows:
        c_idx = {(r["arm"], r["seed"], r["root"]): r for r in c512_rows}
        checked = matched = 0
        for lr in lean_rows:
            if lr["censored_by_call_cap"]:
                continue
            cr = c_idx.get((lr["arm"], lr["seed"], lr["root"]))
            if cr is not None:
                checked += 1
                if cr["v_verifier_calls"] == lr["v_verifier_calls"] and cr["work"] == lr["work"] and cr["closed"] == lr["closed"]:
                    matched += 1
        summary["count_vs_lean_exact_match"] = {"checked": checked, "matched": matched}

    (OUT_DIR / "confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figs = render_svg_figures(summary, FIG_DIR)
    print(json.dumps({"summary_keys": list(summary.keys()), "figs": figs, "certs": summary["certificates"],
                      "lat_by_L": lat_by_l, "confirmatory": summary["confirmatory"],
                      "count_vs_lean": summary.get("count_vs_lean_exact_match")}, indent=2))


if __name__ == "__main__":
    main()
