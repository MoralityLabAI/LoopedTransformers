"""Stage E statistics: paired power preflight and the EvalMath FSI-01 bridge.

No mechanistic-power preflight script exists in the surveyed repos, so power_preflight here
computes, by exact enumeration of the one-sided exact McNemar test, the number of independent
roots needed to detect a registered minimum effect. FSI-01 itself is never re-implemented:
fsi_report calls research_gym/scripts/evalmath_fsi_gate.py with EvalMath on PYTHONPATH.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

GITHUB = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]


def _binom_tail(n: int, k: int) -> float:
    """P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def mcnemar_power(n_roots: int, p_gain: float, p_loss: float, alpha: float) -> float:
    """Power of the one-sided exact McNemar test with n paired roots.

    Discordant count D ~ Bin(n, p_gain + p_loss); given D, gains ~ Bin(D, p_gain/(p_gain+p_loss)).
    """
    pd = p_gain + p_loss
    q = p_gain / pd
    crit = {}
    power = 0.0
    for d in range(n_roots + 1):
        pdd = math.comb(n_roots, d) * pd ** d * (1 - pd) ** (n_roots - d)
        if pdd < 1e-15 or d == 0:
            continue
        if d not in crit:
            k = next((k for k in range(d + 1) if _binom_tail(d, k) <= alpha), d + 1)
            crit[d] = k
        k = crit[d]
        power += pdd * sum(math.comb(d, i) * q ** i * (1 - q) ** (d - i) for i in range(k, d + 1))
    return power


def power_preflight(min_effect: float = 0.10, discordance: float = 0.30, alpha: float = 0.05,
                    comparisons: int = 1, target_power: float = 0.80, n_max: int = 2000) -> dict:
    """Smallest root count reaching target power; Bonferroni alpha over planned comparisons."""
    a = alpha / comparisons
    p_gain = (discordance + min_effect) / 2
    p_loss = (discordance - min_effect) / 2
    if p_loss < 0:
        raise ValueError("discordance must be at least the minimum effect")
    lo = 10
    n = lo
    while n <= n_max:
        if mcnemar_power(n, p_gain, p_loss, a) >= target_power:
            break
        n = int(n * 1.15) + 1
    # refine downward
    best = n
    for m in range(max(lo, int(n / 1.15) - 1), n + 1):
        if mcnemar_power(m, p_gain, p_loss, a) >= target_power:
            best = m
            break
    return {"min_effect": min_effect, "assumed_discordance": discordance, "alpha": alpha,
            "planned_comparisons": comparisons, "per_comparison_alpha": a, "target_power": target_power,
            "required_independent_roots": best if best <= n_max else None,
            "power_at_required": mcnemar_power(best, p_gain, p_loss, a) if best <= n_max else None,
            "test": "one-sided exact McNemar on root-level paired outcomes"}


def registration(experiment_id: str, min_effect: float, comparisons: int, alpha: float = 0.05,
                 min_clusters: int = 20, population: str = "sealed confirmation roots", *,
                 preregistered: bool, sealed: bool, clusters_independent: bool = True,
                 exchangeable: bool = True) -> dict:
    # EvalMath parses probabilities as exact decimals: pass strings, never floats.
    return {"schema_version": 1, "experiment_id": experiment_id, "target_population": population,
            "evaluation_mode": "sampled", "alpha": str(alpha), "minimum_effect": str(min_effect),
            "planned_comparisons": comparisons, "min_independent_clusters": min_clusters,
            "hypothesis_preregistered": preregistered, "sealed_readout": sealed,
            "clusters_independent": clusters_independent, "exchangeable_pair_labels": exchangeable}


def evalmath_src() -> Path | None:
    for name in ("EvalMath", "EvalMath-mit", "EvalMath-vad", "EvalMath-cpf"):
        p = GITHUB / name / "src" / "spiral_ln" / "evaluation_finite_sample.py"
        if p.exists():
            return p.parents[1]
    return None


def fsi_report(cases_jsonl: Path, registration_json: Path, report: Path) -> dict:
    src = evalmath_src()
    if src is None:
        raise RuntimeError("EvalMath FSI-01 source not found next to this repository")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), str(REPO)]), "CUDA_VISIBLE_DEVICES": "-1"}
    cmd = [sys.executable, str(REPO / "research_gym/scripts/evalmath_fsi_gate.py"), "--cases", str(cases_jsonl),
           "--registration", str(registration_json), "--report", str(report)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"FSI gate failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr else proc.returncode}")
    out = json.loads(report.read_text(encoding="utf-8"))
    out["evalmath_src"] = str(src.parent.name)
    return out
