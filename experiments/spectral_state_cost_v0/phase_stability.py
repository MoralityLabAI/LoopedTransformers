"""SPC-02: phase-schedule conditioning and noisy sparse recurrence recovery.

Independent CPU implementation. Fourier matrices here are tiny oracle matrices;
large models may supply a held-out loop-depth trace to the companion CLI.

Conventions: n=m*d, unnormalised n-point Fourier coefficients a[j+r*m],
and demodulated alias measurement b[j,s]=sum_r z_d^(s*r)*a[j,r].
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from typing import Sequence

import numpy as np


def check_offsets(d: int, offsets: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(d, int) or isinstance(d, bool) or d < 1:
        raise ValueError("d must be a positive integer")
    phases = tuple(offsets)
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 or s >= d for s in phases):
        raise ValueError("offsets must be distinct integers in [0,d)")
    if len(set(phases)) != len(phases):
        raise ValueError("duplicate offsets")
    return phases


def phase_matrix(d: int, offsets: Sequence[int]) -> np.ndarray:
    s = np.array(check_offsets(d, offsets), dtype=np.int64)
    r = np.arange(d, dtype=np.int64)
    return np.exp(2j * np.pi * np.outer(s, r) / d)


def critical_support(d: int, k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= d:
        raise ValueError("require 1 <= k <= d")
    return min(2 * k, d)


def certified_gain_floor(d: int, k: int) -> float:
    """Proven uniform ||V_q h|| >= gamma ||h|| on all <=2k-support h.

    Requires q>=p=min(2k,d) consecutive phases or a coprime arithmetic
    progression. This is a conservative analytic guarantee, not an SVD fit.
    """
    p = critical_support(d, k)
    if p == d:
        return math.sqrt(d)
    chord = 2 * math.sin(math.pi / d)
    return (chord ** (p * (p - 1) // 2)) / (p ** (p - 1))


def sharp_singleton_gain(d: int) -> float:
    """Exact restricted minimum gain for the first two phases at k=1."""
    if not isinstance(d, int) or d < 2:
        raise ValueError("require d >= 2")
    return 2 * math.sin(math.pi / (2 * d))


def restricted_min_gain(d: int, offsets: Sequence[int], support: int,
                        max_minors: int = 20000) -> tuple[float, tuple[int, ...]]:
    """Enumerate all supports of size `support` and their least singular gain.

    By extending a smaller support to `support` columns, this is also the
    worst gain over all nonzero vectors with at most `support` entries.
    Return an achieving support as a finite numerical witness.
    """
    phases = check_offsets(d, offsets)
    if not isinstance(support, int) or not 1 <= support <= min(len(phases), d):
        raise ValueError("support must be between 1 and number of probes")
    if math.comb(d, support) > max_minors:
        raise ValueError("too many support minors for CPU oracle")
    v = phase_matrix(d, phases)
    best = math.inf
    best_cols: tuple[int, ...] = ()
    for cols in itertools.combinations(range(d), support):
        # Store a real number, never interpret an approximate singular value
        # of zero as proof of exact algebraic rank loss.
        value = float(np.linalg.svd(v[:, cols], compute_uv=False)[-1])
        if value < best:
            best, best_cols = value, cols
    return best, best_cols


def schedule_frontier(d: int, q: int, k: int,
                      max_cases: int = 20000) -> list[dict[str, object]]:
    """Exhaustively rank small schedules by their uniform 2k-sparse gain."""
    p = critical_support(d, k)
    if not p <= q <= d:
        raise ValueError("the identification frontier needs p <= q <= d")
    count = math.comb(d, q) * math.comb(d, p)
    if count > max_cases:
        raise ValueError(f"schedule search would require {count} minor SVDs")
    scores = []
    for phases in itertools.combinations(range(d), q):
        gain, support = restricted_min_gain(d, phases, p)
        scores.append({"offsets": list(phases), "worst_gain": gain,
                       "limiting_support": list(support)})
    return sorted(scores, key=lambda r: (-float(r["worst_gain"]), r["offsets"]))


def alias_blocks_from_samples(samples: np.ndarray, n: int, d: int,
                              offsets: Sequence[int]) -> np.ndarray:
    """Invert the known j-dependent phase, returning b of shape (m,q)."""
    phases = check_offsets(d, offsets)
    if n < 1 or n % d:
        raise ValueError("n must be divisible by d")
    m = n // d
    y = np.asarray(samples, dtype=np.complex128)
    if y.shape != (len(phases), m):
        raise ValueError(f"expected sampled tensor shape {(len(phases), m)}")
    frequencies = np.arange(m)
    demod = np.exp(-2j * np.pi * np.outer(np.array(phases), frequencies) / n)
    return (d * np.fft.fft(y, axis=1) * demod).T


def sample_signal(signal: Sequence[complex], d: int,
                  offsets: Sequence[int]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.complex128)
    if x.ndim != 1 or x.size % d:
        raise ValueError("signal must be a one-dimensional multiple of d")
    phases = check_offsets(d, offsets)
    m = x.size // d
    return np.stack([x[s + d * np.arange(m)] for s in phases], axis=0)


def coefficients_to_blocks(spectrum: Sequence[complex], d: int) -> np.ndarray:
    a = np.asarray(spectrum, dtype=np.complex128)
    if a.ndim != 1 or a.size % d:
        raise ValueError("spectrum must be a one-dimensional multiple of d")
    m = a.size // d
    return a.reshape(d, m).T.copy()


def blocks_to_coefficients(blocks: np.ndarray) -> np.ndarray:
    a = np.asarray(blocks, dtype=np.complex128)
    if a.ndim != 2:
        raise ValueError("alias blocks must have shape (m,d)")
    return a.T.reshape(-1).copy()


def robust_recover_blocks(measured: np.ndarray, d: int,
                         offsets: Sequence[int], k: int) -> tuple[np.ndarray, float]:
    """Brute-force oracle minimising global residual over <=k modes/block.

    Enumeration is for small d and k. Solves each candidate by least squares;
    floating support inference is *not* a formal certificate or scalable FFT.
    """
    phases = check_offsets(d, offsets)
    p = critical_support(d, k)
    if len(phases) < p:
        raise ValueError("insufficient phases for uniform sparse identification")
    if sum(math.comb(d, i) for i in range(k + 1)) > 5000:
        raise ValueError("support enumeration exceeds bounded CPU budget")
    b = np.asarray(measured, dtype=np.complex128)
    if b.ndim != 2 or b.shape[1] != len(phases):
        raise ValueError("measurements must have shape (alias_classes, probes)")
    v = phase_matrix(d, phases)
    result = np.zeros((b.shape[0], d), dtype=np.complex128)
    residual_sq = 0.0
    supports = [c for i in range(k + 1)
                for c in itertools.combinations(range(d), i)]
    for j in range(b.shape[0]):
        best_sq = math.inf
        best_support: tuple[int, ...] = ()
        best_coeff = np.zeros(0, dtype=np.complex128)
        for support in supports:
            if support:
                matrix = v[:, support]
                coeff, *_ = np.linalg.lstsq(matrix, b[j], rcond=None)
                err = b[j] - matrix @ coeff
            else:
                coeff = np.zeros(0, dtype=np.complex128)
                err = b[j]
            sq = float(np.vdot(err, err).real)
            if sq < best_sq - 1e-24:
                best_sq, best_support, best_coeff = sq, support, coeff
        if best_support:
            result[j, list(best_support)] = best_coeff
        residual_sq += best_sq
    return result, math.sqrt(max(residual_sq, 0.0))


def best_fit_periodic(trace: np.ndarray, period: int) -> tuple[np.ndarray, float]:
    """Least-squares periodization and relative residual over full trace.

    A useful *diagnostic* for actual depth traces. If residual is high,
    the periodic alias theorems apply only to the fitted proxy, not the trace.
    """
    z = np.asarray(trace, dtype=np.complex128)
    if z.ndim not in (1, 2) or period < 1 or len(z) < period:
        raise ValueError("need a depth-first 1-D or 2-D trace at least one period long")
    fitted = np.zeros((period,) + z.shape[1:], dtype=np.complex128)
    for phase in range(period):
        fitted[phase] = z[phase::period].mean(axis=0)
    reconstructed = fitted[np.arange(len(z)) % period]
    residual = float(np.linalg.norm(z - reconstructed))
    relative = residual / max(float(np.linalg.norm(z)), 1e-30)
    return fitted, relative


def trace_spectral_diagnostics(trace: np.ndarray, period: int, d: int, k: int,
                               relative_activity_cutoff: float = 0.05) -> dict[str, float | int]:
    """Assess how much a real loop-depth trace meets the SPC-02 assumptions.

    `trace` has shape (depth,) or (depth,features). The method first fits a
    period-n proxy, then measures per-alias k-mode approximation error.
    Mode cutoff is diagnostic and threshold-dependent, never an exact claim.
    """
    if period % d:
        raise ValueError("period must be a multiple of d")
    critical_support(d, k)
    if not 0 <= relative_activity_cutoff < 1:
        raise ValueError("relative cutoff must be in [0,1)")
    fit, relative_residual = best_fit_periodic(trace, period)
    if fit.ndim == 1:
        fit = fit[:, None]
    spectrum = np.fft.fft(fit, axis=0)
    m, features = period // d, spectrum.shape[1]
    group = spectrum.reshape(d, m, features).transpose(1, 0, 2)
    squared = np.abs(group)**2
    energy = float(squared.sum())
    topk = np.sort(squared, axis=1)[:, -k:, :]
    tail_fraction = max(0.0, (energy-float(topk.sum())) / max(energy, 1e-30))
    block_amplitudes = np.abs(group)
    thresholds = relative_activity_cutoff * block_amplitudes.max(axis=1, keepdims=True)
    active = (block_amplitudes > thresholds) & (block_amplitudes > 1e-12)
    occupied = block_amplitudes.max(axis=1) > 1e-12
    occupied_count = int(occupied.sum())
    within_budget = (active.sum(axis=1) <= k) & occupied
    original = np.asarray(trace, dtype=np.complex128)
    temporal_mean = original.mean(axis=0, keepdims=True)
    centered = original - temporal_mean
    dc_energy = float(len(original) * np.vdot(temporal_mean, temporal_mean).real)
    total_energy = float(np.vdot(original, original).real)
    centered_fit, centered_residual = best_fit_periodic(centered, period)
    if centered_fit.ndim == 1:
        centered_fit = centered_fit[:, None]
    centered_fft = np.fft.fft(centered_fit, axis=0)
    centered_squares = np.abs(centered_fft.reshape(d, m, features))**2
    centered_topk = np.sort(centered_squares, axis=0)[-k:]
    centered_energy = float(centered_squares.sum())
    centered_tail = max(0.0, (centered_energy-float(centered_topk.sum())) /
                        max(centered_energy, 1e-30))
    return {
        "period": period,
        "depth_samples": int(len(trace)),
        "feature_channels": features,
        "alias_classes": m,
        "periodic_proxy_relative_residual": relative_residual,
        "spectral_tail_energy_fraction_after_k_terms": tail_fraction,
        "temporal_dc_energy_fraction": dc_energy / max(total_energy, 1e-30),
        "centered_periodic_proxy_relative_residual": centered_residual,
        "centered_spectral_tail_after_k_terms": centered_tail,
        "occupied_alias_feature_pairs": occupied_count,
        "fraction_occupied_alias_pairs_with_k_or_fewer_modes": (
            float(within_budget.sum()) / max(occupied_count, 1)),
        "relative_activity_cutoff": relative_activity_cutoff,
    }


def demonstration() -> dict[str, object]:
    """Pin a finite d=7 schedule search and one noisy periodic replay."""
    d, m, k = 7, 3, 2
    q = critical_support(d, k)
    frontier = schedule_frontier(d, q, k)
    optimized = frontier[0]
    consecutive = (0, 1, 2, 3)
    consecutive_gain, _ = restricted_min_gain(d, consecutive, q)
    # Construct three sparse alias blocks in the original unnormalized DFT.
    a = np.zeros((m, d), dtype=np.complex128)
    a[0, [0, 3]] = [1.5 + 0.1j, -0.5 + 0.2j]
    a[1, [2, 5]] = [0.4 - 0.7j, 1.2 + 0.8j]
    a[2, [1, 6]] = [0.8 + 0.1j, -0.6 + 0.4j]
    full = blocks_to_coefficients(a)
    signal = np.fft.ifft(full)
    phases = optimized["offsets"]
    clean = sample_signal(signal, d, phases)
    rng = np.random.default_rng(12913002)
    eta = rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    eta = 1e-6 * eta / np.linalg.norm(eta)
    b = alias_blocks_from_samples(clean + eta, d * m, d, phases)
    estimated, residual = robust_recover_blocks(b, d, phases, k)
    err = float(np.linalg.norm(estimated - a))
    analytic_bound = 2 * d * math.sqrt(m) * 1e-6 / certified_gain_floor(d, k)
    # Build a separate nonperiodic perturbation as a controlled failure case.
    time = np.arange(3 * d * m)
    nonperiodic = np.tile(signal, 3) + 0.01 * time / len(time)
    _, periodic_residual = best_fit_periodic(nonperiodic, d * m)
    return {
        "campaign": "SPC-02", "d": d, "m": m, "k": k, "q": q,
        "schedules_enumerated": len(frontier),
        "consecutive_offsets": list(consecutive),
        "consecutive_worst_gain": round(consecutive_gain, 10),
        "optimized_offsets": list(phases),
        "optimized_worst_gain": round(float(optimized["worst_gain"]), 10),
        "conditioning_improvement_factor": round(float(optimized["worst_gain"]) / consecutive_gain, 6),
        "analytic_consecutive_gain_floor": round(certified_gain_floor(d, k), 10),
        "noise_l2": round(float(np.linalg.norm(eta)), 12),
        "recovered_coefficient_error_l2": round(err, 12),
        # The analytic bound is for consecutive/coprime AP schedules only.
        "recovered_residual_l2": round(residual, 12),
        "general_consecutive_error_upper_at_noise": round(analytic_bound, 12),
        "nonperiodic_proxy_relative_residual": round(periodic_residual, 9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", help="optional .npy/.npz depth-first trace (T,) or (T,features)")
    parser.add_argument("--key", default="activations", help="key of .npz array")
    parser.add_argument("--period", type=int, help="period for a provided trace")
    parser.add_argument("--stride", type=int, default=7,
                        help="stride dividing --period (default: 7)")
    parser.add_argument("--max-modes", type=int, default=2,
                        help="assumed maximum active modes per alias class")
    args = parser.parse_args()
    record = demonstration()
    if args.trace:
        if args.period is None:
            parser.error("--period is required with --trace")
        contents = np.load(args.trace, allow_pickle=False)
        try:
            data = contents[args.key] if isinstance(contents, np.lib.npyio.NpzFile) else contents
            z = np.asarray(data)
            if z.ndim == 3:
                # Treat each [batch, depth, features] example separately.
                record["trace_diagnostics"] = [
                    trace_spectral_diagnostics(sample, args.period,
                                               args.stride, args.max_modes)
                    for sample in z]
            else:
                record["trace_diagnostics"] = [trace_spectral_diagnostics(
                    z, args.period, args.stride, args.max_modes)]
        finally:
            if isinstance(contents, np.lib.npyio.NpzFile):
                contents.close()
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
