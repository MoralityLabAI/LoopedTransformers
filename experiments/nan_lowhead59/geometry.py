"""Vocabulary spectral proposal, physical null basis and registered RMS chord."""
import math

from experiments.nan_lattice53.run import next_neighbors
from experiments.nan_readpath37.run import decode_bf16
from experiments.nan_sham56.geometry import drive, remove


def head_spectrum(head, cfg, block_rows, check, np):
    gram = np.zeros((head.width, head.width), dtype=np.float64)
    stride = 2 * head.width
    taken = 0
    gamma = head.gamma.astype(np.float32)
    step, offset = cfg["head_design_stride"], cfg["head_design_offset"]
    head.stream.seek(head.base + head.weight_offset)
    for start in range(0, head.vocab, block_rows):
        check()
        n = min(block_rows, head.vocab - start)
        raw = head.stream.read(n*stride)
        if len(raw) != n*stride:
            raise ValueError("Short vocabulary block")
        weights = decode_bf16(raw, np).reshape(n, head.width)
        first = (offset-start) % step
        selected = weights[first::step] * gamma
        if len(selected):
            gram += selected.T @ selected
            taken += len(selected)
    if taken != len(range(offset, head.vocab, step)):
        raise ValueError("Wrong head-design sample size")
    gram /= taken
    eigen, vectors = np.linalg.eigh(gram)
    width = cfg["spectral_axes"]
    lowest = eigen[:width]
    seed = vectors[:, :width]
    if eigen[0] < -1e-8 or width > len(eigen):
        raise ValueError("Global head sketch lost PSD or proposal width")
    closure = np.max(abs(gram @ seed - seed*lowest))
    if closure > 1e-9:
        raise ValueError("The head eigenspace did not close")
    return seed, {"sampled_vocabulary_rows": taken, "min_eigenvalue": float(eigen[0]),
                  "proposal_limit_eigenvalue": float(lowest[-1]), "median_eigenvalue": float(np.median(eigen)),
                  "max_eigen_residual": float(closure)}


def fine_basis(h, raw, gradient, reached, radius, tag, spectral, cfg, sham_cfg, np):
    down, up = next_neighbors(h.astype(np.float32), np)
    spacing = np.maximum(h-down.astype(np.float64), up.astype(np.float64)-h)
    support = spacing <= .25*radius/math.sqrt(len(h))
    normals = np.column_stack((raw, gradient, *reached))[support]
    lengths = np.linalg.norm(normals, axis=0)
    keep = lengths > 1e-12
    normals = normals[:, keep] / lengths[keep]
    first = drive(tag, sham_cfg["sham_namespace"], len(h), np)
    ambient = drive(tag, "NAN-LOWHEAD-59/ambient", len(h)*(cfg["ambient_axes"]-1), np)
    proposed = np.column_stack((first, spectral, ambient.reshape(len(h), -1)))[support]
    q, r = np.linalg.qr(remove(proposed, normals, np), mode="reduced")
    if min(abs(r.diagonal())) < 1e-8 or np.max(abs(normals.T @ q)) > 1e-9:
        raise ValueError("Screened physical atlas lost rank or null closure")
    result = np.zeros((len(h), q.shape[1]))
    result[support] = q
    return result, int(support.sum())


def designs(atlas, ambient_peak, target_rms, limit, np):
    double = atlas.astype(np.float64)
    gram = double.T @ double / len(atlas)
    del double
    eig, axes = np.linalg.eigh(gram)
    low = axes[:, 0]
    active = ambient_peak.astype(np.float64)
    active /= np.linalg.norm(active)
    if float(low @ active) < 0.:
        low = -low
    if eig[0] < -1e-12:
        raise ValueError("Local derivative Gram is not PSD")
    evaluated = []
    for weight in np.linspace(0., 1., 17):
        q = (1-weight)*low + weight*active
        q /= np.linalg.norm(q)
        rms = math.sqrt(max(0., float(q @ gram @ q)))
        peak = float(np.max(abs(atlas @ q.astype(np.float32))))
        evaluated.append((q, rms, peak, float(weight)))
    ready = [v for v in evaluated if v[2] <= limit]
    chosen = min(ready, key=lambda v: abs(v[1]-target_rms)) if ready else min(evaluated, key=lambda v: v[2])
    return {"low_spectrum_minimum": low, "rms_targeted": chosen[0]}, {
        "predicted_minimum_rms": math.sqrt(max(float(eig[0]), 0.)), "predicted_minimum_peak": evaluated[0][2],
        "target_active_rms": target_rms, "targeted_predicted_rms": chosen[1], "targeted_predicted_peak": chosen[2],
        "chord_weight": chosen[3], "predicted_peak_gate_met": bool(ready)}


def verify(np):
    atlas = np.array([[1., 0.], [0., .125], [0., -.125], [-1., 0.]], dtype=np.float32)
    out, result = designs(atlas, np.array([1., 0.], dtype=np.float32), .125, .2, np)
    assert result["predicted_minimum_rms"] == np.sqrt(.0078125)
    assert result["predicted_peak_gate_met"] and result["targeted_predicted_peak"] <= .2
    assert abs(result["targeted_predicted_rms"] - .125) < .025
    assert abs(np.linalg.norm(out["rms_targeted"]) - 1.) < 1e-12
